#!/usr/bin/env python3
"""Batch computation of CHM Fourier-coefficient covariance matrices.

For each requested circuit size/depth, this script evaluates the exact
parameter-averaged covariance

    Cov[a] = C P C^dagger

by calling the two-copy twirling backend in ``exact_covariance_twirled.py``.
The calculation never materialises the exponentially large CHM column support.

The saved ``.npz`` files contain both explicit twirling keys
``Cov_TW/Corr_TW/var_TW`` and compatibility keys ``Cov_C/Corr_C/var_C`` for the
plotting scripts in this repository.  Optional direct Monte Carlo validation is
controlled by ``--n_theta_samples``; set it to 0 for exact twirling only.

Example:

    python twirled_correlation_matrices_compute.py \
        --out_dir outputs_twirled_corr_C18_RX \
        --ansatz CIRCUIT_18 --n_qubits_list 6 --train_depth_list 1 2 3 4 \
        --n_layers 1 --encoder_axis RX --obs_kind OZ --max_omega 6 \
        --n_theta_samples 4096 --ansatze_path . --progress
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


def _fmt_seconds(seconds: float) -> str:
    """Human-readable duration for progress/ETA messages."""
    if not np.isfinite(seconds) or seconds < 0:
        return "?"
    seconds = int(round(float(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"

try:
    from tqdm import trange
except Exception:  # pragma: no cover
    def trange(*args, **kwargs):
        return range(*args)


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def import_twirl_module(path: Optional[str]):
    if path:
        return _load_module_from_path("exact_covariance_twirled", Path(path).expanduser().resolve())
    try:
        import exact_covariance_twirled as tw  # type: ignore
        return tw
    except Exception:
        here = Path(__file__).resolve().parent
        candidates = [
            here / "exact_covariance_twirled.py",
            Path.cwd() / "exact_covariance_twirled.py",
            Path.cwd() / "chm_covariance_pipeline" / "exact_covariance_twirled.py",
        ]
        for p in candidates:
            if p.exists():
                return _load_module_from_path("exact_covariance_twirled", p)
    raise ImportError(
        "Could not import exact_covariance_twirled.py. Place it next to this script, "
        "in the project root, or pass --twirl_module_path."
    )


def import_ansatze(ansatze_path: Optional[str]):
    if ansatze_path:
        sys.path.insert(0, str(Path(ansatze_path).expanduser().resolve()))
    try:
        from ansatze import CircuitSpec, build_qnode  # type: ignore
        return CircuitSpec, build_qnode
    except Exception:
        from circuit_distance.ansatze import CircuitSpec, build_qnode  # type: ignore
        return CircuitSpec, build_qnode


# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------


def embed_vector(actual_omegas: np.ndarray, values: np.ndarray, omega_grid: np.ndarray, dtype=np.complex128) -> np.ndarray:
    actual_omegas = np.asarray(actual_omegas, dtype=int)
    omega_grid = np.asarray(omega_grid, dtype=int)
    out = np.zeros((len(omega_grid),), dtype=dtype)
    actual_index = {int(o): i for i, o in enumerate(actual_omegas.tolist())}
    for j, om in enumerate(omega_grid.tolist()):
        i = actual_index.get(int(om))
        if i is not None:
            out[j] = values[i]
    return out


def embed_matrix(actual_omegas: np.ndarray, A: np.ndarray, omega_grid: np.ndarray) -> np.ndarray:
    actual_omegas = np.asarray(actual_omegas, dtype=int)
    omega_grid = np.asarray(omega_grid, dtype=int)
    out = np.zeros((len(omega_grid), len(omega_grid)), dtype=np.complex128)
    actual_index = {int(o): i for i, o in enumerate(actual_omegas.tolist())}
    for r, om in enumerate(omega_grid.tolist()):
        i = actual_index.get(int(om))
        if i is None:
            continue
        for c, op in enumerate(omega_grid.tolist()):
            j = actual_index.get(int(op))
            if j is not None:
                out[r, c] = A[i, j]
    return out


def corr_from_cov(Cov: np.ndarray, eps: float = 1e-14) -> Tuple[np.ndarray, np.ndarray]:
    Cov = np.asarray(Cov, dtype=np.complex128)
    var = np.real(np.diag(Cov)).astype(np.float64)
    denom = np.sqrt(np.maximum(var, 0.0))
    Corr = np.zeros_like(Cov)
    good = denom > eps
    if np.any(good):
        Corr[np.ix_(good, good)] = Cov[np.ix_(good, good)] / np.outer(denom[good], denom[good])
    return Corr, var


def covcorr_from_samples(A: np.ndarray, eps: float = 1e-14) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if A.ndim != 2:
        raise ValueError("A must have shape (n_samples, n_omega)")
    S = A.shape[0]
    if S <= 0:
        n_omega = A.shape[1]
        nanv = np.full((n_omega,), np.nan, dtype=np.float64)
        nanm = np.full((n_omega, n_omega), np.nan + 0j, dtype=np.complex128)
        return nanm, nanm, nanv, np.full((n_omega,), np.nan + 0j, dtype=np.complex128)
    mean = np.mean(A, axis=0)
    Ac = A - mean[None, :]
    Cov = (Ac.T @ Ac.conj()) / float(S)
    Cov = 0.5 * (Cov + Cov.conj().T)
    Corr, var = corr_from_cov(Cov, eps=eps)
    return Cov, Corr, var, mean


def offdiag_vector(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    if n <= 1:
        return np.asarray([], dtype=M.dtype)
    return M[np.triu_indices(n, k=1)]


def rel_frob(A: np.ndarray, B: np.ndarray, eps: float = 1e-14) -> float:
    A = np.asarray(A); B = np.asarray(B)
    mask = np.isfinite(A) & np.isfinite(B)
    if not np.any(mask):
        return float("nan")
    return float(np.linalg.norm((A - B)[mask]) / (np.linalg.norm(B[mask]) + eps))


def cosine_similarity(A: np.ndarray, B: np.ndarray, eps: float = 1e-14) -> float:
    a = np.asarray(A).reshape(-1)
    b = np.asarray(B).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return float("nan")
    a = a[mask]; b = b[mask]
    return float(np.real(np.vdot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps)))


def mean_abs_offdiag(M: np.ndarray) -> float:
    v = offdiag_vector(np.asarray(M))
    v = v[np.isfinite(v)]
    return float(np.mean(np.abs(v))) if v.size else float("nan")


def metric_pack(Cov_TW, Corr_TW, Cov_MC, Corr_MC, var_TW, var_MC) -> dict:
    return {
        "complex": {
            "rel_frob_Corr_TW_vs_MC": rel_frob(Corr_TW, Corr_MC),
            "cosine_offdiag_Corr_TW_vs_MC": cosine_similarity(offdiag_vector(Corr_TW), offdiag_vector(Corr_MC)),
            "mean_abs_offdiag_TW": mean_abs_offdiag(Corr_TW),
            "mean_abs_offdiag_MC": mean_abs_offdiag(Corr_MC),
            "rel_frob_Cov_TW_vs_MC": rel_frob(Cov_TW, Cov_MC),
            "cosine_Cov_TW_vs_MC": cosine_similarity(Cov_TW, Cov_MC),
        },
        "real": {
            "rel_frob_ReCorr_TW_vs_MC": rel_frob(np.real(Corr_TW), np.real(Corr_MC)),
            "cosine_offdiag_ReCorr_TW_vs_MC": cosine_similarity(offdiag_vector(np.real(Corr_TW)), offdiag_vector(np.real(Corr_MC))),
            "mean_abs_offdiag_Re_TW": mean_abs_offdiag(np.real(Corr_TW)),
            "mean_abs_offdiag_Re_MC": mean_abs_offdiag(np.real(Corr_MC)),
        },
        "imag": {
            "rel_frob_ImCorr_TW_vs_MC": rel_frob(np.imag(Corr_TW), np.imag(Corr_MC)),
            "cosine_offdiag_ImCorr_TW_vs_MC": cosine_similarity(offdiag_vector(np.imag(Corr_TW)), offdiag_vector(np.imag(Corr_MC))),
            "mean_abs_offdiag_Im_TW": mean_abs_offdiag(np.imag(Corr_TW)),
            "mean_abs_offdiag_Im_MC": mean_abs_offdiag(np.imag(Corr_MC)),
        },
        "variance": {
            "rel_l2_var_TW_vs_MC": rel_frob(var_TW, var_MC),
            "cosine_var_TW_vs_MC": cosine_similarity(var_TW, var_MC),
            "sum_var_TW": float(np.nansum(var_TW)),
            "sum_var_MC": float(np.nansum(var_MC)),
            "max_var_TW": float(np.nanmax(var_TW)) if np.any(np.isfinite(var_TW)) else float("nan"),
            "max_var_MC": float(np.nanmax(var_MC)) if np.any(np.isfinite(var_MC)) else float("nan"),
        },
    }


# ---------------------------------------------------------------------------
# Exact twirled covariance
# ---------------------------------------------------------------------------


def compute_twirled(tw, args, n_qubits: int, train_depth: int, omega_grid: np.ndarray, n_layers: int = None):
    if n_layers is None:
        n_layers = args.n_layers
    encoder_scale_float = float(args.encoder_scale)
    encoder_scale_int = int(round(encoder_scale_float))
    if abs(encoder_scale_float - encoder_scale_int) > 1e-12:
        raise ValueError("twirled Pauli-propagation code currently requires integer encoder_scale")

    theta_period = tw.parse_period(args.theta_period)
    ops, m = tw.build_forward_ops(args.ansatz, int(n_qubits), int(n_layers), int(train_depth), args.encoder_axis, encoder_scale_int)
    obs = tw.observable_terms(int(n_qubits), args.obs_kind, args.ansatz)

    cr_axes = sorted({op.axis for op in ops if op.kind == "cr" and op.axis is not None})
    cr_single_maps = {}
    cr_pair_maps = {}
    for axis in cr_axes:
        if args.progress:
            print(f"  precomputing CR{str(axis)[-1]} maps: q={args.cr_quadrature_points}, period={args.theta_period}", flush=True)
        smap, pmap = tw.build_cr_maps(str(axis), int(args.cr_quadrature_points), theta_period, float(args.combine_tol))
        cr_single_maps[str(axis)] = smap
        cr_pair_maps[str(axis)] = pmap

    if args.progress:
        print(f"  exact twirl: m={m}, ops={len(ops)}, obs_terms={len(obs)}", flush=True)

    mean_omegas, mean_vals_small, mean_terminal = tw.exact_mean(
        ops, obs, int(n_qubits), cr_single_maps, float(args.combine_tol), int(args.max_states), bool(args.progress)
    )
    mean_dict = {int(o): complex(v) for o, v in zip(mean_omegas.tolist(), mean_vals_small.tolist())}
    moment, second_terminal = tw.exact_second_moment(
        ops, obs, int(n_qubits), cr_pair_maps, float(args.combine_tol), int(args.max_states), bool(args.progress)
    )
    actual_omegas, mean_actual, _second_raw, _second_conj, Cov_actual = tw.assemble_matrices(mean_dict, moment)

    Cov_TW = embed_matrix(actual_omegas, Cov_actual, omega_grid)
    mean_TW = embed_vector(actual_omegas, mean_actual, omega_grid)
    Corr_TW, var_TW = corr_from_cov(Cov_TW)
    return {
        "m": int(m),
        "ops": ops,
        "actual_omegas": actual_omegas,
        "mean_TW": mean_TW,
        "Cov_TW": Cov_TW,
        "Corr_TW": Corr_TW,
        "var_TW": var_TW,
        "n_mean_terminal_terms": int(len(mean_terminal)),
        "n_second_terminal_terms": int(len(second_terminal)),
        "n_second_moment_entries": int(len(moment)),
        "cr_axes": cr_axes,
    }


# ---------------------------------------------------------------------------
# Monte Carlo validation
# ---------------------------------------------------------------------------


def compute_mc(args, n_qubits: int, train_depth: int, omega_grid: np.ndarray, m_expected: int, n_layers: int = None):
    if n_layers is None:
        n_layers = args.n_layers
    S = int(args.n_theta_samples)
    n_omega = len(omega_grid)
    if S <= 0:
        nanm = np.full((n_omega, n_omega), np.nan + 0j, dtype=np.complex128)
        return nanm, nanm, np.full((n_omega,), np.nan, dtype=np.float64), np.full((n_omega,), np.nan + 0j, dtype=np.complex128), None

    CircuitSpec, build_qnode = import_ansatze(args.ansatze_path)
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore

    spec = CircuitSpec(
        ansatz=str(args.ansatz),
        n_qubits=int(n_qubits),
        n_layers=int(n_layers),
        train_depth=int(train_depth),
        encoder_axis=str(args.encoder_axis),
        encoder_scale=float(args.encoder_scale),
        obs_kind=str(args.obs_kind),
        device_name="default.qubit",
        diff_method="backprop",
        jit=True,
    )
    qnode, m = build_qnode(spec)
    if int(m) != int(m_expected):
        raise RuntimeError(f"ansatze.py parameter count m={m} does not match twirled m={m_expected}; check circuit conventions/import path")

    x_grid = np.linspace(float(args.x_min), float(args.x_max), int(args.n_x), endpoint=False)
    dx = (float(args.x_max) - float(args.x_min)) / float(args.n_x)
    period = float(args.x_max) - float(args.x_min)
    if args.fourier_normalisation == "coefficient":
        # Fourier-series coefficient: (1/T) integral f(x) exp(-i omega x) dx.
        # This matches the analytic/twirled coefficient convention.
        Phi = np.exp(-1j * np.outer(omega_grid.astype(float), x_grid)) * (dx / period)
    else:
        # Raw-integral convention retained for comparison with externally generated
        # data that uses integral rather than coefficient normalisation.
        Phi = np.exp(-1j * np.outer(omega_grid.astype(float), x_grid)) * dx

    f_vmap_x = jax.jit(jax.vmap(qnode, in_axes=(0, None)))
    rng = np.random.default_rng(int(args.seed) + 1009 * int(n_qubits) + 7919 * int(train_depth))
    theta_period = import_twirl_module(args.twirl_module_path).parse_period(args.theta_period)

    A = np.empty((S, n_omega), dtype=np.complex128)
    done = 0
    t0 = time.time()
    for start in trange(0, S, int(args.batch_size), desc="MC theta", leave=False):
        b = min(int(args.batch_size), S - start)
        thetas = rng.uniform(0.0, theta_period, size=(b, int(m)))
        for j in range(b):
            y = np.asarray(f_vmap_x(jnp.asarray(x_grid), jnp.asarray(thetas[j])), dtype=np.float64)
            A[start + j] = Phi @ y
        done += b
        if args.progress:
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-12)
            eta = (S - done) / max(rate, 1e-12)
            print(
                f"  MC {done}/{S} rate={rate:.2f} samples/s "
                f"elapsed={_fmt_seconds(elapsed)} eta={_fmt_seconds(eta)}",
                flush=True,
            )
    Cov, Corr, var, mean = covcorr_from_samples(A)
    return Cov, Corr, var, mean, A


# ---------------------------------------------------------------------------
# Saving and CLI
# ---------------------------------------------------------------------------


def save_run_npz(path: Path, payload: Dict[str, Any]) -> None:
    diag = payload.pop("diag")
    payload["diag_json"] = np.array(json.dumps(diag, sort_keys=True))
    np.savez_compressed(path, **payload)


def parse_args(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out_dir", default="outputs_twirled_correlation_matrices")
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--n_qubits_list", type=int, nargs="+", default=[3])
    p.add_argument("--train_depth_list", type=int, nargs="+", default=[1, 2, 3, 4, 5])

    p.add_argument("--ansatz", default="CIRCUIT_18")
    p.add_argument("--n_layers", type=int, default=1)
    p.add_argument("--n_layers_list", type=int, nargs="+", default=None, help="Sweep over layer counts as separate runs. Defaults to [--n_layers].")
    p.add_argument("--encoder_axis", default="RX", choices=["RX", "RY", "RZ"])
    p.add_argument("--encoder_scale", default="1")
    p.add_argument("--obs_kind", default="OZ", choices=["OX", "OY", "OZ", "OZZ"])
    p.add_argument("--theta_period", default="2pi", help="Use 4pi for full CRX/CRZ unitary periodicity; default samples theta over [0,2*pi).")

    p.add_argument("--max_omega", type=int, default=15)
    p.add_argument("--omega_values", type=int, nargs="*", default=None, help="Optional explicit omega grid. Overrides --max_omega.")

    p.add_argument("--n_x", type=int, default=256)
    p.add_argument("--x_min", type=float, default=0.0)
    p.add_argument("--x_max", type=float, default=2 * np.pi)
    p.add_argument("--fourier_normalisation", choices=["coefficient", "raw_integral"], default="coefficient", help="coefficient uses (1/T) integral f exp(-iwx). raw_integral omits the 1/T factor for comparison runs.")

    p.add_argument("--n_theta_samples", type=int, default=4096, help="MC validation samples. Set 0 to skip MC.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--save_mc_samples", dest="save_mc_samples", action="store_true", default=True,
                   help="Save raw MC coefficient samples so the plot script can bootstrap error bars (default: on).")
    p.add_argument("--no_save_mc_samples", dest="save_mc_samples", action="store_false",
                   help="Do not save raw MC samples (smaller files; disables plot-side error bars).")

    p.add_argument("--combine_tol", type=float, default=1e-14)
    p.add_argument("--max_states", type=int, default=20_000_000)
    p.add_argument("--cr_quadrature_points", type=int, default=64)
    p.add_argument("--progress", action="store_true")

    p.add_argument("--ansatze_path", default=None)
    p.add_argument("--twirl_module_path", default=None)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tw = import_twirl_module(args.twirl_module_path)

    if args.omega_values:
        omega_grid = np.asarray(sorted(set(int(o) for o in args.omega_values)), dtype=int)
    else:
        omega_grid = np.arange(-int(args.max_omega), int(args.max_omega) + 1, dtype=int)

    index: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "exact_two_copy_parameter_twirled_covariance",
        "out_dir": str(out_dir),
        "global_config": vars(args),
        "omega_grid": omega_grid.tolist(),
        "runs": [],
    }

    layers_list = args.n_layers_list if args.n_layers_list else [args.n_layers]
    for n_qubits in args.n_qubits_list:
        for train_depth in args.train_depth_list:
            for n_layers in layers_list:
                print(f"\n[run] ansatz={args.ansatz} n={n_qubits} L={n_layers} d={train_depth} encoder={args.encoder_axis}", flush=True)
                t0 = time.time()
                tw_data = compute_twirled(tw, args, int(n_qubits), int(train_depth), omega_grid, int(n_layers))
                Cov_TW = tw_data["Cov_TW"]
                Corr_TW = tw_data["Corr_TW"]
                var_TW = tw_data["var_TW"]
                mean_TW = tw_data["mean_TW"]

                try:
                    Cov_MC, Corr_MC, var_MC, mean_MC, samples_MC = compute_mc(args, int(n_qubits), int(train_depth), omega_grid, int(tw_data["m"]), int(n_layers))
                    mc_status = "ok" if int(args.n_theta_samples) > 0 else "skipped"
                    mc_error = None
                except Exception as exc:
                    print(f"[WARN] MC validation failed: {exc}", flush=True)
                    n_omega = len(omega_grid)
                    Cov_MC = np.full((n_omega, n_omega), np.nan + 0j, dtype=np.complex128)
                    Corr_MC = np.full((n_omega, n_omega), np.nan + 0j, dtype=np.complex128)
                    var_MC = np.full((n_omega,), np.nan, dtype=np.float64)
                    mean_MC = np.full((n_omega,), np.nan + 0j, dtype=np.complex128)
                    samples_MC = None
                    mc_status = "failed"
                    mc_error = str(exc)

                diag = {
                    "method": "exact two-copy parameter twirl; *_C aliases are saved for plotting compatibility",
                    "ansatz": args.ansatz,
                    "n_qubits": int(n_qubits),
                    "n_layers": int(n_layers),
                    "train_depth": int(train_depth),
                    "encoder_axis": args.encoder_axis,
                    "encoder_scale": args.encoder_scale,
                    "obs_kind": args.obs_kind,
                    "theta_period": args.theta_period,
                    "m": int(tw_data["m"]),
                    "n_forward_ops": int(len(tw_data["ops"])),
                    "actual_omegas_from_twirled_support": [int(o) for o in tw_data["actual_omegas"].tolist()],
                    "requested_omega_grid": [int(o) for o in omega_grid.tolist()],
                    "n_mean_terminal_terms": int(tw_data["n_mean_terminal_terms"]),
                    "n_second_terminal_terms": int(tw_data["n_second_terminal_terms"]),
                    "n_second_moment_entries": int(tw_data["n_second_moment_entries"]),
                    "cr_axes": [str(x) for x in tw_data["cr_axes"]],
                    "cr_quadrature_points": int(args.cr_quadrature_points),
                    "combine_tol": float(args.combine_tol),
                    "max_states": int(args.max_states),
                    "n_theta_samples_MC": int(args.n_theta_samples),
                    "mc_status": mc_status,
                    "mc_error": mc_error,
                    "runtime_seconds": float(time.time() - t0),
                    "metrics": metric_pack(Cov_TW, Corr_TW, Cov_MC, Corr_MC, var_TW, var_MC),
                }

                fname = f"twirled_corr_{args.ansatz}_n{int(n_qubits)}_L{int(n_layers)}_d{int(train_depth)}_{args.encoder_axis}_{args.obs_kind}.npz"
                path = out_dir / fname
                payload = {
                    "omega_grid": omega_grid.astype(int),
                    "Cov_TW": Cov_TW,
                    "Corr_TW": Corr_TW,
                    "var_TW": var_TW,
                    "mean_TW": mean_TW,
                    # Compatibility aliases used by the plotting scripts.
                    "Cov_C": Cov_TW,
                    "Corr_C": Corr_TW,
                    "var_C": var_TW,
                    "mean_a_C": mean_TW,
                    "Cov_MC": Cov_MC,
                    "Corr_MC": Corr_MC,
                    "var_MC": var_MC,
                    "mean_a_MC": mean_MC,
                    "mean_a_V": mean_MC,
                    "n_qubits": np.array(int(n_qubits)),
                    "train_depth": np.array(int(train_depth)),
                    "n_layers": np.array(int(n_layers)),
                    "n_theta_samples": np.array(int(args.n_theta_samples)),
                    "S_C": np.array(0),
                    "S_V": np.array(int(args.n_theta_samples)),
                    "seed": np.array(int(args.seed)),
                    "ansatz": np.array(str(args.ansatz)),
                    "encoder_axis": np.array(str(args.encoder_axis)),
                    "obs_kind": np.array(str(args.obs_kind)),
                    "diag": diag,
                }
                if bool(args.save_mc_samples) and (samples_MC is not None) and (mc_status == "ok"):
                    # Raw per-theta MC coefficient samples, shape (S, n_omega).
                    # Consumed by the plot script to bootstrap Monte-Carlo standard
                    # errors on the reported metrics (cosine, Frobenius, mean|offdiag|).
                    payload["samples_MC"] = np.asarray(samples_MC, dtype=np.complex128)
                save_run_npz(path, payload)
                run_summary = {
                    "file": fname,
                    "n_qubits": int(n_qubits),
                    "n_layers": int(n_layers),
                    "train_depth": int(train_depth),
                    "m": int(tw_data["m"]),
                    "runtime_seconds": diag["runtime_seconds"],
                    "mc_status": mc_status,
                    "metrics": diag["metrics"],
                }
                index["runs"].append(run_summary)
                print(f"  saved {path}", flush=True)
                print(f"  relFrob Corr(TW,MC): {diag['metrics']['complex']['rel_frob_Corr_TW_vs_MC']:.3e}", flush=True)

    with open(out_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"\nSaved twirled correlation matrix run index to {out_dir / 'index.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
