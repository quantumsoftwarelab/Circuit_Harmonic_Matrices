# correlation_compute.py
#
# STREAMING / LOW-RAM VERSION
#
# Computes and compares the covariance/correlation matrices of the output Fourier
# coefficients a_ω(θ) of a variational quantum circuit, using two estimators:
#
#   (A) C-matrix prediction:
#       Cov_C  = C_nz C_nz†            (drop the k=0 / DC column from C)
#       Corr_C = D_C^{-1/2} Cov_C D_C^{-1/2}
#
#       where C_{ω,k} = E_θ[a_ω(θ) χ_k(θ)*] is the cross-correlation between
#       output harmonics and parameter-space characters χ_k(θ) = exp(i k·θ).
#
#   (B) Direct Monte Carlo:
#       Cov_MC, Corr_MC estimated from centred coefficient samples a(θ)
#
# Uses a split-sample streaming estimator (no storage of all samples):
#   - C-split: accumulates C_sum = Σ a(θ) χ_k(θ)*  ->  C = C_sum / S_C
#   - V-split: accumulates Σ a and Σ a a†           ->  Cov_MC, Corr_MC at end
#
# Saves per-run .npz files containing:
#   K, C, Cov_C, Corr_C, Cov_MC, Corr_MC, mean_a_V, var_a_V, config fields.
#
# Note: this is a simpler version without block-bootstrap standard errors.
# For the full version with block-SE, use correlation_matrices_compute.py.
#
# Requirements: numpy, jax, pennylane, tqdm

import os
import json
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import itertools as it
from tqdm import trange
from datetime import datetime, timezone

from ansatze import CircuitSpec, build_qnode


# ======================================================
# 1) Generate harmonic set K (parameter-side harmonics)
# ======================================================

def generate_K(m, max_hw=1, max_K=None):
    """
    Generate subset of K ⊂ {-1,0,1}^m with Hamming weight <= max_hw (implemented up to 3).
    """
    K_list = []
    zero = np.zeros(m, dtype=np.int8)
    K_list.append(zero)

    def maybe_stop():
        return (max_K is not None) and (len(K_list) >= max_K)

    if max_hw >= 1 and not maybe_stop():
        for a in range(m):
            e = np.zeros(m, dtype=np.int8)
            e[a] = 1
            K_list.append(e.copy())
            K_list.append((-e).copy())
            if maybe_stop():
                return np.stack(K_list, axis=0)

    if max_hw >= 2 and not maybe_stop():
        for a, b in it.combinations(range(m), 2):
            for s_a, s_b in it.product([1, -1], repeat=2):
                v = np.zeros(m, dtype=np.int8)
                v[a] = np.int8(s_a)
                v[b] = np.int8(s_b)
                K_list.append(v)
                if maybe_stop():
                    return np.stack(K_list, axis=0)

    if max_hw >= 3 and not maybe_stop():
        for indices in it.combinations(range(m), 3):
            for signs in it.product([1, -1], repeat=3):
                v = np.zeros(m, dtype=np.int8)
                for idx, s in zip(indices, signs):
                    v[idx] = np.int8(s)
                K_list.append(v)
                if maybe_stop():
                    return np.stack(K_list, axis=0)

    return np.stack(K_list, axis=0)


# ======================================================
# 2) Streaming: batch a_omega(theta) estimator
# ======================================================

def make_a_theta_batch_fn(qnode, omega_grid, n_x, x_min, x_max):
    """
    Returns a function a_batch(theta_batch) -> (B, n_omega) complex128
    that estimates a_ω(θ) via discrete Fourier transform over x.

    IMPORTANT:
      - Vectorises over x (n_x) using jax.vmap
      - Loops over theta_batch in Python to avoid nested batch dimensions
        that can break PennyLane default.qubit under JAX jit/pure_callback.
    """
    omega_grid = np.asarray(omega_grid, dtype=int)
    x_grid = np.linspace(x_min, x_max, n_x, endpoint=False)
    dx = (x_max - x_min) / float(n_x)

    # (n_omega, n_x)
    Phi = np.exp(-1j * np.outer(omega_grid, x_grid)).astype(np.complex128)

    x_array = jnp.array(x_grid)

    # f(x, theta) evaluated for all x in x_grid, for a single theta
    # Output shape: (n_x,)
    f_x = jax.jit(jax.vmap(qnode, in_axes=(0, None)))

    def a_batch(theta_batch_np):
        theta_batch_np = np.asarray(theta_batch_np, dtype=np.float64)
        B = theta_batch_np.shape[0]
        n_omega = Phi.shape[0]

        out = np.empty((B, n_omega), dtype=np.complex128)

        # Loop over theta (no nested vmap)
        for i in range(B):
            theta_i = jnp.asarray(theta_batch_np[i])
            fvals = np.asarray(f_x(x_array, theta_i), dtype=np.float64)  # (n_x,)
            out[i, :] = (Phi @ fvals) * dx

        return out

    return a_batch


# ======================================================
# 3) Cov/Corr from C with k=0 removed
# ======================================================

def compute_cov_corr_from_C(C, K, eps=1e-14):
    """
    Drop k=0 column (DC), then:
      Cov_C  = C_nz C_nz†
      Corr_C = D^{-1/2} Cov_C D^{-1/2},  D=diag(Cov_C)
    """
    n_omega, n_K = C.shape
    is_zero = np.all(K == 0, axis=1)
    zero_idx = np.where(is_zero)[0]
    if len(zero_idx) != 1:
        raise ValueError("Expected exactly one zero vector in K.")
    zero_idx = int(zero_idx[0])

    mask = np.ones(n_K, dtype=bool)
    mask[zero_idx] = False
    C_nz = C[:, mask]

    Cov_C = C_nz @ C_nz.conj().T
    var_C = np.real(np.diag(Cov_C)).astype(np.float64)

    sigma = np.sqrt(np.maximum(var_C, 0.0))
    sigma[sigma < eps] = 1.0
    Corr_C = Cov_C / np.outer(sigma, sigma)

    return Cov_C, Corr_C, var_C, zero_idx


# ======================================================
# 4) Cov/Corr from streaming sufficient statistics
# ======================================================

def covcorr_from_suffstats(sum_a, sum_aaH, S, eps=1e-14):
    """
    Given:
      sum_a   = Σ a_s            shape (n_omega,)
      sum_aaH = Σ a_s a_s†       shape (n_omega, n_omega)
      S       = number of samples

    Returns:
      mean_a, Cov, Corr, var
    """
    if S <= 0:
        raise ValueError("No samples accumulated for V-split.")

    mean_a = sum_a / float(S)
    second = sum_aaH / float(S)
    Cov = second - np.outer(mean_a, mean_a.conj())

    var = np.real(np.diag(Cov)).astype(np.float64)
    sigma = np.sqrt(np.maximum(var, 0.0))
    sigma[sigma < eps] = 1.0
    Corr = Cov / np.outer(sigma, sigma)

    return mean_a, Cov, Corr, var


# ======================================================
# 5) Diagnostics: Cov/Corr agreement
# ======================================================

def covcorr_diagnostics(Cov_MC, Corr_MC, Cov_C, Corr_C, eps=1e-14):
    """
    Returns scalar diagnostics comparing MC vs C-derived matrices.
    """
    Cov_MC = np.asarray(Cov_MC)
    Corr_MC = np.asarray(Corr_MC)
    Cov_C = np.asarray(Cov_C)
    Corr_C = np.asarray(Corr_C)

    def rel_frob(A, B):
        return float(np.linalg.norm(A - B, ord="fro") / (np.linalg.norm(B, ord="fro") + eps))

    n = Cov_C.shape[0]
    iu = np.triu_indices(n, k=1)

    vC_cov = Cov_C[iu]
    vM_cov = Cov_MC[iu]
    vC_cor = Corr_C[iu]
    vM_cor = Corr_MC[iu]

    def complex_cosine(u, v):
        num = np.vdot(u, v)
        den = (np.linalg.norm(u) * np.linalg.norm(v) + eps)
        return float(np.real(num / den))

    def complex_pearson(u, v):
        u0 = u - np.mean(u)
        v0 = v - np.mean(v)
        num = np.vdot(u0, v0)
        den = (np.linalg.norm(u0) * np.linalg.norm(v0) + eps)
        return float(np.real(num / den))

    return {
        "cov_rel_frob": rel_frob(Cov_MC, Cov_C),
        "corr_rel_frob": rel_frob(Corr_MC, Corr_C),
        "cov_offdiag_cosine": complex_cosine(vM_cov, vC_cov),
        "corr_offdiag_cosine": complex_cosine(vM_cor, vC_cor),
        "cov_offdiag_pearson": complex_pearson(vM_cov, vC_cov),
        "corr_offdiag_pearson": complex_pearson(vM_cor, vC_cor),
        "cov_abs_rel_frob": rel_frob(np.abs(Cov_MC), np.abs(Cov_C)),
        "corr_abs_rel_frob": rel_frob(np.abs(Corr_MC), np.abs(Corr_C)),
    }


# ======================================================
# 6) Saving utilities
# ======================================================

def save_run_npz(path, payload: dict):
    """
    Save dict of arrays/scalars. Diagnostics stored as JSON string in diag_json.
    """
    diag = payload.pop("diag")
    payload["diag_json"] = np.array(json.dumps(diag, sort_keys=True))
    np.savez_compressed(path, **payload)


# ======================================================
# 7) Main
# ======================================================

def parse_args():
    p = argparse.ArgumentParser(description="Compute Cov/Corr from C vs direct MC (streaming, low-RAM).")
    p.add_argument("--out_dir", type=str, default="outputs_correlation_compute")
    p.add_argument("--seed", type=int, default=1234)

    # experiment grid
    p.add_argument("--n_qubits_list", type=int, nargs="+", default=[3])
    p.add_argument("--train_depth_list", type=int, nargs="+", default=[1, 2, 3, 4, 5])

    # circuit
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--encoder_scale", type=float, default=1.0)

    # hardcoded circuit choice for now
    p.add_argument("--_ansatz_hardcoded", action="store_true", help=argparse.SUPPRESS)

    # x-side Fourier
    p.add_argument("--n_x", type=int, default=256)
    p.add_argument("--x_min", type=float, default=0.0)
    p.add_argument("--x_max", type=float, default=2 * np.pi)

    # theta-side sampling
    p.add_argument("--n_theta_samples", type=int, default=4096)
    p.add_argument("--split_fraction_for_C", type=float, default=0.5)

    # streaming controls
    p.add_argument("--batch_size", type=int, default=128, help="Theta batch size (controls RAM).")
    p.add_argument(
        "--k_block",
        type=int,
        default=5000,
        help="Block size for K-chunking when updating C_sum (controls RAM).",
    )

    # omega / K
    p.add_argument("--max_omega", type=int, default=15)
    p.add_argument("--max_hw_for_K", type=int, default=3)
    p.add_argument("--max_K_cap", type=int, default=30000)

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    omega_grid = np.arange(-args.max_omega, args.max_omega + 1, dtype=int)
    n_omega = len(omega_grid)
    print(f"[INFO] Omega grid size = {n_omega}")

    # Default circuit selection — edit these to change the circuit being analysed.
    ansatz = "CIRCUIT_17"
    encoder_axis = "RX"
    obs_kind = "OZ"

    index = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "out_dir": args.out_dir,
        "global_config": {
            "n_qubits_list": args.n_qubits_list,
            "train_depth_list": args.train_depth_list,
            "n_layers": args.n_layers,
            "encoder_scale": args.encoder_scale,
            "n_x": args.n_x,
            "x_min": args.x_min,
            "x_max": args.x_max,
            "n_theta_samples": args.n_theta_samples,
            "split_fraction_for_C": args.split_fraction_for_C,
            "batch_size": args.batch_size,
            "k_block": args.k_block,
            "max_omega": args.max_omega,
            "max_hw_for_K": args.max_hw_for_K,
            "max_K_cap": args.max_K_cap,
            "seed": args.seed,
            "ansatz": ansatz,
            "encoder_axis": encoder_axis,
            "obs_kind": obs_kind,
        },
        "runs": [],
    }

    for n_qubits in args.n_qubits_list:
        for train_depth in args.train_depth_list:
            print(f"\n[INFO] Run: n_qubits={n_qubits}, train_depth={train_depth}")

            spec = CircuitSpec(
                ansatz=ansatz,
                n_qubits=int(n_qubits),
                n_layers=int(args.n_layers),
                train_depth=int(train_depth),
                encoder_axis=encoder_axis,
                encoder_scale=float(args.encoder_scale),
                obs_kind=obs_kind,
                device_name="default.qubit",
                diff_method="parameter-shift",
                jit=True,
            )
            qnode, m = build_qnode(spec)
            print(f"  [INFO] m = {m}")

            K = generate_K(m, max_hw=args.max_hw_for_K, max_K=args.max_K_cap)
            n_K = K.shape[0]
            print(f"  [INFO] |K| = {n_K}")

            # Prebuild batched a_omega(theta) estimator
            a_batch_fn = make_a_theta_batch_fn(
                qnode=qnode,
                omega_grid=omega_grid,
                n_x=args.n_x,
                x_min=args.x_min,
                x_max=args.x_max,
            )

            # Streaming accumulators
            # V-split stats:
            sum_a_V = np.zeros((n_omega,), dtype=np.complex128)
            sum_aaH_V = np.zeros((n_omega, n_omega), dtype=np.complex128)
            S_V = 0

            # C-split stats:
            C_sum = np.zeros((n_omega, n_K), dtype=np.complex128)
            S_C = 0

            # Loop in theta batches
            S_total = int(args.n_theta_samples)
            B = int(args.batch_size)
            n_batches = int(np.ceil(S_total / B))

            for b in trange(n_batches, desc="Streaming θ batches"):
                b0 = b * B
                b1 = min(S_total, (b + 1) * B)
                Bb = b1 - b0
                if Bb <= 0:
                    continue

                theta_batch = rng.uniform(0.0, 2 * np.pi, size=(Bb, m)).astype(np.float64)

                # Compute a(θ) for this batch: (Bb, n_omega)
                a_batch = a_batch_fn(theta_batch)

                # On-the-fly split
                is_C = rng.random(Bb) < float(args.split_fraction_for_C)
                # Guard against pathological tiny batches; keep both non-empty when possible
                if (S_C == 0 and not np.any(is_C)) and (Bb >= 2):
                    is_C[0] = True
                if (S_V == 0 and np.all(is_C)) and (Bb >= 2):
                    is_C[0] = False

                # V-split updates
                idxV = np.where(~is_C)[0]
                if idxV.size > 0:
                    aV = a_batch[idxV]  # (Bv, n_omega)
                    sum_a_V += np.sum(aV, axis=0)
                    sum_aaH_V += aV.conj().T @ aV
                    S_V += int(aV.shape[0])

                # C-split updates (with K-chunking)
                idxC = np.where(is_C)[0]
                if idxC.size > 0:
                    aC = a_batch[idxC]          # (Bc, n_omega)
                    thetaC = theta_batch[idxC]  # (Bc, m)

                    # For each K-block, update C_sum[:, sl] += aC.T @ exp(-i thetaC @ Ksl.T)
                    k_block = int(args.k_block)
                    for ks in range(0, n_K, k_block):
                        ke = min(n_K, ks + k_block)
                        Ksl = K[ks:ke].astype(np.float64)  # (kb, m) in {-1,0,1}
                        phases = thetaC @ Ksl.T            # (Bc, kb)
                        char_conj = np.exp(-1j * phases)   # (Bc, kb)
                        C_sum[:, ks:ke] += aC.T @ char_conj

                    S_C += int(aC.shape[0])

            if S_C <= 0 or S_V <= 0:
                raise RuntimeError(
                    f"Split failed: S_C={S_C}, S_V={S_V}. "
                    "Increase n_theta_samples or adjust split_fraction_for_C."
                )

            print(f"  [INFO] split counts: S_C={S_C}, S_V={S_V}")

            # Finalize V-split empirical Cov/Corr
            mean_a_V, Cov_MC, Corr_MC, var_MC = covcorr_from_suffstats(sum_a_V, sum_aaH_V, S_V)

            # Finalize C from C-split
            C = C_sum / float(S_C)
            Cov_C, Corr_C, var_C, zero_idx = compute_cov_corr_from_C(C, K)

            # Diagnostics
            diag = covcorr_diagnostics(Cov_MC, Corr_MC, Cov_C, Corr_C)
            print(
                "  [DIAG] "
                f"cov_rel_frob={diag['cov_rel_frob']:.6f}, "
                f"corr_rel_frob={diag['corr_rel_frob']:.6f}, "
                f"corr_offdiag_cos={diag['corr_offdiag_cosine']:.6f}, "
                f"corr_abs_rel_frob={diag['corr_abs_rel_frob']:.6f}"
            )

            # Save
            tag = (
                f"covcorr_{ansatz}_{encoder_axis}_{obs_kind}"
                f"_n{n_qubits}_depth{train_depth}_L{args.n_layers}"
                f"_mxw{args.max_omega}_hw{args.max_hw_for_K}_Kcap{args.max_K_cap}"
                f"_S{args.n_theta_samples}_SC{S_C}_SV{S_V}"
                f"_X{args.n_x}_B{args.batch_size}_Kb{args.k_block}_seed{args.seed}"
            )
            run_path = os.path.join(args.out_dir, f"{tag}.npz")

            payload = {
                # config
                "omega_grid": np.asarray(omega_grid, dtype=int),
                "n_qubits": int(n_qubits),
                "train_depth": int(train_depth),
                "n_layers": int(args.n_layers),
                "encoder_scale": float(args.encoder_scale),
                "encoder_axis": np.array(encoder_axis),
                "ansatz": np.array(ansatz),
                "obs_kind": np.array(obs_kind),
                "n_x": int(args.n_x),
                "x_min": float(args.x_min),
                "x_max": float(args.x_max),
                "n_theta_samples": int(args.n_theta_samples),
                "seed": int(args.seed),
                "split_fraction_for_C": float(args.split_fraction_for_C),
                "batch_size": int(args.batch_size),
                "k_block": int(args.k_block),
                "S_C": int(S_C),
                "S_V": int(S_V),
                "m": int(m),
                "zero_idx": int(zero_idx),

                # core objects
                "K": np.asarray(K, dtype=np.int8),
                "C": np.asarray(C, dtype=np.complex128),

                "Cov_C": np.asarray(Cov_C, dtype=np.complex128),
                "Corr_C": np.asarray(Corr_C, dtype=np.complex128),
                "var_C": np.asarray(var_C, dtype=np.float64),

                "Cov_MC": np.asarray(Cov_MC, dtype=np.complex128),
                "Corr_MC": np.asarray(Corr_MC, dtype=np.complex128),
                "mean_a_V": np.asarray(mean_a_V, dtype=np.complex128),
                "var_MC": np.asarray(var_MC, dtype=np.float64),

                # diagnostics
                "diag": diag,
            }

            save_run_npz(run_path, payload)
            print(f"  [INFO] Saved: {run_path}")

            index["runs"].append({
                "n_qubits": int(n_qubits),
                "train_depth": int(train_depth),
                "m": int(m),
                "|K|": int(n_K),
                "file": os.path.basename(run_path),
                "S_C": int(S_C),
                "S_V": int(S_V),
                "diag": diag,
            })

    index_path = os.path.join(args.out_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
    print(f"\n[INFO] Wrote index: {index_path}")


if __name__ == "__main__":
    main()