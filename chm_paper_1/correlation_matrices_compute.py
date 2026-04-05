# correlation_matrices_compute.py
#
# STREAMING / LOW-RAM VERSION (HERMITIAN ONLY)
#
# Computes the Hermitian complex Pearson correlation matrix of the output Fourier
# coefficients a_ω(θ) of a variational quantum circuit, using two estimators:
#
#   Cov_MC(ω,ω') = E[(a_ω - μ_ω)(a_ω' - μ_ω')*]_Θ    (direct Monte Carlo)
#   Corr_MC      = Cov_MC / (σ_ω σ_ω')
#
#   Cov_C        = C_nz C_nz†     (drop k=0 / DC column from the C-matrix)
#   Corr_C       = Cov_C / (σ_ω σ_ω')
#
# where C_{ω,k} = E_θ[a_ω(θ) χ_k(θ)*] and χ_k(θ) = exp(i k·θ).
#
# This script includes block-bootstrap standard error estimation for all key
# scalar diagnostics (relative Frobenius error, cosine similarity, mean off-diagonal
# correlation). It is the primary correlation-matrix compute script for the paper.
#
# Uses a split-sample streaming estimator (no storage of all samples):
#   - C-split: accumulates C_sum = Σ a(θ) χ_k(θ)*
#   - V-split: accumulates Σ a and Σ a a†
#
# Outputs per-run .npz files containing:
#   - omega_grid
#   - K, C
#   - Cov_MC, Corr_MC, mean_a_V, var_MC
#   - Cov_C,  Corr_C,  var_C, zero_idx
#   - diag_json (diagnostics + block-bootstrap SE)
#
# Requirements: numpy, jax, pennylane, tqdm

import os
import json
import argparse
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import itertools as it
from tqdm import trange
from datetime import datetime, timezone

from ansatze import CircuitSpec, build_qnode


# ======================================================
# 1) Generate harmonic set K (parameter-side harmonics)
# ======================================================

def generate_K(m, max_hw=1, max_K=None):
    """Generate subset of K ⊂ {-1,0,1}^m with Hamming weight <= max_hw (implemented up to 3)."""
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

    Note: f(x;θ) is assumed real-valued (expectation of Hermitian observable),
    but a_ω(θ) is complex in general.
    """
    omega_grid = np.asarray(omega_grid, dtype=int)
    x_grid = np.linspace(x_min, x_max, n_x, endpoint=False)
    dx = (x_max - x_min) / float(n_x)

    Phi = np.exp(-1j * np.outer(omega_grid, x_grid)).astype(np.complex128)  # (n_omega, n_x)
    x_array = jnp.array(x_grid)

    f_x = jax.jit(jax.vmap(qnode, in_axes=(0, None)))

    def a_batch(theta_batch_np):
        theta_batch_np = np.asarray(theta_batch_np, dtype=np.float64)
        B = theta_batch_np.shape[0]
        n_omega = Phi.shape[0]
        out = np.empty((B, n_omega), dtype=np.complex128)
        for i in range(B):
            theta_i = jnp.asarray(theta_batch_np[i])
            fvals = np.asarray(f_x(x_array, theta_i), dtype=np.float64)  # (n_x,)
            out[i, :] = (Phi @ fvals) * dx
        return out

    return a_batch


# ======================================================
# 3) Core math helpers (Hermitian covariance/correlation)
# ======================================================

def _find_zero_idx(K):
    is_zero = np.all(K == 0, axis=1)
    zero_idx = np.where(is_zero)[0]
    if len(zero_idx) != 1:
        raise ValueError("Expected exactly one zero vector in K.")
    return int(zero_idx[0])


def compute_cov_corr_from_C(C, K, eps=1e-14):
    """
    Drop k=0 column (DC), then:
      Cov_C  = C_nz C_nz†
      Corr_C = D^{-1/2} Cov_C D^{-1/2},  D=diag(Cov_C)
    """
    n_omega, n_K = C.shape
    zero_idx = _find_zero_idx(K)

    mask = np.ones(n_K, dtype=bool)
    mask[zero_idx] = False
    C_nz = C[:, mask]

    Cov_C = C_nz @ C_nz.conj().T
    var_C = np.real(np.diag(Cov_C)).astype(np.float64)

    sigma = np.sqrt(np.maximum(var_C, 0.0))
    sigma[sigma < eps] = 1.0
    Corr_C = Cov_C / np.outer(sigma, sigma)
    return Cov_C, Corr_C, var_C, zero_idx


def covcorr_from_suffstats(sum_a, sum_aaH, S, eps=1e-14):
    """Complex centred covariance and (Hermitian) correlation from sufficient statistics."""
    if S <= 0:
        raise ValueError("No samples accumulated.")
    mean_a = sum_a / float(S)
    second = sum_aaH / float(S)
    Cov = second - np.outer(mean_a, mean_a.conj())
    var = np.real(np.diag(Cov)).astype(np.float64)
    sigma = np.sqrt(np.maximum(var, 0.0))
    sigma[sigma < eps] = 1.0
    Corr = Cov / np.outer(sigma, sigma)
    return mean_a, Cov, Corr, var


def offdiag_vector(A):
    A = np.asarray(A)
    n = A.shape[0]
    iu = np.triu_indices(n, k=1)
    return A[iu]


def cosine_similarity(u, v, eps=1e-14):
    u = np.asarray(u)
    v = np.asarray(v)
    num = np.vdot(u, v)
    den = (np.linalg.norm(u) * np.linalg.norm(v) + eps)
    return float(np.real(num / den))


def rel_frob(A, B, eps=1e-14):
    return float(np.linalg.norm(A - B, ord="fro") / (np.linalg.norm(B, ord="fro") + eps))


def mean_abs_offdiag(A):
    v = offdiag_vector(A)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    return float(np.mean(np.abs(v)))


def spectral_metrics_real(A_mc, A_c, eps=1e-14):
    """Eigenvalue distance for a real matrix: symmetrise then eigvalsh."""
    A_mc = np.asarray(A_mc)
    A_c = np.asarray(A_c)
    A_mc = 0.5 * (A_mc + A_mc.T)
    A_c = 0.5 * (A_c + A_c.T)
    lam_mc = np.linalg.eigvalsh(A_mc)
    lam_c = np.linalg.eigvalsh(A_c)
    diff = lam_mc - lam_c
    return {
        "eig_rel_l2": float(np.linalg.norm(diff) / (np.linalg.norm(lam_c) + eps)),
        "eig_cos": cosine_similarity(lam_mc, lam_c, eps=eps),
        "eig_maxabs": float(np.max(np.abs(diff))),
        "eig_max_mc": float(np.max(lam_mc)),
        "eig_max_c": float(np.max(lam_c)),
    }


def diagnostics_complex(Cov_MC, Corr_MC, Cov_C, Corr_C, eps=1e-14, with_spectral=True):
    """Diagnostics for complex (Hermitian) matrices (MC vs C)."""
    n = Corr_C.shape[0]
    iu = np.triu_indices(n, k=1)

    vC_cor = Corr_C[iu]
    vM_cor = Corr_MC[iu]
    vC_cov = Cov_C[iu]
    vM_cov = Cov_MC[iu]

    out = {
        "cov_rel_frob": rel_frob(Cov_MC, Cov_C, eps=eps),
        "corr_rel_frob": rel_frob(Corr_MC, Corr_C, eps=eps),
        "cov_offdiag_cosine": cosine_similarity(vM_cov, vC_cov, eps=eps),
        "corr_offdiag_cosine": cosine_similarity(vM_cor, vC_cor, eps=eps),
        "cov_abs_rel_frob": rel_frob(np.abs(Cov_MC), np.abs(Cov_C), eps=eps),
        "corr_abs_rel_frob": rel_frob(np.abs(Corr_MC), np.abs(Corr_C), eps=eps),
        "corr_mean_abs_offdiag": mean_abs_offdiag(Corr_MC),
        "corrC_mean_abs_offdiag": mean_abs_offdiag(Corr_C),
        "corr_offdiag_cosine_abs": cosine_similarity(np.abs(vM_cor), np.abs(vC_cor), eps=eps),
    }

    if with_spectral:
        out.update({
            "spec_re": spectral_metrics_real(np.real(Corr_MC), np.real(Corr_C), eps=eps),
            "spec_im": spectral_metrics_real(np.imag(Corr_MC), np.imag(Corr_C), eps=eps),
        })
    return out


def diagnostics_real_parts(A_MC, A_C, eps=1e-14, with_spectral=True):
    """
    Part-separated diagnostics for a complex correlation matrix:
      - computes metrics for Re(A) and Im(A) separately.
    """
    out = {}
    for tag, Am, Ac in (("re", np.real(A_MC), np.real(A_C)), ("im", np.imag(A_MC), np.imag(A_C))):
        n = Ac.shape[0]
        iu = np.triu_indices(n, k=1)
        vM = Am[iu]
        vC = Ac[iu]
        out[f"corr_rel_frob_{tag}"] = rel_frob(Am, Ac, eps=eps)
        out[f"corr_offdiag_cosine_{tag}"] = cosine_similarity(vM, vC, eps=eps)
        out[f"corr_mean_abs_offdiag_{tag}"] = mean_abs_offdiag(Am)
        out[f"corrC_mean_abs_offdiag_{tag}"] = mean_abs_offdiag(Ac)
        if with_spectral:
            out[f"spec_{tag}"] = spectral_metrics_real(Am, Ac, eps=eps)
    return out


def _block_se(values):
    """Standard error of the mean from block values."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return np.nan, int(v.size)
    return float(np.std(v, ddof=1) / np.sqrt(v.size)), int(v.size)


# ======================================================
# 4) Saving utilities
# ======================================================

def save_run_npz(path, payload: dict):
    diag = payload.pop("diag")
    payload["diag_json"] = np.array(json.dumps(diag, sort_keys=True))
    np.savez_compressed(path, **payload)


# ======================================================
# 5) CLI
# ======================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute ONLY Hermitian complex Pearson Corr(a_ω(θ)) (MC vs C) + diagnostics + block-SE (streaming, low-RAM)."
    )
    p.add_argument("--out_dir", type=str, default="outputs_correlation_matrices")
    p.add_argument("--seed", type=int, default=1234)

    # experiment grid
    p.add_argument("--n_qubits_list", type=int, nargs="+", default=[3])
    p.add_argument("--train_depth_list", type=int, nargs="+", default=[1, 2, 3, 4, 5])

    # circuit
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--encoder_scale", type=float, default=1.0)
    p.add_argument("--ansatz", type=str, default="CIRCUIT_17")
    p.add_argument("--encoder_axis", type=str, default="RX")
    p.add_argument("--obs_kind", type=str, default="OZ")

    # x-side Fourier
    p.add_argument("--n_x", type=int, default=256)
    p.add_argument("--x_min", type=float, default=0.0)
    p.add_argument("--x_max", type=float, default=2 * np.pi)

    # theta-side sampling
    p.add_argument("--n_theta_samples", type=int, default=4096)
    p.add_argument("--split_fraction_for_C", type=float, default=0.5)

    # streaming controls
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--k_block", type=int, default=5000)

    # omega / K
    p.add_argument("--max_omega", type=int, default=15)
    p.add_argument("--max_hw_for_K", type=int, default=3)
    p.add_argument("--max_K_cap", type=int, default=30000)

    p.add_argument("--no_spectral", action="store_true", help="Disable spectral diagnostics (faster).")

    return p.parse_args()


# ======================================================
# 6) Main
# ======================================================

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # FULL omega grid
    omega_grid = np.arange(-args.max_omega, args.max_omega + 1, dtype=int)
    n_omega = len(omega_grid)
    print(f"[INFO] Omega grid size (FULL) = {n_omega}")

    index = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "out_dir": args.out_dir,
        "global_config": {
            "n_qubits_list": args.n_qubits_list,
            "train_depth_list": args.train_depth_list,
            "n_layers": args.n_layers,
            "encoder_scale": args.encoder_scale,
            "ansatz": args.ansatz,
            "encoder_axis": args.encoder_axis,
            "obs_kind": args.obs_kind,
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
            "no_spectral": bool(args.no_spectral),
        },
        "runs": [],
    }

    for n_qubits in args.n_qubits_list:
        for train_depth in args.train_depth_list:
            print(f"\n[INFO] Run: n_qubits={n_qubits}, train_depth={train_depth}")

            spec = CircuitSpec(
                ansatz=str(args.ansatz),
                n_qubits=int(n_qubits),
                n_layers=int(args.n_layers),
                train_depth=int(train_depth),
                encoder_axis=str(args.encoder_axis),
                encoder_scale=float(args.encoder_scale),
                obs_kind=str(args.obs_kind),
                device_name="default.qubit",
                diff_method="parameter-shift",
                jit=True,
            )
            qnode, m = build_qnode(spec)
            print(f"  [INFO] m = {m}")

            K = generate_K(m, max_hw=args.max_hw_for_K, max_K=args.max_K_cap)
            n_K = K.shape[0]
            print(f"  [INFO] |K| = {n_K}")

            a_batch_fn = make_a_theta_batch_fn(
                qnode=qnode,
                omega_grid=omega_grid,
                n_x=args.n_x,
                x_min=args.x_min,
                x_max=args.x_max,
            )

            # ---- Global accumulators (V-split: MC Hermitian Cov/Corr) ----
            sum_a_V = np.zeros((n_omega,), dtype=np.complex128)
            sum_aaH_V = np.zeros((n_omega, n_omega), dtype=np.complex128)
            S_V = 0

            # ---- C-split accumulator for C-matrix ----
            C_sum = np.zeros((n_omega, n_K), dtype=np.complex128)
            S_C = 0

            # ---- Block diagnostics containers (for SE) ----
            block_diag_complex_frob = []
            block_diag_complex_cos = []
            block_diag_complex_mean_offdiag_mc = []
            block_diag_complex_mean_offdiag_c = []

            block_diag_complex_re_frob = []
            block_diag_complex_re_cos = []
            block_diag_complex_re_mean_offdiag_mc = []
            block_diag_complex_re_mean_offdiag_c = []
            block_diag_complex_re_eL2 = []
            block_diag_complex_re_eCos = []
            block_diag_complex_re_eMax = []

            block_diag_complex_im_frob = []
            block_diag_complex_im_cos = []
            block_diag_complex_im_mean_offdiag_mc = []
            block_diag_complex_im_mean_offdiag_c = []
            block_diag_complex_im_eL2 = []
            block_diag_complex_im_eCos = []
            block_diag_complex_im_eMax = []

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
                a_batch = a_batch_fn(theta_batch)  # (Bb, n_omega)

                # On-the-fly split
                is_C = rng.random(Bb) < float(args.split_fraction_for_C)
                if (S_C == 0 and not np.any(is_C)) and (Bb >= 2):
                    is_C[0] = True
                if (S_V == 0 and np.all(is_C)) and (Bb >= 2):
                    is_C[0] = False

                idxV = np.where(~is_C)[0]
                idxC = np.where(is_C)[0]

                # ---------------- V-split updates (global) ----------------
                if idxV.size > 0:
                    aV = a_batch[idxV]  # (Bv, n_omega)
                    sum_a_V += np.sum(aV, axis=0)
                    sum_aaH_V += aV.T @ aV.conj()
                    S_V += int(aV.shape[0])

                # ---------------- C-split updates (global) ----------------
                if idxC.size > 0:
                    aC = a_batch[idxC]          # (Bc, n_omega)
                    thetaC = theta_batch[idxC]  # (Bc, m)

                    k_block = int(args.k_block)
                    for ks in range(0, n_K, k_block):
                        ke = min(n_K, ks + k_block)
                        Ksl = K[ks:ke].astype(np.float64)  # (kb, m)
                        phases = thetaC @ Ksl.T            # (Bc, kb)
                        char_conj = np.exp(-1j * phases)   # (Bc, kb)
                        C_sum[:, ks:ke] += aC.T @ char_conj

                    S_C += int(aC.shape[0])

                # ---------------- Block diagnostics for SE ----------------
                if (idxV.size >= 4) and (idxC.size >= 4):
                    # block MC Hermitian corr
                    aVb = a_batch[idxV]
                    mean_b = np.mean(aVb, axis=0)
                    a0b = aVb - mean_b[None, :]
                    Cov_MCb = (a0b.T @ a0b.conj()) / float(a0b.shape[0])
                    var_MCb = np.real(np.diag(Cov_MCb))
                    sig = np.sqrt(np.maximum(var_MCb, 0.0))
                    sig[sig < 1e-14] = 1.0
                    Corr_MCb = Cov_MCb / np.outer(sig, sig)

                    # block C estimate and corr
                    aCb = a_batch[idxC]
                    thetaCb = theta_batch[idxC]

                    C_b = np.zeros((n_omega, n_K), dtype=np.complex128)
                    k_block = int(args.k_block)
                    for ks in range(0, n_K, k_block):
                        ke = min(n_K, ks + k_block)
                        Ksl = K[ks:ke].astype(np.float64)
                        phases = thetaCb @ Ksl.T
                        char_conj = np.exp(-1j * phases)
                        C_b[:, ks:ke] = (aCb.T @ char_conj) / float(aCb.shape[0])

                    Cov_Cb, Corr_Cb, _var_Cb, _ = compute_cov_corr_from_C(C_b, K)

                    # full complex Hermitian block scalars
                    block_diag_complex_frob.append(rel_frob(Corr_MCb, Corr_Cb))
                    block_diag_complex_cos.append(
                        cosine_similarity(offdiag_vector(Corr_MCb), offdiag_vector(Corr_Cb))
                    )
                    block_diag_complex_mean_offdiag_mc.append(mean_abs_offdiag(Corr_MCb))
                    block_diag_complex_mean_offdiag_c.append(mean_abs_offdiag(Corr_Cb))

                    # Re/Im part block scalars (+ spectral)
                    A_re_mc = np.real(Corr_MCb); A_re_c = np.real(Corr_Cb)
                    A_im_mc = np.imag(Corr_MCb); A_im_c = np.imag(Corr_Cb)

                    iu = np.triu_indices(A_re_c.shape[0], k=1)
                    block_diag_complex_re_frob.append(rel_frob(A_re_mc, A_re_c))
                    block_diag_complex_re_cos.append(cosine_similarity(A_re_mc[iu], A_re_c[iu]))
                    block_diag_complex_re_mean_offdiag_mc.append(mean_abs_offdiag(A_re_mc))
                    block_diag_complex_re_mean_offdiag_c.append(mean_abs_offdiag(A_re_c))
                    r = spectral_metrics_real(A_re_mc, A_re_c)
                    block_diag_complex_re_eL2.append(r["eig_rel_l2"])
                    block_diag_complex_re_eCos.append(r["eig_cos"])
                    block_diag_complex_re_eMax.append(r["eig_maxabs"])

                    block_diag_complex_im_frob.append(rel_frob(A_im_mc, A_im_c))
                    block_diag_complex_im_cos.append(cosine_similarity(A_im_mc[iu], A_im_c[iu]))
                    block_diag_complex_im_mean_offdiag_mc.append(mean_abs_offdiag(A_im_mc))
                    block_diag_complex_im_mean_offdiag_c.append(mean_abs_offdiag(A_im_c))
                    r = spectral_metrics_real(A_im_mc, A_im_c)
                    block_diag_complex_im_eL2.append(r["eig_rel_l2"])
                    block_diag_complex_im_eCos.append(r["eig_cos"])
                    block_diag_complex_im_eMax.append(r["eig_maxabs"])

            if S_C <= 0 or S_V <= 0:
                raise RuntimeError(f"Split failed: S_C={S_C}, S_V={S_V}.")

            print(f"  [INFO] split counts: S_C={S_C}, S_V={S_V}")

            with_spectral = (not args.no_spectral)

            # ---------------- Finalise MC (Hermitian) ----------------
            mean_a_V, Cov_MC, Corr_MC, var_MC = covcorr_from_suffstats(sum_a_V, sum_aaH_V, S_V)

            # ---------------- Finalise C-matrix and C Hermitian corr ----------------
            C = C_sum / float(S_C)
            Cov_C, Corr_C, var_C, zero_idx = compute_cov_corr_from_C(C, K)

            # ---------------- Hard sanity checks ----------------
            diag_mc_err = float(np.max(np.abs(np.diag(Corr_MC) - 1.0)))
            diag_c_err  = float(np.max(np.abs(np.diag(Corr_C)  - 1.0)))
            herm_mc_err = float(np.max(np.abs(Corr_MC - Corr_MC.conj().T)))
            herm_c_err  = float(np.max(np.abs(Corr_C  - Corr_C.conj().T)))
            print(f"  [CHECK] max|diag(Corr_MC)-1| = {diag_mc_err:.3e}")
            print(f"  [CHECK] max|diag(Corr_C)-1|  = {diag_c_err:.3e}")
            print(f"  [CHECK] max|Corr_MC-Corr_MC^H| = {herm_mc_err:.3e}")
            print(f"  [CHECK] max|Corr_C-Corr_C^H|   = {herm_c_err:.3e}")

            diag_complex = diagnostics_complex(Cov_MC, Corr_MC, Cov_C, Corr_C, with_spectral=with_spectral)
            diag_complex_parts = diagnostics_real_parts(Corr_MC, Corr_C, with_spectral=with_spectral)

            # ---------------- Block-SE (standard errors from block values) ----------------
            def se_pack(vals):
                se, nblk = _block_se(vals)
                return se, nblk

            se_complex_frob, nblk_c = se_pack(block_diag_complex_frob)
            se_complex_cos, _ = se_pack(block_diag_complex_cos)
            se_complex_mean_offdiag_mc, _ = se_pack(block_diag_complex_mean_offdiag_mc)
            se_complex_mean_offdiag_c, _ = se_pack(block_diag_complex_mean_offdiag_c)

            se_c_re_frob, nblk_re = se_pack(block_diag_complex_re_frob)
            se_c_re_cos, _ = se_pack(block_diag_complex_re_cos)
            se_re_mean_offdiag_mc, _ = se_pack(block_diag_complex_re_mean_offdiag_mc)
            se_re_mean_offdiag_c, _ = se_pack(block_diag_complex_re_mean_offdiag_c)
            se_c_re_eL2, _ = se_pack(block_diag_complex_re_eL2)
            se_c_re_eCos, _ = se_pack(block_diag_complex_re_eCos)
            se_c_re_eMax, _ = se_pack(block_diag_complex_re_eMax)

            se_c_im_frob, nblk_im = se_pack(block_diag_complex_im_frob)
            se_c_im_cos, _ = se_pack(block_diag_complex_im_cos)
            se_im_mean_offdiag_mc, _ = se_pack(block_diag_complex_im_mean_offdiag_mc)
            se_im_mean_offdiag_c, _ = se_pack(block_diag_complex_im_mean_offdiag_c)
            se_c_im_eL2, _ = se_pack(block_diag_complex_im_eL2)
            se_c_im_eCos, _ = se_pack(block_diag_complex_im_eCos)
            se_c_im_eMax, _ = se_pack(block_diag_complex_im_eMax)

            diag = {
                "complex": {
                    **diag_complex,
                    "parts": diag_complex_parts,
                    "sanity": {
                        "max_abs_diag_mc_minus_1": diag_mc_err,
                        "max_abs_diag_c_minus_1": diag_c_err,
                        "max_abs_hermiticity_mc": herm_mc_err,
                        "max_abs_hermiticity_c": herm_c_err,
                    },
                },
                "block_se": {
                    "n_blocks_used_complex": nblk_c,
                    "n_blocks_used_complex_re": nblk_re,
                    "n_blocks_used_complex_im": nblk_im,
                    "complex": {
                        "corr_rel_frob_se": se_complex_frob,
                        "corr_offdiag_cosine_se": se_complex_cos,
                        "corr_mean_abs_offdiag_mc_se": se_complex_mean_offdiag_mc,
                        "corr_mean_abs_offdiag_c_se": se_complex_mean_offdiag_c,
                    },
                    "complex_re": {
                        "corr_rel_frob_se": se_c_re_frob,
                        "corr_offdiag_cosine_se": se_c_re_cos,
                        "corr_mean_abs_offdiag_mc_se": se_re_mean_offdiag_mc,
                        "corr_mean_abs_offdiag_c_se": se_re_mean_offdiag_c,
                        "eig_rel_l2_se": se_c_re_eL2,
                        "eig_cos_se": se_c_re_eCos,
                        "eig_maxabs_se": se_c_re_eMax,
                    },
                    "complex_im": {
                        "corr_rel_frob_se": se_c_im_frob,
                        "corr_offdiag_cosine_se": se_c_im_cos,
                        "corr_mean_abs_offdiag_mc_se": se_im_mean_offdiag_mc,
                        "corr_mean_abs_offdiag_c_se": se_im_mean_offdiag_c,
                        "eig_rel_l2_se": se_c_im_eL2,
                        "eig_cos_se": se_c_im_eCos,
                        "eig_maxabs_se": se_c_im_eMax,
                    },
                },
            }

            # ---------------- Save ----------------
            tag = (
                f"corr_matrices_{args.ansatz}_{args.encoder_axis}_{args.obs_kind}"
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
                "encoder_axis": np.array(str(args.encoder_axis)),
                "ansatz": np.array(str(args.ansatz)),
                "obs_kind": np.array(str(args.obs_kind)),
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

                # Hermitian covariance/correlation
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