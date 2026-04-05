# qntk_compute.py
#
# STREAMING / LOW-RAM VERSION (HERMITIAN ONLY)
#
# Computes the parameter-averaged harmonic Quantum Neural Tangent Kernel (QNTK):
#
#   H(θ) = X(θ) X(θ)†,   where X_{ω,a} = ∂_{θ_a} a_ω(θ)
#   Hbar_MC = E_θ[ H(θ) ]    (direct Monte Carlo average)
#   Hbar_C  = C diag(||k||²) C†   (C-matrix prediction)
#
# where a_ω(θ) are the output Fourier coefficients and C is the C-matrix.
# The QNTK captures how gradient updates in parameter space project onto the
# frequency spectrum of the circuit output.
#
# Also computes correlation-normalised versions:
#   CorrH = H / (sqrt(diag(H)) sqrt(diag(H))^T)
# and runs the same diagnostics used for the correlation matrices (MC vs C).
#
# Uses a low-RAM streaming accumulator:
#   - Gradients computed via JAX autodiff (jacrev), accumulated with lax.scan
#     so the full (B, n_omega, n_omega) tensor is never materialised.
#   - C-matrix computed via the same split-sample estimator as correlation_matrices_compute.py.
#
# Outputs per-run .npz files containing:
#   - omega_grid, K, C
#   - Hbar_MC, CorrH_MC
#   - Hbar_C,  CorrH_C
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


def k_norm_sq(K):
    """||k||_2^2 for k in {-1,0,1}^m equals Hamming weight."""
    K = np.asarray(K, dtype=np.int8)
    return np.sum((K.astype(np.int32) ** 2), axis=1).astype(np.float64)


# ======================================================
# 2) a_omega(theta) estimator (same as your corr code)
# ======================================================

def make_a_theta_batch_fn(qnode, omega_grid, n_x, x_min, x_max):
    omega_grid = np.asarray(omega_grid, dtype=int)
    x_grid = np.linspace(x_min, x_max, n_x, endpoint=False)
    dx = (x_max - x_min) / float(n_x)

    Phi = np.exp(-1j * np.outer(omega_grid, x_grid)).astype(np.complex128)  # (n_omega, n_x)
    x_array = jnp.array(x_grid)

    f_x = jax.jit(jax.vmap(qnode, in_axes=(0, None)))  # (n_x,)

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
# 3) H(theta) batch SUM estimator via autodiff (low-RAM)
# ======================================================

def make_H_theta_batch_sum_fn(qnode, omega_grid, n_x, x_min, x_max):
    """
    Returns a function H_sum(theta_batch) -> (n_omega, n_omega) complex128
    where H_sum is the SUM over theta in the batch of H(theta)=X X†.

    Uses the robust gradient path (matching the correlation compute script):
      f_theta(theta) = vmap_x qnode(x,theta)            -> (n_x,)
      J(theta)       = jacrev_theta f_theta(theta)      -> (n_x,m)
      X(theta)       = Phi @ J(theta) * dx              -> (n_omega,m)
      H(theta)       = X X†                             -> (n_omega,n_omega)

    Low-RAM: accumulates with lax.scan, never materialises (B,n_x,m) or (B,n_omega,n_omega).
    """
    omega_grid = np.asarray(omega_grid, dtype=int)
    x_grid = np.linspace(x_min, x_max, n_x, endpoint=False)
    dx = (x_max - x_min) / float(n_x)

    Phi = jnp.asarray(np.exp(-1j * np.outer(omega_grid, x_grid)).astype(np.complex128))  # (o,x)
    x_array = jnp.asarray(x_grid)

    def f_theta(theta):
        return jax.vmap(qnode, in_axes=(0, None))(x_array, theta)  # (n_x,)

    jac_f_theta = jax.jit(jax.jacrev(f_theta))  # (n_x, m)

    def one_H(theta):
        J = jac_f_theta(theta)                           # (n_x, m)
        X = jnp.einsum("ox,xm->om", Phi, J) * dx         # (o, m)
        H = jnp.einsum("om,pm->op", X, jnp.conj(X))      # (o, o)
        return H

    @jax.jit
    def H_sum(theta_batch):
        # theta_batch: (B, m)
        def body(carry, th):
            return carry + one_H(th), None

        H0 = jnp.zeros((len(omega_grid), len(omega_grid)), dtype=jnp.complex128)
        Htot, _ = jax.lax.scan(body, H0, theta_batch)
        return Htot  # (o,o) SUM over batch

    def out(theta_batch_np):
        theta_batch = jnp.asarray(np.asarray(theta_batch_np, dtype=np.float64))
        return np.asarray(H_sum(theta_batch), dtype=np.complex128)

    return out


# ======================================================
# 4) Core helpers (same diagnostics machinery as your corr code)
# ======================================================

def _find_zero_idx(K):
    is_zero = np.all(K == 0, axis=1)
    zero_idx = np.where(is_zero)[0]
    if len(zero_idx) != 1:
        raise ValueError("Expected exactly one zero vector in K.")
    return int(zero_idx[0])


def corr_from_psd(H, eps=1e-14):
    """Correlation-normalisation for Hermitian PSD H."""
    H = np.asarray(H, dtype=np.complex128)
    var = np.real(np.diag(H)).astype(np.float64)
    sigma = np.sqrt(np.maximum(var, 0.0))
    sigma[sigma < eps] = 1.0
    Corr = H / np.outer(sigma, sigma)
    return Corr, var


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
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return np.nan, int(v.size)
    return float(np.std(v, ddof=1) / np.sqrt(v.size)), int(v.size)


# ======================================================
# 5) Saving utility
# ======================================================

def save_run_npz(path, payload: dict):
    diag = payload.pop("diag")
    payload["diag_json"] = np.array(json.dumps(diag, sort_keys=True))
    np.savez_compressed(path, **payload)


# ======================================================
# 6) CLI
# ======================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute Hermitian harmonic QNTK average Hbar (MC via autodiff vs C prediction) + diagnostics + block-SE (streaming)."
    )
    p.add_argument("--out_dir", type=str, default="outputs_qntk")
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

    # diff method
    p.add_argument(
        "--diff_method",
        type=str,
        default="backprop",
        help="Differentiation method for PennyLane (default: backprop).",
    )

    # x-side Fourier
    p.add_argument("--n_x", type=int, default=256)
    p.add_argument("--x_min", type=float, default=0.0)
    p.add_argument("--x_max", type=float, default=2 * np.pi)

    # theta-side sampling
    p.add_argument("--n_theta_samples", type=int, default=2048)
    p.add_argument("--split_fraction_for_C", type=float, default=0.5)

    # streaming controls
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--k_block", type=int, default=5000)

    # omega / K
    p.add_argument("--max_omega", type=int, default=15)
    p.add_argument("--max_hw_for_K", type=int, default=3)
    p.add_argument("--max_K_cap", type=int, default=30000)

    p.add_argument("--no_spectral", action="store_true", help="Disable spectral diagnostics (faster).")
    return p.parse_args()


# ======================================================
# 7) Main
# ======================================================

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    omega_grid = np.arange(-args.max_omega, args.max_omega + 1, dtype=int)
    n_omega = len(omega_grid)
    print(f"[INFO] Omega grid size (FULL) = {n_omega}")

    index = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "out_dir": args.out_dir,
        "global_config": vars(args),
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
                diff_method=str(args.diff_method),
                jit=True,
            )
            qnode, m = build_qnode(spec)
            print(f"  [INFO] m = {m}")

            K = generate_K(m, max_hw=args.max_hw_for_K, max_K=args.max_K_cap)
            n_K = K.shape[0]
            print(f"  [INFO] |K| = {n_K}")

            # weights ||k||^2 (k=0 gets weight 0 automatically)
            wK = k_norm_sq(K)  # (n_K,)
            sqrt_wK = np.sqrt(wK).astype(np.float64)

            a_batch_fn = make_a_theta_batch_fn(
                qnode=qnode, omega_grid=omega_grid, n_x=args.n_x, x_min=args.x_min, x_max=args.x_max
            )
            # LOW-RAM: returns SUM over batch, not per-sample stack
            H_sum_fn = make_H_theta_batch_sum_fn(
                qnode=qnode, omega_grid=omega_grid, n_x=args.n_x, x_min=args.x_min, x_max=args.x_max
            )

            # ---- Accumulators ----
            H_sum_V = np.zeros((n_omega, n_omega), dtype=np.complex128)
            S_V = 0

            C_sum = np.zeros((n_omega, n_K), dtype=np.complex128)
            S_C = 0

            # ---- Block diagnostics containers (for SE) ----
            block_frob = []
            block_cos = []
            block_mean_offdiag_mc = []
            block_mean_offdiag_c = []

            block_re_frob = []
            block_re_cos = []
            block_re_mean_offdiag_mc = []
            block_re_mean_offdiag_c = []
            block_re_eL2 = []
            block_re_eCos = []
            block_re_eMax = []

            block_im_frob = []
            block_im_cos = []
            block_im_mean_offdiag_mc = []
            block_im_mean_offdiag_c = []
            block_im_eL2 = []
            block_im_eCos = []
            block_im_eMax = []

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

                # a_batch needed for C-side regardless
                a_batch = a_batch_fn(theta_batch)  # (Bb, n_omega)

                # split
                is_C = rng.random(Bb) < float(args.split_fraction_for_C)
                if (S_C == 0 and not np.any(is_C)) and (Bb >= 2):
                    is_C[0] = True
                if (S_V == 0 and np.all(is_C)) and (Bb >= 2):
                    is_C[0] = False

                idxV = np.where(~is_C)[0]
                idxC = np.where(is_C)[0]

                # ---------------- V-split updates: accumulate SUM H(theta) ----------------
                if idxV.size > 0:
                    thetaV = theta_batch[idxV]
                    H_sum_V += H_sum_fn(thetaV)  # (o,o) SUM over thetaV
                    S_V += int(thetaV.shape[0])

                # ---------------- C-split updates: C matrix ----------------
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

                    S_C += int(thetaC.shape[0])

                # ---------------- Block diagnostics for SE ----------------
                if (idxV.size >= 2) and (idxC.size >= 2):
                    # Block MC estimate of Hbar:
                    thetaV = theta_batch[idxV]
                    Hbar_MCb = H_sum_fn(thetaV) / float(thetaV.shape[0])
                    CorrH_MCb, _ = corr_from_psd(Hbar_MCb)

                    # Block C estimate of C and hence Hbar_Cb:
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

                    # Weighted Gram for Hbar_Cb: (C * sqrt(w)) (C * sqrt(w))†
                    Cw = C_b * sqrt_wK[None, :]
                    Hbar_Cb = Cw @ Cw.conj().T
                    CorrH_Cb, _ = corr_from_psd(Hbar_Cb)

                    # Complex block scalars
                    block_frob.append(rel_frob(CorrH_MCb, CorrH_Cb))
                    block_cos.append(cosine_similarity(offdiag_vector(CorrH_MCb), offdiag_vector(CorrH_Cb)))
                    block_mean_offdiag_mc.append(mean_abs_offdiag(CorrH_MCb))
                    block_mean_offdiag_c.append(mean_abs_offdiag(CorrH_Cb))

                    # Re/Im part block scalars (+ spectral)
                    A_re_mc = np.real(CorrH_MCb); A_re_c = np.real(CorrH_Cb)
                    A_im_mc = np.imag(CorrH_MCb); A_im_c = np.imag(CorrH_Cb)

                    iu = np.triu_indices(A_re_c.shape[0], k=1)
                    block_re_frob.append(rel_frob(A_re_mc, A_re_c))
                    block_re_cos.append(cosine_similarity(A_re_mc[iu], A_re_c[iu]))
                    block_re_mean_offdiag_mc.append(mean_abs_offdiag(A_re_mc))
                    block_re_mean_offdiag_c.append(mean_abs_offdiag(A_re_c))
                    r = spectral_metrics_real(A_re_mc, A_re_c)
                    block_re_eL2.append(r["eig_rel_l2"])
                    block_re_eCos.append(r["eig_cos"])
                    block_re_eMax.append(r["eig_maxabs"])

                    block_im_frob.append(rel_frob(A_im_mc, A_im_c))
                    block_im_cos.append(cosine_similarity(A_im_mc[iu], A_im_c[iu]))
                    block_im_mean_offdiag_mc.append(mean_abs_offdiag(A_im_mc))
                    block_im_mean_offdiag_c.append(mean_abs_offdiag(A_im_c))
                    r = spectral_metrics_real(A_im_mc, A_im_c)
                    block_im_eL2.append(r["eig_rel_l2"])
                    block_im_eCos.append(r["eig_cos"])
                    block_im_eMax.append(r["eig_maxabs"])

            if S_C <= 0 or S_V <= 0:
                raise RuntimeError(f"Split failed: S_C={S_C}, S_V={S_V}.")
            print(f"  [INFO] split counts: S_C={S_C}, S_V={S_V}")

            with_spectral = (not args.no_spectral)

            # ---------------- Finalise MC average ----------------
            Hbar_MC = H_sum_V / float(S_V)
            CorrH_MC, varH_MC = corr_from_psd(Hbar_MC)

            # ---------------- Finalise C and C prediction ----------------
            C = C_sum / float(S_C)
            Cw = C * sqrt_wK[None, :]
            Hbar_C = Cw @ Cw.conj().T
            CorrH_C, varH_C = corr_from_psd(Hbar_C)

            # sanity checks
            diag_mc_err = float(np.max(np.abs(np.diag(CorrH_MC) - 1.0)))
            diag_c_err  = float(np.max(np.abs(np.diag(CorrH_C)  - 1.0)))
            herm_mc_err = float(np.max(np.abs(CorrH_MC - CorrH_MC.conj().T)))
            herm_c_err  = float(np.max(np.abs(CorrH_C  - CorrH_C.conj().T)))
            print(f"  [CHECK] max|diag(CorrH_MC)-1| = {diag_mc_err:.3e}")
            print(f"  [CHECK] max|diag(CorrH_C)-1|  = {diag_c_err:.3e}")
            print(f"  [CHECK] max|CorrH_MC-CorrH_MC^H| = {herm_mc_err:.3e}")
            print(f"  [CHECK] max|CorrH_C-CorrH_C^H|   = {herm_c_err:.3e}")

            # diagnostics: treat Cov=Hbar, Corr=CorrH
            diag_complex = diagnostics_complex(Hbar_MC, CorrH_MC, Hbar_C, CorrH_C, with_spectral=with_spectral)
            diag_complex_parts = diagnostics_real_parts(CorrH_MC, CorrH_C, with_spectral=with_spectral)

            # ---------------- Block-SE ----------------
            def se_pack(vals):
                se, nblk = _block_se(vals)
                return se, nblk

            se_frob, nblk = se_pack(block_frob)
            se_cos, _ = se_pack(block_cos)
            se_mean_offdiag_mc, _ = se_pack(block_mean_offdiag_mc)
            se_mean_offdiag_c, _ = se_pack(block_mean_offdiag_c)

            se_re_frob, nblk_re = se_pack(block_re_frob)
            se_re_cos, _ = se_pack(block_re_cos)
            se_re_mean_offdiag_mc, _ = se_pack(block_re_mean_offdiag_mc)
            se_re_mean_offdiag_c, _ = se_pack(block_re_mean_offdiag_c)
            se_re_eL2, _ = se_pack(block_re_eL2)
            se_re_eCos, _ = se_pack(block_re_eCos)
            se_re_eMax, _ = se_pack(block_re_eMax)

            se_im_frob, nblk_im = se_pack(block_im_frob)
            se_im_cos, _ = se_pack(block_im_cos)
            se_im_mean_offdiag_mc, _ = se_pack(block_im_mean_offdiag_mc)
            se_im_mean_offdiag_c, _ = se_pack(block_im_mean_offdiag_c)
            se_im_eL2, _ = se_pack(block_im_eL2)
            se_im_eCos, _ = se_pack(block_im_eCos)
            se_im_eMax, _ = se_pack(block_im_eMax)

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
                    "n_blocks_used": nblk,
                    "n_blocks_used_re": nblk_re,
                    "n_blocks_used_im": nblk_im,
                    "complex": {
                        "corr_rel_frob_se": se_frob,
                        "corr_offdiag_cosine_se": se_cos,
                        "corr_mean_abs_offdiag_mc_se": se_mean_offdiag_mc,
                        "corr_mean_abs_offdiag_c_se": se_mean_offdiag_c,
                    },
                    "complex_re": {
                        "corr_rel_frob_se": se_re_frob,
                        "corr_offdiag_cosine_se": se_re_cos,
                        "corr_mean_abs_offdiag_mc_se": se_re_mean_offdiag_mc,
                        "corr_mean_abs_offdiag_c_se": se_re_mean_offdiag_c,
                        "eig_rel_l2_se": se_re_eL2,
                        "eig_cos_se": se_re_eCos,
                        "eig_maxabs_se": se_re_eMax,
                    },
                    "complex_im": {
                        "corr_rel_frob_se": se_im_frob,
                        "corr_offdiag_cosine_se": se_im_cos,
                        "corr_mean_abs_offdiag_mc_se": se_im_mean_offdiag_mc,
                        "corr_mean_abs_offdiag_c_se": se_im_mean_offdiag_c,
                        "eig_rel_l2_se": se_im_eL2,
                        "eig_cos_se": se_im_eCos,
                        "eig_maxabs_se": se_im_eMax,
                    },
                },
            }

            # ---------------- Save ----------------
            tag = (
                f"qntk_{args.ansatz}_{args.encoder_axis}_{args.obs_kind}"
                f"_diff{args.diff_method}"
                f"_n{n_qubits}_depth{train_depth}_L{args.n_layers}"
                f"_mxw{args.max_omega}_hw{args.max_hw_for_K}_Kcap{args.max_K_cap}"
                f"_S{args.n_theta_samples}_SC{S_C}_SV{S_V}"
                f"_X{args.n_x}_B{args.batch_size}_Kb{args.k_block}_seed{args.seed}"
            )
            run_path = os.path.join(args.out_dir, f"{tag}.npz")

            payload = {
                "omega_grid": np.asarray(omega_grid, dtype=int),
                "n_qubits": int(n_qubits),
                "train_depth": int(train_depth),
                "n_layers": int(args.n_layers),
                "encoder_scale": float(args.encoder_scale),
                "encoder_axis": np.array(str(args.encoder_axis)),
                "ansatz": np.array(str(args.ansatz)),
                "obs_kind": np.array(str(args.obs_kind)),
                "diff_method": np.array(str(args.diff_method)),
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

                "K": np.asarray(K, dtype=np.int8),
                "wK": np.asarray(wK, dtype=np.float64),
                "C": np.asarray(C, dtype=np.complex128),

                "Hbar_MC": np.asarray(Hbar_MC, dtype=np.complex128),
                "CorrH_MC": np.asarray(CorrH_MC, dtype=np.complex128),
                "varH_MC": np.asarray(varH_MC, dtype=np.float64),

                "Hbar_C": np.asarray(Hbar_C, dtype=np.complex128),
                "CorrH_C": np.asarray(CorrH_C, dtype=np.complex128),
                "varH_C": np.asarray(varH_C, dtype=np.float64),

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