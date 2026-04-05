# qntk_plot.py
#
# Visualise the harmonic QNTK correlation matrices CorrH_C and CorrH_MC
# as heatmaps, for a sequence of training depths from a single compute run.
#
# Reads .npz files produced by qntk_compute.py. For each (n_qubits, n_layers) group,
# produces three figures:
#   - Re(CorrH)  : real part of the normalised QNTK
#   - Im(CorrH)  : imaginary part
#   - |CorrH|    : absolute value (magnitude)
# Each figure shows one column per training depth, with two rows:
#   row 0: C-matrix prediction (CorrH_C = C diag(||k||^2) C†, normalised)
#   row 1: Direct Monte Carlo estimate (CorrH_MC, normalised)
# A textbox below each column shows scalar diagnostics (Frobenius error,
# cosine similarity of off-diagonal entries, mean off-diagonal correlation).
#
# Usage:
#   python qntk_plot.py --indir outputs_qntk --outdir figures_qntk
#   python qntk_plot.py --indir outputs_qntk --outdir figures_qntk --mask_diag trivial
#
# Requirements: numpy, matplotlib

import os, glob, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- IO ----------
def load_npz(path):
    d = np.load(path, allow_pickle=False)
    diag = {}
    if "diag_json" in d.files:
        try:
            diag = json.loads(str(d["diag_json"].item()))
        except Exception:
            diag = {}
    meta = {}
    for k in ["n_qubits","train_depth","n_layers","n_theta_samples","S_C","S_V","seed"]:
        if k in d.files:
            try:
                meta[k] = int(d[k])
            except Exception:
                try:
                    meta[k] = float(d[k])
                except Exception:
                    meta[k] = str(d[k])
    return d, meta, diag

# ---------- diag_json helpers ----------
def _dig(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or (k not in cur):
            return default
        cur = cur[k]
    return cur

def _get_se(diag, group, key, default=np.nan):
    v = _dig(diag, "block_se", group, key, default=default)
    try:
        return float(v)
    except Exception:
        return default

def fmt_pm(val, err, fmt_val="{:.3e}", fmt_err="{:.1e}"):
    if not np.isfinite(val):
        return "?"
    if not np.isfinite(err):
        return fmt_val.format(val)
    return fmt_val.format(val) + " ±" + fmt_err.format(err)

# ---------- math ----------
def corr_from_psd(H, eps=1e-14):
    H = np.asarray(H, dtype=np.complex128)
    var = np.real(np.diag(H)).astype(np.float64)
    sigma = np.sqrt(np.maximum(var, 0.0))
    sigma[sigma < eps] = np.nan
    Corr = H / np.outer(sigma, sigma)

    good = np.isfinite(var) & (var > eps)
    if np.any(good):
        Corr = np.array(Corr, copy=True)
        for i in range(len(var)):
            if good[i]:
                Corr[i, i] = 1.0
    return Corr, var, good

def mask_diagonal_for_display(M, mode="zero"):
    M = np.array(M, copy=True)
    if mode == "zero":
        np.fill_diagonal(M, 0.0)
    elif mode == "nan":
        np.fill_diagonal(M, np.nan)
    else:
        raise ValueError("mode must be 'zero' or 'nan'")
    return M

def mask_trivial_diagonal_for_display(M, target=1.0, tol=1e-12, masked_value=np.nan):
    """
    Mask only diagonal entries close to `target`.
    Leaves all other diagonal entries untouched.
    """
    M = np.array(M, copy=True)
    d = np.diag(M).copy()
    idx = np.arange(M.shape[0])
    hit = np.isfinite(d) & (np.abs(d - target) <= tol)
    M[idx[hit], idx[hit]] = masked_value
    return M

def offdiag_vector(M):
    n = M.shape[0]
    iu = np.triu_indices(n, k=1)
    return M[iu]

def rel_frob(A, B, eps=1e-14):
    return float(np.linalg.norm(A - B, "fro") / (np.linalg.norm(B, "fro") + eps))

def cosine_similarity(u, v, eps=1e-14):
    num = np.vdot(u, v)
    den = np.linalg.norm(u) * np.linalg.norm(v) + eps
    return float(np.real(num / den))

def mean_abs_offdiag(A):
    v = offdiag_vector(A)
    v = v[np.isfinite(v)]
    return 0.0 if v.size == 0 else float(np.mean(np.abs(v)))

# ---------- omega restriction ----------
def omega_side_mask(omega_grid, omega_side):
    og = np.asarray(omega_grid, dtype=int)
    if omega_side == "all":
        return np.ones_like(og, bool)
    if omega_side == "nonneg":
        return og >= 0
    if omega_side == "nonpos":
        return og <= 0
    raise ValueError

def omega_phys_mask(omega_grid, omega_phys):
    if omega_phys is None:
        return np.ones_like(np.asarray(omega_grid, int), bool)
    og = np.asarray(omega_grid, dtype=int)
    return np.abs(og) <= int(omega_phys)

def restrict_by_mask(A_list, omega_grid, keep_mask):
    keep = np.asarray(keep_mask, bool)
    A_list_r = [A[np.ix_(keep, keep)] for A in A_list]
    return A_list_r, np.asarray(omega_grid)[keep]

# ---------- color scaling ----------
def offdiag_vmax_diverging(mats, clip_q=0.995, eps=1e-12):
    vals = []
    for A in mats:
        n = A.shape[0]
        iu = np.triu_indices(n, 1)
        v = np.abs(A[iu])
        v = v[np.isfinite(v)]
        if v.size:
            vals.append(v)
    if not vals:
        return 1.0
    vmax = float(np.quantile(np.concatenate(vals), clip_q))
    return max(vmax, eps)

def offdiag_vmax_abs(mats, clip_q=0.995, eps=1e-12):
    vals = []
    for A in mats:
        n = A.shape[0]
        iu = np.triu_indices(n, 1)
        v = A[iu]
        v = v[np.isfinite(v)]
        if v.size:
            vals.append(v)
    if not vals:
        return 1.0
    vmax = float(np.quantile(np.concatenate(vals), clip_q))
    return min(max(vmax, eps), 1.0)

def set_omega_ticks(ax, omega_grid):
    og = np.asarray(omega_grid)
    n = len(og)
    step = max(1, n // 8)
    ticks = np.arange(0, n, step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(int(og[i])) for i in ticks], rotation=45, ha="right")
    ax.set_yticklabels([str(int(og[i])) for i in ticks])
    ax.set_xlabel(r"$\omega'$")
    ax.set_ylabel(r"$\omega$")

# ---------- plot core ----------
def plot_grid(
    rows_by_depth,
    omega_grid,
    title_prefix,
    out_path,
    part="real",
    clip_q=0.995,
    omega_phys=None,
    omega_side="all",
    mask_diag="none",
    trivial_diag_tol=1e-12,
):
    assert part in ("real", "imag", "abs")
    assert omega_side in ("all", "nonneg", "nonpos")
    assert mask_diag in ("none", "zero", "nan", "trivial")

    depths = [r["depth"] for r in rows_by_depth]
    ncols = len(depths)

    C0 = [r["CorrH_C"] for r in rows_by_depth]
    M0 = [r["CorrH_MC"] for r in rows_by_depth]

    if part == "real":
        A_C_raw = [np.real(A) for A in C0]
        A_M_raw = [np.real(A) for A in M0]
        cmap_name = "coolwarm"
        diverging = True
        label = "Re"
        group = "complex_re"
        tag = "re"
    elif part == "imag":
        A_C_raw = [np.imag(A) for A in C0]
        A_M_raw = [np.imag(A) for A in M0]
        cmap_name = "coolwarm"
        diverging = True
        label = "Im"
        group = "complex_im"
        tag = "im"
    else:
        A_C_raw = [np.abs(A) for A in C0]
        A_M_raw = [np.abs(A) for A in M0]
        cmap_name = "viridis"
        diverging = False
        label = "|.|"
        group = "complex"
        tag = None

    cmap = plt.get_cmap(cmap_name).copy()
    if mask_diag in ("nan", "trivial"):
        cmap.set_bad(color="lightgray")

    keep = omega_side_mask(omega_grid, omega_side) & omega_phys_mask(omega_grid, omega_phys)
    A_C_raw, og_r = restrict_by_mask(A_C_raw, omega_grid, keep)
    A_M_raw, _    = restrict_by_mask(A_M_raw, omega_grid, keep)

    # Display arrays (display-only masking)
    if mask_diag == "none":
        A_C_disp = A_C_raw
        A_M_disp = A_M_raw

    elif mask_diag in ("zero", "nan"):
        A_C_disp = [mask_diagonal_for_display(A, mask_diag) for A in A_C_raw]
        A_M_disp = [mask_diagonal_for_display(A, mask_diag) for A in A_M_raw]

    elif mask_diag == "trivial":
        if part in ("real", "abs"):
            A_C_disp = [
                mask_trivial_diagonal_for_display(A, target=1.0, tol=trivial_diag_tol, masked_value=np.nan)
                for A in A_C_raw
            ]
            A_M_disp = [
                mask_trivial_diagonal_for_display(A, target=1.0, tol=trivial_diag_tol, masked_value=np.nan)
                for A in A_M_raw
            ]
        else:
            # match the behaviour of the correlation plotter: do not mask Im diagonal by default
            A_C_disp = A_C_raw
            A_M_disp = A_M_raw

    # scaling
    if diverging:
        vmax = offdiag_vmax_diverging(A_C_disp + A_M_disp, clip_q=clip_q)
        vmin = -vmax
    else:
        vmax = offdiag_vmax_abs(A_C_disp + A_M_disp, clip_q=clip_q)
        vmin = 0.0

    fig_h = 7.2
    fig_w = max(8.0, 3.0 * ncols)
    fig, axes = plt.subplots(
        3, ncols, figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": [1.0, 1.0, 0.42]},
        constrained_layout=True
    )
    if ncols == 1:
        axes = np.array(axes).reshape(3, 1)

    side_note = {"all": "all ω", "nonneg": "ω≥0", "nonpos": "ω≤0"}[omega_side]
    band_note = f", |ω|≤{int(omega_phys)}" if omega_phys is not None else ""
    fig.suptitle(f"Hermitian CorrH | {title_prefix} | {label} | {side_note}{band_note}", fontsize=14)

    ims = []
    for j, depth in enumerate(depths):
        ax0 = axes[0, j]
        im0 = ax0.imshow(A_C_disp[j], origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
        ax0.set_title(f"depth={depth} | {label} | C")
        set_omega_ticks(ax0, og_r)
        ims.append(im0)

        ax1 = axes[1, j]
        im1 = ax1.imshow(A_M_disp[j], origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
        ax1.set_title(f"depth={depth} | {label} | MC")
        set_omega_ticks(ax1, og_r)
        ims.append(im1)

        # fallback metrics computed from matrices
        err_fb = rel_frob(A_M_raw[j], A_C_raw[j])
        cos_fb = cosine_similarity(offdiag_vector(A_M_raw[j]), offdiag_vector(A_C_raw[j]))
        mean_offdiag_corrC_fb = mean_abs_offdiag(A_C_raw[j])
        mean_offdiag_corrM_fb = mean_abs_offdiag(A_M_raw[j])

        diagj = rows_by_depth[j].get("diag", {}) or {}

        err_mean, err_se = err_fb, np.nan
        cos_mean, cos_se = cos_fb, np.nan
        mean_offdiag_corrC, mean_offdiag_corrC_se = mean_offdiag_corrC_fb, np.nan
        mean_offdiag_corrM, mean_offdiag_corrM_se = mean_offdiag_corrM_fb, np.nan

        if isinstance(diagj, dict) and diagj:
            if part in ("real", "imag"):
                parts = _dig(diagj, "complex", "parts", default=None)
                if isinstance(parts, dict):
                    v = parts.get(f"corr_rel_frob_{tag}", None)
                    if v is not None:
                        err_mean = float(v)
                    v = parts.get(f"corr_offdiag_cosine_{tag}", None)
                    if v is not None:
                        cos_mean = float(v)
                    v = parts.get(f"corrC_mean_abs_offdiag_{tag}", None)
                    if v is not None:
                        mean_offdiag_corrC = float(v)
                    v = parts.get(f"corr_mean_abs_offdiag_{tag}", None)
                    if v is not None:
                        mean_offdiag_corrM = float(v)

                err_se  = _get_se(diagj, group, "corr_rel_frob_se", default=np.nan)
                cos_se  = _get_se(diagj, group, "corr_offdiag_cosine_se", default=np.nan)
                mean_offdiag_corrC_se = _get_se(diagj, group, "corr_mean_abs_offdiag_c_se", default=np.nan)
                mean_offdiag_corrM_se = _get_se(diagj, group, "corr_mean_abs_offdiag_mc_se", default=np.nan)

            elif part == "abs":
                comp = _dig(diagj, "complex", default=None)
                if isinstance(comp, dict):
                    v = comp.get("corr_rel_frob", None)
                    if v is not None:
                        err_mean = float(v)
                    v = comp.get("corr_offdiag_cosine", None)
                    if v is not None:
                        cos_mean = float(v)
                    v = comp.get("corrC_mean_abs_offdiag", None)
                    if v is not None:
                        mean_offdiag_corrC = float(v)
                    v = comp.get("corr_mean_abs_offdiag", None)
                    if v is not None:
                        mean_offdiag_corrM = float(v)

                err_se  = _get_se(diagj, "complex", "corr_rel_frob_se", default=np.nan)
                cos_se  = _get_se(diagj, "complex", "corr_offdiag_cosine_se", default=np.nan)
                mean_offdiag_corrC_se = _get_se(diagj, "complex", "corr_mean_abs_offdiag_c_se", default=np.nan)
                mean_offdiag_corrM_se = _get_se(diagj, "complex", "corr_mean_abs_offdiag_mc_se", default=np.nan)

        ax2 = axes[2, j]
        ax2.axis("off")
        ax2.text(
            0.5, 0.5,
            "Frobenius err (MC vs C):\n"
            f"{fmt_pm(err_mean, err_se, fmt_val='{:.3e}', fmt_err='{:.1e}')}\n"
            "Cos(offdiag) (MC vs C):\n"
            f"{fmt_pm(cos_mean, cos_se, fmt_val='{:.3f}', fmt_err='{:.3f}')}\n"
            "mean|offdiag|:\n"
            f"C:  {fmt_pm(mean_offdiag_corrC, mean_offdiag_corrC_se, fmt_val='{:.3e}', fmt_err='{:.1e}')}\n"
            f"MC: {fmt_pm(mean_offdiag_corrM, mean_offdiag_corrM_se, fmt_val='{:.3e}', fmt_err='{:.1e}')}\n",
            ha="center", va="center", fontsize=10.2
        )

    cbar = fig.colorbar(ims[-1], ax=axes[0:2, :], fraction=0.025, pad=0.02)
    if part == "abs":
        cbar_label = r"$|{\rm CorrH}(\omega,\omega')|$"
    else:
        cbar_label = f"{label} value"

    if mask_diag == "nan":
        cbar_label += " (diag masked)"
    elif mask_diag == "zero":
        cbar_label += " (diag set to 0 for display)"
    elif mask_diag == "trivial":
        if part in ("real", "abs"):
            cbar_label += " (trivial diag=1 masked)"
        else:
            cbar_label += " (no diag masking for Im)"

    cbar.set_label(cbar_label)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# ---------- main ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--pattern", default="*.npz")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--clip_q", type=float, default=0.995)
    ap.add_argument("--omega_phys", type=int, default=None)
    ap.add_argument("--omega_side", choices=["all","nonneg","nonpos"], default="all")
    ap.add_argument("--mask_diag", choices=["none","zero","nan","trivial"], default="none")
    ap.add_argument(
        "--trivial_diag_tol",
        type=float,
        default=1e-12,
        help="Absolute tolerance for deciding whether a diagonal entry is the trivial unit value when --mask_diag trivial.",
    )
    ap.add_argument("--force_corr_from_hbar", action="store_true",
                    help="Ignore CorrH_* in file; recompute CorrH from Hbar_*.")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = args.outdir or os.path.join(args.indir, "plots_qntkH_grid")
    os.makedirs(outdir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.indir, args.pattern)))
    if not paths:
        raise FileNotFoundError("No files matched.")

    groups = {}
    for p in paths:
        d, meta, diag = load_npz(p)
        key = (int(d["n_qubits"]), int(d["n_layers"]))
        groups.setdefault(key, []).append((p, d, diag))

    for (nq, L), items in sorted(groups.items()):
        rows = []
        og = None
        for p, d, diag in items:
            if og is None:
                og = np.asarray(d["omega_grid"], dtype=int)

            if (not args.force_corr_from_hbar) and ("CorrH_C" in d.files) and ("CorrH_MC" in d.files):
                CorrC = np.asarray(d["CorrH_C"])
                CorrM = np.asarray(d["CorrH_MC"])
            else:
                if ("Hbar_C" not in d.files) or ("Hbar_MC" not in d.files):
                    raise KeyError(f"{os.path.basename(p)} missing CorrH_* and missing Hbar_*; cannot plot robustly.")
                CorrC, varC, goodC = corr_from_psd(d["Hbar_C"])
                CorrM, varM, goodM = corr_from_psd(d["Hbar_MC"])
                frac_good = float(np.mean(goodC & goodM))
                if frac_good < 0.9:
                    print(f"[WARN] {os.path.basename(p)}: only {frac_good*100:.1f}% frequencies have well-defined CorrH (diag variance > eps).")

            diagC = np.real(np.diag(CorrC))
            if np.nanmax(np.abs(diagC - 1.0)) > 1e-3:
                print(f"[WARN] {os.path.basename(p)}: CorrH_C diagonal not ~1 (max|diag-1|={np.nanmax(np.abs(diagC-1)):.2e}).")

            rows.append({
                "depth": int(d["train_depth"]),
                "CorrH_C": CorrC,
                "CorrH_MC": CorrM,
                "diag": diag,
                "path": p,
            })

        rows.sort(key=lambda r: r["depth"])
        depths_str = "_".join(str(r["depth"]) for r in rows)
        base = f"grid_n{nq}_L{L}_depths_{depths_str}_side{args.omega_side}_diag{args.mask_diag}"

        plot_grid(
            rows, og, f"n={nq}, L={L}", os.path.join(outdir, base + "_COMPLEX_REAL.png"),
            part="real", clip_q=args.clip_q, omega_phys=args.omega_phys,
            omega_side=args.omega_side, mask_diag=args.mask_diag,
            trivial_diag_tol=args.trivial_diag_tol
        )
        plot_grid(
            rows, og, f"n={nq}, L={L}", os.path.join(outdir, base + "_COMPLEX_IMAG.png"),
            part="imag", clip_q=args.clip_q, omega_phys=args.omega_phys,
            omega_side=args.omega_side, mask_diag=args.mask_diag,
            trivial_diag_tol=args.trivial_diag_tol
        )
        plot_grid(
            rows, og, f"n={nq}, L={L}", os.path.join(outdir, base + "_COMPLEX_ABS.png"),
            part="abs", clip_q=args.clip_q, omega_phys=args.omega_phys,
            omega_side=args.omega_side, mask_diag=args.mask_diag,
            trivial_diag_tol=args.trivial_diag_tol
        )

        print(f"[OK] wrote plots for n={nq}, L={L} -> {outdir}")

if __name__ == "__main__":
    main()