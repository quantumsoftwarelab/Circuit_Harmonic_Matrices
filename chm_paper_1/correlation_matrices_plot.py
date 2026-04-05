# correlation_matrices_plot.py
#
# Visualise the Fourier-coefficient correlation matrices Corr_C and Corr_MC
# as heatmaps, for a sequence of training depths from a single compute run.
#
# For each .npz file produced by correlation_matrices_compute.py, this script
# plots three heatmap grids:
#   - Re(Corr)  : real part of the correlation matrix
#   - Im(Corr)  : imaginary part
#   - |Corr|    : absolute value (magnitude)
# Each grid shows one column per training depth, with two rows:
#   row 0: C-matrix prediction (Corr_C)
#   row 1: Direct Monte Carlo estimate (Corr_MC)
# A textbox below each column shows scalar diagnostics (Frobenius error,
# cosine similarity of off-diagonal entries, and mean off-diagonal correlation).
#
# Optional variance-support masking:
#   --mask_support: zero out entries (ω,ω') where Var[a_ω] or Var[a_ω'] falls
#   below a threshold fraction of the maximum variance. This focuses the plot
#   on frequency pairs that actually carry spectral weight.
#
# Usage:
#   python correlation_matrices_plot.py --indir outputs_correlation_matrices --outdir figures_correlation_matrices
#   python correlation_matrices_plot.py --indir outputs_correlation_matrices --outdir figures_correlation_matrices --mask_support --support_threshold 0.01
#
# Requirements: numpy, matplotlib

import os
import glob
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------
# IO helpers
# -------------------------

REQUIRED_KEYS = [
    "Corr_C", "Corr_MC", "omega_grid", "var_C", "var_MC",
]

def load_npz_strict(path):
    d = np.load(path, allow_pickle=False)

    missing = [k for k in REQUIRED_KEYS if k not in d.files]
    if missing:
        raise KeyError(
            f"[ERROR] {os.path.basename(path)} is missing required keys: {missing}\n"
            "This script assumes hermitian-only files produced by correlation_matrices_compute.py."
        )

    meta = {}
    for k in ["n_qubits", "train_depth", "n_layers", "n_theta_samples", "S_C", "S_V", "seed"]:
        if k in d.files:
            try:
                meta[k] = int(d[k])
            except Exception:
                try:
                    meta[k] = float(d[k])
                except Exception:
                    meta[k] = str(d[k])

    diag = {}
    if "diag_json" in d.files:
        try:
            diag = json.loads(str(d["diag_json"].item()))
        except Exception:
            diag = {}

    return d, meta, diag


# -------------------------
# Math helpers (metrics)
# -------------------------

def mask_diagonal_for_display(M, mode="zero"):
    M = np.array(M, copy=True)
    if mode == "zero":
        np.fill_diagonal(M, 0.0)
    elif mode == "nan":
        np.fill_diagonal(M, np.nan)
    else:
        raise ValueError("mode must be 'zero' or 'nan'")
    return M

def offdiag_vector(M):
    M = np.asarray(M)
    n = M.shape[0]
    iu = np.triu_indices(n, k=1)
    return M[iu]

def mean_abs_offdiag(M, use_abs=True):
    v = offdiag_vector(M)
    if use_abs:
        v = np.abs(v)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    return float(np.mean(v))

def rel_frob(A, B, eps=1e-14):
    A = np.asarray(A)
    B = np.asarray(B)
    mask = np.isfinite(A) & np.isfinite(B)
    if not np.any(mask):
        return np.nan
    Avec = A[mask]
    Bvec = B[mask]
    num = np.linalg.norm(Avec - Bvec)
    den = np.linalg.norm(Bvec) + eps
    return float(num / den)

def cosine_similarity(u, v, eps=1e-14):
    u = np.asarray(u)
    v = np.asarray(v)
    mask = np.isfinite(u) & np.isfinite(v)
    if not np.any(mask):
        return np.nan
    u = u[mask]
    v = v[mask]
    num = np.vdot(u, v)
    den = (np.linalg.norm(u) * np.linalg.norm(v) + eps)
    return float(np.real(num / den))


# -------------------------
# Omega restriction helpers
# -------------------------

def omega_side_mask(omega_grid, omega_side):
    og = np.asarray(omega_grid, dtype=int)
    if omega_side == "all":
        return np.ones_like(og, dtype=bool)
    if omega_side == "nonneg":
        return og >= 0
    if omega_side == "nonpos":
        return og <= 0
    raise ValueError("omega_side must be one of {all, nonneg, nonpos}")

def omega_phys_mask(omega_grid, omega_phys):
    if omega_phys is None:
        return np.ones_like(np.asarray(omega_grid, dtype=int), dtype=bool)
    og = np.asarray(omega_grid, dtype=int)
    return np.abs(og) <= int(omega_phys)

def restrict_by_mask(A_list, omega_grid, keep_mask):
    keep = np.asarray(keep_mask, dtype=bool)
    if np.count_nonzero(keep) < 2:
        raise ValueError("Restriction keeps <2 frequencies; cannot plot.")
    A_list_r = [A[np.ix_(keep, keep)] for A in A_list]
    return A_list_r, np.asarray(omega_grid)[keep]

def restrict_vector_list(v_list, omega_grid, keep_mask):
    keep = np.asarray(keep_mask, dtype=bool)
    if np.count_nonzero(keep) < 2:
        raise ValueError("Restriction keeps <2 frequencies; cannot plot.")
    v_list_r = [None if v is None else np.asarray(v)[keep] for v in v_list]
    return v_list_r, np.asarray(omega_grid)[keep]


# -------------------------
# Variance-support masking
# -------------------------

def relative_support_mask(var_vec, rel_thresh):
    v = np.asarray(var_vec, dtype=np.float64)
    vmax = float(np.max(v)) if v.size else 0.0
    if vmax <= 0.0:
        return np.zeros_like(v, dtype=bool)
    return (v / vmax) > float(rel_thresh)

def combine_support_masks(mask_c, mask_mc, mode):
    if mode == "intersection":
        return mask_c & mask_mc
    if mode == "union":
        return mask_c | mask_mc
    if mode == "c":
        return mask_c
    if mode == "mc":
        return mask_mc
    raise ValueError("support_mode must be one of {'intersection','union','c','mc'}")

def apply_support_mask_matrix(A, support_mask):
    """
    Mask rows/cols outside support with NaN for display/metrics.
    """
    A = np.array(A, copy=True)
    bad = ~np.asarray(support_mask, dtype=bool)
    if np.any(bad):
        A[bad, :] = np.nan
        A[:, bad] = np.nan
    return A


# -------------------------
# Color scaling
# -------------------------

def offdiag_vmax_diverging(mats, clip_q=0.995, eps=1e-12):
    vals = []
    for A in mats:
        A = np.asarray(A)
        n = A.shape[0]
        iu = np.triu_indices(n, k=1)
        v = np.abs(A[iu])
        v = v[np.isfinite(v)]
        if v.size:
            vals.append(v)
    if not vals:
        return 1.0
    vals = np.concatenate(vals)
    vmax = float(np.quantile(vals, clip_q))
    return max(vmax, eps)

def offdiag_vmax_abs(mats, clip_q=0.995, eps=1e-12):
    vals = []
    for A in mats:
        A = np.asarray(A)
        n = A.shape[0]
        iu = np.triu_indices(n, k=1)
        v = A[iu]
        v = v[np.isfinite(v)]
        if v.size:
            vals.append(v)
    if not vals:
        return 1.0
    vals = np.concatenate(vals)
    vmax = float(np.quantile(vals, clip_q))
    vmax = min(max(vmax, eps), 1.0)
    return vmax


# -------------------------
# diag_json helpers
# -------------------------

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


# -------------------------
# Tick formatting
# -------------------------

def set_omega_ticks(ax, omega_grid):
    if omega_grid is None:
        return
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


# -------------------------
# Metrics fallback
# -------------------------

def _fallback_metrics_complex_parts(A_C_part, A_MC_part):
    ncols = len(A_C_part)
    errs = [rel_frob(A_MC_part[j], A_C_part[j]) for j in range(ncols)]
    cos_off = []
    mean_offdiag_C = []
    mean_offdiag_MC = []
    for j in range(ncols):
        vC = offdiag_vector(A_C_part[j])
        vM = offdiag_vector(A_MC_part[j])
        cos_off.append(cosine_similarity(vM, vC))
        mean_offdiag_C.append(mean_abs_offdiag(A_C_part[j], use_abs=True))
        mean_offdiag_MC.append(mean_abs_offdiag(A_MC_part[j], use_abs=True))
    return errs, cos_off, mean_offdiag_C, mean_offdiag_MC


# -------------------------
# Plotting core
# -------------------------

def plot_grid_figure_hermitian(
    rows_by_depth,
    omega_grid,
    title_prefix,
    out_path,
    part="real",             # "real" | "imag" | "abs"
    clip_q=0.995,
    omega_phys=None,
    omega_side="all",
    mask_diag="none",
    support_rel_var=None,
    support_mode="intersection",
):
    assert part in ("real", "imag", "abs")
    assert omega_side in ("all", "nonneg", "nonpos")
    assert mask_diag in ("none", "zero", "nan")

    depths = [r["depth"] for r in rows_by_depth]
    ncols = len(depths)

    Corr_C0  = [r["Corr_C"]  for r in rows_by_depth]
    Corr_MC0 = [r["Corr_MC"] for r in rows_by_depth]
    var_C0   = [r["var_C"]   for r in rows_by_depth]
    var_MC0  = [r["var_MC"]  for r in rows_by_depth]

    if part == "real":
        A_C_raw0  = [np.real(A) for A in Corr_C0]
        A_MC_raw0 = [np.real(A) for A in Corr_MC0]
        part_label = "Re"
        cmap = "coolwarm"
        diverging = True
    elif part == "imag":
        A_C_raw0  = [np.imag(A) for A in Corr_C0]
        A_MC_raw0 = [np.imag(A) for A in Corr_MC0]
        part_label = "Im"
        cmap = "coolwarm"
        diverging = True
    else:
        A_C_raw0  = [np.abs(A) for A in Corr_C0]
        A_MC_raw0 = [np.abs(A) for A in Corr_MC0]
        part_label = "|.|"
        cmap = "viridis"
        diverging = False

    # Restrict omega consistently
    keep = omega_side_mask(omega_grid, omega_side) & omega_phys_mask(omega_grid, omega_phys)

    A_C_raw, omega_grid_r  = restrict_by_mask(A_C_raw0,  omega_grid, keep)
    A_MC_raw, _            = restrict_by_mask(A_MC_raw0, omega_grid, keep)
    var_list_r, _          = restrict_vector_list([*var_C0, *var_MC0], omega_grid, keep)
    var_C_r = var_list_r[:len(var_C0)]
    var_MC_r = var_list_r[len(var_C0):]

    # Apply support mask based on relative variances
    support_masks = []
    if support_rel_var is not None:
        A_C_masked = []
        A_MC_masked = []
        for j in range(ncols):
            mC = relative_support_mask(var_C_r[j], support_rel_var)
            mM = relative_support_mask(var_MC_r[j], support_rel_var)
            ms = combine_support_masks(mC, mM, support_mode)
            support_masks.append(ms)
            A_C_masked.append(apply_support_mask_matrix(A_C_raw[j], ms))
            A_MC_masked.append(apply_support_mask_matrix(A_MC_raw[j], ms))
        A_C_raw = A_C_masked
        A_MC_raw = A_MC_masked
    else:
        support_masks = [None] * ncols

    # Display mask (display-only)
    if mask_diag == "none":
        A_C_disp  = [np.array(A, copy=True) for A in A_C_raw]
        A_MC_disp = [np.array(A, copy=True) for A in A_MC_raw]
    else:
        A_C_disp  = [mask_diagonal_for_display(A, mode=mask_diag) for A in A_C_raw]
        A_MC_disp = [mask_diagonal_for_display(A, mode=mask_diag) for A in A_MC_raw]

    # Color scaling
    if diverging:
        vmax = offdiag_vmax_diverging(A_C_disp + A_MC_disp, clip_q=clip_q)
        vmin = -vmax
    else:
        vmax = offdiag_vmax_abs(A_C_disp + A_MC_disp, clip_q=clip_q)
        vmin = 0.0

    # Layout
    fig_h = 7.2
    fig_w = max(8.0, 3.0 * ncols)
    fig, axes = plt.subplots(
        3, ncols, figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": [1.0, 1.0, 0.48]},
        constrained_layout=True
    )
    if ncols == 1:
        axes = np.array(axes).reshape(3, 1)

    side_note = {"all": "all ω", "nonneg": "ω≥0", "nonpos": "ω≤0"}[omega_side]
    band_note = f", |ω|≤{int(omega_phys)}" if omega_phys is not None else ""
    supp_note = ""
    if support_rel_var is not None:
        supp_note = f", support>{support_rel_var:.0e} ({support_mode})"

    fig.suptitle(
        f"Hermitian Corr(a) | {title_prefix} | {part_label} | {side_note}{band_note}{supp_note}",
        fontsize=14
    )

    ims = []

    # Metrics fallback computed from masked meaningful matrices
    fb_errs, fb_cos, fb_mean_offdiag_C, fb_mean_offdiag_M = _fallback_metrics_complex_parts(A_C_raw, A_MC_raw)

    for j, depth in enumerate(depths):
        ax0 = axes[0, j]
        im0 = ax0.imshow(
            np.ma.masked_invalid(A_C_disp[j]),
            origin="lower", aspect="auto",
            vmin=vmin, vmax=vmax, cmap=cmap
        )
        ax0.set_title(f"depth={depth} | {part_label} | C")
        set_omega_ticks(ax0, omega_grid_r)
        ims.append(im0)

        ax1 = axes[1, j]
        im1 = ax1.imshow(
            np.ma.masked_invalid(A_MC_disp[j]),
            origin="lower", aspect="auto",
            vmin=vmin, vmax=vmax, cmap=cmap
        )
        ax1.set_title(f"depth={depth} | {part_label} | MC")
        set_omega_ticks(ax1, omega_grid_r)
        ims.append(im1)

        diag = rows_by_depth[j].get("diag", {}) or {}

        # defaults from fallback on masked matrices
        err_mean = fb_errs[j]; err_se = np.nan
        cos_mean = fb_cos[j];  cos_se = np.nan
        mean_offdiag_C_mean = fb_mean_offdiag_C[j]; mean_offdiag_C_se = np.nan
        mean_offdiag_M_mean = fb_mean_offdiag_M[j]; mean_offdiag_M_se = np.nan

        # Prefer diag_json only when no support masking is applied.
        # Once support masking is applied, the fallback metrics are the relevant ones.
        if (support_rel_var is None) and isinstance(diag, dict) and diag:
            if part in ("real", "imag"):
                parts = _dig(diag, "complex", "parts", default=None)
                group = "complex_re" if part == "real" else "complex_im"
                tag = "re" if part == "real" else "im"

                if isinstance(parts, dict):
                    v = parts.get(f"corr_rel_frob_{tag}", None)
                    if v is not None: err_mean = float(v)
                    v = parts.get(f"corr_offdiag_cosine_{tag}", None)
                    if v is not None: cos_mean = float(v)
                    v = parts.get(f"corrC_mean_abs_offdiag_{tag}", None)
                    if v is not None: mean_offdiag_C_mean = float(v)
                    v = parts.get(f"corr_mean_abs_offdiag_{tag}", None)
                    if v is not None: mean_offdiag_M_mean = float(v)

                err_se             = _get_se(diag, group, "corr_rel_frob_se", default=np.nan)
                cos_se             = _get_se(diag, group, "corr_offdiag_cosine_se", default=np.nan)
                mean_offdiag_C_se  = _get_se(diag, group, "corr_mean_abs_offdiag_c_se", default=np.nan)
                mean_offdiag_M_se  = _get_se(diag, group, "corr_mean_abs_offdiag_mc_se", default=np.nan)

            elif part == "abs":
                comp = _dig(diag, "complex", default=None)
                if isinstance(comp, dict):
                    v = comp.get("corr_rel_frob", None)
                    if v is not None: err_mean = float(v)
                    v = comp.get("corr_offdiag_cosine", None)
                    if v is not None: cos_mean = float(v)
                    v = comp.get("corrC_mean_abs_offdiag", None)
                    if v is not None: mean_offdiag_C_mean = float(v)
                    v = comp.get("corr_mean_abs_offdiag", None)
                    if v is not None: mean_offdiag_M_mean = float(v)

                err_se             = _get_se(diag, "complex", "corr_rel_frob_se", default=np.nan)
                cos_se             = _get_se(diag, "complex", "corr_offdiag_cosine_se", default=np.nan)
                mean_offdiag_C_se  = _get_se(diag, "complex", "corr_mean_abs_offdiag_c_se", default=np.nan)
                mean_offdiag_M_se  = _get_se(diag, "complex", "corr_mean_abs_offdiag_mc_se", default=np.nan)

        ax2 = axes[2, j]
        ax2.axis("off")

        support_txt = ""
        if support_masks[j] is not None:
            nsupp = int(np.count_nonzero(support_masks[j]))
            support_txt = f"support size: {nsupp}/{len(support_masks[j])}\n"

        ax2.text(
            0.5, 0.5,
            support_txt +
            "Frobenius err (MC vs C):\n"
            f"{fmt_pm(err_mean, err_se, fmt_val='{:.3e}', fmt_err='{:.1e}')}\n"
            "Cos(offdiag) (MC vs C):\n"
            f"{fmt_pm(cos_mean, cos_se, fmt_val='{:.3f}', fmt_err='{:.3f}')}\n"
            "mean|offdiag(Corr)|:\n"
            f"C:  {fmt_pm(mean_offdiag_C_mean, mean_offdiag_C_se, fmt_val='{:.3e}', fmt_err='{:.1e}')}\n"
            f"MC: {fmt_pm(mean_offdiag_M_mean, mean_offdiag_M_se, fmt_val='{:.3e}', fmt_err='{:.1e}')}\n",
            ha="center", va="center", fontsize=10.0
        )

    cbar = fig.colorbar(ims[-1], ax=axes[0:2, :], fraction=0.025, pad=0.02)
    if part == "abs":
        cbar.set_label(r"$|{\rm Corr}(\omega,\omega')|$")
    else:
        cbar.set_label(f"{part_label} value" + ("" if mask_diag == "none" else " (diag masked)"))

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -------------------------
# Main
# -------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, required=True, help="Directory containing .npz files from correlation_matrices_compute.py.")
    ap.add_argument("--pattern", type=str, default="*.npz", help="Glob pattern inside indir.")
    ap.add_argument("--outdir", type=str, default=None, help="Output directory (default: indir/plots_covcorrH_grid).")
    ap.add_argument("--clip_q", type=float, default=0.995, help="Quantile for off-diagonal color scaling.")
    ap.add_argument("--omega_phys", type=int, default=None, help="Restrict to |omega|<=omega_phys.")
    ap.add_argument(
        "--omega_side",
        type=str,
        default="all",
        choices=["all", "nonneg", "nonpos"],
        help="Keep only one side of spectrum (e.g. nonneg).",
    )
    ap.add_argument(
        "--mask_diag",
        type=str,
        default="none",
        choices=["none", "zero", "nan"],
        help="Mask diagonal for display only.",
    )
    ap.add_argument(
        "--support_rel_var",
        type=float,
        default=None,
        help="Relative variance threshold for defining omega-support. "
             "Modes with var/max(var) <= threshold are masked out. Example: 1e-3",
    )
    ap.add_argument(
        "--support_mode",
        type=str,
        default="intersection",
        choices=["intersection", "union", "c", "mc"],
        help="How to combine C-side and MC-side support masks.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    indir = args.indir
    outdir = args.outdir or os.path.join(indir, "plots_covcorrH_grid")
    os.makedirs(outdir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(indir, args.pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matched: {os.path.join(indir, args.pattern)}")

    # Group by (n_qubits, n_layers)
    groups = {}
    for p in paths:
        d, meta, diag = load_npz_strict(p)
        key = (int(d["n_qubits"]), int(d["n_layers"]))
        groups.setdefault(key, []).append((p, d, meta, diag))

    summary = {"indir": indir, "pattern": args.pattern, "groups": []}

    for (n_qubits, n_layers), items in sorted(groups.items()):
        rows = []
        omega_grid_complex = None

        for (p, d, meta, diag) in items:
            depth = int(d["train_depth"])
            if omega_grid_complex is None:
                omega_grid_complex = d["omega_grid"].astype(int)

            rows.append({
                "path": p,
                "depth": depth,
                "Corr_C": d["Corr_C"],
                "Corr_MC": d["Corr_MC"],
                "var_C": d["var_C"],
                "var_MC": d["var_MC"],
                "diag": diag,
            })

        rows.sort(key=lambda r: r["depth"])
        title_prefix = f"n={n_qubits}, L={n_layers}"
        depths_str = "_".join(str(r["depth"]) for r in rows)

        base = f"grid_n{n_qubits}_L{n_layers}_depths_{depths_str}_side{args.omega_side}_diag{args.mask_diag}"
        if args.support_rel_var is not None:
            base += f"_supp{args.support_rel_var:.0e}_{args.support_mode}"

        out_re  = os.path.join(outdir, f"{base}_COMPLEX_REAL.png")
        out_im  = os.path.join(outdir, f"{base}_COMPLEX_IMAG.png")
        out_abs = os.path.join(outdir, f"{base}_COMPLEX_ABS.png")

        plot_grid_figure_hermitian(
            rows, omega_grid_complex, title_prefix, out_re,
            part="real", clip_q=args.clip_q, omega_phys=args.omega_phys,
            omega_side=args.omega_side, mask_diag=args.mask_diag,
            support_rel_var=args.support_rel_var, support_mode=args.support_mode
        )
        plot_grid_figure_hermitian(
            rows, omega_grid_complex, title_prefix, out_im,
            part="imag", clip_q=args.clip_q, omega_phys=args.omega_phys,
            omega_side=args.omega_side, mask_diag=args.mask_diag,
            support_rel_var=args.support_rel_var, support_mode=args.support_mode
        )
        plot_grid_figure_hermitian(
            rows, omega_grid_complex, title_prefix, out_abs,
            part="abs", clip_q=args.clip_q, omega_phys=args.omega_phys,
            omega_side=args.omega_side, mask_diag=args.mask_diag,
            support_rel_var=args.support_rel_var, support_mode=args.support_mode
        )

        print(f"[OK] Group n={n_qubits}, L={n_layers}: wrote")
        for outp in [out_re, out_im, out_abs]:
            print(f"     {os.path.basename(outp)}")

        summary["groups"].append({
            "n_qubits": int(n_qubits),
            "n_layers": int(n_layers),
            "n_cols": len(rows),
            "depths": [r["depth"] for r in rows],
            "complex_real": os.path.basename(out_re),
            "complex_imag": os.path.basename(out_im),
            "complex_abs": os.path.basename(out_abs),
            "omega_phys": args.omega_phys,
            "omega_side": args.omega_side,
            "mask_diag": args.mask_diag,
            "support_rel_var": args.support_rel_var,
            "support_mode": args.support_mode,
        })

    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"[INFO] Wrote summary: {os.path.join(outdir, 'summary.json')}")


if __name__ == "__main__":
    main()