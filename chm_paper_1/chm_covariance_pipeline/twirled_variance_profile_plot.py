#!/usr/bin/env python3
"""Plot CHM Fourier-coefficient variance profiles.

The script reads the ``.npz`` files produced by
``twirled_correlation_matrices_compute.py`` and plots ``var_C``/``var_TW``
against the optional Monte Carlo estimate ``var_MC``.  The variance profile
shows how the circuit distributes coefficient energy across output frequencies.

Available plot modes are raw variance, normalised spectral weight, variance
relative to the maximum, and log10 variance relative to the maximum.
"""

import os
import glob
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ======================================================
# Loading / parsing
# ======================================================

REQUIRED_KEYS = ["omega_grid", "var_C", "var_MC"]


def parse_diag_from_npz(npz) -> dict:
    if "diag_json" not in npz.files:
        return {}
    try:
        return json.loads(str(npz["diag_json"].item()))
    except Exception:
        return {}


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def _dig(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or (k not in cur):
            return default
        cur = cur[k]
    return cur


def normalise(v):
    v = np.asarray(v, dtype=np.float64)
    s = float(np.sum(v))
    return v / s if s > 0 else np.zeros_like(v)


def relative_to_max(v):
    v = np.asarray(v, dtype=np.float64)
    m = float(np.max(v)) if v.size else 0.0
    return v / m if m > 0 else np.zeros_like(v)


def log10_relative_to_max(v, floor=1e-16):
    v = np.asarray(v, dtype=np.float64)
    m = float(np.max(v)) if v.size else 0.0
    if m <= 0:
        return np.full_like(v, np.log10(floor), dtype=np.float64)
    rel = v / m
    rel = np.maximum(rel, floor)
    return np.log10(rel)


def get_paths(indir: str, pattern: str | None):
    if pattern is None:
        pattern = "*.npz"
    return sorted(glob.glob(os.path.join(indir, pattern)))


def load_runs(indir: str, pattern: str | None):
    paths = get_paths(indir, pattern)
    out = []
    for path in paths:
        try:
            npz = np.load(path, allow_pickle=False)
        except Exception as e:
            print(f"[WARN] Could not load {path}: {e}")
            continue

        missing = [k for k in REQUIRED_KEYS if k not in npz.files]
        if missing:
            print(f"[SKIP] {os.path.basename(path)} missing keys: {missing}")
            continue

        diag = parse_diag_from_npz(npz)

        row = {
            "path": path,
            "file": os.path.basename(path),
            "omega_grid": npz["omega_grid"].astype(int),
            "var_C": np.asarray(npz["var_C"], dtype=np.float64),
            "var_MC": np.asarray(npz["var_MC"], dtype=np.float64),
            "diag": diag,
            "n_qubits": int(npz["n_qubits"]) if "n_qubits" in npz.files else None,
            "train_depth": int(npz["train_depth"]) if "train_depth" in npz.files else None,
            "n_layers": int(npz["n_layers"]) if "n_layers" in npz.files else None,
            "S_C": int(npz["S_C"]) if "S_C" in npz.files else None,
            "S_V": int(npz["S_V"]) if "S_V" in npz.files else None,
            "seed": int(npz["seed"]) if "seed" in npz.files else None,
            "var_C_se": np.asarray(npz["var_C_se"], dtype=np.float64) if "var_C_se" in npz.files else None,
            "var_MC_se": np.asarray(npz["var_MC_se"], dtype=np.float64) if "var_MC_se" in npz.files else None,
        }
        out.append(row)
    return out


# ======================================================
# Restrictions / metrics
# ======================================================

def restrict_to_omega_band(omega_grid, arrs, omega_phys=None, omega_side="all"):
    og = np.asarray(omega_grid, dtype=int)
    keep = np.ones_like(og, dtype=bool)

    if omega_phys is not None:
        keep &= (np.abs(og) <= int(omega_phys))

    if omega_side == "all":
        pass
    elif omega_side == "nonneg":
        keep &= (og >= 0)
    elif omega_side == "nonpos":
        keep &= (og <= 0)
    else:
        raise ValueError("omega_side must be one of {'all','nonneg','nonpos'}")

    if np.count_nonzero(keep) < 2:
        raise ValueError("Restriction keeps fewer than 2 frequencies.")

    arrs_r = [None if a is None else np.asarray(a)[keep] for a in arrs]
    return og[keep], arrs_r


def rel_l2(u, v, eps=1e-14):
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return float(np.linalg.norm(u - v) / (np.linalg.norm(v) + eps))


def cosine_similarity(u, v, eps=1e-14):
    u = np.asarray(u)
    v = np.asarray(v)
    num = np.vdot(u, v)
    den = (np.linalg.norm(u) * np.linalg.norm(v) + eps)
    return float(np.real(num / den))


def pearson_centered(u, v, eps=1e-14):
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    u0 = u - np.mean(u)
    v0 = v - np.mean(v)
    den = np.linalg.norm(u0) * np.linalg.norm(v0)
    if den < eps:
        return np.nan
    return float(np.dot(u0, v0) / den)


def rmse(u, v):
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return float(np.sqrt(np.mean((u - v) ** 2)))


# ======================================================
# Mode transforms
# ======================================================

def transform_profile(y, y_se, mode, floor=1e-16):
    """
    Apply plotting transform to y and, when sensible, to y_se.
    For logrel, pointwise error bars are omitted by default.
    """
    y = np.asarray(y, dtype=np.float64)
    y_se = None if y_se is None else np.asarray(y_se, dtype=np.float64)

    if mode == "raw":
        return y, y_se, "raw-var", r"$\mathrm{Var}[a_\omega]$"

    if mode == "norm":
        s = max(float(np.sum(y)), 1e-14)
        yt = y / s
        yt_se = None if y_se is None else (y_se / s)
        return yt, yt_se, "norm-var", "Normalised weight"

    if mode == "relmax":
        m = max(float(np.max(y)), 1e-14)
        yt = y / m
        yt_se = None if y_se is None else (y_se / m)
        return yt, yt_se, "relmax-var", "Relative variance"

    if mode == "logrel":
        m = float(np.max(y))
        if m <= 0:
            yt = np.full_like(y, np.log10(floor), dtype=np.float64)
        else:
            yt = np.log10(np.maximum(y / m, floor))
        # Error bars after log transform are not plotted by default.
        return yt, None, "logrel-var", r"$\log_{10}\!\left(\mathrm{Var}/\max\mathrm{Var}\right)$"

    raise ValueError("mode must be one of {'raw','norm','relmax','logrel'}")


# ======================================================
# Textbox helpers
# ======================================================

def textbox_lines(diag, var_C, var_MC, uC, uM, mode_label):
    lines = []

    corr_rel_frob = _dig(diag, "complex", "corr_rel_frob", default=None)
    corr_cos = _dig(diag, "complex", "corr_offdiag_cosine", default=None)
    corr_rel_frob_se = _dig(diag, "block_se", "complex", "corr_rel_frob_se", default=None)
    corr_cos_se = _dig(diag, "block_se", "complex", "corr_offdiag_cosine_se", default=None)

    lines.append(f"{mode_label} pearson: {pearson_centered(uM, uC):.3f}")
    lines.append(f"{mode_label} RMSE: {rmse(uM, uC):.3e}")
    lines.append(f"Linear RMSE: {rmse(var_MC, var_C):.3e}")

    return "\n".join(lines)


# ======================================================
# Plotting
# ======================================================

def plot_grid(
    rows,
    out_path,
    mode="relmax",
    omega_phys=None,
    omega_side="all",
    show_textbox=True,
    show_legend=True,
    ylim=None,
    title=None,
):
    depths = [r["train_depth"] for r in rows]
    ncols = len(rows)

    fig, axes = plt.subplots(
        2, ncols,
        figsize=(4.8 * ncols, 5.4),
        squeeze=False,
        gridspec_kw={"height_ratios": [1.0, 0.35]}
    )

    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.10, top=0.84, wspace=0.25, hspace=0.28)

    title_fs = 18
    axis_label_fs = 15
    tick_fs = 11
    legend_fs = 13
    textbox_fs = 15
    suptitle_fs = 18

    for j, r in enumerate(rows):
        ax = axes[0, j]

        omega, arrs = restrict_to_omega_band(
            r["omega_grid"],
            [r["var_C"], r["var_MC"], r["var_C_se"], r["var_MC_se"]],
            omega_phys=omega_phys,
            omega_side=omega_side,
        )
        var_C, var_MC, var_C_se, var_MC_se = arrs

        yC, yC_se, mode_label, ylabel = transform_profile(var_C, var_C_se, mode)
        yM, yM_se, _, _ = transform_profile(var_MC, var_MC_se, mode)

        if yC_se is not None:
            ax.errorbar(
                omega, yC, yerr=yC_se,
                fmt="s--", linewidth=1.0, markersize=5,
                capsize=2.5, label=r"Row Energy of $C$"
            )
        else:
            ax.plot(
                omega, yC,
                "s--", linewidth=1.0, markersize=5,
                label=r"Row Energy of $C$"
            )

        if yM_se is not None:
            ax.errorbar(
                omega, yM, yerr=yM_se,
                fmt="^-.", linewidth=1.0, markersize=5,
                capsize=2.5, label=r"Variance from MC"
            )
        else:
            ax.plot(
                omega, yM,
                "^-.", linewidth=1.0, markersize=5,
                label=r"Variance from MC"
            )

        ax.set_title(f"depth={r['train_depth']}", fontsize=title_fs)
        ax.set_xlabel(r"frequency $\omega$", fontsize=axis_label_fs)
        ax.set_ylabel(ylabel, fontsize=axis_label_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.grid(True, alpha=0.3)

        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])

        ax_txt = axes[1, j]
        ax_txt.axis("off")
        if show_textbox:
            txt = textbox_lines(r["diag"], var_C, var_MC, yC, yM, mode_label)
            ax_txt.text(
                0.5, 0.5, txt,
                ha="center", va="center", fontsize=textbox_fs
            )

    if show_legend:
        axes[0, 0].legend(fontsize=legend_fs, loc="best")

    if title is None:
        n = rows[0]["n_qubits"]
        L = rows[0]["n_layers"]
        side_note = {"all": "all ω", "nonneg": "ω≥0", "nonpos": "ω≤0"}[omega_side]
        band_note = f", |ω|≤{int(omega_phys)}" if omega_phys is not None else ""
        mode_note = {
            "raw": "raw",
            "norm": "normalised",
            "relmax": "relative-to-max",
            "logrel": "log10 relative-to-max",
        }[mode]
        title = f"Variance profiles ({mode_note}) | n={n}, L={L} | {side_note}{band_note}"

    fig.suptitle(title, fontsize=suptitle_fs)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ======================================================
# CSV summary
# ======================================================

def write_csv_summary(rows, out_csv, mode="relmax", omega_phys=None, omega_side="all"):
    with open(out_csv, "w") as f:
        f.write(
            "n_qubits,n_layers,train_depth,S_C,S_V,"
            "relL2_var,cosine_var,pearson_var,rmse_var,"
            "corr_rel_frob,corr_rel_frob_se,corr_offdiag_cosine,corr_offdiag_cosine_se,"
            "sum_var_C,sum_var_MC,max_var_C,max_var_MC,file\n"
        )

        for r in sorted(rows, key=lambda z: (z["n_qubits"], z["n_layers"], z["train_depth"])):
            omega, arrs = restrict_to_omega_band(
                r["omega_grid"], [r["var_C"], r["var_MC"]],
                omega_phys=omega_phys, omega_side=omega_side
            )
            var_C, var_MC = arrs

            uC, _, _, _ = transform_profile(var_C, None, mode)
            uM, _, _, _ = transform_profile(var_MC, None, mode)

            diag = r["diag"]
            row = [
                r["n_qubits"],
                r["n_layers"],
                r["train_depth"],
                r["S_C"],
                r["S_V"],
                rel_l2(uM, uC),
                cosine_similarity(uM, uC),
                pearson_centered(uM, uC),
                rmse(uM, uC),
                safe_float(_dig(diag, "complex", "corr_rel_frob")),
                safe_float(_dig(diag, "block_se", "complex", "corr_rel_frob_se")),
                safe_float(_dig(diag, "complex", "corr_offdiag_cosine")),
                safe_float(_dig(diag, "block_se", "complex", "corr_offdiag_cosine_se")),
                float(np.sum(var_C)),
                float(np.sum(var_MC)),
                float(np.max(var_C)),
                float(np.max(var_MC)),
                r["file"],
            ]
            f.write(",".join(f"{float(x):.6g}" if isinstance(x, (float, np.floating)) else str(x) for x in row) + "\n")


# ======================================================
# Main
# ======================================================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, required=True, help="Directory containing .npz files from twirled_correlation_matrices_compute.py.")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory for figures + CSV.")
    ap.add_argument("--pattern", type=str, default="*.npz", help="Glob pattern for .npz files.")

    ap.add_argument("--mode", type=str, default="relmax", choices=["raw", "norm", "relmax", "logrel"],
                    help="Plot raw, normalised-by-sum, relative-to-max, or log10(relative-to-max) variance profiles.")
    ap.add_argument("--omega_phys", type=int, default=None,
                    help="Restrict to |omega|<=omega_phys.")
    ap.add_argument("--omega_side", type=str, default="all", choices=["all", "nonneg", "nonpos"],
                    help="Keep all / nonnegative / nonpositive frequencies only.")

    ap.add_argument("--ylim", type=float, nargs=2, default=None,
                    help="Optional y-limits, e.g. --ylim 0 1 or --ylim -16 0 for logrel")
    ap.add_argument("--no_textbox", action="store_true")
    ap.add_argument("--no_legend", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = load_runs(args.indir, args.pattern)
    if not rows:
        raise RuntimeError("No compatible .npz files found.")

    groups = {}
    for r in rows:
        key = (r["n_qubits"], r["n_layers"])
        groups.setdefault(key, []).append(r)

    all_rows_for_csv = []

    for (n_qubits, n_layers), grp in sorted(groups.items()):
        grp = sorted(grp, key=lambda z: z["train_depth"])
        all_rows_for_csv.extend(grp)

        depths_str = "_".join(str(r["train_depth"]) for r in grp)
        base = (
            f"var_profiles_{args.mode}"
            f"_n{n_qubits}_L{n_layers}"
            f"_depths_{depths_str}"
            f"_side{args.omega_side}"
        )
        if args.omega_phys is not None:
            base += f"_w{int(args.omega_phys)}"

        out_png = os.path.join(args.outdir, base + ".png")

        plot_grid(
            grp,
            out_path=out_png,
            mode=args.mode,
            omega_phys=args.omega_phys,
            omega_side=args.omega_side,
            show_textbox=(not args.no_textbox),
            show_legend=(not args.no_legend),
            ylim=args.ylim,
        )
        print(f"[OK] Wrote figure: {out_png}")

    out_csv = os.path.join(args.outdir, f"var_profiles_summary_{args.mode}.csv")
    write_csv_summary(
        all_rows_for_csv,
        out_csv,
        mode=args.mode,
        omega_phys=args.omega_phys,
        omega_side=args.omega_side,
    )
    print(f"[OK] Wrote CSV summary: {out_csv}")


if __name__ == "__main__":
    main()