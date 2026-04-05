# mean_offdiag_correlation_plot.py
#
# Panel plots of mean off-diagonal correlation vs training depth,
# comparing RX vs RY input encoders across multiple circuit ansatzes.
#
# For each ansatz, produces a panel comparing:
#   - Blue (RX): mean |offdiag(Corr)| for RX-encoded circuits
#   - Red  (RY): mean |offdiag(Corr)| for RY-encoded circuits
# across a range of training depths, with block-bootstrap standard error bars.
#
# Reads .npz files produced by correlation_matrices_compute.py, organised in
# per-ansatz folders whose names encode the circuit and encoder axis, e.g.:
#   outputs_corr_C15_RX/
#   outputs_corr_C15_RY/
#   outputs_corr_HEA_RX/
#   ...
# The ansatz and encoder axis are inferred automatically from folder names.
#
# Produces three figures:
#   panel_offdiag_vs_depth_COMPLEX_ABS.png  : mean |offdiag(Corr)|
#   panel_offdiag_vs_depth_REAL_ABS.png     : mean |offdiag(Re(Corr))|
#   panel_offdiag_vs_depth_IMAG_ABS.png     : mean |offdiag(Im(Corr))|
#
# Usage:
#   python mean_offdiag_correlation_plot.py \
#       --root /path/to/results \
#       --glob "outputs_corr_*" \
#       --outdir figures_mean_offdiag
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
# Robust string extraction
# -------------------------

def _safe_get_str(npz, key, default="UNKNOWN"):
    if key not in npz.files:
        return default
    v = npz[key]
    if isinstance(v, np.ndarray) and v.shape == ():
        try:
            return str(v.item())
        except Exception:
            return default
    try:
        return str(v)
    except Exception:
        return default


def _load_diag(npz):
    if "diag_json" not in npz.files:
        return {}
    try:
        return json.loads(str(npz["diag_json"].item()))
    except Exception:
        return {}


def _dig(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or (k not in cur):
            return default
        cur = cur[k]
    return cur


# -------------------------
# Canonical naming helpers
# -------------------------

def _canon_ansatz_from_folder(folder_name):
    """
    Parse ansatz from folder names like:
      outputs_corr_C15_RX_cmplx128
      outputs_corr_C16_RY_cmplx128
      outputs_corr_HEA_RX_cmplx128
      outputs_corr_YZY_ENTANGLING_RY_cmplx128
      outputs_corr_YZY_RX_cmplx128
    """
    s = folder_name.upper()

    if "_C15_" in s:
        return "C15"
    if "_C16_" in s:
        return "C16"
    if "_C17_" in s:
        return "C17"
    if "_C18_" in s:
        return "C18"
    if "_C19_" in s:
        return "C19"
    if "_HEA_" in s:
        return "HEA"
    if "_YZY_ENTANGLING_" in s:
        return "YZY_ENTANGLING"
    if "_YZY_" in s:
        return "YZY"

    return "UNKNOWN"


def _canon_encoder_from_folder(folder_name):
    s = folder_name.upper()
    if "_RX_" in s:
        return "RX"
    if "_RY_" in s:
        return "RY"
    return "UNKNOWN"


def _folder_passes_suffix(folder_path, folder_suffix):
    if folder_suffix is None:
        return True
    return os.path.basename(folder_path).endswith(folder_suffix)


# -------------------------
# Offdiag stat helpers
# -------------------------

def _offdiag_vals(M):
    M = np.asarray(M)
    n = M.shape[0]
    iu = np.triu_indices(n, k=1)
    return M[iu]


def _mean_abs_offdiag_from_matrix(M):
    v = _offdiag_vals(M)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    return float(np.mean(np.abs(v)))


# -------------------------
# Extract (mean, SE) for each "part"
# -------------------------

def get_mean_se_from_diag_or_matrix(part, Corr_MC, diag):
    """
    part:
      - "abs"     : mean |offdiag(Corr_MC)|, SE from block_se["complex"]
      - "realabs" : mean |offdiag(Re(Corr_MC))|, SE from block_se["complex_re"]
      - "imagabs" : mean |offdiag(Im(Corr_MC))|, SE from block_se["complex_im"]
    """
    # --- try diag_json first ---
    if isinstance(diag, dict) and diag:
        if part == "abs":
            mean = _dig(diag, "complex", "corr_mean_abs_offdiag", default=None)
            se   = _dig(diag, "block_se", "complex", "corr_mean_abs_offdiag_mc_se", default=None)
            if mean is not None:
                try:
                    return float(mean), float(se) if se is not None else np.nan
                except Exception:
                    pass

        if part == "realabs":
            mean = _dig(diag, "complex", "parts", "corr_mean_abs_offdiag_re", default=None)
            se   = _dig(diag, "block_se", "complex_re", "corr_mean_abs_offdiag_mc_se", default=None)
            if mean is not None:
                try:
                    return float(mean), float(se) if se is not None else np.nan
                except Exception:
                    pass

        if part == "imagabs":
            mean = _dig(diag, "complex", "parts", "corr_mean_abs_offdiag_im", default=None)
            se   = _dig(diag, "block_se", "complex_im", "corr_mean_abs_offdiag_mc_se", default=None)
            if mean is not None:
                try:
                    return float(mean), float(se) if se is not None else np.nan
                except Exception:
                    pass

    # --- fallback: compute mean from matrix, SE unavailable ---
    if part == "abs":
        return _mean_abs_offdiag_from_matrix(np.abs(Corr_MC)), np.nan
    if part == "realabs":
        return _mean_abs_offdiag_from_matrix(np.real(Corr_MC)), np.nan
    if part == "imagabs":
        return _mean_abs_offdiag_from_matrix(np.imag(Corr_MC)), np.nan

    raise ValueError(f"Unknown part='{part}'")


# -------------------------
# Collect runs
# -------------------------

def collect_runs(root, folder_glob, npz_pattern, folder_suffix=None):
    """
    Returns a list of runs.
    Only folders matching `folder_suffix` are included when specified.
    Ansatz and encoder axis are inferred from the folder name.
    """
    folders = sorted(glob.glob(os.path.join(root, folder_glob)))
    folders = [fd for fd in folders if os.path.isdir(fd)]

    if folder_suffix is not None:
        folders = [fd for fd in folders if _folder_passes_suffix(fd, folder_suffix)]

    if not folders:
        raise FileNotFoundError(
            f"No folders matched: {os.path.join(root, folder_glob)} "
            f"with folder_suffix={folder_suffix!r}"
        )

    runs = []
    for fd in folders:
        folder_name = os.path.basename(fd)
        ansatz = _canon_ansatz_from_folder(folder_name)
        enc = _canon_encoder_from_folder(folder_name)

        if ansatz == "UNKNOWN" or enc == "UNKNOWN":
            print(f"[WARN] Skipping folder with unparsed naming convention: {fd}")
            continue

        npz_files = sorted(glob.glob(os.path.join(fd, "**", npz_pattern), recursive=True))
        if not npz_files:
            continue

        for p in npz_files:
            try:
                npz = np.load(p, allow_pickle=False)
            except Exception as e:
                print(f"[WARN] Could not load {p}: {e}")
                continue

            if "Corr_MC" not in npz.files or "train_depth" not in npz.files:
                continue

            try:
                depth = int(npz["train_depth"])
            except Exception:
                print(f"[WARN] Could not parse train_depth in {p}")
                continue

            diag = _load_diag(npz)

            runs.append({
                "path": p,
                "folder": folder_name,
                "ansatz": ansatz,
                "encoder_axis": enc,
                "depth": depth,
                "Corr_MC": np.asarray(npz["Corr_MC"]),
                "diag": diag,
            })

    if not runs:
        raise RuntimeError("No compatible runs found after folder filtering and NPZ parsing.")

    return runs


# -------------------------
# Plot panels
# -------------------------

def plot_panel_2x3(runs, outpath, part, circuits, marker_map):
    """
    circuits: list of up to 6 ansatz names to include (one per panel).
    """
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.5), constrained_layout=True)
    axes = axes.ravel()

    part_title = {
        "abs": "mean |offdiag(Corr)|",
        "realabs": "mean |offdiag(Re(Corr))|",
        "imagabs": "mean |offdiag(Im(Corr))|",
    }[part]

    # Build quick index: (ansatz, enc) -> list of runs
    idx = {}
    for r in runs:
        key = (r["ansatz"], r["encoder_axis"])
        idx.setdefault(key, []).append(r)

    for key in idx:
        idx[key] = sorted(idx[key], key=lambda rr: (rr["depth"], rr["path"]))

    for i, ansatz in enumerate(circuits):
        ax = axes[i]
        mk = marker_map.get(ansatz, "o")

        for enc, color, label in [("RX", "C0", "RX"), ("RY", "C3", "RY")]:
            rr = idx.get((ansatz, enc), [])
            if not rr:
                continue

            x = np.array([t["depth"] for t in rr], dtype=float)
            y = []
            e = []
            for t in rr:
                mean, se = get_mean_se_from_diag_or_matrix(part, t["Corr_MC"], t["diag"])
                y.append(mean)
                e.append(se)

            y = np.array(y, dtype=float)
            e = np.array(e, dtype=float)

            m = np.isfinite(x) & np.isfinite(y) & (y > 0)
            x, y, e = x[m], y[m], e[m]

            if x.size == 0:
                continue

            ax.errorbar(
                x, y, yerr=e,
                fmt=mk, linestyle="none",
                capsize=2.5, markersize=6.5,
                color=color, label=label
            )

        ax.set_title(ansatz, fontsize=18)
        ax.set_xlabel("train depth d", fontsize=17)
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.0, top=1.0)
        ax.grid(True, alpha=0.25)

        if i % 3 == 0:
            ax.set_ylabel(part_title, fontsize=18)

        ax.tick_params(axis="both", labelsize=16)
        ax.legend(loc="best", fontsize=15, frameon=True)

    # If fewer than 6 panels are used, turn off extras
    for j in range(len(circuits), 6):
        axes[j].axis("off")

    fig.suptitle(part_title + " vs training depth | RX vs RY encoder", fontsize=20)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


# -------------------------
# Main
# -------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory containing per-ansatz output folders."
    )
    ap.add_argument(
        "--glob",
        type=str,
        default="outputs_corr_*",
        help="Folder glob pattern under root."
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="*.npz",
        help="NPZ glob pattern inside folders (recursive)."
    )
    ap.add_argument(
        "--folder_suffix",
        type=str,
        default="_cmplx128",
        help="Only include folders whose names end with this suffix."
    )
    ap.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory for figures + summary."
    )
    ap.add_argument(
        "--max_circuits",
        type=int,
        default=6,
        help="Number of circuits to panel (default: 6)."
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Print discovered folders and grouping information."
    )
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    runs = collect_runs(
        root=args.root,
        folder_glob=args.glob,
        npz_pattern=args.pattern,
        folder_suffix=args.folder_suffix,
    )

    if args.debug:
        print("\n[DEBUG] Unique folders used:")
        for fd in sorted(set(r["folder"] for r in runs)):
            print("  ", fd)

        print("\n[DEBUG] Unique (ansatz, encoder_axis) pairs found:")
        for k in sorted(set((r["ansatz"], r["encoder_axis"]) for r in runs)):
            print("  ", k)

    # Marker map
    marker_map = {
        "YZY": "^",
        "YZY_ENTANGLING": "v",
        "HEA": "o",
        "C15": "s",
        "C16": "P",
        "C17": "X",
        "C18": "D",
        "C19": "*",
        "UNKNOWN": "o",
    }

    # Choose circuits: take those that have at least one RX and one RY run
    present = sorted(set(r["ansatz"] for r in runs))
    good = []
    for a in present:
        has_rx = any((r["ansatz"] == a and r["encoder_axis"] == "RX") for r in runs)
        has_ry = any((r["ansatz"] == a and r["encoder_axis"] == "RY") for r in runs)
        if has_rx and has_ry:
            good.append(a)

    if args.debug:
        print("\n[DEBUG] Circuits with both RX and RY:")
        for a in good:
            print("  ", a)

    if not good:
        raise RuntimeError(
            "No circuits found with BOTH RX and RY data after filtering. "
            "Try running with --debug to inspect parsed folder groups."
        )

    circuits = good[: int(args.max_circuits)]

    # Produce three figures
    out_abs = os.path.join(args.outdir, "panel_offdiag_vs_depth_COMPLEX_ABS.png")
    out_re  = os.path.join(args.outdir, "panel_offdiag_vs_depth_REAL_ABS.png")
    out_im  = os.path.join(args.outdir, "panel_offdiag_vs_depth_IMAG_ABS.png")

    plot_panel_2x3(runs, out_abs, "abs", circuits, marker_map)
    plot_panel_2x3(runs, out_re, "realabs", circuits, marker_map)
    plot_panel_2x3(runs, out_im, "imagabs", circuits, marker_map)

    # Write summary
    summary = {
        "root": args.root,
        "folder_glob": args.glob,
        "folder_suffix": args.folder_suffix,
        "npz_pattern": args.pattern,
        "circuits_panelled": circuits,
        "figures": {
            "complex_abs": os.path.basename(out_abs),
            "real_abs": os.path.basename(out_re),
            "imag_abs": os.path.basename(out_im),
        },
        "note": (
            "Only folders ending with folder_suffix are included. "
            "Circuit and encoder axis are inferred from folder names. "
            "y-values prefer diag_json means; SE from diag_json block_se when available; "
            "fallback computes means from matrices with SE=NaN."
        ),
    }

    summary_path = os.path.join(args.outdir, "summary_offdiag_panels.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("[OK] Wrote:")
    print("  ", out_abs)
    print("  ", out_re)
    print("  ", out_im)
    print("[OK] Summary:")
    print("  ", summary_path)


if __name__ == "__main__":
    main()