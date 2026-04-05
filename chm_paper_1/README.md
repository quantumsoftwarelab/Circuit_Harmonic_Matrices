# Circuit Harmonic Matrices — Codebase Guide

This codebase accompanies the paper on harmonic analysis of variational quantum circuits.
It implements streaming computation and visualisation of two core objects:

1. **Correlation matrices of Fourier coefficients** — `Corr(a_ω(θ))`, the Pearson
   correlation matrix of the output frequency amplitudes across the parameter distribution.
2. **Harmonic QNTK** — the parameter-averaged quantum neural tangent kernel expressed
   in the Fourier frequency basis.

---

## Background

A variational quantum circuit with input `x` and parameters `θ` produces an expectation value

```
f(x; θ) = <O>_{U(x,θ)|0>}
```

which, for the encoder structure used here (identical single-qubit rotations on every qubit),
has a finite Fourier decomposition in `x`:

```
f(x; θ) = Σ_ω  a_ω(θ) e^{iωx}
```

The **C-matrix** captures the cross-correlation between output harmonics and parameter-space
characters:

```
C_{ω,k} = E_θ[ a_ω(θ) χ_k(θ)* ]
```

where `χ_k(θ) = exp(i k·θ)` and `k ∈ K ⊂ {-1,0,1}^m`.

From the C-matrix (dropping the DC column `k=0`), one can *predict* the covariance of
the Fourier coefficients:

```
Cov_C = C_nz C_nz†
Corr_C = D^{-1/2} Cov_C D^{-1/2}
```

This prediction is compared against a direct Monte Carlo estimate (`Cov_MC`, `Corr_MC`)
obtained by sampling `θ ~ Uniform[0, 2π]^m`.

Similarly, the **harmonic QNTK** is:

```
H(θ) = X(θ) X(θ)†,   X_{ω,a} = ∂_{θ_a} a_ω(θ)
Hbar_MC = E_θ[ H(θ) ]        (Monte Carlo)
Hbar_C  = C diag(||k||²) C†  (C-matrix prediction)
```

---

## File Map

```
circuit_harmonic_matrices/chm_paper_1/
│
├── ansatze.py                        # Core circuit library (shared by all compute scripts)
│
├── correlation_compute.py            # Basic Cov/Corr compute (no block-SE)
├── correlation_matrices_compute.py   # Full Cov/Corr compute with block-bootstrap SE  ← main
│
├── qntk_compute.py                   # Harmonic QNTK compute with block-bootstrap SE
│
├── correlation_matrices_plot.py      # Heatmap plots of Corr_C vs Corr_MC
├── qntk_plot.py                      # Heatmap plots of CorrH_C vs CorrH_MC
│
├── variance_profile_plot.py          # Variance profile Var[a_ω] vs frequency ω
├── mean_offdiag_correlation_plot.py  # Mean off-diagonal correlation vs training depth
│
└── README.md
```

### Role of each file

| File | Role |
|---|---|
| `ansatze.py` | Defines all circuit ansatzes, `CircuitSpec`, and `build_qnode`. Imported by all compute scripts. |
| `correlation_compute.py` | Simpler (no block-SE) streaming estimator of Cov/Corr. Good for quick exploration. |
| `correlation_matrices_compute.py` | Full streaming estimator with block-bootstrap SEs. Primary script for the paper results. |
| `qntk_compute.py` | Streaming QNTK computation using JAX autodiff (`jacrev`) and `lax.scan`. |
| `correlation_matrices_plot.py` | Heatmap grids of Corr_C and Corr_MC (Re, Im, |·|) from a single output directory. |
| `qntk_plot.py` | Heatmap grids of CorrH_C and CorrH_MC from a single output directory. |
| `variance_profile_plot.py` | Plots the variance profile Var[a_ω] vs ω, comparing C-matrix prediction to MC. |
| `mean_offdiag_correlation_plot.py` | Panel plots of mean \|offdiag(Corr)\| vs training depth for multiple ansatzes. |

---

## Typical Workflow

```
1. correlation_matrices_compute.py   →  outputs_correlation_matrices/*.npz
        │
        ├──► correlation_matrices_plot.py   →  figures_correlation_matrices/
        ├──► variance_profile_plot.py       →  figures_variance_profiles/
        └──► mean_offdiag_correlation_plot.py  →  figures_mean_offdiag/

2. qntk_compute.py                   →  outputs_qntk/*.npz
        │
        └──► qntk_plot.py            →  figures_qntk/
```

---

## ansatze.py

The shared circuit library. Not run directly — imported by all compute scripts.

### Circuit structure

Each circuit has `n_layers` layers. Each layer applies:
1. Encoder: `R_{axis}(encoder_scale * x)` on every qubit
2. Trainer block, repeated `train_depth` times

Total parameter count: `m = n_layers * train_depth * p_block`
where `p_block` depends on the ansatz.

### Supported ansatzes

| Name | Description | `p_block` |
|---|---|---|
| `YZY` | Per-qubit RY–RZ–RY, no entanglement | `3n` |
| `YZY_ENTANGLING` | YZY + all-to-all (triangular) CNOT | `3n` |
| `HEA` | YZY + ring CNOT entangler | `3n` |
| `CIRCUIT_15` | RY–ring–RY–ring (Strobl Fig. 6d) | `2n` |
| `CIRCUIT_16` | RX+RZ + CZ + sparse RZ extras (Strobl Fig. 6e) | `3n−1` |
| `CIRCUIT_17` | RX+RZ + CZ + sparse RX extras (Strobl Fig. 6f) | `3n−1` |
| `CIRCUIT_18` | RX+RZ + CZ + dense RZ extras (Strobl Fig. 6g) | `3n` |
| `CIRCUIT_19` | RX+RZ + CZ + dense RX extras (Strobl Fig. 6h) | `3n` |

### Key classes and functions

- `CircuitSpec` — frozen dataclass holding all circuit hyperparameters
- `build_qnode(spec)` — returns a JAX-compatible PennyLane QNode and parameter count `m`

---

## correlation_matrices_compute.py

**Primary script** for computing the correlation matrices `Corr_C` and `Corr_MC` with
block-bootstrap standard errors.

### Algorithm

Streams `n_theta_samples` parameter samples in batches of `batch_size`. Each batch is
randomly assigned to one of two splits:

- **C-split** (`S_C` samples): used to estimate the C-matrix via
  `C_sum[:, k] += a(θ).T @ exp(-i θ K.T)`, processed in K-blocks of size `k_block`.
- **V-split** (`S_V` samples): used to estimate `Cov_MC` from sufficient statistics
  `Σ a` and `Σ a a†`.

Block bootstrap SE is computed over per-batch diagnostic values accumulated during streaming.

### Usage

```bash
python correlation_matrices_compute.py \
    --out_dir outputs_correlation_matrices \
    --ansatz CIRCUIT_17 \
    --encoder_axis RX \
    --n_qubits_list 3 4 \
    --train_depth_list 1 2 3 4 5 \
    --n_layers 3 \
    --n_theta_samples 4096 \
    --batch_size 128 \
    --max_omega 15 \
    --max_hw_for_K 1 \
    --seed 42
```

### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--out_dir` | `outputs_correlation_matrices` | Output directory |
| `--ansatz` | `CIRCUIT_17` | Ansatz name (see ansatze.py) |
| `--encoder_axis` | `RX` | Input encoder rotation axis (`RX`, `RY`, `RZ`) |
| `--obs_kind` | `OZ` | Observable (`OX`, `OY`, `OZ`) |
| `--n_qubits_list` | `[3]` | List of qubit counts to sweep |
| `--train_depth_list` | `[1,2,3,4,5]` | List of training depths to sweep |
| `--n_layers` | `3` | Number of circuit layers |
| `--encoder_scale` | `1.0` | Scaling factor on encoder angle |
| `--n_theta_samples` | `4096` | Total number of θ samples |
| `--split_fraction_for_C` | `0.5` | Fraction of samples assigned to C-split |
| `--batch_size` | `128` | Batch size for θ streaming |
| `--k_block` | `5000` | Block size for K-chunking |
| `--max_omega` | `15` | Output frequencies ω ∈ [−max_omega, max_omega] |
| `--max_hw_for_K` | `1` | Maximum Hamming weight of k-vectors in K |
| `--max_K_cap` | `30000` | Hard cap on |K| |
| `--seed` | `1234` | Random seed |

### Output .npz keys

Each run saves one `.npz` file. Key arrays:

| Key | Shape | Description |
|---|---|---|
| `omega_grid` | `(n_omega,)` | Integer frequency grid |
| `K` | `(n_K, m)` | Parameter-space harmonic set |
| `C` | `(n_omega, n_K)` | C-matrix estimate |
| `Corr_C` | `(n_omega, n_omega)` | C-matrix correlation prediction |
| `Corr_MC` | `(n_omega, n_omega)` | Direct MC correlation estimate |
| `Cov_C` | `(n_omega, n_omega)` | C-matrix covariance prediction |
| `Cov_MC` | `(n_omega, n_omega)` | Direct MC covariance estimate |
| `var_C` | `(n_omega,)` | Diagonal of `Cov_C` (variance profile) |
| `var_MC` | `(n_omega,)` | Diagonal of `Cov_MC` (variance profile) |
| `diag_json` | scalar string | JSON blob with scalar diagnostics and block-SE |

---

## correlation_compute.py

A simpler version of `correlation_matrices_compute.py` without block-bootstrap standard
errors. Useful for quick exploration or debugging. The output `.npz` format is compatible
with the plotting scripts.

### Usage

```bash
python correlation_compute.py \
    --out_dir outputs_correlation_compute \
    --n_qubits_list 3 \
    --train_depth_list 1 2 3 \
    --n_layers 3 \
    --n_theta_samples 2048
```

Note: the ansatz, encoder axis, and observable are set inside the script (see the
"Default circuit selection" comment in `main()`).

---

## qntk_compute.py

Computes the harmonic QNTK `Hbar_MC` and its C-matrix prediction `Hbar_C`, along with
their correlation-normalised versions `CorrH_MC` and `CorrH_C`.

Gradients `∂_{θ_a} a_ω(θ)` are computed via `jax.jacrev` applied to the vectorised
circuit output over the x-grid. Accumulation uses `jax.lax.scan` to avoid materialising
the full `(B, n_omega, n_omega)` QNTK tensor.

### Usage

```bash
python qntk_compute.py \
    --out_dir outputs_qntk \
    --ansatz CIRCUIT_17 \
    --encoder_axis RX \
    --n_qubits_list 3 \
    --train_depth_list 1 2 3 4 5 \
    --n_layers 3 \
    --n_theta_samples 512 \
    --batch_size 32 \
    --max_omega 15 \
    --max_hw_for_K 1 \
    --seed 42
```

### CLI arguments

The argument set mirrors `correlation_matrices_compute.py`. Additional/different defaults:

| Argument | Default | Description |
|---|---|---|
| `--out_dir` | `outputs_qntk` | Output directory |
| `--n_theta_samples` | `512` | Fewer samples needed (gradients are more expensive) |
| `--batch_size` | `32` | Smaller batches recommended (JAX compilation overhead) |
| `--diff_method` | `parameter-shift` | PennyLane differentiation method |

### Output .npz keys

| Key | Shape | Description |
|---|---|---|
| `omega_grid` | `(n_omega,)` | Integer frequency grid |
| `Hbar_MC` | `(n_omega, n_omega)` | MC-averaged QNTK |
| `Hbar_C` | `(n_omega, n_omega)` | C-matrix QNTK prediction |
| `CorrH_MC` | `(n_omega, n_omega)` | Correlation-normalised Hbar_MC |
| `CorrH_C` | `(n_omega, n_omega)` | Correlation-normalised Hbar_C |
| `diag_json` | scalar string | JSON blob with scalar diagnostics and block-SE |

---

## correlation_matrices_plot.py

Plots `Corr_C` and `Corr_MC` side-by-side as heatmaps for each training depth,
from a directory of `.npz` files produced by `correlation_matrices_compute.py`.

Three figures are produced per (n_qubits, n_layers) group:
- `*_COMPLEX_REAL.png` — Re(Corr)
- `*_COMPLEX_IMAG.png` — Im(Corr)
- `*_COMPLEX_ABS.png`  — |Corr|

Each column corresponds to one training depth. Row 0 shows `Corr_C`, row 1 shows
`Corr_MC`. A textbox beneath each column reports scalar diagnostics.

### Usage

```bash
python correlation_matrices_plot.py \
    --indir outputs_correlation_matrices \
    --outdir figures_correlation_matrices

# With variance-support masking (zero out entries with low spectral weight):
python correlation_matrices_plot.py \
    --indir outputs_correlation_matrices \
    --outdir figures_correlation_matrices \
    --mask_support \
    --support_threshold 0.01
```

### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--indir` | required | Directory of `.npz` files |
| `--outdir` | `{indir}/plots_corrH_grid` | Output directory |
| `--pattern` | `*.npz` | Glob pattern for input files |
| `--omega_phys` | `None` | Restrict to \|ω\| ≤ omega_phys |
| `--omega_side` | `all` | Restrict to `all`, `nonneg`, or `nonpos` frequencies |
| `--mask_support` | off | Enable variance-support masking |
| `--support_threshold` | `0.01` | Fraction of max variance below which entries are masked |
| `--clip_q` | `0.995` | Quantile for colour scale clipping |
| `--mask_diag` | `none` | Diagonal masking: `none`, `zero`, `nan`, or `trivial` |

---

## qntk_plot.py

Plots `CorrH_C` and `CorrH_MC` side-by-side as heatmaps. Usage mirrors
`correlation_matrices_plot.py`, reading `.npz` files from `qntk_compute.py`.

### Usage

```bash
python qntk_plot.py \
    --indir outputs_qntk \
    --outdir figures_qntk

python qntk_plot.py \
    --indir outputs_qntk \
    --outdir figures_qntk \
    --mask_diag trivial \
    --omega_phys 10
```

### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--indir` | required | Directory of `.npz` files from `qntk_compute.py` |
| `--outdir` | `{indir}/plots_qntkH_grid` | Output directory |
| `--pattern` | `*.npz` | Glob pattern for input files |
| `--omega_phys` | `None` | Restrict to \|ω\| ≤ omega_phys |
| `--omega_side` | `all` | `all`, `nonneg`, or `nonpos` |
| `--clip_q` | `0.995` | Quantile for colour scale |
| `--mask_diag` | `none` | `none`, `zero`, `nan`, or `trivial` |
| `--trivial_diag_tol` | `1e-12` | Tolerance for detecting trivial diagonal entries |
| `--force_corr_from_hbar` | off | Recompute CorrH from Hbar (ignore stored CorrH) |

---

## variance_profile_plot.py

Plots the variance profile `Var[a_ω]` vs frequency `ω`, comparing the C-matrix
prediction (`var_C`, diagonal of `Cov_C`) to the direct MC estimate (`var_MC`).

### Usage

```bash
python variance_profile_plot.py \
    --indir outputs_correlation_matrices \
    --outdir figures_variance_profiles \
    --mode relmax

python variance_profile_plot.py \
    --indir outputs_correlation_matrices \
    --outdir figures_variance_profiles \
    --mode logrel \
    --omega_phys 6
```

### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--indir` | required | Directory of `.npz` files |
| `--outdir` | required | Output directory |
| `--pattern` | `*.npz` | Glob pattern |
| `--mode` | `relmax` | `raw`, `norm`, `relmax`, or `logrel` |
| `--omega_phys` | `None` | Restrict to \|ω\| ≤ omega_phys |
| `--omega_side` | `all` | `all`, `nonneg`, or `nonpos` |
| `--ylim` | `None` | y-axis limits, e.g. `--ylim -16 0` for logrel |
| `--no_textbox` | off | Suppress per-panel summary statistics |
| `--no_legend` | off | Suppress legend |

---

## mean_offdiag_correlation_plot.py

Produces panel plots of mean `|offdiag(Corr)|` vs training depth, comparing RX vs RY
input encoders across multiple circuit ansatzes. Requires outputs organised in per-ansatz
folders whose names encode the circuit and encoder axis (e.g. `outputs_corr_C15_RX/`).

The ansatz and encoder axis are inferred automatically from folder names.

### Usage

```bash
python mean_offdiag_correlation_plot.py \
    --root /path/to/results \
    --glob "outputs_corr_*" \
    --outdir figures_mean_offdiag
```

### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--root` | required | Root directory containing per-ansatz output folders |
| `--glob` | `outputs_corr_*` | Folder glob pattern under root |
| `--pattern` | `*.npz` | NPZ glob within each folder (recursive) |
| `--folder_suffix` | `_cmplx128` | Only include folders ending with this suffix |
| `--outdir` | required | Output directory |
| `--max_circuits` | `6` | Maximum number of circuits to panel |
| `--debug` | off | Print folder parsing information |

---

## Dependencies

```
pennylane       >= 0.38
jax             >= 0.4
jaxlib
numpy
matplotlib
tqdm
```

Install via:
```bash
pip install pennylane jax jaxlib numpy matplotlib tqdm
```

---

## Quick-start Example

Run the full pipeline for a 3-qubit `CIRCUIT_17` circuit with RX encoding:

```bash
# 1. Compute correlation matrices
python correlation_matrices_compute.py \
    --out_dir outputs_C17_RX \
    --ansatz CIRCUIT_17 \
    --encoder_axis RX \
    --n_qubits_list 3 \
    --train_depth_list 1 2 3 4 5 \
    --n_layers 3 \
    --n_theta_samples 4096 \
    --max_omega 10 \
    --max_hw_for_K 1

# 2. Plot correlation matrix heatmaps
python correlation_matrices_plot.py \
    --indir outputs_C17_RX \
    --outdir figures_C17_RX_heatmaps

# 3. Plot variance profiles
python variance_profile_plot.py \
    --indir outputs_C17_RX \
    --outdir figures_C17_RX_variance \
    --mode relmax

# 4. Compute QNTK
python qntk_compute.py \
    --out_dir outputs_C17_RX_qntk \
    --ansatz CIRCUIT_17 \
    --encoder_axis RX \
    --n_qubits_list 3 \
    --train_depth_list 1 2 3 4 5 \
    --n_layers 3 \
    --n_theta_samples 256

# 5. Plot QNTK heatmaps
python qntk_plot.py \
    --indir outputs_C17_RX_qntk \
    --outdir figures_C17_RX_qntk
```