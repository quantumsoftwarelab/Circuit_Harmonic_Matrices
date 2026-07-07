# CHM covariance pipeline

This folder is a cleaned public version of the Circuit Harmonic Matrix covariance workflow.  It contains only the scripts needed to compute and plot Fourier-coefficient covariance/correlation data using exact two-copy parameter twirling, with optional Monte Carlo validation.

## Included files

- `ansatze.py` — minimal PennyLane/JAX circuit library for Monte Carlo validation.
- `exact_covariance_twirled.py` — exact two-copy parameter-twirled covariance for one circuit configuration.
- `twirled_correlation_matrices_compute.py` — batch computation over qubit/depth/layer sweeps.
- `twirled_correlation_matrices_plot.py` — correlation heatmap grids from batch `.npz` outputs.
- `twirled_variance_profile_plot.py` — Fourier-coefficient variance profile plots.
- `experiment_gui.py` — Tkinter launcher for the same command-line workflow.

## Not included from the uploaded working branch

- `cov_estimator.py` was excluded because the uploaded version is tied to parameter-tying experiments and imports `tying`.
- `diagnose_delta_norms.py` was excluded because it is a pruning/Cov-shift diagnostic and imports pruning metadata.
- Phase-alignment, Fisher/QFIM, pruning, gate-schematic, and prediction-launcher paths were removed from the GUI.

## Dependencies

Core exact twirling requires `numpy`; plotting requires `matplotlib`.  Monte Carlo validation additionally requires `pennylane` and `jax`.  `tqdm` is optional for progress bars.

## Example commands

Single exact covariance run:

```bash
python exact_covariance_twirled.py \
  --out_dir outputs/exact_twirled/C18_n6_L1_d4 \
  --ansatz CIRCUIT_18 --n_qubits 6 --n_layers 1 --train_depth 4 \
  --encoder_axis RX --obs_kind OZ --max_states 20000000 --progress
```

Batch covariance/correlation run:

```bash
python twirled_correlation_matrices_compute.py \
  --out_dir outputs/twirled_covariance/C18_n6_d1-4 \
  --ansatz CIRCUIT_18 --n_qubits_list 6 --train_depth_list 1 2 3 4 \
  --n_layers 1 --encoder_axis RX --obs_kind OZ --max_omega 6 \
  --n_theta_samples 4096 --batch_size 64 --progress
```

Correlation plots:

```bash
python twirled_correlation_matrices_plot.py \
  --indir outputs/twirled_covariance/C18_n6_d1-4 \
  --outdir outputs/figures/C18_n6_d1-4/correlation_heatmaps
```

Variance profiles:

```bash
python twirled_variance_profile_plot.py \
  --indir outputs/twirled_covariance/C18_n6_d1-4 \
  --outdir outputs/figures/C18_n6_d1-4/variance_profiles \
  --mode relmax
```

GUI:

```bash
python experiment_gui.py
```
