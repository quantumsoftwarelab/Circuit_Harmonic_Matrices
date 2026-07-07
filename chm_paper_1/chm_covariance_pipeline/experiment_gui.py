#!/usr/bin/env python3
"""Tkinter launcher for the CHM covariance pipeline.

The GUI is a thin subprocess wrapper around the command-line scripts in this
repository.  It is intentionally limited to the public CHM covariance workflow:

1. exact two-copy twirled covariance for one configuration;
2. batched covariance/correlation computation over qubit/depth/layer sweeps;
3. correlation-matrix heatmaps;
4. Fourier-coefficient variance-profile plots.

It does not import PennyLane, JAX, or the numerical modules at startup.  Each run
prints the exact command before execution so that GUI runs remain reproducible
from a terminal.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


ANSATZES = [
    "YZY",
    "YZY_ENTANGLING",
    "HEA",
    "CIRCUIT_15",
    "CIRCUIT_16",
    "CIRCUIT_17",
    "CIRCUIT_18",
    "CIRCUIT_19",
    "CIRCUIT_ZZ",
    "CIRCUIT_MULTIOBS",
    "CIRCUIT_ENTFIRST",
]
AXES = ["RX", "RY", "RZ"]
OBS_KINDS = ["OX", "OY", "OZ", "OZZ"]
THETA_PERIODS = ["2pi", "4pi"]
OMEGA_SIDES = ["all", "nonneg", "nonpos"]
SUPPORT_MODES = ["c", "exact", "twirled", "mc", "intersection", "union"]
MASK_DIAG_MODES = ["none", "zero", "nan"]
COLUMN_AXES = ["depth", "layers"]
VARIANCE_MODES = ["raw", "norm", "relmax", "logrel"]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def shell_join(cmd: Iterable[str]) -> str:
    """Return a readable shell command for logs and dry-run previews."""
    return " ".join(shlex.quote(str(part)) for part in cmd)


def split_words(text: str) -> List[str]:
    """Split a whitespace-separated entry field into CLI tokens."""
    return [item for item in str(text).split() if item.strip()]


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def normalise_subprocess_text(text: str) -> str:
    """Clean progress-bar and ANSI control sequences before writing to Tk text."""
    if not text:
        return text
    if "\r" in text:
        text = text.split("\r")[-1]
    text = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", text)
    return text


@dataclass
class CommandSpec:
    """A command, its required inputs, and its expected outputs."""

    name: str
    cmd: List[str]
    prerequisites: List[Tuple[str, Path]]
    expected_outputs: List[Tuple[str, Path]]


class VarStore:
    """Small wrapper around Tk variables with JSON serialisation support."""

    def __init__(self) -> None:
        self._vars: dict[str, tk.Variable] = {}

    def str(self, name: str, value: str = "") -> tk.StringVar:
        var = tk.StringVar(value=value)
        self._vars[name] = var
        return var

    def bool(self, name: str, value: bool = False) -> tk.BooleanVar:
        var = tk.BooleanVar(value=value)
        self._vars[name] = var
        return var

    def get(self, name: str):
        return self._vars[name].get()

    def set(self, name: str, value) -> None:
        if name in self._vars:
            self._vars[name].set(value)

    def to_dict(self) -> dict[str, object]:
        return {key: var.get() for key, var in self._vars.items()}

    def load_dict(self, data: dict[str, object]) -> None:
        for key, value in data.items():
            if key in self._vars:
                self._vars[key].set(value)


# ---------------------------------------------------------------------------
# GUI application
# ---------------------------------------------------------------------------


class CHMCovarianceGUI(tk.Tk):
    """Main Tk application."""

    def __init__(self) -> None:
        super().__init__()
        self.title("CHM covariance pipeline")
        self.geometry("1180x860")
        self.minsize(980, 720)

        self.vars = VarStore()
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.process: Optional[subprocess.Popen[str]] = None
        self.worker: Optional[threading.Thread] = None
        self.stop_requested = False

        self._make_vars()
        self._build_ui()
        self.after(100, self._poll_log_queue)
        self.refresh_paths()

    # ------------------------------------------------------------------
    # Variables and paths
    # ------------------------------------------------------------------

    def _make_vars(self) -> None:
        default_root = Path(__file__).resolve().parent
        self.vars.str("project_root", str(default_root))
        self.vars.str("python_exe", sys.executable)
        self.vars.str("scripts_subdir", "")

        self.vars.str("ansatz", "CIRCUIT_18")
        self.vars.str("n_qubits", "6")
        self.vars.str("n_qubits_list", "6")
        self.vars.str("n_layers", "1")
        self.vars.str("n_layers_list", "")
        self.vars.str("train_depth", "4")
        self.vars.str("train_depth_list", "1 2 3 4")
        self.vars.str("encoder_axis", "RX")
        self.vars.str("encoder_scale", "1")
        self.vars.str("obs_kind", "OZ")
        self.vars.str("tag_override", "")

        self.vars.str("exact_dir", "")
        self.vars.str("batch_dir", "")
        self.vars.str("figure_dir", "")

        self.vars.str("max_omega", "6")
        self.vars.str("omega_values", "")
        self.vars.str("theta_period", "2pi")
        self.vars.str("cr_quadrature_points", "64")
        self.vars.str("combine_tol", "1e-14")
        self.vars.str("max_states", "20000000")
        self.vars.bool("progress", True)

        self.vars.str("n_theta_samples", "4096")
        self.vars.str("n_x", "256")
        self.vars.str("x_min", "0.0")
        self.vars.str("x_max", "6.283185307179586")
        self.vars.str("batch_size", "64")
        self.vars.str("seed", "1234")
        self.vars.bool("save_mc_samples", True)
        self.vars.str("ansatze_path", "")
        self.vars.str("twirl_module_path", "")

        self.vars.str("plot_pattern", "*.npz")
        self.vars.str("omega_phys", "")
        self.vars.str("omega_side", "all")
        self.vars.str("support_rel_var", "1e-10")
        self.vars.str("support_mode", "c")
        self.vars.str("mask_diag", "none")
        self.vars.str("column_axis", "depth")
        self.vars.str("n_bootstrap", "200")
        self.vars.str("bootstrap_seed", "0")
        self.vars.bool("unit_diag_white", True)
        self.vars.str("clip_q", "0.995")

        self.vars.str("variance_mode", "relmax")
        self.vars.str("variance_ylim", "")
        self.vars.bool("variance_textbox", True)
        self.vars.bool("variance_legend", True)

        self.vars.bool("run_compute", True)
        self.vars.bool("run_corr_plots", True)
        self.vars.bool("run_variance_plots", True)

    def project_root(self) -> Path:
        return Path(str(self.vars.get("project_root"))).expanduser().resolve()

    def scripts_root(self) -> Path:
        subdir = str(self.vars.get("scripts_subdir")).strip()
        return self.project_root() / subdir if subdir else self.project_root()

    def script_path(self, filename: str) -> Path:
        return self.scripts_root() / filename

    def path_from_var(self, name: str) -> Path:
        raw = str(self.vars.get(name)).strip()
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = self.project_root() / path
        return path

    def py_cmd(self, filename: str) -> List[str]:
        return [str(self.vars.get("python_exe")), str(self.script_path(filename))]

    def tag(self) -> str:
        explicit = str(self.vars.get("tag_override")).strip()
        if explicit:
            return explicit
        return (
            f"{self.vars.get('ansatz')}_n{self.vars.get('n_qubits')}"
            f"_L{self.vars.get('n_layers')}_d{self.vars.get('train_depth')}"
            f"_{self.vars.get('encoder_axis')}_{self.vars.get('obs_kind')}"
        )

    def refresh_paths(self) -> None:
        tag = self.tag()
        batch_tag = (
            f"{self.vars.get('ansatz')}_n{'-'.join(split_words(str(self.vars.get('n_qubits_list'))))}"
            f"_d{'-'.join(split_words(str(self.vars.get('train_depth_list'))))}"
            f"_{self.vars.get('encoder_axis')}_{self.vars.get('obs_kind')}"
        )
        self.vars.set("exact_dir", f"outputs/exact_twirled/{tag}")
        self.vars.set("batch_dir", f"outputs/twirled_covariance/{batch_tag}")
        self.vars.set("figure_dir", f"outputs/figures/{batch_tag}")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)

        top = ttk.Frame(root)
        top.pack(fill="x", pady=(0, 8))
        ttk.Button(top, text="Refresh output paths", command=self.refresh_paths).pack(side="left")
        ttk.Button(top, text="Dry-run selected pipeline", command=self.preview_pipeline).pack(side="left", padx=4)
        ttk.Button(top, text="Run selected pipeline", command=self.run_pipeline).pack(side="left", padx=4)
        ttk.Button(top, text="Stop process", command=self.stop_process).pack(side="left", padx=4)
        ttk.Button(top, text="Save preset", command=self.save_preset).pack(side="right", padx=4)
        ttk.Button(top, text="Load preset", command=self.load_preset).pack(side="right")

        body = ttk.PanedWindow(root, orient=tk.VERTICAL)
        body.pack(fill="both", expand=True)

        notebook_frame = ttk.Frame(body)
        self.nb = ttk.Notebook(notebook_frame)
        self.nb.pack(fill="both", expand=True)
        body.add(notebook_frame, weight=3)

        log_frame = ttk.Frame(body)
        self.log = tk.Text(log_frame, height=12, wrap="word", font=("Consolas", 9))
        yscroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=yscroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")
        body.add(log_frame, weight=1)

        self._build_project_tab()
        self._build_circuit_tab()
        self._build_exact_tab()
        self._build_batch_tab()
        self._build_plot_tab()

    def _tab(self, title: str) -> ttk.Frame:
        frame = ttk.Frame(self.nb, padding=10)
        self.nb.add(frame, text=title)
        return frame

    def _entry(
        self,
        parent: ttk.Frame,
        row: int,
        col: int,
        label: str,
        var_name: str,
        width: int = 24,
        browse: Optional[callable] = None,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", pady=2, padx=(0, 4))
        entry = ttk.Entry(parent, textvariable=self.vars._vars[var_name], width=width)
        entry.grid(row=row, column=col + 1, sticky="ew", pady=2)
        if browse:
            ttk.Button(parent, text="Browse", command=browse).grid(row=row, column=col + 2, sticky="w", padx=4)
        parent.columnconfigure(col + 1, weight=1)

    def _combo(self, parent: ttk.Frame, row: int, col: int, label: str, var_name: str, values: list[str]) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", pady=2, padx=(0, 4))
        box = ttk.Combobox(parent, textvariable=self.vars._vars[var_name], values=values, state="readonly", width=22)
        box.grid(row=row, column=col + 1, sticky="ew", pady=2)
        parent.columnconfigure(col + 1, weight=1)

    def _check(self, parent: ttk.Frame, row: int, col: int, label: str, var_name: str) -> None:
        ttk.Checkbutton(parent, text=label, variable=self.vars._vars[var_name]).grid(
            row=row, column=col, columnspan=2, sticky="w", pady=2
        )

    def _build_project_tab(self) -> None:
        frame = self._tab("Project")
        self._entry(frame, 0, 0, "project root", "project_root", width=72, browse=lambda: self.choose_path("project_root", True))
        self._entry(frame, 1, 0, "python executable", "python_exe", width=72, browse=lambda: self.choose_path("python_exe", False))
        self._entry(frame, 2, 0, "scripts subdirectory", "scripts_subdir", width=72)
        ttk.Label(
            frame,
            text="Leave scripts subdirectory blank when the scripts are in the project root. Use e.g. chm_covariance_pipeline if you keep them in a folder.",
            foreground="gray40",
            wraplength=820,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 12))

        self._entry(frame, 4, 0, "exact output dir", "exact_dir", width=72, browse=lambda: self.choose_path("exact_dir", True))
        self._entry(frame, 5, 0, "batch output dir", "batch_dir", width=72, browse=lambda: self.choose_path("batch_dir", True))
        self._entry(frame, 6, 0, "figure output dir", "figure_dir", width=72, browse=lambda: self.choose_path("figure_dir", True))

        self._entry(frame, 8, 0, "ansatze import path", "ansatze_path", width=72, browse=lambda: self.choose_path("ansatze_path", True))
        self._entry(frame, 9, 0, "twirl backend file", "twirl_module_path", width=72, browse=lambda: self.choose_path("twirl_module_path", False))
        ttk.Label(
            frame,
            text="The import path fields are optional. By default the scripts look next to themselves and in the project root.",
            foreground="gray40",
            wraplength=820,
        ).grid(row=10, column=0, columnspan=3, sticky="w", pady=(4, 0))

    def _build_circuit_tab(self) -> None:
        frame = self._tab("Circuit")
        self._combo(frame, 0, 0, "ansatz", "ansatz", ANSATZES)
        self._combo(frame, 1, 0, "encoder axis", "encoder_axis", AXES)
        self._combo(frame, 2, 0, "observable", "obs_kind", OBS_KINDS)
        self._entry(frame, 3, 0, "encoder scale", "encoder_scale")
        self._entry(frame, 4, 0, "single n_qubits", "n_qubits")
        self._entry(frame, 5, 0, "single n_layers", "n_layers")
        self._entry(frame, 6, 0, "single train_depth", "train_depth")
        self._entry(frame, 7, 0, "tag override", "tag_override", width=42)

        ttk.Separator(frame).grid(row=8, column=0, columnspan=3, sticky="ew", pady=10)
        self._entry(frame, 9, 0, "sweep n_qubits_list", "n_qubits_list", width=42)
        self._entry(frame, 10, 0, "sweep train_depth_list", "train_depth_list", width=42)
        self._entry(frame, 11, 0, "sweep n_layers_list", "n_layers_list", width=42)
        ttk.Label(
            frame,
            text="If n_layers_list is blank, the batch compute script uses the single n_layers value.",
            foreground="gray40",
            wraplength=560,
        ).grid(row=12, column=0, columnspan=3, sticky="w", pady=(4, 0))

    def _build_exact_tab(self) -> None:
        frame = self._tab("Exact single run")
        self._entry(frame, 0, 0, "out_dir", "exact_dir", width=64, browse=lambda: self.choose_path("exact_dir", True))
        self._combo(frame, 1, 0, "theta period", "theta_period", THETA_PERIODS)
        self._entry(frame, 2, 0, "CR quadrature points", "cr_quadrature_points")
        self._entry(frame, 3, 0, "combine tolerance", "combine_tol")
        self._entry(frame, 4, 0, "max states", "max_states")
        self._entry(frame, 5, 0, "MC validation samples", "n_theta_samples")
        self._entry(frame, 6, 0, "MC x grid", "n_x")
        self._entry(frame, 7, 0, "MC batch size", "batch_size")
        self._entry(frame, 8, 0, "seed", "seed")
        self._check(frame, 9, 0, "print progress", "progress")
        ttk.Button(frame, text="Run exact covariance", command=lambda: self.run_specs([self.spec_exact()])).grid(
            row=10, column=0, columnspan=2, sticky="ew", pady=10
        )

    def _build_batch_tab(self) -> None:
        frame = self._tab("Batch compute")
        self._entry(frame, 0, 0, "out_dir", "batch_dir", width=64, browse=lambda: self.choose_path("batch_dir", True))
        self._entry(frame, 1, 0, "max omega", "max_omega")
        self._entry(frame, 2, 0, "manual omega values", "omega_values", width=64)
        self._entry(frame, 3, 0, "n_x", "n_x")
        self._entry(frame, 4, 0, "x_min", "x_min")
        self._entry(frame, 5, 0, "x_max", "x_max")
        self._entry(frame, 6, 0, "MC theta samples", "n_theta_samples")
        self._entry(frame, 7, 0, "batch size", "batch_size")
        self._entry(frame, 8, 0, "seed", "seed")
        self._check(frame, 9, 0, "save raw MC samples for bootstrap plots", "save_mc_samples")
        self._check(frame, 10, 0, "print progress", "progress")
        ttk.Button(frame, text="Run batch compute", command=lambda: self.run_specs([self.spec_compute()])).grid(
            row=11, column=0, columnspan=2, sticky="ew", pady=10
        )

    def _build_plot_tab(self) -> None:
        frame = self._tab("Plots")
        self._entry(frame, 0, 0, "input directory", "batch_dir", width=64, browse=lambda: self.choose_path("batch_dir", True))
        self._entry(frame, 1, 0, "figure directory", "figure_dir", width=64, browse=lambda: self.choose_path("figure_dir", True))
        self._entry(frame, 2, 0, "file pattern", "plot_pattern")
        self._entry(frame, 3, 0, "omega_phys", "omega_phys")
        self._combo(frame, 4, 0, "omega side", "omega_side", OMEGA_SIDES)
        self._entry(frame, 5, 0, "support rel var", "support_rel_var")
        self._combo(frame, 6, 0, "support mode", "support_mode", SUPPORT_MODES)
        self._combo(frame, 7, 0, "mask diagonal", "mask_diag", MASK_DIAG_MODES)
        self._combo(frame, 8, 0, "column axis", "column_axis", COLUMN_AXES)
        self._entry(frame, 9, 0, "correlation clip quantile", "clip_q")
        self._entry(frame, 10, 0, "bootstrap resamples", "n_bootstrap")
        self._entry(frame, 11, 0, "bootstrap seed", "bootstrap_seed")
        self._check(frame, 12, 0, "white unit correlation diagonal", "unit_diag_white")

        ttk.Separator(frame).grid(row=13, column=0, columnspan=3, sticky="ew", pady=10)
        self._combo(frame, 14, 0, "variance mode", "variance_mode", VARIANCE_MODES)
        self._entry(frame, 15, 0, "variance y-limits", "variance_ylim", width=32)
        self._check(frame, 16, 0, "show variance text boxes", "variance_textbox")
        self._check(frame, 17, 0, "show variance legend", "variance_legend")

        buttons = ttk.Frame(frame)
        buttons.grid(row=18, column=0, columnspan=3, sticky="ew", pady=10)
        ttk.Button(buttons, text="Run correlation heatmaps", command=lambda: self.run_specs([self.spec_corr_plot()])).pack(side="left", padx=(0, 4))
        ttk.Button(buttons, text="Run variance profiles", command=lambda: self.run_specs([self.spec_variance_plot()])).pack(side="left", padx=4)

        ttk.Separator(frame).grid(row=19, column=0, columnspan=3, sticky="ew", pady=10)
        self._check(frame, 20, 0, "pipeline: batch compute", "run_compute")
        self._check(frame, 21, 0, "pipeline: correlation heatmaps", "run_corr_plots")
        self._check(frame, 22, 0, "pipeline: variance profiles", "run_variance_plots")

    # ------------------------------------------------------------------
    # Command construction
    # ------------------------------------------------------------------

    def common_exact_args(self) -> List[str]:
        args = [
            "--ansatz", str(self.vars.get("ansatz")),
            "--n_qubits", str(self.vars.get("n_qubits")),
            "--n_layers", str(self.vars.get("n_layers")),
            "--train_depth", str(self.vars.get("train_depth")),
            "--encoder_axis", str(self.vars.get("encoder_axis")),
            "--encoder_scale", str(self.vars.get("encoder_scale")),
            "--obs_kind", str(self.vars.get("obs_kind")),
            "--theta_period", str(self.vars.get("theta_period")),
            "--cr_quadrature_points", str(self.vars.get("cr_quadrature_points")),
            "--combine_tol", str(self.vars.get("combine_tol")),
            "--max_states", str(self.vars.get("max_states")),
            "--seed", str(self.vars.get("seed")),
        ]
        ansatze_path = str(self.vars.get("ansatze_path")).strip()
        if ansatze_path:
            args += ["--ansatze_path", ansatze_path]
        if self.vars.get("progress"):
            args.append("--progress")
        return args

    def spec_exact(self) -> CommandSpec:
        out_dir = self.path_from_var("exact_dir")
        cmd = self.py_cmd("exact_covariance_twirled.py") + ["--out_dir", str(out_dir)] + self.common_exact_args()
        samples = int(float(str(self.vars.get("n_theta_samples"))))
        if samples > 0:
            cmd += [
                "--validate_mc_samples", str(samples),
                "--n_x_mc", str(self.vars.get("n_x")),
                "--mc_batch_size", str(self.vars.get("batch_size")),
            ]
        return CommandSpec(
            name="exact two-copy twirled covariance",
            cmd=cmd,
            prerequisites=[("script", self.script_path("exact_covariance_twirled.py"))],
            expected_outputs=[
                ("npz", out_dir / "exact_twirled_covariance.npz"),
                ("summary", out_dir / "summary.json"),
            ],
        )

    def spec_compute(self) -> CommandSpec:
        out_dir = self.path_from_var("batch_dir")
        cmd = self.py_cmd("twirled_correlation_matrices_compute.py") + [
            "--out_dir", str(out_dir),
            "--seed", str(self.vars.get("seed")),
            "--ansatz", str(self.vars.get("ansatz")),
            "--n_qubits_list", *split_words(str(self.vars.get("n_qubits_list"))),
            "--train_depth_list", *split_words(str(self.vars.get("train_depth_list"))),
            "--n_layers", str(self.vars.get("n_layers")),
            "--encoder_axis", str(self.vars.get("encoder_axis")),
            "--encoder_scale", str(self.vars.get("encoder_scale")),
            "--obs_kind", str(self.vars.get("obs_kind")),
            "--theta_period", str(self.vars.get("theta_period")),
            "--n_x", str(self.vars.get("n_x")),
            "--x_min", str(self.vars.get("x_min")),
            "--x_max", str(self.vars.get("x_max")),
            "--n_theta_samples", str(self.vars.get("n_theta_samples")),
            "--batch_size", str(self.vars.get("batch_size")),
            "--combine_tol", str(self.vars.get("combine_tol")),
            "--max_states", str(self.vars.get("max_states")),
            "--cr_quadrature_points", str(self.vars.get("cr_quadrature_points")),
        ]
        n_layers_list = split_words(str(self.vars.get("n_layers_list")))
        if n_layers_list:
            cmd += ["--n_layers_list", *n_layers_list]
        omega_values = split_words(str(self.vars.get("omega_values")))
        if omega_values:
            cmd += ["--omega_values", *omega_values]
        else:
            cmd += ["--max_omega", str(self.vars.get("max_omega"))]
        if self.vars.get("save_mc_samples"):
            cmd.append("--save_mc_samples")
        else:
            cmd.append("--no_save_mc_samples")
        if self.vars.get("progress"):
            cmd.append("--progress")
        ansatze_path = str(self.vars.get("ansatze_path")).strip()
        if ansatze_path:
            cmd += ["--ansatze_path", ansatze_path]
        twirl_path = str(self.vars.get("twirl_module_path")).strip()
        if twirl_path:
            cmd += ["--twirl_module_path", twirl_path]
        return CommandSpec(
            name="batch CHM covariance/correlation compute",
            cmd=cmd,
            prerequisites=[
                ("script", self.script_path("twirled_correlation_matrices_compute.py")),
                ("twirl backend", self.script_path("exact_covariance_twirled.py")),
            ],
            expected_outputs=[("index", out_dir / "index.json")],
        )

    def spec_corr_plot(self) -> CommandSpec:
        in_dir = self.path_from_var("batch_dir")
        out_dir = self.path_from_var("figure_dir") / "correlation_heatmaps"
        cmd = self.py_cmd("twirled_correlation_matrices_plot.py") + [
            "--indir", str(in_dir),
            "--outdir", str(out_dir),
            "--pattern", str(self.vars.get("plot_pattern")),
            "--clip_q", str(self.vars.get("clip_q")),
            "--omega_side", str(self.vars.get("omega_side")),
            "--support_rel_var", str(self.vars.get("support_rel_var")),
            "--support_mode", str(self.vars.get("support_mode")),
            "--mask_diag", str(self.vars.get("mask_diag")),
            "--column_axis", str(self.vars.get("column_axis")),
            "--n_bootstrap", str(self.vars.get("n_bootstrap")),
            "--bootstrap_seed", str(self.vars.get("bootstrap_seed")),
        ]
        omega_phys = str(self.vars.get("omega_phys")).strip()
        if omega_phys:
            cmd += ["--omega_phys", omega_phys]
        cmd.append("--unit_diag_white" if self.vars.get("unit_diag_white") else "--no_unit_diag_white")
        return CommandSpec(
            name="correlation heatmap plots",
            cmd=cmd,
            prerequisites=[("script", self.script_path("twirled_correlation_matrices_plot.py")), ("input directory", in_dir)],
            expected_outputs=[("output directory", out_dir)],
        )

    def spec_variance_plot(self) -> CommandSpec:
        in_dir = self.path_from_var("batch_dir")
        out_dir = self.path_from_var("figure_dir") / "variance_profiles"
        cmd = self.py_cmd("twirled_variance_profile_plot.py") + [
            "--indir", str(in_dir),
            "--outdir", str(out_dir),
            "--pattern", str(self.vars.get("plot_pattern")),
            "--mode", str(self.vars.get("variance_mode")),
            "--omega_side", str(self.vars.get("omega_side")),
        ]
        omega_phys = str(self.vars.get("omega_phys")).strip()
        if omega_phys:
            cmd += ["--omega_phys", omega_phys]
        ylim = split_words(str(self.vars.get("variance_ylim")))
        if ylim:
            if len(ylim) != 2:
                raise ValueError("variance y-limits must contain exactly two numbers, e.g. '-16 0'.")
            cmd += ["--ylim", *ylim]
        if not self.vars.get("variance_textbox"):
            cmd.append("--no_textbox")
        if not self.vars.get("variance_legend"):
            cmd.append("--no_legend")
        return CommandSpec(
            name="variance profile plots",
            cmd=cmd,
            prerequisites=[("script", self.script_path("twirled_variance_profile_plot.py")), ("input directory", in_dir)],
            expected_outputs=[("output directory", out_dir)],
        )

    def selected_pipeline_specs(self) -> List[CommandSpec]:
        specs: List[CommandSpec] = []
        if self.vars.get("run_compute"):
            specs.append(self.spec_compute())
        if self.vars.get("run_corr_plots"):
            specs.append(self.spec_corr_plot())
        if self.vars.get("run_variance_plots"):
            specs.append(self.spec_variance_plot())
        if not specs:
            raise ValueError("No pipeline stages are selected.")
        return specs

    # ------------------------------------------------------------------
    # Running and logging
    # ------------------------------------------------------------------

    def append_log(self, text: str) -> None:
        self.log.insert("end", normalise_subprocess_text(text))
        self.log.see("end")

    def log_line(self, text: str = "") -> None:
        self.append_log(text + "\n")

    def preview_pipeline(self) -> None:
        try:
            specs = self.selected_pipeline_specs()
        except Exception as exc:
            messagebox.showerror("Invalid pipeline", str(exc))
            return
        self.log_line(f"[{timestamp()}] Dry run")
        for spec in specs:
            self.log_line(f"\n# {spec.name}")
            self.log_line(shell_join(spec.cmd))

    def run_pipeline(self) -> None:
        try:
            specs = self.selected_pipeline_specs()
        except Exception as exc:
            messagebox.showerror("Invalid pipeline", str(exc))
            return
        self.run_specs(specs)

    def check_prerequisites(self, spec: CommandSpec) -> bool:
        missing = [(label, path) for label, path in spec.prerequisites if not path.exists()]
        if not missing:
            return True
        message = "Missing required files/directories:\n" + "\n".join(f"- {label}: {path}" for label, path in missing)
        messagebox.showerror(f"Cannot run {spec.name}", message)
        self.log_line(message)
        return False

    def run_specs(self, specs: List[CommandSpec]) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("Process running", "A command is already running.")
            return
        for spec in specs:
            if not self.check_prerequisites(spec):
                return
        self.stop_requested = False
        self.worker = threading.Thread(target=self._run_specs_worker, args=(specs,), daemon=True)
        self.worker.start()

    def _run_specs_worker(self, specs: List[CommandSpec]) -> None:
        for spec in specs:
            if self.stop_requested:
                self.log_queue.put(f"[{timestamp()}] Stopped before {spec.name}.\n")
                return
            self.log_queue.put(f"\n[{timestamp()}] Running {spec.name}\n")
            self.log_queue.put(shell_join(spec.cmd) + "\n")
            try:
                self.process = subprocess.Popen(
                    spec.cmd,
                    cwd=str(self.project_root()),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
                assert self.process.stdout is not None
                for line in self.process.stdout:
                    self.log_queue.put(line)
                rc = self.process.wait()
                self.process = None
                if rc != 0:
                    self.log_queue.put(f"[{timestamp()}] FAILED with exit code {rc}: {spec.name}\n")
                    return
                self.log_queue.put(f"[{timestamp()}] Completed {spec.name}\n")
                for label, path in spec.expected_outputs:
                    status = "OK" if path.exists() else "not found yet"
                    self.log_queue.put(f"    {label}: {path} [{status}]\n")
            except Exception as exc:
                self.process = None
                self.log_queue.put(f"[{timestamp()}] ERROR while running {spec.name}: {exc}\n")
                return
        self.log_queue.put(f"[{timestamp()}] Pipeline complete.\n")

    def _poll_log_queue(self) -> None:
        try:
            while True:
                self.append_log(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        self.after(100, self._poll_log_queue)

    def stop_process(self) -> None:
        self.stop_requested = True
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.log_line(f"[{timestamp()}] Termination requested.")

    # ------------------------------------------------------------------
    # Presets and file choosers
    # ------------------------------------------------------------------

    def choose_path(self, var_name: str, directory: bool) -> None:
        if directory:
            path = filedialog.askdirectory(initialdir=str(self.project_root()))
        else:
            path = filedialog.askopenfilename(initialdir=str(self.project_root()))
        if path:
            self.vars.set(var_name, path)

    def save_preset(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
            initialdir=str(self.project_root()),
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.vars.to_dict(), handle, indent=2, sort_keys=True)
        self.log_line(f"Saved preset: {path}")

    def load_preset(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
            initialdir=str(self.project_root()),
        )
        if not path:
            return
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        self.vars.load_dict(data)
        self.log_line(f"Loaded preset: {path}")


if __name__ == "__main__":
    CHMCovarianceGUI().mainloop()
