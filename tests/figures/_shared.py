"""Shared utilities for Section 8 figure/table generation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("results") / "section8"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PROFILE_LABELS = {
    "speed_oriented": "Speed",
    "safety_first": "Safety",
    "energy_conscious": "Energy",
    "comfort_focused": "Comfort",
    "presentation_focused": "Presentation",
}

PROFILE_COLORS = {
    "speed_oriented": "#1f77b4",
    "safety_first": "#d62728",
    "energy_conscious": "#2ca02c",
    "comfort_focused": "#9467bd",
    "presentation_focused": "#ff7f0e",
}

DIMENSIONS = ["time", "safety", "battery", "proximity", "approach"]
DIM_LABELS = ["Time", "Safety", "Battery", "Proximity", "Approach"]

DEFAULT_THRESHOLD = 0.10
CONVERGENCE_THRESHOLDS = {
    "presentation_focused": 0.15,
}


def load_results(subdir: str) -> List[Dict]:
    path = RESULTS_DIR / subdir
    if not path.exists():
        print(f"  skip: missing {path}")
        return []
    runs = []
    for file in sorted(path.glob("*_seed*.json")):
        with open(file, "r", encoding="utf-8-sig") as f:
            runs.append(json.load(f))
    return runs


def group_by_profile(runs: Iterable[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for run in runs:
        grouped[run.get("profile", "")].append(run)
    return dict(grouped)


def episode_series(run: Dict, key: str, default=np.nan) -> np.ndarray:
    return np.array([ep.get(key, default) for ep in run.get("episodes", [])], dtype=float)


def nested_episode_series(run: Dict, parent: str, key: str, default=np.nan) -> np.ndarray:
    values = []
    for ep in run.get("episodes", []):
        obj = ep.get(parent, {})
        values.append(obj.get(key, default) if isinstance(obj, dict) else default)
    return np.array(values, dtype=float)


def mean_std(series: List[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not series:
        return np.array([]), np.array([])
    n = max(len(s) for s in series)
    mat = np.full((len(series), n), np.nan)
    for i, s in enumerate(series):
        mat[i, : len(s)] = s
    return np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)


def save_fig(fig, name: str):
    path = FIGURES_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def profile_order(runs: Iterable[Dict]) -> List[str]:
    present = {r.get("profile") for r in runs}
    return [p for p in PROFILE_COLORS if p in present]


def ensure_runs(subdir: str) -> List[Dict]:
    runs = load_results(subdir)
    if not runs:
        print(f"  no data for {subdir}; skipping")
    return runs
