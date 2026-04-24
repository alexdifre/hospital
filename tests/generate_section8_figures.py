#!/usr/bin/env python3
"""
Section 8 Figure Generator — MLC Stack Paper (Unified)
======================================================

Reads JSON results from run_section8_experiments.py and generates
ALL figures and tables for Section 8.

Core figures (from --condition full data):
  B1  — Convergence curves (distance vs episode, all profiles)
  B2  — Weight evolution (per-dimension, 3 representative profiles)
  B3  — Final learned vs true weights (grouped bar + diamond markers)
  B4  — Feature centroids (medication vs meal, per profile)
  B5  — Plan diversity (meal type selection proportions)
  B6  — MSE / loss over episodes          [needs learner_mse in JSON]
  B7  — Translator φ parameter evolution  [needs translator_params]
  B8  — MPC trajectory examples (xy)      [needs trajectory_xy]
  B9  — Battery & efficiency comparison   [needs battery_used_pct]

Comparison figures (from --condition baselines/ablations/robustness):
  BL  — Baseline comparison (bar chart)
  AB  — Ablation comparison (bar chart)
  AC  — Ablation convergence curves + bar chart
  NR  — Noise robustness curves
  NR2 — Noise convergence rate bar chart
  IS  — Initialisation sensitivity (box plots)

Table:
  T1  — Master results table (LaTeX)

Usage:
    python generate_section8_figures.py              # all figures
    python generate_section8_figures.py B1 B2 B3 T1  # specific figures
"""

import sys
from pathlib import Path

# Allow running directly from tests/ without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent))

from figures import FIGURE_MAP  # noqa: E402

if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(FIGURE_MAP.keys())
    print(f"\nGenerating {len(targets)} Section 8 figures...\n")
    for key in targets:
        if key in FIGURE_MAP:
            desc, func = FIGURE_MAP[key]
            print(f"[{key}] {desc}")
            func()
        else:
            print(f"[{key}] Unknown — skipping")
    from figures._shared import FIGURES_DIR
    print(f"\nDone. Figures → {FIGURES_DIR}/")
