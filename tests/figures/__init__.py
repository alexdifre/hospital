"""
figures/ — Section 8 figure subpackage.

Each module owns one group of figures:
  core        — B1–B9  (full-system results)
  comparisons — BL, AB, AC  (baselines / ablations)
  robustness  — NR, NR2, IS  (noise & init sensitivity)
  tables      — T1, T2, T3  (LaTeX tables)

FIGURE_MAP maps short keys → (description, callable) and is imported by the
top-level dispatcher (tests/generate_section8_figures.py).
"""

from .core import (
    fig_b1_convergence,
    fig_b2_weight_evolution,
    fig_b3_final_weights,
    fig_b4_feature_space,
    fig_b5_plan_diversity,
    fig_b6_mse_loss,
    fig_b7_translator_params,
    fig_b8_trajectories,
    fig_b9_battery_efficiency,
)
from .comparisons import (
    fig_bl_baselines,
    fig_ab_ablations,
    fig_ac_ablation_curves,
)
from .robustness import (
    fig_nr_noise_robustness,
    fig_nr2_noise_conv_rates,
    fig_is_init_sensitivity,
)
from .tables import table_t1_master, table_t2_baselines, table_t3_ablations, table_t4_robustness

FIGURE_MAP = {
    # Core (from --condition full)
    "B1":  ("Convergence Curves",          fig_b1_convergence),
    "B2":  ("Weight Evolution",             fig_b2_weight_evolution),
    "B3":  ("Final vs True Weights",        fig_b3_final_weights),
    "B4":  ("Feature Space",                fig_b4_feature_space),
    "B5":  ("Plan Diversity",               fig_b5_plan_diversity),
    "B6":  ("MSE / Loss",                   fig_b6_mse_loss),
    "B7":  ("Translator Params",            fig_b7_translator_params),
    "B8":  ("MPC Trajectories",             fig_b8_trajectories),
    "B9":  ("Battery & Efficiency",         fig_b9_battery_efficiency),
    # Comparisons
    "BL":  ("Baseline Comparison",          fig_bl_baselines),
    "AB":  ("Ablation Comparison",          fig_ab_ablations),
    "AC":  ("Ablation Curves + Bar Chart",  fig_ac_ablation_curves),
    # Robustness
    "NR":  ("Noise Robustness (d_best line)", fig_nr_noise_robustness),
    "NR2": ("Noise Conv. Rate Bar Chart",   fig_nr2_noise_conv_rates),
    "IS":  ("Init Sensitivity",             fig_is_init_sensitivity),
    # Tables
    "T1":  ("Master Results Table",         table_t1_master),
    "T2":  ("Baseline Comparison Table",    table_t2_baselines),
    "T3":  ("Ablation Study Table",         table_t3_ablations),
    "T4":  ("Robustness Table",             table_t4_robustness),
}

__all__ = ["FIGURE_MAP"]
