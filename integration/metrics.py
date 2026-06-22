"""
integration/metrics.py — Per-episode and cross-episode evaluation containers.

EpisodeMetrics      — all scalar fields recorded for one episode
LearningCurveTracker — aggregates metrics across episodes; prints/exports summaries
"""

from __future__ import annotations

import csv
from typing import Dict, List, Optional

import numpy as np


class EpisodeMetrics:
    """Comprehensive per-episode evaluation metrics."""

    def __init__(self):
        # Preference learning (outer loop)
        self.preference_distance: float = 0.0
        self.preference_weight_change: float = 0.0
        self.preference_converged: bool = False
        self.dominant_correct: bool = False

        # Translator φ/Q/R updates are disabled; fields remain for compatibility.
        self.phi_gradient_norm: float = 0.0
        self.phi_param_change: float = 0.0
        self.num_sensitivity_samples: int = 0
        self.avg_mpc_cost: float = 0.0

        # MPC solve quality
        self.mpc_total_solves: int = 0
        self.mpc_success_rate: float = 0.0
        self.mpc_avg_solve_time_ms: float = 0.0
        self.mpc_sensitivity_computes: int = 0
        self.mpc_avg_sens_time_ms: float = 0.0

        # Navigation quality
        self.num_legs: int = 0
        self.path_efficiency: float = 0.0
        self.total_distance: float = 0.0
        self.total_time: float = 0.0
        self.min_obstacle_clearance: float = float("inf")
        self.nav_stack_used: bool = False

        # Task completion
        self.delivery_position_error: float = 0.0
        self.delivery_orientation_error: float = 0.0
        self.approach_quality: float = 0.0
        self.battery_used_pct: float = 0.0
        self.battery_net_delta_pct: float = 0.0
        self.battery_remaining_pct: float = 0.0
        self.plan_length: int = 0

        # Feature scores
        self.features: Dict[str, float] = {}
        self.ratings: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        d = {}
        for key, val in self.__dict__.items():
            if isinstance(val, np.ndarray):
                d[key] = val.tolist()
            elif isinstance(val, (float, int, bool, str)):
                d[key] = val
            elif isinstance(val, dict):
                d[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in val.items()
                }
            else:
                d[key] = str(val)
        return d


class LearningCurveTracker:
    """Tracks learning curves across episodes for analysis."""

    def __init__(self):
        self.episode_metrics: List[EpisodeMetrics] = []
        self.preference_distances: List[float] = []
        self.phi_gradient_norms: List[float] = []
        self.mpc_costs: List[float] = []
        self.mpc_success_rates: List[float] = []
        self.path_efficiencies: List[float] = []
        self.delivery_errors: List[float] = []
        self.battery_usage: List[float] = []

    def record(self, metrics: EpisodeMetrics):
        self.episode_metrics.append(metrics)
        self.preference_distances.append(metrics.preference_distance)
        self.phi_gradient_norms.append(metrics.phi_gradient_norm)
        self.mpc_costs.append(metrics.avg_mpc_cost)
        self.mpc_success_rates.append(metrics.mpc_success_rate)
        self.path_efficiencies.append(metrics.path_efficiency)
        self.delivery_errors.append(metrics.delivery_position_error)
        self.battery_usage.append(metrics.battery_used_pct)

    def print_summary(self, last_n: int = 5):
        n = len(self.episode_metrics)
        if n == 0:
            print("No episodes recorded yet.")
            return

        print(f"\n{'='*80}")
        print(f"LEARNING CURVE SUMMARY ({n} episodes)")
        print(f"{'='*80}")

        if n >= 2:
            first = self.preference_distances[0]
            last = self.preference_distances[-1]
            improvement = ((first - last) / first * 100) if first > 0 else 0
            print(f"\nPreference Learning (outer loop):")
            print(f"  Initial distance to w*: {first:.4f}")
            print(f"  Current distance to w*: {last:.4f}")
            print(f"  Improvement: {improvement:.1f}%")
            converged_count = sum(
                1 for m in self.episode_metrics if m.preference_converged
            )
            print(f"  Episodes converged: {converged_count}/{n}")

        if self.phi_gradient_norms:
            recent_grads = self.phi_gradient_norms[-last_n:]
            print(f"\nTranslator φ/Q/R Updates:")
            print("  Status: disabled")
            print(f"  Recent avg gradient norm: {np.mean(recent_grads):.4f}")
            print(
                f"  Total sensitivity samples: "
                f"{sum(m.num_sensitivity_samples for m in self.episode_metrics)}"
            )

        if self.mpc_costs:
            recent_costs = self.mpc_costs[-last_n:]
            recent_rates = self.mpc_success_rates[-last_n:]
            print(f"\nMPC Performance:")
            print(f"  Recent avg cost: {np.mean(recent_costs):.1f}")
            print(f"  Recent avg success rate: {np.mean(recent_rates):.1f}%")

        if self.path_efficiencies:
            recent_eff = [e for e in self.path_efficiencies[-last_n:] if e > 0]
            if recent_eff:
                print(f"\nNavigation Quality:")
                print(f"  Recent avg path efficiency: {np.mean(recent_eff):.2f}")
                print(
                    f"  Recent avg delivery error: "
                    f"{np.mean(self.delivery_errors[-last_n:]):.3f}m"
                )

        print(f"{'='*80}\n")

    def export_csv(self, filepath: str):
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "pref_distance", "pref_converged", "dominant_correct",
                "phi_grad_norm", "phi_change", "sensitivity_samples",
                "mpc_cost", "mpc_success_rate", "mpc_solve_time_ms",
                "path_efficiency", "total_distance", "total_time",
                "delivery_error", "approach_quality", "battery_used_pct",
                "plan_length",
            ])
            for i, m in enumerate(self.episode_metrics, 1):
                writer.writerow([
                    i, m.preference_distance, m.preference_converged,
                    m.dominant_correct, m.phi_gradient_norm, m.phi_param_change,
                    m.num_sensitivity_samples, m.avg_mpc_cost, m.mpc_success_rate,
                    m.mpc_avg_solve_time_ms, m.path_efficiency, m.total_distance,
                    m.total_time, m.delivery_position_error, m.approach_quality,
                    m.battery_used_pct, m.plan_length,
                ])
        print(f"Exported learning curves to: {filepath}")
