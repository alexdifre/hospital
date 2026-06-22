"""
integration/reporting.py — ReportingMixin for FullMedicationDeliverySystem.

Handles all output and persistence:
    _print_episode_summary   — per-episode console output
    _print_final_summary     — end-of-run console summary
    _save_json               — numpy-safe JSON serialiser
    _save_final_summary      — FINAL_SUMMARY.txt writer
    visualize_learning       — delegates to preference learner
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


class ReportingMixin:
    """Output and persistence methods for FullMedicationDeliverySystem."""

    def _print_episode_summary(self, data: Dict):
        lines = []
        lines.append("=" * 80)
        task_label = data.get("task_type", "medication").upper()
        lines.append(f"EPISODE {data['episode']} SUMMARY [{task_label}]")
        lines.append("=" * 80)

        plan = data["plan_structure"]
        if plan.get("task_type") == "meal":
            lines.append(
                f"\nPLAN: meal_type={plan.get('meal_type', '?')}, "
                f"approach={plan.get('approach_choice', '?')}, "
                f"steps={plan.get('plan_length', '?')}"
            )
        else:
            lines.append(
                f"\nPLAN: pharmacy={plan.get('pharmacy_choice', '?')}, "
                f"supply={plan.get('supply_choice', '?')}, "
                f"approach={plan.get('approach_choice', '?')}, "
                f"recharge={'yes' if plan.get('recharge_added') else 'no'}"
            )

        explore = data.get("exploration", {})
        if explore.get("explored"):
            lines.append(
                f"  [Explored] σ={explore['sigma']:.4f}, "
                f"planning_w=[{', '.join(f'{v:.3f}' for v in explore['planning_weights'])}]"
            )

        w = data["weights_after"]
        dim_names = ["time", "safety", "battery", "proximity", "approach"]
        dom_idx = int(np.argmax(w))
        true_w = self.preference_learner.true_profile.weights
        true_dom = int(np.argmax(true_w))

        lines.append(f"\nPREFERENCE LEARNING (outer loop):")
        lines.append(f"  w = [{', '.join(f'{v:.3f}' for v in w)}]")
        lines.append(
            f"  Dominant: {dim_names[dom_idx]} ({w[dom_idx]:.3f})"
            f"  True: {dim_names[true_dom]} ({true_w[true_dom]:.3f})"
            f"  {'✓' if dom_idx == true_dom else '← MISMATCH'}"
        )
        lines.append(f"  Distance to w*: {data['distance_to_true']:.4f}")
        lines.append(f"  Converged: {'✓' if data['converged'] else 'no'}")

        tl = data.get("translator_learning", {})
        lines.append(f"\nTRANSLATOR φ/Q/R UPDATES:")
        lines.append(
            f"  Status: disabled"
        )
        lines.append(f"  ||∂J/∂φ||: {tl.get('phi_gradient_norm', 0):.4f}")
        lines.append(f"  ||Δφ||: {tl.get('phi_param_change', 0):.4f}")
        lines.append(f"  Sensitivity samples: {tl.get('sensitivity_samples', 0)}")
        lines.append(f"  Avg MPC cost: {tl.get('avg_mpc_cost', 0):.1f}")

        mpc = data.get("mpc_stats", {})
        lines.append(f"\nEXECUTION:")
        lines.append(
            f"  Distance: {data['total_distance']:.1f}m, "
            f"Time: {data['total_time']:.1f}s, "
            f"Efficiency: {data.get('path_efficiency', 0):.2f}"
        )
        lines.append(
            f"  MPC: {mpc.get('total_steps', 0)} solves, "
            f"avg {mpc.get('avg_solve_time_ms', 0):.1f}ms, "
            f"{mpc.get('sensitivity_computes', 0)} sens computes"
        )
        net_delta = data.get("battery_net_delta_pct")
        suffix = (
            f"(energy used: {data.get('battery_used_pct', 0):.1f}%, "
            f"net delta: {net_delta:+.1f}%)"
            if net_delta is not None
            else f"(energy used: {data.get('battery_used_pct', 0):.1f}%)"
        )
        lines.append(
            f"  Battery: {data.get('battery_start', 100):.0f}% → "
            f"{data.get('battery_remaining', 0):.0f}% {suffix}"
        )

        if data.get("final_position_error") is not None:
            lines.append(
                f"  Delivery error: {data['final_position_error']:.3f}m, "
                f"approach quality: {data.get('approach_quality', 0):.2f}"
            )

        traj = data.get("trajectory_xy", [])
        if traj:
            lines.append(f"  Trajectory: {len(traj)} xy points recorded")

        lines.append("=" * 80)
        print("\n".join(lines))

    def _print_final_summary(self, results: List[Dict]):
        print(f"\n{'='*80}")
        print("LEARNING COMPLETE")
        print(f"{'='*80}\n")

        self.preference_learner.print_learning_summary()
        self.learning_tracker.print_summary()

        if self.plan_history:
            unique = set()
            for p in self.plan_history:
                if p.get("task_type") == "meal":
                    unique.add(self._get_meal_plan_key(p))
                else:
                    unique.add(self._get_med_plan_key(p))

            print(f"\n{'='*80}")
            print("PLAN DIVERSITY")
            print(f"{'='*80}")
            print(f"Total episodes: {len(self.plan_history)}")
            print(f"Unique plans: {len(unique)}")

            med_plans  = [p for p in self.plan_history if p.get("task_type") != "meal"]
            meal_plans = [p for p in self.plan_history if p.get("task_type") == "meal"]
            if med_plans:
                med_unique = set(self._get_med_plan_key(p) for p in med_plans)
                print(
                    f"  Medication: {len(med_unique)} unique from {len(med_plans)} episodes"
                )
            if meal_plans:
                meal_unique = set(self._get_meal_plan_key(p) for p in meal_plans)
                print(
                    f"  Meal prep:  {len(meal_unique)} unique from {len(meal_plans)} episodes"
                )
                meal_types = set(p.get("meal_type") for p in meal_plans)
                print(f"  Meal types seen: {meal_types}")
            print(f"{'='*80}\n")

        if hasattr(self.translator, "print_learning_summary"):
            self.translator.print_learning_summary()

        if self.save_summaries and self.summary_dir:
            self._save_final_summary(results)
            csv_path = str(self.summary_dir / "learning_curves.csv")
            self.learning_tracker.export_csv(csv_path)

    def _save_json(self, data: Dict, filepath: Path):
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        with open(filepath, "w") as f:
            json.dump(convert(data), f, indent=2)

    def _save_final_summary(self, results: List[Dict]):
        summary_file = self.summary_dir / "FINAL_SUMMARY.txt"
        successful = [r for r in results if r.get("success", False)]

        with open(summary_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("FINAL LEARNING SUMMARY (v4 - terminal-target learning)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total Episodes: {len(results)}\n")
            f.write(f"Successful: {len(successful)}/{len(results)}\n\n")

            if successful:
                f.write("PREFERENCE LEARNING (outer loop):\n")
                f.write(f"  Initial weights: {successful[0]['weights_before']}\n")
                f.write(f"  Final weights:   {successful[-1]['weights_after']}\n")
                f.write(
                    f"  True weights:    "
                    f"{self.preference_learner.true_profile.weights}\n"
                )
                f.write(
                    f"  Final distance:  {successful[-1]['distance_to_true']:.4f}\n"
                )
                f.write(
                    f"  Converged: {'YES' if successful[-1]['converged'] else 'NO'}\n"
                )
                f.write(
                    f"  Dominant correct: "
                    f"{'YES' if successful[-1].get('dominant_correct') else 'NO'}\n\n"
                )

                f.write("TRANSLATOR φ/Q/R UPDATES:\n")
                f.write("  Status: disabled\n")
                f.write("TERMINAL TARGET UPDATES:\n")
                total_target_updates = sum(
                    len(r.get("terminal_target_updates", []))
                    for r in successful
                )
                f.write(f"  Total updates: {total_target_updates}\n")
                total_sens = sum(
                    r.get("translator_learning", {}).get("sensitivity_samples", 0)
                    for r in successful
                )
                f.write(f"  Total sensitivity samples: {total_sens}\n")
                costs = [
                    r.get("translator_learning", {}).get("avg_mpc_cost", 0)
                    for r in successful
                ]
                if len(costs) >= 2 and abs(costs[0]) > 0.01:
                    cost_change = ((costs[-1] - costs[0]) / abs(costs[0])) * 100
                    f.write(
                        f"  MPC cost: {costs[0]:.1f} → {costs[-1]:.1f} "
                        f"({cost_change:+.1f}%)\n"
                    )
                f.write("\n")

                f.write("EXECUTION:\n")
                avg_time = float(np.mean([r["total_time"] for r in successful]))
                avg_dist = float(np.mean([r["total_distance"] for r in successful]))
                avg_eff  = float(
                    np.mean([r.get("path_efficiency", 0) for r in successful])
                )
                f.write(f"  Avg time: {avg_time:.1f}s\n")
                f.write(f"  Avg distance: {avg_dist:.1f}m\n")
                f.write(f"  Avg path efficiency: {avg_eff:.2f}\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"Saved final summary to: {summary_file}")

    def visualize_learning(self, save_path=None):
        return self.preference_learner.visualize_learning(save_path)
