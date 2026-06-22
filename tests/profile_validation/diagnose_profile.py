#!/usr/bin/env python3
"""
Quick Profile Diagnostic — 10 episodes, 1 seed, verbose.

Usage:
    python diagnose_profile.py presentation_focused
    python diagnose_profile.py safety_first
    python diagnose_profile.py speed_oriented
"""

import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from integration.integrator2 import FullMedicationDeliverySystem

DIM_NAMES = ["time", "safety", "battery", "proximity", "approach"]

# True weights for reference
TRUE_WEIGHTS = {
    "speed_oriented": [0.50, 0.12, 0.14, 0.14, 0.10],
    "safety_first": [0.10, 0.50, 0.15, 0.15, 0.10],
    "energy_conscious": [0.15, 0.15, 0.45, 0.15, 0.10],
    "comfort_focused": [0.15, 0.15, 0.10, 0.40, 0.20],
    "presentation_focused": [0.05, 0.10, 0.05, 0.20, 0.60],
}


def diagnose(profile_name: str, num_episodes: int = 20, seed: int = 0):
    np.random.seed(seed)

    true_w = TRUE_WEIGHTS.get(profile_name)
    if true_w is None:
        print(f"Unknown profile: {profile_name}")
        print(f"Available: {list(TRUE_WEIGHTS.keys())}")
        return

    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC: {profile_name}  (seed={seed}, {num_episodes} episodes)")
    print(f"  True weights: {true_w}")
    print(f"{'='*70}\n")

    system = FullMedicationDeliverySystem(
        patient_profile_name=profile_name,
        preference_learning_rate=0.12,
        render=False,
        verbose=False,
        save_summaries=False,
        use_fuzzy=True,
        explore_sigma=0.15,
        explore_decay=0.2,
        rating_noise=0.30,
    )

    distances = []
    all_features = {"medication": [], "meal": []}

    for ep in range(num_episodes):
        task_type = "medication" if ep % 2 == 0 else "meal"
        start_loc = "home" if task_type == "medication" else "pantry"

        try:
            w_before = system.preference_learner.get_current_weights().copy()
        except Exception:
            w_before = np.array([0.2] * 5)

        try:
            result = system.run_episode(start_location=start_loc, task_type=task_type)
        except Exception as e:
            print(f"  Ep {ep} CRASHED: {e}")
            distances.append(1.0)
            continue

        try:
            w_after = system.preference_learner.get_current_weights().copy()
        except Exception:
            w_after = w_before

        dist = result.get("distance_to_true", 1.0)
        features = result.get("features", {})
        plan = result.get("plan_structure", {})

        distances.append(dist)
        if isinstance(features, dict) and features:
            all_features[task_type].append(features)

        # ── Verbose episode summary ──────────────────────────
        converged = "✓ CONV" if dist <= 0.10 else ""
        print(f"  Ep {ep:2d} [{task_type[:4]}]  d={dist:.4f}  {converged}")
        print(f"    weights: [{', '.join(f'{x:.3f}' for x in w_after)}]")

        if isinstance(features, dict):
            feat_str = "  ".join(f"{k}={v:.3f}" for k, v in features.items())
            print(f"    features: {feat_str}")

        plan_sig = plan.get("plan_signature", plan.get("action_sequence", ""))
        meal_type = plan.get("meal_type", "")
        if meal_type:
            print(f"    meal_type: {meal_type}")
        if plan_sig:
            print(f"    plan: {plan_sig}")
        print()

    # ── Final summary ────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC SUMMARY: {profile_name}")
    print(f"{'='*70}")
    print(f"  True:    [{', '.join(f'{x:.3f}' for x in true_w)}]")
    try:
        final_w = system.preference_learner.get_current_weights()
        print(f"  Learned: [{', '.join(f'{x:.3f}' for x in final_w)}]")
    except Exception:
        pass
    print(f"  Final d: {distances[-1]:.4f}")
    print(f"  Best d:  {min(distances):.4f}")
    print(f"  Trajectory: {[f'{d:.3f}' for d in distances]}")

    # Feature analysis
    for task_type in ["medication", "meal"]:
        feats = all_features[task_type]
        if not feats:
            continue
        print(f"\n  {task_type.upper()} feature stats ({len(feats)} episodes):")
        for dim in DIM_NAMES:
            vals = [f.get(dim, 0) for f in feats]
            if vals:
                print(
                    f"    {dim:12s}: mean={np.mean(vals):.3f}  std={np.std(vals):.3f}  range=[{min(vals):.3f}, {max(vals):.3f}]"
                )

    # Key diagnostic: which dimensions have enough variance?
    print(f"\n  SIGNAL STRENGTH (std across all episodes):")
    all_feats = all_features["medication"] + all_features["meal"]
    if all_feats:
        for dim in DIM_NAMES:
            vals = [f.get(dim, 0) for f in all_feats]
            std = np.std(vals)
            marker = "⚠️  LOW" if std < 0.05 else "✓"
            print(f"    {dim:12s}: std={std:.3f}  {marker}")

    system.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_profile.py <profile_name>")
        print(f"Profiles: {list(TRUE_WEIGHTS.keys())}")
        sys.exit(1)

    profile = sys.argv[1]
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    diagnose(profile, num_episodes=episodes)
