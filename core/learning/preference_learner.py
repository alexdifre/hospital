#!/usr/bin/env python3
"""
Preference Learning Engine for Patient-Centered Robot Behavior
==============================================================

Implements preference learning via projected gradient descent on the probability simplex.

Core idea:
- The patient has a hidden preference profile w* (weights on 5 dimensions).
- The robot maintains an estimate w_hat on the simplex (w >= 0, sum(w)=1).
- After each episode, we extract normalized features f in [0,1] for each dimension:
    f = [time, safety, battery, proximity, approach]
  where 0 is best and 1 is worst.
- The patient provides a 5D rating vector r in [1,5] (one rating per dimension).
- The robot updates w_hat via projected gradient descent.

Important: Learning is MULTI-DIMENSIONAL.
We do NOT collapse ratings to a single scalar. Each dimension has its own signal.

Instrumentation (Section 8 figures):
- loss_history / last_mse       → B6 (MSE over episodes)
- per_dim_loss_history           → B6 (per-dimension decomposition)
- gradient_norm_history          → supplementary learning diagnostics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for macOS/headless
import matplotlib.pyplot as plt
import numpy as np


FEATURE_KEYS = ("time", "safety", "battery", "proximity", "approach")
CONCERNS = ["Speed", "Safety", "Battery", "Proximity", "Approach"]


@dataclass
class PatientProfile:
    """
    Patient preference profile (hidden ground truth).

    weights: np.ndarray of shape (5,)
        [w_time, w_safety, w_battery, w_proximity, w_approach]
        Must be on the probability simplex: w >= 0 and sum(w)=1.
    """

    name: str
    weights: np.ndarray

    def __post_init__(self) -> None:
        assert len(self.weights) == 5, "Must have exactly 5 preference weights"
        assert np.all(self.weights >= 0), "Weights must be non-negative"
        assert np.isclose(np.sum(self.weights), 1.0), "Weights must sum to 1"

    def describe(self) -> None:
        """Pretty print profile."""
        print(f"\n{'='*60}")
        print(f"Patient Profile: {self.name}")
        print(f"{'='*60}")
        print(f"Time (speed):      {self.weights[0]:.3f}")
        print(f"Safety:            {self.weights[1]:.3f}")
        print(f"Battery:           {self.weights[2]:.3f}")
        print(f"Proximity:         {self.weights[3]:.3f}")
        print(f"Approach:          {self.weights[4]:.3f}")
        print(f"Sum: {np.sum(self.weights):.3f}")

        max_idx = int(np.argmax(self.weights))
        print(f"\nPrimary concern: {CONCERNS[max_idx]} ({self.weights[max_idx]:.1%})")
        print(f"{'='*60}\n")


# Predefined patient profiles
PATIENT_PROFILES = {
    "speed_oriented": PatientProfile(
        name="Speed-Oriented Patient", weights=np.array([0.50, 0.12, 0.14, 0.14, 0.10])
    ),
    "safety_first": PatientProfile(
        name="Safety-First Patient", weights=np.array([0.10, 0.50, 0.15, 0.15, 0.10])
    ),
    "comfort_focused": PatientProfile(
        name="Comfort-Focused Patient", weights=np.array([0.15, 0.15, 0.10, 0.40, 0.20])
    ),
    "energy_conscious": PatientProfile(
        name="Energy-Conscious Patient",
        weights=np.array([0.15, 0.15, 0.45, 0.15, 0.10]),
    ),
    "presentation_focused": PatientProfile(
        name="Presentation-Focused Patient",
        weights=np.array([0.05, 0.10, 0.05, 0.20, 0.60]),
    ),
}


class PreferenceLearningEngine:
    """
    Learns patient preferences via projected gradient descent.

    Key design choice: multi-dimensional learning.
    We model and fit each rating dimension separately.
    """

    def __init__(
        self,
        true_patient_profile: PatientProfile,
        initial_weights: Optional[np.ndarray] = None,
        learning_rate: float = 0.1,
        rating_noise: float = 0.3,
        lr_decay: float = 0.1,
        ema_alpha: float = 1.0,
    ) -> None:
        """
        Args:
            true_patient_profile: Hidden ground truth preferences (w*).
            initial_weights: Robot's initial estimate (default uniform).
            learning_rate: Base step size (η₀).
            rating_noise: Gaussian noise added to ratings for realism.
            lr_decay: Decay rate for learning rate schedule.
                      Effective lr = η₀ / (1 + lr_decay * episode).
                      Set to 0.0 for constant learning rate (original behavior).
            ema_alpha: Exponential moving average blending factor.
                       1.0 = no smoothing (original behavior).
                       0.8 = 80% new + 20% previous (dampens oscillation).
        """
        self.true_profile = true_patient_profile
        self.base_learning_rate = float(learning_rate)
        self.learning_rate = float(learning_rate)
        self.lr_decay = float(lr_decay)
        self.rating_noise = float(rating_noise)
        self.ema_alpha = float(ema_alpha)

        # Robot's estimated weights (simplex)
        if initial_weights is None:
            self.estimated_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=float)
        else:
            self.estimated_weights = self._project_to_simplex(
                np.array(initial_weights, dtype=float)
            )

        # Multi-dimensional bias (one per rating dimension)
        self.bias_vec = np.zeros(5, dtype=float)
        self.bias_lr = self.learning_rate

        # Optional regularization (disabled by default for stability/diagnosis)
        self.l2_reg = 0.0
        self.entropy_reg = 0.0
        self.w_prior = self.estimated_weights.copy()

        # ── Instrumented loss tracking (B6 figure support) ───────────
        self.last_mse: Optional[float] = None  # most recent scalar MSE
        self.last_loss: Optional[float] = None  # alias (runner probes both)
        self.last_per_dim_mse: Optional[np.ndarray] = None  # shape (5,)
        self.last_gradient_norm: Optional[float] = None

        # History (per-episode)
        self.weight_history = [self.estimated_weights.copy()]
        self.rating_history = []
        self.loss_history = []  # scalar MSE per episode
        self.per_dim_loss_history = []  # [episode][dim] MSE decomposition
        self.gradient_norm_history = []  # ||∇_w L|| per episode
        self.feature_history = []
        self.effective_lr_history = []  # η_eff per episode

        # Convergence tracking
        self.episode_count = 0
        self.converged = False
        self.convergence_threshold = 0.05  # L2 distance threshold

        print("PreferenceLearningEngine initialized")
        print(f"  Learning rate (η): {self.learning_rate}")
        if self.lr_decay > 0:
            print(f"  LR decay: η / (1 + {self.lr_decay} * episode)")
        if self.ema_alpha < 1.0:
            print(f"  EMA smoothing: α={self.ema_alpha} (dampens oscillation)")
        print(f"  Rating noise: {self.rating_noise}")
        print(f"  Initial weights: {self.estimated_weights}")
        print("\n   True patient profile (HIDDEN):")
        self.true_profile.describe()

    # ── Accessors for experiment runner extraction ────────────────────

    @property
    def mse(self) -> Optional[float]:
        """Alias so extract_learner_mse() finds it via getattr probe."""
        return self.last_mse

    def get_loss_summary(self) -> Dict:
        """Return a snapshot of all loss-related state for JSON serialisation."""
        return {
            "last_mse": self.last_mse,
            "last_per_dim_mse": (
                self.last_per_dim_mse.tolist()
                if self.last_per_dim_mse is not None
                else None
            ),
            "last_gradient_norm": self.last_gradient_norm,
            "loss_history": list(self.loss_history),
            "per_dim_loss_history": [
                row.tolist() if isinstance(row, np.ndarray) else row
                for row in self.per_dim_loss_history
            ],
            "gradient_norm_history": list(self.gradient_norm_history),
            "effective_lr_history": list(self.effective_lr_history),
        }

    # ── Core methods ─────────────────────────────────────────────────

    def _get_effective_lr(self) -> float:
        """Compute decayed learning rate for current episode."""
        if self.lr_decay <= 0.0:
            return self.base_learning_rate
        return self.base_learning_rate / (1.0 + self.lr_decay * self.episode_count)

    def generate_patient_ratings(self, features: Dict[str, float]) -> np.ndarray:
        """
        Generate simulated patient ratings based on hidden profile.

        Generator (matches your original intent):
        - features are normalized [0=best, 1=worst]
        - per-dimension rating:
            r_i = 5 - 4 * f_i * w*_i + noise

        Returns:
            ratings: np.ndarray shape (5,) in [1, 5]
        """
        f = np.array([features[k] for k in FEATURE_KEYS], dtype=float)

        base_ratings = 5.0 - (f * self.true_profile.weights * 4.0)
        noise = np.random.randn(5) * self.rating_noise
        ratings = np.clip(base_ratings + noise, 1.0, 5.0)

        print("\n   Patient Ratings (1-5 scale):")
        for i, k in enumerate(FEATURE_KEYS):
            print(
                f"     {k.capitalize():<10}: {ratings[i]:.1f}  (feature={features[k]:.3f})"
            )

        return ratings

    def update_weights(
        self, ratings: np.ndarray, features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Multi-dimensional weight update via projected gradient descent.

        Model we fit (matches generator structure, but with estimated weights):
            r_hat_i = 5 - 4 * f_i * w_i - b_i

        Loss:
            L = mean_i (r_hat_i - r_i)^2

        Then:
            w <- proj_simplex(w - η_eff ∇_w L)

        where η_eff = η₀ / (1 + lr_decay * episode) for annealed convergence.
        """
        self.episode_count += 1

        # Compute effective (decayed) learning rate
        effective_lr = self._get_effective_lr()
        self.learning_rate = effective_lr  # expose for logging

        f = np.array([features[k] for k in FEATURE_KEYS], dtype=float)

        # Predict each dimension's rating
        r_hat = 5.0 - 4.0 * (f * self.estimated_weights) - self.bias_vec
        err = r_hat - ratings  # shape (5,)

        # Gradients for MSE
        grad_w = 2.0 * err * (-4.0) * f
        grad_b = 2.0 * err * (-1.0)

        # Optional regularization (off by default)
        if self.l2_reg > 0.0:
            grad_w += 2.0 * self.l2_reg * (self.estimated_weights - self.w_prior)

        if self.entropy_reg > 0.0:
            eps = 1e-12
            grad_w += self.entropy_reg * (
                1.0 + np.log(np.clip(self.estimated_weights, eps, None))
            )

        old_weights = self.estimated_weights.copy()

        # GD step + simplex projection (using effective decayed lr)
        new_weights = self.estimated_weights - effective_lr * grad_w
        new_weights = self._project_to_simplex(new_weights)

        # EMA smoothing: blend with previous weights to dampen oscillation
        if self.ema_alpha < 1.0:
            new_weights = (
                self.ema_alpha * new_weights + (1.0 - self.ema_alpha) * old_weights
            )
            new_weights = self._project_to_simplex(new_weights)

        self.estimated_weights = new_weights

        # Bias update (also decayed)
        self.bias_vec -= effective_lr * grad_b

        weight_change = float(np.linalg.norm(self.estimated_weights - old_weights))
        distance_to_true = float(
            np.linalg.norm(self.estimated_weights - self.true_profile.weights)
        )

        # ── Loss instrumentation (B6 support) ───────────────────────
        per_dim_mse = err**2  # shape (5,)
        mse = float(np.mean(per_dim_mse))  # scalar
        grad_norm = float(np.linalg.norm(grad_w))

        # Instant values (runner probes these via getattr)
        self.last_mse = mse
        self.last_loss = mse  # alias
        self.last_per_dim_mse = per_dim_mse.copy()
        self.last_gradient_norm = grad_norm

        # Per-episode histories
        self.loss_history.append(mse)
        self.per_dim_loss_history.append(per_dim_mse.copy())
        self.gradient_norm_history.append(grad_norm)
        self.effective_lr_history.append(effective_lr)

        if distance_to_true < self.convergence_threshold and not self.converged:
            self.converged = True

        # Weight + rating history
        self.weight_history.append(self.estimated_weights.copy())
        self.rating_history.append(ratings.copy())

    

        return {
            "episode": self.episode_count,
            "old_weights": old_weights,
            "new_weights": self.estimated_weights.copy(),
            "gradient": grad_w.copy(),
            "weight_change": weight_change,
            "distance_to_true": distance_to_true,
            "converged": self.converged,
            "bias_vec": self.bias_vec.copy(),
            "loss_mse": mse,
            "per_dim_mse": per_dim_mse.tolist(),
            "gradient_norm": grad_norm,
            "effective_lr": effective_lr,
        }

    def process_episode(self, features: Dict[str, float]) -> Tuple[np.ndarray, Dict]:
        """Run one full learning cycle: generate ratings -> update weights."""
        print(f"\n{'='*80}")
        print(f"PREFERENCE LEARNING - EPISODE {self.episode_count + 1}")
        print(f"{'='*80}")

        self.feature_history.append(features.copy())

        ratings = self.generate_patient_ratings(features)
        update_info = self.update_weights(ratings, features)

        return ratings, update_info

    @staticmethod
    def _project_to_simplex(weights: np.ndarray) -> np.ndarray:
        """
        Project weights onto probability simplex: w >= 0, sum(w) = 1.
        Efficient O(n log n) algorithm.
        """
        w = np.array(weights, dtype=float)

        sorted_w = np.sort(w)[::-1]
        cumsum = np.cumsum(sorted_w)
        idx = np.arange(1, len(w) + 1)

        condition = sorted_w - (cumsum - 1.0) / idx > 0
        rho = int(np.where(condition)[0][-1]) if np.any(condition) else 0
        theta = (cumsum[rho] - 1.0) / (rho + 1.0)

        projected = np.maximum(w - theta, 0.0)

        s = float(np.sum(projected))
        if s <= 0.0:
            # fallback: uniform
            return np.ones_like(projected) / len(projected)

        return projected / s

    def get_current_weights(self) -> np.ndarray:
        return self.estimated_weights.copy()

    def get_learning_summary(self) -> Dict:
        if self.episode_count == 0:
            return {
                "episodes": 0,
                "converged": False,
                "message": "No episodes completed yet",
            }

        final_distance = float(
            np.linalg.norm(self.estimated_weights - self.true_profile.weights)
        )

        weight_changes = [
            float(np.linalg.norm(self.weight_history[i + 1] - self.weight_history[i]))
            for i in range(len(self.weight_history) - 1)
        ]

        avg_rating = (
            float(np.mean([np.mean(r) for r in self.rating_history]))
            if self.rating_history
            else 0.0
        )

        return {
            "episodes": self.episode_count,
            "converged": self.converged,
            "final_weights": self.estimated_weights.copy(),
            "true_weights": self.true_profile.weights.copy(),
            "final_distance": final_distance,
            "convergence_threshold": self.convergence_threshold,
            "total_weight_change": float(sum(weight_changes)),
            "average_rating": avg_rating,
            "final_loss": self.loss_history[-1] if self.loss_history else None,
            "weight_history": self.weight_history.copy(),
            "rating_history": self.rating_history.copy(),
            "loss_history": self.loss_history.copy(),
            "per_dim_loss_history": [
                row.tolist() if isinstance(row, np.ndarray) else row
                for row in self.per_dim_loss_history
            ],
            "gradient_norm_history": list(self.gradient_norm_history),
            "effective_lr_history": list(self.effective_lr_history),
        }

    def print_learning_summary(self) -> None:
        summary = self.get_learning_summary()
        if summary["episodes"] == 0:
            print("No learning episodes completed yet.")
            return

        print(f"\n{'='*80}")
        print("PREFERENCE LEARNING SUMMARY")
        print(f"{'='*80}")
        print(f"Episodes completed: {summary['episodes']}")
        print(f"Converged: {' YES' if summary['converged'] else '[FAIL] NO'}")
        print(f"Final distance to true profile: {summary['final_distance']:.4f}")
        print(f"Convergence threshold: {summary['convergence_threshold']:.4f}")

        print("\nWeight Evolution:")
        print(f"  Initial:  {self.weight_history[0]}")
        print(f"  Final:    {summary['final_weights']}")
        print(f"  True:     {summary['true_weights']}")

        print("\nLearning Metrics:")
        print(f"  Total weight change: {summary['total_weight_change']:.4f}")
        print(f"  Average rating: {summary['average_rating']:.2f}/5.0")
        if summary.get("final_loss") is not None:
            print(f"  Final loss (MSE): {summary['final_loss']:.6f}")
        if self.gradient_norm_history:
            print(f"  Final ||∇_w||: {self.gradient_norm_history[-1]:.6f}")
            print(f"  Mean  ||∇_w||: {np.mean(self.gradient_norm_history):.6f}")

        true_max = int(np.argmax(summary["true_weights"]))
        learned_max = int(np.argmax(summary["final_weights"]))

        print("\nWhat Patient Values Most:")
        print(
            f"  True preference: {CONCERNS[true_max]} ({summary['true_weights'][true_max]:.1%})"
        )
        print(
            f"  Robot learned:   {CONCERNS[learned_max]} ({summary['final_weights'][learned_max]:.1%})"
        )

        if true_max == learned_max:
            print("   Robot correctly identified primary concern!")
        else:
            print("  [FAIL] Robot misidentified primary concern")

        print(f"\n{'='*80}")
        print("EPISODE-BY-EPISODE CONVERGENCE STATUS")
        print(f"{'='*80}")
        print(
            f"{'Ep':<4} {'Distance':<10} {'MSE':<10} {'||∇||':<10} {'Converged':<12} {'Top Learned':<20} {'Weight Δ':<10}"
        )
        print(f"{'-'*4} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*20} {'-'*10}")

        for i in range(len(self.weight_history)):
            dist = float(
                np.linalg.norm(self.weight_history[i] - self.true_profile.weights)
            )
            converged = " YES" if dist < self.convergence_threshold else "[FAIL] NO"

            top_idx = int(np.argmax(self.weight_history[i]))
            top_concern = f"{CONCERNS[top_idx]} ({self.weight_history[i][top_idx]:.2f})"

            delta = (
                float(
                    np.linalg.norm(self.weight_history[i] - self.weight_history[i - 1])
                )
                if i > 0
                else 0.0
            )

            # Loss/gradient are offset by 1 (no loss at init)
            ep_mse = (
                self.loss_history[i - 1]
                if (i > 0 and i - 1 < len(self.loss_history))
                else 0.0
            )
            ep_grad = (
                self.gradient_norm_history[i - 1]
                if (i > 0 and i - 1 < len(self.gradient_norm_history))
                else 0.0
            )

            ep_label = "Init" if i == 0 else str(i)
            print(
                f"{ep_label:<4} {dist:<10.4f} {ep_mse:<10.6f} {ep_grad:<10.4f} {converged:<12} {top_concern:<20} {delta:<10.4f}"
            )

        print(f"{'='*80}\n")

        num_converged = sum(
            1
            for w in self.weight_history
            if np.linalg.norm(w - self.true_profile.weights)
            < self.convergence_threshold
        )
        pct = (
            (num_converged / summary["episodes"] * 100.0)
            if summary["episodes"] > 0
            else 0.0
        )
        print("Convergence Statistics:")
        print(
            f"   Episodes converged: {num_converged}/{summary['episodes']} ({pct:.1f}%)"
        )

        first_conv = "Never"
        for j, w in enumerate(self.weight_history):
            if (
                np.linalg.norm(w - self.true_profile.weights)
                < self.convergence_threshold
            ):
                first_conv = f"Episode {j}"
                break
        print(f"   First convergence: {first_conv}")

        # Loss trajectory summary
        if self.loss_history:
            print(
                f"\n   Loss trajectory: start={self.loss_history[0]:.6f}, "
                f"end={self.loss_history[-1]:.6f}, "
                f"min={min(self.loss_history):.6f}, "
                f"mean={np.mean(self.loss_history):.6f}"
            )

        print(f"{'='*80}\n")

    def visualize_learning(self, save_path: Optional[str] = None) -> None:
        """Visualize weight evolution, distance-to-true, MSE loss, and gradient norm."""
        if self.episode_count == 0:
            print("No episodes to visualize yet.")
            return

        fig, axes = plt.subplots(4, 1, figsize=(10, 16))
        episodes = np.arange(len(self.weight_history))

        # Plot 1: Weight Evolution
        ax1 = axes[0]
        weight_array = np.array(self.weight_history)

        for i, label in enumerate(
            ["Time", "Safety", "Battery", "Proximity", "Approach"]
        ):
            ax1.plot(
                episodes,
                weight_array[:, i],
                label=f"{label} (true={self.true_profile.weights[i]:.2f})",
                linewidth=2,
                marker="o",
                markersize=4,
            )

        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Weight Value")
        ax1.set_title("Preference Weight Evolution")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 0.6])

        # Plot 2: Distance to True Profile
        ax2 = axes[1]
        distances = [
            float(np.linalg.norm(w - self.true_profile.weights))
            for w in self.weight_history
        ]
        ax2.plot(episodes, distances, linewidth=2, marker="o")
        ax2.axhline(
            y=self.convergence_threshold,
            linestyle="--",
            label="Convergence threshold",
        )
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("L2 Distance")
        ax2.set_title("Distance to True Patient Profile")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: MSE Loss (B6-style)
        ax3 = axes[2]
        if self.loss_history:
            loss_eps = np.arange(1, len(self.loss_history) + 1)
            ax3.plot(
                loss_eps,
                self.loss_history,
                linewidth=2,
                marker="o",
                color="tab:red",
                label="Total MSE",
            )

            # Per-dimension decomposition (stacked area)
            if self.per_dim_loss_history:
                per_dim = np.array(self.per_dim_loss_history)
                for d, (name, color) in enumerate(
                    zip(
                        FEATURE_KEYS,
                        ["#2196F3", "#F44336", "#FF9800", "#4CAF50", "#9C27B0"],
                    )
                ):
                    ax3.plot(
                        loss_eps,
                        per_dim[:, d],
                        linewidth=1,
                        alpha=0.5,
                        color=color,
                        linestyle="--",
                        label=f"{name.title()} dim",
                    )

            ax3.set_xlabel("Episode")
            ax3.set_ylabel("MSE")
            ax3.set_title("Preference Learner Loss (B6)")
            ax3.legend(fontsize=8, ncol=3)
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(bottom=0)

        # Plot 4: Gradient Norm
        ax4 = axes[3]
        if self.gradient_norm_history:
            grad_eps = np.arange(1, len(self.gradient_norm_history) + 1)
            ax4.plot(
                grad_eps,
                self.gradient_norm_history,
                linewidth=2,
                marker="o",
                color="tab:purple",
            )
            ax4.set_xlabel("Episode")
            ax4.set_ylabel("$\\|\\nabla_w L\\|_2$")
            ax4.set_title("Gradient Norm Over Episodes")
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(bottom=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved learning visualization to: {save_path}")

        plt.close(fig)
        print("Visualization complete")


def test_preference_learning() -> PreferenceLearningEngine:
    """Quick test for the preference learning engine."""
    print("Testing Preference Learning Engine")
    print("=" * 80)

    np.random.seed(0)

    patient = PATIENT_PROFILES["speed_oriented"]
    patient.describe()

    learner = PreferenceLearningEngine(
        true_patient_profile=patient,
        learning_rate=0.05,
        rating_noise=0.2,
    )

    print("\n" + "=" * 80)
    print("SIMULATING 10 LEARNING EPISODES")
    print("=" * 80)

    episode_features = [
        {"time": 0.8, "safety": 0.1, "battery": 0.3, "proximity": 0.2, "approach": 0.3},
        {"time": 0.5, "safety": 0.2, "battery": 0.4, "proximity": 0.3, "approach": 0.2},
        {"time": 0.3, "safety": 0.4, "battery": 0.5, "proximity": 0.4, "approach": 0.3},
        {"time": 0.2, "safety": 0.5, "battery": 0.4, "proximity": 0.3, "approach": 0.2},
        {
            "time": 0.15,
            "safety": 0.4,
            "battery": 0.45,
            "proximity": 0.35,
            "approach": 0.25,
        },
        {
            "time": 0.18,
            "safety": 0.45,
            "battery": 0.42,
            "proximity": 0.32,
            "approach": 0.28,
        },
        {
            "time": 0.16,
            "safety": 0.42,
            "battery": 0.44,
            "proximity": 0.36,
            "approach": 0.24,
        },
        {
            "time": 0.17,
            "safety": 0.43,
            "battery": 0.43,
            "proximity": 0.34,
            "approach": 0.26,
        },
        {
            "time": 0.15,
            "safety": 0.41,
            "battery": 0.45,
            "proximity": 0.33,
            "approach": 0.27,
        },
        {
            "time": 0.16,
            "safety": 0.44,
            "battery": 0.42,
            "proximity": 0.35,
            "approach": 0.25,
        },
    ]

    for i, features in enumerate(episode_features, 1):
        print(f"\n{'='*80}")
        print(f"EPISODE {i}")
        print(f"{'='*80}")
        print(f"Features: {features}")

        learner.process_episode(features)
        print(f"\nCurrent estimated weights: {learner.get_current_weights()}")

    learner.print_learning_summary()
    learner.visualize_learning(save_path="preference_learning_test.png")

    # Verify B6 instrumentation
    print("\n" + "=" * 80)
    print("B6 INSTRUMENTATION CHECK")
    print("=" * 80)
    print(f"  last_mse:            {learner.last_mse}")
    print(f"  last_loss:           {learner.last_loss}")
    print(f"  mse (property):      {learner.mse}")
    print(f"  last_per_dim_mse:    {learner.last_per_dim_mse}")
    print(f"  last_gradient_norm:  {learner.last_gradient_norm}")
    print(f"  loss_history len:    {len(learner.loss_history)}")
    print(f"  per_dim_loss len:    {len(learner.per_dim_loss_history)}")
    print(f"  grad_norm_hist len:  {len(learner.gradient_norm_history)}")
    print(f"  eff_lr_hist len:     {len(learner.effective_lr_history)}")

    # Verify runner extraction would work
    for attr in ("last_mse", "last_loss", "mse"):
        val = getattr(learner, attr, None)
        print(f"  getattr('{attr}'): {val}")
    hist = getattr(learner, "loss_history", None)
    if hist and len(hist) > 0:
        print(f"  loss_history[-1]:    {hist[-1]}")

    print("\nPreference Learning Engine test complete!")
    return learner


if __name__ == "__main__":
    test_preference_learning()
