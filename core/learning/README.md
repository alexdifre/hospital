# core/learning/

Patient preference learning via projected gradient descent on the probability simplex.

## Files

### `preference_learner.py`

The central learning engine. Maintains and updates the robot's estimate of a patient's hidden preference profile.

### `translator_params.py`

Dataclasses for the learnable MPC parameter space.

- `TranslatorParams` — 32-parameter vector `φ`:
  - indices 0–5: `Q_diag` (position/velocity tracking weights)
  - indices 6–8: `r_base_ax, r_base_ay, r_base_alpha` (pre-softplus control cost params)
  - indices 9–19: reserved / padding to align z_target block
  - indices 20–29: `z_target_A` flattened (2×5 matrix)
  - indices 30–31: `z_target_b` (2D bias)
- `MPCParameterGradients` — holds `dQ_dphi (6×32)`, `dR_dphi (3×32)`, `dZ_dphi (2×32)` for the chain-rule update

### `learnable_translator.py`

Maps preference weights `ŵ ∈ Δ⁴` → MPC parameters `(Q, R, z_target)`.

#### Learned Diagonal R (softplus activation)

Each axis has an independent pre-activation parameter:
```
R_i = log(1 + exp(r_base_i)) + ε      # softplus, always positive
```
where `r_base_ax`, `r_base_ay`, `r_base_alpha` are learned. This guarantees `R > 0` without clamping and gives a smooth gradient through zero.

#### Learned z_target

A preference-conditioned 2D position offset applied to every MPC stage cost:
```
z_target(ŵ) = A @ ŵ + b      (A: 2×5, b: 2D)
```
The stage reference becomes `x_ref_stage = [x_ref[:2] + z_target, x_ref[2:]]`, while the terminal cost tracks `x_ref` exactly (no offset). This lets the system learn per-profile path biases (e.g. a safety-conscious patient causes the robot to ride farther from obstacles).

Initialised to zero — identical to fixed `x_ref` tracking until learned.

#### Gradient pipeline for z_target

The IFT engine computes `∂J*/∂z_target` alongside `∂J*/∂Q` and `∂J*/∂R`. The translator then applies the chain rule:
```
∂J*/∂φ  +=  dZ_dphi.T @ dJ_dz_target
```
where `dZ_dphi[i, 20+i*5+j] = ŵ[j]` (A-block) and `dZ_dphi[i, 30+i] = 1.0` (b-block). A and b are updated via the same gradient pipeline with no architectural change.

---

## Problem Formulation

A patient has a hidden preference profile `w* ∈ Δ⁴` (a point on the 5-simplex):
```
w* = [w_time, w_safety, w_battery, w_proximity, w_approach]
w* ≥ 0,  Σ w*_i = 1
```

The robot maintains an estimate `w_hat` and updates it after each episode. Learning is **multi-dimensional** — each dimension carries an independent signal; ratings are never collapsed to a scalar.

---

## Learning Loop (per episode)

```
1. Execute episode → extract features f ∈ [0,1]⁵
       f = [time/max_time, safety_score, battery_used, proximity_error, approach_quality]
       (0 = best, 1 = worst for each dimension)

2. Patient provides ratings r ∈ [1,5]⁵  (one per dimension)

3. Compute loss:
       L(w) = ‖w ⊙ f − r‖²   (per-dimension MSE)

4. Gradient step:
       w ← w − η ∇_w L
       ∇_w L = (w ⊙ f − r) ⊙ f

5. Project back onto simplex:
       w_hat ← Π_Δ(w)

6. Decay learning rate:
       η ← η₀ / (1 + decay × episode)
```

---

## Key Classes

### `PatientProfile`
Dataclass holding a named ground-truth preference vector `w*`. Used only in simulation experiments to generate synthetic ratings; the real robot never has access to it.

### `PreferenceLearningEngine`
Maintains `w_hat` across episodes. Exposes:
- `update(features, ratings) → w_hat` — runs one gradient + projection step
- `loss_history`, `per_dim_loss_history` — convergence data for Section 8 figures
- `gradient_norm_history` — learning diagnostics

---

## Predefined Patient Profiles

Used as ground-truth `w*` in experiments:

| Name | Time | Safety | Battery | Proximity | Approach |
|------|------|--------|---------|-----------|----------|
| uniform | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 |
| speed_oriented | 0.40 | 0.10 | 0.05 | 0.25 | 0.20 |
| safety_first | 0.10 | 0.40 | 0.05 | 0.25 | 0.20 |
| comfort_focused | 0.15 | 0.15 | 0.10 | 0.40 | 0.20 |
| energy_conscious | 0.10 | 0.15 | 0.40 | 0.20 | 0.15 |
| presentation_focused | 0.05 | 0.10 | 0.05 | 0.20 | 0.60 |
| mild_speed | 0.30 | 0.20 | 0.18 | 0.17 | 0.15 |
| mild_safety | 0.18 | 0.30 | 0.20 | 0.17 | 0.15 |

---

## Convergence

Target: `‖w_hat − w*‖ < 0.05` (typically reached in 15–20 episodes).

The `per_dim_loss_history` tracks per-dimension MSE for Section 8 figure B6 — showing which preference dimensions converge fastest and whether any dimensions are structurally harder to learn (e.g., approach quality requires the right task path to generate informative features).
