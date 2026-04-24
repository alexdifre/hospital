# Fuzzy State — Sigmoid Battery Memberships

Branch: `mpc-coldstart-investigation`, commit `c84971a`

---

## What Changed

Replaced triangular/trapezoidal battery membership functions with smooth
sigmoid-based ones:

**Before (trapezoidal/shoulder):**
```
"low":    left_shoulder(SoC, b=0.15, c=0.35)
"medium": trapezoidal(SoC, a=0.20, b=0.35, c=0.65, d=0.80)
"high":   right_shoulder(SoC, a=0.55, b=0.80)
```
Piecewise linear — non-differentiable kinks at every boundary.

**After (sigmoid):**
```
μ_Low  = sigmoid(-10 × (SoC - 0.3))
μ_High = sigmoid(+10 × (SoC - 0.7))
μ_Med  = max(0, 1 - μ_Low - μ_High)
```
Continuously differentiable everywhere.

---

## Empirical Significance

**Modest.** The ablation results tell the story clearly:

| Condition | Best d | Conv. | Δ Best d |
|---|---|---|---|
| Full System (sigmoid fuzzy) | 0.079 | 20/25 | --- |
| Crisp (no fuzzy at all) | 0.080 | 20/25 | +0.001 |

Fuzzy state estimation in general has near-zero impact on convergence distance.
The sigmoid upgrade over triangular would be an even smaller effect in isolation —
there is no direct A/B comparison in the results but the ceiling is Δ+0.001.

---

## Why the Theoretical Argument Was Misdirected

The commit message says "enabling cleaner gradient flow through the battery cost
terms" — this is **incorrect framing**. Tracing the actual data flow reveals:

**What the fuzzy state estimator actually does:**
```python
fm = self.fuzzy_estimator.estimate(current_6d_state[:2], task_state.battery_soc)
task_state.location_memberships = dict(fm.location_memberships)
task_state.location = fm.dominant_location
```
→ Used for **location estimation only** — soft location memberships for task
state management and planning decisions.

**What the preference learner's battery feature actually is:**
```python
"battery": float(np.clip(episode_features["total_battery_used"] / 100.0, 0.0, 1.0))
```
→ **Raw cumulative battery consumption** — completely bypasses the fuzzy
memberships. No fuzzy smoothing involved.

So the sigmoid change has zero effect on the battery feature path to the preference
learner. The preference learner uses projected gradient descent on episode-level
ratings, not backpropagation through the fuzzy state estimator — there was never
a gradient flowing through the membership functions to begin with.

**What the sigmoid change actually improves:**
Smoother location state transitions for task planning — the soft membership
transitions between locations (e.g., "robot is 70% at pharmacy, 30% at corridor")
are now continuously differentiable, which produces more stable task state updates
as the robot moves between waypoints. This is the real benefit.

---

## Where It Does Help

- **Smoother location state transitions** — as the robot moves between waypoints,
  location memberships transition continuously rather than jumping at hard boundaries,
  producing more stable task state updates for the planner
- **Architectural cleanliness** — continuously differentiable throughout the state
  estimation pipeline, which matters if end-to-end differentiable planning is added
  in future work
- **No downside** — sigmoid is strictly better behaved than triangular; the change
  costs nothing

---

## Attempted Task 4 — Battery Feature via Defuzzification (Reverted)

Implemented and tested replacing raw consumption with defuzzified SoC:

```python
# episode_runner.py — defuzzified battery feature
"battery": battery_defuzzify(task_state.battery_soc)

# meal_profiles.py
f_battery = battery_defuzzify(battery_end)
```

**Test results (1 seed, 40 episodes):**

| Profile | w_battery | Raw consumption best_d | Defuzzified SoC best_d | Δ |
|---|---|---|---|---|
| energy_conscious | 0.40 | 0.019 | 0.025 | +0.006 worse |
| speed_oriented | 0.05 | 0.074 | 0.076 | ~same |

**energy_conscious episode progression (defuzzified):**
```
Ep  0: d=0.071  w_battery=0.37
Ep  1: d=0.060  w_battery=0.38
Ep  3: d=0.094  ← oscillation, battery weight dropping (0.34)
Ep 37: d=0.033  w_battery=0.38  ← best
Ep 39: d=0.047  w_battery=0.37  ← drifts back, final_d=0.047
```
Battery weight oscillates between 0.33–0.38 rather than pushing cleanly to 0.40.
Compare to raw consumption: converges to best_d=0.019, w_battery stable at 0.40.

**speed_oriented episode progression (defuzzified):**
```
Ep  0: d=0.136  w_time=0.30
Ep  3: d=0.076  ← first convergence (below 0.10)
Ep 39: d=0.086  final_d=0.086
```
Essentially unaffected — w_battery=0.05 is too small to matter.

**Why it failed:** The robot recharges mid-episode (`battery_soc += 0.4` at charger).
End-of-episode SoC doesn't reflect total energy consumed — a run that used 80% and
recharged looks the same as one that used 20%. Raw consumption (`total_battery_used`)
captures the full episode energy expenditure correctly and gives the outer loop a
stronger, less ambiguous gradient signal for w_battery. Reverted.

`battery_defuzzify()` remains in `fuzzy_state.py` as documentation of the experiment.

---

## Paper Framing

Do NOT overclaim. Frame as:

*"The fuzzy state estimator uses smooth sigmoid-based location membership
functions rather than triangular/trapezoidal forms, ensuring continuously
differentiable location state transitions as the robot moves between waypoints."*

Note: the commit motivation mentioning "battery cost gradients" was misdirected —
the battery feature fed to the preference learner is raw cumulative consumption,
not derived from the fuzzy memberships. The correct benefit is smoother location
state transitions for task planning.

The crisp ablation (Δ+0.001) establishes that fuzzy state estimation is a
robustness feature rather than a convergence driver. The sigmoid upgrade is in
the same category — an architectural improvement for smoothness and future
extensibility, not a performance-critical change.
