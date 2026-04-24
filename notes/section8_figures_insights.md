# Section 8 Figures — Descriptions & Insights for Paper Writing

All figures are in `results/section8/figures/`. Generated from full condition results
(5 profiles × 5 seeds, 40 episodes each) plus baselines and ablations.

---

## B1 — Convergence Curves (`fig8_convergence.png`)

**What it shows:** L2 distance ‖ŵ − w*‖₂ vs episode for all 5 profiles. Shaded bands
show variance across 5 seeds. Dashed lines mark convergence thresholds (0.10 for most
profiles, 0.15 for presentation-focused).

**Key insights:**
- Speed, Safety, Comfort, Energy all converge tightly below threshold by episode 5–10
  and stay there for the remaining 30 episodes — narrow bands confirm robustness across seeds
- Presentation-focused (green) is the dramatic outlier — diverges wildly with large
  oscillations that grow over time rather than shrinking, never crossing its threshold
- The contrast between the flat converged profiles and the oscillating green line is
  visually striking — this is the single best figure for communicating the
  presentation-focused difficulty
- The convergence threshold lines make it immediately obvious which profiles succeeded

**Paper use:** Primary figure for the convergence results section. Leads the comparative
analysis. The visual contrast between 4 converging profiles and 1 oscillating one sets
up the presentation-focused investigation naturally.

---

## B2 — Weight Evolution (`fig8_weight_evolution.png`)

**What it shows:** Per-dimension weight trajectories over episodes for 3 representative
profiles (Speed, Safety, Presentation-focused). Each line is one preference dimension
(time, safety, battery, proximity, approach). Horizontal lines mark true weights w*.

**Key insights:**
- Speed profile: time weight (dominant, true=0.40) rises cleanly and plateaus near
  the true value. Other dimensions settle below their true values. Clean monotone
  convergence.
- Safety profile: safety weight (dominant, true=0.40) rises and converges. Similar
  clean behaviour.
- Presentation-focused: approach weight (dominant, true=0.60) rises toward ~0.45 but
  stalls well short of 0.60. The other dimensions don't fully suppress either. Shows
  exactly why the distance never closes — the learner correctly identifies approach as
  dominant but can't push the weight high enough under the default decay schedule.

**Paper use:** Supports the weight convergence claim with a mechanistic view.
Presentation-focused panel is strong evidence for the "insufficient gradient signal"
argument in the profile-adaptive tuning section.

---

## B3 — Final Learned vs True Weights (`fig8_final_weights.png`)

**What it shows:** Grouped bar chart of final learned weights (bars with error bars)
vs true weights (diamond markers) for all 5 profiles across the 5 weight dimensions.

**Key insights:**
- 4 converging profiles: bars sit very close to their diamonds — the learner has
  accurately recovered the true preference weights. Error bars are small, confirming
  consistency across seeds.
- Presentation-focused (green): approach bar is clearly short of its diamond at 0.60
  — the single most visually obvious gap in the entire figure. All other profiles
  match their diamonds; only presentation-focused has a visible discrepancy on approach.
- Battery and time dimensions are consistently suppressed across all profiles — these
  are the "background" dimensions that all patients de-prioritise relative to their
  dominant concern.

**Paper use:** The definitive "did it work?" figure. Pairs well with T1 (results table).
The single gap for presentation-focused approach weight summarises the entire problem
in one bar.

---

## B4 — Feature Space (`fig8_feature_space.png`)

**What it shows:** Feature centroids per profile for medication vs meal tasks across
the 5 preference dimensions.

**Key insights:**
- Medication and meal tasks occupy different regions of feature space — different
  task types genuinely produce different observational signals
- This is the mechanistic justification for why both task types are needed (supports
  Med Only ablation finding)
- Profiles that are hard to distinguish in one task type become separable in the other

**Paper use:** Supports the ablation discussion. Explains why removing either task
type degrades learning.

---

## B5 — Plan Diversity (`fig8_plan_diversity.png`)

**What it shows:** Meal type selection proportions across profiles and episodes —
how the robot's planning decisions evolve as preferences are learned.

**Key insights:**
- Plan selection adapts over episodes as the preference learner updates weights
- Different profiles drive different plan distributions — the planner is responding
  to the learned weights, not just executing a fixed policy
- Diversity in plans is a signal that the outer loop is actively influencing the
  inner loop's behaviour

**Paper use:** Shows end-to-end coupling between preference learning and task planning.
Demonstrates the system is not just fitting weights in isolation but actually changing
robot behaviour.

---

## B6 — MSE Loss (`fig8_mse_loss.png`)

**What it shows:** Preference learner MSE (rating prediction error) over episodes
for all 5 profiles. Shaded bands show seed variance.

**Key insights:**
- All profiles spike at ep 0–1 (large initial error when ŵ is far from w*) then
  drop sharply and flatten below 0.02 by episode 5
- After ep 5 all profiles are essentially indistinguishable in loss — MSE converges
  even for presentation-focused, even though weight distance does not
- This is an important nuance: the learner is fitting the rating model accurately,
  but accurate rating prediction does not guarantee weight recovery for skewed profiles
- Presentation-focused (green) has the highest initial spike (~0.15) and slowest
  initial drop, consistent with its harder learning problem

**Paper use:** Demonstrates the learning engine is working correctly for all profiles.
The MSE convergence despite weight non-convergence for presentation-focused is a
subtle but important point — the rating signal alone is insufficient to fully
discriminate highly skewed profiles under the default schedule.

---

## B7 — Translator Parameters (`fig8_translator_params.png`)

**What it shows:** Evolution of the learnable translator φ parameters over episodes
— how the inner loop adapts its mapping from preference weights to MPC cost parameters.

**Key insights:**
- Translator parameters evolve in response to the outer loop's weight updates
- Confirms the inner loop is actively learning, not static
- Supports the "inner loop improves convergence stability" argument from the
  outer-only baseline comparison

**Paper use:** Evidence that the inner loop is contributing. Pairs with the baseline
comparison (Outer Only vs Full System) discussion.

---

## B8 — MPC Trajectories (`fig8_trajectories.png`)

**What it shows:** Example xy trajectories for medication delivery (left) and meal
preparation (right). Start = green circle, End = red square. Obstacles shown as
grey circles.

**Key insights:**
- Medication delivery: multi-waypoint path from home through the hospital environment
  to the patient — MPC is doing genuine obstacle-aware trajectory shaping, not
  straight-line moves
- Meal preparation: longer, more lateral trajectory with a different spatial structure
  — navigates through the kitchen/prep area before delivering
- The structural difference between the two trajectories explains the battery and
  efficiency differences seen in B9: meal prep covers more ground (higher battery,
  lower path efficiency)
- The smooth curved paths (not jerky) confirm the MPC is working well — no cold-start
  failures visible in trajectory quality

**Paper use:** Visual proof-of-concept that the robot is executing meaningful,
obstacle-aware behaviour. The trajectory quality also implicitly demonstrates MPC
reliability (no jagged paths from solver failures).

---

## B9 — Battery & Efficiency (`fig8_battery_efficiency.png`)

**What it shows:** Battery usage (%) and path efficiency for medication vs meal tasks,
broken down by profile.

**Key insights:**
- Medication: consistently ~65% battery, ~0.70 path efficiency across all profiles
- Meal preparation: consistently ~95% battery, ~0.40 path efficiency across all profiles
- These differences are driven by task structure (distance, waypoints), not by
  the preference profiles — all 5 profiles show the same pattern
- Path efficiency = direct distance / actual path length; meal prep's lower efficiency
  reflects the more complex multi-stage route through the kitchen

**Paper use:** Shows the system operates consistently across profiles for the same
task type — the robot's physical behaviour is governed by the task structure, while
preference learning governs *how* it executes (speed, approach, proximity), not
*what* route it takes. Clean separation of concerns.

---

## BL — Baseline Comparison (`fig8_baselines.png`)

**What it shows:** Best distance achieved and convergence rate for Full System vs
4 baselines (Uniform no-learn, Random Plan, Outer Only, Bandit).

**Key insights:**
- Full System: best_d ~0.076, 84% convergence — dominates on both metrics
- Uniform (no learning): best_d ~0.30, 0% convergence — confirms learning is necessary
- Outer Only: best_d ~0.082, 80% convergence — competitive on best_d but unreliable;
  inner loop's contribution is convergence *stability* not just distance
- Bandit: 10% convergence despite reasonable best_d — noisy exploration can get lucky
  but is not reliable; justifies gradient descent over bandit methods
- Random Plan: 0% convergence — unstructured task ordering gives no consistent signal

**Paper use:** Main baseline comparison figure. The key message is not just that Full
System achieves lower distance, but that it achieves it *reliably* (convergence rate).

---

## AB / AC — Ablation Study (`fig8_ablations.png`, `fig8_ablation_curves.png`)

**What they show:** Best distance and convergence curves for 5 ablation conditions
(Crisp, No Decay, Med Only, Meal Only, Finite Diff) vs Full System.

**Key insights:**
- Med Only is the critical finding: best_d ~0.177, only 8% convergence, huge variance
  — medication tasks alone are insufficient to learn all 5 preference dimensions
- Meal Only performs much better (~0.090, 76%) — meal prep tasks cover more of the
  preference space, particularly approach and proximity
- Crisp (no fuzzy), No Decay (no LR decay), Finite Diff: all ~0.08, ~80% — nearly
  identical to Full System, confirming these components add robustness/stability
  rather than being strictly necessary for convergence
- Ablation curves confirm Med Only never settles — oscillates with huge spread for
  all 60 episodes; everything else converges by ep 20–30

**Paper use:** Shows which components are load-bearing (dual task types) vs which
add robustness (fuzzy state, LR decay, sensitivity-based gradients). The Med Only
result is the most important ablation finding.

---

## T1 — Master Results Table (`table_results.tex`)

LaTeX table ready for direct insertion. Columns: Profile, threshold, Conv. Ep.,
Best d ± std, Final d, Task Rate, Conv. fraction, Dominant dimension identified.

**Key numbers:**
- Speed: conv ep 3, best_d 0.074 ± 0.011, 5/5 converged
- Safety: conv ep 5, best_d 0.067 ± 0.014, 5/5 converged
- Comfort: conv ep 4, best_d 0.061 ± 0.012, 5/5 converged
- Energy: conv ep 3, best_d 0.019 ± 0.003, 5/5 converged (best performer)
- Presentation: conv ep —, best_d 0.174 ± 0.011, 0/5 converged (default params)

All profiles correctly identify the dominant preference dimension (Dom. = ✓) —
even presentation-focused gets the direction right, just can't fully close the gap.
