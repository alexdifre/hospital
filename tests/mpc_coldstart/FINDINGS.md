# MPC Cold-Start Investigation — Findings

Branch: `mpc-coldstart-investigation`
Tests run: `test_init_strategies.py`, `test_episode_reset.py`, `test_horizon_shift.py`

---

## TL;DR

There are **two distinct cold-start problems** with different root causes and different fixes:

| Problem | Where it hurts | Root cause | Proposed fix |
|---------|---------------|------------|--------------|
| **Zero-init on U_TURN** | First solve of new episode w/ large heading change | Zero trajectory gives the solver nothing to work with | Always use straight-line init (already done in prod — but `reset()` must guarantee it) |
| **Warm-start degradation in MULTI_OBS** | Mid-episode across waypoints | Stale `w_opt` from old robot state passes through obstacles from new position | Discard warm-start when `‖x_warm[0] − x_new‖ > δ` |

---

## Test 1 — Initialisation Strategies (`test_init_strategies.py`)

Single solve per scenario, 3 repeats each.
Strategies: `zero`, `straight_line`, `straight_slack`, `prev_sol`

```
Scenario       zero      straight   straight_s   prev_sol
CLEAR          22ms         21ms        21ms        19ms
BLOCKED        33ms         29ms        28ms        24ms
SIDE           31ms         33ms        31ms        23ms
MULTI_OBS      48ms         41ms        41ms        31ms
LONG_RANGE     38ms         42ms        44ms        30ms
U_TURN        539ms         79ms        78ms        35ms
```

### Key findings

**U_TURN with zero init: 539ms / 266 iterations** — catastrophic.
Zero init forces the solver to discover the direction of motion from scratch across a 33m
heading reversal. The straight-line init drops this to **79ms / 42 iters** (6.8× faster).
`prev_sol` is the best at 35ms — confirming warm-start works well for U_TURN.

**Straight-line vs zero**: zero is rarely better. Always prefer straight-line for the
initial cold solve.

**`straight_slack` (violation-aware slack init)**: Negligible improvement over `straight_line`.
The solver handles soft constraints without help — slack infeasibility is not the bottleneck.

---

## Test 2 — Episode Reset (`test_episode_reset.py`)

Simulates 8-step episode (robot advances along path each step).
Modes: `cold`, `warm_w` (carry w*), `warm_w_lam` (carry w* + λ*), `cold_slack`

```
Scenario      cold(avg)  warm_w(avg)  warm_w_lam   cold_slack   note
CLEAR          18.8ms      19.5ms       19.3ms       18.6ms      no difference
BLOCKED        23.2ms      23.3ms       23.2ms       23.1ms      no difference
MULTI_OBS      48.7ms     195.4ms      145.7ms       50.7ms      ← CRITICAL
U_TURN         39.0ms      29.0ms       28.8ms       38.6ms      warm helps here
```

### Key findings

**MULTI_OBS: warm_w averages 195ms vs cold's 49ms — 4× SLOWER.**
At step 3, warm_w takes 468ms / 225 iterations vs cold's 94ms / 65 iters.

Root cause: as the robot moves from waypoint to waypoint, the previous `w_opt` trajectory
is "anchored" to the old robot state. From the new robot position, this trajectory
may now pass through obstacles, creating constraint violations that are far harder to
escape than starting fresh from a straight line.

**`warm_w_lam` (primal + dual) is slightly better than `warm_w` alone for MULTI_OBS**
(146ms vs 195ms) but both are still much worse than cold.

**`cold_slack` ≈ cold**: violation-aware slack init doesn't help — the solver never struggles
with constraint feasibility here, only with the trajectory shape.

---

## Test 3 — Horizon-Shift Warm Start (`test_horizon_shift.py`)

Tests the standard real-time MPC technique: shift `w*` by one timestep before reuse.

```
Scenario    cold(avg)  raw_warm   shifted_warm
CLEAR          18ms       19ms        18ms      — no difference
BLOCKED        23ms       23ms        23ms      — no difference
MULTI_OBS      48ms      193ms       203ms      ← SHIFT MAKES IT WORSE
U_TURN         38ms       29ms        29ms      — same as raw_warm
```

### Key findings

**Horizon-shift does NOT fix the MULTI_OBS problem.** Both raw_warm and shifted_warm
average ~195–203ms vs cold's 48ms. The issue is not the phase alignment of the
trajectory — it is that the warm-started trajectory passes through obstacles regardless
of whether it is shifted. The curved optimal path for MULTI_OBS diverges significantly
from the straight-line path, so any reuse of a prior solution creates a starting point
in a different "basin" than the one the solver would find from a straight-line.

---

## Root Cause Summary

### Problem 1: Zero init on U_TURN (episode start)

Production code uses `straight_line` init when `w_warm is None`. This is correct.
The risk is if `reset()` is called and then the first call does NOT hit the
straight-line path because a stale or failed warm-start is reused.

If Acados produced a poor solution (e.g. wrong direction), reusing that warm-start
is worse than cold. The fix: only reuse warm-starts whose cost is below a sanity
threshold.

### Problem 2: Stale warm-start across waypoints in MULTI_OBS

Production code: `self.w_warm = w_opt` after every solve. In obstacle-dense
environments with long detours, the optimal trajectory for step k is very different
from step k+1. Reusing it locks the solver into a poor basin.

**Proposed fix — residual-based warm-start selector:**

```python
# Before using w_warm, check if robot has moved significantly from the
# state where w_warm was computed:
x_warm_start = w_warm[:nx]   # first state in warm-start trajectory
dist_to_new  = np.linalg.norm(x_warm_start[:2] - x_new[:2])

if dist_to_new > 1.5:   # robot moved >1.5m from warm-start anchor
    w0 = straight_line_init(x_new, x_ref)   # discard stale warm-start
else:
    w0 = w_warm
```

Threshold 1.5m is a starting point — should be tuned to `2 * max(obstacle_radius)`.

---

## Fix Validation Results (`test_fixes.py`)

All three fixes were implemented in `patched_solvers.py` as subclasses
(no production source modified) and validated against the same scenarios.

### Fix 1 — Residual-Gated Warm Start

Threshold sweep on MULTI_OBS (8-step episode):

```
Threshold  mean ms  vs baseline    vs cold
0.5m        52.0ms   2.98× faster   1.26× faster
1.0m        49.6ms   3.13× faster   1.32× faster  ← best
1.5m        52.7ms   2.94× faster   1.24× faster
2.0m        51.0ms   3.04× faster   1.28× faster
3.0m       154.4ms   1.00× (no help — threshold too wide to trigger)
```

**Threshold = 1.0m is optimal.** In MULTI_OBS the robot moves 2.15m per waypoint
step — always above 1.0m — so every step discards the stale warm-start and uses
cold straight-line. The 400ms+ spikes disappear completely. Max solve drops from
399ms → 104ms.

### Fix 2 — Cost-Gated Warm-Start Rejection

Injecting a bad Acados solution (cost=1e8, zero trajectory) and comparing:

```
Without Fix 2 (poisoned warm-start):  556ms / 266 iters
With Fix 2    (cost-gated):            82ms /  42 iters  ← 6.8× faster
Clean (no injection):                  80ms /  42 iters
```

Fix 2 correctly rejects the bad solution. Solve time with the gate is
essentially identical to a clean run (82ms vs 80ms).

### Fix 3 — Straight-Line reset()

U_TURN first-solve after episode reset (5 runs):

```
zero_reset:       608ms mean (range 554–680ms)
straight_reset:    85ms mean (range 80–89ms)
Speedup:           7.1×
```

The straight-line reset completely eliminates the first-solve penalty.

### Combined — All Fixes Together

```
Scenario   baseline    cold   all_fixes    vs baseline     vs cold
CLEAR        22.1ms   20.9ms    20.9ms   1.06× faster   1.00× faster
BLOCKED      25.8ms   29.1ms    26.1ms   ~1.00×         1.11× faster
MULTI_OBS   163.6ms   54.8ms    51.6ms   3.17× faster   1.06× faster  ✓
U_TURN       29.9ms   40.5ms    39.2ms   1.31× SLOWER   1.03× faster
```

**Note on U_TURN combined result**: U_TURN waypoint spacing is 33m/8 ≈ 4.1m per
step, always above the 1.0m threshold — so Fix 1 discards the warm-start at every
step. In this scenario the warm-start actually helps (baseline 29.9ms < cold 40.5ms),
so Fix 1 is too aggressive. The right solution: apply Fix 1 only in multi-obstacle
mode, or tune the threshold to `2 × max(obstacle_radius)` per episode. Fix 3 is the
critical win for U_TURN (first solve: 608ms → 85ms).

---

## Recommended Changes (to implement on this branch, then test before merging)

1. **Warm-start reuse**:
   - Add a `_dist_from_warm` check before using a previous warm-start
   - Discard stale warm-start if robot has moved > threshold from `w_warm[0:2]`
   - Reject failed/nonsense warm-starts if `sol.cost >= 1e6`

2. **`AcadosSolver.solve()`** (lines 326–331):
   - Current straight-line init is correct
   - After rebuild (`_needs_rebuild`), explicitly reset warm-start to straight-line
     (currently `reset()` sets everything to zeros — triggers the zero-init problem
     on the first solve after an obstacle-count change)

---

## Files

| File | Purpose |
|------|---------|
| `configs.py` | 6 canonical scenarios + strategy definitions |
| `test_init_strategies.py` | Single-solve comparison of 4 init strategies |
| `test_episode_reset.py` | 8-step episode simulation: cold vs warm modes |
| `test_horizon_shift.py` | Tests horizon-shift warm start vs raw warm vs cold |
| `FINDINGS.md` | This document |
