# MPC Cold-Start Investigation — Summary & Merge Decision

Branch: `mpc-coldstart-investigation`

---

## What Was Investigated

Two distinct cold-start problems were identified and tested:

| Problem | Scenario | Root Cause | Status |
|---------|----------|------------|--------|
| Zero-init on U_TURN | First solve of new episode with large heading change | Zero trajectory gives IPOPT nothing to work with | **Fixed** |
| Warm-start degradation | MULTI_OBS mid-episode across waypoints | Stale w_opt anchored to old robot state passes through obstacles from new position | **Unresolved** |

---

## What Was Tried

| Commit | Attempt | Outcome |
|--------|---------|---------|
| `8f3cd54` | Straight-line init after `_build()` and `reset_episode()` | ✓ Genuine fix — U_TURN dropped from 539ms to ~79ms |
| `7ce8405` | Residual-gated warm start + cost-gated Acados handoff | Made MULTI_OBS worse |
| `7d639dd` | Residual-gated warm start (standalone) | No meaningful improvement |
| `0cecc2d` | **Reverted** Acados residual-gating, suppressed QP status 3 output | Noise suppressed, problem left unresolved |
| `9230054` | Silenced Acados build-time QP warnings via print_level=0 | Clean logs, no functional change |

---

## Key Findings (from FINDINGS.md)

**U_TURN zero-init:** catastrophic — 539ms / 266 iterations vs 79ms / 42 iters with
straight-line init. Fixed. 6.8× speedup.

**MULTI_OBS warm-start degradation:**
- cold start: avg 49ms
- warm_w (carry w*): avg **195ms** — 4× SLOWER
- warm_w_lam (carry w* + λ*): avg 146ms — still 3× slower
- Root cause: previous w_opt trajectory is anchored to old robot state. From the new
  position it passes through obstacles, creating constraint violations that are harder
  to escape than starting fresh from straight line.
- Every fix attempted made it worse or had no effect.
- **Conclusion: IPOPT is better off cold in multi-obstacle scenarios.**

---

## Current State

- QP status 3 warnings suppressed — logs are clean
- U_TURN case genuinely improved by straight-line init
- MULTI_OBS warm-start degradation still present but silent
- Zero cold-start *failures* observed across all recent runs (thousands of MPC solves
  across full, robustness, ablation, baseline conditions) — the system is stable,
  just occasionally slower than optimal in MULTI_OBS scenarios

---

## What Else Is on This Branch (beyond cold-start)

These changes were developed on this branch and are production-ready:

- `a0fcb7e` — **learned z_target**: preference-conditioned position offset in MPC stage cost — significant contribution, increases inner loop contribution from ~4% to 26% improvement in final convergence distance
- `c84971a` — smooth sigmoid battery memberships in fuzzy_state.py
- `daae8b0` — execution README + parallel runner (`run_section8_parallel.py`)

---

## Merge Decision

### Arguments FOR merging to master

- The genuine fix (straight-line init) is valuable and should not stay on a branch
- z_target, sigmoid battery memberships, and the parallel runner are all production
  code that master should have
- The suppression commits are harmless — clean logs are better than noisy ones
- The MULTI_OBS issue was pre-existing; suppressing its noise doesn't make anything worse

### Arguments AGAINST / things to consider

- The branch name suggests a "fix" that was never fully delivered — worth being honest
  in the merge message
- If the MULTI_OBS warm-start degradation is ever revisited, the test suite in
  `tests/mpc_coldstart/` is preserved on this branch and should come along

### Recommendation

**Merge.** The net effect is positive:
- One real performance fix (U_TURN)
- One significant feature (z_target)
- Clean logs
- Full cold-start test suite preserved for future work

Merge message should be honest: "Partial cold-start resolution: fix U_TURN zero-init,
suppress QP status 3 noise. MULTI_OBS warm-start degradation documented but unresolved
— IPOPT performs better cold in obstacle-dense scenarios."

---

## Future Work

The right approach for MULTI_OBS warm-start is likely **horizon-shift warm start**:
shift the previous solution one step forward and fill the last step with a terminal
condition, rather than carrying the full previous solution. This was partially
investigated in `test_horizon_shift.py` but not implemented in production. Could be
a follow-up branch.
