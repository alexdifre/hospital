#!/usr/bin/env python3
"""
MPC Cold-Start — Fix Validation
================================

Runs all three fixes from FINDINGS.md against the scenarios that exposed
each problem and directly compares against the original (broken) behaviour.

Fix 1 — Residual-gated warm start     (MULTI_OBS degradation)
Fix 2 — Cost-gated Acados→CasADi handoff  (bad-solution poisoning)
Fix 3 — Straight-line reset()         (U_TURN first-solve penalty)

Usage:
    python tests/mpc_coldstart/test_fixes.py
    python tests/mpc_coldstart/test_fixes.py --fix 1 2 3
    python tests/mpc_coldstart/test_fixes.py --threshold 1.0 2.0 3.0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from configs import (
    ALL_SCENARIOS, CLEAR, BLOCKED, MULTI_OBS, U_TURN, LONG_RANGE,
    HORIZON, DT, NX, NU,
)
from patched_solvers import (
    ResidualGatedNLP,
    CostGatedNLP,
    StraightLineResetNLP,
    _build_nlp, _pack_params, _w0_straight,
)
import casadi as ca


# =============================================================================
# Baseline NLP — mirrors the original production behaviour exactly
# (raw warm-start, no gating, zero reset)
# =============================================================================

class BaselineNLP:
    """
    Original production behaviour:
        - Always reuse w_opt from last solve as warm-start
        - reset() zeros everything (mirrors AcadosSolver.reset())
        - No residual or cost gating
    """

    def __init__(self, n_obstacles: int = 3, horizon: int = HORIZON, dt: float = DT):
        self.N   = horizon
        self.dt  = dt
        self.nx  = NX
        self.nu  = NU
        self.n_obs = n_obstacles

        (self.solver, self.lbw, self.ubw, self.lbg, self.ubg,
         self.n_w, self.n_x_vars) = _build_nlp(n_obstacles, horizon, dt)

        self.w_warm:   Optional[np.ndarray] = None
        self.lam_warm: Optional[np.ndarray] = None

    def reset(self):
        """Fix 3 baseline: zero everything (current production behaviour)."""
        self.w_warm   = np.zeros(self.n_w)    # zero reset — the bug
        self.lam_warm = None

    def reset_none(self):
        """Alternate baseline: w_warm=None (falls back to straight-line in solve)."""
        self.w_warm   = None
        self.lam_warm = None

    def solve(self, xi, xr, obstacles, use_warm=True) -> Dict:
        p  = _pack_params(self.n_obs, xi, xr, obstacles)
        w0 = self.w_warm if (use_warm and self.w_warm is not None) else \
             _w0_straight(self.n_w, self.nx, self.N, xi, xr)
        reason = "warm" if (use_warm and self.w_warm is not None) else "cold_straight"

        kwargs = dict(x0=w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg, p=p)
        if self.lam_warm is not None and use_warm:
            kwargs["lam_g0"] = self.lam_warm

        t0 = time.perf_counter()
        try:
            sol     = self.solver(**kwargs)
            success = self.solver.stats()["success"]
            iters   = self.solver.stats().get("iter_count", -1)
        except Exception as e:
            return {"success": False, "ms": 0, "iters": -1, "cost": np.inf,
                    "w_opt": None, "reason": reason, "error": str(e)}

        ms = (time.perf_counter() - t0) * 1000
        if not success:
            return {"success": False, "ms": ms, "iters": iters,
                    "cost": np.inf, "w_opt": None, "reason": reason}

        w_opt = np.array(sol["x"]).flatten()
        self.w_warm   = w_opt          # always store — no gating
        self.lam_warm = np.array(sol["lam_g"]).flatten()
        return {"success": True, "ms": ms, "iters": iters,
                "cost": float(sol["f"]), "w_opt": w_opt, "reason": reason}


# =============================================================================
# Helper: run an episode (8 waypoint steps) through any solver
# =============================================================================

def _waypoints(xi, xr, n_steps):
    return [xi + (xr - xi) * (s / n_steps) for s in range(n_steps + 1)]


def run_episode(solver, scenario: Dict, n_steps: int = 8, **solve_kwargs) -> List[Dict]:
    xi0 = scenario["x_init"].copy()
    xr  = scenario["x_ref"].copy()
    obs = scenario["obstacles"]
    wps = _waypoints(xi0, xr, n_steps)
    records = []
    for step in range(n_steps):
        r = solver.solve(wps[step], xr, obs, **solve_kwargs)
        records.append({"step": step, **{k: r[k] for k in ("success","ms","iters","cost") if k in r},
                        "reason": r.get("reason","")})
    return records


def _stats(records):
    ok = [r for r in records if r["success"]]
    if not ok:
        return {"success_rate": 0, "mean_ms": np.inf, "mean_iters": np.inf}
    return {
        "success_rate": len(ok)/len(records),
        "mean_ms":      np.mean([r["ms"] for r in ok]),
        "mean_iters":   np.mean([r["iters"] for r in ok]),
        "max_ms":       np.max([r["ms"] for r in ok]),
    }


def _bar(t_baseline, t_fixed):
    ratio = t_baseline / max(t_fixed, 0.1)
    tag   = f"{ratio:.2f}x faster" if t_fixed < t_baseline else f"{1/ratio:.2f}x SLOWER"
    return tag


# =============================================================================
# Fix 1 test — residual gating on MULTI_OBS
# =============================================================================

def test_fix1(n_steps: int = 8, thresholds=(0.5, 1.0, 1.5, 2.5)):
    print("\n" + "=" * 80)
    print("  FIX 1 — RESIDUAL-GATED WARM START")
    print("  Target: MULTI_OBS (warm_w was 4× slower than cold)")
    print("=" * 80)

    scenario = MULTI_OBS
    n_obs = len(scenario["obstacles"])

    # Build baseline once
    baseline = BaselineNLP(n_obs)

    # Baseline: raw warm (production behaviour)
    bl_records = run_episode(baseline, scenario, n_steps)
    bl = _stats(bl_records)
    print(f"\n  BASELINE (raw warm, no gating): {bl['mean_ms']:.1f}ms / {bl['mean_iters']:.1f}iters  max={bl['max_ms']:.1f}ms")

    # Baseline: plain cold (for reference)
    baseline2 = BaselineNLP(n_obs)
    cold_records = run_episode(baseline2, scenario, n_steps, use_warm=False)
    cold = _stats(cold_records)
    print(f"  BASELINE (cold straight-line):  {cold['mean_ms']:.1f}ms / {cold['mean_iters']:.1f}iters  max={cold['max_ms']:.1f}ms")

    print(f"\n  {'Threshold':>10}  {'mean ms':>9}  {'mean iters':>10}  {'max ms':>8}  {'vs baseline':>14}  {'vs cold':>12}")
    print(f"  {'-'*72}")

    best_threshold = None
    best_ms = np.inf
    for thr in thresholds:
        fixed = ResidualGatedNLP(n_obs, dist_threshold=thr)
        records = run_episode(fixed, scenario, n_steps)
        s = _stats(records)
        vs_bl   = _bar(bl['mean_ms'],   s['mean_ms'])
        vs_cold = _bar(cold['mean_ms'], s['mean_ms'])
        print(f"  {thr:>10.1f}m  {s['mean_ms']:>9.1f}  {s['mean_iters']:>10.1f}  {s.get('max_ms',0):>8.1f}  {vs_bl:>14}  {vs_cold:>12}")
        if s['mean_ms'] < best_ms:
            best_ms = s['mean_ms']
            best_threshold = thr

    print(f"\n  ✓ Best threshold: {best_threshold}m  →  {best_ms:.1f}ms mean")

    # Show per-step breakdown for best threshold
    print(f"\n  Per-step detail (threshold={best_threshold}m):")
    fixed_best = ResidualGatedNLP(n_obs, dist_threshold=best_threshold)
    detail = run_episode(fixed_best, scenario, n_steps)
    baseline3 = BaselineNLP(n_obs)
    bl_detail  = run_episode(baseline3, scenario, n_steps)
    print(f"  {'Step':>4}  {'baseline ms':>11}  {'fixed ms':>9}  {'reason'}")
    for i, (b, f) in enumerate(zip(bl_detail, detail)):
        flag = " ← DISCARDED" if "discarded" in f["reason"] else ""
        print(f"  {i:>4}  {b['ms']:>11.1f}  {f['ms']:>9.1f}  {f['reason']}{flag}")


# =============================================================================
# Fix 2 test — cost-gated Acados→CasADi handoff
# =============================================================================

def test_fix2():
    print("\n" + "=" * 80)
    print("  FIX 2 — COST-GATED ACADOS→CASADI WARM-START HANDOFF")
    print("  Target: prevents bad Acados solution poisoning CasADi warm-start")
    print("=" * 80)

    # Use U_TURN — large state change, worst for bad warm-start
    scenario = U_TURN
    n_obs    = len(scenario["obstacles"])
    xi       = scenario["x_init"].copy()
    xr       = scenario["x_ref"].copy()
    obs      = scenario["obstacles"]

    # Scenario: Acados fails and returns a zero-trajectory "solution" with cost=1e8
    print("\n  Simulating: Acados returns a bad solution (cost=1e8, zero trajectory)")

    # WITHOUT Fix 2 (baseline): bad solution gets stored as w_warm
    no_fix = BaselineNLP(n_obs)
    no_fix.w_warm   = np.zeros(no_fix.n_w)    # inject bad zero warm-start
    no_fix.lam_warm = None

    r_nf = no_fix.solve(xi, xr, obs)
    print(f"\n  Without Fix 2 (poisoned warm-start):  {r_nf['ms']:.1f}ms / {r_nf['iters']} iters  success={r_nf['success']}")

    # WITH Fix 2: bad solution is rejected, falls back to straight-line
    with_fix = CostGatedNLP(n_obs, cost_threshold=1e6)
    with_fix.inject_bad_warmstart(xi, xr, cost=1e8)    # simulates Acados returning cost=1e8
    r_wf = with_fix.solve(xi, xr, obs)
    print(f"  With Fix 2    (cost-gated):            {r_wf['ms']:.1f}ms / {r_wf['iters']} iters  success={r_wf['success']}  reason={r_wf['reason']}")

    # Also show clean run (no poisoning) for reference
    clean = CostGatedNLP(n_obs, cost_threshold=1e6)
    r_cl = clean.solve(xi, xr, obs)
    print(f"  Clean (no injection):                  {r_cl['ms']:.1f}ms / {r_cl['iters']} iters  success={r_cl['success']}")

    print(f"\n  Fix 2 correctly {'REJECTED' if 'discarded' in r_wf['reason'] else 'ACCEPTED (check threshold)'} the bad warm-start")
    print(f"  Solve time with Fix 2:  {r_wf['ms']:.1f}ms vs without: {r_nf['ms']:.1f}ms")


# =============================================================================
# Fix 3 test — straight-line reset() on U_TURN
# =============================================================================

def test_fix3():
    print("\n" + "=" * 80)
    print("  FIX 3 — STRAIGHT-LINE reset() INSTEAD OF ZERO RESET")
    print("  Target: U_TURN first-solve penalty (zero reset: 539ms, straight-line: 79ms)")
    print("=" * 80)

    scenario = U_TURN
    n_obs    = len(scenario["obstacles"])
    xi       = scenario["x_init"].copy()
    xr       = scenario["x_ref"].copy()
    obs      = scenario["obstacles"]

    N_REPS = 5
    print(f"\n  Running {N_REPS} first-solve comparisons after episode reset...\n")

    print(f"  {'Run':>4}  {'zero_reset ms':>14}  {'straight_reset ms':>18}  {'speedup':>10}")
    print(f"  {'-'*54}")

    times_zero, times_straight = [], []

    for rep in range(N_REPS):
        # Original: zero reset
        b = BaselineNLP(n_obs)
        b.reset()                       # sets w_warm = zeros
        r_zero = b.solve(xi, xr, obs)
        times_zero.append(r_zero["ms"])

        # Fix 3: straight-line reset
        f = StraightLineResetNLP(n_obs)
        f.reset(xi, xr)                 # pre-populates w_warm with straight-line
        r_sl = f.solve(xi, xr, obs)
        times_straight.append(r_sl["ms"])

        ratio = r_zero["ms"] / max(r_sl["ms"], 0.1)
        print(f"  {rep:>4}  {r_zero['ms']:>14.1f}  {r_sl['ms']:>18.1f}  {ratio:>9.2f}x")

    print(f"\n  Mean zero_reset:       {np.mean(times_zero):.1f}ms")
    print(f"  Mean straight_reset:   {np.mean(times_straight):.1f}ms")
    print(f"  Speedup:               {np.mean(times_zero)/max(np.mean(times_straight),0.1):.2f}x")


# =============================================================================
# Combined test — all fixes together on all problem scenarios
# =============================================================================

def test_all_fixes_combined(n_steps: int = 8):
    print("\n" + "=" * 80)
    print("  ALL FIXES COMBINED — vs BASELINE vs COLD")
    print("=" * 80)

    scenarios_to_test = [CLEAR, BLOCKED, MULTI_OBS, U_TURN]

    print(f"\n  {'Scenario':<14}  {'baseline':>12}  {'cold':>10}  {'all_fixes':>12}  {'vs_baseline':>14}  {'vs_cold':>12}")
    print(f"  {'-'*80}")

    for scenario in scenarios_to_test:
        n_obs = max(len(scenario["obstacles"]), 1)
        name  = scenario["name"]

        # Baseline
        bl_solver = BaselineNLP(n_obs)
        bl_recs   = run_episode(bl_solver, scenario, n_steps)
        bl        = _stats(bl_recs)

        # Cold (reference)
        cold_solver = BaselineNLP(n_obs)
        cold_recs   = run_episode(cold_solver, scenario, n_steps, use_warm=False)
        cold        = _stats(cold_recs)

        # All fixes combined
        fixed_solver = StraightLineResetNLP(n_obs, dist_threshold=1.5, cost_threshold=1e6)
        # Simulate episode reset with xi/xr known (Fix 3)
        fixed_solver.reset(scenario["x_init"], scenario["x_ref"])
        fixed_recs = run_episode(fixed_solver, scenario, n_steps)
        fixed      = _stats(fixed_recs)

        vs_bl   = _bar(bl['mean_ms'],   fixed['mean_ms'])
        vs_cold = _bar(cold['mean_ms'], fixed['mean_ms'])

        print(
            f"  {name:<14}  {bl['mean_ms']:>10.1f}ms  {cold['mean_ms']:>8.1f}ms"
            f"  {fixed['mean_ms']:>10.1f}ms  {vs_bl:>14}  {vs_cold:>12}"
        )

    print()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Validate cold-start fixes")
    parser.add_argument("--fix", nargs="+", type=int, default=[1, 2, 3],
                        help="Which fixes to test (1, 2, 3)")
    parser.add_argument("--threshold", nargs="+", type=float,
                        default=[0.5, 1.0, 1.5, 2.0, 3.0],
                        help="Distance thresholds to sweep for Fix 1")
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()

    if 1 in args.fix:
        test_fix1(n_steps=args.steps, thresholds=args.threshold)

    if 2 in args.fix:
        test_fix2()

    if 3 in args.fix:
        test_fix3()

    if set(args.fix) >= {1, 2, 3}:
        test_all_fixes_combined(n_steps=args.steps)

    print("  ✓ All fix tests complete\n")


if __name__ == "__main__":
    main()
