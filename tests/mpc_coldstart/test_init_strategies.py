#!/usr/bin/env python3
"""
MPC Cold-Start Investigation — Initialisation Strategy Comparison
=================================================================

Runs each scenario under four initialisation strategies and records:
    - solve time (ms)
    - success / failure
    - IPOPT iterations
    - final cost
    - initial feasibility gap (how badly the straight-line init violates constraints)

No existing source files are modified.  This script is read-only w.r.t. core/.

Usage:
    python tests/mpc_coldstart/test_init_strategies.py
    python tests/mpc_coldstart/test_init_strategies.py --scenario BLOCKED
    python tests/mpc_coldstart/test_init_strategies.py --strategy zero straight_line
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import casadi as ca
from configs import (
    ALL_SCENARIOS, BLOCKED, CLEAR, HORIZON, DT, LONG_RANGE, MULTI_OBS,
    SIDE, STRATEGIES, U_TURN, NX, NU, Q_DEFAULT, R_DEFAULT,
)
from core.execution.formulation import SharedMPCFormulation


# =============================================================================
# Standalone NLP (mirrors CasADiSensitivityComputer._build_nlp_solver exactly)
# so we can inject custom initialisations without touching source)
# =============================================================================

class IsolatedNLP:
    """
    Standalone IPOPT NLP matching the production CasADi formulation exactly.
    Exposes the solver so we can inject arbitrary w0.
    """

    def __init__(self, n_obstacles: int = 3, horizon: int = HORIZON, dt: float = DT):
        self.N = horizon
        self.dt = dt
        self.nx = NX
        self.nu = NU
        self.n_obstacles = n_obstacles

        X = ca.MX.sym("X", self.nx, self.N + 1)
        U = ca.MX.sym("U", self.nu, self.N)
        S = ca.MX.sym("S", n_obstacles, self.N + 1)

        Q_diag   = ca.MX.sym("Q_diag", self.nx)
        R_diag   = ca.MX.sym("R_diag", self.nu)
        x_init_s = ca.MX.sym("x_init", self.nx)
        x_ref_s  = ca.MX.sym("x_ref",  self.nx)
        obs_p    = ca.MX.sym("obs", n_obstacles * 3)
        sw       = ca.MX.sym("sw", 1)

        cost = 0
        for k in range(self.N):
            xe = X[:, k] - x_ref_s
            cost += ca.mtimes([xe.T, ca.diag(Q_diag), xe])
            cost += ca.mtimes([U[:, k].T, ca.diag(R_diag), U[:, k]])
            for i in range(n_obstacles):
                cost += sw * S[i, k]
                cost += sw * 0.1 * S[i, k] ** 2
        xe_N = X[:, self.N] - x_ref_s
        cost += SharedMPCFormulation.TERMINAL_COST_MULTIPLIER * ca.mtimes(
            [xe_N.T, ca.diag(Q_diag), xe_N]
        )
        for i in range(n_obstacles):
            cost += sw * S[i, self.N]

        g, lbg, ubg = [], [], []
        g.append(X[:, 0] - x_init_s)
        lbg.extend([0.0] * self.nx); ubg.extend([0.0] * self.nx)

        for k in range(self.N):
            x_next = SharedMPCFormulation.discrete_dynamics(X[:, k], U[:, k], self.dt)
            g.append(X[:, k + 1] - x_next)
            lbg.extend([0.0] * self.nx); ubg.extend([0.0] * self.nx)

        for k in range(self.N + 1):
            for i in range(n_obstacles):
                ox = obs_p[i * 3]; oy = obs_p[i * 3 + 1]; r = obs_p[i * 3 + 2]
                dist_sq = (X[0, k] - ox) ** 2 + (X[1, k] - oy) ** 2
                g.append(r ** 2 - dist_sq - S[i, k])
                lbg.append(-ca.inf); ubg.append(0.0)

        w_sym = ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)), S.reshape((-1, 1)))
        p_sym = ca.vertcat(Q_diag, R_diag, x_init_s, x_ref_s, obs_p, sw)

        lbw, ubw = [], []
        for _ in range(self.N + 1):
            lbw.extend(SharedMPCFormulation.x_min.tolist())
            ubw.extend(SharedMPCFormulation.x_max.tolist())
        for _ in range(self.N):
            lbw.extend(SharedMPCFormulation.u_min.tolist())
            ubw.extend(SharedMPCFormulation.u_max.tolist())
        for _ in range((self.N + 1) * n_obstacles):
            lbw.append(0.0); ubw.append(1e6)

        self.lbw = np.array(lbw)
        self.ubw = np.array(ubw)
        self.lbg = np.array(lbg)
        self.ubg = np.array(ubg)

        self.n_w = w_sym.shape[0]
        self.n_x_vars = self.nx * (self.N + 1)

        nlp  = {"x": w_sym, "f": cost, "g": ca.vertcat(*g), "p": p_sym}
        opts = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
            "ipopt.max_iter": 300,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.tol": 1e-4,
            "ipopt.output_file": "",           # suppress file output
        }
        self.solver = ca.nlpsol("nlp", "ipopt", nlp, opts)
        self.n_obstacles = n_obstacles

    def pack_params(
        self,
        Q_diag: np.ndarray,
        R_diag: np.ndarray,
        x_init: np.ndarray,
        x_ref:  np.ndarray,
        obstacles: List[Dict],
        slack_weight: float = 50000.0,
    ) -> np.ndarray:
        DUMMY = [(50.0, 50.0), (60.0, 50.0), (50.0, 60.0)]
        obs_flat = []
        for i in range(self.n_obstacles):
            if i < len(obstacles):
                o = obstacles[i]
                obs_flat.extend([o["x"], o["y"], o["radius"]])
            else:
                dx, dy = DUMMY[i % len(DUMMY)]
                obs_flat.extend([dx, dy, 0.1])
        return np.concatenate([Q_diag, R_diag, x_init, x_ref, np.array(obs_flat), [slack_weight]])

    # ------------------------------------------------------------------
    # Initial guess builders
    # ------------------------------------------------------------------

    def w0_zero(self) -> np.ndarray:
        """Everything set to zero."""
        return np.zeros(self.n_w)

    def w0_straight_line(self, x_init: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        """States: linear interp; controls: zero; slacks: zero."""
        w0 = np.zeros(self.n_w)
        for k in range(self.N + 1):
            alpha = k / self.N
            x_k = x_init * (1 - alpha) + x_ref * alpha
            w0[k * self.nx : (k + 1) * self.nx] = x_k
        return w0

    def w0_straight_slack(
        self, x_init: np.ndarray, x_ref: np.ndarray, obstacles: List[Dict]
    ) -> np.ndarray:
        """
        States: linear interp; controls: zero.
        Slacks: initialised to max(0, r² - dist²) along the straight line.

        This makes the inequality constraints feasible from the start —
        the key hypothesis for fixing cold-start convergence failure in BLOCKED.
        """
        w0 = self.w0_straight_line(x_init, x_ref)

        n_x_vars = self.nx * (self.N + 1)
        n_u_vars = self.nu * self.N
        slack_start = n_x_vars + n_u_vars

        for k in range(self.N + 1):
            x_k = w0[k * self.nx : (k + 1) * self.nx]
            for i, obs in enumerate(obstacles):
                ox, oy, r = obs["x"], obs["y"], obs["radius"]
                dist_sq = (x_k[0] - ox) ** 2 + (x_k[1] - oy) ** 2
                violation = r ** 2 - dist_sq   # > 0 → inside obstacle
                idx = slack_start + k * self.n_obstacles + i
                if idx < len(w0):
                    w0[idx] = max(0.0, violation * 1.1)  # 10% buffer

        return w0

    # ------------------------------------------------------------------
    # Core solve
    # ------------------------------------------------------------------

    def solve(
        self,
        x_init: np.ndarray,
        x_ref:  np.ndarray,
        obstacles: List[Dict],
        strategy: str,
        Q_diag: np.ndarray = Q_DEFAULT,
        R_diag: np.ndarray = R_DEFAULT,
        w_prev: Optional[np.ndarray] = None,
    ) -> Dict:
        p = self.pack_params(Q_diag, R_diag, x_init, x_ref, obstacles)

        if strategy == "zero":
            w0 = self.w0_zero()
        elif strategy == "straight_line":
            w0 = self.w0_straight_line(x_init, x_ref)
        elif strategy == "straight_slack":
            w0 = self.w0_straight_slack(x_init, x_ref, obstacles)
        elif strategy == "prev_sol":
            w0 = w_prev if w_prev is not None else self.w0_straight_line(x_init, x_ref)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        t0 = time.perf_counter()
        try:
            sol = self.solver(x0=w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg, p=p)
            success = self.solver.stats()["success"]
            stats   = self.solver.stats()
        except Exception as exc:
            return {"success": False, "error": str(exc), "solve_ms": 0.0,
                    "cost": np.inf, "iter_count": -1, "w_opt": None}

        solve_ms = (time.perf_counter() - t0) * 1000.0

        if not success:
            return {"success": False, "solve_ms": solve_ms, "cost": np.inf,
                    "iter_count": stats.get("iter_count", -1), "w_opt": None}

        w_opt = np.array(sol["x"]).flatten()
        u0    = w_opt[self.n_x_vars : self.n_x_vars + self.nu]

        return {
            "success":   True,
            "solve_ms":  solve_ms,
            "cost":      float(sol["f"]),
            "iter_count": stats.get("iter_count", -1),
            "u0":        u0,
            "w_opt":     w_opt,
        }

    # ------------------------------------------------------------------
    # Feasibility check — how badly does w0 violate obstacle constraints?
    # ------------------------------------------------------------------

    def measure_init_infeasibility(
        self,
        x_init: np.ndarray,
        x_ref:  np.ndarray,
        obstacles: List[Dict],
        strategy: str,
    ) -> Dict:
        if strategy == "zero":
            w0 = self.w0_zero()
        elif strategy in ("straight_line", "prev_sol"):
            w0 = self.w0_straight_line(x_init, x_ref)
        elif strategy == "straight_slack":
            w0 = self.w0_straight_slack(x_init, x_ref, obstacles)
        else:
            w0 = self.w0_straight_line(x_init, x_ref)

        max_violation = 0.0
        total_violation = 0.0
        n_violated = 0

        for k in range(self.N + 1):
            x_k = w0[k * self.nx : (k + 1) * self.nx]
            for obs in obstacles:
                ox, oy, r = obs["x"], obs["y"], obs["radius"]
                dist_sq = (x_k[0] - ox) ** 2 + (x_k[1] - oy) ** 2
                violation = r ** 2 - dist_sq   # positive = inside obstacle
                if violation > 0:
                    max_violation = max(max_violation, violation)
                    total_violation += violation
                    n_violated += 1

        return {
            "max_violation": max_violation,
            "total_violation": total_violation,
            "n_violated_steps": n_violated,
        }


# =============================================================================
# Main experiment loop
# =============================================================================

def run_scenario(
    nlp: IsolatedNLP,
    scenario: Dict,
    strategies: List[str],
    n_repeats: int = 3,
) -> Dict:
    """Run all strategies on one scenario, return results dict."""

    name  = scenario["name"]
    xi    = scenario["x_init"]
    xr    = scenario["x_ref"]
    obs   = scenario["obstacles"]

    results = {}
    prev_w  = None   # tracks previous solution for 'prev_sol' strategy

    for strat in strategies:
        times, costs, iters, successes = [], [], [], []

        for rep in range(n_repeats):
            # For prev_sol on first rep, fall back to straight_line
            r = nlp.solve(xi, xr, obs, strategy=strat, w_prev=prev_w)
            times.append(r["solve_ms"])
            costs.append(r["cost"])
            iters.append(r["iter_count"])
            successes.append(r["success"])

            if strat == "prev_sol" and r.get("w_opt") is not None:
                prev_w = r["w_opt"]   # warm-start next rep from this solution

        infeas = nlp.measure_init_infeasibility(xi, xr, obs, strat)

        results[strat] = {
            "success_rate":  sum(successes) / n_repeats,
            "mean_ms":       np.mean(times),
            "std_ms":        np.std(times),
            "mean_cost":     np.mean([c for c in costs if c < np.inf]) if any(successes) else np.inf,
            "mean_iters":    np.mean(iters),
            "infeasibility": infeas,
            "raw":           {"times": times, "costs": costs, "iters": iters, "successes": successes},
        }

    return results


def print_results(scenario: Dict, results: Dict):
    name = scenario["name"]
    note = scenario.get("note", "")
    n_obs = len(scenario["obstacles"])

    print(f"\n{'='*80}")
    print(f"  SCENARIO: {name}  ({n_obs} obstacles)")
    print(f"  {note}")
    print(f"{'='*80}")
    print(f"  {'Strategy':<18} {'Success':>8} {'Time(ms)':>10} {'±':>6} {'Iters':>7} {'Cost':>12} {'MaxViol':>10}")
    print(f"  {'-'*75}")

    for strat, r in results.items():
        infeas = r["infeasibility"]["max_violation"]
        cost_str = f"{r['mean_cost']:>12.1f}" if r['mean_cost'] < np.inf else "      FAILED"
        print(
            f"  {strat:<18} {r['success_rate']:>7.0%}  {r['mean_ms']:>9.1f}  "
            f"{r['std_ms']:>5.1f}  {r['mean_iters']:>6.1f}  {cost_str}  {infeas:>10.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description="MPC cold-start init strategy test")
    parser.add_argument("--scenario", nargs="+", default=None,
                        help="Scenario names to run (default: all)")
    parser.add_argument("--strategy", nargs="+", default=None,
                        help="Strategies to test (default: all)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Repetitions per (scenario, strategy) cell")
    args = parser.parse_args()

    scenario_map = {s["name"]: s for s in ALL_SCENARIOS}
    if args.scenario:
        selected_scenarios = [scenario_map[n] for n in args.scenario if n in scenario_map]
    else:
        selected_scenarios = ALL_SCENARIOS

    all_strategies = list(STRATEGIES.keys())
    if args.strategy:
        selected_strategies = [s for s in args.strategy if s in all_strategies]
    else:
        selected_strategies = all_strategies

    print("=" * 80)
    print("  MPC COLD-START: INITIALISATION STRATEGY COMPARISON")
    print("=" * 80)
    print(f"  Scenarios:  {[s['name'] for s in selected_scenarios]}")
    print(f"  Strategies: {selected_strategies}")
    print(f"  Repeats:    {args.repeats}")
    print(f"  Horizon:    N={HORIZON}, dt={DT}s")

    # Build one NLP per obstacle count to avoid unnecessary rebuilds
    nlp_cache: Dict[int, IsolatedNLP] = {}

    all_results = {}
    for scenario in selected_scenarios:
        n_obs = max(len(scenario["obstacles"]), 1)   # always include ≥1 dummy
        if n_obs not in nlp_cache:
            print(f"\n  [Building NLP for {n_obs} obstacles...]")
            nlp_cache[n_obs] = IsolatedNLP(n_obstacles=n_obs)

        nlp = nlp_cache[n_obs]
        results = run_scenario(nlp, scenario, selected_strategies, args.repeats)
        print_results(scenario, results)
        all_results[scenario["name"]] = results

    # ── Summary table ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  SUMMARY — COLD vs WARM TIME RATIO")
    print(f"{'='*80}")
    print(f"  {'Scenario':<14}", end="")
    for strat in selected_strategies:
        print(f"  {strat[:12]:>12}", end="")
    print()

    for sname, results in all_results.items():
        print(f"  {sname:<14}", end="")
        for strat in selected_strategies:
            r = results.get(strat, {})
            if r.get("success_rate", 0) > 0:
                print(f"  {r['mean_ms']:>10.1f}ms", end="")
            else:
                print(f"  {'FAIL':>12}", end="")
        print()

    print(f"\n  ✓ Done\n")


if __name__ == "__main__":
    main()
