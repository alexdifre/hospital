#!/usr/bin/env python3
"""
MPC Cold-Start Investigation — Episode Reset Behaviour
=======================================================

The production system calls `mpc.reset_episode()` between episodes, which
clears w_warm and lam_warm.  This test measures the cost of that reset:

    1. Run N consecutive solves WITH warm-starting (simulates mid-episode)
    2. Run the SAME sequence but clear warm-start before each solve (simulates
       start-of-episode cold start)
    3. Run with a partial warm-start (carry w* but NOT lam* from prior episode)

Also measures:
    - How many IPOPT iterations the first solve after reset takes vs the 2nd+
    - Whether lam_warm makes a material difference vs w_warm alone
    - How solve time evolves across a simulated 8-step segment

Usage:
    python tests/mpc_coldstart/test_episode_reset.py
    python tests/mpc_coldstart/test_episode_reset.py --scenario BLOCKED
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import casadi as ca
from configs import (
    ALL_SCENARIOS, BLOCKED, CLEAR, HORIZON, DT, NX, NU, Q_DEFAULT, R_DEFAULT,
)
from core.execution.formulation import SharedMPCFormulation


# Reuse the IsolatedNLP from the other test (inline a minimal version here
# to keep the files self-contained)

class EpisodeNLP:
    """
    NLP with explicit warm-start control.
    Exposes solve_warm() / solve_cold() / solve_partial() entry points.
    """

    def __init__(self, n_obstacles: int = 3, horizon: int = HORIZON, dt: float = DT):
        self.N = horizon
        self.dt = dt
        self.nx = NX
        self.nu = NU
        self.n_obs = n_obstacles
        self.n_x_vars = self.nx * (self.N + 1)

        X = ca.MX.sym("X", self.nx, self.N + 1)
        U = ca.MX.sym("U", self.nu, self.N)
        S = ca.MX.sym("S", n_obstacles, self.N + 1)

        Q_d  = ca.MX.sym("Q_diag", self.nx)
        R_d  = ca.MX.sym("R_diag", self.nu)
        xi_s = ca.MX.sym("xi",  self.nx)
        xr_s = ca.MX.sym("xr",  self.nx)
        obs  = ca.MX.sym("obs", n_obstacles * 3)
        sw   = ca.MX.sym("sw",  1)

        cost = 0
        for k in range(self.N):
            xe = X[:, k] - xr_s
            cost += ca.mtimes([xe.T, ca.diag(Q_d), xe])
            cost += ca.mtimes([U[:, k].T, ca.diag(R_d), U[:, k]])
            for i in range(n_obstacles):
                cost += sw * S[i, k] + sw * 0.1 * S[i, k] ** 2
        xe_N = X[:, self.N] - xr_s
        cost += SharedMPCFormulation.TERMINAL_COST_MULTIPLIER * ca.mtimes(
            [xe_N.T, ca.diag(Q_d), xe_N]
        )
        for i in range(n_obstacles):
            cost += sw * S[i, self.N]

        g, lbg, ubg = [], [], []
        g.append(X[:, 0] - xi_s)
        lbg += [0.0] * self.nx; ubg += [0.0] * self.nx

        for k in range(self.N):
            xn = SharedMPCFormulation.discrete_dynamics(X[:, k], U[:, k], self.dt)
            g.append(X[:, k + 1] - xn)
            lbg += [0.0] * self.nx; ubg += [0.0] * self.nx

        for k in range(self.N + 1):
            for i in range(n_obstacles):
                ox = obs[i*3]; oy = obs[i*3+1]; r = obs[i*3+2]
                d2 = (X[0,k]-ox)**2 + (X[1,k]-oy)**2
                g.append(r**2 - d2 - S[i,k])
                lbg.append(-ca.inf); ubg.append(0.0)

        w_sym = ca.vertcat(X.reshape((-1,1)), U.reshape((-1,1)), S.reshape((-1,1)))
        p_sym = ca.vertcat(Q_d, R_d, xi_s, xr_s, obs, sw)

        lbw, ubw = [], []
        for _ in range(self.N+1):
            lbw += SharedMPCFormulation.x_min.tolist()
            ubw += SharedMPCFormulation.x_max.tolist()
        for _ in range(self.N):
            lbw += SharedMPCFormulation.u_min.tolist()
            ubw += SharedMPCFormulation.u_max.tolist()
        for _ in range((self.N+1)*n_obstacles):
            lbw.append(0.0); ubw.append(1e6)

        self.lbw = np.array(lbw); self.ubw = np.array(ubw)
        self.lbg = np.array(lbg); self.ubg = np.array(ubg)
        self.n_w = w_sym.shape[0]

        nlp  = {"x": w_sym, "f": cost, "g": ca.vertcat(*g), "p": p_sym}
        opts = {
            "ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": 0,
            "ipopt.max_iter": 300, "ipopt.warm_start_init_point": "yes",
            "ipopt.tol": 1e-4,
        }
        self.solver = ca.nlpsol("ep_nlp", "ipopt", nlp, opts)

    def _pack(self, xi, xr, obstacles, Q=Q_DEFAULT, R=R_DEFAULT, sw=50000.0):
        DUMMY = [(50.,50.),(60.,50.),(50.,60.)]
        of = []
        for i in range(self.n_obs):
            if i < len(obstacles):
                o = obstacles[i]; of.extend([o["x"],o["y"],o["radius"]])
            else:
                dx,dy = DUMMY[i%3]; of.extend([dx,dy,0.1])
        return np.concatenate([Q, R, xi, xr, np.array(of), [sw]])

    def _w0_straight(self, xi, xr):
        w0 = np.zeros(self.n_w)
        for k in range(self.N+1):
            a = k/self.N
            w0[k*self.nx:(k+1)*self.nx] = xi*(1-a) + xr*a
        return w0

    def _w0_slack_aware(self, xi, xr, obstacles):
        w0 = self._w0_straight(xi, xr)
        ns = self.nx*(self.N+1) + self.nu*self.N
        for k in range(self.N+1):
            xk = w0[k*self.nx:(k+1)*self.nx]
            for i,o in enumerate(obstacles):
                d2 = (xk[0]-o["x"])**2 + (xk[1]-o["y"])**2
                viol = o["radius"]**2 - d2
                idx = ns + k*self.n_obs + i
                if idx < self.n_w:
                    w0[idx] = max(0.0, viol*1.1)
        return w0

    def _solve(self, xi, xr, obstacles, w0, lam0=None):
        p = self._pack(xi, xr, obstacles)
        kwargs = dict(x0=w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg, p=p)
        if lam0 is not None:
            kwargs["lam_g0"] = lam0
        t0 = time.perf_counter()
        try:
            sol = self.solver(**kwargs)
            success = self.solver.stats()["success"]
            iters   = self.solver.stats().get("iter_count", -1)
        except Exception as e:
            return {"success": False, "ms": 0, "iters": -1, "cost": np.inf,
                    "w_opt": None, "lam_opt": None, "error": str(e)}
        ms = (time.perf_counter()-t0)*1000
        if not success:
            return {"success": False, "ms": ms, "iters": iters, "cost": np.inf,
                    "w_opt": None, "lam_opt": None}
        return {
            "success": True, "ms": ms, "iters": iters,
            "cost": float(sol["f"]),
            "w_opt": np.array(sol["x"]).flatten(),
            "lam_opt": np.array(sol["lam_g"]).flatten(),
        }

    def solve_cold(self, xi, xr, obstacles, slack_aware=False):
        """First solve with no prior solution."""
        w0 = self._w0_slack_aware(xi, xr, obstacles) if slack_aware else self._w0_straight(xi, xr)
        return self._solve(xi, xr, obstacles, w0)

    def solve_warm(self, xi, xr, obstacles, w_prev, lam_prev=None):
        """Warm-started from previous solution."""
        return self._solve(xi, xr, obstacles, w_prev, lam_prev)


# =============================================================================
# Simulated episode: robot moves one waypoint step at a time
# =============================================================================

def simulate_episode(nlp: EpisodeNLP, scenario: Dict, n_steps: int = 8):
    """
    Simulate a robot navigating along a straight-line path to x_ref.
    At each step the robot advances ~0.3m and re-solves the MPC.

    Returns per-step solve stats for three modes:
        cold          — reset warm-start every step (worst case)
        warm_w        — carry only primal w* (common warm-start)
        warm_w_lam    — carry primal + dual (full warm-start)
        cold_slack    — reset but use slack-aware init
    """
    xi0  = scenario["x_init"].copy()
    xr   = scenario["x_ref"].copy()
    obs  = scenario["obstacles"]

    # Generate waypoints along the straight line
    waypoints = [xi0 + (xr - xi0) * (s / n_steps) for s in range(n_steps + 1)]

    modes = {
        "cold":       {"w": None, "lam": None},
        "warm_w":     {"w": None, "lam": None},
        "warm_w_lam": {"w": None, "lam": None},
        "cold_slack": {"w": None, "lam": None},
    }

    records = {m: [] for m in modes}

    for step in range(n_steps):
        xi_step = waypoints[step]

        for mode in modes:
            cold = mode.startswith("cold")
            slack = mode == "cold_slack"

            if cold:
                r = nlp.solve_cold(xi_step, xr, obs, slack_aware=slack)
            else:
                w_prev   = modes[mode]["w"]
                lam_prev = modes[mode]["lam"] if mode == "warm_w_lam" else None
                if w_prev is None:
                    r = nlp.solve_cold(xi_step, xr, obs)
                else:
                    r = nlp.solve_warm(xi_step, xr, obs, w_prev, lam_prev)

            if r["success"]:
                modes[mode]["w"]   = r["w_opt"]
                modes[mode]["lam"] = r.get("lam_opt")

            records[mode].append({
                "step": step, "success": r["success"],
                "ms": r["ms"], "iters": r["iters"], "cost": r["cost"],
            })

    return records


def print_episode_results(scenario: Dict, records: Dict):
    name  = scenario["name"]
    n_obs = len(scenario["obstacles"])

    print(f"\n{'='*80}")
    print(f"  EPISODE SIMULATION: {name}  ({n_obs} obstacles)")
    print(f"  {scenario.get('note','')}")
    print(f"{'='*80}")

    modes = list(records.keys())
    print(f"  {'Step':>4}  |  ", end="")
    for m in modes:
        print(f"  {m:<14} ms / iters", end="")
    print()
    print(f"  {'-'*78}")

    n_steps = len(list(records.values())[0])
    for s in range(n_steps):
        print(f"  {s:>4}  |  ", end="")
        for m in modes:
            r = records[m][s]
            if r["success"]:
                print(f"  {r['ms']:>8.1f} / {r['iters']:>4}    ", end="")
            else:
                print(f"  {'FAIL':>8}   {'--':>4}    ", end="")
        print()

    print(f"\n  {'MEAN':>4}  |  ", end="")
    for m in modes:
        recs = records[m]
        ok = [r for r in recs if r["success"]]
        if ok:
            mean_ms = np.mean([r["ms"] for r in ok])
            mean_it = np.mean([r["iters"] for r in ok])
            print(f"  {mean_ms:>8.1f} / {mean_it:>4.1f}    ", end="")
        else:
            print(f"  {'ALL FAIL':>8}   {'--':>4}    ", end="")
    print()

    # Speedup summary
    print(f"\n  Speedup vs cold (warm_w vs cold):")
    cold_ok = [r for r in records["cold"] if r["success"]]
    warm_ok = [r for r in records["warm_w"] if r["success"]]
    if cold_ok and warm_ok:
        speedup = np.mean([r["ms"] for r in cold_ok]) / max(np.mean([r["ms"] for r in warm_ok]), 1e-9)
        print(f"    {speedup:.2f}x faster with warm_w")

    slack_ok = [r for r in records["cold_slack"] if r["success"]]
    if cold_ok and slack_ok:
        speedup2 = np.mean([r["ms"] for r in cold_ok]) / max(np.mean([r["ms"] for r in slack_ok]), 1e-9)
        print(f"    {speedup2:.2f}x faster with cold_slack vs plain cold")


def main():
    parser = argparse.ArgumentParser(description="Episode reset warm-start cost analysis")
    parser.add_argument("--scenario", nargs="+", default=["CLEAR", "BLOCKED", "MULTI_OBS"])
    parser.add_argument("--steps", type=int, default=8,
                        help="Number of MPC steps to simulate per episode")
    args = parser.parse_args()

    scenario_map = {s["name"]: s for s in ALL_SCENARIOS}

    print("=" * 80)
    print("  MPC COLD-START: EPISODE RESET — WARM VS COLD COMPARISON")
    print("=" * 80)
    print(f"  Simulated steps per episode: {args.steps}")
    print(f"  Scenarios: {args.scenario}")

    nlp_cache = {}
    for sname in args.scenario:
        if sname not in scenario_map:
            print(f"  WARNING: unknown scenario {sname}, skipping")
            continue
        scenario = scenario_map[sname]
        n_obs = max(len(scenario["obstacles"]), 1)
        if n_obs not in nlp_cache:
            print(f"\n  [Building NLP for {n_obs} obstacles...]")
            nlp_cache[n_obs] = EpisodeNLP(n_obstacles=n_obs)

        nlp     = nlp_cache[n_obs]
        records = simulate_episode(nlp, scenario, args.steps)
        print_episode_results(scenario, records)

    print(f"\n  ✓ Done\n")


if __name__ == "__main__":
    main()
