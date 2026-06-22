#!/usr/bin/env python3
"""
MPC Cold-Start Investigation — Horizon-Shift Warm Start
========================================================

FINDING from test_episode_reset.py:
    MULTI_OBS: warm_w averages 195ms vs cold's 48ms — warm start is 4x SLOWER.

WHY: After each MPC step the robot moves to a new position. The previous w_opt
trajectory is "anchored" to the old state. When reused as-is, it often passes
through obstacles from the new robot position → constraint violations → many
iterations to escape.

THE FIX: Horizon-shift warm start (standard real-time MPC technique).
Instead of reusing w* directly, shift it by one timestep:

    x_warm[k] = x*[k+1]         for k = 0..N-1
    x_warm[N] = x*[N] + Δ       (extrapolate or hold)
    u_warm[k] = u*[k+1]         for k = 0..N-2
    u_warm[N-1] = u*[N-1]       (hold last control)

This produces a trajectory that starts at the NEW robot position (because
x*[1] is where the MPC predicted the robot would be after step 0), which
is close to the actual new position in a well-tracking system.

Compare:
    raw_warm    — current production behaviour (copy w* as-is)
    shifted_warm — proposed fix (horizon-shift before reuse)
    cold        — baseline (straight-line, no prior solution)

Usage:
    python tests/mpc_coldstart/test_horizon_shift.py
    python tests/mpc_coldstart/test_horizon_shift.py --scenario MULTI_OBS U_TURN
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
    ALL_SCENARIOS, BLOCKED, CLEAR, HORIZON, DT, MULTI_OBS, U_TURN,
    NX, NU, Q_DEFAULT, R_DEFAULT,
)
from core.execution.formulation import SharedMPCFormulation


# =============================================================================
# NLP with shift-aware warm-start support
# =============================================================================

class ShiftNLP:
    def __init__(self, n_obstacles: int = 3, horizon: int = HORIZON, dt: float = DT):
        self.N   = horizon
        self.dt  = dt
        self.nx  = NX
        self.nu  = NU
        self.n_obs = n_obstacles
        self.n_x_vars = self.nx * (self.N + 1)
        self.n_u_vars = self.nu * self.N

        X   = ca.MX.sym("X",   self.nx, self.N + 1)
        U   = ca.MX.sym("U",   self.nu, self.N)
        S   = ca.MX.sym("S",   n_obstacles, self.N + 1)
        Q_d = ca.MX.sym("Q_d", self.nx)
        R_d = ca.MX.sym("R_d", self.nu)
        xi  = ca.MX.sym("xi",  self.nx)
        xr  = ca.MX.sym("xr",  self.nx)
        obs = ca.MX.sym("obs", n_obstacles * 3)
        sw  = ca.MX.sym("sw",  1)

        cost = 0
        for k in range(self.N):
            xe = X[:, k] - xr
            cost += ca.mtimes([xe.T, ca.diag(Q_d), xe])
            cost += ca.mtimes([U[:, k].T, ca.diag(R_d), U[:, k]])
            for i in range(n_obstacles):
                cost += sw * S[i, k] + sw * 0.1 * S[i, k] ** 2
        xeN = X[:, self.N] - xr
        cost += SharedMPCFormulation.TERMINAL_COST_MULTIPLIER * ca.mtimes(
            [xeN.T, ca.diag(Q_d), xeN]
        )
        for i in range(n_obstacles): cost += sw * S[i, self.N]

        g, lbg, ubg = [], [], []
        g.append(X[:, 0] - xi);  lbg += [0.]*self.nx; ubg += [0.]*self.nx
        for k in range(self.N):
            xn = SharedMPCFormulation.discrete_dynamics(X[:, k], U[:, k], self.dt)
            g.append(X[:, k+1] - xn); lbg += [0.]*self.nx; ubg += [0.]*self.nx
        for k in range(self.N+1):
            for i in range(n_obstacles):
                ox=obs[i*3]; oy=obs[i*3+1]; r=obs[i*3+2]
                d2 = (X[0,k]-ox)**2+(X[1,k]-oy)**2
                g.append(r**2-d2-S[i,k]); lbg.append(-ca.inf); ubg.append(0.)

        w_s = ca.vertcat(X.reshape((-1,1)), U.reshape((-1,1)), S.reshape((-1,1)))
        p_s = ca.vertcat(Q_d, R_d, xi, xr, obs, sw)

        lbw, ubw = [], []
        for _ in range(self.N+1): lbw+=SharedMPCFormulation.x_min.tolist(); ubw+=SharedMPCFormulation.x_max.tolist()
        for _ in range(self.N):   lbw+=SharedMPCFormulation.u_min.tolist(); ubw+=SharedMPCFormulation.u_max.tolist()
        for _ in range((self.N+1)*n_obstacles): lbw.append(0.); ubw.append(1e6)

        self.lbw=np.array(lbw); self.ubw=np.array(ubw)
        self.lbg=np.array(lbg); self.ubg=np.array(ubg)
        self.n_w = w_s.shape[0]

        nlp  = {"x": w_s, "f": cost, "g": ca.vertcat(*g), "p": p_s}
        opts = {
            "ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": 0,
            "ipopt.max_iter": 300, "ipopt.warm_start_init_point": "yes",
            "ipopt.tol": 1e-4,
        }
        self.solver = ca.nlpsol("shift_nlp", "ipopt", nlp, opts)

    def _pack(self, xi, xr, obstacles):
        DUMMY = [(50.,50.),(60.,50.),(50.,60.)]
        of = []
        for i in range(self.n_obs):
            if i<len(obstacles): o=obstacles[i]; of+=[o["x"],o["y"],o["radius"]]
            else: dx,dy=DUMMY[i%3]; of+=[dx,dy,0.1]
        return np.concatenate([Q_DEFAULT, R_DEFAULT, xi, xr, np.array(of), [50000.]])

    def _w0_straight(self, xi, xr):
        w0 = np.zeros(self.n_w)
        for k in range(self.N+1):
            a = k/self.N
            w0[k*self.nx:(k+1)*self.nx] = xi*(1-a)+xr*a
        return w0

    def shift_warm_start(self, w_prev: np.ndarray, x_new: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        """
        Horizon-shift warm start.

        Shifts the trajectory by one step so it starts at the new robot position:
            x_warm[k] = x*[k+1]     k = 0..N-1
            x_warm[N] = x*[N] + (x_ref - x*[N]) * dt  (small step toward goal)
            u_warm[k] = u*[k+1]     k = 0..N-2
            u_warm[N-1] = u*[N-1]   (hold last control)
            slacks: shift by one step, append 0 at end

        This ensures the warm-started trajectory starts from (approximately)
        the new robot state, keeping it feasible.
        """
        nx, nu, N = self.nx, self.nu, self.N
        n_obs = self.n_obs

        # Unpack previous solution
        X_prev = w_prev[:nx*(N+1)].reshape((N+1, nx))
        U_prev = w_prev[nx*(N+1):nx*(N+1)+nu*N].reshape((N, nu))
        S_prev = w_prev[nx*(N+1)+nu*N:].reshape((N+1, n_obs))

        # Shift states
        X_new = np.zeros_like(X_prev)
        X_new[:N] = X_prev[1:]                    # x[k] ← x*[k+1]
        # Terminal: small step toward x_ref
        step = (x_ref - X_prev[N]) * self.dt
        X_new[N] = X_prev[N] + np.clip(step, -0.5, 0.5)

        # Override x[0] with actual new robot state (enforce initial condition)
        X_new[0] = x_new

        # Shift controls
        U_new = np.zeros_like(U_prev)
        U_new[:N-1] = U_prev[1:]                  # u[k] ← u*[k+1]
        U_new[N-1] = U_prev[N-1]                  # hold last

        # Shift slacks
        S_new = np.zeros_like(S_prev)
        S_new[:N] = S_prev[1:]
        S_new[N] = S_prev[N]

        return np.concatenate([X_new.flatten(), U_new.flatten(), S_new.flatten()])

    def _solve(self, xi, xr, obstacles, w0):
        p = self._pack(xi, xr, obstacles)
        t0 = time.perf_counter()
        try:
            sol = self.solver(x0=w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg, p=p)
            success = self.solver.stats()["success"]
            iters   = self.solver.stats().get("iter_count", -1)
        except Exception as e:
            return {"success": False, "ms": 0, "iters": -1, "cost": np.inf, "w_opt": None}
        ms = (time.perf_counter()-t0)*1000
        if not success:
            return {"success": False, "ms": ms, "iters": iters, "cost": np.inf, "w_opt": None}
        w_opt = np.array(sol["x"]).flatten()
        return {"success": True, "ms": ms, "iters": iters, "cost": float(sol["f"]), "w_opt": w_opt}


# =============================================================================
# Episode simulation comparing raw_warm vs shifted_warm vs cold
# =============================================================================

def simulate(nlp: ShiftNLP, scenario: Dict, n_steps: int = 8):
    xi0 = scenario["x_init"].copy()
    xr  = scenario["x_ref"].copy()
    obs = scenario["obstacles"]

    waypoints = [xi0 + (xr - xi0) * (s / n_steps) for s in range(n_steps + 1)]

    modes = {
        "cold":         None,
        "raw_warm":     None,   # stores previous w_opt
        "shifted_warm": None,   # stores shifted w_opt
    }
    records = {m: [] for m in modes}

    for step in range(n_steps):
        xi_step = waypoints[step]

        # --- cold ---
        w0 = nlp._w0_straight(xi_step, xr)
        r_cold = nlp._solve(xi_step, xr, obs, w0)

        # --- raw_warm ---
        w0_raw = modes["raw_warm"] if modes["raw_warm"] is not None else nlp._w0_straight(xi_step, xr)
        r_raw = nlp._solve(xi_step, xr, obs, w0_raw)
        if r_raw["success"]:
            modes["raw_warm"] = r_raw["w_opt"]

        # --- shifted_warm ---
        if modes["shifted_warm"] is not None:
            w0_shift = nlp.shift_warm_start(modes["shifted_warm"], xi_step, xr)
        else:
            w0_shift = nlp._w0_straight(xi_step, xr)
        r_shift = nlp._solve(xi_step, xr, obs, w0_shift)
        if r_shift["success"]:
            modes["shifted_warm"] = r_shift["w_opt"]

        records["cold"].append({"step": step, **{k: r_cold[k] for k in ("success","ms","iters","cost")}})
        records["raw_warm"].append({"step": step, **{k: r_raw[k] for k in ("success","ms","iters","cost")}})
        records["shifted_warm"].append({"step": step, **{k: r_shift[k] for k in ("success","ms","iters","cost")}})

    return records


def print_results(scenario: Dict, records: Dict):
    name  = scenario["name"]
    n_obs = len(scenario["obstacles"])
    print(f"\n{'='*80}")
    print(f"  HORIZON-SHIFT TEST: {name}  ({n_obs} obstacles)")
    print(f"  {scenario.get('note','')}")
    print(f"{'='*80}")
    print(f"  {'Step':>4}  {'cold ms':>9}  {'cold it':>7}  {'raw_warm ms':>11}  {'raw_it':>6}  {'shift ms':>9}  {'shift_it':>8}")
    print(f"  {'-'*72}")
    for s in range(len(records["cold"])):
        rc = records["cold"][s]
        rr = records["raw_warm"][s]
        rs = records["shifted_warm"][s]
        def fmt(r): return f"{r['ms']:>9.1f}  {r['iters']:>7}" if r["success"] else f"{'FAIL':>9}  {'--':>7}"
        print(f"  {s:>4}  {fmt(rc)}  {fmt(rr)}  {fmt(rs)}")

    print(f"\n  {'MEAN':>4}", end="")
    for mode, label in [("cold","cold"), ("raw_warm","raw_warm"), ("shifted_warm","shifted_warm")]:
        ok = [r for r in records[mode] if r["success"]]
        if ok:
            print(f"  mean={np.mean([r['ms'] for r in ok]):.1f}ms / {np.mean([r['iters'] for r in ok]):.1f}iters", end="")
        else:
            print(f"  ALL FAIL", end="")
    print()

    # Speedup
    cold_ms    = np.mean([r["ms"] for r in records["cold"] if r["success"]] or [1])
    raw_ms     = np.mean([r["ms"] for r in records["raw_warm"] if r["success"]] or [1])
    shift_ms   = np.mean([r["ms"] for r in records["shifted_warm"] if r["success"]] or [1])
    print(f"\n  raw_warm vs cold:     {cold_ms/raw_ms:.2f}x  ({'faster' if raw_ms < cold_ms else 'SLOWER'})")
    print(f"  shifted_warm vs cold: {cold_ms/shift_ms:.2f}x  ({'faster' if shift_ms < cold_ms else 'SLOWER'})")
    print(f"  shifted vs raw_warm:  {raw_ms/shift_ms:.2f}x  ({'faster' if shift_ms < raw_ms else 'SLOWER'})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", nargs="+", default=["CLEAR","BLOCKED","MULTI_OBS","U_TURN"])
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()

    scenario_map = {s["name"]: s for s in ALL_SCENARIOS}
    print("=" * 80)
    print("  MPC COLD-START: HORIZON-SHIFT WARM START TEST")
    print("=" * 80)

    nlp_cache = {}
    for sname in args.scenario:
        if sname not in scenario_map: continue
        scenario = scenario_map[sname]
        n_obs = max(len(scenario["obstacles"]), 1)
        if n_obs not in nlp_cache:
            print(f"\n  [Building NLP for {n_obs} obstacles...]")
            nlp_cache[n_obs] = ShiftNLP(n_obstacles=n_obs)
        records = simulate(nlp_cache[n_obs], scenario, args.steps)
        print_results(scenario, records)

    print(f"\n  ✓ Done\n")


if __name__ == "__main__":
    main()
