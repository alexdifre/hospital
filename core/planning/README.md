# core/planning/

State estimation for bridging continuous robot pose and discrete task logic.

## Files

### `fuzzy_state.py` — Fuzzy State Estimator

Bridges the continuous MuJoCo robot state and the discrete task planner. Without fuzzy membership, the planner would receive crisp "robot is at pharmacy" / "robot is not at pharmacy" signals that produce chattering and abrupt plan changes near location boundaries.

**Three fuzzification layers:**

#### 1. Position → Location Membership
Gaussian membership for each named location:
```
μ_L(x, y) = exp(−d(robot, location)² / (2 σ_L²))
```
Output: `{'pharmacy_north': 0.85, 'supply_A': 0.02, ...}`

#### 2. Battery → {low, medium, high}
Sigmoid membership functions replacing the old trapezoidal model. Smooth, differentiable transitions centred at SoC = 0.3 and 0.7:

```
μ_Low(SoC)  = σ(−10 · (SoC − 0.3))          # 1 below 0.3, 0 above
μ_High(SoC) = σ(+10 · (SoC − 0.7))          # 0 below 0.7, 1 above
μ_Med(SoC)  = max(0, 1 − μ_Low − μ_High)    # peaks near SoC = 0.5
```

This eliminates the flat "plateau" regions of trapezoids and gives a single continuous gradient everywhere, which is more informative when battery state influences task priority.

#### 3. Congestion/Risk → {safe, moderate, hazardous}
Continuous fuzzy risk level. Congestion penalties scale continuously rather than jumping at an if/else boundary.

**Output:** `FuzzyMemberships` dataclass:
- `location_memberships: Dict[str, float]`
- `battery_memberships: Dict[str, float]`
- `risk_level: Dict[str, float]`
- `dominant_location: str`
- `is_at(location, threshold=0.5) → bool`

## How They Work Together

```
MuJoCo state (x, y, θ, vx, vy, ω_z)
        ↓
  FuzzyStateEstimator
        ↓
  FuzzyMemberships (soft location + battery + risk)
        ↓
  Task Planner (A* uses fuzzy cost estimates)
        ↓
  Next action → target location
        ↓
  Direct 21-point waypoint reference
        ↓
  HybridMPC
```
