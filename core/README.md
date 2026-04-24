# core/

Task-independent framework components. Everything here is reusable across different hospital tasks (medication delivery, meal preparation, etc.).

## Sub-packages

| Directory | Purpose |
|-----------|---------|
| [execution/](execution/) | MPC controllers — trajectory tracking and sensitivity computation |
| [planning/](planning/) | Spatial A* path planner and fuzzy state estimation |
| [learning/](learning/) | Patient preference learning engine |
| [environment/](environment/) | MuJoCo hospital ward simulation |

## Design Philosophy

Each layer in the hierarchy exposes a clean interface to the layer above it:

- **Environment** exposes a step/reset API (MuJoCo physics)
- **Execution** takes waypoints and MPC parameters, returns control + sensitivities
- **Planning** takes a start pose and goal, returns waypoints
- **Learning** takes episode features and patient ratings, returns updated preference weights

Task-specific logic (which actions exist, what the reward signal means, how preferences map to MPC cost matrices) lives in `tasks/` rather than here.
