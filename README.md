# Hospital Robot System

**Numerical Implementation of:**

> *Mission-Aligned and Safe Scientifically Informed Deterministic Policy Reinforcement Learning — Part 2: Algorithms and Numerical Simulation*
> Vyacheslav Kungurtsev, Monicah Naibei, Haozhe Tian, Akhil Anand, Sebastien Gros, Homayoun Hamedmoghadam — 2025

---

## What This Is

A simulation framework for autonomous hospital service robots that learn individual patient preferences over repeated task episodes. The system demonstrates that **deterministic, interpretable parameter updates** can achieve adaptive behavior comparable to reinforcement learning — critical for safety-critical healthcare applications where black-box policies are unacceptable.

The robot learns to ask: *"Does this patient want speed, safety, energy efficiency, gentle proximity, or precise positioning?"* and adapts its behaviour accordingly across episodes.

---

## Core Innovation

Three ideas working together:

1. **Hierarchical control** — symbolic task planning guides continuous MPC execution through an explicit translation layer, making each level independently interpretable.

2. **Preference learning with terminal-target adaptation** — the active outer loop learns patient preference weights `w` from episode ratings; MPC `Q/R` translator parameters `φ` remain fixed, while the terminal target `z_target` is adjusted from `1 - μ(goal_location)` after applying a location-specific observation offset.

3. **Deterministic execution** — same parameters produce same trajectories, enabling formal safety verification and reproducible experiments.

---

## Architecture

```
                    ┌─────────────────────────────┐
  Patient Profile   │   LAYER 1: PDDL Task Planner │   ENHSP-opt over PDDL
  (hidden w*)  ──►  │   (unified_planning/*.pddl)  │   symbolic task models
                    └──────────────┬──────────────┘
                                   │  action sequence
                    ┌──────────────▼──────────────┐
  Translator φ ──►  │   LearnableTranslator        │   w → MPC cost matrices
                    │   (core/learning/)           │
                    └──────────────┬──────────────┘
                                   │  Q, R, horizon
                    ┌──────────────▼──────────────┐
                    │   LAYER 2: Navigation Stack  │   A* grid plan → waypoints
                    │   (core/planning/)           │
                    └──────────────┬──────────────┘
                                   │  waypoints
                    ┌──────────────▼──────────────┐
                    │   HybridMPC                  │   Acados (control)
                    │   (core/execution/)          │
                    └──────────────┬──────────────┘
                                   │  u*
                    ┌──────────────▼──────────────┐
                    │   LAYER 3: MuJoCo Env        │   6-DOF physics simulation
                    │   (core/environment/)        │
                    └──────────────┬──────────────┘
                                   │  features f ∈ [0,1]⁵
                    ┌──────────────▼──────────────┐
                    │   Preference Learner         │   Projected gradient descent
                    │   (core/learning/)           │   on simplex → w update
                    └─────────────────────────────┘
```

**Episode cycle:** Plan → Translate → Navigate → Execute → Measure → Learn → Repeat

---

## Preference Dimensions

Every episode produces a 5-dimensional feature vector; patients rate each dimension independently:

| Dimension | Measures | Preferred Value |
|-----------|----------|-----------------|
| **Time** | Episode duration | Low (fast) |
| **Safety** | Min distance to obstacles/patient | Low (high margin) |
| **Battery** | Total energy consumed | Low (efficient) |
| **Proximity** | Movement comfort near patient | Low (gentle) |
| **Approach** | Final positioning precision | Low (accurate) |

The robot maintains an estimate `w_hat` on the 5-simplex and updates it after every episode via projected gradient descent toward the patient's true profile `w*`.

---

## Hospital Tasks

### Medication Delivery *(fully implemented)*

Robot navigates from home → pharmacy → (optional supply) → patient bedside.

- 10 locations, 2 pharmacy options, 2 supply depots, 2 charging stations
- Location-specific risk, congestion, and stock tracking
- 5 preference dimensions learned over episodes

### Meal Preparation *(fully implemented)*

Robot prepares and delivers a patient meal with three complexity paths:

| Path | Steps | Time | Approach Quality |
|------|-------|------|-----------------|
| Sandwich | collect → assemble → deliver | Fast | Poor |
| Soup | collect → chop → cook → deliver | Medium | Medium |
| Full Meal | collect → chop → cook → plate → deliver | Slow | Excellent |

The task planner chooses the path based on learned preferences: a presentation-focused patient will learn to receive full meals; a speed-oriented patient gets sandwiches.

### Patient Entertainment *(planned)*

Activity selection, setup, interactive engagement, and mood adaptation.

---

## Patient Profiles

Five predefined archetypes for experiments (hidden ground-truth `w*`):

| Profile | Time | Safety | Battery | Proximity | Approach |
|---------|------|--------|---------|-----------|----------|
| speed_oriented | 0.50 | 0.12 | 0.14 | 0.14 | 0.10 |
| safety_first | 0.10 | 0.50 | 0.15 | 0.15 | 0.10 |
| comfort_focused | 0.15 | 0.15 | 0.10 | 0.40 | 0.20 |
| energy_conscious | 0.15 | 0.15 | 0.45 | 0.15 | 0.10 |
| presentation_focused | 0.05 | 0.10 | 0.05 | 0.20 | 0.60 |

---

## Repository Structure

```
hospital-robot-system/
│
├── core/                       # Reusable framework (task-independent)
│   ├── execution/              # HybridMPC — formulation, IFT engine, Acados solver
│   ├── planning/               # A* spatial planner + fuzzy state bridge
│   ├── learning/               # Preference learner + LearnableTranslator
│   ├── task_planning/          # Shared task-state mixins and PDDL engine selection
│   └── environment/            # MuJoCo hospital simulation
│
├── tasks/                      # Task-specific implementations
│   ├── medication_delivery/    # PDDL-aligned actions, state machine, reward engine
│   └── meal_preparation/       # PDDL-aligned actions, meal profiles, state machine
│
├── integration/                # Full system (system.py, episode_runner.py, metrics.py)
│
├── tests/                      # Experimental validation (Sections 7 & 8)
│   ├── profile_validation/     # Per-profile integration tests + unified runner
│   ├── results/                # Experiment outputs (section7, section8)
│   ├── generate_section7_figures.py
│   ├── generate_section8_figures.py
│   └── run_section8_experiments.py
│
├── environment.yml             # Conda environment spec
├── ARCHITECTURE.md             # Deep technical design notes
└── SETUP.md                    # Installation instructions
```

---

## Installation

```bash
# 1. Install Acados (MPC solver) — see SETUP.md
# 2. Create Python environment
conda env create -f environment.yml
conda activate mlc-stack

# 3. Run the full experiment suite
python tests/run_section8_experiments.py

# 4. Generate paper figures
python tests/generate_section7_figures.py
python tests/generate_section8_figures.py
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Preference convergence | 15–20 episodes to `‖w_hat − w*‖ < 0.05` |
| Learning rate | η = 0.03 with decay |
| Control frequency | 5 Hz (0.2 s timesteps) |
| Tracking error at convergence | < 1.5 m |
| Episode duration (medication) | ~45 seconds |
| MPC solver | Acados SQP-RTI (1–5 ms per step) |

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Optimization (MPC) | Acados SQP-RTI + CasADi symbolic math |
| Physics simulation | MuJoCo 3.3.5 (6-DOF) |
| Learning | NumPy projected gradient descent |
| Task planning | ENHSP-opt via Unified Planning/PDDL; direct waypoint references for MPC |
| Language | Python 3.11 |

---

## Citation

```bibtex
@article{kungurtsev2025mission,
  author  = {Kungurtsev, Vyacheslav and Naibei, Monicah and Tian, Haozhe and
             Anand, Akhil and Gros, Sebastien and Hamedmoghadam, Homayoun},
  title   = {Mission-Aligned and Safe Scientifically Informed Deterministic
             Policy Reinforcement Learning Part 2: Algorithms and Numerical Simulation},
  year    = {2025}
}
```

---

**Authors:** Monicah Naibei, Vyacheslav Kungurtsev, Haozhe Tian, Akhil Anand, Sebastien Gros, Homayoun Hamedmoghadam
**Institutions:**  CTU Prague · Imperial College London · NTNU · UC San Diego
**Contact:** cheronai37@gmail.com
# hospital
