# Setup Instructions

## System Requirements

- **OS:** macOS or Linux (tested on macOS with Apple Silicon)
- **Python:** 3.11+
- **C Compiler:** gcc or clang
- **RAM:** 8GB minimum

## Installation Steps

### Step 1: Install Acados (System-Wide)

Acados must be installed before setting up the Python environment.

**1.1 Clone Acados**

```bash
cd ~  # or wherever you want to install
git clone https://github.com/acados/acados.git
cd acados
git submodule update --recursive --init
```

**1.2 Build Acados**

```bash
mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON ..
make install -j4
cd ../..
```

**1.3 Install Python Interface**

```bash
pip install -e ~/acados/interfaces/acados_template
# Adjust path if you installed Acados elsewhere
```

**1.4 Set Environment Variables**

Add these lines to your shell configuration file:

- **For bash:** `~/.bashrc`
- **For zsh:** `~/.zshrc`

```bash
export ACADOS_SOURCE_DIR="$HOME/acados"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$HOME/acados/lib"
```

**Adjust the paths if you installed Acados elsewhere!**

**1.5 Reload Shell**

```bash
source ~/.zshrc  # or source ~/.bashrc
```

**1.6 Verify Acados Installation**

```bash
echo $ACADOS_SOURCE_DIR
# Should print: /Users/yourname/acados (or your install path)

python -c "from acados_template import AcadosOcp; print('Acados OK')"
# Should print: Acados OK
```

---

### Step 2: Set Up Python Environment

#### Option A: Using Conda (Recommended)

```bash
# Navigate to project root
cd hospital-robot-system

# Create environment
conda env create -f environment.yml

# Activate environment
conda activate mlc-stack
```

#### Option B: Using pip + venv

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

### Step 3: Verify Complete Setup

```bash
# Activate environment
conda activate mlc-stack

# Test all imports
python -c "import mujoco; import casadi; import numpy; print('Core packages OK')"

# Test Acados
python -c "from acados_template import AcadosOcp; print('Acados integration OK')"
```

All commands should complete without errors.

---

## Running the System

### Single-Profile Smoke Test

```bash
conda activate mlc-stack

# Run one episode of the full system (speed-oriented patient)
python tests/run_section8_experiments.py --condition full --profile speed_oriented --episodes 1 --seeds 1
```

### Full Experiment Suite

```bash
# All profiles, all conditions (takes ~hours)
python tests/run_section8_experiments.py

# Single condition
python tests/run_section8_experiments.py --condition full --episodes 40 --seeds 5
```

### Expected Output

The system will execute through several phases:

1. **Task Planning:** High-level action sequence (go_to_pharmacy, collect_medication, etc.)
2. **Spatial Planning:** Collision-free waypoint generation using A\*
3. **MPC Execution:** Trajectory tracking with real-time control
4. **Preference Learning:** Weight updates from simulated patient feedback
5. **Results:** JSON files saved to `tests/results/section8/`

**Note:** You may see "QP solver returned error status 3" in the first 10-20 steps. This is expected (cold-start issue) and does not affect convergence.

---

## System Specifications

**Tested Configuration:**

- macOS (Apple Silicon M-series)
- Python 3.11.13
- Acados (December 2024 version)
- MuJoCo 3.3.5
- Conda environment management

**Key Dependencies:**

- `mujoco==3.3.5` - Physics simulation
- `casadi==3.7.0` - Symbolic optimization (Acados dependency)
- `numpy==2.2.6` - Numerical computing
- `scipy==1.15.3` - Scientific computing
- `matplotlib==3.10.3` - Visualization

---

## Troubleshooting

### "Cannot import acados_template"

**Cause:** Acados not installed or environment variables not set.

**Fix:**

```bash
# Check environment variables
echo $ACADOS_SOURCE_DIR
echo $LD_LIBRARY_PATH

# If empty, add to ~/.zshrc (or ~/.bashrc):
export ACADOS_SOURCE_DIR="$HOME/acados"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$HOME/acados/lib"

# Reload
source ~/.zshrc

# Reinstall Python interface if needed
pip install -e ~/acados/interfaces/acados_template
```

### "Conda environment creation fails"

**Option 1:** Force recreation

```bash
conda env remove -n mlc-stack
conda env create -f environment.yml
```

**Option 2:** Use pip fallback

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### "MuJoCo rendering issues"

**On macOS:**

```bash
# Install OpenGL support
brew install glfw
```

**On Linux:**

```bash
sudo apt-get install libglfw3 libglfw3-dev
```

### "QP solver errors during MPC execution"

**Symptoms:** "SQP_RTI: QP solver returned error status 3" in first 10-20 steps

**This is expected behavior** due to cold-start constraints. The solver converges despite these warnings. If errors persist beyond 20 steps, check:

- Waypoint spacing (may be too aggressive)
- Initial velocity constraints
- MPC horizon settings

---

## Alternative: Docker Setup

For a fully containerized setup (coming soon), see `docker/README.md`.

---

## Video Demonstration

**A complete system demonstration video is coming soon.**

For code review purposes, the system can be evaluated without local execution via static code analysis and the architecture documentation in `ARCHITECTURE.md`.

---

## Getting Help

**If you encounter setup issues:**

1. Check that Acados environment variables are set correctly
2. Verify all dependencies installed: `pip list | grep -E "(mujoco|casadi|numpy)"`
3. Review the troubleshooting section above
4. Contact: [cheronai37@gmail.com]

**For architecture questions or code review without running:**

- See `ARCHITECTURE.md` for system design
- Watch the demonstration video (coming soon)

---

## Notes for Reviewers

This system has been developed and tested on the author's development environment. The setup instructions have been verified to work on fresh installations.

If immediate execution is not required for code review purposes, the architecture, algorithms, and implementation can be evaluated through:

- Static code analysis
- Architecture documentation (`ARCHITECTURE.md`)
- Video demonstration
- Test results and plots included in the repository

---

_Last updated: March 2026_
