# Pacman Machine Learning Project

This project combines a classic Pacman game written in C99 with a Python-based machine learning framework for AI training and automated gameplay. The C Pacman game exposes an HTTP API, allowing the Python ML agents to interact with and control the game programmatically.

## Project Structure

- **C Pacman Game:**  
  A minimal-dependency Pacman clone for Windows, macOS, Linux, and WASM.  
  [WASM version](https://floooh.github.io/pacman.c/pacman.html)  
  For implementation details, see comments in the `pacman.c` source file.

- **Python Machine Learning:**  
  Q-Learning and Deep Q-Learning agents implemented in Python, which communicate with the C game via the HTTP API for training and evaluation.

## Features

- HTTP API for programmatic game control (enables AI training)
- Q-Learning and Deep Q-Learning agents
- Cross-platform C game (Windows, macOS, Linux, WASM)
- Easy integration between C and Python via HTTP

## Quick Start

### 1. Build and Run the C Pacman Game with HTTP API

**Dependencies:**  
On Linux, install OpenGL, X11, and ALSA development packages (e.g. `mesa-common-dev`, `libx11-dev`, `libasound2-dev`).

```bash
# Clone repository and initialize submodules (for HTTP library)
git clone https://github.com/floooh/pacman.c
cd pacman.c
git submodule update --init --recursive

# Build with HTTP API enabled
cmake -B build -DPACMAN_ENABLE_API=ON
cmake --build build

# Run Pacman with HTTP API enabled
./build/pacman --api --ghosts=1
```

On macOS/Linux, the executable is `./build/pacman`.  
On Windows, it is `Debug/pacman.exe`.

Alternatively, with Nix:
```bash
nix run github:floooh/pacman.c
```

### 2. Run Python Machine Learning Agents

The Python ML scripts are in the `qlearning` and `deep-qlearning` subdirectories.  
They require [Poetry](https://python-poetry.org/) for dependency management.

#### Q-Learning Agent

```bash
poetry install
poetry run python qlearning/train.py --episodes 5000
```

#### Deep Q-Learning Agent

```bash
poetry install
poetry run python deep-qlearning/train.py --episodes 5000
```

## Example Results

### Q-Learning (after 5000 episodes)
```
Window          Avg Score   Avg Dots  Max Score   Max Dots
------------------------------------------------------------
1-500               699.1       63.0       2160        127
...
4501-5000          1238.3      102.0       3420        214

=== BEST PERFORMANCES ===
Top 10 by dots eaten:
  Ep 3969: 218 dots, score 3340
  ...
Episodes with 200+ dots: 7
Episodes with 180+ dots: 28
```

### Deep Q-Learning (after 5000 episodes)
```
============================================================
DQN AGENT STATS
============================================================
Device: cpu
Epsilon: 0.0500
Learning Rate: 1.00e-05
Steps: 2000
Network Updates: 1554734
Buffer Size: 100000
============================================================

============================================================
TRAINING COMPLETE
============================================================
Episodes:      2000
Avg Score:     1981.7
Avg Dots:      163.5
Avg Deaths:    2.97 / 3
Total Wins:    48 (2.4%)
Best Score:    4490 (Ep 133)
Best Dots:     244
============================================================
```

## Related Projects

- Original version: https://github.com/floooh/pacman.c
- Zig version: https://github.com/floooh/pacman.zig

---
This project is ideal for experimenting with reinforcement learning and game AI in a real-time environment.
