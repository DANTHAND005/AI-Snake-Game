# AI Snake — Deep Q-Learning
**Daniel Than | Northern Kentucky University**
Based on research presented at Spring Celebration 2025

---

## Project Structure

```
snake_ai/
├── game.py           # Snake game environment (pygame)
├── model.py          # Deep Q-Network (PyTorch) — 11 → 256 → 3
├── agent.py          # DQN agent (ε-greedy + experience replay)
├── tracker.py        # Trial logger (CSV + JSON)
├── train.py          # Main training loop
├── plot_results.py   # Plot training graphs from logs
├── logs/             # Auto-created during training
│   ├── trials.csv
│   ├── trials.json
│   └── session_<timestamp>.csv
└── checkpoints/      # Auto-created during training
    └── model.pth
```

---

## Setup

```bash
pip install torch pygame numpy matplotlib
```

---

## Usage

### Train (with game window)
```bash
python train.py
```

### Train headless 
```bash
python train.py --no-render
```

### Train for exactly 300 games 
```bash
python train.py --games 300 --no-render
```

### Resume training from last checkpoint
```bash
python train.py --load
```

### Plot results after training
```bash
python plot_results.py
```

---

## How It Works

### State (11 values fed into the network)
| # | Value |
|---|-------|
| 0 | Danger straight |
| 1 | Danger right |
| 2 | Danger left |
| 3–6 | Current direction (L/R/U/D) |
| 7–10 | Food location relative to head |

### Action Space (one-hot, 3 values)
| Action | Meaning |
|--------|---------|
| [1,0,0] | Go straight |
| [0,1,0] | Turn right |
| [0,0,1] | Turn left |

### Rewards
| Event | Reward |
|-------|--------|
| Eat food | +10 |
| Die (wall/self) | −10 |
| Survive step | 0 |

### Network Architecture
```
Input (11) → Linear → ReLU → Hidden (256) → Linear → Output (3)
```

### Training
- **Short-term**: Train on each step immediately (online learning)
- **Long-term**: Sample random mini-batch from replay buffer (experience replay)
- **ε-greedy**: Explore randomly for first ~80 games, then exploit Q-network

---

## Logs

Every game is recorded automatically to `logs/trials.csv`:

| Column | Description |
|--------|-------------|
| trial | Game number |
| score | Food eaten this game |
| record | Best score ever |
| avg_score | Rolling average |
| epsilon | Exploration rate |
| duration_s | How long the game lasted |

---


