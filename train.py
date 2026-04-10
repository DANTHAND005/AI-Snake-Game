"""
snake_ai/train.py
-----------------
Main training loop.

Usage:
    python train.py                   # train with pygame window
    python train.py --no-render       # headless (faster)
    python train.py --games 300       # stop after N games
    python train.py --load            # resume from last checkpoint
"""

import argparse
import os
import sys

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train the AI Snake agent")
parser.add_argument("--no-render", action="store_true",  help="Disable pygame window (faster training)")
parser.add_argument("--games",     type=int, default=0,  help="Stop after this many games (0 = run forever)")
parser.add_argument("--load",      action="store_true",  help="Load existing checkpoint before training")
args = parser.parse_args()

RENDER         = not args.no_render
MAX_GAMES      = args.games
CHECKPOINT     = "checkpoints/model.pth"
PRINT_EVERY    = 10    # print a summary line every N games
SAVE_EVERY     = 50    # save checkpoint every N games

# ── Imports ───────────────────────────────────────────────────────────────────
from game    import SnakeGame
from agent   import Agent
from tracker import TrialTracker


def train():
    tracker = TrialTracker(log_dir="logs")
    agent   = Agent()
    game    = SnakeGame(render=RENDER)

    if args.load and os.path.exists(CHECKPOINT):
        agent.load(CHECKPOINT)

    print("=" * 60)
    print("  AI Snake — Deep Q-Learning Training")
    print(f"  Render: {RENDER}  |  Max games: {MAX_GAMES or '∞'}")
    print("  Logs  → logs/trials.csv")
    print("  Model → checkpoints/model.pth")
    print("=" * 60)

    while True:
        agent.n_games += 1
        trial = agent.n_games

        if MAX_GAMES and trial > MAX_GAMES:
            break

        tracker.start_game()
        state = game.reset()

        # ── Play one full game ─────────────────────────────────────────────
        while True:
            action                        = agent.get_action(state)
            next_state, reward, done, score = game.step(action)

            # Short-term training (online)
            agent.train_short(state, action, reward, next_state, done)
            agent.remember(state, action, reward, next_state, done)

            state = next_state

            if done:
                break

        # ── Post-game ─────────────────────────────────────────────────────
        # Long-term training (replay buffer)
        agent.train_long()

        row = tracker.log(trial, score, agent.epsilon)

        # Console output
        print(
            f"Game {row['trial']:>4} | Score {row['score']:>3} | "
            f"Record {row['record']:>3} | Avg {row['avg_score']:>6.2f} | "
            f"ε {row['epsilon']:>4}"
        )

        if trial % PRINT_EVERY == 0:
            tracker.print_summary()

        if trial % SAVE_EVERY == 0 or score >= tracker.record:
            os.makedirs("checkpoints", exist_ok=True)
            agent.save(CHECKPOINT)

    # ── Session end ───────────────────────────────────────────────────────────
    tracker.save_json()
    tracker.print_summary()
    print(f"\nTraining complete — {trial} games played.")
    print(f"Best score : {tracker.record}")
    print(f"Avg score  : {round(tracker.total_score / trial, 2)}")
    print(f"Logs saved : logs/trials.csv  |  logs/trials.json")


if __name__ == "__main__":
    train()
