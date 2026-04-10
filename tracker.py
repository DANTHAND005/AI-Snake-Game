"""
snake_ai/tracker.py
--------------------
Records every trial (game) to:
  - logs/trials.csv   → easy to open in Excel / pandas
  - logs/trials.json  → full session summary with metadata
  - logs/session_<timestamp>.csv → per-session archive

Columns saved per trial:
  trial       | game number (1-based)
  score       | food eaten this game
  record      | best score seen so far
  avg_score   | rolling average over all games so far
  epsilon     | exploration rate at end of game
  duration_s  | wall-clock seconds the game lasted
"""

import csv
import json
import os
import time
from datetime import datetime


class TrialTracker:

    FIELDNAMES = ["trial", "score", "record", "avg_score", "epsilon", "duration_s"]

    def __init__(self, log_dir: str = "logs"):
        self.log_dir    = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.csv_path   = os.path.join(log_dir, "trials.csv")
        self.json_path  = os.path.join(log_dir, "trials.json")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_csv = os.path.join(log_dir, f"session_{ts}.csv")

        # State
        self.trials    : list[dict] = []
        self.record    : int        = 0
        self.total_score: int       = 0
        self._game_start: float     = time.time()

        self._init_csv()

    # ── Public API ────────────────────────────────────────────────────────────

    def start_game(self):
        """Call at the beginning of each game."""
        self._game_start = time.time()

    def log(self, trial: int, score: int, epsilon: int) -> dict:
        """
        Record one completed game.

        Returns the row dict (useful for printing).
        """
        duration = round(time.time() - self._game_start, 2)

        if score > self.record:
            self.record = score

        self.total_score += score
        avg = round(self.total_score / trial, 3)

        row = {
            "trial"     : trial,
            "score"     : score,
            "record"    : self.record,
            "avg_score" : avg,
            "epsilon"   : epsilon,
            "duration_s": duration,
        }
        self.trials.append(row)

        self._append_csv(row)
        return row

    def save_json(self):
        """Flush full session summary to JSON."""
        summary = {
            "session_start"   : datetime.now().isoformat(),
            "total_trials"    : len(self.trials),
            "record_score"    : self.record,
            "avg_score"       : round(self.total_score / max(len(self.trials), 1), 3),
            "trials"          : self.trials,
        }
        with open(self.json_path, "w") as f:
            json.dump(summary, f, indent=2)

    def print_summary(self):
        """Print a concise table of all trials."""
        print(f"\n{'─'*60}")
        print(f"{'Trial':>6}  {'Score':>6}  {'Record':>7}  {'Avg':>7}  {'Eps':>4}")
        print(f"{'─'*60}")
        for r in self.trials[-20:]:   # show last 20
            print(
                f"{r['trial']:>6}  {r['score']:>6}  {r['record']:>7}  "
                f"{r['avg_score']:>7}  {r['epsilon']:>4}"
            )
        print(f"{'─'*60}\n")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _init_csv(self):
        for path in (self.csv_path, self.session_csv):
            write_header = not os.path.exists(path) or os.path.getsize(path) == 0
            if write_header:
                with open(path, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=self.FIELDNAMES).writeheader()

    def _append_csv(self, row: dict):
        for path in (self.csv_path, self.session_csv):
            with open(path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDNAMES).writerow(row)
