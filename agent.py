"""
snake_ai/agent.py
-----------------
DQN agent with:
  - Epsilon-greedy exploration
  - Experience replay (memory buffer)
  - Short-term (single step) + long-term (batch replay) training
"""

import random
import numpy as np
from collections import deque

import torch

from model import DQN, QTrainer

# ── Hyper-parameters ──────────────────────────────────────────────────────────
MAX_MEMORY   = 100_000   # replay buffer size
BATCH_SIZE   = 1_000     # mini-batch for long-term training
LR           = 1e-3      # Adam learning rate
GAMMA        = 0.9       # discount factor
EPSILON_MAX  = 80        # exploration threshold (games played)


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0           # exploration rate (decays with games)
        self.memory  = deque(maxlen=MAX_MEMORY)
        self.model   = DQN(input_size=11, hidden_size=256, output_size=3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=GAMMA)

    # ── Memory ────────────────────────────────────────────────────────────────
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # ── Training ──────────────────────────────────────────────────────────────
    def train_short(self, state, action, reward, next_state, done):
        """Train on a single transition immediately after each step."""
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long(self):
        """Sample a random mini-batch from replay buffer and train."""
        if len(self.memory) < BATCH_SIZE:
            sample = list(self.memory)
        else:
            sample = random.sample(self.memory, BATCH_SIZE)

        states, actions, rewards, next_states, dones = zip(*sample)
        self.trainer.train_step(
            list(states), list(actions), list(rewards),
            list(next_states), list(dones)
        )

    # ── Action selection (ε-greedy) ───────────────────────────────────────────
    def get_action(self, state) -> list:
        """
        With probability ε: random action (explore).
        Otherwise: best action from Q-network (exploit).
        ε decreases as the agent plays more games.
        """
        self.epsilon = EPSILON_MAX - self.n_games
        action = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            idx = random.randint(0, 2)
        else:
            state_t = torch.tensor(state, dtype=torch.float)
            pred    = self.model(state_t)
            idx     = torch.argmax(pred).item()

        action[idx] = 1
        return action

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, path: str = "checkpoints/model.pth"):
        self.model.save(path)

    def load(self, path: str = "checkpoints/model.pth"):
        self.model.load(path)
