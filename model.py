"""
snake_ai/model.py
-----------------
Deep Q-Network (DQN) built with PyTorch.

Architecture:
    Input  → 11 neurons  (game state)
    Hidden → 256 neurons
    Output → 3 neurons   (straight / right / left)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class DQN(nn.Module):
    """
    Fully-connected neural network that approximates the Q-function.
    Q(state) → [Q_straight, Q_right, Q_left]
    """

    def __init__(self, input_size: int = 11, hidden_size: int = 256, output_size: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def save(self, path: str = "model.pth"):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"[Model] Saved → {path}")

    def load(self, path: str = "model.pth"):
        self.load_state_dict(torch.load(path, map_location="cpu"))
        self.eval()
        print(f"[Model] Loaded ← {path}")


class QTrainer:
    """
    Handles one step of the Bellman-equation update:
        Q_target = r + γ · max_a Q(s', a)     (if not done)
        Q_target = r                            (if done)

    Loss = MSE(Q_predicted, Q_target)
    """

    def __init__(self, model: DQN, lr: float = 1e-3, gamma: float = 0.9):
        self.model     = model
        self.gamma     = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self, states, actions, rewards, next_states, dones):
        states      = torch.tensor(states,      dtype=torch.float)
        actions     = torch.tensor(actions,     dtype=torch.long)
        rewards     = torch.tensor(rewards,     dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        dones       = torch.tensor(dones,       dtype=torch.bool)

        # Ensure batch dimension
        if states.dim() == 1:
            states      = states.unsqueeze(0)
            actions     = actions.unsqueeze(0)
            rewards     = rewards.unsqueeze(0)
            next_states = next_states.unsqueeze(0)
            dones       = dones.unsqueeze(0)

        # Current Q-values for all actions
        q_pred = self.model(states)

        # Compute targets
        q_target = q_pred.clone().detach()
        with torch.no_grad():
            q_next = self.model(next_states)

        for i in range(len(dones)):
            q_new = rewards[i]
            if not dones[i]:
                q_new = rewards[i] + self.gamma * torch.max(q_next[i])
            action_idx = torch.argmax(actions[i]).item()
            q_target[i][action_idx] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(q_pred, q_target)
        loss.backward()
        self.optimizer.step()
