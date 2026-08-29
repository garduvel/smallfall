"""Termination predictor MLP (CLAUDE.md S8.2).

hidden=(128, 128), activation ELU, single logit output. No RNN, LSTM, GRU,
transformer, attention, residual blocks, layer norm, or dropout (S12#16).
input_dim = 6 * history (5 obs + 1 action, stacked over `history` frames).
"""

from __future__ import annotations

import torch
from torch import nn

OBS_ACTION_DIM = 6  # 5 obs + 1 action, CLAUDE.md S8.1


class TerminationPredictor(nn.Module):
    def __init__(self, history: int = 1, hidden: tuple[int, int] = (128, 128)):
        super().__init__()
        self.history = history
        input_dim = OBS_ACTION_DIM * history
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ELU(),
            nn.Linear(h1, h2),
            nn.ELU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, 6 * history] -> logits: [batch, 1]."""
        return self.net(x)
