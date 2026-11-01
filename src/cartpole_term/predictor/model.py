"""Termination predictor MLP (CLAUDE.md S8.2).

hidden=(128, 128), activation ELU, single logit output. No RNN, LSTM, GRU,
transformer, attention, residual blocks, layer norm, or dropout (S12#16).
input_dim = obs_action_dim * history (obs + action, stacked over `history`
frames).

obs_action_dim defaults to 6 (cartpole: 5 obs + 1 action, CLAUDE.md S8.1) so
every existing cartpole call site (`TerminationPredictor(history=1)`, etc.)
keeps constructing the exact same network unchanged. Phase-2 tasks with a
different obs/action width (e.g. the Go1 velocity task) pass their own
`obs_action_dim` explicitly rather than this file guessing or hardcoding a
second literal -- see cartpole_term/go1/predictor.py.
"""

from __future__ import annotations

import torch
from torch import nn

OBS_ACTION_DIM = 6  # 5 obs + 1 action, CLAUDE.md S8.1 -- cartpole's default only


class TerminationPredictor(nn.Module):
    def __init__(
        self,
        history: int = 1,
        hidden: tuple[int, int] = (128, 128),
        obs_action_dim: int = OBS_ACTION_DIM,
    ):
        super().__init__()
        self.history = history
        self.obs_action_dim = obs_action_dim
        input_dim = obs_action_dim * history
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ELU(),
            nn.Linear(h1, h2),
            nn.ELU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, obs_action_dim * history] -> logits: [batch, 1]."""
        return self.net(x)
