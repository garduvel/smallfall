"""Offline supervised training for the termination predictor (CLAUDE.md S8.3).

Loss: plain BCEWithLogitsLoss. No class reweighting, no pos_weight, no focal
loss -- CLAUDE.md S8.3: base rates here (roughly 20%/11%/4% across the three
operating points) are well inside the range where plain BCE is fine, and
reweighting would distort calibration, which is a reported metric.

Optimizer: Adam, lr=1e-3, weight_decay=0.0, max_grad_norm=1.0. Batch size
1024, up to 50 epochs, early stopping on validation AUPRC (patience 5),
best-validation weights restored before returning. The test split is never
touched here -- CLAUDE.md S8.3: "the test split is touched exactly once, at
the end", which happens in scripts/eval_predictor.py, after this returns.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainResult:
    state_dict: dict
    best_epoch: int
    best_val_auprc: float
    history: list[dict] = field(default_factory=list)


def train_predictor(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    device: str = "cpu",
    seed: int = 0,
    batch_size: int = 1024,
    max_epochs: int = 50,
    patience: int = 5,
    lr: float = 1e-3,
    max_grad_norm: float = 1.0,
) -> TrainResult:
    if len(train_x) == 0:
        raise ValueError("train_predictor: empty training set")
    if len(val_x) == 0:
        raise ValueError("train_predictor: empty validation set")

    torch.manual_seed(seed)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    loss_fn = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(
        torch.as_tensor(train_x, dtype=torch.float32),
        torch.as_tensor(train_y, dtype=torch.float32),
    )
    # batch_size may exceed a tiny (e.g. --smoke) dataset; DataLoader handles
    # that fine (one under-full batch), no special-casing needed.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_x_t = torch.as_tensor(val_x, dtype=torch.float32).to(device)

    best_val_auprc = -1.0
    best_state: dict | None = None
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict] = []

    val_has_both_classes = len(np.unique(val_y)) > 1

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x_t).squeeze(-1)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
        val_auprc = (
            average_precision_score(val_y, val_probs) if val_has_both_classes else 0.0
        )

        history.append(
            {"epoch": epoch, "train_loss": total_loss / max(n_batches, 1), "val_auprc": val_auprc}
        )

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)

    return TrainResult(
        state_dict=best_state, best_epoch=best_epoch, best_val_auprc=best_val_auprc, history=history
    )
