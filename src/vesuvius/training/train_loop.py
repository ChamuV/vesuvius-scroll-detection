# src/vesuvius/training/train_loop.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from vesuvius.training.train_epoch_3d import train_epoch_3d
from vesuvius.training.validate_epoch_3d import validate_epoch_3d


@dataclass
class TrainHistory:
    train_loss: list[float]
    val_loss: list[float]
    dice: list[float]


def train_loop(
    *,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    device: torch.device,
    epochs: int = 50,
    log_every: int = 5,
    threshold: float = 0.5,
) -> TrainHistory:
    hist = TrainHistory(train_loss=[], val_loss=[], dice=[])

    for ep in range(1, epochs + 1):
        tr = train_epoch_3d(model, train_loader, loss_fn, optimizer, device)
        va, mets = validate_epoch_3d(model, val_loader, loss_fn, device, threshold=threshold)

        hist.train_loss.append(float(tr))
        hist.val_loss.append(float(va))
        hist.dice.append(float(mets.get("dice", 0.0)))

        if ep == 1 or ep % log_every == 0:
            print(
                f"epoch {ep:03d} | train_loss={tr:.6f} | val_loss={va:.6f} | "
                f"dice={mets.get('dice', 0.0):.4f}"
            )

    return hist