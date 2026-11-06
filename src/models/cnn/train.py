"""Training loop, loss computation with class-weighting, cosine scheduling, and early stopping."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import logging
import time
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader

from .architectures.common import seed_everything
from .config import CnnConfig
from .data import CnnDatasetPartition, InMemCnnDataset

logger = logging.getLogger("research")


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Immutable result from CNN model training."""

    best_epoch: int
    train_elapsed_s: float
    history: tuple[dict[str, float], ...]
    best_val_pr_auc: float
    best_state_dict: Mapping[str, torch.Tensor]


def _evaluate_subset(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[float, float, float, float]:
    """Evaluate loss, accuracy, PR-AUC, and F1 on a validation tensor batch."""
    model.eval()
    if len(labels) == 0:
        return 0.0, 0.0, 0.0, 0.0

    total_loss = 0.0
    all_probs: list[float] = []
    all_preds: list[int] = []
    all_targets: list[int] = labels.tolist()

    n = len(labels)
    with torch.inference_mode():
        for start_idx in range(0, n, batch_size):
            end_idx = min(start_idx + batch_size, n)
            xb = images[start_idx:end_idx].to(device)
            yb = labels[start_idx:end_idx].to(device)

            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * len(yb)

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())

    mean_loss = total_loss / max(1, n)
    acc = float(accuracy_score(all_targets, all_preds))
    f1 = float(f1_score(all_targets, all_preds, zero_division=0))

    if len(np.unique(all_targets)) > 1:
        pr_auc = float(average_precision_score(all_targets, all_probs))
    else:
        pr_auc = 0.0

    return mean_loss, acc, pr_auc, f1


def train_model(
    model: nn.Module,
    partition: CnnDatasetPartition,
    cfg: CnnConfig,
    seed: int,
    device: torch.device,
) -> TrainResult:
    """Train lightweight CNN model according to TOML specifications.

    Args:
        model: PyTorch nn.Module architecture.
        partition: CnnDatasetPartition with training and validation data.
        cfg: Validated CnnConfig instance.
        seed: Random seed for reproducibility.
        device: Target execution device.

    Returns:
        TrainResult with best model state dict, history, and training duration.
    """
    seed_everything(seed)
    model = model.to(device)

    # 1. Setup loss criterion with optional balanced weighting
    if cfg.training.class_weighting == "balanced":
        n_pos = int((partition.y_train == 1).sum().item())
        n_neg = int((partition.y_train == 0).sum().item())
        n_tot = n_pos + n_neg
        w0 = n_tot / (2.0 * max(1, n_neg))
        w1 = n_tot / (2.0 * max(1, n_pos))
        weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # 2. Setup optimizer
    if cfg.training.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
    elif cfg.training.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer '{cfg.training.optimizer}'")

    # 3. Setup learning rate scheduler
    scheduler: Any = None
    if cfg.training.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.epochs, eta_min=1e-6
        )
    elif cfg.training.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, cfg.training.epochs // 2), gamma=0.1
        )

    # 4. DataLoader
    train_dataset = InMemCnnDataset(
        partition.X_train,
        partition.y_train,
        hflip=cfg.augmentation.hflip,
        vflip=cfg.augmentation.vflip,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(cfg.training.batch_size, max(2, len(train_dataset))),
        shuffle=True,
        num_workers=cfg.training.num_workers,
        drop_last=False,
    )

    history: list[dict[str, float]] = []
    best_epoch = 1
    best_val_pr_auc = -1.0
    best_state_dict: Mapping[str, torch.Tensor] = {
        k: v.cpu().clone() for k, v in model.state_dict().items()
    }
    patience_counter = 0
    has_val = len(partition.y_val) > 0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0_train = time.perf_counter()

    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss encountered at epoch {epoch}: {loss.item()}")

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(yb)
            train_correct += (logits.argmax(dim=1) == yb).sum().item()
            train_samples += len(yb)

        current_lr = float(optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step()

        epoch_train_loss = train_loss / max(1, train_samples)
        epoch_train_acc = train_correct / max(1, train_samples)

        if has_val:
            val_loss, val_acc, val_pr_auc, val_f1 = _evaluate_subset(
                model, partition.X_val, partition.y_val, criterion, device
            )
            # Early stopping metric: PR-AUC is primary for imbalanced anomaly detection
            current_metric = val_pr_auc
            if current_metric > best_val_pr_auc + 1e-4:
                best_val_pr_auc = current_metric
                best_epoch = epoch
                best_state_dict = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
        else:
            val_loss, val_acc, val_pr_auc, val_f1 = 0.0, 0.0, 0.0, 0.0
            best_epoch = epoch
            best_state_dict = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

        record = {
            "epoch": float(epoch),
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_pr_auc": val_pr_auc,
            "val_f1": val_f1,
            "lr": current_lr,
        }
        history.append(record)

        if has_val and patience_counter >= cfg.training.early_stopping_patience:
            logger.info(
                "Early stopping triggered at epoch %d (patience=%d, best_val_pr_auc=%.4f at epoch %d)",
                epoch,
                cfg.training.early_stopping_patience,
                best_val_pr_auc,
                best_epoch,
            )
            break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_elapsed_s = time.perf_counter() - t0_train

    # Restore best weights
    model.load_state_dict(best_state_dict)

    return TrainResult(
        best_epoch=best_epoch,
        train_elapsed_s=train_elapsed_s,
        history=tuple(history),
        best_val_pr_auc=best_val_pr_auc if has_val else 0.0,
        best_state_dict=best_state_dict,
    )
