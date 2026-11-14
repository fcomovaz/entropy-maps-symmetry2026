"""Scaler-folded 1-hidden-layer ReLU MLP classifier for low-footprint tabular inference.

Promoted from Understanding Lab 14 (`understanding/14_ml_classifiers_mcu/models.py`) to
`src/models/` so formal descriptor evaluation and C99 export share a single deterministic
implementation. StandardScaler parameters are algebraically folded into the first hidden
linear layer weights and bias (`w1_eff_`, `b1_eff_`), eliminating scaler transforms at
inference time ("dot" convention matching `LinearDotCoreClassifier`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn


@dataclass(frozen=True, slots=True)
class MlpHyperparams:
    """Immutable hyperparameters for MlpDotCoreClassifier."""

    hidden_dims: tuple[int, ...] = (16,)
    epochs: int = 200
    learning_rate: float = 0.002
    weight_decay: float = 0.05
    dropout: float = 0.1
    positive_weight_mode: str = "balanced"
    train_threads: int = 1


class _TorchMlp1Hidden(nn.Module):
    """Compact 1-hidden-layer ReLU MLP supporting training-time dropout."""

    def __init__(self, d_in: int, d_hidden: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.fc2 = nn.Linear(d_hidden, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.act(self.fc1(x)))
        return self.fc2(h)


class MlpDotCoreClassifier(ClassifierMixin, BaseEstimator):
    """Binary 1-hidden-layer ReLU MLP with StandardScaler folded into input dense weights."""

    def __init__(
        self,
        hidden_dims: Sequence[int] = (16,),
        epochs: int = 200,
        learning_rate: float = 0.002,
        weight_decay: float = 0.05,
        dropout: float = 0.1,
        positive_weight_mode: str = "balanced",
        train_threads: int = 1,
        random_state: int | None = 42,
    ) -> None:
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.dropout = float(dropout)
        self.positive_weight_mode = str(positive_weight_mode)
        self.train_threads = int(train_threads)
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> MlpDotCoreClassifier:
        """Fit StandardScaler + PyTorch MLP and fold scaler into layer 1 weights/bias."""
        x_arr = np.asarray(X, dtype=np.float64)
        if x_arr.ndim != 2:
            raise ValueError(f"Expected 2D array [N, F], got shape {x_arr.shape}")
        y_arr = np.asarray(y, dtype=np.int64)
        uniq = np.unique(y_arr)
        if len(uniq) != 2:
            raise ValueError(f"MlpDotCoreClassifier requires 2 classes, got {uniq}")
        if len(self.hidden_dims) != 1 or self.hidden_dims[0] < 1:
            raise ValueError(f"Expected single positive hidden_dim, got {self.hidden_dims}")

        scaler = StandardScaler().fit(x_arr)
        mu = np.asarray(scaler.mean_, dtype=np.float64)
        sigma = np.asarray(scaler.scale_, dtype=np.float64).copy()
        sigma[sigma == 0.0] = 1.0
        inv_sigma = 1.0 / sigma

        seed = 42 if self.random_state is None else int(self.random_state)
        d_in = int(x_arr.shape[1])
        d_hidden = int(self.hidden_dims[0])

        prev_threads = torch.get_num_threads()
        torch.manual_seed(seed)
        torch.set_num_threads(max(1, self.train_threads))
        try:
            net = _TorchMlp1Hidden(d_in=d_in, d_hidden=d_hidden, dropout=self.dropout)
            xt = torch.tensor(scaler.transform(x_arr), dtype=torch.float32)
            y_bin = (y_arr == uniq[1]).astype(np.float32)
            yt = torch.tensor(y_bin, dtype=torch.float32)

            if self.positive_weight_mode == "balanced":
                n_pos = float(np.sum(y_bin == 1.0))
                n_neg = float(np.sum(y_bin == 0.0))
                pos_w = torch.tensor([n_neg / max(1.0, n_pos)], dtype=torch.float32)
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
            else:
                loss_fn = nn.BCEWithLogitsLoss()

            opt = torch.optim.AdamW(
                net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
            net.train()
            for _ in range(self.epochs):
                opt.zero_grad()
                logits = net(xt).squeeze(1)
                loss = loss_fn(logits, yt)
                loss.backward()
                opt.step()
            net.eval()
        finally:
            torch.set_num_threads(prev_threads)

        with torch.no_grad():
            w1_raw = net.fc1.weight.detach().cpu().numpy().astype(np.float64)
            b1_raw = net.fc1.bias.detach().cpu().numpy().astype(np.float64)
            w2_raw = net.fc2.weight.detach().cpu().numpy().astype(np.float64).ravel()
            b2_raw = float(net.fc2.bias.detach().cpu().numpy().astype(np.float64).item())

        w1_eff = np.ascontiguousarray(w1_raw * inv_sigma[None, :], dtype=np.float64)
        b1_eff = np.ascontiguousarray(b1_raw - w1_eff @ mu, dtype=np.float64)

        self.scaler_: StandardScaler = scaler
        self.net_: nn.Module = net
        self.w1_eff_: np.ndarray = w1_eff
        self.b1_eff_: np.ndarray = b1_eff
        self.w2_eff_: np.ndarray = np.ascontiguousarray(w2_raw, dtype=np.float64)
        self.b2_eff_: float = b2_raw
        self.classes_: np.ndarray = np.asarray(uniq)
        self.n_features_in_: int = d_in
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute folded logit `relu(X @ w1_eff_.T + b1_eff_) @ w2_eff_ + b2_eff_`."""
        if not hasattr(self, "w1_eff_"):
            raise NotFittedError("This MlpDotCoreClassifier instance is not fitted yet.")
        x_arr = np.asarray(X, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        elif x_arr.ndim != 2:
            raise ValueError(f"Expected 2D array [N, F], got shape {x_arr.shape}")
        if int(x_arr.shape[1]) != self.n_features_in_:
            raise ValueError(
                f"X has {x_arr.shape[1]} features, expected {self.n_features_in_}"
            )
        h = np.maximum(0.0, x_arr @ self.w1_eff_.T + self.b1_eff_)
        return np.asarray(h @ self.w2_eff_ + self.b2_eff_, dtype=np.float64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute binary class probabilities via logistic sigmoid on folded logits."""
        scores = self.decision_function(X)
        p1 = expit(scores)
        proba = np.empty((scores.shape[0], 2), dtype=np.float64)
        proba[:, 1] = p1
        proba[:, 0] = 1.0 - p1
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels from `decision_function(X) > 0`."""
        scores = self.decision_function(X)
        return np.where(scores > 0.0, self.classes_[1], self.classes_[0])
