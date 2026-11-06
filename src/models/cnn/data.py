"""Data loading, metadata indexing, stratified splitting, and PyTorch dataset creation.

Reads preprocessed 128x128 grayscale PNG files directly from data/preprocessed/
while using metadata CSV files solely for deterministic ordering and split parity
with the baseline classifier (task_06A).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

from .config import CnnConfig

logger = logging.getLogger("research")


@dataclass(frozen=True, slots=True)
class CnnDatasetPartition:
    """Immutable train/val/test image partition for CNN experiments."""

    dataset_name: str
    object_name: str | None
    inverted: bool
    split_mode: str
    X_train: torch.Tensor  # [N_tr, 1, H, W]
    y_train: torch.Tensor  # [N_tr]
    X_val: torch.Tensor  # [N_val, 1, H, W]
    y_val: torch.Tensor  # [N_val]
    X_test: torch.Tensor  # [N_te, 1, H, W]
    y_test: torch.Tensor  # [N_te]
    train_counts: dict[str, int]
    val_counts: dict[str, int]
    test_counts: dict[str, int]
    test_file_names: tuple[str, ...]
    load_elapsed_s: float


class InMemCnnDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """In-memory PyTorch dataset with deterministic label-preserving augmentation."""

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        hflip: bool = False,
        vflip: bool = False,
    ) -> None:
        """Initialize dataset.

        Args:
            images: Tensor of shape [N, 1, H, W].
            labels: Tensor of shape [N].
            hflip: Whether to apply random horizontal flip.
            vflip: Whether to apply random vertical flip.
        """
        if images.shape[0] != labels.shape[0]:
            raise ValueError("images and labels must have the same length")
        self.images = images
        self.labels = labels
        self.hflip = hflip
        self.vflip = vflip

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx]
        lbl = self.labels[idx]

        if self.hflip and torch.rand(1).item() > 0.5:
            img = torch.flip(img, dims=[-1])
        if self.vflip and torch.rand(1).item() > 0.5:
            img = torch.flip(img, dims=[-2])

        return img, lbl


def _resolve_image_relative_path(
    row: pd.Series, object_col: str, has_defect_col: bool
) -> Path:
    """Resolve relative sub-path to the preprocessed PNG given metadata row."""
    split_val = str(row["split"])
    file_name = str(row["file_name"])

    if object_col and object_col in row:
        obj_val = str(row[object_col])
        defect_val = str(row["defect_name"]) if has_defect_col and "defect_name" in row else str(row["type"])
        return Path(obj_val) / split_val / defect_val / file_name

    type_val = str(row["type"])
    return Path(split_val) / type_val / file_name


def _load_single_image(
    png_path: Path, expected_size: int, mean: float, std: float
) -> torch.Tensor:
    """Load a single PNG file as float32 tensor of shape [1, H, W] normalized.

    Args:
        png_path: Absolute or resolved Path to PNG image.
        expected_size: Expected image width and height (e.g. 128).
        mean: Normalization mean (e.g. 0.5).
        std: Normalization standard deviation (e.g. 0.5).

    Returns:
        torch.Tensor of shape [1, expected_size, expected_size].

    Raises:
        FileNotFoundError: If the PNG does not exist.
        ValueError: If dimensions do not match expected_size.
    """
    if not png_path.is_file():
        raise FileNotFoundError(f"Image PNG not found: {png_path}")

    with Image.open(png_path) as img:
        if img.size != (expected_size, expected_size):
            raise ValueError(
                f"Image {png_path} size {img.size} does not match expected ({expected_size}, {expected_size})"
            )
        img_gray = img.convert("L")
        arr = np.asarray(img_gray, dtype=np.float32) / 255.0

    arr = (arr - mean) / std
    return torch.from_numpy(arr).unsqueeze(0)


def load_cnn_partition(
    cfg: CnnConfig,
    object_name: str | None = None,
    seed: int | None = None,
) -> CnnDatasetPartition:
    """Load dataset partition directly from preprocessed PNG files with baseline parity.

    Args:
        cfg: Validated CnnConfig instance.
        object_name: Optional MVTec-AD object override.
        seed: Random seed for splitting (defaults to cfg.seed).

    Returns:
        CnnDatasetPartition with cached train, val, and test tensors.

    Raises:
        FileNotFoundError: If metadata CSV or images cannot be located.
        ValueError: If filtering results in empty sets or fewer than 2 classes.
    """
    start_t = time.perf_counter()
    active_seed = seed if seed is not None else cfg.seed

    meta_csv_path = cfg.effective_metadata_csv
    if not meta_csv_path.is_file():
        raise FileNotFoundError(f"Metadata CSV file not found: {meta_csv_path}")

    # Read strictly the metadata columns (fast, lightweight, memory-efficient)
    object_col = cfg.data.object_column
    needed_cols = [cfg.data.split_column, cfg.data.label_column, "file_name"]
    if object_col:
        needed_cols.append(object_col)

    # Peek headers
    with meta_csv_path.open("r", encoding="utf-8") as f:
        header_line = f.readline().strip().split(",")
    has_defect_col = "defect_name" in header_line
    if has_defect_col:
        needed_cols.append("defect_name")

    for col in needed_cols:
        if col not in header_line:
            raise ValueError(f"Required metadata column '{col}' missing from {meta_csv_path}")

    df = pd.read_csv(meta_csv_path, usecols=needed_cols)

    # 1. Strictly filter allowed_splits (excludes ground_truth masks in MVTec-AD)
    df = df[df[cfg.data.split_column].isin(cfg.data.allowed_splits)].copy()
    if df.empty:
        raise ValueError(
            f"No rows remain after filtering allowed_splits={cfg.data.allowed_splits}"
        )

    # 2. Filter by object_name when object_column is configured
    target_object: str | None = None
    if object_col:
        target_object = object_name if object_name is not None else cfg.default_object
        if target_object is None:
            raise ValueError(
                f"Dataset requires an object selection ('{object_col}'), "
                f"but neither --object nor default_object was provided."
            )
        if cfg.subdatasets and target_object not in cfg.subdatasets:
            raise ValueError(
                f"Unknown object_name '{target_object}'. Valid subdatasets: {cfg.subdatasets}"
            )
        df = df[df[object_col] == target_object].copy()
        if df.empty:
            raise ValueError(
                f"No rows found for {object_col} == '{target_object}'"
            )

    # 3. Validate and encode binary labels
    label_map = {cfg.data.negative_label: 0, cfg.data.positive_label: 1}
    unknown_labels = set(df[cfg.data.label_column].unique()) - set(label_map.keys())
    if unknown_labels:
        raise ValueError(
            f"Unexpected label values {sorted(unknown_labels)} in '{cfg.data.label_column}'. "
            f"Expected only {sorted(label_map.keys())}"
        )
    y_all = df[cfg.data.label_column].map(label_map).to_numpy(dtype=np.int64)
    n_samples = len(df)
    indices_all = np.arange(n_samples)

    # 4. Partition into Train Full and Test
    if cfg.data.split_mode == "presplit":
        train_full_mask = (df[cfg.data.split_column] == cfg.data.train_split_name).to_numpy()
        test_mask = (df[cfg.data.split_column] == cfg.data.test_split_name).to_numpy()
        train_full_idx = indices_all[train_full_mask]
        test_idx = indices_all[test_mask]
    elif cfg.data.split_mode == "stratified":
        if len(np.unique(y_all)) < 2:
            raise ValueError(
                "Pooled dataset has fewer than 2 unique classes; cannot perform stratified split"
            )
        train_full_idx, test_idx = train_test_split(
            indices_all,
            test_size=cfg.data.test_size,
            stratify=y_all,
            random_state=active_seed,
        )
    else:
        raise ValueError(f"Unsupported split_mode: {cfg.data.split_mode}")

    # Validate test classes
    y_test_arr = y_all[test_idx]
    if len(np.unique(y_test_arr)) < 2:
        raise ValueError("Test split has fewer than 2 unique classes")

    y_train_full = y_all[train_full_idx]
    if len(np.unique(y_train_full)) < 2:
        raise ValueError("Training split has fewer than 2 unique classes")

    # 5. Carve validation split from train_full ONLY (for early stopping)
    if cfg.data.val_size > 0.0 and len(train_full_idx) > 5:
        # Check if stratified carve is possible
        min_class_count = min(
            int(np.sum(y_train_full == 0)), int(np.sum(y_train_full == 1))
        )
        if min_class_count >= 2:
            train_idx, val_idx = train_test_split(
                train_full_idx,
                test_size=cfg.data.val_size,
                stratify=y_train_full,
                random_state=active_seed,
            )
        else:
            # Fallback to unstratified split if extreme imbalance
            train_idx, val_idx = train_test_split(
                train_full_idx,
                test_size=cfg.data.val_size,
                random_state=active_seed,
            )
    else:
        train_idx = train_full_idx
        val_idx = np.empty(0, dtype=np.int64)

    # 6. Load images into in-memory tensors
    root_preprocessed = cfg.effective_preprocessed_dir
    if not root_preprocessed.is_dir():
        raise FileNotFoundError(
            f"Preprocessed image root directory not found: {root_preprocessed}"
        )

    # Pre-resolve relative paths
    relative_paths = [
        _resolve_image_relative_path(row, object_col, has_defect_col)
        for _, row in df.iterrows()
    ]

    def _load_batch(selected_indices: np.ndarray) -> torch.Tensor:
        if len(selected_indices) == 0:
            return torch.empty(
                (0, 1, cfg.data.image_size, cfg.data.image_size), dtype=torch.float32
            )
        tensors = []
        for idx in selected_indices:
            full_path = root_preprocessed / relative_paths[idx]
            tensor = _load_single_image(
                full_path,
                cfg.data.image_size,
                cfg.data.normalize_mean,
                cfg.data.normalize_std,
            )
            tensors.append(tensor)
        return torch.stack(tensors, dim=0)

    X_train = _load_batch(train_idx)
    y_train = torch.from_numpy(y_all[train_idx])

    X_val = _load_batch(val_idx)
    y_val = torch.from_numpy(y_all[val_idx])

    X_test = _load_batch(test_idx)
    y_test = torch.from_numpy(y_all[test_idx])

    test_file_names = tuple(str(df.iloc[i]["file_name"]) for i in test_idx)

    train_counts = {
        cfg.data.negative_label: int((y_train == 0).sum().item()),
        cfg.data.positive_label: int((y_train == 1).sum().item()),
    }
    val_counts = {
        cfg.data.negative_label: int((y_val == 0).sum().item()),
        cfg.data.positive_label: int((y_val == 1).sum().item()),
    }
    test_counts = {
        cfg.data.negative_label: int((y_test == 0).sum().item()),
        cfg.data.positive_label: int((y_test == 1).sum().item()),
    }
    load_elapsed_s = time.perf_counter() - start_t

    return CnnDatasetPartition(
        dataset_name=cfg.dataset_name,
        object_name=target_object,
        inverted=cfg.inverted,
        split_mode=cfg.data.split_mode,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        train_counts=train_counts,
        val_counts=val_counts,
        test_counts=test_counts,
        test_file_names=test_file_names,
        load_elapsed_s=load_elapsed_s,
    )
