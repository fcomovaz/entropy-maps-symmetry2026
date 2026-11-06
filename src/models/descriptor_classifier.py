"""Supervised classification, subset ablation, and explainability pipeline for x94 descriptors.

Provides TOML-driven configuration loading, in-memory filtering and partitioning
(native presplit for KolektorSDD2, seeded stratified split per object for MVTec-AD),
Random Forest and k-NN training/evaluation with timing separation, and multi-scale
explainability breakdowns (x85 entropy vs x9 asymmetry, per-scale contributions,
and 2D spatial importance heatmaps).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
import tomllib
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.asymmetry_maps import asymmetry_column_names, entropy_column_names

logger = logging.getLogger("research")


def _require(table: Mapping[str, object], key: str, where: str) -> object:
    """Extract required key from dictionary or raise KeyError with context."""
    if key not in table:
        raise KeyError(f"Missing required key '{key}' in {where}")
    return table[key]


@dataclass(frozen=True, slots=True)
class RfHyperparams:
    """Immutable hyperparameters for RandomForestClassifier."""

    n_estimators: int
    max_depth: int | None
    min_samples_leaf: int
    class_weight: str | None
    n_jobs: int


@dataclass(frozen=True, slots=True)
class DtHyperparams:
    """Immutable hyperparameters for DecisionTreeClassifier."""

    max_depth: int | None
    min_samples_leaf: int
    class_weight: str | None


@dataclass(frozen=True, slots=True)
class LrHyperparams:
    """Immutable hyperparameters for LogisticRegression."""

    C: float
    penalty: str
    solver: str
    class_weight: str | None
    scale_features: bool


@dataclass(frozen=True, slots=True)
class KnnHyperparams:
    """Immutable hyperparameters for KNeighborsClassifier."""

    n_neighbors: int
    weights: str
    metric: str
    p: int
    scale_features: bool


@dataclass(frozen=True, slots=True)
class ClassificationConfig:
    """Immutable configuration for x94 descriptor classification and explainability."""

    dataset_key: str
    dataset_name: str
    input_csv: Path
    input_csv_inverted: Path
    results_dir: Path
    inverted: bool
    overwrite: bool
    seed: int
    default_classifier: str
    default_object: str | None
    subdatasets: tuple[str, ...]
    split_column: str
    label_column: str
    object_column: str
    positive_label: str
    negative_label: str
    allowed_splits: tuple[str, ...]
    split_mode: str
    train_split_name: str
    test_split_name: str
    test_size: float
    active_subset: str
    compare_subsets: tuple[str, ...]
    rf_params: RfHyperparams
    knn_params: KnnHyperparams
    dt_params: DtHyperparams
    lr_params: LrHyperparams
    explainability_enabled: bool
    permutation_repeats: int
    top_k_features: int
    dpi: int
    float_precision: int
    feature_name_template: str
    global_scale: int
    asymmetry_scales: tuple[int, ...]
    asymmetry_kinds: tuple[str, ...]
    asymmetry_templates: dict[str, str]

    @property
    def effective_input_csv(self) -> Path:
        """Return active x94 CSV path according to the inversion setting."""
        return self.input_csv_inverted if self.inverted else self.input_csv

    @property
    def effective_dataset_tag(self) -> str:
        """Return dataset folder name including '-inverted' suffix when active."""
        base_tag = self.dataset_name.lower()
        return f"{base_tag}-inverted" if self.inverted else base_tag

    @property
    def available_classifiers(self) -> tuple[str, ...]:
        """Return tuple of supported classifier names configured in this pipeline."""
        return ("lr", "dt", "knn", "rf")

    def effective_results_dir(
        self, object_name: str | None, classifier_name: str
    ) -> Path:
        """Resolve output directory for a specific dataset variant, object, and classifier."""
        target_tag = object_name if object_name else "full"
        return (
            self.results_dir
            / self.effective_dataset_tag
            / f"{target_tag}_{classifier_name.lower()}"
        )


def load_classification_config(
    config_path: Path, project_root: Path
) -> ClassificationConfig:
    """Load, resolve, and validate classification configuration from TOML files.

    Args:
        config_path: Path to the classification task TOML file.
        project_root: Repository root directory for anchoring relative paths.

    Returns:
        Validated ClassificationConfig instance.

    Raises:
        FileNotFoundError: If configuration files do not exist.
        KeyError: If required sections or keys are missing.
        ValueError: If parameter values violate type or domain constraints.
    """
    resolved_cfg_path = (project_root / config_path).resolve()
    if not resolved_cfg_path.is_file():
        raise FileNotFoundError(
            f"Classification config file not found: {resolved_cfg_path}"
        )

    with resolved_cfg_path.open("rb") as f:
        raw_cfg = tomllib.load(f)

    cls_table_obj = _require(raw_cfg, "classification", "classification config")
    if not isinstance(cls_table_obj, dict):
        raise ValueError("[classification] section must be a table")
    cls_table: dict[str, object] = cls_table_obj

    base_config_rel = str(_require(cls_table, "base_config", "[classification]"))
    asym_config_rel = str(_require(cls_table, "asymmetry_config", "[classification]"))
    dataset_key = str(_require(cls_table, "dataset_key", "[classification]"))
    inverted = _require(cls_table, "inverted", "[classification]")
    overwrite = _require(cls_table, "overwrite", "[classification]")
    seed_raw = _require(cls_table, "seed", "[classification]")
    default_classifier = str(
        _require(cls_table, "default_classifier", "[classification]")
    ).lower()
    default_object_raw = cls_table.get("default_object")
    default_object = str(default_object_raw) if default_object_raw is not None else None

    if not isinstance(inverted, bool):
        raise ValueError("'inverted' in [classification] must be a boolean")
    if not isinstance(overwrite, bool):
        raise ValueError("'overwrite' in [classification] must be a boolean")
    if isinstance(seed_raw, bool) or not isinstance(seed_raw, int):
        raise ValueError("'seed' in [classification] must be an integer")
    if default_classifier not in {"rf", "knn", "dt", "lr"}:
        raise ValueError(
            f"'default_classifier' must be 'rf', 'knn', 'dt', or 'lr', got '{default_classifier}'"
        )

    # Data section
    data_table_obj = _require(cls_table, "data", "[classification]")
    if not isinstance(data_table_obj, dict):
        raise ValueError("[classification.data] must be a table")
    data_table: dict[str, object] = data_table_obj

    split_column = str(_require(data_table, "split_column", "[classification.data]"))
    label_column = str(_require(data_table, "label_column", "[classification.data]"))
    object_column = str(_require(data_table, "object_column", "[classification.data]"))
    positive_label = str(
        _require(data_table, "positive_label", "[classification.data]")
    )
    negative_label = str(
        _require(data_table, "negative_label", "[classification.data]")
    )
    raw_allowed_splits = _require(data_table, "allowed_splits", "[classification.data]")
    split_mode = str(
        _require(data_table, "split_mode", "[classification.data]")
    ).lower()
    train_split_name = str(
        _require(data_table, "train_split_name", "[classification.data]")
    )
    test_split_name = str(
        _require(data_table, "test_split_name", "[classification.data]")
    )
    test_size = float(_require(data_table, "test_size", "[classification.data]"))

    if not isinstance(raw_allowed_splits, list) or not raw_allowed_splits:
        raise ValueError(
            "'allowed_splits' in [classification.data] must be a non-empty list"
        )
    allowed_splits = tuple(str(s) for s in raw_allowed_splits)

    if split_mode not in {"presplit", "stratified"}:
        raise ValueError(
            f"'split_mode' in [classification.data] must be 'presplit' or 'stratified', got '{split_mode}'"
        )
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"'test_size' must be strictly in (0.0, 1.0), got {test_size}")
    if positive_label == negative_label:
        raise ValueError("positive_label and negative_label must be distinct")

    # Subsets section
    subsets_table_obj = _require(cls_table, "subsets", "[classification]")
    if not isinstance(subsets_table_obj, dict):
        raise ValueError("[classification.subsets] must be a table")
    active_subset = str(
        _require(subsets_table_obj, "active", "[classification.subsets]")
    )
    raw_compare_subsets = _require(
        subsets_table_obj, "compare_subsets", "[classification.subsets]"
    )
    if not isinstance(raw_compare_subsets, list) or not raw_compare_subsets:
        raise ValueError(
            "'compare_subsets' in [classification.subsets] must be a non-empty list"
        )
    compare_subsets = tuple(str(s) for s in raw_compare_subsets)

    # Models section
    models_table_obj = _require(cls_table, "models", "[classification]")
    if not isinstance(models_table_obj, dict):
        raise ValueError("[classification.models] must be a table")

    # LR section (optional)
    lr_table_obj = models_table_obj.get("lr")
    if lr_table_obj is not None:
        if not isinstance(lr_table_obj, dict):
            raise ValueError("[classification.models.lr] must be a table")
        C = float(_require(lr_table_obj, "C", "[classification.models.lr]"))
        penalty = str(_require(lr_table_obj, "penalty", "[classification.models.lr]"))
        solver = str(_require(lr_table_obj, "solver", "[classification.models.lr]"))
        class_weight_raw = str(
            _require(lr_table_obj, "class_weight", "[classification.models.lr]")
        )
        scale_features = bool(
            _require(lr_table_obj, "scale_features", "[classification.models.lr]")
        )
        lr_params = LrHyperparams(
            C=C,
            penalty=penalty,
            solver=solver,
            class_weight=class_weight_raw,
            scale_features=scale_features,
        )
    else:
        lr_params = LrHyperparams(
            C=1.0,
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            scale_features=True,
        )

    # DT section (optional)
    dt_table_obj = models_table_obj.get("dt")
    if dt_table_obj is not None:
        if not isinstance(dt_table_obj, dict):
            raise ValueError("[classification.models.dt] must be a table")
        max_depth = int(_require(dt_table_obj, "max_depth", "[classification.models.dt]"))
        min_samples_leaf = int(
            _require(dt_table_obj, "min_samples_leaf", "[classification.models.dt]")
        )
        class_weight_raw = str(
            _require(dt_table_obj, "class_weight", "[classification.models.dt]")
        )
        if max_depth < 1 or min_samples_leaf < 1:
            raise ValueError("DT max_depth and min_samples_leaf must be >= 1")
        dt_params = DtHyperparams(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight_raw,
        )
    else:
        dt_params = DtHyperparams(
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
        )

    # RF section
    rf_table_obj = _require(models_table_obj, "rf", "[classification.models]")
    if not isinstance(rf_table_obj, dict):
        raise ValueError("[classification.models.rf] must be a table")
    n_estimators = int(
        _require(rf_table_obj, "n_estimators", "[classification.models.rf]")
    )
    max_depth_raw = _require(rf_table_obj, "max_depth", "[classification.models.rf]")
    max_depth = int(max_depth_raw) if int(max_depth_raw) > 0 else None
    min_samples_leaf = int(
        _require(rf_table_obj, "min_samples_leaf", "[classification.models.rf]")
    )
    class_weight_raw = str(
        _require(rf_table_obj, "class_weight", "[classification.models.rf]")
    )
    class_weight = None if class_weight_raw.lower() == "none" else class_weight_raw
    n_jobs = int(_require(rf_table_obj, "n_jobs", "[classification.models.rf]"))
    if n_estimators < 1 or min_samples_leaf < 1:
        raise ValueError("RF n_estimators and min_samples_leaf must be >= 1")
    rf_params = RfHyperparams(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        n_jobs=n_jobs,
    )

    # KNN section
    knn_table_obj = _require(models_table_obj, "knn", "[classification.models]")
    if not isinstance(knn_table_obj, dict):
        raise ValueError("[classification.models.knn] must be a table")
    n_neighbors = int(
        _require(knn_table_obj, "n_neighbors", "[classification.models.knn]")
    )
    weights = str(_require(knn_table_obj, "weights", "[classification.models.knn]"))
    metric = str(_require(knn_table_obj, "metric", "[classification.models.knn]"))
    p_val = int(_require(knn_table_obj, "p", "[classification.models.knn]"))
    scale_features = bool(
        _require(knn_table_obj, "scale_features", "[classification.models.knn]")
    )
    if n_neighbors < 1 or p_val < 1:
        raise ValueError("k-NN n_neighbors and p must be >= 1")
    knn_params = KnnHyperparams(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        p=p_val,
        scale_features=scale_features,
    )

    # Explainability section
    exp_table_obj = _require(cls_table, "explainability", "[classification]")
    if not isinstance(exp_table_obj, dict):
        raise ValueError("[classification.explainability] must be a table")
    explainability_enabled = bool(
        _require(exp_table_obj, "enabled", "[classification.explainability]")
    )
    permutation_repeats = int(
        _require(
            exp_table_obj, "permutation_repeats", "[classification.explainability]"
        )
    )
    top_k_features = int(
        _require(exp_table_obj, "top_k_features", "[classification.explainability]")
    )
    if permutation_repeats < 1 or top_k_features < 1:
        raise ValueError("permutation_repeats and top_k_features must be >= 1")

    # Output section
    out_table_obj = _require(cls_table, "output", "[classification]")
    if not isinstance(out_table_obj, dict):
        raise ValueError("[classification.output] must be a table")
    results_dir_rel = str(
        _require(out_table_obj, "results_dir", "[classification.output]")
    )
    dpi = int(_require(out_table_obj, "dpi", "[classification.output]"))
    float_precision = int(
        _require(out_table_obj, "float_precision", "[classification.output]")
    )
    if dpi < 50 or float_precision < 1:
        raise ValueError("dpi must be >= 50 and float_precision >= 1")

    # Load base config
    base_cfg_path = (project_root / base_config_rel).resolve()
    if not base_cfg_path.is_file():
        raise FileNotFoundError(f"Base config file not found: {base_cfg_path}")
    with base_cfg_path.open("rb") as f_base:
        base_data = tomllib.load(f_base)

    datasets_table = _require(base_data, "datasets", "base config")
    if not isinstance(datasets_table, dict):
        raise ValueError("[datasets] in base config must be a table")
    ds_spec = _require(
        datasets_table, dataset_key, f"base config [datasets.{dataset_key}]"
    )
    if not isinstance(ds_spec, dict):
        raise ValueError(f"[datasets.{dataset_key}] in base config must be a table")

    dataset_name = str(_require(ds_spec, "name", f"[datasets.{dataset_key}]"))
    input_csv_rel = str(_require(ds_spec, "full_csv_path", f"[datasets.{dataset_key}]"))
    input_csv_inv_rel = str(
        _require(ds_spec, "full_csv_path_inverted", f"[datasets.{dataset_key}]")
    )
    raw_subdatasets = ds_spec.get("subdatasets", [])
    subdatasets = (
        tuple(str(s) for s in raw_subdatasets)
        if isinstance(raw_subdatasets, list)
        else ()
    )

    if (
        object_column
        and default_object is not None
        and subdatasets
        and default_object not in subdatasets
    ):
        raise ValueError(
            f"default_object '{default_object}' not found in subdatasets {subdatasets}"
        )

    # Load asymmetry config for feature naming templates
    asym_cfg_path = (project_root / asym_config_rel).resolve()
    if not asym_cfg_path.is_file():
        raise FileNotFoundError(f"Asymmetry config file not found: {asym_cfg_path}")
    with asym_cfg_path.open("rb") as f_asym:
        asym_data = tomllib.load(f_asym)

    asym_table = _require(asym_data, "asymmetry_features", "asymmetry config")
    if not isinstance(asym_table, dict):
        raise ValueError("[asymmetry_features] in asymmetry config must be a table")
    asym_regions = _require(asym_table, "regions", "[asymmetry_features]")
    asym_source = _require(asym_table, "source", "[asymmetry_features]")
    asym_output = _require(asym_table, "output", "[asymmetry_features]")
    if not (
        isinstance(asym_regions, dict)
        and isinstance(asym_source, dict)
        and isinstance(asym_output, dict)
    ):
        raise ValueError("Invalid structure in linked asymmetry config")

    asymmetry_scales = tuple(int(s) for s in _require(asym_regions, "scales", "[asymmetry_features.regions]"))  # type: ignore[arg-type]
    asymmetry_kinds = tuple(str(k) for k in _require(asym_regions, "kinds", "[asymmetry_features.regions]"))  # type: ignore[arg-type]
    feature_name_template = str(
        _require(asym_source, "feature_name_template", "[asymmetry_features.source]")
    )
    global_scale = int(
        _require(asym_source, "global_scale", "[asymmetry_features.source]")
    )
    raw_templates = _require(
        asym_output, "column_templates", "[asymmetry_features.output]"
    )
    if not isinstance(raw_templates, dict):
        raise ValueError("column_templates in asymmetry config must be a table")
    asymmetry_templates = {str(k): str(v) for k, v in raw_templates.items()}

    return ClassificationConfig(
        dataset_key=dataset_key,
        dataset_name=dataset_name,
        input_csv=(project_root / input_csv_rel).resolve(),
        input_csv_inverted=(project_root / input_csv_inv_rel).resolve(),
        results_dir=(project_root / results_dir_rel).resolve(),
        inverted=inverted,
        overwrite=overwrite,
        seed=seed_raw,
        default_classifier=default_classifier,
        default_object=default_object,
        subdatasets=subdatasets,
        split_column=split_column,
        label_column=label_column,
        object_column=object_column,
        positive_label=positive_label,
        negative_label=negative_label,
        allowed_splits=allowed_splits,
        split_mode=split_mode,
        train_split_name=train_split_name,
        test_split_name=test_split_name,
        test_size=test_size,
        active_subset=active_subset,
        compare_subsets=compare_subsets,
        rf_params=rf_params,
        knn_params=knn_params,
        dt_params=dt_params,
        lr_params=lr_params,
        explainability_enabled=explainability_enabled,
        permutation_repeats=permutation_repeats,
        top_k_features=top_k_features,
        dpi=dpi,
        float_precision=float_precision,
        feature_name_template=feature_name_template,
        global_scale=global_scale,
        asymmetry_scales=asymmetry_scales,
        asymmetry_kinds=asymmetry_kinds,
        asymmetry_templates=asymmetry_templates,
    )


@dataclass(frozen=True, slots=True)
class DatasetPartition:
    """Immutable in-memory train/test partition for descriptor classification."""

    dataset_name: str
    object_name: str | None
    inverted: bool
    split_mode: str
    subset_name: str
    feature_columns: tuple[str, ...]
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_counts: dict[str, int]
    test_counts: dict[str, int]
    load_elapsed_s: float


def resolve_feature_groups(
    header: Sequence[str], cfg: ClassificationConfig
) -> dict[str, tuple[str, ...]]:
    """Build and validate canonical feature subsets and scale groups against CSV header.

    Args:
        header: Sequence of column names present in the descriptor CSV.
        cfg: Validated ClassificationConfig instance.

    Returns:
        Dictionary mapping group/subset names to ordered tuples of column names.

    Raises:
        ValueError: If any expected entropy or asymmetry column is missing from header.
    """
    header_set = set(header)
    all_scales = (cfg.global_scale, *cfg.asymmetry_scales)

    groups: dict[str, tuple[str, ...]] = {}
    entropy_cols_all: list[str] = []

    for g in all_scales:
        cols_g = tuple(entropy_column_names(g, cfg.feature_name_template))
        for c in cols_g:
            if c not in header_set:
                raise ValueError(
                    f"Required entropy column '{c}' missing from CSV header"
                )
        groups[f"E_{g}x{g}"] = cols_g
        entropy_cols_all.extend(cols_g)

    asym_cols_all = tuple(
        asymmetry_column_names(
            cfg.asymmetry_scales, cfg.asymmetry_kinds, cfg.asymmetry_templates
        )
    )
    for c in asym_cols_all:
        if c not in header_set:
            raise ValueError(f"Required asymmetry column '{c}' missing from CSV header")

    for g in cfg.asymmetry_scales:
        cols_asym_g = tuple(
            asymmetry_column_names((g,), cfg.asymmetry_kinds, cfg.asymmetry_templates)
        )
        groups[f"asym_{g}x{g}"] = cols_asym_g

    for kind in cfg.asymmetry_kinds:
        cols_kind = tuple(
            asymmetry_column_names(
                cfg.asymmetry_scales, (kind,), cfg.asymmetry_templates
            )
        )
        groups[f"asym_{kind}"] = cols_kind

    groups["x85_only"] = tuple(entropy_cols_all)
    groups["x9_only"] = asym_cols_all
    groups["x94_full"] = tuple(entropy_cols_all) + asym_cols_all

    return groups


def load_and_split_descriptors(
    cfg: ClassificationConfig,
    object_name: str | None = None,
    subset_name: str | None = None,
) -> DatasetPartition:
    """Load x94 CSV, filter allowed splits and object in memory, and partition into train/test.

    Args:
        cfg: Validated ClassificationConfig instance.
        object_name: Optional MVTec-AD object category override.
        subset_name: Optional feature subset override (e.g. 'x94_full', 'x85_only', 'x9_only').

    Returns:
        DatasetPartition ready for classifier training and evaluation.

    Raises:
        FileNotFoundError: If the effective CSV file does not exist.
        ValueError: If object_name or subset_name is unknown, labels are invalid,
            or resulting train/test splits contain fewer than 2 unique classes.
    """
    csv_path = cfg.effective_input_csv
    if not csv_path.is_file():
        raise FileNotFoundError(f"Descriptor CSV file not found: {csv_path}")

    start_t = time.perf_counter()
    df = pd.read_csv(csv_path)

    if cfg.split_column not in df.columns:
        raise ValueError(f"Configured split_column '{cfg.split_column}' not in CSV")
    if cfg.label_column not in df.columns:
        raise ValueError(f"Configured label_column '{cfg.label_column}' not in CSV")

    groups = resolve_feature_groups(list(df.columns), cfg)
    active_sub = subset_name if subset_name is not None else cfg.active_subset
    if active_sub not in groups:
        raise ValueError(
            f"Unknown feature subset '{active_sub}'. Available: {sorted(groups.keys())}"
        )
    feature_cols = groups[active_sub]

    # 1. Strictly filter allowed_splits (excludes 'ground_truth' masks in MVTec-AD)
    df = df[df[cfg.split_column].isin(cfg.allowed_splits)].copy()
    if df.empty:
        raise ValueError(
            f"No rows remain after filtering allowed_splits={cfg.allowed_splits}"
        )

    # 2. Filter by object_name when object_column is configured
    target_object: str | None = None
    if cfg.object_column:
        if cfg.object_column not in df.columns:
            raise ValueError(
                f"Configured object_column '{cfg.object_column}' not in CSV"
            )
        target_object = object_name if object_name is not None else cfg.default_object
        if target_object is None:
            raise ValueError(
                f"Dataset requires an object selection ('{cfg.object_column}'), "
                f"but neither --object nor default_object was provided."
            )
        if cfg.subdatasets and target_object not in cfg.subdatasets:
            raise ValueError(
                f"Unknown object_name '{target_object}'. Valid subdatasets: {cfg.subdatasets}"
            )
        df = df[df[cfg.object_column] == target_object].copy()
        if df.empty:
            raise ValueError(
                f"No rows found for {cfg.object_column} == '{target_object}'"
            )

    # 3. Validate and encode binary labels
    label_map = {cfg.negative_label: 0, cfg.positive_label: 1}
    unknown_labels = set(df[cfg.label_column].unique()) - set(label_map.keys())
    if unknown_labels:
        raise ValueError(
            f"Unexpected label values {sorted(unknown_labels)} in '{cfg.label_column}'. "
            f"Expected only {sorted(label_map.keys())}"
        )

    X_all = df[list(feature_cols)].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(X_all)):
        raise ValueError("Descriptor feature matrix contains NaN or infinite values")
    y_all = df[cfg.label_column].map(label_map).to_numpy(dtype=np.int64)

    # 4. Partition into train and test
    if cfg.split_mode == "presplit":
        train_mask = (df[cfg.split_column] == cfg.train_split_name).to_numpy()
        test_mask = (df[cfg.split_column] == cfg.test_split_name).to_numpy()
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]
    elif cfg.split_mode == "stratified":
        if len(np.unique(y_all)) < 2:
            raise ValueError(
                "Pooled dataset has fewer than 2 unique classes; cannot perform stratified split"
            )
        X_train, X_test, y_train, y_test = train_test_split(
            X_all,
            y_all,
            test_size=cfg.test_size,
            stratify=y_all,
            random_state=cfg.seed,
        )
    else:
        raise ValueError(f"Unsupported split_mode: {cfg.split_mode}")

    if len(np.unique(y_train)) < 2:
        raise ValueError(
            f"Training split has fewer than 2 unique classes "
            f"(counts: {cfg.negative_label}={int(np.sum(y_train == 0))}, "
            f"{cfg.positive_label}={int(np.sum(y_train == 1))}). "
            f"For MVTec-AD, use split_mode='stratified' because native 'train' has only '{cfg.negative_label}'."
        )
    if len(np.unique(y_test)) < 2:
        raise ValueError("Test split has fewer than 2 unique classes")

    train_counts = {
        cfg.negative_label: int(np.sum(y_train == 0)),
        cfg.positive_label: int(np.sum(y_train == 1)),
    }
    test_counts = {
        cfg.negative_label: int(np.sum(y_test == 0)),
        cfg.positive_label: int(np.sum(y_test == 1)),
    }
    load_elapsed_s = time.perf_counter() - start_t

    return DatasetPartition(
        dataset_name=cfg.dataset_name,
        object_name=target_object,
        inverted=cfg.inverted,
        split_mode=cfg.split_mode,
        subset_name=active_sub,
        feature_columns=feature_cols,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_counts=train_counts,
        test_counts=test_counts,
        load_elapsed_s=load_elapsed_s,
    )


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Immutable evaluation metrics and timing record for a trained classifier."""

    classifier_name: str
    subset_name: str
    n_features: int
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]
    fpr: np.ndarray
    tpr: np.ndarray
    pr_precision: np.ndarray
    pr_recall: np.ndarray
    train_elapsed_s: float
    inference_elapsed_s: float
    inference_ms_per_sample: float


def build_classifier(
    classifier_name: str, cfg: ClassificationConfig
) -> ClassifierMixin | Pipeline:
    """Instantiate configured Random Forest or k-NN classifier pipeline.

    Args:
        classifier_name: Identifier ('rf' or 'knn').
        cfg: Validated ClassificationConfig instance.

    Returns:
        Unfitted scikit-learn estimator or Pipeline.

    Raises:
        ValueError: If classifier_name is not supported.
    """
    cls_key = classifier_name.lower()
    if cls_key == "rf":
        rf = cfg.rf_params
        return RandomForestClassifier(
            n_estimators=rf.n_estimators,
            max_depth=rf.max_depth,
            min_samples_leaf=rf.min_samples_leaf,
            class_weight=rf.class_weight,
            random_state=cfg.seed,
            n_jobs=rf.n_jobs,
        )
    if cls_key == "dt":
        dt = cfg.dt_params
        return DecisionTreeClassifier(
            max_depth=dt.max_depth,
            min_samples_leaf=dt.min_samples_leaf,
            class_weight=dt.class_weight,
        )
    if cls_key == "lr":
        estimator = LogisticRegression(
            C=cfg.lr_params.C,
            penalty=cfg.lr_params.penalty,
            solver=cfg.lr_params.solver,
            class_weight=cfg.lr_params.class_weight,
        )
        if cfg.lr_params.scale_features:
            return Pipeline([("scaler", StandardScaler()), ("lr", estimator)])
        return estimator
    if cls_key == "knn":
        knn = cfg.knn_params
        estimator = KNeighborsClassifier(
            n_neighbors=knn.n_neighbors,
            weights=knn.weights,
            metric=knn.metric,
            p=knn.p,
            n_jobs=-1,
        )
        if knn.scale_features:
            return Pipeline([("scaler", StandardScaler()), ("knn", estimator)])
        return estimator
    raise ValueError(
        f"Unsupported classifier '{classifier_name}'. Choose 'rf' or 'knn' or 'dt' or 'lr'."
    )


def train_and_evaluate(
    partition: DatasetPartition,
    classifier_name: str,
    cfg: ClassificationConfig,
    subset_name: str | None = None,
) -> tuple[ClassifierMixin | Pipeline, ClassificationMetrics]:
    """Fit classifier on X_train and evaluate metrics and latency on X_test.

    Args:
        partition: DatasetPartition containing X_train [N_tr, F], y_train, X_test [N_te, F], y_test.
        classifier_name: 'rf' or 'knn'.
        cfg: Validated ClassificationConfig instance.
        subset_name: Optional override for the subset label recorded in metrics.

    Returns:
        Tuple of (fitted_model, ClassificationMetrics).
    """
    model = build_classifier(classifier_name, cfg)
    sub_label = subset_name if subset_name is not None else partition.subset_name

    t0_train = time.perf_counter()
    model.fit(partition.X_train, partition.y_train)
    train_elapsed_s = time.perf_counter() - t0_train

    t0_inf = time.perf_counter()
    y_pred = model.predict(partition.X_test)
    y_prob = model.predict_proba(partition.X_test)[:, 1]
    inference_elapsed_s = time.perf_counter() - t0_inf

    n_test = max(1, len(partition.y_test))
    inference_ms_per_sample = (inference_elapsed_s * 1000.0) / n_test

    acc = float(accuracy_score(partition.y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(partition.y_test, y_pred))
    prec = float(precision_score(partition.y_test, y_pred, zero_division=0))
    rec = float(recall_score(partition.y_test, y_pred, zero_division=0))
    f1 = float(f1_score(partition.y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(partition.y_test, y_prob))
    pr_auc = float(average_precision_score(partition.y_test, y_prob))

    cm_raw = confusion_matrix(partition.y_test, y_pred, labels=[0, 1])
    cm_tuple = (
        (int(cm_raw[0, 0]), int(cm_raw[0, 1])),
        (int(cm_raw[1, 0]), int(cm_raw[1, 1])),
    )
    fpr, tpr, _ = roc_curve(partition.y_test, y_prob)
    pr_precision, pr_recall, _ = precision_recall_curve(partition.y_test, y_prob)

    metrics = ClassificationMetrics(
        classifier_name=classifier_name.lower(),
        subset_name=sub_label,
        n_features=int(partition.X_train.shape[1]),
        accuracy=acc,
        balanced_accuracy=bal_acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        confusion_matrix=cm_tuple,
        fpr=np.asarray(fpr, dtype=np.float64),
        tpr=np.asarray(tpr, dtype=np.float64),
        pr_precision=np.asarray(pr_precision, dtype=np.float64),
        pr_recall=np.asarray(pr_recall, dtype=np.float64),
        train_elapsed_s=train_elapsed_s,
        inference_elapsed_s=inference_elapsed_s,
        inference_ms_per_sample=inference_ms_per_sample,
    )
    return model, metrics


def slice_partition_subset(
    base_partition: DatasetPartition,
    subset_name: str,
    groups: Mapping[str, tuple[str, ...]],
) -> DatasetPartition:
    """Slice an existing x94_full DatasetPartition to a target feature subset without re-splitting."""
    if subset_name not in groups:
        raise ValueError(
            f"Unknown subset '{subset_name}'. Available: {sorted(groups.keys())}"
        )
    target_cols = groups[subset_name]
    col_to_idx = {col: idx for idx, col in enumerate(base_partition.feature_columns)}
    indices = [col_to_idx[c] for c in target_cols]
    return DatasetPartition(
        dataset_name=base_partition.dataset_name,
        object_name=base_partition.object_name,
        inverted=base_partition.inverted,
        split_mode=base_partition.split_mode,
        subset_name=subset_name,
        feature_columns=target_cols,
        X_train=base_partition.X_train[:, indices],
        y_train=base_partition.y_train,
        X_test=base_partition.X_test[:, indices],
        y_test=base_partition.y_test,
        train_counts=base_partition.train_counts,
        test_counts=base_partition.test_counts,
        load_elapsed_s=base_partition.load_elapsed_s,
    )


def run_subset_ablation(
    cfg: ClassificationConfig,
    object_name: str | None,
    classifier_name: str,
    base_partition: DatasetPartition | None = None,
) -> tuple[ClassificationMetrics, ...]:
    """Evaluate configured compare_subsets on the exact same train/test split.

    Args:
        cfg: Validated ClassificationConfig instance.
        object_name: Optional MVTec-AD object category.
        classifier_name: 'rf' or 'knn'.
        base_partition: Optional pre-loaded 'x94_full' partition to reuse.

    Returns:
        Tuple of ClassificationMetrics in the order of cfg.compare_subsets.
    """
    part_full = (
        base_partition
        if base_partition is not None and base_partition.subset_name == "x94_full"
        else load_and_split_descriptors(
            cfg, object_name=object_name, subset_name="x94_full"
        )
    )
    groups = resolve_feature_groups(part_full.feature_columns, cfg)

    results: list[ClassificationMetrics] = []
    for sub_name in cfg.compare_subsets:
        sub_part = slice_partition_subset(part_full, sub_name, groups)
        _, metrics = train_and_evaluate(
            sub_part, classifier_name, cfg, subset_name=sub_name
        )
        results.append(metrics)
    return tuple(results)


@dataclass(frozen=True, slots=True)
class GroupImportance:
    """Aggregated importance statistics for a feature family, scale, or symmetry axis."""

    group_name: str
    n_features: int
    total_importance: float
    share_pct: float
    mean_importance_per_feature: float


@dataclass(frozen=True, slots=True)
class ExplainabilityReport:
    """Comprehensive explainability analysis for x94 descriptors."""

    method: str
    feature_names: tuple[str, ...]
    importances: np.ndarray
    top_k_indices: tuple[int, ...]
    family_breakdown: tuple[GroupImportance, ...]
    scale_breakdown: tuple[GroupImportance, ...]
    asymmetry_kind_breakdown: tuple[GroupImportance, ...]
    spatial_heatmaps: dict[int, np.ndarray]


def _build_group_breakdown(
    group_specs: Sequence[tuple[str, tuple[str, ...]]],
    col_to_idx: Mapping[str, int],
    importances: np.ndarray,
    denom_total: float,
) -> tuple[GroupImportance, ...]:
    """Compute GroupImportance items for an ordered list of (group_name, column_names)."""
    out: list[GroupImportance] = []
    for g_name, cols in group_specs:
        present_indices = [col_to_idx[c] for c in cols if c in col_to_idx]
        n_feat = len(present_indices)
        if n_feat == 0:
            continue
        tot = float(np.sum(importances[present_indices]))
        share = (tot / denom_total) * 100.0 if denom_total > 0.0 else 0.0
        mean_per_feat = tot / n_feat
        out.append(
            GroupImportance(
                group_name=g_name,
                n_features=n_feat,
                total_importance=tot,
                share_pct=share,
                mean_importance_per_feature=mean_per_feat,
            )
        )
    return tuple(out)


def compute_explainability(
    model: object,
    partition: DatasetPartition,
    cfg: ClassificationConfig,
) -> ExplainabilityReport:
    """Compute feature importances, family/scale/axis breakdowns, and 2D spatial heatmaps.

    Uses Gini impurity reduction (MDI) when the fitted model exposes `feature_importances_`
    (e.g., RandomForestClassifier) and falls back to test-set permutation importance
    (scoring='f1') for distance-based models such as k-NN.

    Args:
        model: Fitted classifier or Pipeline.
        partition: DatasetPartition over which the model was trained/evaluated.
        cfg: Validated ClassificationConfig instance.

    Returns:
        ExplainabilityReport with normalized shares, per-feature means, and 2D heatmaps.
    """
    if hasattr(model, "feature_importances_"):
        raw_imp = np.asarray(getattr(model, "feature_importances_"), dtype=np.float64)
        method = "gini_mdi"
    else:
        perm = permutation_importance(
            model,
            partition.X_test,
            partition.y_test,
            scoring="average_precision",
            n_repeats=cfg.permutation_repeats,
            random_state=cfg.seed,
        )
        raw_imp = np.maximum(np.asarray(perm.importances_mean, dtype=np.float64), 0.0)
        method = "permutation_importance"
        if float(np.sum(raw_imp)) <= 1e-12 or int(np.count_nonzero(raw_imp)) < 3:
            # In high-dimensional k-NN (94-D), permuting a single coordinate at a time
            # may not alter neighbor rankings when multiple features carry redundant signal.
            # Fall back to standardized Fisher inter-class distance ratio on X_train.
            x0 = partition.X_train[partition.y_train == 0]
            x1 = partition.X_train[partition.y_train == 1]
            mu0 = np.mean(x0, axis=0)
            mu1 = np.mean(x1, axis=0)
            var0 = np.var(x0, axis=0)
            var1 = np.var(x1, axis=0)
            raw_imp = np.square(mu1 - mu0) / (var0 + var1 + 1e-12)
            method = "knn_fisher_distance"

    total_imp = float(np.sum(raw_imp))
    if total_imp > 0.0:
        importances = raw_imp / total_imp
    else:
        importances = np.zeros_like(raw_imp)
    norm_total = float(np.sum(importances))

    k = min(cfg.top_k_features, len(importances))
    top_k_indices = tuple(int(i) for i in np.argsort(importances)[::-1][:k])

    groups = resolve_feature_groups(partition.feature_columns, cfg)
    col_to_idx = {col: idx for idx, col in enumerate(partition.feature_columns)}

    family_specs = [
        ("x85_entropy", groups["x85_only"]),
        ("x9_asymmetry", groups["x9_only"]),
    ]
    family_breakdown = _build_group_breakdown(
        family_specs, col_to_idx, importances, norm_total
    )

    all_entropy_scales = (cfg.global_scale, *cfg.asymmetry_scales)
    scale_specs: list[tuple[str, tuple[str, ...]]] = [
        (f"E_{g}x{g}", groups[f"E_{g}x{g}"]) for g in all_entropy_scales
    ] + [(f"asym_{g}x{g}", groups[f"asym_{g}x{g}"]) for g in cfg.asymmetry_scales]
    scale_breakdown = _build_group_breakdown(
        scale_specs, col_to_idx, importances, norm_total
    )

    asym_total = float(
        np.sum(
            [importances[col_to_idx[c]] for c in groups["x9_only"] if c in col_to_idx]
        )
    )
    kind_specs: list[tuple[str, tuple[str, ...]]] = [
        (f"asym_{kind}", groups[f"asym_{kind}"]) for kind in cfg.asymmetry_kinds
    ]
    asymmetry_kind_breakdown = _build_group_breakdown(
        kind_specs, col_to_idx, importances, asym_total
    )

    spatial_heatmaps: dict[int, np.ndarray] = {}
    for g in cfg.asymmetry_scales:
        cols_g = groups[f"E_{g}x{g}"]
        if all(c in col_to_idx for c in cols_g):
            vals_g = np.array(
                [importances[col_to_idx[c]] for c in cols_g], dtype=np.float64
            )
            spatial_heatmaps[g] = vals_g.reshape((g, g))

    return ExplainabilityReport(
        method=method,
        feature_names=partition.feature_columns,
        importances=importances,
        top_k_indices=top_k_indices,
        family_breakdown=family_breakdown,
        scale_breakdown=scale_breakdown,
        asymmetry_kind_breakdown=asymmetry_kind_breakdown,
        spatial_heatmaps=spatial_heatmaps,
    )


@dataclass(frozen=True, slots=True)
class EvaluationRunSummary:
    """Summary returned by a full or dry-run evaluation pipeline execution."""

    dataset_name: str
    object_name: str | None
    inverted: bool
    classifier_name: str
    split_mode: str
    train_counts: dict[str, int]
    test_counts: dict[str, int]
    active_metrics: ClassificationMetrics | None
    ablation_metrics: tuple[ClassificationMetrics, ...]
    explainability: ExplainabilityReport | None
    output_dir: Path | None
    dry_run: bool


def _feature_family_and_scale(col_name: str) -> tuple[str, str]:
    """Classify a column name into (family, scale_group)."""
    if col_name.startswith("asym_"):
        parts = col_name.split("_")
        scale_tag = parts[-1] if len(parts) >= 3 else "unknown"
        return ("x9_asymmetry", f"asym_{scale_tag}")
    if col_name.startswith("E_"):
        parts = col_name.split("_")
        scale_tag = parts[1] if len(parts) >= 2 else "unknown"
        return ("x85_entropy", f"E_{scale_tag}")
    return ("other", "other")


def run_evaluation_pipeline(
    cfg: ClassificationConfig,
    classifier_name: str | None = None,
    object_name: str | None = None,
    dry_run: bool = False,
) -> EvaluationRunSummary:
    """Execute end-to-end descriptor loading, classification, ablation, explainability, and plotting.

    Args:
        cfg: Validated ClassificationConfig instance.
        classifier_name: Optional override ('rf' or 'knn').
        object_name: Optional MVTec-AD object override.
        dry_run: When True, validates data loading and partitioning without training or writing files.

    Returns:
        EvaluationRunSummary containing metrics, breakdowns, and output directory.
    """
    from src.models.classification_plots import (
        plot_confusion_and_curves,
        plot_explainability_breakdown,
        plot_spatial_importance_heatmaps,
        plot_subset_ablation,
    )

    cls_name = (classifier_name or cfg.default_classifier).lower()
    part_full = load_and_split_descriptors(
        cfg, object_name=object_name, subset_name="x94_full"
    )

    if dry_run:
        return EvaluationRunSummary(
            dataset_name=part_full.dataset_name,
            object_name=part_full.object_name,
            inverted=part_full.inverted,
            classifier_name=cls_name,
            split_mode=part_full.split_mode,
            train_counts=part_full.train_counts,
            test_counts=part_full.test_counts,
            active_metrics=None,
            ablation_metrics=(),
            explainability=None,
            output_dir=None,
            dry_run=True,
        )

    out_dir = cfg.effective_results_dir(part_full.object_name, cls_name)
    if out_dir.exists() and not cfg.overwrite:
        raise FileExistsError(
            f"Output directory already exists and overwrite=False: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = resolve_feature_groups(part_full.feature_columns, cfg)
    active_part = (
        part_full
        if cfg.active_subset == "x94_full"
        else slice_partition_subset(part_full, cfg.active_subset, groups)
    )

    fitted_model, active_metrics = train_and_evaluate(
        active_part, cls_name, cfg, subset_name=cfg.active_subset
    )
    ablation_metrics = run_subset_ablation(
        cfg,
        object_name=part_full.object_name,
        classifier_name=cls_name,
        base_partition=part_full,
    )

    # Compute explainability on the full x94 descriptor so all 85 + 9 features are analyzed together
    if cfg.active_subset == "x94_full":
        model_for_exp = fitted_model
    else:
        model_for_exp, _ = train_and_evaluate(
            part_full, cls_name, cfg, subset_name="x94_full"
        )
    exp_report = compute_explainability(model_for_exp, part_full, cfg)

    variant_label = (
        f"{part_full.dataset_name}{' (Inverted)' if part_full.inverted else ''}"
    )
    if part_full.object_name:
        title_prefix = f"{variant_label} [{part_full.object_name}]"
    else:
        title_prefix = variant_label

    # 1. Write ablation_table.csv
    p_fmt = f"%.{cfg.float_precision}f"
    ablation_csv_path = out_dir / "ablation_table.csv"
    with ablation_csv_path.open("w", newline="", encoding="utf-8") as f_ab:
        writer = csv.writer(f_ab)
        writer.writerow(
            [
                "subset_name",
                "n_features",
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "train_elapsed_s",
                "inference_elapsed_s",
                "inference_ms_per_sample",
            ]
        )
        for m in ablation_metrics:
            writer.writerow(
                [
                    m.subset_name,
                    m.n_features,
                    p_fmt % m.accuracy,
                    p_fmt % m.balanced_accuracy,
                    p_fmt % m.precision,
                    p_fmt % m.recall,
                    p_fmt % m.f1,
                    p_fmt % m.roc_auc,
                    p_fmt % m.pr_auc,
                    p_fmt % m.train_elapsed_s,
                    p_fmt % m.inference_elapsed_s,
                    p_fmt % m.inference_ms_per_sample,
                ]
            )

    # 2. Write feature_importance.csv (sorted descending)
    feat_imp_csv_path = out_dir / "feature_importance.csv"
    sorted_indices = np.argsort(exp_report.importances)[::-1]
    with feat_imp_csv_path.open("w", newline="", encoding="utf-8") as f_fi:
        writer = csv.writer(f_fi)
        writer.writerow(
            ["rank", "feature_name", "family", "scale_group", "importance", "share_pct"]
        )
        for rank_idx, feat_idx in enumerate(sorted_indices, start=1):
            fname = exp_report.feature_names[int(feat_idx)]
            fam, sc_grp = _feature_family_and_scale(fname)
            imp_val = float(exp_report.importances[int(feat_idx)])
            writer.writerow(
                [
                    rank_idx,
                    fname,
                    fam,
                    sc_grp,
                    p_fmt % imp_val,
                    p_fmt % (imp_val * 100.0),
                ]
            )

    # 3. Write scale_and_family_breakdown.csv
    breakdown_csv_path = out_dir / "scale_and_family_breakdown.csv"
    with breakdown_csv_path.open("w", newline="", encoding="utf-8") as f_bd:
        writer = csv.writer(f_bd)
        writer.writerow(
            [
                "category",
                "group_name",
                "n_features",
                "total_importance",
                "share_pct",
                "mean_importance_per_feature_pct",
            ]
        )
        for cat_label, items in [
            ("family", exp_report.family_breakdown),
            ("scale", exp_report.scale_breakdown),
            ("asymmetry_axis", exp_report.asymmetry_kind_breakdown),
        ]:
            for g in items:
                writer.writerow(
                    [
                        cat_label,
                        g.group_name,
                        g.n_features,
                        p_fmt % g.total_importance,
                        p_fmt % g.share_pct,
                        p_fmt % (g.mean_importance_per_feature * 100.0),
                    ]
                )

    # 4. Write metrics_summary.json
    by_sub = {m.subset_name: m for m in ablation_metrics}
    symmetry_gain_f1 = (
        round(by_sub["x94_full"].f1 - by_sub["x85_only"].f1, cfg.float_precision)
        if "x94_full" in by_sub and "x85_only" in by_sub
        else None
    )
    symmetry_gain_roc_auc = (
        round(
            by_sub["x94_full"].roc_auc - by_sub["x85_only"].roc_auc, cfg.float_precision
        )
        if "x94_full" in by_sub and "x85_only" in by_sub
        else None
    )

    scale_candidates = [m for m in ablation_metrics if m.subset_name.startswith("E_")]
    best_single_entropy_scale = (
        max(scale_candidates, key=lambda m: (m.f1, m.roc_auc)).subset_name
        if scale_candidates
        else None
    )

    summary_payload = {
        "dataset_name": part_full.dataset_name,
        "object_name": part_full.object_name,
        "inverted": part_full.inverted,
        "source_csv": str(cfg.effective_input_csv),
        "split_mode": part_full.split_mode,
        "seed": cfg.seed,
        "classifier": cls_name,
        "train_counts": part_full.train_counts,
        "test_counts": part_full.test_counts,
        "timings": {
            "T_data_load_s": round(part_full.load_elapsed_s, cfg.float_precision),
            "T_model_train_s": round(
                active_metrics.train_elapsed_s, cfg.float_precision
            ),
            "T_model_inference_s": round(
                active_metrics.inference_elapsed_s, cfg.float_precision
            ),
            "inference_ms_per_sample": round(
                active_metrics.inference_ms_per_sample, cfg.float_precision
            ),
        },
        "active_subset_metrics": {
            "subset_name": active_metrics.subset_name,
            "n_features": active_metrics.n_features,
            "accuracy": round(active_metrics.accuracy, cfg.float_precision),
            "balanced_accuracy": round(
                active_metrics.balanced_accuracy, cfg.float_precision
            ),
            "precision": round(active_metrics.precision, cfg.float_precision),
            "recall": round(active_metrics.recall, cfg.float_precision),
            "f1": round(active_metrics.f1, cfg.float_precision),
            "roc_auc": round(active_metrics.roc_auc, cfg.float_precision),
            "pr_auc": round(active_metrics.pr_auc, cfg.float_precision),
            "confusion_matrix": [list(row) for row in active_metrics.confusion_matrix],
        },
        "research_findings": {
            "symmetry_gain_f1_x94_minus_x85": symmetry_gain_f1,
            "symmetry_gain_roc_auc_x94_minus_x85": symmetry_gain_roc_auc,
            "best_single_entropy_scale_by_f1": best_single_entropy_scale,
            "explainability_method": exp_report.method,
            "family_breakdown": [
                {
                    "group_name": g.group_name,
                    "n_features": g.n_features,
                    "share_pct": round(g.share_pct, 4),
                    "mean_importance_per_feature_pct": round(
                        g.mean_importance_per_feature * 100.0, 4
                    ),
                }
                for g in exp_report.family_breakdown
            ],
            "scale_breakdown": [
                {
                    "group_name": g.group_name,
                    "n_features": g.n_features,
                    "share_pct": round(g.share_pct, 4),
                    "mean_importance_per_feature_pct": round(
                        g.mean_importance_per_feature * 100.0, 4
                    ),
                }
                for g in exp_report.scale_breakdown
            ],
            "asymmetry_axis_breakdown": [
                {
                    "group_name": g.group_name,
                    "n_features": g.n_features,
                    "share_within_x9_pct": round(g.share_pct, 4),
                }
                for g in exp_report.asymmetry_kind_breakdown
            ],
            "top_features": [
                {
                    "rank": idx + 1,
                    "feature_name": exp_report.feature_names[i],
                    "share_pct": round(float(exp_report.importances[i]) * 100.0, 4),
                }
                for idx, i in enumerate(exp_report.top_k_indices)
            ],
        },
    }
    with (out_dir / "metrics_summary.json").open("w", encoding="utf-8") as f_json:
        json.dump(summary_payload, f_json, indent=2)

    # 5. Render the 4 diagnostic figures
    plot_confusion_and_curves(
        active_metrics,
        title_prefix=title_prefix,
        out_path=out_dir / "confusion_matrix_and_curves.png",
        dpi=cfg.dpi,
    )
    plot_subset_ablation(
        ablation_metrics,
        title_prefix=title_prefix,
        out_path=out_dir / "descriptor_ablation_x85_x9_x94.png",
        dpi=cfg.dpi,
    )
    plot_explainability_breakdown(
        exp_report,
        title_prefix=title_prefix,
        out_path=out_dir / "explainability_importance_breakdown.png",
        dpi=cfg.dpi,
    )
    plot_spatial_importance_heatmaps(
        exp_report,
        title_prefix=title_prefix,
        out_path=out_dir / "spatial_importance_heatmaps.png",
        dpi=cfg.dpi,
    )

    return EvaluationRunSummary(
        dataset_name=part_full.dataset_name,
        object_name=part_full.object_name,
        inverted=part_full.inverted,
        classifier_name=cls_name,
        split_mode=part_full.split_mode,
        train_counts=part_full.train_counts,
        test_counts=part_full.test_counts,
        active_metrics=active_metrics,
        ablation_metrics=ablation_metrics,
        explainability=exp_report,
        output_dir=out_dir,
        dry_run=False,
    )
