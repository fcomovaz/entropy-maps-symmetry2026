"""Configuration loader and schema validation for lightweight CNN experiments.

Follows the project's strict fail-fast TOML validation principles and frozen dataclass architecture.
All filesystem paths are resolved dynamically relative to the project root.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Mapping


def _require(table: Mapping[str, object], key: str, where: str) -> object:
    """Extract required key from dictionary or raise KeyError with context."""
    if key not in table:
        raise KeyError(f"Missing required key '{key}' in {where}")
    return table[key]


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Configuration for dataset ingestion, indexing, and splitting."""

    split_column: str
    label_column: str
    object_column: str
    positive_label: str
    negative_label: str
    split_mode: str
    train_split_name: str
    test_split_name: str
    test_size: float
    val_size: float
    image_size: int
    normalize_mean: float
    normalize_std: float
    use_metadata_index: bool
    metadata_csv_key: str
    metadata_csv_key_inverted: str
    allowed_splits: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Hyperparameters and execution options for CNN training."""

    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    optimizer: str
    scheduler: str
    early_stopping_patience: int
    class_weighting: str
    num_workers: int
    device: str
    amp: bool
    deterministic: bool
    threshold_mode: str


@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    """Label-preserving data augmentations for training."""

    hflip: bool
    vflip: bool
    rotate_deg: int
    brightness: float


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Settings for standalone inference timing and FLOP profiling."""

    warmup_iters: int
    timed_iters: int
    batch_sizes: tuple[int, ...]
    devices: tuple[str, ...]
    threads: int
    measure_flops: bool


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Output directories and reporting artifacts configuration."""

    results_dir: Path
    dpi: int
    float_precision: int
    save_checkpoints: bool


@dataclass(frozen=True, slots=True)
class CnnConfig:
    """Root configuration aggregating dataset, training, benchmarking, and output options."""

    dataset_key: str
    dataset_name: str
    preprocessed_dir: Path
    preprocessed_dir_inverted: Path
    metadata_csv_path: Path
    metadata_csv_path_inverted: Path
    results_dir: Path
    inverted: bool
    overwrite: bool
    seed: int
    seeds: tuple[int, ...]
    default_model: str
    models: tuple[str, ...]
    default_object: str | None
    subdatasets: tuple[str, ...]
    data: DataConfig
    training: TrainingConfig
    augmentation: AugmentationConfig
    benchmark: BenchmarkConfig
    output: OutputConfig

    @property
    def effective_preprocessed_dir(self) -> Path:
        """Return active preprocessed image directory according to inversion flag."""
        return self.preprocessed_dir_inverted if self.inverted else self.preprocessed_dir

    @property
    def effective_metadata_csv(self) -> Path:
        """Return active metadata CSV path according to inversion flag."""
        return (
            self.metadata_csv_path_inverted
            if self.inverted
            else self.metadata_csv_path
        )

    @property
    def effective_dataset_tag(self) -> str:
        """Return dataset folder name including '-inverted' suffix when active."""
        base_tag = self.dataset_name.lower()
        return f"{base_tag}-inverted" if self.inverted else base_tag

    def effective_results_dir(
        self, object_name: str | None, model_name: str
    ) -> Path:
        """Resolve output directory for a specific object and model."""
        target_tag = object_name if object_name else "full"
        return (
            self.results_dir
            / self.effective_dataset_tag
            / target_tag
            / model_name.lower()
        )


def load_cnn_config(config_path: Path, project_root: Path) -> CnnConfig:
    """Load, resolve, and validate CNN experiment configuration from TOML files.

    Args:
        config_path: Path to the CNN task TOML file.
        project_root: Repository root directory for anchoring relative paths.

    Returns:
        Validated CnnConfig instance.

    Raises:
        FileNotFoundError: If configuration files or base configuration do not exist.
        KeyError: If required sections or keys are missing.
        ValueError: If parameter values violate type or domain constraints.
    """
    resolved_cfg_path = (project_root / config_path).resolve()
    if not resolved_cfg_path.is_file():
        raise FileNotFoundError(f"CNN config file not found: {resolved_cfg_path}")

    with resolved_cfg_path.open("rb") as f:
        raw_cfg = tomllib.load(f)

    cnn_table_obj = _require(raw_cfg, "cnn", "CNN config")
    if not isinstance(cnn_table_obj, dict):
        raise ValueError("[cnn] section must be a table")
    cnn_table: dict[str, object] = cnn_table_obj

    base_config_rel = str(_require(cnn_table, "base_config", "[cnn]"))
    dataset_key = str(_require(cnn_table, "dataset_key", "[cnn]"))
    seed_raw = _require(cnn_table, "seed", "[cnn]")
    raw_seeds = _require(cnn_table, "seeds", "[cnn]")
    inverted = _require(cnn_table, "inverted", "[cnn]")
    overwrite = _require(cnn_table, "overwrite", "[cnn]")
    default_model = str(_require(cnn_table, "default_model", "[cnn]")).lower()
    raw_models = _require(cnn_table, "models", "[cnn]")
    default_object_raw = cnn_table.get("default_object")
    default_object = str(default_object_raw) if default_object_raw is not None else None

    if not isinstance(inverted, bool):
        raise ValueError("'inverted' in [cnn] must be a boolean")
    if not isinstance(overwrite, bool):
        raise ValueError("'overwrite' in [cnn] must be a boolean")
    if isinstance(seed_raw, bool) or not isinstance(seed_raw, int):
        raise ValueError("'seed' in [cnn] must be an integer")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError("'seeds' in [cnn] must be a non-empty list of integers")
    seeds = tuple(int(s) for s in raw_seeds)
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("'models' in [cnn] must be a non-empty list of strings")
    models = tuple(str(m) for m in raw_models)

    # 1. Base config resolution
    resolved_base_path = (project_root / base_config_rel).resolve()
    if not resolved_base_path.is_file():
        raise FileNotFoundError(f"Base config file not found: {resolved_base_path}")

    with resolved_base_path.open("rb") as f_base:
        base_data = tomllib.load(f_base)

    datasets_table = _require(base_data, "datasets", "base config")
    if not isinstance(datasets_table, dict):
        raise ValueError("[datasets] in base config must be a table")

    dataset_def = _require(datasets_table, dataset_key, f"[datasets.{dataset_key}]")
    if not isinstance(dataset_def, dict):
        raise ValueError(f"[datasets.{dataset_key}] must be a table")

    dataset_name = str(_require(dataset_def, "name", f"[datasets.{dataset_key}]"))
    out_path_rel = str(_require(dataset_def, "output_path", f"[datasets.{dataset_key}]"))
    out_inv_path_rel = str(
        _require(dataset_def, "output_path_inverted", f"[datasets.{dataset_key}]")
    )

    subdatasets_raw = dataset_def.get("subdatasets", [])
    subdatasets = (
        tuple(str(s) for s in subdatasets_raw) if isinstance(subdatasets_raw, list) else ()
    )

    if default_object is not None and subdatasets and default_object not in subdatasets:
        raise ValueError(
            f"default_object '{default_object}' not found in subdatasets {subdatasets}"
        )

    # 2. Data section
    data_table_obj = _require(cnn_table, "data", "[cnn]")
    if not isinstance(data_table_obj, dict):
        raise ValueError("[cnn.data] must be a table")
    data_table: dict[str, object] = data_table_obj

    split_column = str(_require(data_table, "split_column", "[cnn.data]"))
    label_column = str(_require(data_table, "label_column", "[cnn.data]"))
    object_column = str(_require(data_table, "object_column", "[cnn.data]"))
    positive_label = str(_require(data_table, "positive_label", "[cnn.data]"))
    negative_label = str(_require(data_table, "negative_label", "[cnn.data]"))
    split_mode = str(_require(data_table, "split_mode", "[cnn.data]")).lower()
    train_split_name = str(_require(data_table, "train_split_name", "[cnn.data]"))
    test_split_name = str(_require(data_table, "test_split_name", "[cnn.data]"))
    test_size = float(_require(data_table, "test_size", "[cnn.data]"))
    val_size = float(_require(data_table, "val_size", "[cnn.data]"))
    image_size = int(_require(data_table, "image_size", "[cnn.data]"))
    normalize_mean = float(_require(data_table, "normalize_mean", "[cnn.data]"))
    normalize_std = float(_require(data_table, "normalize_std", "[cnn.data]"))
    use_metadata_index = bool(_require(data_table, "use_metadata_index", "[cnn.data]"))
    meta_key = str(_require(data_table, "metadata_csv_key", "[cnn.data]"))
    meta_key_inv = str(_require(data_table, "metadata_csv_key_inverted", "[cnn.data]"))
    raw_allowed_splits = _require(data_table, "allowed_splits", "[cnn.data]")

    if not isinstance(raw_allowed_splits, list) or not raw_allowed_splits:
        raise ValueError("'allowed_splits' in [cnn.data] must be a non-empty list")
    allowed_splits = tuple(str(s) for s in raw_allowed_splits)

    if split_mode not in {"presplit", "stratified"}:
        raise ValueError(
            f"'split_mode' must be 'presplit' or 'stratified', got '{split_mode}'"
        )
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"'test_size' must be in (0.0, 1.0), got {test_size}")
    if not (0.0 <= val_size < 1.0):
        raise ValueError(f"'val_size' must be in [0.0, 1.0), got {val_size}")
    if image_size < 16:
        raise ValueError(f"'image_size' must be >= 16, got {image_size}")
    if normalize_std <= 0:
        raise ValueError(f"'normalize_std' must be > 0, got {normalize_std}")
    if positive_label == negative_label:
        raise ValueError("positive_label and negative_label must be distinct")

    meta_csv_rel = str(_require(dataset_def, meta_key, f"[datasets.{dataset_key}]"))
    meta_csv_inv_rel = str(_require(dataset_def, meta_key_inv, f"[datasets.{dataset_key}]"))

    data_cfg = DataConfig(
        split_column=split_column,
        label_column=label_column,
        object_column=object_column,
        positive_label=positive_label,
        negative_label=negative_label,
        split_mode=split_mode,
        train_split_name=train_split_name,
        test_split_name=test_split_name,
        test_size=test_size,
        val_size=val_size,
        image_size=image_size,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        use_metadata_index=use_metadata_index,
        metadata_csv_key=meta_key,
        metadata_csv_key_inverted=meta_key_inv,
        allowed_splits=allowed_splits,
    )

    # 3. Training section
    training_table_obj = _require(cnn_table, "training", "[cnn]")
    if not isinstance(training_table_obj, dict):
        raise ValueError("[cnn.training] must be a table")
    training_table: dict[str, object] = training_table_obj

    epochs = int(_require(training_table, "epochs", "[cnn.training]"))
    batch_size = int(_require(training_table, "batch_size", "[cnn.training]"))
    learning_rate = float(_require(training_table, "learning_rate", "[cnn.training]"))
    weight_decay = float(_require(training_table, "weight_decay", "[cnn.training]"))
    optimizer = str(_require(training_table, "optimizer", "[cnn.training]")).lower()
    scheduler = str(_require(training_table, "scheduler", "[cnn.training]")).lower()
    early_stopping_patience = int(
        _require(training_table, "early_stopping_patience", "[cnn.training]")
    )
    class_weighting = str(
        _require(training_table, "class_weighting", "[cnn.training]")
    ).lower()
    num_workers = int(_require(training_table, "num_workers", "[cnn.training]"))
    device = str(_require(training_table, "device", "[cnn.training]")).lower()
    amp = bool(_require(training_table, "amp", "[cnn.training]"))
    deterministic = bool(_require(training_table, "deterministic", "[cnn.training]"))
    threshold_mode = str(training_table.get("threshold_mode", "val_f1")).lower()
    if threshold_mode not in {"val_f1", "fixed_05", "val_bal_acc"}:
        raise ValueError(
            f"Unsupported threshold_mode '{threshold_mode}'. Choose 'val_f1', 'fixed_05', or 'val_bal_acc'."
        )

    if epochs < 1:
        raise ValueError(f"'epochs' must be >= 1, got {epochs}")
    if batch_size < 1:
        raise ValueError(f"'batch_size' must be >= 1, got {batch_size}")
    if learning_rate <= 0:
        raise ValueError(f"'learning_rate' must be > 0, got {learning_rate}")
    if optimizer not in {"adamw", "adam"}:
        raise ValueError(f"Unsupported optimizer '{optimizer}'")
    if scheduler not in {"cosine", "step", "none"}:
        raise ValueError(f"Unsupported scheduler '{scheduler}'")

    train_cfg = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping_patience=early_stopping_patience,
        class_weighting=class_weighting,
        num_workers=num_workers,
        device=device,
        amp=amp,
        deterministic=deterministic,
        threshold_mode=threshold_mode,
    )

    # 4. Augmentation section
    aug_table_obj = _require(cnn_table, "augmentation", "[cnn]")
    if not isinstance(aug_table_obj, dict):
        raise ValueError("[cnn.augmentation] must be a table")
    aug_table: dict[str, object] = aug_table_obj

    hflip = bool(_require(aug_table, "hflip", "[cnn.augmentation]"))
    vflip = bool(_require(aug_table, "vflip", "[cnn.augmentation]"))
    rotate_deg = int(_require(aug_table, "rotate_deg", "[cnn.augmentation]"))
    brightness = float(_require(aug_table, "brightness", "[cnn.augmentation]"))

    aug_cfg = AugmentationConfig(
        hflip=hflip,
        vflip=vflip,
        rotate_deg=rotate_deg,
        brightness=brightness,
    )

    # 5. Benchmark section
    bench_table_obj = _require(cnn_table, "benchmark", "[cnn]")
    if not isinstance(bench_table_obj, dict):
        raise ValueError("[cnn.benchmark] must be a table")
    bench_table: dict[str, object] = bench_table_obj

    warmup_iters = int(_require(bench_table, "warmup_iters", "[cnn.benchmark]"))
    timed_iters = int(_require(bench_table, "timed_iters", "[cnn.benchmark]"))
    raw_b_sizes = _require(bench_table, "batch_sizes", "[cnn.benchmark]")
    raw_b_devices = _require(bench_table, "devices", "[cnn.benchmark]")
    threads = int(_require(bench_table, "threads", "[cnn.benchmark]"))
    measure_flops = bool(_require(bench_table, "measure_flops", "[cnn.benchmark]"))

    if not isinstance(raw_b_sizes, list) or not raw_b_sizes:
        raise ValueError("'batch_sizes' in [cnn.benchmark] must be a non-empty list")
    b_sizes = tuple(int(b) for b in raw_b_sizes)
    if not isinstance(raw_b_devices, list) or not raw_b_devices:
        raise ValueError("'devices' in [cnn.benchmark] must be a non-empty list")
    b_devices = tuple(str(d) for d in raw_b_devices)

    bench_cfg = BenchmarkConfig(
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        batch_sizes=b_sizes,
        devices=b_devices,
        threads=threads,
        measure_flops=measure_flops,
    )

    # 6. Output section
    out_table_obj = _require(cnn_table, "output", "[cnn]")
    if not isinstance(out_table_obj, dict):
        raise ValueError("[cnn.output] must be a table")
    out_table: dict[str, object] = out_table_obj

    res_dir_rel = str(_require(out_table, "results_dir", "[cnn.output]"))
    dpi = int(_require(out_table, "dpi", "[cnn.output]"))
    float_precision = int(_require(out_table, "float_precision", "[cnn.output]"))
    save_checkpoints = bool(_require(out_table, "save_checkpoints", "[cnn.output]"))

    output_cfg = OutputConfig(
        results_dir=(project_root / res_dir_rel).resolve(),
        dpi=dpi,
        float_precision=float_precision,
        save_checkpoints=save_checkpoints,
    )

    return CnnConfig(
        dataset_key=dataset_key,
        dataset_name=dataset_name,
        preprocessed_dir=(project_root / out_path_rel).resolve(),
        preprocessed_dir_inverted=(project_root / out_inv_path_rel).resolve(),
        metadata_csv_path=(project_root / meta_csv_rel).resolve(),
        metadata_csv_path_inverted=(project_root / meta_csv_inv_rel).resolve(),
        results_dir=(project_root / res_dir_rel).resolve(),
        inverted=inverted,
        overwrite=overwrite,
        seed=seed_raw,
        seeds=seeds,
        default_model=default_model,
        models=models,
        default_object=default_object,
        subdatasets=subdatasets,
        data=data_cfg,
        training=train_cfg,
        augmentation=aug_cfg,
        benchmark=bench_cfg,
        output=output_cfg,
    )
