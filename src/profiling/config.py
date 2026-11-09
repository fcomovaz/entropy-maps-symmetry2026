"""Strict TOML configuration loader and validator for profiling and CPU-fair comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Mapping

import psutil


def _require(table: Mapping[str, object], key: str, where: str) -> object:
    """Extract required key from TOML mapping or raise KeyError with section context."""
    if key not in table:
        raise KeyError(f"Missing required key '{key}' in {where}")
    return table[key]


def _require_table(table: Mapping[str, object], key: str, where: str) -> dict[str, object]:
    """Extract required sub-table from TOML mapping with validation."""
    obj = _require(table, key, where)
    if not isinstance(obj, dict):
        raise ValueError(f"Section '{key}' in {where} must be a TOML table")
    return dict(obj)


@dataclass(frozen=True, slots=True)
class ProfilingProtocolConfig:
    """Protocol constraints enforcing fair CPU comparison and repeatable timing."""

    device: str
    include_cuda_reference: bool
    primary_threads: int
    threads_list: tuple[int, ...]
    include_physical_cores: bool
    primary_batch_size: int
    batch_sizes: tuple[int, ...]
    warmup_iters: int
    timed_iters: int
    batch_timed_iters: int
    repeats: int
    n_real_samples: int
    disable_gc: bool

    def effective_threads_list(self) -> tuple[int, ...]:
        """Return deduplicated, sorted thread counts including physical core count when requested."""
        values = {int(t) for t in self.threads_list if int(t) >= 1}
        values.add(int(self.primary_threads))
        if self.include_physical_cores:
            phys = psutil.cpu_count(logical=False)
            if phys is not None and phys >= 1:
                values.add(int(phys))
        return tuple(sorted(values))


@dataclass(frozen=True, slots=True)
class DescriptorProfilingConfig:
    """Parameters governing x94 descriptor stage breakdown and scalability benchmarks."""

    image_size: int
    grid_levels: tuple[int, ...]
    bins: int
    asymmetry_scales: tuple[int, ...]
    asymmetry_kinds: tuple[str, ...]
    scalability_image_sizes: tuple[int, ...]
    scalability_bins: tuple[int, ...]
    scalability_grid_levels: tuple[tuple[int, ...], ...]
    cprofile_top_n: int


@dataclass(frozen=True, slots=True)
class CnnProfilingConfig:
    """Parameters governing CNN layer hook profiling, torch.profiler, and parity checks."""

    torch_profiler_warmup: int
    torch_profiler_active: int
    hook_overhead_warn_pct: float
    metrics_tolerance: float
    include_checkpoint_load_and_cold_start: bool


@dataclass(frozen=True, slots=True)
class StatisticsProfilingConfig:
    """Statistical uncertainty and significance test configuration."""

    bootstrap_resamples: int
    confidence_level: float
    mann_whitney_alpha: float
    ml_baseline_models: tuple[str, ...]
    cnn_reference_models: tuple[str, ...]
    ablation_subsets: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OutputProfilingConfig:
    """Output destinations, plotting parameters, and inverted-variant behavior."""

    comparison_cpu_dir: Path
    profiling_dir: Path
    dpi: int
    float_precision: int
    run_full_profiling_on_inverted: bool


@dataclass(frozen=True, slots=True)
class ProfilingConfig:
    """Validated root configuration for CPU-fair comparison and deep profiling."""

    project_root: Path
    base_config_path: Path
    classification_config_path: Path
    cnn_config_path: Path
    entropy_config_path: Path
    asymmetry_config_path: Path
    dataset_key: str
    dataset_name: str
    subdatasets: tuple[str, ...]
    inverted: bool
    overwrite: bool
    seed: int
    objects: tuple[str, ...]
    default_object: str | None
    classifiers: tuple[str, ...]
    models: tuple[str, ...]
    protocol: ProfilingProtocolConfig
    descriptor: DescriptorProfilingConfig
    cnn: CnnProfilingConfig
    statistics: StatisticsProfilingConfig
    output: OutputProfilingConfig

    @property
    def base_dataset_tag(self) -> str:
        """Return canonical dataset base tag including '-inverted' when enabled."""
        base = self.dataset_name.lower()
        return f"{base}-inverted" if self.inverted else base


def effective_variant_tag(dataset_name: str, object_name: str | None, inverted: bool) -> str:
    """Build canonical dataset-variant tag for folders and filenames.

    Examples:
        ('KolektorSDD2', None, False) -> 'kolektorsdd2'
        ('KolektorSDD2', None, True) -> 'kolektorsdd2-inverted'
        ('mvtec-ad', 'bottle', False) -> 'mvtec-ad_bottle'
        ('mvtec-ad', 'bottle', True) -> 'mvtec-ad-inverted_bottle'
    """
    base = f"{dataset_name.lower()}-inverted" if inverted else dataset_name.lower()
    return f"{base}_{object_name}" if object_name else base


def resolve_target_objects(
    cfg: ProfilingConfig,
    object_name: str | None = None,
    objects: tuple[str, ...] | list[str] | None = None,
    all_objects: bool = False,
) -> tuple[str | None, ...]:
    """Resolve target object names respecting CLI flags and dataset schema."""
    has_subdatasets = bool(cfg.subdatasets)

    if all_objects:
        if not has_subdatasets:
            raise ValueError(
                f"--all-objects requires subdatasets in base.toml for '{cfg.dataset_key}'"
            )
        return cfg.subdatasets

    if objects:
        if not has_subdatasets:
            raise ValueError(
                f"--objects is only valid for multi-object datasets; '{cfg.dataset_key}' has none"
            )
        cleaned = tuple(str(obj).strip() for obj in objects if str(obj).strip())
        for obj in cleaned:
            if obj not in cfg.subdatasets:
                raise ValueError(
                    f"Unknown object '{obj}' for '{cfg.dataset_key}'. Valid: {cfg.subdatasets}"
                )
        return cleaned

    if object_name is not None:
        if has_subdatasets and object_name not in cfg.subdatasets:
            raise ValueError(
                f"Unknown object '{object_name}' for '{cfg.dataset_key}'. Valid: {cfg.subdatasets}"
            )
        return (object_name,)

    if has_subdatasets:
        if cfg.objects:
            for obj in cfg.objects:
                if obj not in cfg.subdatasets:
                    raise ValueError(
                        f"Configured object '{obj}' not in subdatasets {cfg.subdatasets}"
                    )
            return cfg.objects
        if cfg.default_object is not None:
            return (cfg.default_object,)
        raise ValueError(
            f"Dataset '{cfg.dataset_key}' requires an object via --object, --objects, or config 'objects'"
        )

    return (None,)


def load_profiling_config(config_path: Path, project_root: Path) -> ProfilingConfig:
    """Load, anchor paths, and fail-fast validate profiling TOML configuration."""
    resolved_root = project_root.resolve()
    resolved_cfg = (resolved_root / config_path).resolve()
    if not resolved_cfg.is_file():
        raise FileNotFoundError(f"Profiling config file not found: {resolved_cfg}")

    with resolved_cfg.open("rb") as f:
        raw_cfg = tomllib.load(f)

    prof = _require_table(raw_cfg, "profiling", str(resolved_cfg))

    base_config_path = (resolved_root / str(_require(prof, "base_config", "[profiling]"))).resolve()
    cls_config_path = (
        resolved_root / str(_require(prof, "classification_config", "[profiling]"))
    ).resolve()
    cnn_config_path = (resolved_root / str(_require(prof, "cnn_config", "[profiling]"))).resolve()
    ent_config_path = (
        resolved_root / str(_require(prof, "entropy_config", "[profiling]"))
    ).resolve()
    asym_config_path = (
        resolved_root / str(_require(prof, "asymmetry_config", "[profiling]"))
    ).resolve()

    for linked in (base_config_path, cls_config_path, cnn_config_path, ent_config_path, asym_config_path):
        if not linked.is_file():
            raise FileNotFoundError(f"Linked configuration file not found: {linked}")

    dataset_key = str(_require(prof, "dataset_key", "[profiling]"))
    inverted = _require(prof, "inverted", "[profiling]")
    overwrite = _require(prof, "overwrite", "[profiling]")
    seed = _require(prof, "seed", "[profiling]")
    if not isinstance(inverted, bool):
        raise ValueError("'inverted' in [profiling] must be boolean")
    if not isinstance(overwrite, bool):
        raise ValueError("'overwrite' in [profiling] must be boolean")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("'seed' in [profiling] must be integer")

    raw_objects = _require(prof, "objects", "[profiling]")
    if not isinstance(raw_objects, list):
        raise ValueError("'objects' in [profiling] must be a list")
    objects = tuple(str(o) for o in raw_objects if str(o).strip())

    raw_def_obj = str(_require(prof, "default_object", "[profiling]")).strip()
    default_object = raw_def_obj if raw_def_obj else None

    raw_classifiers = _require(prof, "classifiers", "[profiling]")
    raw_models = _require(prof, "models", "[profiling]")
    if not isinstance(raw_classifiers, list) or not raw_classifiers:
        raise ValueError("'classifiers' in [profiling] must be a non-empty list")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("'models' in [profiling] must be a non-empty list")
    classifiers = tuple(str(c).lower() for c in raw_classifiers)
    models = tuple(str(m).lower() for m in raw_models)

    # Resolve dataset metadata from base.toml
    with base_config_path.open("rb") as f_base:
        base_data = tomllib.load(f_base)
    datasets_tbl = _require_table(base_data, "datasets", "base config")
    ds_spec = _require_table(datasets_tbl, dataset_key, f"base config [datasets.{dataset_key}]")
    dataset_name = str(_require(ds_spec, "name", f"[datasets.{dataset_key}]"))
    raw_sub = ds_spec.get("subdatasets", [])
    subdatasets = tuple(str(s) for s in raw_sub) if isinstance(raw_sub, list) else ()

    # [profiling.protocol]
    proto_tbl = _require_table(prof, "protocol", "[profiling]")
    device = str(_require(proto_tbl, "device", "[profiling.protocol]")).lower()
    if device != "cpu":
        raise ValueError(
            f"Primary device in [profiling.protocol] must be 'cpu' for fair comparison, got '{device}'"
        )
    include_cuda_reference = bool(_require(proto_tbl, "include_cuda_reference", "[profiling.protocol]"))
    primary_threads = int(_require(proto_tbl, "primary_threads", "[profiling.protocol]"))
    raw_threads = _require(proto_tbl, "threads_list", "[profiling.protocol]")
    if not isinstance(raw_threads, list) or not raw_threads:
        raise ValueError("'threads_list' in [profiling.protocol] must be a non-empty list")
    threads_list = tuple(int(t) for t in raw_threads)
    if primary_threads < 1 or any(t < 1 for t in threads_list):
        raise ValueError("Thread counts in [profiling.protocol] must be >= 1")

    include_physical_cores = bool(_require(proto_tbl, "include_physical_cores", "[profiling.protocol]"))
    primary_batch_size = int(_require(proto_tbl, "primary_batch_size", "[profiling.protocol]"))
    raw_batches = _require(proto_tbl, "batch_sizes", "[profiling.protocol]")
    if not isinstance(raw_batches, list) or not raw_batches:
        raise ValueError("'batch_sizes' in [profiling.protocol] must be a non-empty list")
    batch_sizes = tuple(int(b) for b in raw_batches)
    if primary_batch_size < 1 or any(b < 1 for b in batch_sizes):
        raise ValueError("Batch sizes in [profiling.protocol] must be >= 1")

    warmup_iters = int(_require(proto_tbl, "warmup_iters", "[profiling.protocol]"))
    timed_iters = int(_require(proto_tbl, "timed_iters", "[profiling.protocol]"))
    batch_timed_iters = int(_require(proto_tbl, "batch_timed_iters", "[profiling.protocol]"))
    repeats = int(_require(proto_tbl, "repeats", "[profiling.protocol]"))
    n_real_samples = int(_require(proto_tbl, "n_real_samples", "[profiling.protocol]"))
    disable_gc = bool(_require(proto_tbl, "disable_gc", "[profiling.protocol]"))
    if min(warmup_iters, timed_iters, batch_timed_iters, repeats, n_real_samples) < 1:
        raise ValueError("Iteration, repeat, and sample counts in [profiling.protocol] must be >= 1")

    protocol_cfg = ProfilingProtocolConfig(
        device=device,
        include_cuda_reference=include_cuda_reference,
        primary_threads=primary_threads,
        threads_list=threads_list,
        include_physical_cores=include_physical_cores,
        primary_batch_size=primary_batch_size,
        batch_sizes=batch_sizes,
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        batch_timed_iters=batch_timed_iters,
        repeats=repeats,
        n_real_samples=n_real_samples,
        disable_gc=disable_gc,
    )

    # [profiling.descriptor]
    desc_tbl = _require_table(prof, "descriptor", "[profiling]")
    desc_img_size = int(_require(desc_tbl, "image_size", "[profiling.descriptor]"))
    raw_levels = _require(desc_tbl, "grid_levels", "[profiling.descriptor]")
    desc_bins = int(_require(desc_tbl, "bins", "[profiling.descriptor]"))
    raw_asym_scales = _require(desc_tbl, "asymmetry_scales", "[profiling.descriptor]")
    raw_asym_kinds = _require(desc_tbl, "asymmetry_kinds", "[profiling.descriptor]")
    raw_scale_sizes = _require(desc_tbl, "scalability_image_sizes", "[profiling.descriptor]")
    raw_scale_bins = _require(desc_tbl, "scalability_bins", "[profiling.descriptor]")
    raw_scale_levels = _require(desc_tbl, "scalability_grid_levels", "[profiling.descriptor]")
    cprofile_top_n = int(_require(desc_tbl, "cprofile_top_n", "[profiling.descriptor]"))

    if not all(isinstance(x, list) and x for x in (raw_levels, raw_asym_scales, raw_asym_kinds, raw_scale_sizes, raw_scale_bins, raw_scale_levels)):
        raise ValueError("Descriptor list fields in [profiling.descriptor] must be non-empty lists")

    descriptor_cfg = DescriptorProfilingConfig(
        image_size=desc_img_size,
        grid_levels=tuple(int(g) for g in raw_levels),
        bins=desc_bins,
        asymmetry_scales=tuple(int(g) for g in raw_asym_scales),
        asymmetry_kinds=tuple(str(k) for k in raw_asym_kinds),
        scalability_image_sizes=tuple(int(s) for s in raw_scale_sizes),
        scalability_bins=tuple(int(b) for b in raw_scale_bins),
        scalability_grid_levels=tuple(tuple(int(g) for g in lv) for lv in raw_scale_levels),
        cprofile_top_n=cprofile_top_n,
    )

    # [profiling.cnn]
    cnn_tbl = _require_table(prof, "cnn", "[profiling]")
    cnn_prof_cfg = CnnProfilingConfig(
        torch_profiler_warmup=int(_require(cnn_tbl, "torch_profiler_warmup", "[profiling.cnn]")),
        torch_profiler_active=int(_require(cnn_tbl, "torch_profiler_active", "[profiling.cnn]")),
        hook_overhead_warn_pct=float(_require(cnn_tbl, "hook_overhead_warn_pct", "[profiling.cnn]")),
        metrics_tolerance=float(_require(cnn_tbl, "metrics_tolerance", "[profiling.cnn]")),
        include_checkpoint_load_and_cold_start=bool(
            _require(cnn_tbl, "include_checkpoint_load_and_cold_start", "[profiling.cnn]")
        ),
    )

    # [profiling.statistics]
    stat_tbl = _require_table(prof, "statistics", "[profiling]")
    bootstrap_resamples = int(_require(stat_tbl, "bootstrap_resamples", "[profiling.statistics]"))
    confidence_level = float(_require(stat_tbl, "confidence_level", "[profiling.statistics]"))
    mann_whitney_alpha = float(_require(stat_tbl, "mann_whitney_alpha", "[profiling.statistics]"))
    raw_ml_base = _require(stat_tbl, "ml_baseline_models", "[profiling.statistics]")
    raw_cnn_ref = _require(stat_tbl, "cnn_reference_models", "[profiling.statistics]")
    raw_ablation = _require(stat_tbl, "ablation_subsets", "[profiling.statistics]")
    if bootstrap_resamples < 10 or not (0.5 < confidence_level < 1.0) or not (0.0 < mann_whitney_alpha < 0.5):
        raise ValueError("Invalid statistical bounds in [profiling.statistics]")
    if not (isinstance(raw_ml_base, list) and isinstance(raw_cnn_ref, list) and isinstance(raw_ablation, list)):
        raise ValueError("Statistical model lists in [profiling.statistics] must be lists")

    stat_cfg = StatisticsProfilingConfig(
        bootstrap_resamples=bootstrap_resamples,
        confidence_level=confidence_level,
        mann_whitney_alpha=mann_whitney_alpha,
        ml_baseline_models=tuple(str(m).lower() for m in raw_ml_base),
        cnn_reference_models=tuple(str(m).lower() for m in raw_cnn_ref),
        ablation_subsets=tuple(str(s) for s in raw_ablation),
    )

    # [profiling.output]
    out_tbl = _require_table(prof, "output", "[profiling]")
    out_cfg = OutputProfilingConfig(
        comparison_cpu_dir=(
            resolved_root / str(_require(out_tbl, "comparison_cpu_dir", "[profiling.output]"))
        ).resolve(),
        profiling_dir=(
            resolved_root / str(_require(out_tbl, "profiling_dir", "[profiling.output]"))
        ).resolve(),
        dpi=int(_require(out_tbl, "dpi", "[profiling.output]")),
        float_precision=int(_require(out_tbl, "float_precision", "[profiling.output]")),
        run_full_profiling_on_inverted=bool(
            _require(out_tbl, "run_full_profiling_on_inverted", "[profiling.output]")
        ),
    )

    return ProfilingConfig(
        project_root=resolved_root,
        base_config_path=base_config_path,
        classification_config_path=cls_config_path,
        cnn_config_path=cnn_config_path,
        entropy_config_path=ent_config_path,
        asymmetry_config_path=asym_config_path,
        dataset_key=dataset_key,
        dataset_name=dataset_name,
        subdatasets=subdatasets,
        inverted=inverted,
        overwrite=overwrite,
        seed=seed,
        objects=objects,
        default_object=default_object,
        classifiers=classifiers,
        models=models,
        protocol=protocol_cfg,
        descriptor=descriptor_cfg,
        cnn=cnn_prof_cfg,
        statistics=stat_cfg,
        output=out_cfg,
    )
