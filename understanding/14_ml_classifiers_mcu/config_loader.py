"""Configuration loading and validation for Lab 14 MCU classifier evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import tomllib
from typing import Any

LAB_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Top-level experiment reproducibility and output settings."""

    seed: int
    output_dir: Path
    float_precision: int
    dpi: int
    export_sample_header_dataset: str
    export_sample_header_models: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DescriptorReferenceConfig:
    """Settings matching x94 descriptor extraction for T_desc timing."""

    entropy_backend: str
    image_size: int
    bins: int
    grid_levels: tuple[int, ...]
    asymmetry_scales: tuple[int, ...]
    asymmetry_kinds: tuple[str, ...]
    n_real_images: int


@dataclass(frozen=True, slots=True)
class ProfilingConfig:
    """CPU-fair single-thread latency and statistical testing protocol."""

    threads: int
    warmup_iters: int
    timed_iters: int
    batch_timed_iters: int
    repeats: int
    batch_sizes: tuple[int, ...]
    disable_gc: bool
    bootstrap_resamples: int
    confidence_level: float
    mann_whitney_alpha: float
    reference_baseline_model: str
    heavy_baseline_model: str


@dataclass(frozen=True, slots=True)
class ThresholdsConfig:
    """Numerical parity and sparsity tolerances."""

    parity_abs_tol_f64: float
    parity_abs_tol_f32: float
    c_parity_abs_tol_f32: float
    c_parity_rel_tol_f32: float
    zero_weight_eps: float


@dataclass(frozen=True, slots=True)
class NativeBuildConfig:
    """Configuration for optional C99 reference DLL compilation and FFI bridge."""

    enabled: bool
    source_file: Path
    header_file: Path
    build_dir: Path
    dll_name: str
    so_name: str
    build_info_name: str
    vswhere_hint: str
    c_bench_inner_iters: int

    @property
    def binary_path(self) -> Path:
        """Return platform-specific shared library path inside build_dir."""
        filename = self.dll_name if sys.platform == "win32" else self.so_name
        return self.build_dir / filename

    @property
    def build_info_path(self) -> Path:
        """Return build provenance metadata JSON path."""
        return self.build_dir / self.build_info_name


@dataclass(frozen=True, slots=True)
class BenchmarkTargetConfig:
    """Dataset benchmark entry reusing canonical src/ classification configs."""

    tag: str
    classification_config: Path
    object_name: str | None
    image_dir: Path
    image_dir_inverted: Path
    benchmark_latency: bool


@dataclass(frozen=True, slots=True)
class MvtecSweepConfig:
    """Cross-object predictive generalization sweep settings for MVTec-AD."""

    enabled: bool
    classification_config: Path
    objects: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelsConfig:
    """Active model list and per-model hyperparameter mappings from TOML."""

    active: tuple[str, ...]
    params_by_model: dict[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class McuLabConfig:
    """Root validated configuration container for Lab 14."""

    config_path: Path
    experiment: ExperimentConfig
    descriptor_reference: DescriptorReferenceConfig
    profiling: ProfilingConfig
    thresholds: ThresholdsConfig
    native: NativeBuildConfig
    targets: tuple[BenchmarkTargetConfig, ...]
    mvtec_sweep: MvtecSweepConfig
    models: ModelsConfig


def resolve_project_path(raw_path: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve a path relative to project_root when not already absolute."""
    p = Path(raw_path)
    return p.resolve() if p.is_absolute() else (project_root / p).resolve()


def _require_key(table: dict[str, Any], key: str, section: str) -> Any:
    """Fail fast with contextual KeyError when a required key is absent."""
    if key not in table:
        raise KeyError(f"Missing required key '{key}' in section '{section}'")
    return table[key]


def _require_table(raw: dict[str, Any], key: str) -> dict[str, Any]:
    """Require a TOML sub-table as dictionary."""
    val = _require_key(raw, key, "root")
    if not isinstance(val, dict):
        raise ValueError(f"Section [{key}] must be a TOML table")
    return val


def load_lab_config(
    config_path: str | Path,
    project_root: Path = PROJECT_ROOT,
) -> McuLabConfig:
    """Load, validate, and resolve Lab 14 TOML configuration cleanly."""
    resolved_cfg = resolve_project_path(config_path, project_root)
    if not resolved_cfg.is_file():
        raise FileNotFoundError(f"Lab 14 config file not found: {resolved_cfg}")

    with resolved_cfg.open("rb") as f:
        raw = tomllib.load(f)

    exp_tbl = _require_table(raw, "experiment")
    seed = int(_require_key(exp_tbl, "seed", "experiment"))
    output_dir = resolve_project_path(
        str(_require_key(exp_tbl, "output_dir", "experiment")),
        project_root,
    )
    float_precision = int(_require_key(exp_tbl, "float_precision", "experiment"))
    dpi = int(_require_key(exp_tbl, "dpi", "experiment"))
    export_sample_ds = str(_require_key(exp_tbl, "export_sample_header_dataset", "experiment"))
    export_sample_models = tuple(
        str(x) for x in _require_key(exp_tbl, "export_sample_header_models", "experiment")
    )
    if float_precision < 1 or dpi < 72:
        raise ValueError("Invalid float_precision or dpi in [experiment]")

    experiment = ExperimentConfig(
        seed=seed,
        output_dir=output_dir,
        float_precision=float_precision,
        dpi=dpi,
        export_sample_header_dataset=export_sample_ds,
        export_sample_header_models=export_sample_models,
    )

    desc_tbl = _require_table(raw, "descriptor_reference")
    descriptor_reference = DescriptorReferenceConfig(
        entropy_backend=str(_require_key(desc_tbl, "entropy_backend", "descriptor_reference")),
        image_size=int(_require_key(desc_tbl, "image_size", "descriptor_reference")),
        bins=int(_require_key(desc_tbl, "bins", "descriptor_reference")),
        grid_levels=tuple(
            int(v) for v in _require_key(desc_tbl, "grid_levels", "descriptor_reference")
        ),
        asymmetry_scales=tuple(
            int(v) for v in _require_key(desc_tbl, "asymmetry_scales", "descriptor_reference")
        ),
        asymmetry_kinds=tuple(
            str(v) for v in _require_key(desc_tbl, "asymmetry_kinds", "descriptor_reference")
        ),
        n_real_images=int(_require_key(desc_tbl, "n_real_images", "descriptor_reference")),
    )
    if descriptor_reference.image_size <= 0 or descriptor_reference.bins < 2:
        raise ValueError("Invalid image_size or bins in [descriptor_reference]")

    prof_tbl = _require_table(raw, "profiling")
    profiling = ProfilingConfig(
        threads=int(_require_key(prof_tbl, "threads", "profiling")),
        warmup_iters=int(_require_key(prof_tbl, "warmup_iters", "profiling")),
        timed_iters=int(_require_key(prof_tbl, "timed_iters", "profiling")),
        batch_timed_iters=int(_require_key(prof_tbl, "batch_timed_iters", "profiling")),
        repeats=int(_require_key(prof_tbl, "repeats", "profiling")),
        batch_sizes=tuple(int(b) for b in _require_key(prof_tbl, "batch_sizes", "profiling")),
        disable_gc=bool(_require_key(prof_tbl, "disable_gc", "profiling")),
        bootstrap_resamples=int(_require_key(prof_tbl, "bootstrap_resamples", "profiling")),
        confidence_level=float(_require_key(prof_tbl, "confidence_level", "profiling")),
        mann_whitney_alpha=float(_require_key(prof_tbl, "mann_whitney_alpha", "profiling")),
        reference_baseline_model=str(
            _require_key(prof_tbl, "reference_baseline_model", "profiling")
        ),
        heavy_baseline_model=str(_require_key(prof_tbl, "heavy_baseline_model", "profiling")),
    )
    if profiling.threads < 1 or profiling.timed_iters < 1 or profiling.repeats < 1:
        raise ValueError("threads, timed_iters, and repeats must be >= 1")
    if any(b < 1 for b in profiling.batch_sizes):
        raise ValueError("All profiling batch_sizes must be >= 1")

    th_tbl = _require_table(raw, "thresholds")
    thresholds = ThresholdsConfig(
        parity_abs_tol_f64=float(_require_key(th_tbl, "parity_abs_tol_f64", "thresholds")),
        parity_abs_tol_f32=float(_require_key(th_tbl, "parity_abs_tol_f32", "thresholds")),
        c_parity_abs_tol_f32=float(_require_key(th_tbl, "c_parity_abs_tol_f32", "thresholds")),
        c_parity_rel_tol_f32=float(_require_key(th_tbl, "c_parity_rel_tol_f32", "thresholds")),
        zero_weight_eps=float(_require_key(th_tbl, "zero_weight_eps", "thresholds")),
    )

    nat_tbl = _require_table(raw, "native")
    native = NativeBuildConfig(
        enabled=bool(_require_key(nat_tbl, "enabled", "native")),
        source_file=resolve_project_path(
            str(_require_key(nat_tbl, "source_file", "native")), project_root
        ),
        header_file=resolve_project_path(
            str(_require_key(nat_tbl, "header_file", "native")), project_root
        ),
        build_dir=resolve_project_path(
            str(_require_key(nat_tbl, "build_dir", "native")), project_root
        ),
        dll_name=str(_require_key(nat_tbl, "dll_name", "native")),
        so_name=str(_require_key(nat_tbl, "so_name", "native")),
        build_info_name=str(_require_key(nat_tbl, "build_info_name", "native")),
        vswhere_hint=str(_require_key(nat_tbl, "vswhere_hint", "native")),
        c_bench_inner_iters=int(_require_key(nat_tbl, "c_bench_inner_iters", "native")),
    )
    if native.c_bench_inner_iters < 1:
        raise ValueError("c_bench_inner_iters must be >= 1")

    raw_targets = _require_key(raw, "targets", "root")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("[[targets]] must be a non-empty list of tables")
    targets: list[BenchmarkTargetConfig] = []
    for idx, entry in enumerate(raw_targets):
        if not isinstance(entry, dict):
            raise ValueError(f"targets[{idx}] must be a table")
        obj_raw = str(_require_key(entry, "object_name", f"targets[{idx}]")).strip()
        cls_cfg = resolve_project_path(
            str(_require_key(entry, "classification_config", f"targets[{idx}]")),
            project_root,
        )
        if not cls_cfg.is_file():
            raise FileNotFoundError(f"Target classification config missing: {cls_cfg}")
        targets.append(
            BenchmarkTargetConfig(
                tag=str(_require_key(entry, "tag", f"targets[{idx}]")),
                classification_config=cls_cfg,
                object_name=obj_raw if obj_raw else None,
                image_dir=resolve_project_path(
                    str(_require_key(entry, "image_dir", f"targets[{idx}]")), project_root
                ),
                image_dir_inverted=resolve_project_path(
                    str(_require_key(entry, "image_dir_inverted", f"targets[{idx}]")),
                    project_root,
                ),
                benchmark_latency=bool(_require_key(entry, "benchmark_latency", f"targets[{idx}]")),
            )
        )

    sweep_tbl = _require_table(raw, "mvtec_sweep")
    sweep_cfg_path = resolve_project_path(
        str(_require_key(sweep_tbl, "classification_config", "mvtec_sweep")), project_root
    )
    if not sweep_cfg_path.is_file():
        raise FileNotFoundError(f"Sweep classification config missing: {sweep_cfg_path}")
    mvtec_sweep = MvtecSweepConfig(
        enabled=bool(_require_key(sweep_tbl, "enabled", "mvtec_sweep")),
        classification_config=sweep_cfg_path,
        objects=tuple(str(o) for o in _require_key(sweep_tbl, "objects", "mvtec_sweep")),
    )

    models_tbl = _require_table(raw, "models")
    active_models = tuple(str(m) for m in _require_key(models_tbl, "active", "models"))
    if not active_models:
        raise ValueError("[models].active cannot be empty")
    params_by_model: dict[str, dict[str, Any]] = {}
    for k, v in models_tbl.items():
        if k == "active":
            continue
        if isinstance(v, dict):
            params_by_model[k] = dict(v)

    return McuLabConfig(
        config_path=resolved_cfg,
        experiment=experiment,
        descriptor_reference=descriptor_reference,
        profiling=profiling,
        thresholds=thresholds,
        native=native,
        targets=tuple(targets),
        mvtec_sweep=mvtec_sweep,
        models=ModelsConfig(active=active_models, params_by_model=params_by_model),
    )
