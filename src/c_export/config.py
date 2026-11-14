"""Configuration loading and validation for C99 pipeline export and verification.

References existing dataset, entropy, asymmetry, classification, and CNN TOMLs as the
single source of truth rather than duplicating dataset parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import tomllib
from typing import Mapping

from src.features.asymmetry_csv import AsymmetryCsvConfig, load_asymmetry_config
from src.features.entropy_csv import EntropyCsvConfig, load_entropy_config
from src.features.entropy_maps import validate_bins, validate_nested_grid_levels
from src.models.cnn.architectures.registry import MODEL_REGISTRY
from src.models.cnn.config import CnnConfig, load_cnn_config
from src.models.descriptor_classifier import (
    ClassificationConfig,
    load_classification_config,
)

ALLOWED_NUMERIC_MODES: tuple[str, ...] = ("f32", "q15")
ALLOWED_SHANNON_MODES: tuple[str, ...] = ("nlogn_full", "term_per_level", "log2_mantissa")
ALLOWED_WEIGHT_FORMATS: tuple[str, ...] = ("q15", "int8_per_channel")
ALLOWED_CLASSIFIER_MODELS: tuple[str, ...] = ("lr_dot", "mlp_16_dot")


def _require(table: Mapping[str, object], key: str, where: str) -> object:
    """Return a required TOML key or raise KeyError with full section context."""
    if key not in table:
        raise KeyError(f"Missing required key '{key}' in {where}")
    return table[key]


def _require_int(table: Mapping[str, object], key: str, where: str) -> int:
    """Return an exact non-boolean integer from a TOML table."""
    val = _require(table, key, where)
    if isinstance(val, bool) or not isinstance(val, int):
        raise ValueError(f"Key '{key}' in {where} must be an integer, got {val!r}")
    return int(val)


@dataclass(frozen=True, slots=True)
class DescriptorSpec:
    """Derived descriptor geometry shared by C LUTs, C descriptor runtime, and Python mirror."""

    image_size: int
    bins: int
    grid_levels: tuple[int, ...]
    asym_scales: tuple[int, ...]
    asym_kinds: tuple[str, ...]
    schema: str
    n_x85: int
    n_x9: int
    n_x94: int


@dataclass(frozen=True, slots=True)
class CImpPaths:
    """Anchored filesystem directories used during C generation, compilation, and reporting."""

    runtime_dir: Path
    app_dir: Path
    c_tests_dir: Path
    generated_root: Path
    build_root: Path
    results_root: Path


@dataclass(frozen=True, slots=True)
class CImpNumeric:
    """Numeric precision, accumulator bit-width, and rounding policy."""

    modes: tuple[str, ...]
    feature_frac_bits: int
    accumulator_bits: int
    weight_storage_bits: int
    rounding: str


@dataclass(frozen=True, slots=True)
class CImpDescriptor:
    """Descriptor LUT mode configuration."""

    shannon_lut_mode: str
    log2_mantissa_bits: int


@dataclass(frozen=True, slots=True)
class CImpClassifier:
    """Classical descriptor classifier export and cross-check tolerances."""

    models: tuple[str, ...]
    metrics_crosscheck_dir: Path
    metrics_crosscheck_abs_tol: float


@dataclass(frozen=True, slots=True)
class CImpCnn:
    """CNN weight import, quantization calibration, and embedded memory budget parameters."""

    models: tuple[str, ...]
    allowed_weight_formats: tuple[str, ...]
    flash_budget_bytes: int
    sram_budget_bytes: int
    psram_budget_bytes: int
    checkpoint_variant: str
    calibration_split: str
    calibration_max_images: int
    activation_range_percentile: float
    weight_granularity: str
    input_frac_bits: int
    fuse_conv_relu_maxpool: bool


@dataclass(frozen=True, slots=True)
class CImpTestVectors:
    """Test vector selection and embedded C header count."""

    split: str
    max_images: int
    embed_headers_count: int


@dataclass(frozen=True, slots=True)
class CImpBuild:
    """MSVC desktop build configuration."""

    compiler: str
    c_standard_flag: str
    compiler_flags: tuple[str, ...]
    vswhere_hint: str
    executable_name: str


@dataclass(frozen=True, slots=True)
class CImpVerification:
    """Acceptance gates and desktop benchmark iteration counts."""

    f32_descriptor_abs_tol: float
    f32_logit_abs_tol: float
    q15_must_be_bit_exact_vs_python_model: bool
    q15_descriptor_report_only: bool
    timing_warmup_iters: int
    timing_timed_iters: int


@dataclass(frozen=True, slots=True)
class CImpVisualization:
    """Visualization figure output directory, resolution, and error audit limit."""

    output_subdir: str
    dpi: int
    top_k_worst_errors: int


@dataclass(frozen=True, slots=True)
class CImpConfig:
    """Root validated C99 pipeline export and verification configuration."""

    config_path: Path
    project_root: Path
    target_tag: str
    object_name: str | None
    seed: int
    inverted: bool
    entropy_config_path: Path
    asymmetry_config_path: Path
    classification_config_path: Path
    cnn_config_path: Path
    entropy_cfg: EntropyCsvConfig
    asymmetry_cfg: AsymmetryCsvConfig
    classification_cfg: ClassificationConfig
    cnn_cfg: CnnConfig
    descriptor_spec: DescriptorSpec
    paths: CImpPaths
    numeric: CImpNumeric
    descriptor: CImpDescriptor
    classifier: CImpClassifier
    cnn: CImpCnn
    test_vectors: CImpTestVectors
    build: CImpBuild
    verification: CImpVerification
    visualization: CImpVisualization

    @property
    def effective_target_tag(self) -> str:
        """Return unique target subdirectory tag including object and inversion suffix."""
        tag = self.target_tag
        if self.object_name:
            tag = f"{tag}_{self.object_name}"
        if self.inverted:
            tag = f"{tag}-inverted"
        return tag

    @property
    def generated_dir(self) -> Path:
        """Return target-specific generated C header/source directory."""
        return self.paths.generated_root / self.effective_target_tag

    @property
    def build_dir(self) -> Path:
        """Return target-specific compiled desktop binary directory."""
        return self.paths.build_root / self.effective_target_tag

    @property
    def results_dir(self) -> Path:
        """Return target-specific parity and memory reports directory."""
        return self.paths.results_root / self.effective_target_tag

    @property
    def visualization_dir(self) -> Path:
        """Return target-specific directory for generated figures and summary reports."""
        return self.results_dir / self.visualization.output_subdir


def load_c_imp_config(
    config_path: Path,
    project_root: Path,
    object_name: str | None = None,
    inverted: bool | None = None,
) -> CImpConfig:
    """Load, resolve, and fail-fast validate C99 export configuration from TOML.

    Args:
        config_path: Path to configs/c_imp/*.toml (relative to project_root or absolute).
        project_root: Repository 'code/' root directory for path anchoring.
        object_name: Optional MVTec-AD subdataset override (e.g. 'bottle').
        inverted: Optional override for image intensity inversion flag.

    Returns:
        Validated immutable CImpConfig instance.
    """
    root = Path(project_root).resolve()
    resolved_cfg_path = config_path if config_path.is_absolute() else (root / config_path).resolve()
    if not resolved_cfg_path.is_file():
        raise FileNotFoundError(f"C export config file not found: {resolved_cfg_path}")

    with resolved_cfg_path.open("rb") as f:
        raw = tomllib.load(f)

    top = _require(raw, "c_imp", str(resolved_cfg_path))
    if not isinstance(top, dict):
        raise ValueError("[c_imp] section must be a TOML table")

    target_tag = str(_require(top, "target_tag", "[c_imp]")).strip()
    if not target_tag:
        raise ValueError("'target_tag' in [c_imp] must not be empty")

    seed = _require_int(top, "seed", "[c_imp]")
    inv_cfg = _require(top, "inverted", "[c_imp]")
    if not isinstance(inv_cfg, bool):
        raise ValueError("'inverted' in [c_imp] must be a boolean")
    effective_inverted = inv_cfg if inverted is None else bool(inverted)

    raw_obj = top.get("object_name")
    active_object = object_name if object_name is not None else (str(raw_obj) if raw_obj else None)

    entropy_rel = Path(str(_require(top, "entropy_config", "[c_imp]")))
    asym_rel = Path(str(_require(top, "asymmetry_config", "[c_imp]")))
    cls_rel = Path(str(_require(top, "classification_config", "[c_imp]")))
    cnn_rel = Path(str(_require(top, "cnn_config", "[c_imp]")))

    # Load referenced single-source configs and propagate active inverted flag
    entropy_cfg = replace(load_entropy_config(entropy_rel, root), inverted=effective_inverted)
    asymmetry_cfg = replace(load_asymmetry_config(asym_rel, root), inverted=effective_inverted)
    classification_cfg = replace(
        load_classification_config(cls_rel, root), inverted=effective_inverted
    )
    cnn_cfg = replace(load_cnn_config(cnn_rel, root), inverted=effective_inverted)

    if cnn_cfg.subdatasets:
        if active_object is None:
            active_object = cnn_cfg.default_object
        if active_object is None or active_object not in cnn_cfg.subdatasets:
            raise ValueError(
                f"Object '{active_object}' invalid for dataset '{cnn_cfg.dataset_name}'. "
                f"Valid subdatasets: {cnn_cfg.subdatasets}"
            )
    elif active_object is not None and active_object.strip() == "":
        active_object = None

    # Validate descriptor geometry & C nesting requirement
    validate_bins(entropy_cfg.bins)
    nested_levels = validate_nested_grid_levels(entropy_cfg.image_size, entropy_cfg.grid_levels)
    n_x85 = sum(g * g for g in nested_levels)
    n_x9 = len(asymmetry_cfg.scales) * len(asymmetry_cfg.kinds)
    desc_spec = DescriptorSpec(
        image_size=entropy_cfg.image_size,
        bins=entropy_cfg.bins,
        grid_levels=nested_levels,
        asym_scales=asymmetry_cfg.scales,
        asym_kinds=asymmetry_cfg.kinds,
        schema=asymmetry_cfg.schema,
        n_x85=n_x85,
        n_x9=n_x9,
        n_x94=n_x85 + n_x9,
    )

    # [c_imp.paths]
    p_tbl = _require(top, "paths", "[c_imp]")
    if not isinstance(p_tbl, dict):
        raise ValueError("[c_imp.paths] must be a table")
    paths = CImpPaths(
        runtime_dir=(root / str(_require(p_tbl, "runtime_dir", "[c_imp.paths]"))).resolve(),
        app_dir=(root / str(_require(p_tbl, "app_dir", "[c_imp.paths]"))).resolve(),
        c_tests_dir=(root / str(_require(p_tbl, "c_tests_dir", "[c_imp.paths]"))).resolve(),
        generated_root=(root / str(_require(p_tbl, "generated_dir", "[c_imp.paths]"))).resolve(),
        build_root=(root / str(_require(p_tbl, "build_dir", "[c_imp.paths]"))).resolve(),
        results_root=(root / str(_require(p_tbl, "results_dir", "[c_imp.paths]"))).resolve(),
    )

    # [c_imp.numeric]
    num_tbl = _require(top, "numeric", "[c_imp]")
    if not isinstance(num_tbl, dict):
        raise ValueError("[c_imp.numeric] must be a table")
    raw_modes = _require(num_tbl, "modes", "[c_imp.numeric]")
    if not isinstance(raw_modes, list) or not raw_modes:
        raise ValueError("'modes' in [c_imp.numeric] must be a non-empty list")
    modes = tuple(str(m).strip().lower() for m in raw_modes)
    for m in modes:
        if m not in ALLOWED_NUMERIC_MODES:
            raise ValueError(f"Unsupported numeric mode '{m}'. Allowed: {ALLOWED_NUMERIC_MODES}")
    feature_frac = _require_int(num_tbl, "feature_frac_bits", "[c_imp.numeric]")
    if not (1 <= feature_frac <= 15):
        raise ValueError(f"feature_frac_bits must be in [1, 15], got {feature_frac}")
    acc_bits = _require_int(num_tbl, "accumulator_bits", "[c_imp.numeric]")
    if acc_bits not in (32, 64):
        raise ValueError(f"accumulator_bits must be 32 or 64, got {acc_bits}")
    w_bits = _require_int(num_tbl, "weight_storage_bits", "[c_imp.numeric]")
    if w_bits not in (8, 16):
        raise ValueError(f"weight_storage_bits must be 8 or 16, got {w_bits}")
    rounding = str(_require(num_tbl, "rounding", "[c_imp.numeric]")).strip().lower()
    if rounding != "half_up":
        raise ValueError(f"Unsupported rounding '{rounding}'; expected 'half_up'")
    numeric = CImpNumeric(
        modes=modes,
        feature_frac_bits=feature_frac,
        accumulator_bits=acc_bits,
        weight_storage_bits=w_bits,
        rounding=rounding,
    )

    # [c_imp.descriptor]
    d_tbl = _require(top, "descriptor", "[c_imp]")
    if not isinstance(d_tbl, dict):
        raise ValueError("[c_imp.descriptor] must be a table")
    shannon_mode = str(_require(d_tbl, "shannon_lut_mode", "[c_imp.descriptor]")).strip().lower()
    if shannon_mode not in ALLOWED_SHANNON_MODES:
        raise ValueError(
            f"Unknown shannon_lut_mode '{shannon_mode}'. Allowed: {ALLOWED_SHANNON_MODES}"
        )
    mant_bits = _require_int(d_tbl, "log2_mantissa_bits", "[c_imp.descriptor]")
    if not (4 <= mant_bits <= 12):
        raise ValueError(f"log2_mantissa_bits must be in [4, 12], got {mant_bits}")
    descriptor = CImpDescriptor(shannon_lut_mode=shannon_mode, log2_mantissa_bits=mant_bits)

    # [c_imp.classifier]
    cls_tbl = _require(top, "classifier", "[c_imp]")
    if not isinstance(cls_tbl, dict):
        raise ValueError("[c_imp.classifier] must be a table")
    raw_cls_models = _require(cls_tbl, "models", "[c_imp.classifier]")
    if not isinstance(raw_cls_models, list) or not raw_cls_models:
        raise ValueError("[c_imp.classifier].models must be a non-empty list")
    cls_models = tuple(str(m).strip().lower() for m in raw_cls_models)
    for cm in cls_models:
        if cm not in ALLOWED_CLASSIFIER_MODELS:
            raise ValueError(
                f"Unsupported classifier export model '{cm}'. Allowed: {ALLOWED_CLASSIFIER_MODELS}"
            )
    classifier = CImpClassifier(
        models=cls_models,
        metrics_crosscheck_dir=(
            root / str(_require(cls_tbl, "metrics_crosscheck_dir", "[c_imp.classifier]"))
        ).resolve(),
        metrics_crosscheck_abs_tol=float(
            _require(cls_tbl, "metrics_crosscheck_abs_tol", "[c_imp.classifier]")
        ),
    )

    # [c_imp.cnn]
    cnn_tbl = _require(top, "cnn", "[c_imp]")
    if not isinstance(cnn_tbl, dict):
        raise ValueError("[c_imp.cnn] must be a table")
    raw_cnn_models = _require(cnn_tbl, "models", "[c_imp.cnn]")
    if not isinstance(raw_cnn_models, list) or not raw_cnn_models:
        raise ValueError("[c_imp.cnn].models must be a non-empty list")
    cnn_models = tuple(str(m).strip() for m in raw_cnn_models)
    for m in cnn_models:
        if m not in MODEL_REGISTRY:
            raise ValueError(f"Unsupported CNN model '{m}'. Available: {list(MODEL_REGISTRY.keys())}")
    raw_w_fmts = _require(cnn_tbl, "allowed_weight_formats", "[c_imp.cnn]")
    if not isinstance(raw_w_fmts, list) or not raw_w_fmts:
        raise ValueError("[c_imp.cnn].allowed_weight_formats must be a non-empty list")
    w_fmts = tuple(str(w).strip() for w in raw_w_fmts)
    for wf in w_fmts:
        if wf not in ALLOWED_WEIGHT_FORMATS:
            raise ValueError(f"Unsupported weight format '{wf}'. Allowed: {ALLOWED_WEIGHT_FORMATS}")
    input_frac = _require_int(cnn_tbl, "input_frac_bits", "[c_imp.cnn]")
    if not (1 <= input_frac <= 15):
        raise ValueError(f"input_frac_bits must be in [1, 15], got {input_frac}")
    flash_budget = _require_int(cnn_tbl, "flash_budget_bytes", "[c_imp.cnn]")
    sram_budget = _require_int(cnn_tbl, "sram_budget_bytes", "[c_imp.cnn]")
    psram_budget = _require_int(cnn_tbl, "psram_budget_bytes", "[c_imp.cnn]")
    if flash_budget <= 0 or sram_budget <= 0 or psram_budget < 0:
        raise ValueError("Memory budgets must be positive (psram >= 0)")
    cnn_sec = CImpCnn(
        models=cnn_models,
        allowed_weight_formats=w_fmts,
        flash_budget_bytes=flash_budget,
        sram_budget_bytes=sram_budget,
        psram_budget_bytes=psram_budget,
        checkpoint_variant=str(_require(cnn_tbl, "checkpoint_variant", "[c_imp.cnn]")),
        calibration_split=str(_require(cnn_tbl, "calibration_split", "[c_imp.cnn]")),
        calibration_max_images=_require_int(cnn_tbl, "calibration_max_images", "[c_imp.cnn]"),
        activation_range_percentile=float(
            _require(cnn_tbl, "activation_range_percentile", "[c_imp.cnn]")
        ),
        weight_granularity=str(_require(cnn_tbl, "weight_granularity", "[c_imp.cnn]")),
        input_frac_bits=input_frac,
        fuse_conv_relu_maxpool=bool(cnn_tbl.get("fuse_conv_relu_maxpool", False)),
    )

    # [c_imp.test_vectors]
    tv_tbl = _require(top, "test_vectors", "[c_imp]")
    if not isinstance(tv_tbl, dict):
        raise ValueError("[c_imp.test_vectors] must be a table")
    tv = CImpTestVectors(
        split=str(_require(tv_tbl, "split", "[c_imp.test_vectors]")),
        max_images=_require_int(tv_tbl, "max_images", "[c_imp.test_vectors]"),
        embed_headers_count=_require_int(tv_tbl, "embed_headers_count", "[c_imp.test_vectors]"),
    )

    # [c_imp.build]
    b_tbl = _require(top, "build", "[c_imp]")
    if not isinstance(b_tbl, dict):
        raise ValueError("[c_imp.build] must be a table")
    raw_flags = _require(b_tbl, "compiler_flags", "[c_imp.build]")
    if not isinstance(raw_flags, list):
        raise ValueError("compiler_flags must be a list")
    build = CImpBuild(
        compiler=str(_require(b_tbl, "compiler", "[c_imp.build]")),
        c_standard_flag=str(_require(b_tbl, "c_standard_flag", "[c_imp.build]")),
        compiler_flags=tuple(str(f) for f in raw_flags),
        vswhere_hint=str(_require(b_tbl, "vswhere_hint", "[c_imp.build]")),
        executable_name=str(_require(b_tbl, "executable_name", "[c_imp.build]")),
    )

    # [c_imp.verification]
    v_tbl = _require(top, "verification", "[c_imp]")
    if not isinstance(v_tbl, dict):
        raise ValueError("[c_imp.verification] must be a table")
    ver = CImpVerification(
        f32_descriptor_abs_tol=float(_require(v_tbl, "f32_descriptor_abs_tol", "[c_imp.verification]")),
        f32_logit_abs_tol=float(_require(v_tbl, "f32_logit_abs_tol", "[c_imp.verification]")),
        q15_must_be_bit_exact_vs_python_model=bool(
            _require(v_tbl, "q15_must_be_bit_exact_vs_python_model", "[c_imp.verification]")
        ),
        q15_descriptor_report_only=bool(
            _require(v_tbl, "q15_descriptor_report_only", "[c_imp.verification]")
        ),
        timing_warmup_iters=_require_int(v_tbl, "timing_warmup_iters", "[c_imp.verification]"),
        timing_timed_iters=_require_int(v_tbl, "timing_timed_iters", "[c_imp.verification]"),
    )

    # [c_imp.visualization]
    vis_tbl = _require(top, "visualization", "[c_imp]")
    if not isinstance(vis_tbl, dict):
        raise ValueError("[c_imp.visualization] must be a table")
    vis_subdir = str(_require(vis_tbl, "output_subdir", "[c_imp.visualization]")).strip()
    if not vis_subdir:
        raise ValueError("'output_subdir' in [c_imp.visualization] must not be empty")
    vis_dpi = _require_int(vis_tbl, "dpi", "[c_imp.visualization]")
    if vis_dpi <= 0:
        raise ValueError(f"'dpi' in [c_imp.visualization] must be positive, got {vis_dpi}")
    top_k = _require_int(vis_tbl, "top_k_worst_errors", "[c_imp.visualization]")
    if top_k <= 0:
        raise ValueError(
            f"'top_k_worst_errors' in [c_imp.visualization] must be positive, got {top_k}"
        )
    visualization = CImpVisualization(
        output_subdir=vis_subdir,
        dpi=vis_dpi,
        top_k_worst_errors=top_k,
    )

    return CImpConfig(
        config_path=resolved_cfg_path,
        project_root=root,
        target_tag=target_tag,
        object_name=active_object,
        seed=seed,
        inverted=effective_inverted,
        entropy_config_path=(root / entropy_rel).resolve(),
        asymmetry_config_path=(root / asym_rel).resolve(),
        classification_config_path=(root / cls_rel).resolve(),
        cnn_config_path=(root / cnn_rel).resolve(),
        entropy_cfg=entropy_cfg,
        asymmetry_cfg=asymmetry_cfg,
        classification_cfg=classification_cfg,
        cnn_cfg=cnn_cfg,
        descriptor_spec=desc_spec,
        paths=paths,
        numeric=numeric,
        descriptor=descriptor,
        classifier=classifier,
        cnn=cnn_sec,
        test_vectors=tv,
        build=build,
        verification=ver,
        visualization=visualization,
    )
