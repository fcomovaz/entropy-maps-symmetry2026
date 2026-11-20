"""Configuration loading, path anchoring, and fail-fast validation for `hw_test`.

Validates board hardware limits, wire protocol timing, image geometry, and model
allowlists against the referenced `c_imp` TOML configuration without requiring
firmware source directories or WiFi secrets to exist at load time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from types import MappingProxyType
from typing import Mapping

from configs.logging import logger
from src.c_export.config import CImpConfig, _require, _require_int, load_c_imp_config

ALLOWED_BOARD_FAMILIES: tuple[str, ...] = ("avr", "esp8266", "esp32")
ALLOWED_TRANSPORTS: tuple[str, ...] = ("uart", "wifi")
ALLOWED_ARENA_LOCATIONS: tuple[str, ...] = ("none", "sram", "psram", "auto")
ALLOWED_NUMERIC_MODES: tuple[str, ...] = ("q15", "f32")
ALLOWED_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {"warmup_iters", "timed_iters", "result_timeout_s", "numeric_modes", "arena_location"}
)


def _require_table(parent: Mapping[str, object], key: str, where: str) -> Mapping[str, object]:
    """Return a required TOML subtable or raise KeyError/ValueError with context."""
    val = _require(parent, key, where)
    if not isinstance(val, dict):
        raise ValueError(f"Section '{key}' in {where} must be a TOML table")
    return val


def _require_str(table: Mapping[str, object], key: str, where: str, *, allow_empty: bool = False) -> str:
    """Return a required string from a TOML table."""
    val = _require(table, key, where)
    if not isinstance(val, str):
        raise ValueError(f"Key '{key}' in {where} must be a string, got {val!r}")
    if not allow_empty and not val.strip():
        raise ValueError(f"Key '{key}' in {where} must not be empty")
    return val


def _require_float(table: Mapping[str, object], key: str, where: str) -> float:
    """Return a non-boolean numeric value as float from a TOML table."""
    val = _require(table, key, where)
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"Key '{key}' in {where} must be a number, got {val!r}")
    return float(val)


def _require_bool(table: Mapping[str, object], key: str, where: str) -> bool:
    """Return an exact boolean from a TOML table."""
    val = _require(table, key, where)
    if not isinstance(val, bool):
        raise ValueError(f"Key '{key}' in {where} must be a boolean, got {val!r}")
    return val


def _require_str_list(
    table: Mapping[str, object],
    key: str,
    where: str,
    *,
    allow_empty: bool = False,
    lowercase: bool = False,
) -> tuple[str, ...]:
    """Return a validated tuple of strings from a TOML list."""
    val = _require(table, key, where)
    if not isinstance(val, list):
        raise ValueError(f"Key '{key}' in {where} must be a list of strings, got {val!r}")
    if not allow_empty and not val:
        raise ValueError(f"Key '{key}' in {where} must be a non-empty list")
    out: list[str] = []
    for idx, item in enumerate(val):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Element {idx} of '{key}' in {where} must be a non-empty string")
        cleaned = item.strip().lower() if lowercase else item.strip()
        out.append(cleaned)
    return tuple(out)


def _validate_unique_allowed(
    values: tuple[str, ...],
    allowed: tuple[str, ...],
    key: str,
    where: str,
) -> None:
    """Validate that values contain no duplicates and only permitted members."""
    seen: set[str] = set()
    for val in values:
        if val not in allowed:
            raise ValueError(f"Invalid entry '{val}' in '{key}' at {where}. Allowed: {allowed}")
        if val in seen:
            raise ValueError(f"Duplicate entry '{val}' in '{key}' at {where}")
        seen.add(val)


@dataclass(frozen=True, slots=True)
class Paths:
    """Project-root-anchored paths for images, firmware, staging, results, and C runtime."""

    images_dir: Path
    firmware_dir: Path
    staging_dir: Path
    results_dir: Path
    runtime_dir: Path


@dataclass(frozen=True, slots=True)
class ImagesSpec:
    """Image discovery and raw pixel validation parameters."""

    allowed_extensions: tuple[str, ...]
    width: int
    height: int
    mode: str


@dataclass(frozen=True, slots=True)
class ProtocolSpec:
    """Binary framing and host-side timeout/retry policy."""

    magic: int
    version: int
    ack_timeout_s: float
    result_timeout_s: float
    max_retries: int


@dataclass(frozen=True, slots=True)
class TimingSpec:
    """Default warm-up and timed iteration counts."""

    warmup_iters: int
    timed_iters: int


@dataclass(frozen=True, slots=True)
class FitSpec:
    """Latency viability threshold and offline metrics directories."""

    max_latency_ms: float
    classification_results_dir: Path
    cnn_results_dir: Path
    c_imp_results_dir: Path


@dataclass(frozen=True, slots=True)
class ArduinoCliSpec:
    """`arduino-cli` binary and extra compile arguments."""

    executable: str
    compile_extra_args: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelOverride:
    """Optional per-(board, model) overrides for iterations, timeout, modes, and arena."""

    warmup_iters: int | None = None
    timed_iters: int | None = None
    result_timeout_s: float | None = None
    numeric_modes: tuple[str, ...] | None = None
    arena_location: str | None = None


@dataclass(frozen=True, slots=True)
class EffectiveModelSettings:
    """Resolved runtime parameters for a specific `(board, model)` pair."""

    board_id: str
    model: str
    warmup_iters: int
    timed_iters: int
    result_timeout_s: float
    numeric_modes: tuple[str, ...]
    arena_location: str


@dataclass(frozen=True, slots=True)
class BoardSpec:
    """Validated hardware profile, memory budgets, transport settings, and model allowlist."""

    board_id: str
    fqbn: str
    firmware: str
    family: str
    transports: tuple[str, ...]
    uart_baud: int
    upload_baud: int | None
    serial_reset_on_open: bool
    boot_wait_s: float
    tcp_port: int | None
    physical_sram_bytes: int
    usable_sram_budget_bytes: int
    flash_app_bytes: int
    psram_bytes: int
    cpu_hz: int
    streaming: bool
    band_rows: int
    max_payload_bytes: int
    warmup_iters: int
    timed_iters: int
    result_timeout_s: float
    numeric_modes: tuple[str, ...]
    models: tuple[str, ...]
    arena_location: str
    model_overrides: Mapping[str, ModelOverride]

    def effective_settings(self, model: str) -> EffectiveModelSettings:
        """Return effective timing, mode, and arena policy for `model` on this board."""
        if model not in self.models:
            raise ValueError(
                f"Model '{model}' is not configured for board '{self.board_id}'. "
                f"Configured models: {self.models}"
            )
        override = self.model_overrides.get(model)
        if override is None:
            return EffectiveModelSettings(
                board_id=self.board_id,
                model=model,
                warmup_iters=self.warmup_iters,
                timed_iters=self.timed_iters,
                result_timeout_s=self.result_timeout_s,
                numeric_modes=self.numeric_modes,
                arena_location=self.arena_location,
            )
        return EffectiveModelSettings(
            board_id=self.board_id,
            model=model,
            warmup_iters=self.warmup_iters if override.warmup_iters is None else override.warmup_iters,
            timed_iters=self.timed_iters if override.timed_iters is None else override.timed_iters,
            result_timeout_s=(
                self.result_timeout_s
                if override.result_timeout_s is None
                else override.result_timeout_s
            ),
            numeric_modes=(
                self.numeric_modes if override.numeric_modes is None else override.numeric_modes
            ),
            arena_location=(
                self.arena_location if override.arena_location is None else override.arena_location
            ),
        )


@dataclass(frozen=True, slots=True)
class HwTestConfig:
    """Root validated hardware test bench configuration."""

    config_path: Path
    project_root: Path
    c_imp_config_path: Path
    secrets_config_path: Path
    seed: int
    c_imp_cfg: CImpConfig
    paths: Paths
    images: ImagesSpec
    protocol: ProtocolSpec
    timing: TimingSpec
    fit: FitSpec
    arduino_cli: ArduinoCliSpec
    boards: Mapping[str, BoardSpec]

    def get_board(self, board_id: str) -> BoardSpec:
        """Return the validated `BoardSpec` for `board_id` or raise ValueError."""
        if board_id not in self.boards:
            raise ValueError(
                f"Unknown board '{board_id}'. Available boards: {tuple(self.boards.keys())}"
            )
        return self.boards[board_id]

    def effective_settings(self, board_id: str, model: str) -> EffectiveModelSettings:
        """Return resolved effective settings for `(board_id, model)`."""
        return self.get_board(board_id).effective_settings(model)

    @property
    def known_models(self) -> tuple[str, ...]:
        """Return all tabular and CNN model names registered in `c_imp_cfg`."""
        return self.c_imp_cfg.classifier.models + self.c_imp_cfg.cnn.models

    def is_cnn_model(self, model: str) -> bool:
        """Return True if `model` is a CNN in `c_imp_cfg`, False if tabular; raise if unknown."""
        if model in self.c_imp_cfg.cnn.models:
            return True
        if model in self.c_imp_cfg.classifier.models:
            return False
        raise ValueError(f"Unknown model '{model}'. Known models: {self.known_models}")


@dataclass(frozen=True, slots=True)
class WifiSecrets:
    """WiFi station credentials loaded lazily only for `--transport wifi`."""

    ssid: str
    password: str


def _parse_model_override(
    raw_tbl: Mapping[str, object],
    model_name: str,
    board_id: str,
    psram_bytes: int,
    c_imp_modes: tuple[str, ...],
) -> ModelOverride:
    """Validate and construct one `[hw_test.boards.<id>.model_overrides.<model>]` entry."""
    where = f"[hw_test.boards.{board_id}.model_overrides.{model_name}]"
    unknown_keys = set(raw_tbl.keys()) - ALLOWED_OVERRIDE_KEYS
    if unknown_keys:
        raise ValueError(f"Unknown key(s) {sorted(unknown_keys)} in {where}")

    warmup: int | None = None
    if "warmup_iters" in raw_tbl:
        warmup = _require_int(raw_tbl, "warmup_iters", where)
        if warmup < 0:
            raise ValueError(f"'warmup_iters' in {where} must be >= 0, got {warmup}")

    timed: int | None = None
    if "timed_iters" in raw_tbl:
        timed = _require_int(raw_tbl, "timed_iters", where)
        if timed < 1:
            raise ValueError(f"'timed_iters' in {where} must be >= 1, got {timed}")

    timeout: float | None = None
    if "result_timeout_s" in raw_tbl:
        timeout = _require_float(raw_tbl, "result_timeout_s", where)
        if timeout <= 0.0:
            raise ValueError(f"'result_timeout_s' in {where} must be > 0, got {timeout}")

    modes: tuple[str, ...] | None = None
    if "numeric_modes" in raw_tbl:
        modes = _require_str_list(raw_tbl, "numeric_modes", where, lowercase=True)
        _validate_unique_allowed(modes, ALLOWED_NUMERIC_MODES, "numeric_modes", where)
        for m in modes:
            if m not in c_imp_modes:
                raise ValueError(
                    f"Numeric mode '{m}' in {where} is not enabled in c_imp config ({c_imp_modes})"
                )

    arena_loc: str | None = None
    if "arena_location" in raw_tbl:
        arena_loc = _require_str(raw_tbl, "arena_location", where).strip().lower()
        if arena_loc not in ALLOWED_ARENA_LOCATIONS:
            raise ValueError(
                f"Invalid arena_location '{arena_loc}' in {where}. Allowed: {ALLOWED_ARENA_LOCATIONS}"
            )
        if arena_loc == "psram" and psram_bytes == 0:
            raise ValueError(f"arena_location='psram' in {where} requires psram_bytes > 0")

    return ModelOverride(
        warmup_iters=warmup,
        timed_iters=timed,
        result_timeout_s=timeout,
        numeric_modes=modes,
        arena_location=arena_loc,
    )


def _parse_board(
    board_id: str,
    b_tbl: Mapping[str, object],
    images: ImagesSpec,
    timing: TimingSpec,
    protocol: ProtocolSpec,
    c_imp_cfg: CImpConfig,
) -> BoardSpec:
    """Validate and parse a single `[hw_test.boards.<id>]` table."""
    where = f"[hw_test.boards.{board_id}]"
    fqbn = _require_str(b_tbl, "fqbn", where).strip()
    firmware = _require_str(b_tbl, "firmware", where).strip()
    family = _require_str(b_tbl, "family", where).strip().lower()
    if family not in ALLOWED_BOARD_FAMILIES:
        raise ValueError(
            f"Invalid family '{family}' in {where}. Allowed: {ALLOWED_BOARD_FAMILIES}"
        )

    transports = _require_str_list(b_tbl, "transports", where, lowercase=True)
    _validate_unique_allowed(transports, ALLOWED_TRANSPORTS, "transports", where)

    tcp_port: int | None = None
    if "wifi" in transports:
        tcp_port = _require_int(b_tbl, "tcp_port", where)
        if not (1 <= tcp_port <= 65535):
            raise ValueError(f"'tcp_port' in {where} must be in [1, 65535], got {tcp_port}")
    elif "tcp_port" in b_tbl:
        tcp_port = _require_int(b_tbl, "tcp_port", where)
        if not (1 <= tcp_port <= 65535):
            raise ValueError(f"'tcp_port' in {where} must be in [1, 65535], got {tcp_port}")

    uart_baud = _require_int(b_tbl, "uart_baud", where)
    if uart_baud <= 0:
        raise ValueError(f"'uart_baud' in {where} must be > 0, got {uart_baud}")

    upload_baud: int | None = None
    if "upload_baud" in b_tbl:
        upload_baud = _require_int(b_tbl, "upload_baud", where)
        if upload_baud <= 0:
            raise ValueError(f"'upload_baud' in {where} must be > 0, got {upload_baud}")

    serial_reset_on_open = _require_bool(b_tbl, "serial_reset_on_open", where)

    boot_wait_s = _require_float(b_tbl, "boot_wait_s", where)
    if boot_wait_s < 0.0:
        raise ValueError(f"'boot_wait_s' in {where} must be >= 0, got {boot_wait_s}")

    physical_sram = _require_int(b_tbl, "physical_sram_bytes", where)
    usable_sram = _require_int(b_tbl, "usable_sram_budget_bytes", where)
    flash_app = _require_int(b_tbl, "flash_app_bytes", where)
    psram = _require_int(b_tbl, "psram_bytes", where)
    cpu_hz = _require_int(b_tbl, "cpu_hz", where)
    if physical_sram <= 0 or usable_sram <= 0 or flash_app <= 0 or cpu_hz <= 0:
        raise ValueError(
            f"Memory budgets and cpu_hz in {where} must be positive: "
            f"physical_sram={physical_sram}, usable_sram={usable_sram}, "
            f"flash_app={flash_app}, cpu_hz={cpu_hz}"
        )
    if usable_sram > physical_sram:
        raise ValueError(
            f"'usable_sram_budget_bytes' ({usable_sram}) in {where} cannot exceed "
            f"'physical_sram_bytes' ({physical_sram})"
        )
    if psram < 0:
        raise ValueError(f"'psram_bytes' in {where} must be >= 0, got {psram}")

    streaming = _require_bool(b_tbl, "streaming", where)
    band_rows = _require_int(b_tbl, "band_rows", where)
    if band_rows <= 0 or (images.height % band_rows) != 0:
        raise ValueError(
            f"'band_rows' ({band_rows}) in {where} must be > 0 and divide image height ({images.height})"
        )

    max_payload = _require_int(b_tbl, "max_payload_bytes", where)
    min_payload = 4 + band_rows * images.width
    if max_payload < min_payload:
        raise ValueError(
            f"'max_payload_bytes' ({max_payload}) in {where} must be >= 4 + band_rows * width ({min_payload})"
        )
    if max_payload > 65535:
        raise ValueError(
            f"'max_payload_bytes' ({max_payload}) in {where} exceeds uint16 wire frame limit (65535)"
        )

    warmup_iters = (
        _require_int(b_tbl, "warmup_iters", where)
        if "warmup_iters" in b_tbl
        else timing.warmup_iters
    )
    if warmup_iters < 0:
        raise ValueError(f"'warmup_iters' in {where} must be >= 0, got {warmup_iters}")

    timed_iters = (
        _require_int(b_tbl, "timed_iters", where)
        if "timed_iters" in b_tbl
        else timing.timed_iters
    )
    if timed_iters < 1:
        raise ValueError(f"'timed_iters' in {where} must be >= 1, got {timed_iters}")

    result_timeout_s = (
        _require_float(b_tbl, "result_timeout_s", where)
        if "result_timeout_s" in b_tbl
        else protocol.result_timeout_s
    )
    if result_timeout_s <= 0.0:
        raise ValueError(f"'result_timeout_s' in {where} must be > 0, got {result_timeout_s}")

    numeric_modes = _require_str_list(b_tbl, "numeric_modes", where, lowercase=True)
    _validate_unique_allowed(numeric_modes, ALLOWED_NUMERIC_MODES, "numeric_modes", where)
    for mode in numeric_modes:
        if mode not in c_imp_cfg.numeric.modes:
            raise ValueError(
                f"Numeric mode '{mode}' in {where} is not enabled in c_imp config ({c_imp_cfg.numeric.modes})"
            )

    known_models = c_imp_cfg.classifier.models + c_imp_cfg.cnn.models
    models = _require_str_list(b_tbl, "models", where, lowercase=False)
    _validate_unique_allowed(models, known_models, "models", where)

    arena_location = _require_str(b_tbl, "arena_location", where).strip().lower()
    if arena_location not in ALLOWED_ARENA_LOCATIONS:
        raise ValueError(
            f"Invalid arena_location '{arena_location}' in {where}. Allowed: {ALLOWED_ARENA_LOCATIONS}"
        )
    if arena_location == "psram" and psram == 0:
        raise ValueError(f"arena_location='psram' in {where} requires psram_bytes > 0")

    overrides_dict: dict[str, ModelOverride] = {}
    if "model_overrides" in b_tbl:
        ov_tbl = _require_table(b_tbl, "model_overrides", where)
        for model_name, raw_ov in ov_tbl.items():
            if model_name not in models:
                raise ValueError(
                    f"Model override '{model_name}' in {where}.model_overrides is not in board 'models' ({models})"
                )
            if not isinstance(raw_ov, dict):
                raise ValueError(f"{where}.model_overrides.{model_name} must be a TOML table")
            overrides_dict[model_name] = _parse_model_override(
                raw_ov,
                model_name=model_name,
                board_id=board_id,
                psram_bytes=psram,
                c_imp_modes=c_imp_cfg.numeric.modes,
            )

    return BoardSpec(
        board_id=board_id,
        fqbn=fqbn,
        firmware=firmware,
        family=family,
        transports=transports,
        uart_baud=uart_baud,
        upload_baud=upload_baud,
        serial_reset_on_open=serial_reset_on_open,
        boot_wait_s=boot_wait_s,
        tcp_port=tcp_port,
        physical_sram_bytes=physical_sram,
        usable_sram_budget_bytes=usable_sram,
        flash_app_bytes=flash_app,
        psram_bytes=psram,
        cpu_hz=cpu_hz,
        streaming=streaming,
        band_rows=band_rows,
        max_payload_bytes=max_payload,
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        result_timeout_s=result_timeout_s,
        numeric_modes=numeric_modes,
        models=models,
        arena_location=arena_location,
        model_overrides=MappingProxyType(overrides_dict),
    )


def load_hw_test_config(config_path: Path, project_root: Path) -> HwTestConfig:
    """Load, anchor, and fail-fast validate `configs/hw_test/boards.toml`.

    Does not check that `paths.firmware_dir` or `secrets_config` exist on disk so
    dry-run and non-WiFi runs work cleanly without firmware or secrets.
    """
    root = Path(project_root).resolve()
    cfg_path = Path(config_path)
    resolved_cfg_path = cfg_path.resolve() if cfg_path.is_absolute() else (root / cfg_path).resolve()
    if not resolved_cfg_path.is_file():
        raise FileNotFoundError(f"Hardware test config file not found: {resolved_cfg_path}")

    with resolved_cfg_path.open("rb") as f:
        raw = tomllib.load(f)

    top = _require_table(raw, "hw_test", str(resolved_cfg_path))
    c_imp_rel = Path(_require_str(top, "c_imp_config", "[hw_test]").strip())
    secrets_rel = Path(_require_str(top, "secrets_config", "[hw_test]").strip())
    seed = _require_int(top, "seed", "[hw_test]")

    c_imp_config_path = (
        c_imp_rel.resolve() if c_imp_rel.is_absolute() else (root / c_imp_rel).resolve()
    )
    secrets_config_path = (
        secrets_rel.resolve() if secrets_rel.is_absolute() else (root / secrets_rel).resolve()
    )
    c_imp_cfg = load_c_imp_config(c_imp_config_path, root)

    # [hw_test.paths]
    p_tbl = _require_table(top, "paths", "[hw_test]")
    paths = Paths(
        images_dir=(root / _require_str(p_tbl, "images_dir", "[hw_test.paths]").strip()).resolve(),
        firmware_dir=(
            root / _require_str(p_tbl, "firmware_dir", "[hw_test.paths]").strip()
        ).resolve(),
        staging_dir=(
            root / _require_str(p_tbl, "staging_dir", "[hw_test.paths]").strip()
        ).resolve(),
        results_dir=(
            root / _require_str(p_tbl, "results_dir", "[hw_test.paths]").strip()
        ).resolve(),
        runtime_dir=(
            root / _require_str(p_tbl, "runtime_dir", "[hw_test.paths]").strip()
        ).resolve(),
    )

    # [hw_test.images]
    im_tbl = _require_table(top, "images", "[hw_test]")
    exts = _require_str_list(im_tbl, "allowed_extensions", "[hw_test.images]", lowercase=True)
    for ext in exts:
        if not ext.startswith("."):
            raise ValueError(
                f"Extension '{ext}' in [hw_test.images].allowed_extensions must start with '.'"
            )
    width = _require_int(im_tbl, "width", "[hw_test.images]")
    height = _require_int(im_tbl, "height", "[hw_test.images]")
    if width <= 0 or height <= 0 or width != height:
        raise ValueError(
            f"[hw_test.images] width ({width}) and height ({height}) must be positive and equal"
        )
    desc_size = c_imp_cfg.descriptor_spec.image_size
    if width != desc_size or height != desc_size:
        raise ValueError(
            f"[hw_test.images] dimensions ({width}x{height}) do not match c_imp descriptor "
            f"image_size ({desc_size}x{desc_size})"
        )
    mode = _require_str(im_tbl, "mode", "[hw_test.images]").strip()
    if mode != "L":
        raise ValueError(f"[hw_test.images].mode must be 'L' (8-bit grayscale), got {mode!r}")
    images = ImagesSpec(allowed_extensions=exts, width=width, height=height, mode=mode)

    # [hw_test.protocol]
    pr_tbl = _require_table(top, "protocol", "[hw_test]")
    magic = _require_int(pr_tbl, "magic", "[hw_test.protocol]")
    if not (0 <= magic <= 0xFFFF):
        raise ValueError(f"'magic' in [hw_test.protocol] must be uint16, got {magic}")
    version = _require_int(pr_tbl, "version", "[hw_test.protocol]")
    if not (1 <= version <= 0xFF):
        raise ValueError(f"'version' in [hw_test.protocol] must be in [1, 255], got {version}")
    ack_timeout_s = _require_float(pr_tbl, "ack_timeout_s", "[hw_test.protocol]")
    result_timeout_s = _require_float(pr_tbl, "result_timeout_s", "[hw_test.protocol]")
    max_retries = _require_int(pr_tbl, "max_retries", "[hw_test.protocol]")
    if ack_timeout_s <= 0.0 or result_timeout_s <= 0.0:
        raise ValueError("[hw_test.protocol] timeouts must be > 0")
    if max_retries < 0:
        raise ValueError(f"'max_retries' in [hw_test.protocol] must be >= 0, got {max_retries}")
    protocol = ProtocolSpec(
        magic=magic,
        version=version,
        ack_timeout_s=ack_timeout_s,
        result_timeout_s=result_timeout_s,
        max_retries=max_retries,
    )

    # [hw_test.timing]
    tm_tbl = _require_table(top, "timing", "[hw_test]")
    warmup_iters = _require_int(tm_tbl, "warmup_iters", "[hw_test.timing]")
    timed_iters = _require_int(tm_tbl, "timed_iters", "[hw_test.timing]")
    if warmup_iters < 0:
        raise ValueError(f"'warmup_iters' in [hw_test.timing] must be >= 0, got {warmup_iters}")
    if timed_iters < 1:
        raise ValueError(f"'timed_iters' in [hw_test.timing] must be >= 1, got {timed_iters}")
    timing = TimingSpec(warmup_iters=warmup_iters, timed_iters=timed_iters)

    # [hw_test.fit]
    ft_tbl = _require_table(top, "fit", "[hw_test]")
    max_latency_ms = _require_float(ft_tbl, "max_latency_ms", "[hw_test.fit]")
    if max_latency_ms <= 0.0:
        raise ValueError(f"'max_latency_ms' in [hw_test.fit] must be > 0, got {max_latency_ms}")
    fit = FitSpec(
        max_latency_ms=max_latency_ms,
        classification_results_dir=(
            root / _require_str(ft_tbl, "classification_results_dir", "[hw_test.fit]").strip()
        ).resolve(),
        cnn_results_dir=(
            root / _require_str(ft_tbl, "cnn_results_dir", "[hw_test.fit]").strip()
        ).resolve(),
        c_imp_results_dir=(
            root / _require_str(ft_tbl, "c_imp_results_dir", "[hw_test.fit]").strip()
        ).resolve(),
    )

    # [hw_test.arduino_cli]
    ac_tbl = _require_table(top, "arduino_cli", "[hw_test]")
    ac_exe = _require_str(ac_tbl, "executable", "[hw_test.arduino_cli]").strip()
    ac_args = _require_str_list(
        ac_tbl, "compile_extra_args", "[hw_test.arduino_cli]", allow_empty=True
    )
    arduino_cli = ArduinoCliSpec(executable=ac_exe, compile_extra_args=ac_args)

    # [hw_test.boards]
    boards_tbl = _require_table(top, "boards", "[hw_test]")
    if not boards_tbl:
        raise ValueError("[hw_test.boards] must define at least one board")
    boards: dict[str, BoardSpec] = {}
    for board_id, raw_board in boards_tbl.items():
        if not isinstance(raw_board, dict):
            raise ValueError(f"[hw_test.boards.{board_id}] must be a TOML table")
        boards[board_id] = _parse_board(
            board_id=board_id,
            b_tbl=raw_board,
            images=images,
            timing=timing,
            protocol=protocol,
            c_imp_cfg=c_imp_cfg,
        )

    logger.debug("Loaded hw_test config %s with boards=%s", resolved_cfg_path, list(boards.keys()))
    return HwTestConfig(
        config_path=resolved_cfg_path,
        project_root=root,
        c_imp_config_path=c_imp_config_path,
        secrets_config_path=secrets_config_path,
        seed=seed,
        c_imp_cfg=c_imp_cfg,
        paths=paths,
        images=images,
        protocol=protocol,
        timing=timing,
        fit=fit,
        arduino_cli=arduino_cli,
        boards=MappingProxyType(boards),
    )


def load_wifi_secrets(path: Path) -> WifiSecrets:
    """Load WiFi STA credentials from `configs/hw_test/secrets.toml` when WiFi is used."""
    secrets_path = Path(path)
    if not secrets_path.is_file():
        raise FileNotFoundError(
            f"WiFi secrets file not found: {secrets_path}. "
            "Copy configs/hw_test/secrets.example.toml and fill in your credentials."
        )
    with secrets_path.open("rb") as f:
        raw = tomllib.load(f)
    wifi_tbl = _require_table(raw, "wifi", str(secrets_path))
    ssid = _require_str(wifi_tbl, "ssid", "[wifi]").strip()
    password = _require_str(wifi_tbl, "password", "[wifi]", allow_empty=True)
    return WifiSecrets(ssid=ssid, password=password)
