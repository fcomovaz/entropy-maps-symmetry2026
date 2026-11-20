"""Sketch staging, single-model registry generation, and `arduino-cli` build/flash integration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Callable, Mapping

from configs.logging import logger
from src.hw_test.config import BoardSpec, HwTestConfig, WifiSecrets, load_wifi_secrets
from src.hw_test.protocol import ARENA_LOC_NONE, ARENA_LOC_PSRAM, ARENA_LOC_SRAM

ARENA_PREF_TO_INT: dict[str, int] = {
    "none": ARENA_LOC_NONE,
    "sram": ARENA_LOC_SRAM,
    "psram": ARENA_LOC_PSRAM,
    "auto": 3,
}

_FLASH_RE = re.compile(r"Code in flash.*?used\s+(\d+)\s*/\s*(\d+)\s*bytes", re.IGNORECASE)
_RAM_RE = re.compile(
    r"Variables and constants in RAM.*?used\s+(\d+)\s*/\s*(\d+)\s*bytes", re.IGNORECASE
)
_IRAM_RE = re.compile(r"Instruction RAM.*?used\s+(\d+)\s*/\s*(\d+)\s*bytes", re.IGNORECASE)
_RODATA_RE = re.compile(r"RODATA\s+(\d+)", re.IGNORECASE)

SubprocessRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class BuildResult:
    """Measured compiler outcome, memory footprint, and board budget verdict."""

    board_id: str
    model: str
    mode: str
    transport: str
    fqbn: str
    sketch_dir: Path
    build_dir: Path
    compile_ok: bool
    flash_used_bytes: int | None
    flash_max_bytes: int | None
    ram_used_bytes: int | None
    ram_max_bytes: int | None
    arena_bytes: int
    fits_flash: bool
    fits_ram: bool
    fits_arena: bool
    fits_budget: bool
    reason: str
    core_id: str
    core_version: str | None
    cli_version: str | None
    compiler_out: str
    compiler_err: str


@dataclass(frozen=True, slots=True)
class ConnectedBoard:
    """One serial/port entry reported by `arduino-cli board list`."""

    port: str
    protocol: str
    protocol_label: str
    board_name: str
    fqbn: str | None
    vid: str | None
    pid: str | None


@dataclass(frozen=True, slots=True)
class ToolchainInfo:
    """Installed `arduino-cli` and core version metadata."""

    cli_version: str
    core_versions: Mapping[str, str]


def _default_runner(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 600,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=timeout,
    )


def _c_escape(value: str) -> str:
    """Escape a string literal safely for C preprocessor `#define` string emission."""
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )


def _parse_json_output(raw: str, context: str) -> Any:
    cleaned = raw.lstrip("\ufeff").strip()
    if not cleaned:
        raise RuntimeError(f"Empty JSON output from {context}")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON output from {context}: {exc}") from exc


def core_id_from_fqbn(fqbn: str) -> str:
    """Extract `packager:arch` core identifier from a full FQBN string."""
    parts = fqbn.split(":")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid FQBN format: {fqbn!r}")
    return f"{parts[0]}:{parts[1]}"


def get_git_short_hash(
    project_root: Path,
    runner: SubprocessRunner = _default_runner,
) -> str:
    """Return 7-character git short commit hash, or `'unknown'` if unavailable."""
    try:
        proc = runner(["git", "rev-parse", "--short=7", "HEAD"], cwd=project_root, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    text = proc.stdout.strip()
    if not text:
        return "unknown"
    return text[:8]


def get_toolchain_versions(
    cfg: HwTestConfig,
    runner: SubprocessRunner = _default_runner,
) -> ToolchainInfo:
    """Query installed `arduino-cli` version and installed board cores."""
    exe = cfg.arduino_cli.executable
    ver_proc = runner([exe, "version", "--format", "json"], cwd=cfg.project_root, timeout=30)
    if ver_proc.returncode != 0:
        detail = (ver_proc.stderr or ver_proc.stdout).strip()
        raise RuntimeError(f"Failed to run '{exe} version': {detail}")
    ver_json = _parse_json_output(ver_proc.stdout, f"{exe} version")
    cli_version = str(ver_json.get("VersionString") or ver_json.get("version") or "unknown")

    core_proc = runner(
        [exe, "core", "list", "--format", "json"], cwd=cfg.project_root, timeout=30
    )
    if core_proc.returncode != 0:
        detail = (core_proc.stderr or core_proc.stdout).strip()
        raise RuntimeError(f"Failed to run '{exe} core list': {detail}")
    core_json = _parse_json_output(core_proc.stdout, f"{exe} core list")

    platforms = (
        core_json.get("platforms", []) if isinstance(core_json, dict) else core_json
    )
    versions: dict[str, str] = {}
    if isinstance(platforms, list):
        for item in platforms:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("id", "")).strip()
            inst = str(item.get("installed_version") or item.get("installed") or "").strip()
            if pid and inst:
                versions[pid] = inst
    return ToolchainInfo(cli_version=cli_version, core_versions=versions)


def ensure_core_installed(
    cfg: HwTestConfig,
    fqbn: str,
    runner: SubprocessRunner = _default_runner,
) -> ToolchainInfo:
    """Verify the Arduino core required by `fqbn` is installed, or fail fast with install cmd."""
    core_id = core_id_from_fqbn(fqbn)
    info = get_toolchain_versions(cfg, runner=runner)
    if core_id not in info.core_versions:
        raise RuntimeError(
            f"Required Arduino core '{core_id}' is not installed. "
            f"Install it with: {cfg.arduino_cli.executable} core install {core_id}"
        )
    return info


def list_connected_boards(
    cfg: HwTestConfig,
    runner: SubprocessRunner = _default_runner,
) -> list[ConnectedBoard]:
    """Query `arduino-cli board list --format json`, marking unrecognized serial ports unknown."""
    exe = cfg.arduino_cli.executable
    proc = runner([exe, "board", "list", "--format", "json"], cwd=cfg.project_root, timeout=30)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(f"Failed to run '{exe} board list': {detail}")
    data = _parse_json_output(proc.stdout, f"{exe} board list")

    raw_ports = data.get("detected_ports", []) if isinstance(data, dict) else data
    boards: list[ConnectedBoard] = []
    if not isinstance(raw_ports, list):
        return boards

    for entry in raw_ports:
        if not isinstance(entry, dict):
            continue
        port_obj = entry.get("port") or {}
        address = str(port_obj.get("address") or port_obj.get("label") or "").strip()
        if not address:
            continue
        props = port_obj.get("properties") or {}
        matching = entry.get("matching_boards") or []
        if matching and isinstance(matching, list) and isinstance(matching[0], dict):
            board_name = str(matching[0].get("name") or "unknown")
            fqbn = matching[0].get("fqbn")
            fqbn_str = str(fqbn) if fqbn else None
        else:
            board_name = "unknown"
            fqbn_str = None
        boards.append(
            ConnectedBoard(
                port=address,
                protocol=str(port_obj.get("protocol") or "unknown"),
                protocol_label=str(port_obj.get("protocol_label") or ""),
                board_name=board_name,
                fqbn=fqbn_str,
                vid=str(props["vid"]) if props.get("vid") else None,
                pid=str(props["pid"]) if props.get("pid") else None,
            )
        )
    return boards


def get_model_arena_bytes(cfg: HwTestConfig, model: str, mode: str) -> int:
    """Return predicted CNN activation arena bytes for `(model, mode)`, or 0 for tabular."""
    if not cfg.is_cnn_model(model):
        return 0
    norm_mode = mode.strip().lower()
    if norm_mode not in ("q15", "f32"):
        raise ValueError(f"Unsupported mode: {mode!r}")

    mem_path = cfg.c_imp_cfg.results_dir / "memory_report.json"
    if mem_path.is_file():
        data = json.loads(mem_path.read_text(encoding="utf-8"))
        for row in data.get("models", []):
            if row.get("model_name") == model:
                key = "peak_arena_q15_bytes" if norm_mode == "q15" else "peak_arena_f32_bytes"
                if key in row:
                    return int(row[key])

    header_path = cfg.c_imp_cfg.generated_dir / f"model_{model}.h"
    if header_path.is_file():
        macro = (
            f"CIMP_CNN_{model.upper()}_ARENA_Q15_BYTES"
            if norm_mode == "q15"
            else f"CIMP_CNN_{model.upper()}_ARENA_F32_BYTES"
        )
        m = re.search(rf"#define\s+{re.escape(macro)}\s+(\d+)", header_path.read_text(encoding="utf-8"))
        if m:
            return int(m.group(1))

    raise FileNotFoundError(
        f"Cannot determine arena bytes for CNN '{model}' ({norm_mode}): "
        f"neither {mem_path} nor {header_path} contained arena metadata"
    )


def _validate_target_combination(
    cfg: HwTestConfig,
    board: str,
    model: str,
    mode: str,
    transport: str,
) -> tuple[BoardSpec, bool]:
    board_spec = cfg.get_board(board)
    eff = board_spec.effective_settings(model)
    norm_mode = mode.strip().lower()
    norm_transport = transport.strip().lower()
    if norm_mode not in eff.numeric_modes:
        raise ValueError(
            f"Mode '{norm_mode}' is not enabled for ({board}, {model}). "
            f"Allowed: {eff.numeric_modes}"
        )
    if norm_transport not in board_spec.transports:
        raise ValueError(
            f"Transport '{norm_transport}' is not supported on board '{board}'. "
            f"Allowed: {board_spec.transports}"
        )
    is_cnn = cfg.is_cnn_model(model)
    if board_spec.streaming and is_cnn:
        raise ValueError(f"Streaming board '{board}' cannot run CNN model '{model}'")
    if board_spec.streaming and norm_mode != "q15":
        raise ValueError(f"Streaming board '{board}' supports 'q15' only, got '{norm_mode}'")
    return board_spec, is_cnn


def _render_single_model_registry(cfg: HwTestConfig, model: str, arena_bytes: int) -> str:
    lines = [
        "/* AUTO-GENERATED STAGING FILE - DO NOT EDIT BY HAND. */",
        "#ifndef CIMP_MODELS_REGISTRY_H",
        "#define CIMP_MODELS_REGISTRY_H",
        "",
    ]
    for tab in cfg.c_imp_cfg.classifier.models:
        macro = f"CIMP_HAS_MODEL_{tab.upper()}"
        enabled = 1 if tab == model else 0
        lines.append(f"#define {macro} {enabled}")
        if enabled:
            lines.append(f'#include "model_{tab}.h"')
    for cnn in cfg.c_imp_cfg.cnn.models:
        macro = f"CIMP_HAS_CNN_{cnn.upper()}"
        enabled = 1 if cnn == model else 0
        lines.append(f"#define {macro} {enabled}")
        if enabled:
            lines.append(f'#include "model_{cnn}.h"')
    lines.extend(
        [
            "",
            f"#define CIMP_MAX_CNN_ARENA_BYTES {arena_bytes}",
            "",
            "#endif /* CIMP_MODELS_REGISTRY_H */",
            "",
        ]
    )
    return "\n".join(lines)


def _render_hw_build_config(
    *,
    cfg: HwTestConfig,
    board_spec: BoardSpec,
    model: str,
    is_cnn: bool,
    mode: str,
    transport: str,
    wifi_secrets: WifiSecrets | None,
    arena_bytes: int,
    arena_pref_name: str,
    git_hash: str,
) -> str:
    timer_res_us = 4 if board_spec.family == "avr" else 1
    use_wifi = 1 if transport == "wifi" else 0
    ssid = wifi_secrets.ssid if wifi_secrets is not None else ""
    password = wifi_secrets.password if wifi_secrets is not None else ""
    f32_enabled = 0 if mode == "q15" else 1
    mode_q15 = 1 if mode == "q15" else 0
    arena_pref = ARENA_PREF_TO_INT[arena_pref_name]
    tcp_port = board_spec.tcp_port if board_spec.tcp_port is not None else 0

    return "\n".join(
        [
            "/* AUTO-GENERATED STAGING FILE - DO NOT EDIT BY HAND. */",
            "#ifndef HW_BUILD_CONFIG_H",
            "#define HW_BUILD_CONFIG_H",
            "",
            f'#define HW_BOARD_ID "{_c_escape(board_spec.board_id)}"',
            f'#define HW_MODEL_NAME "{_c_escape(model)}"',
            f"#define HW_MODEL_KIND {1 if is_cnn else 0}",
            f"#define HW_MODE_Q15 {mode_q15}",
            f"#define HW_PROTOCOL_VERSION {cfg.protocol.version}",
            f"#define HW_IMAGE_WIDTH {cfg.images.width}",
            f"#define HW_IMAGE_HEIGHT {cfg.images.height}",
            f"#define HW_STREAMING {1 if board_spec.streaming else 0}",
            f"#define HW_BAND_ROWS {board_spec.band_rows}",
            f"#define HW_MAX_PAYLOAD_BYTES {board_spec.max_payload_bytes}",
            f"#define HW_UART_BAUD {board_spec.uart_baud}UL",
            f"#define HW_USE_WIFI {use_wifi}",
            f'#define HW_WIFI_SSID "{_c_escape(ssid)}"',
            f'#define HW_WIFI_PASS "{_c_escape(password)}"',
            f"#define HW_TCP_PORT {tcp_port}",
            f"#define HW_TIMER_RES_US {timer_res_us}",
            f"#define HW_ARENA_BYTES {arena_bytes}UL",
            f"#define HW_ARENA_PREF {arena_pref}",
            f"#define HW_USABLE_SRAM_BUDGET {board_spec.usable_sram_budget_bytes}UL",
            f'#define HW_GIT_HASH "{_c_escape(git_hash)}"',
            "",
            "#ifndef CIMP_ENABLE_F32",
            f"#define CIMP_ENABLE_F32 {f32_enabled}",
            "#endif",
            "",
            "#endif /* HW_BUILD_CONFIG_H */",
            "",
        ]
    )


def _copy_required(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"Required firmware/runtime file not found: {src}")
    shutil.copy2(src, dst)


def stage_sketch(
    cfg: HwTestConfig,
    board: str,
    model: str,
    mode: str,
    transport: str = "uart",
    wifi_secrets: WifiSecrets | None = None,
    *,
    runner: SubprocessRunner = _default_runner,
) -> Path:
    """Assemble an isolated Arduino sketch directory for one `(board, model, mode, transport)`."""
    board_spec, is_cnn = _validate_target_combination(cfg, board, model, mode, transport)
    norm_mode = mode.strip().lower()
    norm_transport = transport.strip().lower()

    resolved_secrets = wifi_secrets
    if norm_transport == "wifi" and resolved_secrets is None:
        resolved_secrets = load_wifi_secrets(cfg.secrets_config_path)

    sketch_name = f"{board_spec.board_id}_{model}_{norm_mode}_{norm_transport}"
    sketch_dir = cfg.paths.staging_dir / sketch_name
    if sketch_dir.exists():
        shutil.rmtree(sketch_dir)
    sketch_dir.mkdir(parents=True, exist_ok=True)

    common_dir = cfg.paths.firmware_dir / "common"
    if not common_dir.is_dir():
        raise FileNotFoundError(f"Firmware common directory not found: {common_dir}")
    for src in sorted(common_dir.iterdir()):
        if src.is_file():
            shutil.copy2(src, sketch_dir / src.name)

    board_fw_dir = cfg.paths.firmware_dir / board_spec.firmware
    ino_src = board_fw_dir / f"{board_spec.firmware}.ino"
    _copy_required(ino_src, sketch_dir / f"{sketch_name}.ino")
    for extra in sorted(board_fw_dir.iterdir()):
        if extra.is_file() and extra.suffix.lower() != ".ino":
            shutil.copy2(extra, sketch_dir / extra.name)

    rt_dir = cfg.paths.runtime_dir
    gen_dir = cfg.c_imp_cfg.generated_dir
    for name in (
        "cimp_fixed_point.h",
        "cimp_tensor.h",
        "cimp_platform.h",
        "cimp_platform.c",
        "cimp_pipeline.h",
        "cimp_pipeline.c",
    ):
        _copy_required(rt_dir / name, sketch_dir / name)

    _copy_required(gen_dir / "cimp_config.h", sketch_dir / "cimp_config.h")

    if is_cnn:
        # Headers only for descriptor/sklearn (cimp_pipeline.h includes them, no .c needed).
        for name in ("cimp_descriptor.h", "infra_sklearn.h", "infra_pytorch.h", "infra_pytorch.c"):
            _copy_required(rt_dir / name, sketch_dir / name)
        _copy_required(gen_dir / f"model_{model}.h", sketch_dir / f"model_{model}.h")
        _copy_required(gen_dir / f"model_{model}.c", sketch_dir / f"model_{model}.c")
    else:
        for name in (
            "cimp_descriptor.h",
            "cimp_descriptor.c",
            "infra_sklearn.h",
            "infra_sklearn.c",
        ):
            _copy_required(rt_dir / name, sketch_dir / name)
        for name in ("lut_bin_quant.h", "lut_shannon.h", f"model_{model}.h"):
            _copy_required(gen_dir / name, sketch_dir / name)

    arena_bytes = get_model_arena_bytes(cfg, model, norm_mode)
    eff = board_spec.effective_settings(model)
    git_hash = get_git_short_hash(cfg.project_root, runner=runner)

    (sketch_dir / "cimp_models_registry.h").write_text(
        _render_single_model_registry(cfg, model, arena_bytes),
        encoding="utf-8",
    )
    (sketch_dir / "hw_build_config.h").write_text(
        _render_hw_build_config(
            cfg=cfg,
            board_spec=board_spec,
            model=model,
            is_cnn=is_cnn,
            mode=norm_mode,
            transport=norm_transport,
            wifi_secrets=resolved_secrets,
            arena_bytes=arena_bytes,
            arena_pref_name=eff.arena_location,
            git_hash=git_hash,
        ),
        encoding="utf-8",
    )
    logger.info("Staged sketch %s", sketch_dir)
    return sketch_dir


def _parse_compile_sizes(
    family: str,
    builder_result: Mapping[str, Any],
    compiler_out: str,
    compiler_err: str,
) -> tuple[int | None, int | None, int | None, int | None]:
    if family in ("avr", "esp32"):
        sections = builder_result.get("executable_sections_size") or []
        flash_used = flash_max = ram_used = ram_max = None
        for sec in sections:
            if not isinstance(sec, dict):
                continue
            name = str(sec.get("name", "")).strip().lower()
            if name == "text":
                flash_used = int(sec["size"])
                flash_max = int(sec["max_size"]) if sec.get("max_size") is not None else None
            elif name == "data":
                ram_used = int(sec["size"])
                ram_max = int(sec["max_size"]) if sec.get("max_size") is not None else None
        return flash_used, flash_max, ram_used, ram_max

    if family == "esp8266":
        combined = f"{compiler_err}\n{compiler_out}"
        flash_m = _FLASH_RE.search(combined)
        ram_m = _RAM_RE.search(combined)
        flash_used = int(flash_m.group(1)) if flash_m else None
        flash_max = int(flash_m.group(2)) if flash_m else None
        ram_used = int(ram_m.group(1)) if ram_m else None
        ram_max = int(ram_m.group(2)) if ram_m else None
        rodata_m = _RODATA_RE.search(combined)
        iram_m = _IRAM_RE.search(combined)
        if rodata_m or iram_m:
            logger.debug(
                "ESP8266 extra sections: rodata=%s B, iram=%s/%s B",
                rodata_m.group(1) if rodata_m else "n/a",
                iram_m.group(1) if iram_m else "n/a",
                iram_m.group(2) if iram_m else "n/a",
            )
        return flash_used, flash_max, ram_used, ram_max

    raise ValueError(f"Unsupported board family: {family!r}")


def _evaluate_fit(
    board_spec: BoardSpec,
    arena_bytes: int,
    arena_pref: str,
    compile_ok: bool,
    flash_used: int | None,
    flash_max: int | None,
    ram_used: int | None,
    ram_max: int | None,
    compiler_err: str,
) -> tuple[bool, bool, bool, bool, str]:
    if not compile_ok:
        first_line = next((ln.strip() for ln in compiler_err.splitlines() if ln.strip()), "compile failed")
        return False, False, False, False, f"compile failed: {first_line[:160]}"

    flash_limit = (
        min(board_spec.flash_app_bytes, flash_max)
        if flash_max is not None
        else board_spec.flash_app_bytes
    )
    fits_flash = (flash_used is not None) and (flash_used <= flash_limit)

    if board_spec.family == "avr":
        ram_limit = board_spec.usable_sram_budget_bytes
    elif board_spec.family == "esp8266":
        ram_limit = (
            min(board_spec.physical_sram_bytes, ram_max)
            if ram_max is not None
            else board_spec.physical_sram_bytes
        )
    else:  # esp32
        ram_limit = board_spec.usable_sram_budget_bytes
    fits_ram = (ram_used is not None) and (ram_used <= ram_limit)

    if arena_bytes == 0:
        fits_arena = True
    elif arena_pref == "none":
        fits_arena = False
    elif arena_pref == "sram":
        fits_arena = arena_bytes <= board_spec.usable_sram_budget_bytes
    elif arena_pref == "psram":
        fits_arena = arena_bytes <= board_spec.psram_bytes
    else:  # auto
        fits_arena = (
            arena_bytes <= board_spec.usable_sram_budget_bytes
            or arena_bytes <= board_spec.psram_bytes
        )

    reasons: list[str] = []
    if not fits_flash:
        reasons.append(f"flash {flash_used} > {flash_limit} B")
    if not fits_ram:
        reasons.append(f"ram {ram_used} > {ram_limit} B")
    if not fits_arena:
        reasons.append(
            f"arena {arena_bytes} B exceeds sram={board_spec.usable_sram_budget_bytes} B "
            f"/ psram={board_spec.psram_bytes} B"
        )

    fits_budget = fits_flash and fits_ram and fits_arena
    return fits_flash, fits_ram, fits_arena, fits_budget, "; ".join(reasons) if reasons else "ok"


def _write_build_cache(cfg: HwTestConfig, result: BuildResult) -> Path:
    board_results = cfg.paths.results_dir / result.board_id
    board_results.mkdir(parents=True, exist_ok=True)
    cache_path = board_results / "build_cache.json"
    if cache_path.is_file():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        cache = {"board_id": result.board_id, "builds": {}}

    builds = cache.setdefault("builds", {})
    key = f"{result.model}|{result.mode}"
    builds[key] = {
        "board_id": result.board_id,
        "model": result.model,
        "mode": result.mode,
        "transport": result.transport,
        "fqbn": result.fqbn,
        "compile_ok": result.compile_ok,
        "flash_used_bytes": result.flash_used_bytes,
        "flash_max_bytes": result.flash_max_bytes,
        "ram_used_bytes": result.ram_used_bytes,
        "ram_max_bytes": result.ram_max_bytes,
        "arena_bytes": result.arena_bytes,
        "fits_flash": result.fits_flash,
        "fits_ram": result.fits_ram,
        "fits_arena": result.fits_arena,
        "fits_budget": result.fits_budget,
        "reason": result.reason,
        "core_id": result.core_id,
        "core_version": result.core_version,
        "cli_version": result.cli_version,
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return cache_path


def load_build_cache(results_dir: Path, board_id: str) -> dict[tuple[str, str], dict[str, Any]]:
    """Load persisted measured compiler sizes from `results/hw_test/<board>/build_cache.json`."""
    cache_path = Path(results_dir) / board_id / "build_cache.json"
    if not cache_path.is_file():
        return {}
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in (raw.get("builds") or {}).values():
        model = str(entry.get("model", "")).strip()
        mode = str(entry.get("mode", "")).strip().lower()
        if model and mode:
            out[(model, mode)] = entry
    return out


def compile_sketch(
    cfg: HwTestConfig,
    board: str,
    model: str,
    mode: str,
    transport: str = "uart",
    wifi_secrets: WifiSecrets | None = None,
    *,
    runner: SubprocessRunner = _default_runner,
) -> BuildResult:
    """Stage and compile one single-model sketch via `arduino-cli`, caching measured sizes."""
    board_spec, _ = _validate_target_combination(cfg, board, model, mode, transport)
    norm_mode = mode.strip().lower()
    norm_transport = transport.strip().lower()

    toolchain = ensure_core_installed(cfg, board_spec.fqbn, runner=runner)
    sketch_dir = stage_sketch(
        cfg,
        board=board,
        model=model,
        mode=norm_mode,
        transport=norm_transport,
        wifi_secrets=wifi_secrets,
        runner=runner,
    )
    build_dir = cfg.paths.staging_dir / "_arduino_build" / sketch_dir.name
    build_dir.mkdir(parents=True, exist_ok=True)

    f32_flag = 0 if norm_mode == "q15" else 1
    cmd = [
        cfg.arduino_cli.executable,
        "compile",
        "--fqbn",
        board_spec.fqbn,
        "--format",
        "json",
        "--build-path",
        str(build_dir),
        "--build-property",
        f"compiler.c.extra_flags=-DCIMP_ENABLE_F32={f32_flag}",
        "--build-property",
        f"compiler.cpp.extra_flags=-DCIMP_ENABLE_F32={f32_flag}",
        *cfg.arduino_cli.compile_extra_args,
        str(sketch_dir),
    ]
    logger.info("Compiling %s (%s, %s) via %s", sketch_dir.name, board_spec.fqbn, norm_mode, cfg.arduino_cli.executable)
    proc = runner(cmd, cwd=cfg.project_root, timeout=600)

    payload: dict[str, Any] = {}
    if proc.stdout.strip():
        try:
            payload = _parse_json_output(proc.stdout, f"{cfg.arduino_cli.executable} compile")
        except RuntimeError:
            payload = {}

    compile_ok = (proc.returncode == 0) and bool(payload.get("success", True))
    compiler_out = str(payload.get("compiler_out", proc.stdout))
    compiler_err = str(payload.get("compiler_err", proc.stderr))
    builder_result = payload.get("builder_result") or {}
    core_id = core_id_from_fqbn(board_spec.fqbn)
    core_version = (
        str((builder_result.get("board_platform") or {}).get("version"))
        if (builder_result.get("board_platform") or {}).get("version")
        else toolchain.core_versions.get(core_id)
    )

    flash_used, flash_max, ram_used, ram_max = _parse_compile_sizes(
        board_spec.family, builder_result, compiler_out, compiler_err
    )
    arena_bytes = get_model_arena_bytes(cfg, model, norm_mode)
    eff = board_spec.effective_settings(model)

    fits_flash, fits_ram, fits_arena, fits_budget, reason = _evaluate_fit(
        board_spec=board_spec,
        arena_bytes=arena_bytes,
        arena_pref=eff.arena_location,
        compile_ok=compile_ok,
        flash_used=flash_used,
        flash_max=flash_max,
        ram_used=ram_used,
        ram_max=ram_max,
        compiler_err=compiler_err,
    )

    result = BuildResult(
        board_id=board_spec.board_id,
        model=model,
        mode=norm_mode,
        transport=norm_transport,
        fqbn=board_spec.fqbn,
        sketch_dir=sketch_dir,
        build_dir=build_dir,
        compile_ok=compile_ok,
        flash_used_bytes=flash_used,
        flash_max_bytes=flash_max,
        ram_used_bytes=ram_used,
        ram_max_bytes=ram_max,
        arena_bytes=arena_bytes,
        fits_flash=fits_flash,
        fits_ram=fits_ram,
        fits_arena=fits_arena,
        fits_budget=fits_budget,
        reason=reason,
        core_id=core_id,
        core_version=core_version,
        cli_version=toolchain.cli_version,
        compiler_out=compiler_out,
        compiler_err=compiler_err,
    )
    _write_build_cache(cfg, result)
    return result


def upload_sketch(
    cfg: HwTestConfig,
    board: str,
    model: str,
    mode: str,
    port: str,
    transport: str = "uart",
    wifi_secrets: WifiSecrets | None = None,
    *,
    compile_first: bool = True,
    runner: SubprocessRunner = _default_runner,
) -> BuildResult | None:
    """Compile (by default) and upload one sketch to a connected board via `arduino-cli`."""
    if not port.strip():
        raise ValueError("Serial port must not be empty for upload")
    board_spec, _ = _validate_target_combination(cfg, board, model, mode, transport)
    norm_mode = mode.strip().lower()
    norm_transport = transport.strip().lower()

    build_res: BuildResult | None = None
    if compile_first:
        build_res = compile_sketch(
            cfg,
            board=board,
            model=model,
            mode=norm_mode,
            transport=norm_transport,
            wifi_secrets=wifi_secrets,
            runner=runner,
        )
        if not build_res.compile_ok:
            raise RuntimeError(
                f"Cannot upload ({board}, {model}, {norm_mode}): {build_res.reason}"
            )
        sketch_dir = build_res.sketch_dir
        build_dir = build_res.build_dir
    else:
        ensure_core_installed(cfg, board_spec.fqbn, runner=runner)
        sketch_name = f"{board_spec.board_id}_{model}_{norm_mode}_{norm_transport}"
        sketch_dir = cfg.paths.staging_dir / sketch_name
        build_dir = cfg.paths.staging_dir / "_arduino_build" / sketch_name
        if not sketch_dir.is_dir():
            raise FileNotFoundError(f"Staged sketch not found: {sketch_dir}")

    cmd = [
        cfg.arduino_cli.executable,
        "upload",
        "--fqbn",
        board_spec.fqbn,
        "--port",
        port,
        "--build-path",
        str(build_dir),
    ]
    if board_spec.upload_baud is not None:
        cmd.extend(["--upload-property", f"upload.speed={board_spec.upload_baud}"])
    cmd.append(str(sketch_dir))
    logger.info("Uploading %s to %s (%s)", sketch_dir.name, port, board_spec.fqbn)
    proc = runner(cmd, cwd=cfg.project_root, timeout=300)
    if proc.returncode != 0:
        detail = "\n".join(p for p in (proc.stdout, proc.stderr) if p).strip()
        raise RuntimeError(f"Upload failed on {port} ({board_spec.fqbn}): {detail}")
    return build_res
