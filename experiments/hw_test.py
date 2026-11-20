"""CLI entry point for the microcontroller hardware test bench (`PC -> MCU -> PC`).

Manual execution (PowerShell from `code/` directory):
    .\\.venv\\Scripts\\python.exe -m experiments.hw_test boards --config configs/hw_test/boards.toml
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable

import serial

from configs.logging import configure_logging, logger
from src.hw_test.config import BoardSpec, HwTestConfig, WifiSecrets, load_hw_test_config, load_wifi_secrets
from src.hw_test.firmware import (
    BuildResult,
    SubprocessRunner,
    compile_sketch,
    ensure_core_installed,
    get_git_short_hash,
    get_toolchain_versions,
    list_connected_boards,
    upload_sketch,
)
from src.hw_test.fit_report import generate_fit_report
from src.hw_test.images import GoldenLookup, prepare_images
from src.hw_test.report import write_run_report
from src.hw_test.session import run_batch
from src.hw_test.transport import SerialTransport, TcpTransport, Transport

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TransportFactory = Callable[[HwTestConfig, BoardSpec, str, str | None, str | None], Transport]


def _default_transport_factory(
    cfg: HwTestConfig,
    board_spec: BoardSpec,
    transport_name: str,
    port: str | None,
    host: str | None,
) -> Transport:
    if transport_name == "uart":
        if not port:
            raise ValueError("--port is required when --transport uart is selected")
        return SerialTransport(
            port=port,
            baudrate=board_spec.uart_baud,
            boot_wait_s=board_spec.boot_wait_s,
            reset_on_open=board_spec.serial_reset_on_open,
        )
    if transport_name == "wifi":
        if not host:
            raise ValueError("--host is required when --transport wifi is selected")
        if board_spec.tcp_port is None:
            raise ValueError(f"Board '{board_spec.board_id}' has no tcp_port configured")
        return TcpTransport(
            host=host,
            port=board_spec.tcp_port,
            connect_timeout_s=max(5.0, cfg.protocol.ack_timeout_s),
        )
    raise ValueError(f"Unsupported transport: {transport_name!r}")


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


def _add_common_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        dest="sub_config",
        type=Path,
        default=None,
        help="Path to hardware bench TOML config (default: configs/hw_test/boards.toml).",
    )
    parser.add_argument(
        "--dry-run",
        dest="sub_dry_run",
        action="store_true",
        default=False,
        help="Validate config, images, and toolchain without accessing hardware.",
    )


def _add_image_selector_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--image", type=str, default=None, help="Single image filename or path.")
    group.add_argument("--glob", dest="glob_pattern", type=str, default=None, help="Glob pattern inside images_dir.")
    group.add_argument("--all", dest="select_all", action="store_true", default=False, help="Select all images in images_dir.")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser supporting `boards`, `images`, `build`, `flash`, `send`, `bench`, `report`, `clean`."""
    parser = argparse.ArgumentParser(
        prog="python -m experiments.hw_test",
        description="Microcontroller hardware test bench (PC -> MCU -> PC) for x94 & CNN models.",
    )
    parser.add_argument(
        "--config",
        dest="top_config",
        type=Path,
        default=Path("configs/hw_test/boards.toml"),
        help="Path to hardware bench TOML config (default: configs/hw_test/boards.toml).",
    )
    parser.add_argument(
        "--dry-run",
        dest="top_dry_run",
        action="store_true",
        default=False,
        help="Validate inputs/config/toolchain without hardware access.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_boards = sub.add_parser("boards", help="List configured boards and connected serial ports.")
    _add_common_flags(p_boards)

    p_images = sub.add_parser("images", help="Validate and list images + golden parity availability.")
    _add_common_flags(p_images)
    _add_image_selector_flags(p_images)

    p_build = sub.add_parser("build", help="Stage and compile single-model firmware via arduino-cli.")
    _add_common_flags(p_build)
    p_build.add_argument("--board", required=True, help="Target board id (nano, uno, nodemcu, esp32cam).")
    p_build.add_argument("--model", required=True, help="Single model name to compile.")
    p_build.add_argument("--mode", default="q15", choices=["q15", "f32"], help="Numeric arithmetic mode.")
    p_build.add_argument("--transport", default="uart", choices=["uart", "wifi"], help="Firmware transport mode.")

    p_flash = sub.add_parser("flash", help="Compile and upload single-model firmware to a serial port.")
    _add_common_flags(p_flash)
    p_flash.add_argument("--board", required=True, help="Target board id.")
    p_flash.add_argument("--model", required=True, help="Single model name.")
    p_flash.add_argument("--mode", default="q15", choices=["q15", "f32"], help="Numeric arithmetic mode.")
    p_flash.add_argument("--transport", default="uart", choices=["uart", "wifi"], help="Firmware transport mode.")
    p_flash.add_argument("--port", default=None, help="Serial port (e.g. COM4).")
    p_flash.add_argument("--no-compile", action="store_true", default=False, help="Upload previously built binary.")

    p_send = sub.add_parser("send", help="Send validated images to flashed MCU and write run report.")
    _add_common_flags(p_send)
    p_send.add_argument("--board", required=True, help="Target board id.")
    p_send.add_argument("--model", required=True, help="Model expected on flashed MCU.")
    p_send.add_argument("--mode", default="q15", choices=["q15", "f32"], help="Numeric arithmetic mode.")
    p_send.add_argument("--transport", required=True, choices=["uart", "wifi"], help="Wire transport.")
    p_send.add_argument("--port", default=None, help="Serial port when --transport uart.")
    p_send.add_argument("--host", default=None, help="Device IP/hostname when --transport wifi.")
    _add_image_selector_flags(p_send)
    p_send.add_argument("--repeats", type=int, default=None, help="Override timed_iters.")
    p_send.add_argument("--warmup", type=int, default=None, help="Override warmup_iters.")
    p_send.add_argument("--dump-descriptor", action="store_true", default=False, help="Request RESULT_X94 after each tabular run.")

    p_bench = sub.add_parser("bench", help="Compile all board models unattended; flash & measure selected models.")
    _add_common_flags(p_bench)
    p_bench.add_argument("--board", required=True, help="Target board id.")
    p_bench.add_argument("--transport", default="uart", choices=["uart", "wifi"], help="Wire transport.")
    p_bench.add_argument("--port", default=None, help="Serial port (required for upload / uart runs).")
    p_bench.add_argument("--host", default=None, help="Device IP/hostname when --transport wifi.")
    p_bench.add_argument("--models", nargs="+", default=None, help="Optional subset of models to flash & run.")
    p_bench.add_argument("--modes", nargs="+", default=None, choices=["q15", "f32"], help="Optional numeric modes filter.")
    _add_image_selector_flags(p_bench)
    p_bench.add_argument("--repeats", type=int, default=None, help="Override timed_iters.")
    p_bench.add_argument("--warmup", type=int, default=None, help="Override warmup_iters.")
    p_bench.add_argument("--dump-descriptor", action="store_true", default=False, help="Fetch x94 descriptor on tabular models.")
    p_bench.add_argument("--compile-only", action="store_true", default=False, help="Compile all models and skip flashing/running.")
    p_bench.add_argument("--no-prompt", action="store_true", default=False, help="Do not pause before ESP32-CAM uploads.")

    p_report = sub.add_parser("report", help="Generate fit_report.{csv,md} from cached builds and run summaries.")
    _add_common_flags(p_report)

    p_clean = sub.add_parser(
        "clean",
        help="Remove generated staging build folders and/or results runs & reports (preserving README.md).",
    )
    _add_common_flags(p_clean)
    p_clean.add_argument("--board", default=None, help="Optional board id to clean exclusively.")
    clean_scope = p_clean.add_mutually_exclusive_group()
    clean_scope.add_argument(
        "--staging-only",
        action="store_true",
        default=False,
        help="Clean only hw_test/build/ staging directories.",
    )
    clean_scope.add_argument(
        "--results-only",
        action="store_true",
        default=False,
        help="Clean only results/hw_test/ run folders and fit_report.{csv,md}.",
    )
    p_clean.add_argument(
        "--keep-build-cache",
        action="store_true",
        default=False,
        help="Preserve results/hw_test/<board>/build_cache.json when cleaning results.",
    )

    return parser


def _resolve_images(cfg: HwTestConfig, args: argparse.Namespace, *, default_all: bool = False):
    has_sel = any(
        [args.image is not None, args.glob_pattern is not None, bool(args.select_all)]
    )
    select_all = bool(args.select_all) or (default_all and not has_sel)
    return prepare_images(
        cfg.paths.images_dir,
        cfg.images,
        image=args.image,
        glob_pattern=args.glob_pattern,
        select_all=select_all,
    )


def _parse_models_arg(raw_models: list[str] | None, board_spec: BoardSpec) -> tuple[str, ...]:
    if not raw_models:
        return board_spec.models
    out: list[str] = []
    for item in raw_models:
        for tok in item.split(","):
            name = tok.strip()
            if name:
                if name not in board_spec.models:
                    raise ValueError(
                        f"Model '{name}' is not configured on board '{board_spec.board_id}'. "
                        f"Available: {board_spec.models}"
                    )
                if name not in out:
                    out.append(name)
    if not out:
        raise ValueError("--models list resolved to empty")
    return tuple(out)


def _cmd_boards(cfg: HwTestConfig, dry_run: bool, runner: SubprocessRunner) -> int:
    print("Configured boards:")
    for bid, b in cfg.boards.items():
        print(
            f"  - {bid}: fqbn={b.fqbn} family={b.family} transports={','.join(b.transports)} "
            f"models={','.join(b.models)}"
        )
    if dry_run:
        logger.info("boards --dry-run: skipped arduino-cli board list")
        return 0
    connected = list_connected_boards(cfg, runner=runner)
    print("Connected ports:")
    if not connected:
        print("  (none detected)")
    for c in connected:
        fqbn_txt = c.fqbn if c.fqbn else "unknown"
        print(f"  - {c.port}: board={c.board_name} fqbn={fqbn_txt} ({c.protocol_label})")
    return 0


def _cmd_images(cfg: HwTestConfig, args: argparse.Namespace) -> int:
    imgs = _resolve_images(cfg, args, default_all=True)
    golden = GoldenLookup.from_config(cfg)
    print(f"Validated {len(imgs)} image(s) from {cfg.paths.images_dir}:")
    for im in imgs:
        entry = golden.lookup_manifest(im)
        manifest_tag = f"manifest_idx={entry.index} label={entry.label}" if entry else "manifest=n/a"
        print(f"  - [{im.image_id}] {im.name} ({im.width}x{im.height} {im.mode}) crc32=0x{im.crc32:08X} {manifest_tag}")
    return 0


def _cmd_build(
    cfg: HwTestConfig,
    args: argparse.Namespace,
    dry_run: bool,
    runner: SubprocessRunner,
) -> int:
    board_spec = cfg.get_board(args.board)
    eff = board_spec.effective_settings(args.model)
    if args.mode not in eff.numeric_modes:
        raise ValueError(f"Mode '{args.mode}' not enabled for ({args.board}, {args.model}): {eff.numeric_modes}")
    if args.transport not in board_spec.transports:
        raise ValueError(f"Transport '{args.transport}' not enabled for {args.board}: {board_spec.transports}")

    secrets: WifiSecrets | None = None
    if args.transport == "wifi":
        secrets = load_wifi_secrets(cfg.secrets_config_path)

    if dry_run:
        ensure_core_installed(cfg, board_spec.fqbn, runner=runner)
        logger.info(
            "build --dry-run OK: board=%s model=%s mode=%s transport=%s fqbn=%s",
            args.board,
            args.model,
            args.mode,
            args.transport,
            board_spec.fqbn,
        )
        return 0

    res = compile_sketch(
        cfg,
        board=args.board,
        model=args.model,
        mode=args.mode,
        transport=args.transport,
        wifi_secrets=secrets,
        runner=runner,
    )
    print(
        f"Build {res.board_id}/{res.model}/{res.mode}: compile_ok={res.compile_ok} "
        f"flash={res.flash_used_bytes}/{res.flash_max_bytes} B "
        f"ram={res.ram_used_bytes}/{res.ram_max_bytes} B "
        f"arena={res.arena_bytes} B fits_budget={res.fits_budget} ({res.reason})"
    )
    if not res.compile_ok or not res.fits_budget:
        raise RuntimeError(f"Build failed or exceeded budget: {res.reason}")
    return 0


def _cmd_flash(
    cfg: HwTestConfig,
    args: argparse.Namespace,
    dry_run: bool,
    runner: SubprocessRunner,
) -> int:
    board_spec = cfg.get_board(args.board)
    eff = board_spec.effective_settings(args.model)
    if args.mode not in eff.numeric_modes:
        raise ValueError(f"Mode '{args.mode}' not enabled for ({args.board}, {args.model})")
    if args.transport not in board_spec.transports:
        raise ValueError(f"Transport '{args.transport}' not enabled for {args.board}")
    secrets: WifiSecrets | None = None
    if args.transport == "wifi":
        secrets = load_wifi_secrets(cfg.secrets_config_path)

    if dry_run:
        ensure_core_installed(cfg, board_spec.fqbn, runner=runner)
        logger.info("flash --dry-run OK: board=%s model=%s mode=%s", args.board, args.model, args.mode)
        return 0

    if not args.port:
        raise ValueError("--port is required for flash")
    upload_sketch(
        cfg,
        board=args.board,
        model=args.model,
        mode=args.mode,
        port=args.port,
        transport=args.transport,
        wifi_secrets=secrets,
        compile_first=not args.no_compile,
        runner=runner,
    )
    print(f"Flashed {args.board}/{args.model}/{args.mode} ({args.transport}) to {args.port}")
    return 0


def _cmd_send(
    cfg: HwTestConfig,
    args: argparse.Namespace,
    dry_run: bool,
    transport_factory: TransportFactory,
    runner: SubprocessRunner,
) -> int:
    board_spec = cfg.get_board(args.board)
    eff = board_spec.effective_settings(args.model)
    if args.mode not in eff.numeric_modes:
        raise ValueError(f"Mode '{args.mode}' not enabled for ({args.board}, {args.model}): {eff.numeric_modes}")
    if args.transport not in board_spec.transports:
        raise ValueError(f"Transport '{args.transport}' not enabled for {args.board}: {board_spec.transports}")

    images = _resolve_images(cfg, args, default_all=False)
    golden = GoldenLookup.from_config(cfg)

    if dry_run:
        logger.info(
            "send --dry-run OK: board=%s model=%s mode=%s transport=%s n_images=%d",
            args.board,
            args.model,
            args.mode,
            args.transport,
            len(images),
        )
        return 0

    port_or_host = str(args.port if args.transport == "uart" else args.host or "")
    transport = transport_factory(cfg, board_spec, args.transport, args.port, args.host)

    try:
        toolchain = get_toolchain_versions(cfg, runner=runner)
    except RuntimeError:
        toolchain = None
    git_hash = get_git_short_hash(cfg.project_root, runner=runner)

    with transport:
        batch = run_batch(
            transport,
            cfg,
            board=args.board,
            images=images,
            model=args.model,
            mode=args.mode,
            repeats=args.repeats,
            warmup=args.warmup,
            dump_descriptor=args.dump_descriptor,
            golden_lookup=golden,
            transport_name=args.transport,
            port_or_host=port_or_host,
        )
    out = write_run_report(cfg, batch, toolchain=toolchain, host_git_hash=git_hash)
    n_pass = sum(1 for r in batch.records if r.parity_status == "pass")
    n_matched = sum(1 for r in batch.records if r.parity_status in ("pass", "fail"))
    print(f"Run complete: {out.run_dir} (images={len(batch.records)}, parity={n_pass}/{n_matched})")
    return 0


def _cmd_bench(
    cfg: HwTestConfig,
    args: argparse.Namespace,
    dry_run: bool,
    transport_factory: TransportFactory,
    runner: SubprocessRunner,
    input_fn: Callable[[str], str],
) -> int:
    board_spec = cfg.get_board(args.board)
    if args.transport not in board_spec.transports:
        raise ValueError(f"Transport '{args.transport}' not supported on {args.board}: {board_spec.transports}")

    selected_models = _parse_models_arg(args.models, board_spec)
    modes_filter = set(args.modes) if args.modes else None
    images = _resolve_images(cfg, args, default_all=True)

    secrets: WifiSecrets | None = None
    if args.transport == "wifi":
        secrets = load_wifi_secrets(cfg.secrets_config_path)

    if dry_run:
        ensure_core_installed(cfg, board_spec.fqbn, runner=runner)
        logger.info(
            "bench --dry-run OK: board=%s transport=%s models=%s images=%d",
            args.board,
            args.transport,
            selected_models,
            len(images),
        )
        return 0

    if not args.compile_only:
        if not args.port:
            raise ValueError("--port is required to flash during bench (or pass --compile-only)")
        if args.transport == "wifi" and not args.host:
            raise ValueError("--host is required when running bench over --transport wifi")

    # 1. Compile all board models unattended so fit_report captures full compiler measurements
    compile_results: dict[tuple[str, str], BuildResult] = {}
    for model in board_spec.models:
        eff = board_spec.effective_settings(model)
        for mode in eff.numeric_modes:
            if modes_filter is not None and mode not in modes_filter:
                continue
            res = compile_sketch(
                cfg,
                board=board_spec.board_id,
                model=model,
                mode=mode,
                transport=args.transport,
                wifi_secrets=secrets,
                runner=runner,
            )
            compile_results[(model, mode)] = res
            logger.info(
                "Bench build %s/%s/%s: compile_ok=%s fits=%s (%s)",
                board_spec.board_id,
                model,
                mode,
                res.compile_ok,
                res.fits_budget,
                res.reason,
            )

    # 2. Flash and execute selected fitting models
    if not args.compile_only:
        golden = GoldenLookup.from_config(cfg)
        toolchain = get_toolchain_versions(cfg, runner=runner)
        git_hash = get_git_short_hash(cfg.project_root, runner=runner)
        port_or_host = str(args.port if args.transport == "uart" else args.host or "")

        for model in selected_models:
            eff = board_spec.effective_settings(model)
            for mode in eff.numeric_modes:
                if modes_filter is not None and mode not in modes_filter:
                    continue
                bres = compile_results[(model, mode)]
                if not bres.compile_ok or not bres.fits_budget:
                    logger.warning(
                        "Skipping %s/%s/%s: does not fit (%s)",
                        board_spec.board_id,
                        model,
                        mode,
                        bres.reason,
                    )
                    continue

                if board_spec.family == "esp32" and not args.no_prompt:
                    ans = input_fn(
                        f"Put {board_spec.board_id} in flash mode (GPIO0->GND + reset) for "
                        f"{model}/{mode} and press Enter (or 's' to skip): "
                    ).strip().lower()
                    if ans == "s":
                        logger.info("Skipped %s/%s at user prompt", model, mode)
                        continue

                upload_sketch(
                    cfg,
                    board=board_spec.board_id,
                    model=model,
                    mode=mode,
                    port=args.port,
                    transport=args.transport,
                    wifi_secrets=secrets,
                    compile_first=False,
                    runner=runner,
                )
                transport = transport_factory(cfg, board_spec, args.transport, args.port, args.host)
                with transport:
                    batch = run_batch(
                        transport,
                        cfg,
                        board=board_spec.board_id,
                        images=images,
                        model=model,
                        mode=mode,
                        repeats=args.repeats,
                        warmup=args.warmup,
                        dump_descriptor=args.dump_descriptor,
                        golden_lookup=golden,
                        transport_name=args.transport,
                        port_or_host=port_or_host,
                    )
                write_run_report(cfg, batch, toolchain=toolchain, host_git_hash=git_hash)

    fit_paths = generate_fit_report(cfg)
    print(f"Bench complete: fit report written to {fit_paths.md_path}")
    return 0


def _cmd_report(cfg: HwTestConfig, dry_run: bool) -> int:
    if dry_run:
        logger.info("report --dry-run OK: output dir %s", cfg.paths.results_dir)
        return 0
    fit_paths = generate_fit_report(cfg)
    print(f"Fit report written: {fit_paths.csv_path} and {fit_paths.md_path} ({len(fit_paths.rows)} rows)")
    return 0


def _collect_clean_targets(cfg: HwTestConfig, args: argparse.Namespace) -> list[Path]:
    if args.board is not None:
        cfg.get_board(args.board)

    targets: list[Path] = []
    protected_names = {"README.md", ".gitkeep"}

    if not args.results_only and cfg.paths.staging_dir.is_dir():
        if args.board is not None:
            prefix = f"{args.board}_"
            targets.extend(
                sorted(
                    p
                    for p in cfg.paths.staging_dir.iterdir()
                    if p.name not in protected_names and p.name.startswith(prefix)
                )
            )
            arduino_bld = cfg.paths.staging_dir / "_arduino_build"
            if arduino_bld.is_dir():
                targets.extend(
                    sorted(p for p in arduino_bld.iterdir() if p.name.startswith(prefix))
                )
        else:
            targets.extend(
                sorted(p for p in cfg.paths.staging_dir.iterdir() if p.name not in protected_names)
            )

    if not args.staging_only and cfg.paths.results_dir.is_dir():
        if args.board is not None:
            bdir = cfg.paths.results_dir / args.board
            if bdir.is_dir():
                if args.keep_build_cache:
                    targets.extend(
                        sorted(
                            c
                            for c in bdir.iterdir()
                            if c.name not in protected_names and c.name != "build_cache.json"
                        )
                    )
                else:
                    targets.append(bdir)
        else:
            for p in sorted(cfg.paths.results_dir.iterdir()):
                if p.name in protected_names:
                    continue
                if p.is_dir() and args.keep_build_cache:
                    targets.extend(
                        sorted(
                            c
                            for c in p.iterdir()
                            if c.name not in protected_names and c.name != "build_cache.json"
                        )
                    )
                else:
                    targets.append(p)

    return targets


def _cmd_clean(cfg: HwTestConfig, args: argparse.Namespace, dry_run: bool) -> int:
    targets = _collect_clean_targets(cfg, args)
    if dry_run:
        logger.info("clean --dry-run OK: %d target(s) would be removed", len(targets))
        print(f"Clean --dry-run ({len(targets)} target(s)):")
        for t in targets:
            print(f"  - {t}")
        return 0

    for t in targets:
        if t.is_dir():
            shutil.rmtree(t)
        elif t.exists():
            t.unlink()
    logger.info("Cleaned %d hardware test artifact target(s)", len(targets))
    print(f"Cleaned {len(targets)} target(s).")
    return 0


def main(
    argv: list[str] | None = None,
    *,
    project_root: Path = PROJECT_ROOT,
    transport_factory: TransportFactory = _default_transport_factory,
    runner: SubprocessRunner = _default_runner,
    input_fn: Callable[[str], str] = input,
) -> int:
    """CLI entry point for `experiments.hw_test`."""
    configure_logging()
    args = build_parser().parse_args(argv)
    config_path = args.sub_config if args.sub_config is not None else args.top_config
    dry_run = bool(args.top_dry_run or args.sub_dry_run)

    try:
        cfg = load_hw_test_config(config_path=config_path, project_root=project_root)
        cmd = args.command
        if cmd == "boards":
            return _cmd_boards(cfg, dry_run, runner)
        if cmd == "images":
            return _cmd_images(cfg, args)
        if cmd == "build":
            return _cmd_build(cfg, args, dry_run, runner)
        if cmd == "flash":
            return _cmd_flash(cfg, args, dry_run, runner)
        if cmd == "send":
            return _cmd_send(cfg, args, dry_run, transport_factory, runner)
        if cmd == "bench":
            return _cmd_bench(cfg, args, dry_run, transport_factory, runner, input_fn)
        if cmd == "report":
            return _cmd_report(cfg, dry_run)
        if cmd == "clean":
            return _cmd_clean(cfg, args, dry_run)
        raise ValueError(f"Unsupported command: {cmd}")
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        serial.SerialException,
        TimeoutError,
    ) as exc:
        logger.error("hw_test failed: %s", exc)
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
