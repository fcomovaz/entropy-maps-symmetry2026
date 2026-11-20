"""Per-run hardware test result serialization (`run_meta.json`, `per_image.csv`, `summary.json`)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from configs.logging import logger
from src.hw_test.config import HwTestConfig
from src.hw_test.firmware import ToolchainInfo, core_id_from_fqbn, get_git_short_hash
from src.hw_test.protocol import UNAVAILABLE_U32
from src.hw_test.session import BatchRunResult, ImageRunRecord

PER_IMAGE_COLUMNS: tuple[str, ...] = (
    "image",
    "image_id",
    "true_label",
    "label",
    "logit0",
    "logit1",
    "margin",
    "logit0_raw",
    "logit1_raw",
    "frac_bits",
    "golden_label",
    "golden_logit0",
    "golden_logit1",
    "golden_logit0_raw",
    "golden_logit1_raw",
    "parity_status",
    "parity_ok",
    "x94_crc_ok",
    "first_selector_us",
    "first_model_us",
    "first_total_us",
    "min_selector_us",
    "min_model_us",
    "min_total_us",
    "median_selector_us",
    "median_model_us",
    "median_total_us",
    "mean_selector_us",
    "mean_model_us",
    "mean_total_us",
    "max_selector_us",
    "max_model_us",
    "max_total_us",
    "T_transfer_host_ms",
    "includes_compute",
    "free_heap_min",
    "stack_hwm",
    "free_psram_min",
    "arena_bytes",
    "arena_location",
)


@dataclass(frozen=True, slots=True)
class RunReportPaths:
    """Output paths produced for one hardware run."""

    run_id: str
    run_dir: Path
    run_meta_path: Path
    per_image_csv_path: Path
    summary_json_path: Path
    x94_dump_path: Path | None = None


def make_run_id(
    model: str,
    mode: str,
    transport: str,
    *,
    now: datetime | None = None,
) -> str:
    """Format Windows-safe UTC run identifier: `YYYYMMDDTHHMMSSZ_<model>_<mode>_<transport>`."""
    ts = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{model}_{mode}_{transport}"


def _opt_probe(value: int) -> int | None:
    return None if value == UNAVAILABLE_U32 else int(value)


def _fmt_opt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def _record_to_csv_row(rec: ImageRunRecord) -> dict[str, str]:
    res = rec.result
    st = res.stats
    logit0 = rec.logits_float[0] if len(rec.logits_float) >= 1 else None
    logit1 = rec.logits_float[1] if (rec.is_cnn and len(rec.logits_float) >= 2) else None

    golden_label: int | None = None
    g_l0: float | None = None
    g_l1: float | None = None
    g_l0_raw: int | None = None
    g_l1_raw: int | None = None
    true_label: int | None = None
    if rec.golden is not None:
        true_label = rec.golden.true_label
        if rec.mode == "q15":
            golden_label = rec.golden.expected_q15_label
            g_l0_raw = rec.golden.expected_q15_logits_raw[0]
            scale = float(1 << res.frac_bits) if res.frac_bits > 0 else 1.0
            g_l0 = float(g_l0_raw) / scale
            if rec.is_cnn and len(rec.golden.expected_q15_logits_raw) >= 2:
                g_l1_raw = rec.golden.expected_q15_logits_raw[1]
                g_l1 = float(g_l1_raw) / scale
        else:
            golden_label = rec.golden.expected_f32_label
            g_l0 = rec.golden.expected_f32_logits[0]
            if rec.is_cnn and len(rec.golden.expected_f32_logits) >= 2:
                g_l1 = rec.golden.expected_f32_logits[1]

    return {
        "image": rec.image.name,
        "image_id": str(rec.image.image_id),
        "true_label": _fmt_opt(true_label),
        "label": str(res.label),
        "logit0": _fmt_opt(logit0),
        "logit1": _fmt_opt(logit1),
        "margin": _fmt_opt(rec.margin_float if rec.is_cnn else None),
        "logit0_raw": str(res.logit_raw0) if rec.mode == "q15" else "",
        "logit1_raw": str(res.logit_raw1) if (rec.is_cnn and rec.mode == "q15") else "",
        "frac_bits": str(res.frac_bits),
        "golden_label": _fmt_opt(golden_label),
        "golden_logit0": _fmt_opt(g_l0),
        "golden_logit1": _fmt_opt(g_l1),
        "golden_logit0_raw": _fmt_opt(g_l0_raw),
        "golden_logit1_raw": _fmt_opt(g_l1_raw),
        "parity_status": rec.parity_status,
        "parity_ok": _fmt_opt(rec.parity_ok),
        "x94_crc_ok": _fmt_opt(rec.x94_crc_ok),
        "first_selector_us": str(st.first.selector_us),
        "first_model_us": str(st.first.model_us),
        "first_total_us": str(st.first.total_us),
        "min_selector_us": str(st.minimum.selector_us),
        "min_model_us": str(st.minimum.model_us),
        "min_total_us": str(st.minimum.total_us),
        "median_selector_us": str(st.median.selector_us),
        "median_model_us": str(st.median.model_us),
        "median_total_us": str(st.median.total_us),
        "mean_selector_us": str(st.mean.selector_us),
        "mean_model_us": str(st.mean.model_us),
        "mean_total_us": str(st.mean.total_us),
        "max_selector_us": str(st.maximum.selector_us),
        "max_model_us": str(st.maximum.model_us),
        "max_total_us": str(st.maximum.total_us),
        "T_transfer_host_ms": f"{rec.t_transfer_host_ms:.6f}",
        "includes_compute": str(rec.includes_compute),
        "free_heap_min": _fmt_opt(_opt_probe(res.free_heap_min)),
        "stack_hwm": _fmt_opt(_opt_probe(res.stack_hwm)),
        "free_psram_min": _fmt_opt(_opt_probe(res.free_psram_min)),
        "arena_bytes": str(res.arena_bytes),
        "arena_location": res.arena_location_name,
    }


def _min_available(values: list[int | None]) -> int | None:
    valid = [v for v in values if v is not None]
    return min(valid) if valid else None


def write_run_report(
    cfg: HwTestConfig,
    batch: BatchRunResult,
    *,
    toolchain: ToolchainInfo | None = None,
    host_git_hash: str | None = None,
    run_id: str | None = None,
) -> RunReportPaths:
    """Persist `run_meta.json`, `per_image.csv`, and `summary.json` for a completed batch."""
    if not batch.records:
        raise ValueError("Cannot write run report for empty BatchRunResult")

    board_spec = cfg.get_board(batch.board_id)
    eff = board_spec.effective_settings(batch.model)
    board_dir = cfg.paths.results_dir / batch.board_id
    board_dir.mkdir(parents=True, exist_ok=True)

    base_id = run_id or make_run_id(batch.model, batch.mode, batch.transport_name)
    resolved_id = base_id
    suffix = 1
    while (board_dir / resolved_id).exists():
        resolved_id = f"{base_id}_{suffix:02d}"
        suffix += 1

    run_dir = board_dir / resolved_id
    run_dir.mkdir(parents=True, exist_ok=True)

    core_id = core_id_from_fqbn(board_spec.fqbn)
    core_version = toolchain.core_versions.get(core_id) if toolchain is not None else None
    cli_version = toolchain.cli_version if toolchain is not None else None
    git_hash = host_git_hash if host_git_hash is not None else get_git_short_hash(cfg.project_root)

    first_res = batch.records[0].result
    timer_res_us = first_res.timer_resolution_us
    avr_bound_us = (
        timer_res_us * cfg.images.height if board_spec.streaming else 0
    )

    run_meta = {
        "run_id": resolved_id,
        "started_at_utc": batch.started_at_utc,
        "finished_at_utc": batch.finished_at_utc,
        "board_id": batch.board_id,
        "model": batch.model,
        "mode": batch.mode,
        "transport": batch.transport_name,
        "port_or_host": batch.port_or_host,
        "host_git_hash": git_hash,
        "device_git_hash": batch.info.git_hash,
        "fqbn": board_spec.fqbn,
        "core_id": core_id,
        "core_version": core_version,
        "arduino_cli_version": cli_version,
        "timer_resolution_us": timer_res_us,
        "avr_quantization_bound_us": avr_bound_us if board_spec.streaming else None,
        "includes_compute": board_spec.streaming,
        "effective_settings": {
            "warmup_iters_config": eff.warmup_iters,
            "timed_iters_config": eff.timed_iters,
            "warmup_iters_run": batch.warmup_requested,
            "timed_iters_run": batch.repeats_requested,
            "result_timeout_s": eff.result_timeout_s,
            "arena_location": eff.arena_location,
        },
        "device_info": {
            "board_id": batch.info.board_id,
            "model": batch.info.model,
            "mode": batch.info.mode,
            "protocol_version": batch.info.protocol_version,
            "width": batch.info.width,
            "height": batch.info.height,
            "streaming": batch.info.streaming,
            "max_payload": batch.info.max_payload,
            "free_ram_boot": batch.info.free_ram_boot,
            "psram_size": batch.info.psram_size,
            "git_hash": batch.info.git_hash,
        },
        "config_snapshot": {
            "physical_sram_bytes": board_spec.physical_sram_bytes,
            "usable_sram_budget_bytes": board_spec.usable_sram_budget_bytes,
            "flash_app_bytes": board_spec.flash_app_bytes,
            "psram_bytes": board_spec.psram_bytes,
            "cpu_hz": board_spec.cpu_hz,
            "band_rows": board_spec.band_rows,
            "max_payload_bytes": board_spec.max_payload_bytes,
            "ack_timeout_s": cfg.protocol.ack_timeout_s,
            "max_retries": cfg.protocol.max_retries,
        },
    }
    run_meta_path = run_dir / "run_meta.json"
    run_meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    per_image_csv_path = run_dir / "per_image.csv"
    with per_image_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(PER_IMAGE_COLUMNS))
        writer.writeheader()
        for rec in batch.records:
            writer.writerow(_record_to_csv_row(rec))

    n_pass = sum(1 for r in batch.records if r.parity_status == "pass")
    n_fail = sum(1 for r in batch.records if r.parity_status == "fail")
    n_na = sum(1 for r in batch.records if r.parity_status == "n/a")
    n_matched = n_pass + n_fail
    parity_rate = (float(n_pass) / float(n_matched)) if n_matched > 0 else None

    diffs = [r.logit_max_abs_diff for r in batch.records if r.logit_max_abs_diff is not None]
    max_logit_abs_diff = max(diffs) if diffs else None

    x94_checks = [r.x94_crc_ok for r in batch.records if r.x94_crc_ok is not None]
    x94_pass = sum(1 for x in x94_checks if x)

    med_sel = [float(r.result.stats.median.selector_us) for r in batch.records]
    med_mod = [float(r.result.stats.median.model_us) for r in batch.records]
    med_tot = [float(r.result.stats.median.total_us) for r in batch.records]
    first_tot = [float(r.result.stats.first.total_us) for r in batch.records]
    min_tot = [float(r.result.stats.minimum.total_us) for r in batch.records]
    max_tot = [float(r.result.stats.maximum.total_us) for r in batch.records]
    tx_ms = [float(r.t_transfer_host_ms) for r in batch.records]

    agg_med_sel_us = float(statistics.median(med_sel))
    agg_med_mod_us = float(statistics.median(med_mod))
    agg_med_tot_us = float(statistics.median(med_tot))
    agg_mean_tot_us = float(statistics.mean(float(r.result.stats.mean.total_us) for r in batch.records))
    agg_min_tot_us = float(min(min_tot))
    agg_max_tot_us = float(max(max_tot))
    agg_first_tot_us = float(statistics.median(first_tot))

    heap_mins = [_opt_probe(r.result.free_heap_min) for r in batch.records]
    stack_hwms = [_opt_probe(r.result.stack_hwm) for r in batch.records]
    psram_mins = [_opt_probe(r.result.free_psram_min) for r in batch.records]

    summary = {
        "run_id": resolved_id,
        "board_id": batch.board_id,
        "model": batch.model,
        "mode": batch.mode,
        "transport": batch.transport_name,
        "n_images": len(batch.records),
        "n_golden_matched": n_matched,
        "parity": {
            "n_pass": n_pass,
            "n_fail": n_fail,
            "n_na": n_na,
            "parity_rate": parity_rate,
            "max_logit_abs_diff": max_logit_abs_diff,
            "x94_crc_pass": x94_pass,
            "x94_crc_checked": len(x94_checks),
        },
        "timing_us": {
            "median_selector_us": agg_med_sel_us,
            "median_model_us": agg_med_mod_us,
            "median_total_us": agg_med_tot_us,
            "mean_total_us": agg_mean_tot_us,
            "min_total_us": agg_min_tot_us,
            "max_total_us": agg_max_tot_us,
            "first_total_us": agg_first_tot_us,
        },
        "timing_ms": {
            "median_selector_ms": agg_med_sel_us / 1000.0,
            "median_model_ms": agg_med_mod_us / 1000.0,
            "median_total_ms": agg_med_tot_us / 1000.0,
            "mean_total_ms": agg_mean_tot_us / 1000.0,
            "min_total_ms": agg_min_tot_us / 1000.0,
            "max_total_ms": agg_max_tot_us / 1000.0,
            "first_total_ms": agg_first_tot_us / 1000.0,
        },
        "transfer_host_ms": {
            "median_ms": float(statistics.median(tx_ms)),
            "mean_ms": float(statistics.mean(tx_ms)),
            "includes_compute": board_spec.streaming,
        },
        "timer_resolution_us": timer_res_us,
        "avr_quantization_bound_us": avr_bound_us,
        "memory": {
            "free_ram_boot": batch.info.free_ram_boot,
            "free_heap_min_overall": _min_available(heap_mins),
            "stack_hwm_min_overall": _min_available(stack_hwms),
            "free_psram_min_overall": _min_available(psram_mins),
            "arena_bytes": first_res.arena_bytes,
            "arena_location": first_res.arena_location_name,
        },
    }
    summary_json_path = run_dir / "summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    x94_dump_path: Path | None = None
    dumped_entries = [
        {
            "image": r.image.name,
            "image_id": r.image.image_id,
            "mode": r.dumped_x94.mode,
            "x94_crc32": r.result.x94_crc32,
            "values": list(r.dumped_x94.values),
        }
        for r in batch.records
        if r.dumped_x94 is not None
    ]
    if dumped_entries:
        x94_dump_path = run_dir / "x94_dumped.json"
        x94_dump_path.write_text(json.dumps(dumped_entries, indent=2), encoding="utf-8")

    logger.info(
        "Wrote hw_test run report to %s (n=%d, parity=%d/%d, median_total=%.3f ms)",
        run_dir,
        len(batch.records),
        n_pass,
        n_matched,
        agg_med_tot_us / 1000.0,
    )
    return RunReportPaths(
        run_id=resolved_id,
        run_dir=run_dir,
        run_meta_path=run_meta_path,
        per_image_csv_path=per_image_csv_path,
        summary_json_path=summary_json_path,
        x94_dump_path=x94_dump_path,
    )
