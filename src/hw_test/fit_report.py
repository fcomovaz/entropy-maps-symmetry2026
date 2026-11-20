"""Cross-board hardware fit, memory, latency, and offline-accuracy Pareto verdict report."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from configs.logging import logger
from src.hw_test.config import BoardSpec, HwTestConfig
from src.hw_test.firmware import load_build_cache

VERDICT_DOES_NOT_FIT = "does_not_fit"
VERDICT_NOT_MEASURED = "not_measured"
VERDICT_FITS_EXCEEDS_LATENCY = "fits_exceeds_latency"
VERDICT_FITS_DOMINATED = "fits_dominated"
VERDICT_FITS_VIABLE = "fits_viable"


@dataclass(frozen=True, slots=True)
class OfflineModelMetrics:
    """Offline desktop classification metrics loaded from `results/`."""

    roc_auc: float | None
    f1: float | None
    source: str


@dataclass(frozen=True, slots=True)
class PredictedMemory:
    """Predicted flash weights and activation arena size from `c_imp` reports."""

    weight_flash_bytes: int | None
    arena_bytes: int


@dataclass(frozen=True, slots=True)
class FitRow:
    """One `(board, model, mode)` entry in `fit_report.{csv,md}`."""

    board_id: str
    model: str
    mode: str
    fit_possible: bool
    verdict: str
    reason: str
    measured_transport: str | None
    pred_weight_flash_bytes: int | None
    pred_arena_bytes: int
    compile_ok: bool | None
    flash_used_bytes: int | None
    flash_max_bytes: int | None
    ram_used_bytes: int | None
    ram_max_bytes: int | None
    free_ram_boot: int | None
    free_heap_min: int | None
    stack_hwm: int | None
    free_psram_min: int | None
    arena_location: str | None
    median_selector_ms: float | None
    median_model_ms: float | None
    median_total_ms: float | None
    parity_rate: float | None
    offline_roc_auc: float | None
    offline_f1: float | None


@dataclass(frozen=True, slots=True)
class FitReportPaths:
    """Paths written by `generate_fit_report`."""

    csv_path: Path
    md_path: Path
    rows: tuple[FitRow, ...]


FIT_CSV_COLUMNS: tuple[str, ...] = (
    "board",
    "model",
    "mode",
    "verdict",
    "reason",
    "measured_transport",
    "pred_weight_flash_bytes [predicted]",
    "pred_arena_bytes [predicted]",
    "compile_ok [measured]",
    "flash_used_bytes [measured]",
    "flash_max_bytes [measured]",
    "ram_used_bytes [measured]",
    "ram_max_bytes [measured]",
    "free_ram_boot [measured]",
    "free_heap_min [measured]",
    "stack_hwm [measured]",
    "free_psram_min [measured]",
    "arena_location [measured]",
    "median_selector_ms [measured]",
    "median_model_ms [measured]",
    "median_total_ms [measured]",
    "parity_rate [measured]",
    "offline_roc_auc [offline]",
    "offline_f1 [offline]",
)


def _opt_float(val: object) -> float | None:
    if val is None or val == "":
        return None
    return float(val)


def _opt_int(val: object) -> int | None:
    if val is None or val == "":
        return None
    return int(val)


def _load_predicted_memory(cfg: HwTestConfig) -> dict[tuple[str, str], PredictedMemory]:
    target = cfg.c_imp_cfg.effective_target_tag
    c_imp_dir = cfg.fit.c_imp_results_dir / target
    gen_dir = cfg.c_imp_cfg.generated_dir

    out: dict[tuple[str, str], PredictedMemory] = {}
    mem_path = c_imp_dir / "memory_report.json"
    if mem_path.is_file():
        mem = json.loads(mem_path.read_text(encoding="utf-8"))
        for row in mem.get("models", []):
            mname = str(row.get("model_name", "")).strip()
            if not mname:
                continue
            w_bytes = _opt_int(row.get("weight_flash_bytes"))
            q15_arena = int(row.get("peak_arena_q15_bytes", 0))
            f32_arena = int(row.get("peak_arena_f32_bytes", 0))
            out[(mname, "q15")] = PredictedMemory(weight_flash_bytes=w_bytes, arena_bytes=q15_arena)
            out[(mname, "f32")] = PredictedMemory(weight_flash_bytes=w_bytes, arena_bytes=f32_arena)

    manifest_path = gen_dir / "export_manifest.json"
    if manifest_path.is_file():
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
        for mname, info in (man.get("tabular_models") or {}).items():
            w_bytes = _opt_int(info.get("weight_bytes"))
            for mode in ("q15", "f32"):
                out.setdefault((mname, mode), PredictedMemory(weight_flash_bytes=w_bytes, arena_bytes=0))
        for mname, info in (man.get("cnn_models") or {}).items():
            w_bytes = _opt_int(info.get("weight_flash_bytes"))
            q15_arena = int(info.get("peak_arena_q15_bytes", 0))
            out.setdefault((mname, "q15"), PredictedMemory(weight_flash_bytes=w_bytes, arena_bytes=q15_arena))
            out.setdefault(
                (mname, "f32"),
                PredictedMemory(weight_flash_bytes=w_bytes, arena_bytes=q15_arena * 2),
            )
    return out


def _load_offline_metrics(cfg: HwTestConfig) -> dict[tuple[str, str], OfflineModelMetrics]:
    target = cfg.c_imp_cfg.effective_target_tag
    out: dict[tuple[str, str], OfflineModelMetrics] = {}

    summary_csv = cfg.fit.c_imp_results_dir / target / "figures" / "summary_models_metrics.csv"
    if summary_csv.is_file():
        with summary_csv.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mname = str(row.get("model", "")).strip()
                if not mname:
                    continue
                out[(mname, "q15")] = OfflineModelMetrics(
                    roc_auc=_opt_float(row.get("q15_roc_auc")),
                    f1=_opt_float(row.get("q15_f1")),
                    source="summary_models_metrics.csv",
                )
                out[(mname, "f32")] = OfflineModelMetrics(
                    roc_auc=_opt_float(row.get("f32_roc_auc")),
                    f1=_opt_float(row.get("f32_f1")),
                    source="summary_models_metrics.csv",
                )

    for model in cfg.known_models:
        is_cnn = cfg.is_cnn_model(model)
        if not is_cnn:
            m_path = cfg.fit.classification_results_dir / target / f"full_{model}" / "metrics_summary.json"
            if m_path.is_file():
                data = json.loads(m_path.read_text(encoding="utf-8")).get("active_subset_metrics") or {}
                metrics = OfflineModelMetrics(
                    roc_auc=_opt_float(data.get("roc_auc")),
                    f1=_opt_float(data.get("f1")),
                    source=str(m_path.name),
                )
                for mode in ("q15", "f32"):
                    out.setdefault((model, mode), metrics)
        else:
            m_path = cfg.fit.cnn_results_dir / target / "full" / model / "metrics_summary.json"
            if m_path.is_file():
                data = json.loads(m_path.read_text(encoding="utf-8")).get("metrics") or {}
                metrics = OfflineModelMetrics(
                    roc_auc=_opt_float(data.get("roc_auc")),
                    f1=_opt_float(data.get("f1")),
                    source=str(m_path.name),
                )
                for mode in ("q15", "f32"):
                    out.setdefault((model, mode), metrics)
    return out


def _load_device_summaries(results_dir: Path, board_id: str) -> dict[tuple[str, str], dict[str, Any]]:
    board_dir = Path(results_dir) / board_id
    if not board_dir.is_dir():
        return {}
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for run_dir in sorted(board_dir.iterdir(), key=lambda p: p.name):
        if not run_dir.is_dir():
            continue
        sum_path = run_dir / "summary.json"
        if not sum_path.is_file():
            continue
        summary = json.loads(sum_path.read_text(encoding="utf-8"))
        model = str(summary.get("model", "")).strip()
        mode = str(summary.get("mode", "")).strip().lower()
        if model and mode:
            latest[(model, mode)] = summary
    return latest


def _assess_static_and_configured_fit(
    cfg: HwTestConfig,
    board: BoardSpec,
    model: str,
    mode: str,
    pred_mem: PredictedMemory | None,
    build_entry: Mapping[str, Any] | None,
) -> tuple[bool, str]:
    is_cnn = cfg.is_cnn_model(model)
    arena = pred_mem.arena_bytes if pred_mem is not None else 0
    w_flash = pred_mem.weight_flash_bytes if pred_mem is not None else None

    if build_entry is not None:
        if not bool(build_entry.get("compile_ok")) or not bool(build_entry.get("fits_budget")):
            return False, str(build_entry.get("reason") or "compiler/budget check failed")

    if model not in board.models:
        if is_cnn:
            if board.streaming:
                return (
                    False,
                    f"predicted arena {arena} B exceeds usable SRAM {board.usable_sram_budget_bytes} B "
                    f"(AVR streaming supports tabular only)",
                )
            return (
                False,
                f"predicted arena {arena} B exceeds usable SRAM {board.usable_sram_budget_bytes} B "
                f"with psram={board.psram_bytes} B",
            )
        return False, f"model '{model}' not configured for board '{board.board_id}'"

    eff = board.effective_settings(model)
    if mode not in eff.numeric_modes:
        return False, f"mode '{mode}' disabled for ({board.board_id}, {model}) [Decision D3]"

    if w_flash is not None and w_flash > board.flash_app_bytes:
        return False, f"predicted weight flash {w_flash} B > flash_app_bytes {board.flash_app_bytes} B"

    if is_cnn and arena > 0:
        loc = eff.arena_location
        if loc == "none":
            return False, f"arena {arena} B disallowed (arena_location='none')"
        if loc == "sram" and arena > board.usable_sram_budget_bytes:
            return False, f"predicted arena {arena} B > usable SRAM {board.usable_sram_budget_bytes} B"
        if loc == "psram" and arena > board.psram_bytes:
            return False, f"predicted arena {arena} B > PSRAM {board.psram_bytes} B"
        if loc == "auto" and arena > board.usable_sram_budget_bytes and arena > board.psram_bytes:
            return (
                False,
                f"predicted arena {arena} B > SRAM {board.usable_sram_budget_bytes} B and PSRAM {board.psram_bytes} B",
            )

    return True, "ok"


def evaluate_verdicts(rows: Sequence[FitRow], max_latency_ms: float) -> tuple[FitRow, ...]:
    """Apply Stage 8 verdict taxonomy in strict order over candidate `FitRow`s.

    Order:
        1. `does_not_fit`
        2. `not_measured`
        3. `fits_exceeds_latency`
        4. `fits_dominated` (`lr_dot` baseline is exempt)
        5. `fits_viable`
    """
    if max_latency_ms <= 0.0:
        raise ValueError(f"max_latency_ms must be positive, got {max_latency_ms}")

    stage1: list[FitRow] = []
    for row in rows:
        if not row.fit_possible:
            stage1.append(replace(row, verdict=VERDICT_DOES_NOT_FIT))
        elif row.median_total_ms is None:
            stage1.append(
                replace(
                    row,
                    verdict=VERDICT_NOT_MEASURED,
                    reason=row.reason if row.reason and row.reason != "ok" else "no device measurement yet",
                )
            )
        elif row.median_total_ms > max_latency_ms:
            stage1.append(
                replace(
                    row,
                    verdict=VERDICT_FITS_EXCEEDS_LATENCY,
                    reason=f"median_total_ms {row.median_total_ms:.3f} > max_latency_ms {max_latency_ms:.3f}",
                )
            )
        else:
            stage1.append(replace(row, verdict=VERDICT_FITS_VIABLE))

    final: list[FitRow] = []
    for idx, row in enumerate(stage1):
        if row.verdict != VERDICT_FITS_VIABLE or row.model == "lr_dot":
            final.append(row)
            continue

        assert row.median_total_ms is not None
        dominator: FitRow | None = None
        if row.offline_roc_auc is not None:
            for jdx, other in enumerate(stage1):
                if idx == jdx or other.board_id != row.board_id or other.model == row.model:
                    continue
                if not other.fit_possible or other.median_total_ms is None or other.offline_roc_auc is None:
                    continue
                if (
                    other.median_total_ms < row.median_total_ms
                    and other.offline_roc_auc > row.offline_roc_auc
                ):
                    dominator = other
                    break

        if dominator is not None:
            final.append(
                replace(
                    row,
                    verdict=VERDICT_FITS_DOMINATED,
                    reason=(
                        f"dominated by {dominator.model}/{dominator.mode} "
                        f"({dominator.median_total_ms:.3f} ms, AUC {dominator.offline_roc_auc:.4f})"
                    ),
                )
            )
        else:
            final.append(replace(row, reason="viable"))
    return tuple(final)


def _fmt_csv(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _fmt_md(value: object, unit: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}{unit}" if unit else f"{value:.4f}"
    return f"{value}{unit}"


def generate_fit_report(cfg: HwTestConfig) -> FitReportPaths:
    """Build and write `results/hw_test/fit_report.csv` and `results/hw_test/fit_report.md`."""
    pred_mem_map = _load_predicted_memory(cfg)
    offline_map = _load_offline_metrics(cfg)

    candidates: list[FitRow] = []
    for board_id, board_spec in cfg.boards.items():
        build_cache = load_build_cache(cfg.paths.results_dir, board_id)
        device_runs = _load_device_summaries(cfg.paths.results_dir, board_id)

        for model in cfg.known_models:
            modes = (
                board_spec.effective_settings(model).numeric_modes
                if model in board_spec.models
                else board_spec.numeric_modes
            )
            for mode in modes:
                pred_mem = pred_mem_map.get((model, mode))
                off = offline_map.get((model, mode))
                b_entry = build_cache.get((model, mode))
                run_sum = device_runs.get((model, mode))

                fit_ok, reason = _assess_static_and_configured_fit(
                    cfg, board_spec, model, mode, pred_mem, b_entry
                )

                timing_ms = (run_sum.get("timing_ms") or {}) if run_sum else {}
                parity_obj = (run_sum.get("parity") or {}) if run_sum else {}
                mem_obj = (run_sum.get("memory") or {}) if run_sum else {}

                candidates.append(
                    FitRow(
                        board_id=board_id,
                        model=model,
                        mode=mode,
                        fit_possible=fit_ok,
                        verdict="",
                        reason=reason,
                        measured_transport=str(run_sum["transport"]) if run_sum else None,
                        pred_weight_flash_bytes=(
                            pred_mem.weight_flash_bytes if pred_mem is not None else None
                        ),
                        pred_arena_bytes=pred_mem.arena_bytes if pred_mem is not None else 0,
                        compile_ok=(
                            bool(b_entry["compile_ok"]) if b_entry and "compile_ok" in b_entry else None
                        ),
                        flash_used_bytes=_opt_int(b_entry.get("flash_used_bytes")) if b_entry else None,
                        flash_max_bytes=_opt_int(b_entry.get("flash_max_bytes")) if b_entry else None,
                        ram_used_bytes=_opt_int(b_entry.get("ram_used_bytes")) if b_entry else None,
                        ram_max_bytes=_opt_int(b_entry.get("ram_max_bytes")) if b_entry else None,
                        free_ram_boot=_opt_int(mem_obj.get("free_ram_boot")) if run_sum else None,
                        free_heap_min=_opt_int(mem_obj.get("free_heap_min_overall")) if run_sum else None,
                        stack_hwm=_opt_int(mem_obj.get("stack_hwm_min_overall")) if run_sum else None,
                        free_psram_min=_opt_int(mem_obj.get("free_psram_min_overall")) if run_sum else None,
                        arena_location=str(mem_obj["arena_location"]) if mem_obj.get("arena_location") else None,
                        median_selector_ms=_opt_float(timing_ms.get("median_selector_ms")),
                        median_model_ms=_opt_float(timing_ms.get("median_model_ms")),
                        median_total_ms=_opt_float(timing_ms.get("median_total_ms")),
                        parity_rate=_opt_float(parity_obj.get("parity_rate")),
                        offline_roc_auc=off.roc_auc if off is not None else None,
                        offline_f1=off.f1 if off is not None else None,
                    )
                )

    evaluated = evaluate_verdicts(candidates, max_latency_ms=cfg.fit.max_latency_ms)

    out_dir = cfg.paths.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "fit_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(FIT_CSV_COLUMNS))
        writer.writeheader()
        for row in evaluated:
            writer.writerow(
                {
                    "board": row.board_id,
                    "model": row.model,
                    "mode": row.mode,
                    "verdict": row.verdict,
                    "reason": row.reason,
                    "measured_transport": row.measured_transport or "",
                    "pred_weight_flash_bytes [predicted]": _fmt_csv(row.pred_weight_flash_bytes),
                    "pred_arena_bytes [predicted]": _fmt_csv(row.pred_arena_bytes),
                    "compile_ok [measured]": _fmt_csv(row.compile_ok),
                    "flash_used_bytes [measured]": _fmt_csv(row.flash_used_bytes),
                    "flash_max_bytes [measured]": _fmt_csv(row.flash_max_bytes),
                    "ram_used_bytes [measured]": _fmt_csv(row.ram_used_bytes),
                    "ram_max_bytes [measured]": _fmt_csv(row.ram_max_bytes),
                    "free_ram_boot [measured]": _fmt_csv(row.free_ram_boot),
                    "free_heap_min [measured]": _fmt_csv(row.free_heap_min),
                    "stack_hwm [measured]": _fmt_csv(row.stack_hwm),
                    "free_psram_min [measured]": _fmt_csv(row.free_psram_min),
                    "arena_location [measured]": row.arena_location or "",
                    "median_selector_ms [measured]": _fmt_csv(row.median_selector_ms),
                    "median_model_ms [measured]": _fmt_csv(row.median_model_ms),
                    "median_total_ms [measured]": _fmt_csv(row.median_total_ms),
                    "parity_rate [measured]": _fmt_csv(row.parity_rate),
                    "offline_roc_auc [offline]": _fmt_csv(row.offline_roc_auc),
                    "offline_f1 [offline]": _fmt_csv(row.offline_f1),
                }
            )

    md_lines = [
        "# Hardware Fit & Pareto Report (`hw_test`)",
        "",
        f"- Target: `{cfg.c_imp_cfg.effective_target_tag}`",
        f"- Latency budget (`max_latency_ms`): `{cfg.fit.max_latency_ms:.1f} ms`",
        "- Column tags: `[predicted]` (`memory_report.json` / `export_manifest.json`), "
        "`[measured]` (compiler output / MCU run), `[offline]` (desktop evaluation).",
        "",
        "| board | model | mode | pred_weights_B [predicted] | pred_arena_B [predicted] | "
        "flash_B [measured] | ram_B [measured] | free_heap_min [measured] | stack_hwm [measured] | "
        "median_sel_ms [measured] | median_model_ms [measured] | median_total_ms [measured] | "
        "parity [measured] | roc_auc [offline] | f1 [offline] | verdict | reason |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in evaluated:
        flash_str = (
            f"{r.flash_used_bytes}/{r.flash_max_bytes}"
            if r.flash_used_bytes is not None and r.flash_max_bytes is not None
            else _fmt_md(r.flash_used_bytes)
        )
        ram_str = (
            f"{r.ram_used_bytes}/{r.ram_max_bytes}"
            if r.ram_used_bytes is not None and r.ram_max_bytes is not None
            else _fmt_md(r.ram_used_bytes)
        )
        parity_str = f"{r.parity_rate * 100:.1f}%" if r.parity_rate is not None else "-"
        md_lines.append(
            f"| {r.board_id} | {r.model} | {r.mode} | "
            f"{_fmt_md(r.pred_weight_flash_bytes)} | {_fmt_md(r.pred_arena_bytes)} | "
            f"{flash_str} | {ram_str} | "
            f"{_fmt_md(r.free_heap_min)} | {_fmt_md(r.stack_hwm)} | "
            f"{_fmt_md(r.median_selector_ms)} | {_fmt_md(r.median_model_ms)} | {_fmt_md(r.median_total_ms)} | "
            f"{parity_str} | {_fmt_md(r.offline_roc_auc)} | {_fmt_md(r.offline_f1)} | "
            f"`{r.verdict}` | {r.reason} |"
        )
    md_path = out_dir / "fit_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    logger.info("Wrote fit report (%d rows) to %s and %s", len(evaluated), csv_path, md_path)
    return FitReportPaths(csv_path=csv_path, md_path=md_path, rows=evaluated)
