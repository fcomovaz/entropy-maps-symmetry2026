"""Load, aggregate, and summarize generated C99 export and parity CSV/JSON outputs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import CImpConfig


@dataclass(frozen=True, slots=True)
class BinaryClassificationStats:
    """Binary classification metrics and confusion counts for one precision mode."""

    n_samples: int
    n_positives: int
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    tn: int
    fp: int
    fn: int
    tp: int


@dataclass(frozen=True, slots=True)
class ModelSeriesBundle:
    """Aligned per-sample arrays and derived statistics for one exported model."""

    name: str
    family: str
    file_names: tuple[str, ...]
    y_true: np.ndarray
    py_f32_label: np.ndarray
    py_f32_score: np.ndarray
    py_q15_label: np.ndarray
    py_q15_score: np.ndarray
    py_q15_raw: np.ndarray
    c_f32_label: np.ndarray | None
    c_f32_score: np.ndarray | None
    c_q15_label: np.ndarray | None
    c_q15_score: np.ndarray | None
    stats_f32: BinaryClassificationStats
    stats_q15: BinaryClassificationStats
    score_diff_mean: float
    score_diff_p95: float
    score_diff_max: float
    score_pearson_r: float
    decision_threshold: float
    q15_vs_f32_flips: int
    weight_format: str
    weight_flash_bytes: int | None
    peak_arena_f32_bytes: int | None
    peak_arena_q15_bytes: int | None
    fits_internal_sram_q15: bool | None
    fits_psram_q15: bool | None
    fits_flash: bool | None
    c_f32_max_abs_diff: float | None
    c_f32_label_mismatches: int | None
    c_q15_max_raw_diff: int | None
    c_q15_label_mismatches: int | None
    c_parity_passed: bool | None
    bench_f32_selector_ms: float | None
    bench_f32_model_ms: float | None
    bench_f32_total_ms: float | None
    bench_q15_selector_ms: float | None
    bench_q15_model_ms: float | None
    bench_q15_total_ms: float | None


@dataclass(frozen=True, slots=True)
class DescriptorParityRecord:
    """Standalone descriptor C parity and timing summary when present."""

    n_samples: int
    f32_max_abs_diff: float
    q15_vs_py_q15_max_raw_diff: int
    passed: bool
    bench_f32_ms: float | None
    bench_q15_ms: float | None


@dataclass(frozen=True, slots=True)
class DescriptorFeatureInsights:
    """Class-wise x94 spatial entropy, asymmetry, effect size, and Q0.15 error stats."""

    feature_names: tuple[str, ...]
    n_samples: int
    n_pos: int
    n_neg: int
    mean_neg_8x8: np.ndarray
    mean_pos_8x8: np.ndarray
    asym_names: tuple[str, ...]
    asym_mean_neg: np.ndarray
    asym_mean_pos: np.ndarray
    top_effect_names: tuple[str, ...]
    top_effect_values: np.ndarray
    q15_abs_err_mean: np.ndarray
    q15_abs_err_max: np.ndarray


@dataclass(frozen=True, slots=True)
class MemoryBudgets:
    """Target MCU Flash, internal SRAM, and PSRAM budgets in bytes."""

    flash_budget_bytes: int
    sram_budget_bytes: int
    psram_budget_bytes: int


@dataclass(frozen=True, slots=True)
class TargetVisualizationBundle:
    """Complete loaded artifact bundle for one exported/verified C99 target."""

    target_tag: str
    models: tuple[ModelSeriesBundle, ...]
    budgets: MemoryBudgets
    descriptor_parity: DescriptorParityRecord | None
    per_sample_f32_abs_diff: Mapping[str, np.ndarray]
    descriptor_insights: DescriptorFeatureInsights | None
    compiler_version: str | None
    all_gates_passed: bool | None


@dataclass(frozen=True, slots=True)
class VisualizationReport:
    """Paths to all summary files and PNG figures created during visualization."""

    target_tag: str
    output_dir: Path
    summary_csv_path: Path
    summary_md_path: Path
    figure_paths: tuple[Path, ...]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_float(val: Any) -> float | None:
    if val is None or val == "":
        return None
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def _compute_binary_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> BinaryClassificationStats:
    n = int(len(y_true))
    n_pos = int(np.sum(y_true == 1))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    has_two_classes = len(np.unique(y_true)) >= 2
    roc = float(roc_auc_score(y_true, y_score)) if has_two_classes else float("nan")
    pr = float(average_precision_score(y_true, y_score)) if has_two_classes else float("nan")
    bal_acc = float(balanced_accuracy_score(y_true, y_pred)) if has_two_classes else float("nan")

    return BinaryClassificationStats(
        n_samples=n,
        n_positives=n_pos,
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=bal_acc,
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc,
        pr_auc=pr,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
    )


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


_THRESHOLD_RE = re.compile(
    r"#define\s+CIMP_[A-Z0-9_]+_THRESHOLD(?:_LOGIT)?_F32\s+\(?([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)f?\)?"
)


def _extract_decision_threshold(
    generated_dir: Path,
    model_name: str,
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Read exact C header threshold constant, falling back to score/label separation midpoint."""
    header_path = generated_dir / f"model_{model_name}.h"
    if header_path.is_file():
        text = header_path.read_text(encoding="utf-8")
        match = _THRESHOLD_RE.search(text)
        if match is not None:
            try:
                return float(match.group(1))
            except ValueError:
                pass
    neg = scores[labels == 0]
    pos = scores[labels == 1]
    if len(neg) > 0 and len(pos) > 0:
        lo = float(np.max(neg))
        hi = float(np.min(pos))
        if lo <= hi:
            return 0.5 * (lo + hi)
    return 0.0


def _discover_golden_files(generated_dir: Path, cfg: CImpConfig) -> list[tuple[str, str, Path]]:
    """Return ordered list of `(model_name, family, golden_csv_path)` present in `generated_dir`."""
    if not generated_dir.is_dir():
        raise FileNotFoundError(
            f"Generated C directory not found: {generated_dir}. "
            "Run 'python -m experiments.export_c_pipeline --config ...' first."
        )

    discovered: dict[str, tuple[str, Path]] = {}
    for p in sorted(generated_dir.glob("golden_*.csv")):
        stem = p.stem
        if stem.startswith("golden_cnn_"):
            name = stem[len("golden_cnn_") :]
            discovered[name] = ("cnn", p)
        else:
            name = stem[len("golden_") :]
            discovered[name] = ("tabular", p)

    ordered: list[tuple[str, str, Path]] = []
    seen: set[str] = set()
    for tab_name in cfg.classifier.models:
        if tab_name in discovered:
            fam, p = discovered[tab_name]
            ordered.append((tab_name, fam, p))
            seen.add(tab_name)
    for cnn_name in cfg.cnn.models:
        if cnn_name in discovered:
            fam, p = discovered[cnn_name]
            ordered.append((cnn_name, fam, p))
            seen.add(cnn_name)
    for name in sorted(discovered):
        if name not in seen:
            fam, p = discovered[name]
            ordered.append((name, fam, p))

    if not ordered:
        raise FileNotFoundError(
            f"No golden_*.csv prediction files found in {generated_dir}."
        )
    return ordered


def _load_c_output_arrays(
    results_dir: Path,
    model_name: str,
    family: str,
    expected_len: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    raw_dir = results_dir / "c_outputs"
    prefix = f"cnn_{model_name}" if family == "cnn" else model_name
    f32_path = raw_dir / f"{prefix}_f32.csv"
    q15_path = raw_dir / f"{prefix}_q15.csv"
    score_col = "c_margin" if family == "cnn" else "c_logit"

    def _read_pair(path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not path.is_file():
            return None, None
        rows = _read_csv_rows(path)
        if len(rows) != expected_len:
            return None, None
        labels = np.array([int(r["c_label"]) for r in rows], dtype=np.int64)
        scores = np.array([float(r[score_col]) for r in rows], dtype=np.float64)
        return labels, scores

    c_f32_lbl, c_f32_score = _read_pair(f32_path)
    c_q15_lbl, c_q15_score = _read_pair(q15_path)
    return c_f32_lbl, c_f32_score, c_q15_lbl, c_q15_score


def _load_descriptor_insights(cfg: CImpConfig) -> DescriptorFeatureInsights | None:
    vectors_dir = cfg.results_dir / "vectors"
    f64_path = vectors_dir / "golden_x94_f64.csv"
    q15_path = vectors_dir / "golden_x94_q15.csv"
    if not f64_path.is_file():
        return None

    rows_f64 = _read_csv_rows(f64_path)
    if not rows_f64:
        return None

    feat_names = tuple(k for k in rows_f64[0].keys() if k not in ("index", "label"))
    labels = np.array([int(r["label"]) for r in rows_f64], dtype=np.int64)
    mat_f64 = np.array([[float(r[k]) for k in feat_names] for r in rows_f64], dtype=np.float64)

    neg_mask = labels == 0
    pos_mask = labels == 1
    n_neg = int(np.sum(neg_mask))
    n_pos = int(np.sum(pos_mask))

    # 8x8 entropy columns follow schema E_8x8_r<row>_c<col>
    grid_8x8_neg = np.zeros((8, 8), dtype=np.float64)
    grid_8x8_pos = np.zeros((8, 8), dtype=np.float64)
    name_to_idx = {name: i for i, name in enumerate(feat_names)}
    for r_idx in range(8):
        for c_idx in range(8):
            col_name = f"E_8x8_r{r_idx}_c{c_idx}"
            if col_name in name_to_idx:
                j = name_to_idx[col_name]
                grid_8x8_neg[r_idx, c_idx] = float(np.mean(mat_f64[neg_mask, j])) if n_neg > 0 else 0.0
                grid_8x8_pos[r_idx, c_idx] = float(np.mean(mat_f64[pos_mask, j])) if n_pos > 0 else 0.0

    asym_names = tuple(name for name in feat_names if name.startswith("asym_"))
    asym_neg = np.array(
        [float(np.mean(mat_f64[neg_mask, name_to_idx[n]])) if n_neg > 0 else 0.0 for n in asym_names],
        dtype=np.float64,
    )
    asym_pos = np.array(
        [float(np.mean(mat_f64[pos_mask, name_to_idx[n]])) if n_pos > 0 else 0.0 for n in asym_names],
        dtype=np.float64,
    )

    # Signed Cohen's d effect size (pos - neg) / pooled_std across all features
    if n_neg > 1 and n_pos > 1:
        m0 = np.mean(mat_f64[neg_mask], axis=0)
        m1 = np.mean(mat_f64[pos_mask], axis=0)
        v0 = np.var(mat_f64[neg_mask], axis=0, ddof=1)
        v1 = np.var(mat_f64[pos_mask], axis=0, ddof=1)
        pooled = np.sqrt(np.maximum(((n_neg - 1) * v0 + (n_pos - 1) * v1) / max(1, n_neg + n_pos - 2), 1e-18))
        effect = (m1 - m0) / pooled
    else:
        effect = np.zeros(len(feat_names), dtype=np.float64)

    top_k = min(12, len(feat_names))
    top_order = np.argsort(np.abs(effect))[::-1][:top_k]
    top_names = tuple(feat_names[int(i)] for i in top_order)
    top_vals = effect[top_order]

    # Quantization error if golden_x94_q15.csv is available
    q15_abs_mean = np.zeros(len(feat_names), dtype=np.float64)
    q15_abs_max = np.zeros(len(feat_names), dtype=np.float64)
    if q15_path.is_file():
        rows_q15 = _read_csv_rows(q15_path)
        if len(rows_q15) == len(rows_f64):
            q_keys = [k for k in rows_q15[0].keys() if k not in ("index", "label")]
            if len(q_keys) == len(feat_names):
                scale = float(1 << cfg.numeric.feature_frac_bits)
                mat_q15 = np.array([[float(r[k]) / scale for k in q_keys] for r in rows_q15], dtype=np.float64)
                abs_err = np.abs(mat_q15 - mat_f64)
                q15_abs_mean = np.mean(abs_err, axis=0)
                q15_abs_max = np.max(abs_err, axis=0)

    return DescriptorFeatureInsights(
        feature_names=feat_names,
        n_samples=len(rows_f64),
        n_pos=n_pos,
        n_neg=n_neg,
        mean_neg_8x8=grid_8x8_neg,
        mean_pos_8x8=grid_8x8_pos,
        asym_names=asym_names,
        asym_mean_neg=asym_neg,
        asym_mean_pos=asym_pos,
        top_effect_names=top_names,
        top_effect_values=top_vals,
        q15_abs_err_mean=q15_abs_mean,
        q15_abs_err_max=q15_abs_max,
    )


def load_target_bundle(cfg: CImpConfig) -> TargetVisualizationBundle:
    """Load generated CSVs plus optional memory, parity, timing, and descriptor reports."""
    golden_entries = _discover_golden_files(cfg.generated_dir, cfg)

    manifest_path = cfg.generated_dir / "export_manifest.json"
    manifest_data: dict[str, Any] = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    tabular_meta: Mapping[str, Any] = manifest_data.get("tabular_models", {})

    mem_path = cfg.results_dir / "memory_report.json"
    mem_by_model: dict[str, Mapping[str, Any]] = {}
    budgets = MemoryBudgets(
        flash_budget_bytes=cfg.cnn.flash_budget_bytes,
        sram_budget_bytes=cfg.cnn.sram_budget_bytes,
        psram_budget_bytes=cfg.cnn.psram_budget_bytes,
    )
    if mem_path.is_file():
        mem_json = json.loads(mem_path.read_text(encoding="utf-8"))
        b_raw = mem_json.get("budgets", {})
        budgets = MemoryBudgets(
            flash_budget_bytes=int(b_raw.get("flash_budget_bytes", budgets.flash_budget_bytes)),
            sram_budget_bytes=int(b_raw.get("sram_budget_bytes", budgets.sram_budget_bytes)),
            psram_budget_bytes=int(b_raw.get("psram_budget_bytes", budgets.psram_budget_bytes)),
        )
        for item in mem_json.get("models", []):
            mem_by_model[str(item["model_name"])] = item

    parity_path = cfg.results_dir / "parity_report.json"
    parity_by_model: dict[str, Mapping[str, Any]] = {}
    compiler_version: str | None = None
    all_gates_passed: bool | None = None
    descriptor_parity: DescriptorParityRecord | None = None
    if parity_path.is_file():
        p_json = json.loads(parity_path.read_text(encoding="utf-8"))
        compiler_version = str(p_json.get("compiler_version", "")) or None
        all_gates_passed = bool(p_json.get("all_gates_passed"))
        for item in p_json.get("models", []):
            parity_by_model[str(item["name"])] = item

    timing_path = cfg.results_dir / "timing_benchmark.csv"
    timing_lookup: dict[tuple[str, str], Mapping[str, str]] = {}
    if timing_path.is_file():
        for trow in _read_csv_rows(timing_path):
            timing_lookup[(trow["model"], trow["arith"])] = trow

    if "descriptor_x94" in parity_by_model:
        dp = parity_by_model["descriptor_x94"]
        desc_f32 = timing_lookup.get(("x94_only", "f32"))
        desc_q15 = timing_lookup.get(("x94_only", "q15"))
        descriptor_parity = DescriptorParityRecord(
            n_samples=int(dp["n_samples"]),
            f32_max_abs_diff=float(dp["f32_max_abs_diff"]),
            q15_vs_py_q15_max_raw_diff=int(dp["q15_vs_py_q15_max_raw_diff"]),
            passed=bool(dp["passed"]),
            bench_f32_ms=_safe_float(desc_f32.get("total_ms_per_img") if desc_f32 else None),
            bench_q15_ms=_safe_float(desc_q15.get("total_ms_per_img") if desc_q15 else None),
        )

    per_sample_path = cfg.results_dir / "parity_per_sample.csv"
    per_sample_diffs: dict[str, np.ndarray] = {}
    if per_sample_path.is_file():
        grouped: dict[str, list[float]] = {}
        for prow in _read_csv_rows(per_sample_path):
            grouped.setdefault(str(prow["model"]), []).append(float(prow["f32_abs_diff"]))
        per_sample_diffs = {k: np.array(v, dtype=np.float64) for k, v in grouped.items()}

    bundles: list[ModelSeriesBundle] = []
    for name, family, g_path in golden_entries:
        rows = _read_csv_rows(g_path)
        if not rows:
            continue
        f32_score_col = "py_f32_margin" if family == "cnn" else "py_f32_logit"
        q15_score_col = "py_q15_margin_f64" if family == "cnn" else "py_q15_logit_f64"
        q15_raw_col = "py_q15_margin_raw" if family == "cnn" else "py_q15_logit_raw"

        file_names = tuple(r["file_name"] for r in rows)
        y_true = np.array([int(r["true_label"]) for r in rows], dtype=np.int64)
        py_f32_lbl = np.array([int(r["py_f32_label"]) for r in rows], dtype=np.int64)
        py_f32_sc = np.array([float(r[f32_score_col]) for r in rows], dtype=np.float64)
        py_q15_lbl = np.array([int(r["py_q15_label"]) for r in rows], dtype=np.int64)
        py_q15_sc = np.array([float(r[q15_score_col]) for r in rows], dtype=np.float64)
        py_q15_raw = np.array([int(r[q15_raw_col]) for r in rows], dtype=np.int64)

        c_f32_lbl, c_f32_sc, c_q15_lbl, c_q15_sc = _load_c_output_arrays(
            cfg.results_dir, name, family, len(rows)
        )

        abs_diff = np.abs(py_q15_sc - py_f32_sc)
        dec_thr = _extract_decision_threshold(cfg.generated_dir, name, py_f32_sc, py_f32_lbl)
        m_mem = mem_by_model.get(name, {})
        if family == "tabular":
            t_meta = tabular_meta.get(name, {})
            w_bits = int(t_meta["weight_bits"]) if "weight_bits" in t_meta else 16
            w_fmt = "q31" if w_bits == 32 else ("q7" if w_bits == 8 else "q15")
            w_flash = int(t_meta["weight_bytes"]) if "weight_bytes" in t_meta else None
            arena_f32 = None
            arena_q15 = None
            fits_sram = True
            fits_psram = True
            fits_flash = (w_flash <= budgets.flash_budget_bytes) if w_flash is not None else None
        else:
            w_fmt = str(m_mem.get("weight_format", "q15"))
            w_flash = int(m_mem["weight_flash_bytes"]) if "weight_flash_bytes" in m_mem else None
            arena_f32 = int(m_mem["peak_arena_f32_bytes"]) if "peak_arena_f32_bytes" in m_mem else None
            arena_q15 = int(m_mem["peak_arena_q15_bytes"]) if "peak_arena_q15_bytes" in m_mem else None
            fits_sram = bool(m_mem["fits_internal_sram_q15"]) if "fits_internal_sram_q15" in m_mem else None
            fits_psram = bool(m_mem["fits_psram_q15"]) if "fits_psram_q15" in m_mem else None
            fits_flash = bool(m_mem["fits_flash"]) if "fits_flash" in m_mem else None

        p_meta = parity_by_model.get(name, {})
        tf32 = timing_lookup.get((name, "f32"), {})
        tq15 = timing_lookup.get((name, "q15"), {})

        bundles.append(
            ModelSeriesBundle(
                name=name,
                family=family,
                file_names=file_names,
                y_true=y_true,
                py_f32_label=py_f32_lbl,
                py_f32_score=py_f32_sc,
                py_q15_label=py_q15_lbl,
                py_q15_score=py_q15_sc,
                py_q15_raw=py_q15_raw,
                c_f32_label=c_f32_lbl,
                c_f32_score=c_f32_sc,
                c_q15_label=c_q15_lbl,
                c_q15_score=c_q15_sc,
                stats_f32=_compute_binary_stats(y_true, py_f32_lbl, py_f32_sc),
                stats_q15=_compute_binary_stats(y_true, py_q15_lbl, py_q15_sc),
                score_diff_mean=float(np.mean(abs_diff)),
                score_diff_p95=float(np.percentile(abs_diff, 95)),
                score_diff_max=float(np.max(abs_diff)),
                score_pearson_r=_safe_pearson(py_f32_sc, py_q15_sc),
                decision_threshold=dec_thr,
                q15_vs_f32_flips=int(np.sum(py_f32_lbl != py_q15_lbl)),
                weight_format=w_fmt,
                weight_flash_bytes=w_flash,
                peak_arena_f32_bytes=arena_f32,
                peak_arena_q15_bytes=arena_q15,
                fits_internal_sram_q15=fits_sram,
                fits_psram_q15=fits_psram,
                fits_flash=fits_flash,
                c_f32_max_abs_diff=_safe_float(p_meta.get("f32_max_abs_diff")),
                c_f32_label_mismatches=int(p_meta["f32_label_mismatches"]) if "f32_label_mismatches" in p_meta else None,
                c_q15_max_raw_diff=int(p_meta["q15_vs_py_q15_max_raw_diff"]) if "q15_vs_py_q15_max_raw_diff" in p_meta else None,
                c_q15_label_mismatches=int(p_meta["q15_vs_py_q15_label_mismatches"]) if "q15_vs_py_q15_label_mismatches" in p_meta else None,
                c_parity_passed=bool(p_meta["passed"]) if "passed" in p_meta else None,
                bench_f32_selector_ms=_safe_float(tf32.get("selector_ms_per_img")),
                bench_f32_model_ms=_safe_float(tf32.get("model_ms_per_img")),
                bench_f32_total_ms=_safe_float(tf32.get("total_ms_per_img")),
                bench_q15_selector_ms=_safe_float(tq15.get("selector_ms_per_img")),
                bench_q15_model_ms=_safe_float(tq15.get("model_ms_per_img")),
                bench_q15_total_ms=_safe_float(tq15.get("total_ms_per_img")),
            )
        )

    return TargetVisualizationBundle(
        target_tag=cfg.effective_target_tag,
        models=tuple(bundles),
        budgets=budgets,
        descriptor_parity=descriptor_parity,
        per_sample_f32_abs_diff=per_sample_diffs,
        descriptor_insights=_load_descriptor_insights(cfg),
        compiler_version=compiler_version,
        all_gates_passed=all_gates_passed,
    )


def _fmt_opt_float(val: float | None, precision: int = 4) -> str:
    if val is None or math.isnan(val):
        return ""
    return f"{val:.{precision}f}"


def write_summary_tables(
    bundle: TargetVisualizationBundle,
    output_dir: Path,
    figure_paths: Sequence[Path],
) -> tuple[Path, Path]:
    """Write human-readable `summary_models_metrics.csv` and `summary_report.md`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary_models_metrics.csv"

    fieldnames = [
        "model",
        "family",
        "n_samples",
        "n_positives",
        "weight_format",
        "weight_flash_kb",
        "arena_q15_kb",
        "fits_internal_sram_q15",
        "f32_accuracy",
        "f32_balanced_acc",
        "f32_precision",
        "f32_recall",
        "f32_f1",
        "f32_roc_auc",
        "f32_pr_auc",
        "q15_accuracy",
        "q15_balanced_acc",
        "q15_precision",
        "q15_recall",
        "q15_f1",
        "q15_roc_auc",
        "q15_pr_auc",
        "delta_f1_q15_minus_f32",
        "q15_vs_f32_flips",
        "score_diff_mean",
        "score_diff_max",
        "score_pearson_r",
        "tp_q15",
        "fp_q15",
        "tn_q15",
        "fn_q15",
        "c_f32_max_abs_diff",
        "c_q15_max_raw_diff",
        "c_parity_passed",
        "bench_f32_total_ms",
        "bench_q15_total_ms",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in bundle.models:
            s32 = m.stats_f32
            sq = m.stats_q15
            writer.writerow(
                {
                    "model": m.name,
                    "family": m.family,
                    "n_samples": sq.n_samples,
                    "n_positives": sq.n_positives,
                    "weight_format": m.weight_format,
                    "weight_flash_kb": f"{m.weight_flash_bytes / 1024.0:.2f}" if m.weight_flash_bytes is not None else "",
                    "arena_q15_kb": f"{m.peak_arena_q15_bytes / 1024.0:.2f}" if m.peak_arena_q15_bytes is not None else "",
                    "fits_internal_sram_q15": m.fits_internal_sram_q15 if m.fits_internal_sram_q15 is not None else "",
                    "f32_accuracy": f"{s32.accuracy:.4f}",
                    "f32_balanced_acc": _fmt_opt_float(s32.balanced_accuracy),
                    "f32_precision": f"{s32.precision:.4f}",
                    "f32_recall": f"{s32.recall:.4f}",
                    "f32_f1": f"{s32.f1:.4f}",
                    "f32_roc_auc": _fmt_opt_float(s32.roc_auc),
                    "f32_pr_auc": _fmt_opt_float(s32.pr_auc),
                    "q15_accuracy": f"{sq.accuracy:.4f}",
                    "q15_balanced_acc": _fmt_opt_float(sq.balanced_accuracy),
                    "q15_precision": f"{sq.precision:.4f}",
                    "q15_recall": f"{sq.recall:.4f}",
                    "q15_f1": f"{sq.f1:.4f}",
                    "q15_roc_auc": _fmt_opt_float(sq.roc_auc),
                    "q15_pr_auc": _fmt_opt_float(sq.pr_auc),
                    "delta_f1_q15_minus_f32": f"{sq.f1 - s32.f1:+.4f}",
                    "q15_vs_f32_flips": m.q15_vs_f32_flips,
                    "score_diff_mean": f"{m.score_diff_mean:.6f}",
                    "score_diff_max": f"{m.score_diff_max:.6f}",
                    "score_pearson_r": _fmt_opt_float(m.score_pearson_r, precision=6),
                    "tp_q15": sq.tp,
                    "fp_q15": sq.fp,
                    "tn_q15": sq.tn,
                    "fn_q15": sq.fn,
                    "c_f32_max_abs_diff": f"{m.c_f32_max_abs_diff:.3e}" if m.c_f32_max_abs_diff is not None else "",
                    "c_q15_max_raw_diff": m.c_q15_max_raw_diff if m.c_q15_max_raw_diff is not None else "",
                    "c_parity_passed": m.c_parity_passed if m.c_parity_passed is not None else "",
                    "bench_f32_total_ms": _fmt_opt_float(m.bench_f32_total_ms, precision=4),
                    "bench_q15_total_ms": _fmt_opt_float(m.bench_q15_total_ms, precision=4),
                }
            )

    md_path = output_dir / "summary_report.md"
    md_lines = _build_markdown_summary(bundle, figure_paths)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def _build_markdown_summary(
    bundle: TargetVisualizationBundle,
    figure_paths: Sequence[Path],
) -> list[str]:
    models = bundle.models
    n_samples = models[0].stats_q15.n_samples
    n_pos = models[0].stats_q15.n_positives
    pos_pct = (100.0 * n_pos / n_samples) if n_samples > 0 else 0.0

    best_f1 = max(models, key=lambda m: (m.stats_q15.f1, m.stats_q15.accuracy))
    sram_models = [m for m in models if m.fits_internal_sram_q15]
    best_sram = (
        max(sram_models, key=lambda m: (m.stats_q15.f1, m.stats_q15.accuracy))
        if sram_models
        else None
    )
    timed_q15 = [m for m in models if m.bench_q15_total_ms is not None]
    fastest = min(timed_q15, key=lambda m: float(m.bench_q15_total_ms)) if timed_q15 else None

    lines = [
        f"# C99 Fixed-Point Export & Verification Summary: `{bundle.target_tag}`",
        "",
        "## What the Raw `golden_*.csv` Columns Mean",
        "",
        f"- **Dataset slice**: `{n_samples}` test images (`{n_pos}` defective / `{n_samples - n_pos}` normal, `{pos_pct:.1f}%` positives).",
        "- **Decision score (`logit` for `lr_dot`/`mlp_16_dot`, `margin = logit_1 - logit_0` for CNNs)**: values `> threshold` predict defect (`label = 1`); values `<= threshold` predict normal (`label = 0`), with `threshold = 0.0` for tabular models and each CNN's calibrated C threshold (`model_<name>.h`).",
        "- **`py_f32_*` vs `py_q15_*`**: continuous float32 reference vs integer fixed-point (`Q15` / `int8_per_channel` weights + `int64` accumulators). Comparing them shows how much accuracy or score margin changes under embedded integer quantization.",
        "- **`c_f32_*` & `c_q15_*`** (when `verify_c_pipeline` has been run): compiled C99 executable outputs compared against Python (`q15` raw bit diff = `0` verifies exact integer implementation parity).",
        "",
        "## Key Takeaways",
        "",
        f"- **Highest Q15 F1 overall**: `{best_f1.name}` (`Q15 F1 = {best_f1.stats_q15.f1:.3f}`, `F32 F1 = {best_f1.stats_f32.f1:.3f}`, `Q15 vs F32 flips = {best_f1.q15_vs_f32_flips}/{n_samples}`).",
    ]
    if best_sram is not None:
        arena_str = f"{best_sram.peak_arena_q15_bytes / 1024:.1f} KB" if best_sram.peak_arena_q15_bytes is not None else "stateless"
        lines.append(
            f"- **Best internal-SRAM model (<= {bundle.budgets.sram_budget_bytes // 1024} KB)**: `{best_sram.name}` (`Q15 F1 = {best_sram.stats_q15.f1:.3f}`, arena `{arena_str}`)."
        )
    if fastest is not None and fastest.bench_q15_total_ms is not None:
        lines.append(
            f"- **Fastest verified C99 Q15 pipeline**: `{fastest.name}` (`{fastest.bench_q15_total_ms:.4f} ms/img`, `Q15 F1 = {fastest.stats_q15.f1:.3f}`)."
        )
    if bundle.all_gates_passed is not None:
        lines.append(f"- **C99 Parity Verification Gate (`all_gates_passed`)**: `{bundle.all_gates_passed}`.")
    if bundle.descriptor_parity is not None:
        dp = bundle.descriptor_parity
        lines.append(
            f"- **Descriptor `x94` Parity**: `f32_max_abs_diff = {dp.f32_max_abs_diff:.3e}`, `q15_raw_diff = {dp.q15_vs_py_q15_max_raw_diff}`, `passed = {dp.passed}`."
        )

    lines.extend(
        [
            "",
            "## Model Comparison Table (`Q15` vs `float32`)",
            "",
            "| Model | Family | Weights | F32 F1 | Q15 F1 | Q15 PR-AUC | Q15 ROC-AUC | Flips (Q15 vs F32) | Max |Delta Score| | Flash (KB) | Arena Q15 (KB) | Q15 C (ms/img) | C Parity |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )

    for m in models:
        pr_str = f"{m.stats_q15.pr_auc:.3f}" if not math.isnan(m.stats_q15.pr_auc) else "n/a"
        roc_str = f"{m.stats_q15.roc_auc:.3f}" if not math.isnan(m.stats_q15.roc_auc) else "n/a"
        flash_str = f"{m.weight_flash_bytes / 1024.0:.1f}" if m.weight_flash_bytes is not None else "-"
        arena_str = f"{m.peak_arena_q15_bytes / 1024.0:.1f}" if m.peak_arena_q15_bytes is not None else "0.0"
        lat_str = f"{m.bench_q15_total_ms:.3f}" if m.bench_q15_total_ms is not None else "-"
        par_str = (
            f"PASS (q15 diff={m.c_q15_max_raw_diff})"
            if m.c_parity_passed
            else ("FAIL" if m.c_parity_passed is False else "-")
        )
        lines.append(
            f"| `{m.name}` | {m.family} | `{m.weight_format}` | {m.stats_f32.f1:.3f} | {m.stats_q15.f1:.3f} | "
            f"{pr_str} | {roc_str} | {m.q15_vs_f32_flips} | {m.score_diff_max:.4f} | "
            f"{flash_str} | {arena_str} | {lat_str} | {par_str} |"
        )

    lines.extend(["", "## Generated Figures", ""])
    for fp in figure_paths:
        lines.append(f"- `{fp.name}`")
    return lines
