"""End-to-end Python vs compiled C99 parity verification and desktop latency benchmarking."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess

import numpy as np

from configs.logging import logger

from .build_c import build_c_runtime_test, build_desktop_runner
from .config import CImpConfig


@dataclass(frozen=True, slots=True)
class ModelParitySummary:
    """Parity and gate outcome for one descriptor, tabular, or CNN pipeline."""

    name: str
    mode: str
    n_samples: int
    f32_max_abs_diff: float
    f32_label_mismatches: int
    q15_vs_py_q15_max_raw_diff: int
    q15_vs_py_q15_label_mismatches: int
    q15_vs_f32_label_mismatches: int
    passed: bool


@dataclass(frozen=True, slots=True)
class ParityReport:
    """Aggregate report emitted by `run_full_parity_verification`."""

    target_tag: str
    report_path: Path
    per_sample_csv_path: Path
    timing_csv_path: Path | None
    runtime_tests_passed: bool
    all_gates_passed: bool
    models: tuple[ModelParitySummary, ...]


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_x94_matrix(csv_path: Path) -> np.ndarray:
    rows = _read_csv_dicts(csv_path)
    if not rows:
        raise ValueError(f"Empty descriptor CSV: {csv_path}")
    feat_keys = [k for k in rows[0].keys() if k not in ("index", "label")]
    return np.array([[float(r[k]) for k in feat_keys] for r in rows], dtype=np.float64)


def _run_exe(cmd: list[str], cwd: Path) -> None:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    if completed.returncode != 0:
        msg = "\n".join(p for p in (completed.stdout, completed.stderr) if p).strip()
        raise RuntimeError(f"Command failed (exit={completed.returncode}): {' '.join(cmd)}\n{msg}")


def _verify_descriptor(
    cfg: CImpConfig,
    exe_path: Path,
    bench_reps: int,
    bench_warmup: int,
    bench_rows: list[dict[str, str]],
) -> ModelParitySummary:
    vectors_dir = cfg.results_dir / "vectors"
    manifest = vectors_dir / "manifest.csv"
    images_dir = vectors_dir / "images"
    raw_dir = cfg.results_dir / "c_outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)

    f32_csv = raw_dir / "desc_f32.csv"
    f32_x94_csv = raw_dir / "c_x94_f32.csv"
    q15_csv = raw_dir / "desc_q15.csv"
    q15_x94_csv = raw_dir / "c_x94_q15.csv"

    _run_exe(
        [
            str(exe_path),
            "--manifest",
            str(manifest),
            "--images-dir",
            str(images_dir),
            "--mode",
            "descriptor",
            "--model",
            "x94_only",
            "--arith",
            "f32",
            "--out-csv",
            str(f32_csv),
            "--x94-csv",
            str(f32_x94_csv),
        ],
        cwd=cfg.project_root,
    )
    _run_exe(
        [
            str(exe_path),
            "--manifest",
            str(manifest),
            "--images-dir",
            str(images_dir),
            "--mode",
            "descriptor",
            "--model",
            "x94_only",
            "--arith",
            "q15",
            "--out-csv",
            str(q15_csv),
            "--x94-csv",
            str(q15_x94_csv),
        ],
        cwd=cfg.project_root,
    )

    if bench_reps > 0:
        for arith_tag in ("f32", "q15"):
            b_csv = raw_dir / f"bench_descriptor_x94_only_{arith_tag}.csv"
            _run_exe(
                [
                    str(exe_path),
                    "--manifest",
                    str(manifest),
                    "--images-dir",
                    str(images_dir),
                    "--mode",
                    "descriptor",
                    "--model",
                    "x94_only",
                    "--arith",
                    arith_tag,
                    "--out-csv",
                    str(raw_dir / f"_bench_tmp_{arith_tag}.csv"),
                    "--bench-warmup",
                    str(bench_warmup),
                    "--bench-reps",
                    str(bench_reps),
                    "--bench-csv",
                    str(b_csv),
                ],
                cwd=cfg.project_root,
            )
            bench_rows.extend(_read_csv_dicts(b_csv))

    py_f64 = _load_x94_matrix(vectors_dir / "golden_x94_f64.csv")
    c_f32 = _load_x94_matrix(f32_x94_csv)
    py_q15 = _load_x94_matrix(vectors_dir / "golden_x94_q15.csv").astype(np.int64)
    c_q15 = _load_x94_matrix(q15_x94_csv).astype(np.int64)

    f32_diff = float(np.max(np.abs(py_f64 - c_f32)))
    q15_raw_diff = int(np.max(np.abs(py_q15 - c_q15)))

    passed = f32_diff <= cfg.verification.f32_descriptor_abs_tol and (
        not cfg.verification.q15_must_be_bit_exact_vs_python_model or q15_raw_diff == 0
    )
    return ModelParitySummary(
        name="descriptor_x94",
        mode="descriptor",
        n_samples=int(py_f64.shape[0]),
        f32_max_abs_diff=f32_diff,
        f32_label_mismatches=0,
        q15_vs_py_q15_max_raw_diff=q15_raw_diff,
        q15_vs_py_q15_label_mismatches=0,
        q15_vs_f32_label_mismatches=0,
        passed=passed,
    )


def _verify_tabular_model(
    cfg: CImpConfig,
    exe_path: Path,
    model_name: str,
    bench_reps: int,
    bench_warmup: int,
    bench_rows: list[dict[str, str]],
    per_sample_rows: list[dict[str, object]],
) -> ModelParitySummary:
    vectors_dir = cfg.results_dir / "vectors"
    manifest = vectors_dir / "manifest.csv"
    images_dir = vectors_dir / "images"
    raw_dir = cfg.results_dir / "c_outputs"
    golden = _read_csv_dicts(cfg.generated_dir / f"golden_{model_name}.csv")

    f32_csv = raw_dir / f"{model_name}_f32.csv"
    q15_csv = raw_dir / f"{model_name}_q15.csv"
    for arith_tag, out_p in (("f32", f32_csv), ("q15", q15_csv)):
        cmd = [
            str(exe_path),
            "--manifest",
            str(manifest),
            "--images-dir",
            str(images_dir),
            "--mode",
            "descriptor",
            "--model",
            model_name,
            "--arith",
            arith_tag,
            "--out-csv",
            str(out_p),
        ]
        if bench_reps > 0:
            b_csv = raw_dir / f"bench_{model_name}_{arith_tag}.csv"
            cmd.extend(
                [
                    "--bench-warmup",
                    str(bench_warmup),
                    "--bench-reps",
                    str(bench_reps),
                    "--bench-csv",
                    str(b_csv),
                ]
            )
            _run_exe(cmd, cwd=cfg.project_root)
            bench_rows.extend(_read_csv_dicts(b_csv))
        else:
            _run_exe(cmd, cwd=cfg.project_root)

    c_f32 = _read_csv_dicts(f32_csv)
    c_q15 = _read_csv_dicts(q15_csv)
    if len(c_f32) != len(golden) or len(c_q15) != len(golden):
        raise ValueError(f"Row count mismatch for tabular model {model_name}")

    max_f32_diff = 0.0
    f32_lbl_mm = 0
    max_q_raw_diff = 0
    q_py_lbl_mm = 0
    q_f32_lbl_mm = 0

    for g_row, cf_row, cq_row in zip(golden, c_f32, c_q15):
        f_diff = abs(float(cf_row["c_logit"]) - float(g_row["py_f32_logit"]))
        max_f32_diff = max(max_f32_diff, f_diff)
        if int(cf_row["c_label"]) != int(g_row["py_f32_label"]):
            f32_lbl_mm += 1
        q_diff = abs(int(cq_row["c_logit_raw_q"]) - int(g_row["py_q15_logit_raw"]))
        max_q_raw_diff = max(max_q_raw_diff, q_diff)
        if int(cq_row["c_label"]) != int(g_row["py_q15_label"]):
            q_py_lbl_mm += 1
        if int(cq_row["c_label"]) != int(g_row["py_f32_label"]):
            q_f32_lbl_mm += 1
        per_sample_rows.append(
            {
                "model": model_name,
                "index": g_row["index"],
                "file_name": g_row["file_name"],
                "true_label": g_row["true_label"],
                "py_f32_label": g_row["py_f32_label"],
                "c_f32_label": cf_row["c_label"],
                "py_q15_label": g_row["py_q15_label"],
                "c_q15_label": cq_row["c_label"],
                "f32_abs_diff": f"{f_diff:.9g}",
                "q15_raw_abs_diff": q_diff,
            }
        )

    passed = (
        max_f32_diff <= cfg.verification.f32_logit_abs_tol
        and f32_lbl_mm == 0
        and (
            not cfg.verification.q15_must_be_bit_exact_vs_python_model
            or (max_q_raw_diff == 0 and q_py_lbl_mm == 0)
        )
    )
    return ModelParitySummary(
        name=model_name,
        mode="descriptor",
        n_samples=len(golden),
        f32_max_abs_diff=max_f32_diff,
        f32_label_mismatches=f32_lbl_mm,
        q15_vs_py_q15_max_raw_diff=max_q_raw_diff,
        q15_vs_py_q15_label_mismatches=q_py_lbl_mm,
        q15_vs_f32_label_mismatches=q_f32_lbl_mm,
        passed=passed,
    )


def _verify_cnn_model(
    cfg: CImpConfig,
    exe_path: Path,
    model_name: str,
    bench_reps: int,
    bench_warmup: int,
    bench_rows: list[dict[str, str]],
    per_sample_rows: list[dict[str, object]],
) -> ModelParitySummary:
    vectors_dir = cfg.results_dir / "vectors"
    manifest = vectors_dir / "manifest.csv"
    images_dir = vectors_dir / "images"
    raw_dir = cfg.results_dir / "c_outputs"
    golden = _read_csv_dicts(cfg.generated_dir / f"golden_cnn_{model_name}.csv")

    f32_csv = raw_dir / f"cnn_{model_name}_f32.csv"
    q15_csv = raw_dir / f"cnn_{model_name}_q15.csv"
    for arith_tag, out_p in (("f32", f32_csv), ("q15", q15_csv)):
        cmd = [
            str(exe_path),
            "--manifest",
            str(manifest),
            "--images-dir",
            str(images_dir),
            "--mode",
            "cnn",
            "--model",
            model_name,
            "--arith",
            arith_tag,
            "--out-csv",
            str(out_p),
        ]
        if bench_reps > 0:
            b_csv = raw_dir / f"bench_cnn_{model_name}_{arith_tag}.csv"
            cmd.extend(
                [
                    "--bench-warmup",
                    str(bench_warmup),
                    "--bench-reps",
                    str(bench_reps),
                    "--bench-csv",
                    str(b_csv),
                ]
            )
            _run_exe(cmd, cwd=cfg.project_root)
            bench_rows.extend(_read_csv_dicts(b_csv))
        else:
            _run_exe(cmd, cwd=cfg.project_root)

    c_f32 = _read_csv_dicts(f32_csv)
    c_q15 = _read_csv_dicts(q15_csv)
    if len(c_f32) != len(golden) or len(c_q15) != len(golden):
        raise ValueError(f"Row count mismatch for CNN model {model_name}")

    max_f32_diff = 0.0
    f32_lbl_mm = 0
    max_q_raw_diff = 0
    q_py_lbl_mm = 0
    q_f32_lbl_mm = 0

    for g_row, cf_row, cq_row in zip(golden, c_f32, c_q15):
        d0 = abs(float(cf_row["c_logit0"]) - float(g_row["py_f32_logit0"]))
        d1 = abs(float(cf_row["c_logit1"]) - float(g_row["py_f32_logit1"]))
        f_diff = max(d0, d1)
        max_f32_diff = max(max_f32_diff, f_diff)
        if int(cf_row["c_label"]) != int(g_row["py_f32_label"]):
            f32_lbl_mm += 1
        qd0 = abs(int(cq_row["c_logit0_raw_q"]) - int(g_row["py_q15_logit0_raw"]))
        qd1 = abs(int(cq_row["c_logit1_raw_q"]) - int(g_row["py_q15_logit1_raw"]))
        q_diff = max(qd0, qd1)
        max_q_raw_diff = max(max_q_raw_diff, q_diff)
        if int(cq_row["c_label"]) != int(g_row["py_q15_label"]):
            q_py_lbl_mm += 1
        if int(cq_row["c_label"]) != int(g_row["py_f32_label"]):
            q_f32_lbl_mm += 1
        per_sample_rows.append(
            {
                "model": model_name,
                "index": g_row["index"],
                "file_name": g_row["file_name"],
                "true_label": g_row["true_label"],
                "py_f32_label": g_row["py_f32_label"],
                "c_f32_label": cf_row["c_label"],
                "py_q15_label": g_row["py_q15_label"],
                "c_q15_label": cq_row["c_label"],
                "f32_abs_diff": f"{f_diff:.9g}",
                "q15_raw_abs_diff": q_diff,
            }
        )

    passed = (
        max_f32_diff <= cfg.verification.f32_logit_abs_tol
        and f32_lbl_mm == 0
        and (
            not cfg.verification.q15_must_be_bit_exact_vs_python_model
            or (max_q_raw_diff == 0 and q_py_lbl_mm == 0)
        )
    )
    return ModelParitySummary(
        name=model_name,
        mode="cnn",
        n_samples=len(golden),
        f32_max_abs_diff=max_f32_diff,
        f32_label_mismatches=f32_lbl_mm,
        q15_vs_py_q15_max_raw_diff=max_q_raw_diff,
        q15_vs_py_q15_label_mismatches=q_py_lbl_mm,
        q15_vs_f32_label_mismatches=q_f32_lbl_mm,
        passed=passed,
    )


def run_full_parity_verification(
    cfg: CImpConfig,
    benchmark: bool = False,
) -> ParityReport:
    """Compile C runtime tests + desktop binary and verify Python-vs-C bit-exact parity."""
    runtime_build = build_c_runtime_test(cfg)
    _run_exe([str(runtime_build.exe_path)], cwd=cfg.project_root)

    desktop_build = build_desktop_runner(cfg)
    manifest_path = cfg.generated_dir / "export_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing export manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    bench_reps = cfg.verification.timing_timed_iters if benchmark else 0
    bench_warmup = cfg.verification.timing_warmup_iters if benchmark else 0
    bench_rows: list[dict[str, str]] = []
    per_sample_rows: list[dict[str, object]] = []
    summaries: list[ModelParitySummary] = []

    summaries.append(
        _verify_descriptor(cfg, desktop_build.exe_path, bench_reps, bench_warmup, bench_rows)
    )

    for tab_name in manifest.get("tabular_models", {}):
        summaries.append(
            _verify_tabular_model(
                cfg,
                desktop_build.exe_path,
                tab_name,
                bench_reps,
                bench_warmup,
                bench_rows,
                per_sample_rows,
            )
        )

    for cnn_name in manifest.get("cnn_models", {}):
        summaries.append(
            _verify_cnn_model(
                cfg,
                desktop_build.exe_path,
                cnn_name,
                bench_reps,
                bench_warmup,
                bench_rows,
                per_sample_rows,
            )
        )

    all_passed = all(s.passed for s in summaries)

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    per_sample_csv_path = cfg.results_dir / "parity_per_sample.csv"
    with per_sample_csv_path.open("w", newline="", encoding="utf-8") as pf:
        fieldnames = [
            "model",
            "index",
            "file_name",
            "true_label",
            "py_f32_label",
            "c_f32_label",
            "py_q15_label",
            "c_q15_label",
            "f32_abs_diff",
            "q15_raw_abs_diff",
        ]
        writer = csv.DictWriter(pf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_sample_rows)

    timing_csv_path: Path | None = None
    if bench_rows:
        timing_csv_path = cfg.results_dir / "timing_benchmark.csv"
        with timing_csv_path.open("w", newline="", encoding="utf-8") as tf:
            writer = csv.DictWriter(tf, fieldnames=list(bench_rows[0].keys()))
            writer.writeheader()
            writer.writerows(bench_rows)

    report_path = cfg.results_dir / "parity_report.json"
    report_payload = {
        "target_tag": cfg.effective_target_tag,
        "runtime_tests_passed": True,
        "all_gates_passed": all_passed,
        "desktop_exe": str(desktop_build.exe_path),
        "compiler_version": desktop_build.compiler_version,
        "models": [
            {
                "name": s.name,
                "mode": s.mode,
                "n_samples": s.n_samples,
                "f32_max_abs_diff": s.f32_max_abs_diff,
                "f32_label_mismatches": s.f32_label_mismatches,
                "q15_vs_py_q15_max_raw_diff": s.q15_vs_py_q15_max_raw_diff,
                "q15_vs_py_q15_label_mismatches": s.q15_vs_py_q15_label_mismatches,
                "q15_vs_f32_label_mismatches": s.q15_vs_f32_label_mismatches,
                "passed": s.passed,
            }
            for s in summaries
        ],
    }
    report_path.write_text(json.dumps(report_payload, indent=2) + "\n", encoding="utf-8")

    for s in summaries:
        logger.info(
            "Parity %-22s | f32_diff=%.3e (lbl_mm=%d) | q15_raw_diff=%d (lbl_mm=%d) | passed=%s",
            s.name,
            s.f32_max_abs_diff,
            s.f32_label_mismatches,
            s.q15_vs_py_q15_max_raw_diff,
            s.q15_vs_py_q15_label_mismatches,
            s.passed,
        )

    if not all_passed:
        failed = [s.name for s in summaries if not s.passed]
        raise ValueError(f"Parity verification failed for models: {failed} (see {report_path})")

    return ParityReport(
        target_tag=cfg.effective_target_tag,
        report_path=report_path,
        per_sample_csv_path=per_sample_csv_path,
        timing_csv_path=timing_csv_path,
        runtime_tests_passed=True,
        all_gates_passed=all_passed,
        models=tuple(summaries),
    )
