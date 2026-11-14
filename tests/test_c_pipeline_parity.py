"""End-to-end hermetic parity tests compiling C99 kernels via MSVC and comparing with Python."""

from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.c_export.c_writer import format_c_array, sha256_bytes, write_header
from src.c_export.cnn_codegen import emit_cnn_model, write_registry_and_reports
from src.c_export.cnn_graph import extract_cnn_ir_from_module
from src.c_export.cnn_quantize import quantize_cnn_graph
from src.c_export.config import load_c_imp_config
from src.c_export.descriptor_reference import extract_x94_f64_oracle, extract_x94_q15_batch
from src.c_export.luts import emit_config_and_lut_headers
from src.c_export.memory_plan import build_memory_plan
from src.c_export.parity import run_full_parity_verification
from src.models.cnn.architectures.registry import create_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestCPipelineParityEndToEnd(unittest.TestCase):
    def test_synthetic_descriptor_and_cnn_c_parity(self) -> None:
        base_cfg = load_c_imp_config(Path("configs/c_imp/kolektorsdd2.toml"), PROJECT_ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            paths = replace(
                base_cfg.paths,
                generated_root=tmp_root / "generated",
                build_root=tmp_root / "build",
                results_root=tmp_root / "results",
            )
            cfg = replace(
                base_cfg,
                paths=paths,
                cnn=replace(base_cfg.cnn, models=("ultra_tiny", "tiny_cnn", "tiny_resnet")),
                classifier=replace(base_cfg.classifier, models=()),
            )

            emit_config_and_lut_headers(cfg)

            # Synthetic 4x [128, 128] uint8 test images and golden x94 descriptors
            rng = np.random.default_rng(2026)
            imgs_u8 = rng.integers(0, 256, size=(4, 128, 128), dtype=np.uint8)
            labels = np.array([0, 1, 0, 1], dtype=np.int64)
            fnames = tuple(f"synth_{i}.png" for i in range(4))
            imgs_f32 = ((imgs_u8[:, None, :, :].astype(np.float32) / 255.0) - 0.5) / 0.5

            vectors_dir = cfg.results_dir / "vectors"
            images_dir = vectors_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = vectors_dir / "manifest.csv"
            with manifest_path.open("w", newline="", encoding="utf-8") as mf:
                w = csv.writer(mf)
                w.writerow(["index", "file_name", "split", "label", "bin_relpath", "sha256"])
                for idx in range(4):
                    raw = imgs_u8[idx].tobytes(order="C")
                    rel = f"{idx:05d}.bin"
                    (images_dir / rel).write_bytes(raw)
                    w.writerow([idx, fnames[idx], "test", int(labels[idx]), rel, sha256_bytes(raw)])

            x94_f64 = extract_x94_f64_oracle(imgs_u8, cfg)
            x94_q15 = extract_x94_q15_batch(imgs_u8, cfg)
            with (vectors_dir / "golden_x94_f64.csv").open("w", newline="", encoding="utf-8") as ff:
                w = csv.writer(ff)
                w.writerow(["index", "label", *(f"f{k}" for k in range(94))])
                for idx in range(4):
                    w.writerow([idx, int(labels[idx]), *(f"{float(v):.10g}" for v in x94_f64[idx])])
            with (vectors_dir / "golden_x94_q15.csv").open("w", newline="", encoding="utf-8") as qf:
                w = csv.writer(qf)
                w.writerow(["index", "label", *(f"f{k}" for k in range(94))])
                for idx in range(4):
                    w.writerow([idx, int(labels[idx]), *(int(v) for v in x94_q15[idx])])

            # Also emit embedded header for image 0
            write_header(
                cfg.generated_dir / "test_image_0.h",
                "CIMP_TEST_IMAGE_0_H",
                ["<stdint.h>"],
                format_c_array("CIMP_TEST_IMAGE_0_U8", imgs_u8[0].ravel(), "uint8_t", per_line=16, hex_int8=True),
            )

            # Export ultra_tiny, tiny_cnn, and tiny_resnet with synthetic weights
            exported_cnns = []
            torch.manual_seed(99)
            for cname in ("ultra_tiny", "tiny_cnn", "tiny_resnet"):
                m = create_model(cname, num_classes=2)
                m.eval()
                graph = extract_cnn_ir_from_module(
                    model=m,
                    model_name=cname,
                    input_shape=(1, 128, 128),
                    norm_mean=0.5,
                    norm_std=0.5,
                    selected_threshold=0.5,
                    fuse_relu=True,
                )
                qmodel = quantize_cnn_graph(graph, imgs_f32, cfg)
                plan = build_memory_plan(graph, qmodel.weight_format, qmodel.weight_bytes, cfg)
                exported_cnns.append(
                    emit_cnn_model(
                        cfg=cfg,
                        graph=graph,
                        qmodel=qmodel,
                        plan=plan,
                        eval_file_names=fnames,
                        eval_true_labels=labels,
                        eval_images_u8=imgs_u8,
                        eval_images_f32=imgs_f32,
                    )
                )

            write_registry_and_reports(cfg, exported_tabular={}, exported_cnns=exported_cnns)

            report = run_full_parity_verification(cfg, benchmark=True)
            self.assertTrue(report.runtime_tests_passed)
            self.assertTrue(report.all_gates_passed)
            self.assertIsNotNone(report.timing_csv_path)
            assert report.timing_csv_path is not None
            self.assertTrue(report.timing_csv_path.is_file())
            payload = json.loads(report.report_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["all_gates_passed"])
            for m_rep in payload["models"]:
                self.assertEqual(m_rep["q15_vs_py_q15_max_raw_diff"], 0)
                self.assertEqual(m_rep["f32_label_mismatches"], 0)


if __name__ == "__main__":
    unittest.main()
