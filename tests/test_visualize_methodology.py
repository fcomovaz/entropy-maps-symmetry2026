"""Unit and CLI tests for `experiments.visualize_methodology`."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.visualize_methodology import main as methodology_cli_main
from src.features.asymmetry_maps import asymmetry_column_names, compute_asymmetry_features
from src.features.entropy_maps import compute_entropy_features, feature_names


def _write_synthetic_methodology_project(root: Path) -> Path:
    (root / "configs" / "preprocessing").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "entropy").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "asymmetry").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "visualization").mkdir(parents=True, exist_ok=True)
    (root / "data" / "csv").mkdir(parents=True, exist_ok=True)

    (root / "configs" / "base.toml").write_text(
        """
[configs]
path = "configs/base.toml"

[datasets.mvtecad]
name = "mvtec-ad"
full_csv_path = "data/csv/mvtec-ad-entropy-full.csv"
full_csv_path_inverted = "data/csv/mvtec-ad-inverted-entropy-full.csv"
output_path = "data/preprocessed/mvtec-ad"
output_path_inverted = "data/preprocessed/mvtec-ad-inverted"
subdatasets = ["bottle", "tile"]

[datasets.kolektorsdd2]
name = "KolektorSDD2"
full_csv_path = "data/csv/kolektorsdd2-entropy-full.csv"
full_csv_path_inverted = "data/csv/kolektorsdd2-inverted-entropy-full.csv"
output_path = "data/preprocessed/KolektorSDD2"
output_path_inverted = "data/preprocessed/KolektorSDD2-inverted"
splits = ["train", "test"]
""",
        encoding="utf-8",
    )

    for ds_key, fname in (("kolektorsdd2", "kolektorsdd2.toml"), ("mvtecad", "mvtec_ad.toml")):
        (root / "configs" / "preprocessing" / fname).write_text(
            "[preprocessing]\ntarget_size = 32\n",
            encoding="utf-8",
        )
        (root / "configs" / "entropy" / fname).write_text(
            f"""
[entropy_maps]
base_config = "configs/base.toml"
dataset_key = "{ds_key}"
preprocessing_config = "configs/preprocessing/{fname}"
inverted = false
overwrite = false

[entropy_maps.regions]
grid_levels = [1, 2, 4, 8]
bins = 16

[entropy_maps.output]
feature_name_template = "E_{{g}}x{{g}}_r{{r}}_c{{c}}"
""",
            encoding="utf-8",
        )
        (root / "configs" / "asymmetry" / fname).write_text(
            f"""
[asymmetry_features]
base_config = "configs/base.toml"
entropy_config = "configs/entropy/{fname}"
dataset_key = "{ds_key}"
inverted = false
overwrite = false
schema = "x94-v1"

[asymmetry_features.regions]
scales = [2, 4, 8]
kinds = ["lr", "tb", "rot180"]

[asymmetry_features.output]
column_templates = {{ lr = "asym_lr_{{g}}x{{g}}", tb = "asym_tb_{{g}}x{{g}}", rot180 = "asym_rot180_{{g}}x{{g}}" }}
""",
            encoding="utf-8",
        )

    vis_cfg = Path("configs/visualization/methodology.toml")
    (root / vis_cfg).write_text(
        """
[methodology_viz]
base_config = "configs/base.toml"
kolektor_entropy_config = "configs/entropy/kolektorsdd2.toml"
kolektor_asymmetry_config = "configs/asymmetry/kolektorsdd2.toml"
mvtec_entropy_config = "configs/entropy/mvtec_ad.toml"
mvtec_asymmetry_config = "configs/asymmetry/mvtec_ad.toml"
output_dir = "results/methodology"
journal = "mdpi"
palette = "editorial"
dpi = 120
seed = 42
mvtec_object = "bottle"
mvtec_texture = "tile"
""",
        encoding="utf-8",
    )

    e_cols = list(feature_names((1, 2, 4, 8), template="E_{g}x{g}_r{r}_c{c}"))
    a_cols = list(
        asymmetry_column_names(
            (2, 4, 8),
            ("lr", "tb", "rot180"),
            {"lr": "asym_lr_{g}x{g}", "tb": "asym_tb_{g}x{g}", "rot180": "asym_rot180_{g}x{g}"},
        )
    )

    rng = np.random.default_rng(42)

    # KolektorSDD2 standard & inverted (letterbox black borders on left/right)
    for inv in (False, True):
        img_root = root / "data" / "preprocessed" / ("KolektorSDD2-inverted" if inv else "KolektorSDD2")
        csv_path = root / "data" / "csv" / (
            "kolektorsdd2-inverted-entropy-full.csv" if inv else "kolektorsdd2-entropy-full.csv"
        )
        rows = []
        for lbl, fname in (("good", "k_g.png"), ("defect", "k_d.png")):
            arr = np.zeros((32, 32), dtype=np.uint8)
            core = rng.integers(80, 140, size=(32, 12), dtype=np.uint8)
            if inv:
                core = (255 - core).astype(np.uint8)
            arr[:, 10:22] = core
            if lbl == "defect":
                arr[6:12, 12:18] = 240 if not inv else 15
            out_png = img_root / "test" / lbl / fname
            out_png.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr, mode="L").save(out_png)
            e85 = compute_entropy_features(arr, grid_levels=(1, 2, 4, 8), bins=16)[0]
            e_map = {
                1: e85[:1].reshape(1, 1),
                2: e85[1:5].reshape(1, 4),
                4: e85[5:21].reshape(1, 16),
                8: e85[21:85].reshape(1, 64),
            }
            a9 = compute_asymmetry_features(e_map, scales=(2, 4, 8), kinds=("lr", "tb", "rot180"))[0]
            rows.append(["test", lbl, fname, 16, "1|2|4|8", *e85.tolist(), *a9.tolist()])
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["split", "type", "file_name", "bins", "grid_levels", *e_cols, *a_cols])
            writer.writerows(rows)

    # MVTec-AD standard & inverted
    for inv in (False, True):
        img_root = root / "data" / "preprocessed" / ("mvtec-ad-inverted" if inv else "mvtec-ad")
        csv_path = root / "data" / "csv" / (
            "mvtec-ad-inverted-entropy-full.csv" if inv else "mvtec-ad-entropy-full.csv"
        )
        rows = []
        for obj in ("bottle", "tile"):
            for lbl, dname, fname in (("good", "good", "000.png"), ("defect", "crack", "001.png")):
                arr = rng.integers(40, 200, size=(32, 32), dtype=np.uint8)
                if lbl == "defect":
                    arr[4:12, 4:14] = 250
                if inv:
                    arr = (255 - arr).astype(np.uint8)
                out_png = img_root / obj / "test" / dname / fname
                out_png.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(arr, mode="L").save(out_png)
                e85 = compute_entropy_features(arr, grid_levels=(1, 2, 4, 8), bins=16)[0]
                e_map = {
                    2: e85[1:5].reshape(1, 4),
                    4: e85[5:21].reshape(1, 16),
                    8: e85[21:85].reshape(1, 64),
                }
                a9 = compute_asymmetry_features(e_map, scales=(2, 4, 8), kinds=("lr", "tb", "rot180"))[0]
                rows.append([obj, "test", lbl, dname, fname, 16, "1|2|4|8", *e85.tolist(), *a9.tolist()])
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["object_name", "split", "type", "defect_name", "file_name", "bins", "grid_levels", *e_cols, *a_cols]
            )
            writer.writerows(rows)

    return vis_cfg


class TestVisualizeMethodology(unittest.TestCase):
    """Verify dry-run, 300-DPI-ready figure generation, and overwrite semantics."""

    def test_dry_run_render_and_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_rel = _write_synthetic_methodology_project(root)
            out_dir = root / "results" / "methodology"

            rc_dry = methodology_cli_main(["--config", str(cfg_rel), "--dry-run"], project_root=root)
            self.assertEqual(rc_dry, 0)
            self.assertFalse(out_dir.exists())

            rc_run = methodology_cli_main(["--config", str(cfg_rel)], project_root=root)
            self.assertEqual(rc_run, 0)
            expected = (
                "kolektorsdd2_x94_methodology.png",
                "mvtec_ad_bottle_x94_methodology.png",
                "polarity_inversion_invariance_comparison.png",
            )
            for name in expected:
                p = out_dir / name
                self.assertTrue(p.is_file(), f"Missing {name}")
                self.assertGreater(p.stat().st_size, 1000)

            mtime_before = (out_dir / expected[0]).stat().st_mtime_ns
            rc_skip = methodology_cli_main(["--config", str(cfg_rel)], project_root=root)
            self.assertEqual(rc_skip, 0)
            self.assertEqual((out_dir / expected[0]).stat().st_mtime_ns, mtime_before)

            rc_ov = methodology_cli_main(["--config", str(cfg_rel), "--overwrite"], project_root=root)
            self.assertEqual(rc_ov, 0)
            self.assertTrue((out_dir / expected[0]).is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
