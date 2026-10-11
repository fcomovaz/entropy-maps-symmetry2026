"""Scientific and artifact checks; no predictive evaluation or downloads."""
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image
import compare_descriptors as cd
from synthetic_scenes import checkerboard, noise_texture, cellwise_collision, equal_histogram_pair
from perturbations import intensity, noise, blur, insert_defect, translate
from experiments import cases, EXPERIMENTS, FILENAMES


class HelperTests(unittest.TestCase):
    def test_known_metrics_and_tolerance(self):
        self.assertEqual(cd.metrics([0, 0], [3, 4]), {"l1": 7, "mad": 3.5, "l2": 5, "max_abs": 4, "changed": 2})
        self.assertEqual(cd.metrics([0, 0], [1e-6, 2e-6])["changed"], 1)
        for a, b, tolerance in (([], [], 0), ([0], [0, 1], 0), ([0], [float("nan")], 0), ([0], [0], -1)):
            with self.assertRaises(ValueError):
                cd.metrics(a, b, tolerance)

    def test_canonical_schema_and_blocks(self):
        self.assertEqual(cd.NAMES, cd.api.get_feature_names())
        self.assertEqual([len(v) for v in cd.BLOCKS.values()], [1, 4, 16, 64, 3, 3, 3])
        self.assertEqual([i for block in cd.BLOCKS.values() for i in block], list(range(94)))
        image = checkerboard(128)
        self.assertEqual(cd.extract(image), cd.api.extract_descriptor(image).vector)

    def test_identity_perturbations(self):
        image = checkerboard(128)
        for modified in (intensity(image), noise(image, 0, 42), blur(image, 0), translate(image, 0, 0),
                         insert_defect(image, noise_texture(8, 42), 10, 10, 0)[0]):
            self.assertEqual(image.tobytes(), modified.tobytes())

    def test_rounding_clipping_padding(self):
        image = Image.frombytes("L", (3, 1), bytes([0, 127, 255]))
        self.assertEqual(list(intensity(image, offset=20).tobytes()), [20, 147, 255])
        self.assertEqual(list(translate(image, 1, 0).tobytes()), [0, 0, 127])
        self.assertEqual(list(translate(image, -1, 0).tobytes()), [127, 255, 0])

    def test_seed_and_no_mutation(self):
        image = checkerboard(128)
        before = image.tobytes()
        self.assertEqual(noise(image, 10, 42).tobytes(), noise(image, 10, 42).tobytes())
        self.assertNotEqual(noise(image, 10, 42).tobytes(), noise(image, 10, 43).tobytes())
        insert_defect(image, noise_texture(8, 42), 10, 10)
        self.assertEqual(image.tobytes(), before)

    def test_defect_support_and_fixed_moving_texture(self):
        image = checkerboard(128)
        target = noise_texture(8, 42)
        for x in (8, 12, 16):
            modified, box = insert_defect(image, target, x, 20)
            self.assertEqual(modified.crop(box).tobytes(), target.tobytes())
            for yy in range(128):
                for xx in range(128):
                    if not (x <= xx < x+8 and 20 <= yy < 28):
                        self.assertEqual(image.getpixel((xx, yy)), modified.getpixel((xx, yy)))
        with self.assertRaises(ValueError):
            insert_defect(image, target, 127, 127)

    def test_locality_crossing_cell(self):
        image = checkerboard(128)
        modified, box = insert_defect(image, noise_texture(8, 42), 12, 20)
        result = cd.locality(image.size, [box], cd.extract(image), cd.extract(modified), 1e-6)
        self.assertEqual(result["expected_affected_cells"], ["entropy_8x8_r1_c0", "entropy_8x8_r1_c1"])
        self.assertTrue(result["locality_match"])

    def test_exact_histogram_collision(self):
        for size in (128, 256, 257):
            a, b = cellwise_collision(size)
            self.assertNotEqual(a.tobytes(), b.tobytes())
            for g in (1, 2, 4, 8):
                for box in cd.partition_boxes(size, size, g):
                    self.assertEqual(a.crop(box).histogram(), b.crop(box).histogram())
            for bins in (8, 16, 256):
                self.assertEqual(cd.extract(a, bins), cd.extract(b, bins))

    def test_patch_collision_depends_on_partition(self):
        a, b = equal_histogram_pair(32, 32)
        self.assertEqual(a.histogram(), b.histogram())
        self.assertEqual(cd.entropy_details(a)["entropy_bits"], 1)
        self.assertEqual(cd.entropy_details(b)["entropy_bits"], 1)
        self.assertNotEqual(cd.extract(a), cd.extract(b))
        rows = list(cases("same_histogram", 256, 20261001, 16))
        for row in rows[1:3]:
            self.assertEqual(cd.extract(row.base), cd.extract(row.image))
        self.assertNotEqual(cd.extract(rows[3].base), cd.extract(rows[3].image))

    def test_flip_permutation_and_invariants(self):
        for case in cases("transformations", 256, 20261001, 16):
            self.assertLess(case.extras["permuted_entropy_max_error"], 1e-12)
            self.assertEqual(cd.extract(case.base)[85:], cd.extract(case.image)[85:])
            self.assertLess(abs(case.extras["entropy_l2_norm_change"]), 1e-12)

    def test_symmetric_equal_area_control(self):
        pair, left, single = list(cases("symmetric_defects", 256, 20261001, 16))
        self.assertTrue(pair.extras["pixel_lr_symmetric"])
        self.assertFalse(left.extras["pixel_lr_symmetric"])
        self.assertEqual(pair.parameters["replacement_area"], left.parameters["replacement_area"])
        self.assertEqual(cd.extract(pair.base)[85:], [0]*9)
        for i in (85, 88, 91):
            self.assertEqual(cd.extract(pair.image)[i], 0)
            self.assertGreater(cd.extract(left.image)[i], 0)

    def test_boundary_independent_count_arithmetic(self):
        # g=8 boundary x=32, defect [28,76,36,84): two cells, 4x8 pixels each.
        case = next(c for c in cases("grid_boundary", 256, 20261001, 16)
                    if c.series == "grid_8" and c.parameter == 0)
        self.assertEqual(case.boxes, [(28, 76, 36, 84)])
        trace = {"box": case.boxes[0], "cells": []}
        vector = cd.extract(case.image)
        for col in (0, 1):
            counts = [0]*16
            counts[5] = counts[11] = 512  # 80 and 176, balanced base cell.
            x0, x1 = (28, 32) if col == 0 else (32, 36)
            for y in range(76, 84):
                for x in range(x0, x1):
                    counts[case.base.getpixel((x, y))*16//256] -= 1
                    counts[case.image.getpixel((x, y))*16//256] += 1
            expected = -sum((n/1024)*math.log2(n/1024) for n in counts if n)/4
            index = 21+2*8+col
            self.assertAlmostEqual(expected, vector[index], places=14)
            trace["cells"].append({"name": cd.NAMES[index], "counts": counts,
                                   "expected_entropy": expected, "actual_entropy": vector[index]})
        (ROOT / "results").mkdir(exist_ok=True)
        (ROOT / "results" / "boundary_manual_check.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")

    def test_curve_labels(self):
        self.assertEqual(cd.curve_shape([0, 0]), "approximately flat")
        self.assertEqual(cd.curve_shape([0, 1, 2]), "monotonic nondecreasing")
        self.assertEqual(cd.curve_shape([2, 1, 0]), "monotonic nonincreasing")
        self.assertEqual(cd.curve_shape([0, 1, 0]), "non-monotonic")

    def test_cli_help_and_invalid_inputs(self):
        result = subprocess.run([sys.executable, str(ROOT / "run_all.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("grid_boundary", result.stdout)
        for args in (("--size", "8"), ("--bins", "1"), ("--tolerance", "nan"), ("--output", str(ROOT.parent))):
            result = subprocess.run([sys.executable, str(ROOT / "run_all.py"), *args], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)


class ExecutedArtifactsTests(unittest.TestCase):
    """Requires default run first: checks every exported image, vector and seed replay."""
    def test_all_experiments_csv_images_and_reproducibility(self):
        count = 0
        for experiment in EXPERIMENTS:
            path = ROOT / "results" / (FILENAMES.get(experiment, experiment)+".csv")
            self.assertTrue(path.exists(), f"Run run_all.py first: missing {path}")
            manifest = json.loads(path.with_name(path.stem+"_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["feature_names"], cd.NAMES)
            for relative, expected_hash in manifest["source_sha256"].items():
                self.assertEqual(hashlib.sha256((cd.CODE / relative).read_bytes()).hexdigest(), expected_hash,
                                 f"Source changed after run: {relative}; rerun run_all.py")
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            base_rows = [r for r in rows if r["relation"] == "base"]
            regenerated = list(cases(experiment, 256, 20261001, 16))
            self.assertEqual(len(base_rows), len(regenerated))
            exported_vectors = {}
            for case, row in zip(regenerated, base_rows, strict=True):
                for field, image, bins in (("base", case.base, case.base_bins), ("modified", case.image, case.bins)):
                    vector = json.loads(row[field+"_vector"])
                    self.assertEqual(len(vector), 94)
                    self.assertEqual(vector, cd.extract(image, bins))
                    exported_vectors[(row[field+"_image"], bins)] = vector
                    with Image.open(ROOT / row[field+"_image"]) as saved:
                        saved.load()
                        self.assertEqual(saved.tobytes(), image.tobytes())
                    self.assertEqual(hashlib.sha256(image.tobytes()).hexdigest(), row[field+"_sha256"])
                delta = json.loads(row["delta"])
                self.assertEqual(len(delta), 94)
                self.assertTrue(all(math.isfinite(v) for v in delta))
            for row in rows:
                a, b = json.loads(row["base_vector"]), json.loads(row["modified_vector"])
                self.assertEqual(a, exported_vectors[(row["base_image"], int(row["base_bins"]))])
                self.assertEqual(b, exported_vectors[(row["modified_image"], int(row["bins"]))])
                for key, value in cd.compare(a, b).items():
                    self.assertEqual(float(row[key]), value)
                self.assertEqual(json.loads(row["top10"]), cd.top_features(a, b))
                count += 1
        self.assertEqual(count, 279)
        for directory in (ROOT / "outputs" / "images", ROOT / "outputs" / "plots"):
            for path in directory.rglob("*.png"):
                with Image.open(path) as image:
                    image.load()


if __name__ == "__main__":
    unittest.main(verbosity=2)
