"""Mathematical and integration tests; no training, benchmarks or new packages."""
import csv
from importlib import import_module
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT.parents[1]
sys.path.insert(0, str(CODE))
PACKAGE = "understanding.06_entropy_pyramid_descriptor"
api = import_module(PACKAGE)
descriptor = import_module(PACKAGE + ".descriptor")
entropy = import_module(PACKAGE + ".entropy")
partition = import_module(PACKAGE + ".partition")
names = import_module(PACKAGE + ".feature_names")
synthetic = import_module(PACKAGE + ".synthetic_examples")
sf = import_module("understanding.05_symmetry_features.symmetry_features")
reference = import_module("understanding.03_entropy_benchmark.entropy_benchmark")


class DescriptorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = synthetic.generate_examples()

    def test_dimensions(self):
        result = api.extract_descriptor(self.examples["iid_noise"])
        self.assertEqual((len(result.entropy_features), len(result.asymmetry_features),
                          len(result.vector)), (85, 9, 94))
        self.assertTrue(all(type(value) is float for value in result.vector))
        self.assertEqual(api.extract_descriptor_vector(self.examples["iid_noise"]), result.vector)

    def test_feature_names_and_exact_order(self):
        expected_entropy = [f"entropy_{g}x{g}_r{r}_c{c}"
                            for g in [1, 2, 4, 8] for r in range(g) for c in range(g)]
        expected_asymmetry = [
            "asym_lr_2x2", "asym_tb_2x2", "asym_rot180_2x2",
            "asym_lr_4x4", "asym_tb_4x4", "asym_rot180_4x4",
            "asym_lr_8x8", "asym_tb_8x8", "asym_rot180_8x8"]
        self.assertEqual(api.get_entropy_feature_names(), expected_entropy)
        self.assertEqual(api.get_asymmetry_feature_names(), expected_asymmetry)
        self.assertEqual(api.get_feature_names(), expected_entropy + expected_asymmetry)
        self.assertEqual([len(api.get_entropy_feature_names()), len(api.get_asymmetry_feature_names()),
                          len(api.get_feature_names()), len(set(api.get_feature_names()))], [85, 9, 94, 94])
        modified = api.get_feature_names()
        modified[0] = "changed"
        self.assertEqual(api.get_feature_names()[0], "entropy_1x1_r0_c0")

    def test_row_major_with_distinct_values(self):
        values = [i / 100 for i in range(85)]
        with patch.object(descriptor, "entropy_details", side_effect=[
                {"normalized_entropy": value} for value in values]):
            result = api.extract_descriptor(Image.new("L", (32, 32)))
        self.assertEqual(result.entropy_features, values)
        for offset, g in [(0, 1), (1, 2), (5, 4), (21, 8)]:
            for r in range(g):
                for c in range(g):
                    self.assertEqual(result.entropy_grids[g][r][c], values[offset + r * g + c])

    def test_asymmetry_not_dictionary_order(self):
        image = self.examples["asymmetric_layout"]
        expected = api.extract_descriptor(image).vector
        def reversed_keys(*grids):
            return dict(reversed(list(sf.compute_nine_asymmetry_features(*grids).items())))
        with patch.object(descriptor, "compute_nine_asymmetry_features", side_effect=reversed_keys):
            self.assertEqual(api.extract_descriptor_vector(image), expected)

    def test_uniform_zero(self):
        for value in (0, 91, 255):
            self.assertEqual(api.extract_descriptor_vector(Image.new("L", (128, 128), value)), [0.] * 94)

    def test_range(self):
        for bins in (2, 7, 16, 256):
            for image in self.examples.values():
                result = api.extract_descriptor(image, bins)
                for value in result.entropy_features:
                    self.assertTrue(math.isfinite(value) and -1e-12 <= value <= 1 + 1e-12)
                for value in result.asymmetry_features:
                    self.assertTrue(math.isfinite(value) and 0 <= value <= 1)

    def test_noise_high_entropy(self):
        values = api.extract_descriptor(self.examples["iid_noise"]).entropy_features
        self.assertGreater(min(values), .95)  # Fixed seed, >=1024 pixels/cell, B=16.

    def test_localization(self):
        result = api.extract_descriptor(self.examples["localized_texture"])
        for g, target in [(2, (0, 0)), (4, (1, 1)), (8, (3, 2))]:
            nonzero = [(r, c) for r in range(g) for c in range(g) if result.entropy_grids[g][r][c] > 0]
            self.assertEqual(nonzero, [target])
        self.assertTrue(all(value > 0 for value in result.asymmetry_features))

    def test_symmetric_and_asymmetric_layouts(self):
        self.assertEqual(api.extract_descriptor(self.examples["symmetric_layout"]).asymmetry_features, [0.] * 9)
        self.assertTrue(all(value > 0 for value in api.extract_descriptor(
            self.examples["asymmetric_layout"]).asymmetry_features))

    def test_determinism_and_no_mutation(self):
        image = self.examples["iid_noise"]
        before = image.tobytes()
        self.assertEqual(api.extract_descriptor(image), api.extract_descriptor(image))
        self.assertEqual(image.tobytes(), before)
        self.assertEqual(synthetic.generate_examples()["iid_noise"].tobytes(), before)

    def test_resolutions_and_rectangles(self):
        for width, height in [(8, 8), (128, 128), (256, 256), (512, 512), (257, 251), (500, 333), (8, 65)]:
            with self.subTest(size=(width, height)):
                rng = random.Random(19)
                image = Image.frombytes("L", (width, height), bytes(rng.randrange(256) for _ in range(width * height)))
                result = api.extract_descriptor(image)
                self.assertEqual(len(result.vector), 94)
                self.assertEqual((result.image_width, result.image_height), (width, height))
                self.assertEqual(result, api.extract_descriptor(image))

    def test_partition_coverage_no_gaps_no_overlaps(self):
        for width, height in [(8, 8), (128, 128), (257, 251), (500, 333)]:
            for g in (1, 2, 4, 8):
                boxes = partition.partition_boxes(width, height, g)
                self.assertEqual(boxes, partition.partition_boxes(width, height, g))
                visits = bytearray(width * height)
                for i, (x0, y0, x1, y1) in enumerate(boxes):
                    r, c = divmod(i, g)
                    self.assertEqual((x0, y0, x1, y1),
                                     (c * width // g, r * height // g, (c + 1) * width // g, (r + 1) * height // g))
                    self.assertTrue(0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height)
                    for y in range(y0, y1):
                        for x in range(x0, x1):
                            visits[y * width + x] += 1
                self.assertEqual(set(visits), {1})
                self.assertEqual(len(boxes), g * g)

    def test_rgb_and_path(self):
        image = Image.frombytes("RGB", (17, 19), bytes((i * 31) % 256 for i in range(17 * 19 * 3)))
        expected = api.extract_descriptor(image.convert("L"))
        self.assertEqual(api.extract_descriptor(image), expected)
        self.assertEqual(image.mode, "RGB")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rgb.png"
            image.save(path)
            self.assertEqual(api.extract_descriptor(path), expected)
            self.assertEqual(api.extract_descriptor(str(path)), expected)

    def test_quantization_boundaries(self):
        for bins in range(2, 257):
            image = Image.frombytes("L", (256, 1), bytes(range(256)))
            counts = entropy.entropy_details(image, bins)["counts"]
            expected = [sum(b * 256 <= p * bins < (b + 1) * 256 for p in range(256)) for b in range(bins)]
            self.assertEqual(counts, expected)
            self.assertEqual(sum(counts), 256)

    def test_equiprobability_all_supported_bins(self):
        for bins in range(2, 257):
            # ceil(256*b/B) supplies one legal intensity in every bin.
            alphabet = bytes((256 * b + bins - 1) // bins for b in range(bins))
            image = Image.frombytes("L", (8 * bins, 8), alphabet * 64)
            result = api.extract_descriptor(image, bins)
            self.assertEqual(result.entropy_features, [1.] * 85)
            self.assertEqual(result.asymmetry_features, [0.] * 9)

    def test_entropy_trace_and_reference03(self):
        for image in self.examples.values():
            details = entropy.entropy_details(image)
            self.assertEqual(sum(details["counts"]), image.width * image.height)
            self.assertAlmostEqual(sum(details["probabilities"]), 1)
            self.assertAlmostEqual(-sum(details["p_log2_p"]), details["entropy_bits"])
            self.assertAlmostEqual(details["entropy_bits"], reference.e1(details["counts"], image.width * image.height))

    def test_consistency_with05(self):
        for image in self.examples.values():
            result = api.extract_descriptor(image)
            expected = [function(result.entropy_grids[g]) for g in (2, 4, 8)
                        for function in (sf.reflection_asymmetry_lr, sf.reflection_asymmetry_tb, sf.rot180_asymmetry)]
            self.assertEqual(result.asymmetry_features, expected)  # Exact, not almostEqual.

    def test_vector_name_value_consistency(self):
        image = self.examples["asymmetric_layout"]
        result = api.extract_descriptor(image)
        for index, name in enumerate(result.feature_names):
            if name.startswith("entropy_"):
                _, size, row, col = name.split("_")
                g, r, c = int(size.split("x")[0]), int(row[1:]), int(col[1:])
                box = partition.partition_boxes(*image.size, g)[r * g + c]
                expected = entropy.entropy_details(image.crop(box))["normalized_entropy"]
                self.assertEqual(expected, result.entropy_grids[g][r][c])
            else:
                _, measure, size = name.split("_")
                g = int(size.split("x")[0])
                expected = getattr(sf, f"asymmetry_{measure}_{g}x{g}")(result.entropy_grids[g])
            self.assertEqual(result.vector[index], expected, name)
        self.assertEqual(result.vector, result.entropy_features + result.asymmetry_features)

    def test_manual_example(self):
        image = synthetic.worked_image()
        result = api.extract_descriptor(image)
        expected_global = -(7/8 * math.log2(7/8) + 1/8 * math.log2(1/8)) / 4
        self.assertAlmostEqual(result.vector[0], expected_global)
        self.assertEqual(entropy.entropy_details(image)["counts"], [224] + [0] * 14 + [32])
        for g in (2, 4, 8):
            self.assertEqual(result.entropy_grids[g], [[.25 if r < g//2 and c < g//2 else 0.
                                                       for c in range(g)] for r in range(g)])
        self.assertEqual(result.asymmetry_features, [.125] * 9)

    def test_coarse_entropy_is_not_average(self):
        image = Image.new("L", (16, 16))
        image.paste(255, (8, 0, 16, 16))
        result = api.extract_descriptor(image)
        self.assertEqual(result.vector[0], .25)
        self.assertEqual(result.entropy_grids[2], [[0., 0.], [0., 0.]])

    def test_invalid_inputs(self):
        for bins in (0, 1, 257, 2.0, True, "16"):
            with self.assertRaises(ValueError):
                api.extract_descriptor(self.examples["uniform"], bins)
        for size in [(7, 8), (8, 7), (0, 0)]:
            with self.assertRaises(ValueError):
                api.extract_descriptor(Image.new("L", size))
        with self.assertRaises(ValueError):
            entropy.entropy_details(Image.new("L", (0, 0)))
        for args in [(8, 8, 0), (7, 8, 8), (8.0, 8, 2)]:
            with self.assertRaises(ValueError):
                partition.partition_boxes(*args)


class IntegrationTests(unittest.TestCase):
    def run_cli(self, *args, cwd=None):
        completed = subprocess.run([sys.executable, str(ROOT / "run_demo.py"), *map(str, args)],
                                   cwd=cwd, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        return completed.stdout

    def check_exports(self, path):
        payload = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        with path.with_suffix(".csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(payload["feature_names"], api.get_feature_names())
        self.assertEqual(list(payload["features"]), api.get_feature_names())
        self.assertEqual([row["feature_name"] for row in rows], api.get_feature_names())
        self.assertEqual([float(row["value"]) for row in rows], payload["vector"])
        self.assertEqual(list(payload["features"].values()), payload["vector"])
        self.assertEqual(len(payload["vector"]), 94)
        return payload

    def test_default_demo_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            stdout = self.run_cli("--output", out)
            self.assertIn("Saved 6 cases", stdout)
            for name in synthetic.generate_examples():
                self.check_exports(out / "descriptors" / f"{name}.json")
                self.assertTrue((out / "grids" / name / "image_with_grid_8x8.png").is_file())
            pngs = list(out.rglob("*.png"))
            self.assertEqual(len(pngs), 54)
            for path in pngs:
                with Image.open(path) as image:
                    image.load()
            trace = json.loads((out / "descriptors" / "worked_trace.json").read_text())
            self.assertEqual(len(trace["matrix"]), 16)
            self.assertEqual([len(trace["levels"][str(g)]) for g in (1, 2, 4, 8)], [1, 4, 16, 64])
            self.assertEqual(trace["asymmetry_features"], [.125] * 9)

    def test_cli_rgb_nondivisible_and_explicit_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = folder / "rgb input.png"
            Image.new("RGB", (257, 251), (12, 128, 237)).save(source)
            for suffix in (".json", ".csv"):
                output = folder / ("export" + suffix)
                self.run_cli("--image", source, "--bins", 16, "--output", output, cwd=folder)
                payload = self.check_exports(output)
                self.assertEqual((payload["metadata"]["width"], payload["metadata"]["height"]), (257, 251))
                self.assertEqual(payload["vector"], [0.] * 94)

    def test_help_and_bad_cli(self):
        self.assertIn("--bins", self.run_cli("--help"))
        for args in [("--bins", "1"), ("--output", "ambiguous.json")]:
            result = subprocess.run([sys.executable, str(ROOT / "run_demo.py"), *args], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("error:", result.stderr)

    def test_downstream_import_without_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            downstream = Path(tmp) / "07_descriptor_behavior"
            downstream.mkdir()
            script = downstream / "consume.py"
            script.write_text(
                "from importlib import import_module\nimport sys\nfrom PIL import Image\n"
                f"sys.path.insert(0, {str(CODE)!r})\n"
                f"api = import_module({PACKAGE!r})\n"
                "result = api.extract_descriptor(Image.new('L', (128, 128)))\n"
                "assert len(result.vector) == len(api.get_feature_names()) == 94\n"
                "assert api.extract_descriptor_vector(Image.new('L', (128, 128))) == result.vector\n",
                encoding="utf-8")
            result = subprocess.run([sys.executable, str(script)], cwd=downstream, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertEqual(list(downstream.iterdir()), [script])


if __name__ == "__main__":
    unittest.main(verbosity=2)
