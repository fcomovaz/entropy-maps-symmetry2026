"""Mathematical, image-property and end-to-end artifact checks (stdlib unittest)."""
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image
import symmetry_features as sf
from entropy_pyramid import compute_entropy_pyramid, entropy_details, flatten_entropy_features
from synthetic_images import generate_images
from synthetic_matrices import controlled_matrices, progressive_defects

METRICS = (sf.reflection_asymmetry_lr, sf.reflection_asymmetry_tb, sf.rot180_asymmetry)
TRANSFORMS = (sf.mirror_left_right, sf.mirror_top_bottom, sf.rotate_180)


class MathematicalTests(unittest.TestCase):
    def test_controlled_symmetries(self):
        expected = [(0, 0, 0), (0, .425, .425), (.425, 0, .425),
                    (.2, .2, 0), (.4, .2, .4)]
        for (name, grid), values in zip(controlled_matrices().items(), expected):
            for metric, target in zip(METRICS, values):
                with self.subTest(name=name, metric=metric.__name__):
                    self.assertAlmostEqual(metric(grid), target, places=14)

    def test_manual_elementwise_differences(self):
        grid = [[.1, .7], [.2, .4]]
        expected = ([[.6, .6], [.2, .2]], [[.1, .3], [.1, .3]],
                    [[.3, .5], [.5, .3]])
        for transform, target in zip(TRANSFORMS, expected):
            actual = sf.absolute_difference(grid, transform(grid))
            for row, expected_row in zip(actual, target):
                for value, reference in zip(row, expected_row):
                    self.assertAlmostEqual(value, reference)

    def test_involutions_and_nonmutation(self):
        grid = [[.1, .7], [.2, .4]]
        for transform in TRANSFORMS:
            self.assertEqual(transform(transform(grid)), grid)
            result = transform(grid)
            result[0][0] = 1
            self.assertEqual(grid, [[.1, .7], [.2, .4]])
        self.assertEqual(sf.rotate_180(grid), sf.mirror_top_bottom(sf.mirror_left_right(grid)))

    def test_all_nine_wrappers_and_dimensions(self):
        for g in (2, 4, 8):
            grid = [[(i*g+j)/(g*g) for j in range(g)] for i in range(g)]
            for name, reference in zip(("lr", "tb", "rot180"), METRICS):
                function = getattr(sf, f"asymmetry_{name}_{g}x{g}")
                self.assertEqual(function(grid), reference(grid))
                for wrong in (1, 2, 4, 8):
                    if wrong != g:
                        with self.assertRaisesRegex(ValueError, f"Expected {g}x{g}"):
                            function([[.5]*wrong for _ in range(wrong)])

    def test_invalid_values_and_shapes(self):
        for grid in ([], [[]], [[.1, .2]], [[.1], [.2, .3]],
                     [[-0.01]], [[1.01]], [[math.nan]], [[math.inf]]):
            for function in (*TRANSFORMS, *METRICS):
                with self.assertRaises(ValueError):
                    function(grid)

    def test_range_and_triangle_relations(self):
        rng = random.Random(20261001)
        for g in (1, 2, 3, 4, 8):
            for _ in range(40):
                grid = [[rng.random() for _ in range(g)] for _ in range(g)]
                a, b, c = [metric(grid) for metric in METRICS]
                for value in (a, b, c):
                    self.assertGreaterEqual(value, 0)
                    self.assertLessEqual(value, 1)
                self.assertLessEqual(abs(a-b), c+1e-14)
                self.assertLessEqual(c, a+b+1e-14)

    def test_upper_bounds_are_attainable(self):
        for g in (2, 4, 8):
            lr_max = [[float(j >= g//2) for j in range(g)] for _ in range(g)]
            tb_max = [[float(i >= g//2)]*g for i in range(g)]
            self.assertEqual(sf.reflection_asymmetry_lr(lr_max), 1)
            self.assertEqual(sf.reflection_asymmetry_tb(tb_max), 1)
            self.assertEqual(sf.rot180_asymmetry(lr_max), 1)

    def test_one_cell_is_always_zero(self):
        for value in (0, .15, .5, 1):
            self.assertEqual([f([[value]]) for f in METRICS], [0, 0, 0])

    def test_progressive_defect(self):
        for delta, grid in progressive_defects().items():
            for metric in METRICS:
                self.assertAlmostEqual(metric(grid), 2*delta/16)

    def test_order_and_lengths(self):
        grids = {g: [[.5]*g for _ in range(g)] for g in (1, 2, 4, 8)}
        features = sf.compute_nine_asymmetry_features(grids[2], grids[4], grids[8])
        self.assertEqual(list(features), [f"{key}_{g}x{g}" for g in (2, 4, 8)
                                          for key in ("lr", "tb", "rot180")])
        entropy = flatten_entropy_features(grids)
        self.assertEqual((len(entropy), len(features), len(entropy+list(features.values()))),
                         (85, 9, 94))


class ImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.images = generate_images()
        cls.pyramids = {name: compute_entropy_pyramid(image)
                        for name, image in cls.images.items()}

    def test_image_symmetry_properties_all_scales(self):
        expected_zero = [(True, True, True), (True, False, False),
                         (False, True, False), (False, False, True),
                         (False, False, False), (False, False, False)]
        for (name, grids), zeros in zip(self.pyramids.items(), expected_zero):
            for g in (2, 4, 8):
                for metric, zero in zip(METRICS, zeros):
                    with self.subTest(image=name, scale=g, metric=metric.__name__):
                        value = metric(grids[g])
                        if zero:
                            self.assertAlmostEqual(value, 0, places=14)
                        else:
                            self.assertGreater(value, 1e-6)

    def test_entropy_reference_and_trace(self):
        uniform = entropy_details(Image.new("L", (32, 32), 80))
        self.assertEqual(uniform["normalized_entropy"], 0)
        all_levels = Image.frombytes("L", (16, 16), bytes(range(256)))
        for bins in (2, 7, 16, 30, 256):
            details = entropy_details(all_levels, bins)
            self.assertEqual(sum(details["counts"]), 256)
            self.assertAlmostEqual(sum(details["probabilities"]), 1)
            self.assertAlmostEqual(-sum(details["p_log2_p"]), details["entropy_bits"])
            self.assertTrue(0 <= details["normalized_entropy"] <= 1)
        self.assertEqual(entropy_details(all_levels, 16)["normalized_entropy"], 1)

    def test_invalid_bins_and_image_dimensions(self):
        for bins in (0, 1, 257, 16.5, True):
            with self.assertRaisesRegex(ValueError, "bins"):
                entropy_details(Image.new("L", (8, 8)), bins)
        for shape in ((7, 8), (8, 9), (0, 0)):
            with self.assertRaises(ValueError):
                compute_entropy_pyramid(Image.new("L", shape))
        self.assertEqual(len(flatten_entropy_features(
            compute_entropy_pyramid(Image.new("RGB", (16, 24))))), 85)

    def test_scale_defect_matches_analytic_binary_entropy(self):
        base = self.pyramids["image_07_scale_baseline"]
        defect = self.pyramids["image_08_scale_defect"]
        values = []
        for g in (2, 4, 8):
            for metric in METRICS:
                self.assertEqual(metric(base[g]), 0)
            # One 32x32 tile is replaced within a (256/g)^2 cell.
            fraction = (g/8)**2
            p = (1+fraction)/2
            h = -sum(q*math.log2(q) for q in (p, 1-p) if q)
            expected = 2*(1-h)/(4*g*g)
            values.append(sf.reflection_asymmetry_lr(defect[g]))
            for metric in METRICS:
                self.assertAlmostEqual(metric(defect[g]), expected, places=14)
        self.assertLess(values[0], values[1])
        self.assertLess(values[1], values[2])

    def test_scale_need_not_be_monotonic(self):
        grids = self.pyramids["image_06_symmetric_plus_defect"]
        self.assertGreater(sf.reflection_asymmetry_lr(grids[4]),
                           sf.reflection_asymmetry_lr(grids[8]))

    def test_equal_entropy_different_pixels(self):
        name = "image_09_equal_entropy_different_structure"
        image = self.images[name]
        for transpose in (Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM,
                          Image.Transpose.ROTATE_180):
            self.assertNotEqual(image.tobytes(), image.transpose(transpose).tobytes())
        for g in (2, 4, 8):
            for metric in METRICS:
                self.assertEqual(metric(self.pyramids[name][g]), 0)

    def test_reproducibility(self):
        repeated = generate_images()
        for name, image in self.images.items():
            self.assertEqual(image.tobytes(), repeated[name].tobytes())

    def test_all_descriptors_and_alternative_bins(self):
        for bins in (7, 16, 256):
            for image in self.images.values():
                grids = compute_entropy_pyramid(image, bins)
                entropies = flatten_entropy_features(grids)
                features = sf.compute_nine_asymmetry_features(grids[2], grids[4], grids[8])
                descriptor = entropies + list(features.values())
                self.assertEqual((len(entropies), len(features), len(descriptor)), (85, 9, 94))
                self.assertTrue(all(0 <= value <= 1 for value in descriptor))


class EndToEndTests(unittest.TestCase):
    def test_cli_and_generated_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            result = subprocess.run([sys.executable, str(ROOT / "run_demo.py"),
                                     "--bins", "16", "--output", str(out)],
                                    capture_output=True, text=True, check=True)
            self.assertIn("Saved 10 matrix cases and 9 images", result.stdout)
            payload = json.loads((out / "results.json").read_text())
            self.assertEqual(len(payload["images"]), 9)
            for result in payload["images"].values():
                self.assertEqual(len(result["entropy_features"]), 85)
                self.assertEqual(len(result["asymmetry_features"]), 9)
                self.assertEqual(result["combined_descriptor"], result["entropy_features"]
                                 + list(result["asymmetry_features"].values()))
            pngs = list(out.rglob("*.png"))
            self.assertEqual(len(pngs), 304)
            for path in pngs:
                with Image.open(path) as image:
                    image.verify()
            self.assertTrue((out / "scale.csv").exists())
            self.assertTrue((out / "matrices" / "progressive_defect.csv").exists())

    def test_cli_help_and_invalid_bins(self):
        script = str(ROOT / "run_demo.py")
        result = subprocess.run([sys.executable, script, "--help"], capture_output=True)
        self.assertEqual(result.returncode, 0)
        result = subprocess.run([sys.executable, script, "--bins", "1"], capture_output=True)
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
