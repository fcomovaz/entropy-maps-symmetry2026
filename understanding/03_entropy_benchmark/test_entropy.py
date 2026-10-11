"""Run with the project Python; no dependency beyond the benchmark itself."""
import math
import random
import unittest

import entropy_benchmark as eb


class EntropyTests(unittest.TestCase):
    def test_analytical_and_fixed_bounds(self):
        for n in (256,1024,4096,16384):
            lc,lt = eb.count_lut(n),eb.term_lut(n)
            histograms = [[n]+[0]*255,[n//2,n//2]+[0]*254,[n//256]*256]
            for h,expected in zip(histograms,(0,1,8)):
                self.assertAlmostEqual(eb.e1(h,n),expected,places=12)
                self.assertAlmostEqual(eb.e2(h,n,math.log2(n)),expected,places=12)
                self.assertAlmostEqual(eb.e3(h,n,math.log2(n),lc),expected,places=12)
                self.assertAlmostEqual(eb.e4(h,lt),expected,places=12)
            rng = random.Random(n)
            for _ in range(25):
                h = eb.histogram(bytes(rng.randrange(256) for _ in range(n)))
                for bits in (16,30):
                    table = [math.floor(x*(1<<bits)+.5) for x in lt]
                    value = eb.fixed_entropy(h,table,bits)
                    bound = sum(c>0 for c in h)/(2*(1<<bits))
                    self.assertLessEqual(abs(value-eb.e1(h,n)),bound+1e-12)
                    self.assertLess(max(table),1<<(16 if bits==16 else 32))
                    self.assertLess(sum(table[c] for c in h),1<<(32 if bits==16 else 64))

    def test_count_lut_reuse_non_power_of_two(self):
        table = eb.count_lut(16384)
        for n in (1,3,37,255,676,16383):
            h = eb.histogram(bytes(i%37 for i in range(n)))
            self.assertLessEqual(abs(eb.e3(h,n,math.log2(n),table)-eb.e1(h,n)),1e-12)
            self.assertLessEqual(abs(eb.e4(h,eb.term_lut(n))-eb.e1(h,n)),1e-12)

    def test_sparse_and_histogram_remainders(self):
        for pixels in (b'',b'\x00',b'\x01\x00\x01',bytes(range(256))*3+b'\xff'):
            hist = eb.histogram(pixels)
            state = eb.histogram_sparse(pixels)
            self.assertEqual(hist,state[0])
            self.assertEqual(len(state[1]),sum(c>0 for c in hist))
            self.assertEqual(state[1],list(dict.fromkeys(pixels)))
            for fn in (eb.histogram_array,eb.histogram_unroll4,eb.histogram_counter):
                self.assertEqual(fn(pixels),hist)

    def test_ranking_detects_reversal_and_ties(self):
        metric = eb.ranking_metrics([0,1,2],[2,1,0])
        self.assertEqual(metric['strict_inversions'],3)
        self.assertEqual(metric['top_1'],0)
        self.assertEqual(eb.ranking_metrics([0,1],[0,0])['collapsed_pairs'],1)
        self.assertEqual(eb.ranking_metrics([1,1],[1,2])['broken_reference_ties'],1)

    def test_real_patches_match_saved_experiment_01_samples(self):
        from PIL import Image
        for size in (32,64,128):
            _,ranking,_ = eb.datasets(size)
            real = {p.name:p.pixels for p in ranking if p.group=='real'}
            folder = eb.experiment01_root()/f'outputs/A_divisible_patch_{size}/sample_patches'
            paths = list(folder.glob('patch_*.png'))
            self.assertTrue(paths)
            for path in paths:
                index = int(path.stem.split('_')[1])
                with Image.open(path) as image:
                    self.assertEqual(real[f'real_{index:04d}'],image.tobytes())


if __name__ == '__main__':
    unittest.main(verbosity=2)
