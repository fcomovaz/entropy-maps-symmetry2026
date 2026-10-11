"""Scalar Shannon entropy study. Standard library; Pillow only decodes images.

Kernels assume valid nonnegative counts summing to N > 0. Validation and image
decoding are outside timing. All generated artifacts stay beside this script.
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys
import time
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[2]
SIZES = (16, 32, 64, 128)
SEED = 20260928
TOL = 1e-12


def experiment01_root() -> Path:
    """Locate sibling experiment 01 without depending on the working directory."""
    candidate = ROOT.parent/'01_patch_partition'
    if (candidate/'outputs/A_divisible_patch_32/partition_input.png').exists():
        return candidate
    raise FileNotFoundError(f'Experiment 01 input missing from {candidate}')


def histogram(pixels: bytes) -> list[int]:
    """H1 from experiment 02: 256 direct-address counters."""
    hist = [0] * 256
    for pixel in pixels:
        hist[pixel] += 1
    return hist


def histogram_array(pixels: bytes) -> list[int]:
    """H2 from experiment 02, including its conversion back to list."""
    hist = array('I', [0]) * 256
    for pixel in pixels:
        hist[pixel] += 1
    return list(hist)


def histogram_unroll4(pixels: bytes) -> list[int]:
    """H5 from experiment 02, four scalar increments and a remainder loop."""
    hist = [0] * 256
    stop = len(pixels) - len(pixels) % 4
    for i in range(0, stop, 4):
        hist[pixels[i]] += 1
        hist[pixels[i+1]] += 1
        hist[pixels[i+2]] += 1
        hist[pixels[i+3]] += 1
    for i in range(stop, len(pixels)):
        hist[pixels[i]] += 1
    return hist


def histogram_counter(pixels: bytes) -> list[int]:
    """H3 from experiment 02; native Counter hash baseline, not MCU evidence."""
    counts = Counter(pixels)
    return [counts[i] for i in range(256)]


def histogram_sparse(pixels: bytes) -> tuple[list[int], list[int]]:
    """E5: record first encounters in deterministic pixel order."""
    hist = [0] * 256
    used = []
    for pixel in pixels:
        if hist[pixel] == 0:
            used.append(pixel)
        hist[pixel] += 1
    return hist, used


def e1(hist: Sequence[int], n: int) -> float:
    """Reference H = -sum(p log2 p), skipping zeros."""
    result = 0.0
    for c in hist:
        if c:
            p = c / n
            result -= p * math.log2(p)
    return result


def e2(hist: Sequence[int], n: int, logn: float) -> float:
    """Count identity, with log2(N) initialized once per patch size."""
    total = 0.0
    for c in hist:
        if c:
            total += c * math.log2(c)
    return logn - total / n


def count_lut(n: int) -> list[float]:
    """N+1 entries; reusable for any smaller N with correct normalization."""
    return [0.0] + [c * math.log2(c) for c in range(1, n+1)]


def term_lut(n: int) -> list[float]:
    """N-specific complete Shannon terms, including zero convention."""
    return [0.0] + [-(c/n) * math.log2(c/n) for c in range(1, n+1)]


def e3(hist: Sequence[int], n: int, logn: float, lut: Sequence[float]) -> float:
    """Branch-free traversal of all 256 counts."""
    total = 0.0
    for c in hist:
        total += lut[c]
    return logn - total / n


def e4(hist: Sequence[int], lut: Sequence[float]) -> float:
    """Complete-term LUT, no normalization in the patch loop."""
    total = 0.0
    for c in hist:
        total += lut[c]
    return total


def sparse_entropy(state: tuple[list[int], list[int]], n: int, logn: float,
                   lut: Sequence[float], method: str) -> float:
    """Compute E1/E2/E3/E4 over occupied indices, without repacking counts."""
    hist, used = state
    total = 0.0
    for i in used:
        c = hist[i]
        if method == 'E1':
            p = c/n
            total -= p * math.log2(p)
        elif method == 'E2':
            total += c * math.log2(c)
        else:
            total += lut[c]
    return logn - total/n if method in ('E2', 'E3') else total


def fixed_entropy(hist: Sequence[int], lut: Sequence[int], bits: int) -> float:
    """Integer accumulation; convert only once for comparison/reporting."""
    total = 0
    for c in hist:
        total += lut[c]
    return total / (1 << bits)


@dataclass
class Patch:
    name: str
    group: str
    pixels: bytes
    hist: list[int]


def patch(name: str, group: str, pixels: bytes) -> Patch:
    return Patch(name, group, pixels, histogram(pixels))


def datasets(size: int) -> tuple[list[Patch], list[Patch], dict]:
    """Synthetic pixels plus exact row-major tiles of experiment-01 input."""
    from PIL import Image
    n = size*size
    rng = random.Random(SEED + size)
    synthetic = {
        'single_level': bytes([127])*n,
        'uniform_256': bytes(range(256))*(n//256),
        'two_levels': bytes([32, 224])*(n//2),
        'few_levels': bytes(rng.choices([24, 25, 26, 27], [90, 5, 3, 2], k=n)),
        'gradient': bytes(round(255*x/(size-1)) for y in range(size) for x in range(size)),
        'approximately_uniform_256': bytes((i % 256 if i % 31 else 0) for i in range(n)),
        'random': bytes(rng.randrange(256) for _ in range(n)),
    }
    bench = [patch(k, k, v) for k, v in synthetic.items()]
    source = experiment01_root()/'outputs/A_divisible_patch_32/partition_input.png'
    with Image.open(source) as image:
        if image.mode != 'L':
            raise ValueError('Expected existing grayscale L image')
        width, height = image.size
        pixels = image.tobytes()
    real = []
    boxes = []
    for y in range(0, height-size+1, size):
        for x in range(0, width-size+1, size):
            data = b''.join(pixels[row*width+x:row*width+x+size] for row in range(y, y+size))
            real.append(patch(f'real_{len(real):04d}', 'real', data))
            boxes.append((x, y, x+size, y+size))
    # Verify existing experiment-01 coordinates, rather than silently redefining its tiling.
    if size != 16:
        coords = experiment01_root()/f'outputs/A_divisible_patch_{size}/patch_coordinates.csv'
        with coords.open(newline='', encoding='utf-8') as f:
            recorded = [tuple(int(r[k]) for k in ('x0','y0','x1','y1')) for r in csv.DictReader(f)]
        assert boxes == recorded
    indices = sorted({round(i*(len(real)-1)/7) for i in range(8)})
    bench += [real[i] for i in indices]
    ranking = list(bench[:7]) + real
    # Diverse alphabets and very close high-entropy pairs, not only easy rankings.
    for j in range(96):
        alphabet = rng.sample(range(256), rng.choice((2,4,8,16,64,256)))
        data = bytes(rng.choices(alphabet, weights=[rng.random()**3+.0001 for _ in alphabet], k=n))
        ranking.append(patch(f'mixed_{j:03d}', 'mixed', data))
    for j in range(96):
        counts = [n//256]*256
        for _ in range(j+1):
            a,b = rng.sample(range(256), 2)
            if counts[a]:
                counts[a] -= 1
                counts[b] += 1
        data = b''.join(bytes([i])*c for i,c in enumerate(counts))
        ranking.append(patch(f'near_tie_{j:03d}', 'near_tie', data))
    return bench, ranking, {'source':str(source.relative_to(REPOSITORY_ROOT)),
        'sha256':hashlib.sha256(source.read_bytes()).hexdigest(), 'real_patches':len(real)}


def measure(jobs: dict[str, Callable], repeats: int, rounds: int, rng: random.Random) -> dict:
    """Warm up each kernel, shuffle order per round, retain all batch timings."""
    samples = {name: [] for name in jobs}
    for fn in jobs.values():
        for _ in range(5):
            fn()
    for _ in range(rounds):
        order = list(jobs)
        rng.shuffle(order)
        for name in order:
            fn = jobs[name]
            start = time.perf_counter_ns()
            for _ in range(repeats):
                fn()
            samples[name].append((time.perf_counter_ns()-start)/repeats)
    return {name: {'median_ns':statistics.median(v), 'min_ns':min(v), 'max_ns':max(v),
                   'samples_ns':json.dumps(v), 'repetitions':repeats, 'rounds':rounds}
            for name,v in samples.items()}


def write_csv(name: str, rows: list[dict]) -> None:
    """Write reproducible tabular artifacts without pandas."""
    with (ROOT/'results'/name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ranking_metrics(ref: list[float], values: list[float]) -> dict:
    """Strict inversions and ties separately, reference equivalence tolerance 1e-12."""
    inversions = collapsed = broken = comparable = 0
    for i in range(len(ref)):
        for j in range(i):
            a,b = ref[i]-ref[j], values[i]-values[j]
            if abs(a) <= TOL:
                broken += abs(b) > TOL
            else:
                comparable += 1
                collapsed += abs(b) <= TOL
                inversions += abs(b) > TOL and a*b < 0
    order = sorted(range(len(ref)), key=lambda i: (-ref[i], i))
    approx = sorted(range(len(ref)), key=lambda i: (-values[i], i))
    out = dict(strict_inversions=inversions, comparable_pairs=comparable,
               collapsed_pairs=collapsed, broken_reference_ties=broken,
               position_discrepancies=sum(a!=b for a,b in zip(order,approx)))
    for k in sorted({1,5,10,max(1,len(ref)//10),max(1,len(ref)//4)}):
        out[f'top_{k}'] = len(set(order[:k]) & set(approx[:k]))/k
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--entropy-repeats', type=int, default=300)
    parser.add_argument('--target-pixels', type=int, default=300000)
    args = parser.parse_args()
    if min(args.rounds, args.entropy_repeats, args.target_pixels) < 1:
        parser.error('All timing parameters must be positive')
    (ROOT/'results').mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    entropy_rows, pipeline_rows, hist_rows, lut_rows = [], [], [], []
    errors, ranks, scores, manifest = [], [], [], []
    max_exact_error = 0.0
    validations = 0
    hist_methods = {'H1':histogram, 'H2_array':histogram_array,
                    'H5_unroll4':histogram_unroll4, 'H3_Counter':histogram_counter}
    for size in SIZES:
        n, logn = size*size, math.log2(size*size)
        bench, ranking, provenance = datasets(size)
        lc, lt = count_lut(n), term_lut(n)
        fixed = {bits:[math.floor(x*(1<<bits)+.5) for x in lt] for bits in (16,30)}
        build = measure({'E3':lambda:count_lut(n), 'E4':lambda:term_lut(n)}, 20, args.rounds, rng)
        for name,lut in [('E3',lc),('E4',lt)]:
            lut_rows.append(dict(size=size,N=n,method=name,entries=n+1,
                c_float64_bytes=(n+1)*8,c_float32_bytes=(n+1)*4,
                c_uint32_bytes=(n+1)*4,c_uint16_bytes=(n+1)*2,
                max_entry=max(lut),python_deep_bytes=sys.getsizeof(lut)+sum(sys.getsizeof(x) for x in lut),
                uint16_format='Q0.16 terms' if name=='E4' else ('integer only, fractional precision unspecified' if max(lut)<=65535 else 'range overflow even Q0'),
                **build[name]))
        # Essential analytical cases plus all real/synthetic ranking cases.
        for p in ranking:
            h = p.hist
            assert len(h)==256 and sum(h)==n and min(h)>=0
            ref = e1(h,n)
            exact = {'E2':e2(h,n,logn), 'E3':e3(h,n,logn,lc),'E4':e4(h,lt)}
            for method,value in exact.items():
                error = abs(value-ref)
                assert error <= TOL, (size,p.name,method,error)
                max_exact_error = max(max_exact_error,error)
                validations += 1
            if p.name in ('single_level','two_levels','uniform_256'):
                assert abs(ref-{'single_level':0,'two_levels':1,'uniform_256':8}[p.name]) <= TOL
            manifest.append(dict(size=size,patch=p.name,group=p.group,N=n,occupied=sum(c>0 for c in h),
                pixel_sha256=hashlib.sha256(p.pixels).hexdigest()))
            scores.append(dict(size=size,patch=p.name,group=p.group,E1=ref,**exact,
                Q16=fixed_entropy(h,fixed[16],16),Q30=fixed_entropy(h,fixed[30],30)))
        for group in ('all','real','near_tie'):
            selected = [r for r in scores if r['size']==size and (group=='all' or r['group']==group)]
            reference = [r['E1'] for r in selected]
            for method in ('E2','E3','E4','Q16','Q30'):
                values = [r[method] for r in selected]
                absolute = [abs(a-b) for a,b in zip(reference,values)]
                relative = [abs(a-b)/abs(a) for a,b in zip(reference,values) if abs(a)>TOL]
                errors.append(dict(size=size,group=group,method=method,patches=len(values),
                    max_absolute_error=max(absolute),mean_absolute_error=statistics.mean(absolute),
                    max_relative_error=max(relative,default=0),mean_relative_error=statistics.mean(relative) if relative else 0))
                metric = ranking_metrics(reference,values)
                tops = {k:v for k,v in metric.items() if k.startswith('top_')}
                base = {k:v for k,v in metric.items() if not k.startswith('top_')}
                for key,agreement in tops.items():
                    ranks.append(dict(size=size,group=group,method=method,patches=len(values),**base,
                                      k=int(key[4:]),top_k_agreement=agreement))
        for p in bench:
            hist, state = p.hist, histogram_sparse(p.pixels)
            assert state[0] == hist
            jobs = {'E1':lambda:e1(hist,n),'E2':lambda:e2(hist,n,logn),
                    'E3':lambda:e3(hist,n,logn,lc),'E4':lambda:e4(hist,lt)}
            # Specialize outside timed loops; no method-dispatch branch per occupied bin.
            sparse_jobs = {
                'E1':lambda:e1([state[0][i] for i in state[1]],n),
                'E2':lambda:e2([state[0][i] for i in state[1]],n,logn),
                'E3':lambda:logn-sum(lc[state[0][i]] for i in state[1])/n,
                'E4':lambda:sum(lt[state[0][i]] for i in state[1])}
            # For E1/E2, materialization is explicitly included in the sparse kernel cost.
            isolated = dict(jobs)
            isolated.update({'E5_'+k:v for k,v in sparse_jobs.items()})
            for name,fn in isolated.items():
                assert abs(fn()-e1(hist,n)) <= TOL, (p.name,name)
            timings = measure(isolated,args.entropy_repeats,args.rounds,rng)
            info = dict(size=size,N=n,patch=p.name,group=p.group,occupied=sum(c>0 for c in hist))
            for name,timing in timings.items():
                entropy_rows.append(dict(**info,method=name,**timing))
            h_jobs = {name:(lambda fn=fn:fn(p.pixels)) for name,fn in hist_methods.items()}
            h_jobs['H1_sparse'] = lambda:histogram_sparse(p.pixels)
            for name,fn in h_jobs.items():
                assert (fn()[0] if name=='H1_sparse' else fn()) == hist
            repetitions = max(20,args.target_pixels//n)
            ht = measure(h_jobs,repetitions,args.rounds,rng)
            for name,timing in ht.items():
                hist_rows.append(dict(**info,method=name,**timing))
            pipelines = {}
            for hname,hfn in hist_methods.items():
                pipelines[hname+'+E1'] = lambda f=hfn:e1(f(p.pixels),n)
                pipelines[hname+'+E2'] = lambda f=hfn:e2(f(p.pixels),n,logn)
                pipelines[hname+'+E3'] = lambda f=hfn:e3(f(p.pixels),n,logn,lc)
                pipelines[hname+'+E4'] = lambda f=hfn:e4(f(p.pixels),lt)
            def sp(method: str) -> float:
                s = histogram_sparse(p.pixels)
                if method=='E1':
                    return e1([s[0][i] for i in s[1]],n)
                if method=='E2':
                    return e2([s[0][i] for i in s[1]],n,logn)
                if method=='E3':
                    return logn-sum(lc[s[0][i]] for i in s[1])/n
                return sum(lt[s[0][i]] for i in s[1])
            for method in jobs:
                pipelines['H1_sparse+'+method] = lambda m=method:sp(m)
            for name,fn in pipelines.items():
                assert abs(fn()-e1(hist,n)) <= TOL, (p.name,name)
                validations += 1
            pt = measure(pipelines,repetitions,args.rounds,rng)
            for name,timing in pt.items():
                hname,ename = name.split('+')
                htime = ht[hname]['median_ns']
                etime = timings[('E5_' if hname=='H1_sparse' else '')+ename]['median_ns']
                pipeline_rows.append(dict(**info,method=name,**timing,histogram_ns=htime,entropy_ns=etime,
                    estimated_histogram_fraction=htime/(htime+etime),estimated_entropy_fraction=etime/(htime+etime),
                    component_sum_ns=htime+etime,residual_ns=timing['median_ns']-htime-etime))
        print(f'Completed {size}x{size}: {len(bench)} timing patches, {len(ranking)} ranking patches',flush=True)
    for filename,rows in [('entropy_benchmark.csv',entropy_rows),('pipeline_benchmark.csv',pipeline_rows),
        ('histogram_benchmark.csv',hist_rows),('lut_memory_build.csv',lut_rows),('errors.csv',errors),
        ('ranking.csv',ranks),('patch_scores.csv',scores),('manifest.csv',manifest)]:
        write_csv(filename,rows)
    metadata = dict(timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),python=sys.version,
        platform=platform.platform(),processor=platform.processor(),seed=SEED,arguments=vars(args),
        warmup_calls=5,timer='perf_counter_ns',clock=vars(time.get_clock_info('perf_counter')),
        source=provenance,experiment_02_results_available=(ROOT.parent/'02_histogram_benchmark/results').exists(),
        exact_tolerance=TOL,max_exact_error=max_exact_error,exact_comparisons=validations,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'results/run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(f'PASS: {validations} comparisons; maximum exact error {max_exact_error:.3g}',flush=True)


if __name__ == '__main__':
    main()
