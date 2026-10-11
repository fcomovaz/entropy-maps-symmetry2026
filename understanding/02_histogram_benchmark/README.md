# Histogram benchmark: exact grayscale uint8 histograms

## Guía rápida para retomar

Este laboratorio compara **ocho implementaciones exactas** sobre 35 entradas
(cinco tamaños × siete tipos). Empezar por `histogram_direct`, después revisar
`METHODS`, `prepare_inputs`, `validate_methods` y `benchmark_kernel` dentro de
`histogram_benchmark.py`. `benchmark_end_to_end` mide además recorte → bytes →
histograma → entropía; no incluye una CNN.

Desde la raíz en PowerShell:

```powershell
.\code\.venv\Scripts\python.exe .\code\understanding\02_histogram_benchmark\histogram_benchmark.py --help
# Protocolo completo, escribe en results/ de este laboratorio:
.\code\.venv\Scripts\python.exe .\code\understanding\02_histogram_benchmark\histogram_benchmark.py
```

Usar preferentemente la [copia de ejecución](../../README.md#repetir-01-03-sin-sobrescribir-evidencia)
para preservar los resultados históricos. Para una corrida corta agregar
`--target-pixels-per-run 10000`; mantiene las 35 entradas y cinco rondas, reduce
repeticiones del kernel y conserva la prueba end-to-end. No usar sus tiempos
como sustituto del protocolo completo (1,000,000 píxeles objetivo por ronda).

La entrada real se toma primero de `../01_patch_partition/assets/derived_grayscale_500x333.png`;
si falta, usa el JPEG 512×512 y lo convierte a grayscale. Ese fallback cambia
los patches centrales: para reproducir las tablas, conservar el PNG derivado
o ejecutar primero 01 con el mismo JPEG. Si faltan ambas imágenes, el script falla.

Resultado esperado: `Validated 35 inputs and wrote 280 kernel rows`.
Abrir `results/summary_table.md` y consultar `benchmark.csv` para detalle.
Cada ejecución sobrescribe `benchmark.csv`, `end_to_end.csv`,
`summary_table.md` y `run_metadata.json`. No crea automáticamente los snapshots
`benchmark_run1.csv`/`benchmark_run2.csv`: son copias históricas conservadas.
Las cifras y conclusiones siguientes describen esas corridas históricas; el
README no se regenera al ejecutar el script.

## Research question and answer

This experiment asks whether an exact 256-level grayscale histogram justifies an algorithm more elaborate than:

```python
hist = [0] * 256
for pixel in pixels:
    hist[pixel] += 1
```

**Answer: no for the current non-overlapping-patch design.** Every one of the `N` pixels must contribute once, so an exact histogram has a lower bound of Ω(N) work. A direct-address array performs one indexed increment per pixel and is Θ(N + 256), including initialization/output of the 256 bins. The tested alternatives do not improve this algorithmic complexity. Some change CPython constants; none eliminates the required per-pixel contribution in a transferable scalar C implementation.

This is an understanding experiment only. It does not modify the main pipeline.

## Scope and dependencies

The implementation uses the Python 3.11 standard library plus the Pillow already present in `code/.venv`. It does not use NumPy, OpenCV, scikit-image, PyTorch, pandas, numba, Cython, or a new dependency. The real-image cases use the grayscale image produced by `code/understanding/01_patch_partition`; file reading, grayscale conversion, cropping, and pixel preparation occur before kernel timing.

Run from the repository root:

```powershell
code\.venv\Scripts\python.exe code\understanding\02_histogram_benchmark\histogram_benchmark.py
```

The seed is fixed at `20260928`. The script fails immediately if any method differs from H1 at any of the 256 positions or if `sum(histogram) != number_of_pixels`.

## Experimental design

The matrix contains 5 square sizes (16, 32, 64, 128, 256) and 7 inputs: uniform, two-level, gradient, checkerboard, low-entropy random, full-range random, and a centered real-image patch. Pixel buffers and Pillow images are prepared before timing.

Each configuration has 3 warm-up calls and 5 independent timed runs. Each run repeats enough calls to process approximately 1,000,000 pixels, subject to 7–2,000 repetitions. Timing uses `time.perf_counter_ns()`. The CSV records median, minimum, p25, p75, maximum, IQR, ns/pixel, throughput, and speedup against the matching H1 input and size. Thus the reported dispersion is between independent run medians, not between individual pixels.

Methods:

- H1 direct-address `list`: fundamental scalar baseline.
- H1-local: same algorithm with a local alias, included only to test a modest CPython detail. The alias has no meaningful C interpretation.
- H2 `array.array('I')`: fixed-width contiguous counters closer in representation to C, but operations still pass through Python.
- H3 `collections.Counter`: hash/mapping comparison. Its loop is implemented natively in CPython; this is not evidence that hashing is preferable on a microcontroller.
- H4 `defaultdict`: hash/mapping negative control with a Python loop.
- H5 unroll ×4 and ×8: simple manual unrolling with exact tail handling.
- H6 Pillow: **library/C implementation baseline**. It shows the effect of native compiled code but is not a directly transferable embedded candidate.

## Observed CPython results

The following is the median of each method's ns/pixel and matching-case speedup across all 35 input/size cases from the canonical second full run. Aggregation is for readability; all 280 configuration rows and their dispersion are in `results/benchmark.csv`.

| Method | Median ns/pixel across all cases | Median speedup vs H1 | Interpretation |
|---|---:|---:|---|
| H1 direct-address list | 53.699 | 1.000× | scalar reference |
| H1-local direct-address list | 57.688 | 0.980× | no consistent useful gain |
| H2 array.array | 92.270 | 0.608× | closer storage model, slower Python access |
| H3 collections.Counter | 45.141 | 1.481× | CPython native implementation; hashing is not the cause of an algorithmic win |
| H4 defaultdict | 97.987 | 0.590× | mapping negative control |
| H5 unroll ×4 | 74.630 | 0.748× | Python indexing overhead outweighs unrolling |
| H5 unroll ×8 | 71.924 | 0.765× | same conclusion |
| H6 Pillow histogram [library/C implementation baseline] | 1.971 | 29.408× | native library baseline only |

These timings rank implementations in this CPython environment, not C algorithms. `Counter` is faster than the Python H1 loop because it performs its counting loop in compiled interpreter/library code. Conversely, `array.array` stores compact C integers but each Python-level indexed update is expensive. Neither observation predicts that a hash table beats a direct array in C. Manual unrolling changes constants only; whether it helps C depends on compiler, target, memory system, and measurement on that target.

The small end-to-end check processes 16 real 64×64 patches (65,536 pixels total) per repetition. It includes Pillow crop, `tobytes`, H1 histogram, and entropy. The canonical run measured 5,823,566 ns median for that pipeline and 4,776,078 ns for the prepared-pixel histogram-only loop, an illustrative 82.013% ratio. This is deliberately a small CPython decomposition, not an embedded latency claim; crop and entropy implementations would differ in C.

## Reproducibility check

Two complete executions were retained as `benchmark_run1.csv` and `benchmark_run2.csv`; the canonical `benchmark.csv` equals run 2. Median ns/pixel changes from run 1 to run 2 were −5.3% for H1, −1.4% for H1-local, −1.9% for `array.array`, +2.5% for `Counter`, +1.1% for `defaultdict`, +1.2% for unroll ×4, and −3.9% for unroll ×8. Pillow changed +18.9% because its individual calls are extremely short and therefore more sensitive to host noise. The substantive ranking and algorithmic conclusion did not change. Exact machine, Python, Pillow, timer, seed, and repetition settings are recorded in `results/run_metadata.json`.

## Algorithmic interpretation

### Exact histogram lower bound

For an arbitrary grayscale input, ignoring a pixel can change one bin without the algorithm observing it. Therefore every pixel must be read at least once: Ω(N). The direct-address implementation reaches Θ(N + 256), uses constant-size state, avoids hashing, and has predictable memory access. Loop unrolling can reduce branch/index overhead or expose instruction-level parallelism, but it remains Θ(N) and is a compiler/target-level constant-factor question.

A sparse mapping may avoid clearing all 256 counters when very few levels occur, but it adds hashing, keys, collision handling, and less predictable memory access. Once an exact 256-position vector is required for entropy or comparison, materializing that output still touches 256 positions. For a fixed uint8 alphabet, 256 is small and known in advance; a mapping is not algorithmically attractive.

### Non-overlapping grid

If selected patches do not overlap, their total histogram work is proportional to the number of selected pixels:

```text
T_histograms = Θ(sum of pixels in selected patches) + Θ(256 × patch count)
```

Each real pixel is counted at most once in the grid. There is no duplicated spatial work to reuse, so rolling or integral techniques do not improve the asymptotic pixel-counting cost. The useful system-level lever is selecting fewer pixels/patches while accounting for selector cost, not replacing the direct histogram counter.

### Sliding or overlapping windows

Overlapping windows create repeated work and would change the decision. A rolling histogram can subtract the departing column and add the entering column, reducing a horizontal window shift from Θ(window area) to Θ(window height), while retaining 256 counters and enough image/column state. Two-dimensional traversal also needs careful row-transition handling.

A 256-bin integral histogram enables region queries by combining prefix values, but stores 256 counters per image position: Θ(256 × width × height) memory. Even with 16-bit counters, a 320×240 frame would require about 39.3 MB, generally unreasonable for a small microcontroller and potentially subject to overflow. Tiled, compressed, or partial variants trade simplicity for memory and control overhead. These approaches are justified only if future experiments introduce many overlapping window queries; they are not implemented here.

## Embedded counter memory and overflow

The relevant C memory is the counter array itself, not `sys.getsizeof()` of a CPython list. Worst-case sizing assumes every pixel in a patch has the same value, so one bin can reach `N`.

| Counter type | 256-bin memory | Maximum safe pixels in one histogram | Largest safe square patch |
|---|---:|---:|---:|
| `uint8_t` | 256 B | 255 | 15×15 (225 pixels) |
| `uint16_t` | 512 B | 65,535 | 255×255 (65,025 pixels) |
| `uint32_t` | 1,024 B | 4,294,967,295 | 65,535×65,535 |

Therefore the requested 16×16 case already needs more than `uint8_t` in the worst case, and a uniform 256×256 patch overflows `uint16_t` by one. `uint16_t` is sufficient through 128×128 and for 255×255, while `uint32_t` safely covers all benchmark sizes. A narrower counter is valid only when patch size or a proven per-bin bound prevents overflow; saturation would make the histogram inexact.

Direct counters can be static or stack allocated, have no required dynamic allocation, and need no hashing. A straightforward C version is easy to audit and offers predictable timing. Target-specific improvements worth leaving to a later C microbenchmark include counter width, alignment, clearing strategy, compiler optimization, and possibly compiler-generated unrolling—not a new histogram algorithm.

## Output files

```text
results/benchmark.csv          canonical full kernel results (run 2)
results/benchmark_run1.csv     first full reproducibility run
results/benchmark_run2.csv     second full reproducibility run
results/end_to_end.csv         small pipeline decomposition (run 2)
results/run_metadata.json      environment and protocol
results/summary_table.md       generated readable aggregate (run 2)
```

## Decision for the paper

**B + C.** For the present non-overlapping patches, the direct array of 256 counters is already essentially the right algorithm. There is no demonstrated algorithmic optimization beyond one required contribution per selected pixel, and continuing to optimize histogram construction in Python is unlikely to add scientific value to the paper. We should **discard this as a paper contribution and redirect effort to region selection, total selected pixels, and the complete `T_total = T_selector + T_model` trade-off.**

Option C applies only narrowly to implementation constants: before deployment, run a small C/C++ microbenchmark on the actual microcontroller to choose counter width and confirm compiler/target effects such as clearing and unrolling. That engineering check should not reopen the claim that a different exact-histogram algorithm is needed. Revisit rolling or integral histograms only if the project changes to many overlapping windows.
