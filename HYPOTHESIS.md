# HYPOTHESIS.md — Post-Merge Hypothesis Verification & Pre-MCU Viability (Python vs. C99 vs. CNNs)

- **Synthesis Date:** 2026-11-11
- **Evaluated Branch & Revision:** `main` @ `486c111` *(after merging `feat/fast-entropy-opt`: C99 `c_single_pass` kernel, pyramidal NumPy variants `V1/V2`, and fused linear classifier `lr_dot`)*
- **Measured Protocol:** CPU `Intel(R) Core(TM) i7-7700HQ @ 2.80 GHz`, `threads = 1`, `batch_size = 1` (embedded latency) and `batch_size = 32` (throughput), `seed = 42` aligned across classical ML and CNN checkpoints (CUDA reference on NVIDIA GTX 1050 `cu126`).
- **Related Documents:** [README.md](README.md), [RESUME.md](RESUME.md), [c_imp/README.md](c_imp/README.md), [results/c_imp/README.md](results/c_imp/README.md), [results/classification/README.md](results/classification/README.md), and [understanding/13_entropy_maps_optimization/README.md](understanding/13_entropy_maps_optimization/README.md).

> [!NOTE]
> **Exact scope of on-disk artifacts (`486c111`):** Formal post-merge runs on `main` cover `kolektorsdd2` (`normal` and `inverted`) and `mvtec-ad_bottle` (`normal` and `inverted`), along with isolated pure-Python comparisons under `results/comparison_cpu_python/` (Git-ignored). Any reference to `mvtec-ad_tile` or the `--all-objects` Random Forest sweep explicitly originates from the historical `bf4a29f` snapshot documented in [results/classification/README.md](results/classification/README.md).
>
> **Updated canonical 2026-11-26 sweep (full 15-category MVTec-AD + KolektorSDD2 + live ESP32-CAM):** The tables below preserve the `486c111` post-merge baseline snapshot; for complete updated measurements across **all 15 MVTec-AD categories**, KolektorSDD2, C99 `f32`/`Q15` on `16/16` targets, and physical **ESP32-CAM (`COM5`)** runs, see [results/FINAL_PAPER_REPORT.md](results/FINAL_PAPER_REPORT.md) and [results/hw_test/README.md](results/hw_test/README.md) (reproducible end-to-end via [reproduce_paper_pipeline.ps1](reproduce_paper_pipeline.ps1)).

---

## 1. Master Verdict on Research Hypotheses

| Hypothesis | Statement | Post-Merge Verdict (`486c111`) | Key Quantitative Evidence |
| :--- | :--- | :--- | :--- |
| **$H_{\text{sym}}$ (Symmetry–Entropy Synergy)** | Appending the 9 spatial asymmetry residuals ($\mathbf{x}_9$) to the 85 pyramidal Shannon entropies ($\mathbf{x}_{85}$) improves anomaly detection over using entropy alone. | **VERIFIED** | On `kolektorsdd2`, `lr` rises from `F1 = 0.2687` (`ROC-AUC = 0.6179`) on $\mathbf{x}_{85}$ to **`F1 = 0.4265` (`ROC-AUC = 0.8061`)** on $\mathbf{x}_{94}$ ($\mathbf{+0.1579}$ F1 / $\mathbf{+0.1883}$ AUC); on `mvtec-ad_bottle`, `lr` rises from `0.7333` to **`0.8000`** ($\mathbf{+0.0667}$ F1) and `rf` from `0.5263` to **`0.7273`** ($\mathbf{+0.2010}$ F1). *(Historical ref. `bf4a29f` on `mvtec-ad_tile`: `lr` $+0.1315$ F1, `rf` $+0.3887$ F1).* |
| **$H_{\text{inv}}$ (Photometric Invariance $255 - I$)** | On unpadded images, $\mathbf{x}_{94}$ is invariant to polarity inversion $255 - I$, whereas standard CNNs without photometric augmentation degrade. | **VERIFIED** | Exact identity by commutativity of $-\sum p_b \log_2 p_b$. On `mvtec-ad-inverted_bottle`, `lr_dot (x94)` preserves **`F1 = 0.8000, BalAcc = 0.9072`** (`ROC-AUC = 0.9231 → 0.9214`), whereas `mobilenet_like` collapses (`F1 = 0.7586 → 0.3611`, `ROC-AUC = 0.9415 → 0.6739`). On `kolektorsdd2-inverted`, fixed black letterboxing (`pad=0`) introduces a minor controlled drift (`lr_dot` `F1 = 0.4265 → 0.4153`, `ROC-AUC = 0.8061 → 0.8151`), whereas `tiny_cnn` and `torchvision_mobilenet` collapse (`BalAcc ≈ 0.50–0.54`). |
| **$H_{\text{pareto}}$ (CPU Pareto of $\mathbf{x}_{94}$ + Lightweight Classifier vs. CNNs)** | On CPU (`threads=1, batch=1`) for structural defects, $\mathbf{x}_{94}$ + lightweight classifier matches or outperforms lightweight CNNs with orders-of-magnitude fewer parameters and lower latency. | **VERIFIED IN COMPILED REGIME** | With `c_single_pass` + `lr_dot`, the pipeline takes **`0.3002 ms`** (`3,331.1 img/s`) on `mvtec-ad_bottle` and **`0.3799 ms`** (`2,632.6 img/s`) on `kolektorsdd2` (`95` trainable parameters). On `bottle` (`seed=42`) it outperforms **all 7 CNNs** on CPU in latency (`3.05×` vs `ultra_tiny`, **`13.76×`** vs `mobilenet_like`), parameters (`745×` fewer than `mobilenet_like`), and F1 (**`0.8000` vs `0.7586`** for `mobilenet_like`, the only CNN without `val_f1` threshold collapse). **Key nuance:** requires compiled execution (`c_single_pass`); in pure Python it loses to `ultra_tiny`. |
| **$H_{\text{univ}}$ (Universal Superiority on Micro-Cracks & Pure Python)** | $\mathbf{x}_{94}$ also wins on sub-cell micro-cracks and beats CNNs without native compilation. | **REFUTED** | (1) **Physical resolution floor ($16\times16$ px per cell at $8\times8$):** on `kolektorsdd2`, a 1–2 px crack barely alters a 256-pixel histogram, so **all 7 CNNs win in F1** (`tiny_resnet`: **`F1 = 0.7130, ROC-AUC = 0.9213`** vs `lr_dot`: `F1 = 0.4265, ROC-AUC = 0.8061`). (2) **Interpreter overhead floor:** with `numpy_baseline`, `lr_dot` takes `2.807–2.867 ms` (`B=1`), running **`2.36×–2.75×` slower** than `ultra_tiny` (`1.042–1.189 ms`). |
| **$H_{\text{opt-1}}$ (Single-Pass Pyramid in NumPy `V1/V2`)** | Replacing the 85-region loop with a single fine-grid histogram + pyramidal bottom-up sum substantially accelerates $\mathbf{x}_{85}$ in NumPy while maintaining exact parity ($\le 10^{-12}$). | **VERIFIED ON $\mathbf{x}_{85}$** | In Lab 13 (`200` real samples), `V1_numpy_finest_pyramid` achieves **`9.40×`** speedup over `V0` at `B=1` (`0.2371 ms` vs `2.2282 ms`, $p = 4.83 \times 10^{-67}$) and `V2` achieves **`10.49×`** at `B=1`, although `V2` loses to `V1` for batches $\ge 8$ and at `512x512` due to memory pressure on `[N, 85, B]`. Numerical parity $\le 1.89 \times 10^{-15}$. |
| **$H_{\text{opt-2}}$ (C99 Kernel `c_single_pass` + `lr_dot`)** | A single-pass C99 kernel (`entropy_pyramid.c`) plus algebraic folding of `StandardScaler + LR` (`lr_dot`) eliminates Python/sklearn overhead and outperforms all CNNs in CPU latency. | **VERIFIED** | In Lab 13 (`x85`), `V3_c_single_pass` achieves **`92.65×`** over `V0` (`0.02405 ms`). In the full E2E pipeline (`x94 + lr_dot`), `lr_dot` is **`33.0×–50.8×` faster** than `sklearn lr` (`0.0077–0.0081 ms` vs `0.2669–0.3908 ms` in micro-benchmarks) and enables `x94 + lr_dot` to beat `ultra_tiny` (`3.64×` on `kolektorsdd2`, **`2.67×` on `bottle`** in profiling, $p \le 1.19 \times 10^{-66}$). On `x85_only + lr_dot`, `T_total` drops to **`0.0317–0.0326 ms`** (`~31,000 img/s`). |

---

## 2. Post-Merge Quantitative Evidence (`main @ 486c111`, `seed=42`)

### 2.1 Official 100% CPU Comparison: `c_single_pass` + ML vs. 7 CNNs (`threads=1`)

Data extracted directly from [results/comparison_cpu/kolektorsdd2_baseline_vs_cnn_cpu.csv](results/comparison_cpu/kolektorsdd2_baseline_vs_cnn_cpu.csv), [kolektorsdd2-inverted_baseline_vs_cnn_cpu.csv](results/comparison_cpu/kolektorsdd2-inverted_baseline_vs_cnn_cpu.csv), [mvtec-ad_bottle_baseline_vs_cnn_cpu.csv](results/comparison_cpu/mvtec-ad_bottle_baseline_vs_cnn_cpu.csv), and [mvtec-ad-inverted_bottle_baseline_vs_cnn_cpu.csv](results/comparison_cpu/mvtec-ad-inverted_bottle_baseline_vs_cnn_cpu.csv) (`benchmark_cpu_fair.py`, $N=200$ timed iterations per model):

#### A. `kolektorsdd2` (`N_test = 1,004`: `894 good / 110 defect`, CNN `val = 350`)

| Model | Family | Parameters (`model`) | F1 | BalAcc | ROC-AUC | PR-AUC | `T_desc/prep` `B=1` (ms) | `T_model` `B=1` (ms) | **`T_total` `B=1` (ms)** | `Thru B=1` (`img/s`) | **`T_total` `B=32` (ms/img)** | **`F1` Inv.** | `ROC-AUC` Inv. |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`lr_dot (x94)`** | `ml` | **95** | 0.4265 | 0.7452 | 0.8061 | 0.4360 | 0.3713 | **0.0075** | **0.3799** | **2,632.6** | **0.0271** (`36,913`) | 0.4153 | 0.8151 |
| `dt (x94)` | `ml` | 285 | 0.2798 | 0.5992 | 0.5962 | 0.1805 | 0.3713 | 0.1410 | 0.5298 | 1,887.5 | 0.0296 (`33,729`) | 0.3141 | 0.6228 |
| `lr (x94)` | `ml` | 95 | 0.4265 | 0.7452 | 0.8061 | 0.4360 | 0.3713 | 0.2221 | 0.6225 | 1,606.4 | 0.0383 (`26,105`) | 0.4153 | 0.8151 |
| `knn (x94)` | `ml` | 219,114 | 0.1500 | 0.5404 | 0.6652 | 0.3441 | 0.3713 | 0.8336 | 1.2467 | 802.1 | 0.0816 (`12,256`) | 0.1500 | 0.6607 |
| `rf (x94)` *(200t)* | `ml` | 13,208 | 0.4298 | 0.6841 | 0.7904 | 0.4223 | 0.3713 | 10.4926 | 10.9062 | 91.7 | 0.3615 (`2,766`) | 0.4237 | 0.7978 |
| `ultra_tiny` | `cnn` | 1,154 | 0.6304 | 0.7547 | 0.8815 | 0.7055 | 0.0432 | 0.9450 | 0.9919 | 1,008.2 | 0.4466 (`2,239`) | 0.5185 | 0.8179 |
| `ds_cnn` | `cnn` | 12,866 | 0.6563 | 0.7757 | 0.8779 | 0.7144 | 0.0432 | 2.7509 | 2.7948 | 357.8 | 3.5950 (`278`) | 0.5665 | 0.8673 |
| `mobilenet_like` | `cnn` | 70,786 | 0.6250 | 0.7438 | 0.8622 | 0.6846 | 0.0432 | 3.9495 | 4.0077 | 249.5 | 3.8238 (`262`) | 0.4845 | 0.7900 |
| **`tiny_resnet`** | `cnn` | 77,234 | **0.7130** | **0.8515** | **0.9213** | **0.7811** | 0.0432 | 4.2858 | 4.3355 | 230.7 | 3.5943 (`278`) | **0.6509** | **0.9131** |
| `tiny_cnn` | `cnn` | 23,538 | 0.6154 | 0.7324 | 0.8714 | 0.6652 | 0.0432 | 5.5675 | 5.6102 | 178.2 | 6.3075 (`159`) | 0.2098 | 0.8281 |
| `small_cnn` | `cnn` | 97,650 | 0.5965 | 0.7262 | 0.8927 | 0.7112 | 0.0432 | 8.5095 | 8.5604 | 116.8 | 7.6074 (`131`) | 0.5682 | 0.8684 |
| `torchvision_mobilenet` | `cnn` | 1,519,618 | 0.5976 | 0.7199 | 0.7332 | 0.5800 | 0.0432 | 8.6648 | 8.7329 | 114.5 | 2.8878 (`346`) | 0.1975 | 0.5369 |

#### B. `mvtec-ad_bottle` (`N_test = 59`: `46 good / 13 defect`, CNN `val = 35`)

| Model | Family | Parameters (`model`) | F1 | BalAcc | ROC-AUC | PR-AUC | `T_desc/prep` `B=1` (ms) | `T_model` `B=1` (ms) | **`T_total` `B=1` (ms)** | `Thru B=1` (`img/s`) | **`T_total` `B=32` (ms/img)** | **`F1` Inv.** | `ROC-AUC` Inv. |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`lr_dot (x94)`** | `ml` | **95** | **0.8000** | **0.9072** | 0.9231 | **0.8963** | 0.2925 | **0.0078** | **0.3002** | **3,331.1** | **0.0320** (`31,245`) | **0.8000** | 0.9214 |
| `dt (x94)` | `ml` | **15** | 0.5833 | 0.7258 | 0.6614 | 0.5167 | 0.2925 | 0.1357 | 0.4349 | 2,299.6 | 0.0351 (`28,529`) | 0.5833 | 0.6614 |
| `lr (x94)` | `ml` | 95 | **0.8000** | **0.9072** | 0.9231 | **0.8963** | 0.2925 | 0.2221 | 0.5496 | 1,819.5 | 0.0436 (`22,953`) | **0.8000** | 0.9214 |
| `knn (x94)` | `ml` | 21,902 | 0.3750 | 0.6154 | 0.7057 | 0.5907 | 0.2925 | 0.7085 | 1.0939 | 914.2 | 0.0608 (`16,450`) | 0.3750 | 0.7057 |
| `rf (x94)` *(100t)* | `ml` | 3,496 | 0.7273 | 0.7968 | 0.9030 | 0.8266 | 0.2925 | 5.4326 | 5.7947 | 172.6 | 0.2066 (`4,841`) | 0.7273 | 0.9013 |
| `ultra_tiny` | `cnn` | 1,154 | 0.3611 | 0.5000 | 0.5268 | 0.3505 | 0.0400 | 0.8749 | 0.9171 | 1,090.4 | 0.5183 (`1,929`) | 0.4138 | 0.5987 |
| `ds_cnn` | `cnn` | 12,866 | 0.3611 | 0.5000 | 0.5535 | 0.3966 | 0.0400 | 3.2770 | 3.3181 | 301.4 | 3.6253 (`276`) | 0.6190 | 0.9365 |
| **`mobilenet_like`** | `cnn` | 70,786 | 0.7586 | 0.8687 | **0.9415** | 0.8901 | 0.0400 | 4.0806 | 4.1313 | 242.1 | 3.8399 (`260`) | 0.3611 | 0.6739 |
| `tiny_resnet` | `cnn` | 77,234 | 0.3662 | 0.5109 | 0.7391 | 0.6320 | 0.0400 | 5.0154 | 5.0555 | 197.8 | 3.6397 (`275`) | 0.5625 | 0.8127 |
| `tiny_cnn` | `cnn` | 23,538 | 0.3571 | 0.5836 | 0.6472 | 0.4117 | 0.0400 | 6.2127 | 6.2530 | 159.9 | 7.3557 (`136`) | 0.3636 | 0.6187 |
| `small_cnn` | `cnn` | 97,650 | 0.3611 | 0.5000 | 0.6639 | 0.4250 | 0.0400 | 7.8983 | 7.9384 | 126.0 | 8.2924 (`121`) | **0.8000** | **0.9916** |
| `torchvision_mobilenet` | `cnn` | 1,519,618 | 0.3611 | 0.5000 | 0.5000 | 0.2203 | 0.0400 | 9.4212 | 9.4679 | 105.6 | 2.9356 (`341`) | 0.3611 | 0.5109 |

### 2.2 Exact Diagnosis of CNN Collapse on `mvtec-ad_bottle` (`seed=42`)

When aligning classical ML and CNNs on the exact same partition (`seed=42`) on `mvtec-ad_bottle`, the validation-set sample gap explains why CNNs collapse here but not on `kolektorsdd2`:

1. **Critical validation set size (`N_val = 35`: `27 good / 8 defect`) + early `best_epoch` + `val_f1`:**
   - A degenerate all-positive predictor on validation achieves `val F1 = 2 * 8 / (35 + 8) = 16 / 43 = 0.3721`. Projecting that threshold $\tau^*$ onto the test split (`46 good / 13 defect`) classifies every sample as `defect` (`[[0, 46], [0, 13]]`, `F1 = 26 / 72 = 0.3611`, `BalAcc = 0.5000`, `Acc = 0.2203`).
   - This exact collapse occurred in 5 out of 7 architectures (`ultra_tiny` at epoch `8` with $\tau^*=0.44$, `ds_cnn` epoch `4` $\tau^*=0.41$, `small_cnn` epoch `1` $\tau^*=0.40$, `torchvision_mobilenet` epoch `7` $\tau^*=0.32$, and near-complete collapse in `tiny_resnet` epoch `2` $\tau^*=0.66$ with `[[1, 45], [0, 13]]`). At the fixed standard threshold $\tau = 0.50$, the first four networks predicted all-negative (`f1_at_05 = 0.0`).
   - The **only non-collapsed CNN on `bottle` (`seed=42`)** is **`mobilenet_like`** (`best_epoch = 23`, $\tau^* = 0.20 \to$ `F1 = 0.7586`, `ROC-AUC = 0.9415`; even at $\tau = 0.50$ it reaches `F1 = 0.8333`). Against it, **`lr_dot (x94)` achieves `F1 = 0.8000` (`+0.0414` F1 under the official protocol, running `13.76×` faster on CPU at `0.3002 ms` vs `4.1313 ms` with `745×` fewer parameters)**.
   - On `kolektorsdd2` (`N_val = 350`: `313 / 37`), no CNN collapses on `normal` (`tiny_resnet` reaches `F1 = 0.7130` at $\tau^* = 0.69$, vs `0.5973` at $\tau = 0.50$), although inverting without augmentation causes `tiny_cnn` and `torchvision_mobilenet` to collapse.

### 2.3 Statistical Significance & `lr_dot` Micro-Benchmark ([results/profiling/](results/profiling/))

In dedicated runs of [experiments/profile_pipelines.py](experiments/profile_pipelines.py) (`N = 200`, `threads = 1`, `B = 1`), non-parametric **Mann-Whitney U** tests confirm overwhelming statistical significance (`p < 1e-55`):

| Comparison (`profile_pipelines.py`, `B=1`) | `kolektorsdd2` Median A vs B (ms) | `kolektorsdd2` Ratio & p-value | `mvtec-ad_bottle` Median A vs B (ms) | `mvtec-ad_bottle` Ratio & p-value |
| :--- | :---: | :---: | :---: | :---: |
| **`lr_dot (x94)` vs `ultra_tiny`** | `0.3629` vs `1.3211` | **`0.275×` (`3.64×` faster)**, `p = 4.83e-67` | `0.3873` vs `1.0328` | **`0.375×` (`2.67×` faster)**, `p = 1.19e-66` |
| **`lr_dot (x94)` vs `ds_cnn`** | `0.3629` vs `2.3431` | **`0.155×` (`6.46×` faster)**, `p = 4.83e-67` | `0.3873` vs `2.9051` | **`0.133×` (`7.50×` faster)**, `p = 4.83e-67` |
| `lr (x94)` vs `ultra_tiny` | `0.6018` vs `1.3211` | `0.456×` (`2.20×`), `p = 4.69e-66` | `0.6558` vs `1.0328` | `0.635×` (`1.57×`), `p = 4.86e-56` |
| `dt (x94)` vs `ultra_tiny` | `0.4759` vs `1.3211` | `0.360×` (`2.78×`), `p = 4.83e-67` | `0.5446` vs `1.0328` | `0.527×` (`1.90×`), `p = 9.82e-63` |

In the isolated classifier micro-benchmark ([classifier_times.csv](results/profiling/kolektorsdd2/classifier_times.csv)):
- **`kolektorsdd2` (`B=1`):** [`LinearDotCoreClassifier`](src/models/descriptor_classifier.py#L92-L170) (`lr_dot`) executes in **`0.00770 ms`** (`0.00610` core + `0.00160` dispatch) vs `0.39080 ms` for `sklearn Pipeline lr` (**`50.8×` faster**, and **`30.6×`** at `B=32`: `0.000277` vs `0.008483 ms/img`).
- **`mvtec-ad_bottle` (`B=1`):** `lr_dot` takes **`0.00810 ms`** vs `0.26690 ms` for `lr` (**`33.0×` faster**).

### 2.4 Multi-Scale Ablation & Explainability ($\mathbf{x}_{85}$ vs. $\mathbf{x}_9$) at `486c111`

Results from [results/classification/](results/classification/README.md) and [ablation_quality_vs_cost.csv](results/profiling/mvtec-ad_bottle/ablation_quality_vs_cost.csv):

| Dataset | Classifier | `x85_only` (`F1 / AUC`) | `x9_only` (`F1 / AUC`) | **`x94_full` (`F1 / AUC`)** | $\Delta\text{F1}(\mathbf{x}_{94}-\mathbf{x}_{85})$ | Cost `lr_dot x85` (`ms`) | Cost `lr_dot x94` (`ms`) | $\mathbf{x}_9$ Attribution (Top axes & density vs $\mathbf{x}_{85}$) |
| :--- | :--- | :---: | :---: | :---: | ---: | ---: | ---: | :--- |
| **`kolektorsdd2`** | `lr` / `lr_dot` | `0.2687 / 0.6179` | `0.3591 / 0.7371` | **`0.4265 / 0.8061`** | **`+0.1579`** | **`0.0326`** | `0.2923` | `x9 = 27.11%` perm. (`3.01%/feat`, **`3.51×`** density vs `0.86%`). Axes: `asym_lr 51.14%`, `asym_tb 43.56%`, `asym_rot180 5.30%`. Top-1: `asym_lr_8x8` (`11.75%`), #5 `asym_tb_8x8` (`8.59%`). |
| **`kolektorsdd2`** | `rf` *(200t)* | `0.4286 / 0.8255` | `0.3765 / 0.7429` | `0.4298 / 0.7904` | `+0.0013` | `10.559` *(rf)* | `10.090` *(rf)* | `x9 = 47.52%` Gini (**`8.55×`** density): **ranks 1 through 9 are all 9 `asym_*` features**. |
| **`mvtec-ad_bottle`** | `lr` / `lr_dot` | `0.7333 / 0.9181` | `0.7333 / 0.9130` | **`0.8000 / 0.9231`** | **`+0.0667`** | **`0.0317`** | `0.2216` | `x9 = 19.33%` perm. (`2.15%/feat`, **`2.26×`** density). Axes: **`asym_tb 54.25%`**, `asym_lr 32.38%`, `asym_rot180 13.37%`. Top-3: `asym_tb_8x8` (`9.17%`), `E_2x2_r1_c0` (`7.31%`), `asym_lr_8x8` (`5.64%`). |
| **`mvtec-ad_bottle`** | `rf` *(100t)* | `0.5263 / 0.8696` | `0.6923 / 0.8980` | `0.7273 / 0.9030` | **`+0.2010`** | `5.443` *(rf)* | `5.653` *(rf)* | `x9 = 38.12%` Gini (**`5.82×`** density: `4.24%` vs `0.73%/feat`). Axes: `asym_tb 35.18%`, `asym_lr 34.73%`, `asym_rot180 30.09%`. |
| **`mvtec-ad_bottle`** | `dt` *(depth 3)* | `0.6364 / 0.7525` | `0.5882 / 0.7584` | `0.5833 / 0.6614` | `-0.0530` | — | — | In `ablation_quality_vs_cost.csv` (retrained), `dt x85` scores `0.5714` due to feature-tie randomness without `random_state` in [`build_classifier`](src/models/descriptor_classifier.py#L931-L937). |
| *Historical `bf4a29f` (`mvtec-ad_tile`)* | `lr` / `rf` / `dt` | `0.733 / 0.435 / 0.313` | `0.833 / 0.857 / 0.727` | `0.865 / 0.824 / 0.839` | `+0.132 / +0.389 / +0.526` | — | — | Historical `bf4a29f` snapshot ([results/classification/README.md](results/classification/README.md); `tile` not re-evaluated with CNNs at `486c111`). |

---

## 3. Direct Pre-MCU Answer: Is Pure Python Enough Before Migrating to C on a Microcontroller?

To answer empirically without speculation, isolated runs were executed in `results/comparison_cpu_python/` comparing the two pure-Python backends (`numpy_baseline` equivalent to the original `V0` and `numpy_finest_pyramid` optimized `V1`) against `c_single_pass` (`V3` in C99) and the CNNs on CPU (`threads=1`, `seed=42`).

### 3.1 Real Comparison: Pure Python vs. C99 vs. `ultra_tiny` (`threads=1`)

| Pipeline / Backend | `kolektorsdd2` `B=1` (`ms`) | `kolektorsdd2` `B=32` (`ms/img`) | `mvtec-ad_bottle` `B=1` (`ms`) | `mvtec-ad_bottle` `B=32` (`ms/img`) | `kolektorsdd2-inv` `B=1` (`ms`) | `mvtec-ad-inv_bottle` `B=1` (`ms`) | Peak `tracemalloc` fine-hist stage | Relative Verdict |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **`numpy_baseline` + `lr_dot`** | `2.8665` (`349 img/s`) | — | `2.8066` (`356 img/s`) | — | — | — | `193–201 KB` (`V0`) | **Loses `2.36×–2.75×` vs `ultra_tiny`** and runs **`7.55×–9.35×` slower than C99**. |
| **`numpy_baseline` + `lr` (`sklearn`)** | `3.1890` | — | `3.1414` | — | — | — | `193–201 KB` | **Loses `2.64×–3.06×` vs `ultra_tiny`**. |
| **`numpy_finest_pyramid` + `lr_dot`** | `0.7951` (`1,258 img/s`) | `0.3227` (`3,099/s`) | `0.9717` (`1,029 img/s`) | `0.3088` (`3,239/s`) | `0.8188` | `0.8153` | `200.8 KB` (`B=1`)<br>**`6,408.6 KB` (`B=32`)** | At `B=1` it slightly edges out `ultra_tiny`, but `lr` (`1.097–1.236 ms`) and `dt` on `bottle` (`1.121 ms`) still lose. At `B=32` it runs **`5.5×–12.3×` slower than C99**. |
| **`c_single_pass` (`C99 x85`) + `lr_dot` (`x94`)** | **`0.3799`** (`2,633 img/s`) | **`0.0271`** (`36,913/s`) | **`0.3002`** (`3,331 img/s`) | **`0.0320`** (`31,245/s`) | **`0.2450`** | **`0.2395`** | **`1.86–1.91 KB`** | **Dominates all CNNs** (`2.67×–3.64×` vs `ultra_tiny`, `6.46×–13.76×` vs `ds_cnn`/`mobilenet_like`). |
| **`ultra_tiny` (CNN PyTorch C++/AVX2)** | `0.9919–1.0623` | `0.4466–0.5373` | `0.9171–1.1888` | `0.5183–0.5197` | `0.9645–1.0783` | `0.9610–1.0240` | `128 KB` activations | Fastest CPU CNN baseline; beats the pure-Python baseline. |

### 3.2 Internal Diagnosis of the New Bottleneck (Why C99 Changes the Landscape)

Measuring stage by stage ([results/profiling/](results/profiling/kolektorsdd2/descriptor_stage_times.csv) vs `results/comparison_cpu_python/profiling/`) reveals where computational cost actually resides:

1. **With `numpy_finest_pyramid` (pure Python + NumPy, `B=1` and `B=32`):**
   - The stage sum at `B=1` is `0.4746 ms` (`kolektorsdd2`) / `0.4687 ms` (`bottle`): `quantize (0.049/0.047 ms)` + `fine_grid_histograms (0.120/0.119 ms, 25.2%–25.4%)` + `pyramid_sum (0.024 ms)` + `ent_g1..8 (0.077/0.078 ms)` + `asymmetry_g2/4/8 (0.194/0.190 ms, ~40.7%)` + `concat (~0.011 ms)`.
   - Increasing to `B=32` causes `fine_grid_histograms` with `np.bincount` to explode to **`73.5%–74.0%`** of the descriptor time (`0.2039–0.2161 ms/img`) and **`6,408.6 KB`** in `tracemalloc`, making it ~`10×–12×` slower than C99.
2. **With `c_single_pass` (C99 for $\mathbf{x}_{85}$, Python/NumPy still for $\mathbf{x}_9$):**
   - The entire C99 extraction of the 85 entropies (`c_fine_hist 0.0330–0.0336 ms + c_pyramid_sum 0.0110–0.0116 ms + c_entropy_lut 0.0114–0.0125 ms`) costs only **`0.0554–0.0577 ms`** (`22.8%–23.0%` of descriptor stages) and **`1.9–2.1 KB`** on the static stack with zero `malloc`.
   - The **remaining `74.2%–74.4%`** of the current descriptor (`0.1807–0.1859 ms`) comes entirely from **`asymmetry_g2/4/8` still running in Python/NumPy** (`0.0677–0.0703 + 0.0572–0.0589 + 0.0558–0.0568 ms`), which in C amounts to ~100 `fabsf()` subtractions on the stack (`< 0.003 ms`). Indeed, in [ablation_quality_vs_cost.csv](results/profiling/mvtec-ad_bottle/ablation_quality_vs_cost.csv), **`lr_dot (x85_only)` already reaches `0.0317–0.0326 ms` (`~31,000 img/s`)**.

### 3.3 Concrete Conclusion: What Can and Cannot Be Validated in Pure Python Before the MCU

- **CAN be finalized in pure Python (`numpy_finest_pyramid` + `lr_dot`):**
  1. Predictive quality (`F1`, `ROC-AUC`, `PR-AUC`, confusion matrix), threshold calibration, $\mathbf{x}_{85}$ vs $\mathbf{x}_9$ ablations, explainability, and photometric invariance $255 - I$ — mathematical parity against C99 is **`0.000e+00`** (difference $\le 1.89 \times 10^{-15}$).
  2. Export of final trained parameters (`w_eff[94], b_eff`) algebraically folded without depending on `StandardScaler` or `scikit-learn`.
- **MUST NOT use pure Python for:**
  1. **Comparing CPU latency against CNNs:** PyTorch runs convolutions in compiled C++/AVX2 (`ATen`). Comparing against Python `V0` unfairly penalizes the classical method by `~7.5×–9.4×`, and even `numpy_finest_pyramid` is held back at `B=1` by Python dispatch (`asymmetry`) and at `B=32` by `np.bincount` (`9.65×–12.30×` vs C99). **For a fair pre-MCU comparison you already have C99 (`c_single_pass`); there is no need to restrict latency claims to pure Python.**
  2. **Estimating MCU SRAM, Flash, cycle counts, or energy:** An x86-64 Core i7 hides LUT footprint, L1 cache collisions, lack of FPU/SIMD, and real Flash/SRAM usage on a Cortex-M or RISC-V target.

### 3.4 Exact Pre-MCU Portability Checklist (4 Steps from Desktop C99 $\to$ C Firmware)

> [!NOTE]
> **Status update (landed on `main` via `feat/c-migration` & `feat/hw-test`, desktop + physical ESP32-CAM, Nano & Uno verified):** All four C99 portability steps identified at `486c111` below are **implemented and verified on desktop across all `16/16` targets** in [c_imp/](c_imp/README.md) + `src/c_export/` ([results/c_imp/README.md](results/c_imp/README.md)) and **measured on physical ESP32-CAM (`COM5`), Arduino Nano (`COM4`), and Arduino Uno (`COM6`)** in [hw_test/](hw_test/README.md) + `src/hw_test/` ([results/hw_test/README.md](results/hw_test/README.md), [results/FINAL_PAPER_REPORT.md](results/FINAL_PAPER_REPORT.md)): on ESP32-CAM, `lr_dot/q15` runs in **`3.191 ms`** (`3.178 ms` descriptor + `0.012 ms` model, **`53.3×` faster than `ultra_tiny/q15` at `170.007 ms`** and **`624×–2,716×` faster than larger `q15` CNNs at `1,991–8,665 ms`**) and `mlp_16_dot/q15` runs in **`3.366 ms`** with **`100.0%` bit-exact parity (`0` LSB raw diff across all 9 `q15` models, `96/96` images)**. Furthermore, streaming `CIMP_DescStream` (`1,060 B` state) enables `lr_dot/q15` (`12,780 B` Flash, `1,636 B` static RAM, `78.4 ms`) and `mlp_16_dot/q15` (`19,038 B` Flash, `1,636 B` static RAM, `101.8–101.9 ms`) to run live with `100.0%` parity (`32/32` images) on 8-bit `ATmega328P` (`nano` and `uno`, `2 KiB` SRAM).

1. **Reuse the heap-free core of [`entropy_pyramid.c`](src/features/native/entropy_pyramid.c#L26-L31):** Its `small` path already operates entirely on the stack (`1,024 × uint16_t` fine counts + `384 × uint32_t` coarse counts = **`3,584 B ≈ 3.5 KiB`** for `[1,2,4,8], bins=16`), with zero `malloc/free` and zero OS dependencies. Even `qlut[256]` disappears using inline `px >> 4` when `bins=16`. If the MCU lacks `16 KiB` of free SRAM for a full `128x128` frame, rows can be streamed in `16`-pixel strips (`2,048 B`, one row of $8\times8$ cells) accumulating the fine grid on the fly.
2. **Move `asymmetry_g2/4/8` into C99:** Porting the 9 `LR/TB/R180` differences over `float E[85]` inside the same C kernel eliminates the remaining `74%` Python latency on desktop (`0.24 ms → ~0.06–0.08 ms`) and makes the $\mathbf{x}_{94}$ extractor 100% self-contained in C.
3. **Export `lr_dot` as a static `const` array:** `LinearDotCoreClassifier` reduces the entire classifier + normalization to **`94 + 1` weights (`380 B` in `float32`, or `~190 B` in `int16_t` Q15)**.
4. **Compact `nlogn_lut` for embedded Flash:** On PC, [`prepare_native_buffers`](src/features/native_entropy.py#L355-L364) allocates `(128/1)^2 + 1 = 16,385` `float64` entries (**`131,080 B ≈ 128 KiB`**) just to cover the `1x1` cell (`16,384` pixels). In MCU firmware, the `8x8` level (`64` cells, `1,024` counts) only needs **`257` `float32` entries (`1,028 B` in Flash)**, while the **`21` coarse cells** (`4x4 + 2x2 + 1x1` $\to$ `336` total iterations) can be evaluated with `c * log2f((float)c)` or fixed-point arithmetic without spending `128 KiB` of Flash.

---

## 4. Open Gaps Before Final Manuscript & Real MCU Benchmark

1. **`M2` (Non-Degenerate CNN Threshold Guard + Dual Threshold Reporting):** Add a guard in [`find_optimal_threshold`](src/models/cnn/evaluate.py#L54-L93) when predictions degenerate to all-positive on small validation splits (`bottle` `N_val=35`) and report both regimes ($\tau = 0.50$ fixed and validated/OOB $\tau^*$) across both classical ML and CNNs.
2. **`M3` (Multi-Category MVTec-AD Coverage & Multi-Seed Sweeps):** Full 15-category MVTec-AD training, CPU benchmarking, and C99 export (`seed=42`) are now completed in [results/FINAL_PAPER_REPORT.md](results/FINAL_PAPER_REPORT.md); multi-seed sweeps (`42, 43, 44`) remain open to strengthen variance bounds.
3. **`DecisionTreeClassifier` Determinism:** Pass `random_state=cfg.seed` in [`build_classifier`](src/models/descriptor_classifier.py#L931-L937) to eliminate minor tie-breaking variance across runs (`0.6364` vs `0.5714` on `bottle_dt`).
4. **Remaining Physical Board/Transport Measurements & Dynamic Region Selectors:** Both the standalone zero-malloc C99 port ([results/c_imp/README.md](results/c_imp/README.md)) and live UART hardware runs on **ESP32-CAM (`COM5`, `96/96` parity across 12 runs)**, **Arduino Nano (`COM4`, `16/16` parity)**, and **Arduino Uno (`COM6`, `16/16` parity)** ([results/hw_test/README.md](results/hw_test/README.md)) are now landed and verified. Remaining open work is measuring physical `nodemcu` (compile-verified 4/4; host UART blocked by `ch341ser.sys 3.9.2024.9`), `WiFi/TCP` transport (`secrets.toml`), and the 5 remaining `f32` PSRAM CNNs on `esp32cam`, plus exploring finer grids (`16×16`) or dynamic top-$k$/threshold region selectors (`T_total = T_selector + T_model`).
