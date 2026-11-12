# HYPOTHESIS.md — Verificación de Hipótesis Post-Merge y Viabilidad Pre-MCU (Python vs. C99 vs. CNNs)

- **Fecha de síntesis:** 2026-11-11
- **Rama y revisión evaluada:** `main` @ `486c111` *(tras integrar `feat/fast-entropy-opt`: kernel C99 `c_single_pass`, variantes NumPy piramidales `V1/V2`, y clasificador lineal fusionado `lr_dot`)*
- **Protocolo medido:** CPU `Intel(R) Core(TM) i7-7700HQ @ 2.80 GHz`, `threads = 1`, `batch_size = 1` (latencia embebida) y `batch_size = 32` (throughput), `seed = 42` alineado entre ML clásico y checkpoints CNN (referencia CUDA en NVIDIA GTX 1050 `cu126`).
- **Documentos relacionados:** [README.md](README.md), [RESUME.md](RESUME.md), [REVIEW_01.md](REVIEW_01.md) *(auditoría forense histórica `bf4a29f`)* y [understanding/13_entropy_maps_optimization/README.md](understanding/13_entropy_maps_optimization/README.md).

> [!NOTE]
> **Alcance exacto de los artefactos en disco (`486c111`):** Las corridas formales posteriores al merge sobre `main` cubren `kolektorsdd2` (`normal` e `inverted`) y `mvtec-ad_bottle` (`normal` e `inverted`), junto con comparaciones aisladas en puro Python bajo `results/comparison_cpu_python/` (ignorado por Git). Cualquier mención a `mvtec-ad_tile` o al barrido `--all-objects` de Random Forest proviene explícitamente de la instantánea histórica `bf4a29f` auditada en [REVIEW_01.md](REVIEW_01.md) y [results/classification/README.md](results/classification/README.md).

---

## 1. Veredicto Maestro de las Hipótesis de Investigación

| Hipótesis | Enunciado | Veredicto Post-Merge (`486c111`) | Evidencia Cuantitativa Clave |
| :--- | :--- | :--- | :--- |
| **$H_{\text{sym}}$ (Sinergia Simetría–Entropía)** | Añadir los 9 residuos de asimetría espacial ($\mathbf{x}_9$) a las 85 entropías piramidales de Shannon ($\mathbf{x}_{85}$) mejora la detección frente a usar solo entropía. | **VERIFICADA** | En `kolektorsdd2`, `lr` sube de `F1 = 0.2687` (`ROC-AUC = 0.6179`) en $\mathbf{x}_{85}$ a **`F1 = 0.4265` (`ROC-AUC = 0.8061`)** en $\mathbf{x}_{94}$ ($\mathbf{+0.1579}$ F1 / $\mathbf{+0.1883}$ AUC); en `mvtec-ad_bottle`, `lr` sube de `0.7333` a **`0.8000`** ($\mathbf{+0.0667}$ F1) y `rf` de `0.5263` a **`0.7273`** ($\mathbf{+0.2010}$ F1). *(Ref. histórica `bf4a29f` en `mvtec-ad_tile`: `lr` $+0.1315$ F1, `rf` $+0.3887$ F1).* |
| **$H_{\text{inv}}$ (Invarianza Fotométrica $255 - I$)** | En imágenes sin padding, $\mathbf{x}_{94}$ es invariante a la inversión de polaridad $255 - I$, mientras que CNNs estándar sin aumentación fotométrica degradan. | **VERIFICADA** | Identidad exacta por conmutatividad de $-\sum p_b \log_2 p_b$. En `mvtec-ad-inverted_bottle`, `lr_dot (x94)` conserva **`F1 = 0.8000, BalAcc = 0.9072`** (`ROC-AUC = 0.9231 → 0.9214`), mientras `mobilenet_like` colapsa (`F1 = 0.7586 → 0.3611`, `ROC-AUC = 0.9415 → 0.6739`). En `kolektorsdd2-inverted`, el `letterbox` negro fijo (`pad=0`) introduce una deriva menor controlada (`lr_dot` `F1 = 0.4265 → 0.4153`, `ROC-AUC = 0.8061 → 0.8151`), pero `tiny_cnn` y `torchvision_mobilenet` colapsan (`BalAcc ≈ 0.50–0.54`). |
| **$H_{\text{pareto}}$ (Pareto en CPU de $\mathbf{x}_{94}$ + Clasificador Ligero vs. CNNs)** | En CPU (`threads=1, batch=1`) sobre defectos estructurales, $\mathbf{x}_{94}$ + clasificador ligero supera o iguala a CNNs ligeras con órdenes de magnitud menos parámetros y menor latencia. | **VERIFICADA EN RÉGIMEN COMPILADO** | Con `c_single_pass` + `lr_dot`, el pipeline tarda **`0.3002 ms`** (`3,331.1 img/s`) en `mvtec-ad_bottle` y **`0.3799 ms`** (`2,632.6 img/s`) en `kolektorsdd2` (`95` parámetros entrenables). En `bottle` (`seed=42`) supera en CPU a **las 7 CNNs** en latencia (`3.05×` vs `ultra_tiny`, **`13.76×`** vs `mobilenet_like`), parámetros (`745×` menos que `mobilenet_like`) y F1 (**`0.8000` vs `0.7586`** de `mobilenet_like`, única CNN sin colapso de umbral `val_f1`). **Matiz clave:** requiere ejecución compilada (`c_single_pass`); en Python puro pierde frente a `ultra_tiny`. |
| **$H_{\text{univ}}$ (Superioridad Universal en Micro-Grietas y en Puro Python)** | $\mathbf{x}_{94}$ vence también en micro-grietas sub-celda y supera a CNNs sin necesidad de compilación nativa. | **REFUTADA** | (1) **Límite físico de resolución ($16\times16$ px por celda en $8\times8$):** en `kolektorsdd2`, una grieta de 1–2 px altera poco el histograma de 256 px, por lo que **las 7 CNNs vencen en F1** (`tiny_resnet`: **`F1 = 0.7130, ROC-AUC = 0.9213`** vs `lr_dot`: `F1 = 0.4265, ROC-AUC = 0.8061`). (2) **Límite de intérprete:** con `numpy_baseline`, `lr_dot` tarda `2.807–2.867 ms` (`B=1`), siendo **`2.36×–2.75×` más lento** que `ultra_tiny` (`1.042–1.189 ms`). |
| **$H_{\text{opt-1}}$ (Pirámide Single-Pass en NumPy `V1/V2`)** | Reemplazar el loop de 85 regiones por un solo histograma fino + suma piramidal acelera sustancialmente $\mathbf{x}_{85}$ en NumPy manteniendo paridad exacta ($\le 10^{-12}$). | **VERIFICADA EN $\mathbf{x}_{85}$** | En Lab 13 (`200` muestras reales), `V1_numpy_finest_pyramid` alcanza **`9.40×`** sobre `V0` en `B=1` (`0.2371 ms` vs `2.2282 ms`, $p = 4.83 \times 10^{-67}$) y `V2` **`10.49×`** en `B=1`, aunque `V2` pierde frente a `V1` en lotes $\ge 8$ y en `512x512` por presión de memoria en `[N, 85, B]`. Paridad numérica $\le 1.89 \times 10^{-15}$. |
| **$H_{\text{opt-2}}$ (Kernel C99 `c_single_pass` + `lr_dot`)** | Un kernel C99 de paso único (`entropy_pyramid.c`) más plegado algebraico de `StandardScaler + LR` (`lr_dot`) elimina el sesgo de Python/sklearn y supera en latencia a todas las CNNs en CPU. | **VERIFICADA** | En Lab 13 (`x85`), `V3_c_single_pass` logra **`92.65×`** sobre `V0` (`0.02405 ms`). En pipeline completo E2E (`x94 + lr_dot`), `lr_dot` acelera **`33.0×–50.8×`** respecto a `sklearn lr` (`0.0077–0.0081 ms` vs `0.2669–0.3908 ms` en micro-benchmark) y permite que `x94 + lr_dot` venza a `ultra_tiny` (`3.64×` en `kolektorsdd2`, **`2.67×` en `bottle`** en profiling, $p \le 1.19 \times 10^{-66}$). En `x85_only + lr_dot`, `T_total` cae a **`0.0317–0.0326 ms`** (`~31,000 img/s`). |

---

## 2. Evidencia Cuantitativa Post-Merge (`main @ 486c111`, `seed=42`)

### 2.1 Comparativa Oficial 100% CPU: `c_single_pass` + ML vs. 7 CNNs (`threads=1`)

Datos extraídos directamente de [results/comparison_cpu/kolektorsdd2_baseline_vs_cnn_cpu.csv](results/comparison_cpu/kolektorsdd2_baseline_vs_cnn_cpu.csv), [kolektorsdd2-inverted_baseline_vs_cnn_cpu.csv](results/comparison_cpu/kolektorsdd2-inverted_baseline_vs_cnn_cpu.csv), [mvtec-ad_bottle_baseline_vs_cnn_cpu.csv](results/comparison_cpu/mvtec-ad_bottle_baseline_vs_cnn_cpu.csv) y [mvtec-ad-inverted_bottle_baseline_vs_cnn_cpu.csv](results/comparison_cpu/mvtec-ad-inverted_bottle_baseline_vs_cnn_cpu.csv) (`benchmark_cpu_fair.py`, $N=200$ iteraciones cronometradas por modelo):

#### A. `kolektorsdd2` (`N_test = 1,004`: `894 good / 110 defect`, CNN `val = 350`)

| Modelo | Familia | Parámetros (`modelo`) | F1 | BalAcc | ROC-AUC | PR-AUC | `T_desc/prep` `B=1` (ms) | `T_model` `B=1` (ms) | **`T_total` `B=1` (ms)** | `Thru B=1` (`img/s`) | **`T_total` `B=32` (ms/img)** | **`F1` Inv.** | `ROC-AUC` Inv. |
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

| Modelo | Familia | Parámetros (`modelo`) | F1 | BalAcc | ROC-AUC | PR-AUC | `T_desc/prep` `B=1` (ms) | `T_model` `B=1` (ms) | **`T_total` `B=1` (ms)** | `Thru B=1` (`img/s`) | **`T_total` `B=32` (ms/img)** | **`F1` Inv.** | `ROC-AUC` Inv. |
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

### 2.2 Diagnóstico Exacto del Colapso de CNNs en `mvtec-ad_bottle` (`seed=42`)

Al alinear ML y CNNs exactamente sobre la misma partición (`seed=42`) en `mvtec-ad_bottle`, la brecha muestral entre conjuntos de validación explica por qué las CNNs colapsan aquí pero no en `kolektorsdd2`:

1. **Tamaño crítico del set de validación (`N_val = 35`: `27 good / 8 defect`) + `best_epoch` temprano + `val_f1`:**
   - Un predictor degenerado todo-positivo en validación obtiene `val F1 = 2 * 8 / (35 + 8) = 16 / 43 = 0.3721`. Al proyectar ese umbral $\tau^*$ al conjunto de test (`46 good / 13 defect`), todo queda clasificado como `defect` (`[[0, 46], [0, 13]]`, `F1 = 26 / 72 = 0.3611`, `BalAcc = 0.5000`, `Acc = 0.2203`).
   - Exactamente eso ocurrió en 5 de 7 arquitecturas (`ultra_tiny` en época `8` con $\tau^*=0.44$, `ds_cnn` época `4` $\tau^*=0.41$, `small_cnn` época `1` $\tau^*=0.40$, `torchvision_mobilenet` época `7` $\tau^*=0.32$, y casi completo en `tiny_resnet` época `2` $\tau^*=0.66$ con `[[1, 45], [0, 13]]`). Con el umbral fijo estándar $\tau = 0.50$, las cuatro primeras redes predecían todo negativo (`f1_at_05 = 0.0`).
   - La **única CNN no colapsada en `bottle` (`seed=42`)** es **`mobilenet_like`** (`best_epoch = 23`, $\tau^* = 0.20 \to$ `F1 = 0.7586`, `ROC-AUC = 0.9415`; incluso en $\tau = 0.50$ logra `F1 = 0.8333`). Frente a ella, **`lr_dot (x94)` obtiene `F1 = 0.8000` (`+0.0414` F1 con el protocolo oficial, `13.76×` más rápido en CPU `0.3002 ms` vs `4.1313 ms` y `745×` menos parámetros)**.
   - En `kolektorsdd2` (`N_val = 350`: `313 / 37`), ninguna CNN colapsa en `normal` (`tiny_resnet` alcanza `F1 = 0.7130` en $\tau^* = 0.69$, frente a `0.5973` en $\tau = 0.50$), aunque al invertir sin aumentación sí colapsan `tiny_cnn` y `torchvision_mobilenet`.

### 2.3 Significancia Estadística y Micro-Benchmark de `lr_dot` ([results/profiling/](results/profiling/))

En las corridas dedicadas de [experiments/profile_pipelines.py](experiments/profile_pipelines.py) (`N = 200`, `threads = 1`, `B = 1`), las pruebas no paramétricas **Mann-Whitney U** confirman ganancias estadísticamente abrumadoras (`p < 1e-55`):

| Comparación (`profile_pipelines.py`, `B=1`) | `kolektorsdd2` Mediana A vs B (ms) | `kolektorsdd2` Ratio & p-valor | `mvtec-ad_bottle` Mediana A vs B (ms) | `mvtec-ad_bottle` Ratio & p-valor |
| :--- | :---: | :---: | :---: | :---: |
| **`lr_dot (x94)` vs `ultra_tiny`** | `0.3629` vs `1.3211` | **`0.275×` (`3.64×` más rápido)**, `p = 4.83e-67` | `0.3873` vs `1.0328` | **`0.375×` (`2.67×` más rápido)**, `p = 1.19e-66` |
| **`lr_dot (x94)` vs `ds_cnn`** | `0.3629` vs `2.3431` | **`0.155×` (`6.46×` más rápido)**, `p = 4.83e-67` | `0.3873` vs `2.9051` | **`0.133×` (`7.50×` más rápido)**, `p = 4.83e-67` |
| `lr (x94)` vs `ultra_tiny` | `0.6018` vs `1.3211` | `0.456×` (`2.20×`), `p = 4.69e-66` | `0.6558` vs `1.0328` | `0.635×` (`1.57×`), `p = 4.86e-56` |
| `dt (x94)` vs `ultra_tiny` | `0.4759` vs `1.3211` | `0.360×` (`2.78×`), `p = 4.83e-67` | `0.5446` vs `1.0328` | `0.527×` (`1.90×`), `p = 9.82e-63` |

En el micro-benchmark aislado de clasificadores ([classifier_times.csv](results/profiling/kolektorsdd2/classifier_times.csv)):
- **`kolektorsdd2` (`B=1`):** [`LinearDotCoreClassifier`](src/models/descriptor_classifier.py#L92-L170) (`lr_dot`) ejecuta en **`0.00770 ms`** (`0.00610` núcleo + `0.00160` despacho) frente a `0.39080 ms` de `sklearn Pipeline lr` (**`50.8×` más rápido**, y **`30.6×`** en `B=32`: `0.000277` vs `0.008483 ms/img`).
- **`mvtec-ad_bottle` (`B=1`):** `lr_dot` tarda **`0.00810 ms`** vs `0.26690 ms` de `lr` (**`33.0×` más rápido**).

### 2.4 Ablación Multi-Escala y Explicabilidad ($\mathbf{x}_{85}$ vs. $\mathbf{x}_9$) en `486c111`

Resultados de [results/classification/](results/classification/README.md) y [ablation_quality_vs_cost.csv](results/profiling/mvtec-ad_bottle/ablation_quality_vs_cost.csv):

| Dataset | Clasificador | `x85_only` (`F1 / AUC`) | `x9_only` (`F1 / AUC`) | **`x94_full` (`F1 / AUC`)** | $\Delta\text{F1}(\mathbf{x}_{94}-\mathbf{x}_{85})$ | Coste `lr_dot x85` (`ms`) | Coste `lr_dot x94` (`ms`) | Atribución $\mathbf{x}_9$ (Top ejes y densidad vs $\mathbf{x}_{85}$) |
| :--- | :--- | :---: | :---: | :---: | ---: | ---: | ---: | :--- |
| **`kolektorsdd2`** | `lr` / `lr_dot` | `0.2687 / 0.6179` | `0.3591 / 0.7371` | **`0.4265 / 0.8061`** | **`+0.1579`** | **`0.0326`** | `0.2923` | `x9 = 27.11%` perm. (`3.01%/feat`, **`3.51×`** densidad vs `0.86%`). Ejes: `asym_lr 51.14%`, `asym_tb 43.56%`, `asym_rot180 5.30%`. Top-1: `asym_lr_8x8` (`11.75%`), #5 `asym_tb_8x8` (`8.59%`). |
| **`kolektorsdd2`** | `rf` *(200t)* | `0.4286 / 0.8255` | `0.3765 / 0.7429` | `0.4298 / 0.7904` | `+0.0013` | `10.559` *(rf)* | `10.090` *(rf)* | `x9 = 47.52%` Gini (**`8.55×`** densidad): **los puestos 1 al 9 completos son las 9 variables `asym_*`**. |
| **`mvtec-ad_bottle`** | `lr` / `lr_dot` | `0.7333 / 0.9181` | `0.7333 / 0.9130` | **`0.8000 / 0.9231`** | **`+0.0667`** | **`0.0317`** | `0.2216` | `x9 = 19.33%` perm. (`2.15%/feat`, **`2.26×`** densidad). Ejes: **`asym_tb 54.25%`**, `asym_lr 32.38%`, `asym_rot180 13.37%`. Top-3: `asym_tb_8x8` (`9.17%`), `E_2x2_r1_c0` (`7.31%`), `asym_lr_8x8` (`5.64%`). |
| **`mvtec-ad_bottle`** | `rf` *(100t)* | `0.5263 / 0.8696` | `0.6923 / 0.8980` | `0.7273 / 0.9030` | **`+0.2010`** | `5.443` *(rf)* | `5.653` *(rf)* | `x9 = 38.12%` Gini (**`5.82×`** densidad: `4.24%` vs `0.73%/feat`). Ejes: `asym_tb 35.18%`, `asym_lr 34.73%`, `asym_rot180 30.09%`. |
| **`mvtec-ad_bottle`** | `dt` *(prof. 3)* | `0.6364 / 0.7525` | `0.5882 / 0.7584` | `0.5833 / 0.6614` | `-0.0530` | — | — | En `ablation_quality_vs_cost.csv` (reentrenado) `dt x85` marca `0.5714` por empates sin `random_state` en [`build_classifier`](src/models/descriptor_classifier.py#L931-L937). |
| *Histórico `bf4a29f` (`mvtec-ad_tile`)* | `lr` / `rf` / `dt` | `0.733 / 0.435 / 0.313` | `0.833 / 0.857 / 0.727` | `0.865 / 0.824 / 0.839` | `+0.132 / +0.389 / +0.526` | — | — | Snapshot histórico en [REVIEW_01.md](REVIEW_01.md) (`tile` no re-evaluado con CNNs en `486c111`). |

---

## 3. Respuesta Directa Pre-MCU: ¿Puro Python Basta Antes de Migrar a C en Microcontrolador?

Para responder empíricamente sin especular, se ejecutaron corridas aisladas en `results/comparison_cpu_python/` comparando los dos backends puros en Python (`numpy_baseline` equivalente al `V0` original y `numpy_finest_pyramid` optimizado `V1`) frente a `c_single_pass` (`V3` en C99) y las CNNs en CPU (`threads=1`, `seed=42`).

### 3.1 Comparativa Real: Puro Python vs. C99 vs. `ultra_tiny` (`threads=1`)

| Pipeline / Backend | `kolektorsdd2` `B=1` (`ms`) | `kolektorsdd2` `B=32` (`ms/img`) | `mvtec-ad_bottle` `B=1` (`ms`) | `mvtec-ad_bottle` `B=32` (`ms/img`) | `kolektorsdd2-inv` `B=1` (`ms`) | `mvtec-ad-inv_bottle` `B=1` (`ms`) | Pico `tracemalloc` etapa hist. fina | Veredicto Relativo |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **`numpy_baseline` + `lr_dot`** | `2.8665` (`349 img/s`) | — | `2.8066` (`356 img/s`) | — | — | — | `193–201 KB` (`V0`) | **Pierde `2.36×–2.75×` vs `ultra_tiny`** y es **`7.55×–9.35×` más lento que C99**. |
| **`numpy_baseline` + `lr` (`sklearn`)** | `3.1890` | — | `3.1414` | — | — | — | `193–201 KB` | **Pierde `2.64×–3.06×` vs `ultra_tiny`**. |
| **`numpy_finest_pyramid` + `lr_dot`** | `0.7951` (`1,258 img/s`) | `0.3227` (`3,099/s`) | `0.9717` (`1,029 img/s`) | `0.3088` (`3,239/s`) | `0.8188` | `0.8153` | `200.8 KB` (`B=1`)<br>**`6,408.6 KB` (`B=32`)** | En `B=1` supera ligeramente a `ultra_tiny`, pero `lr` (`1.097–1.236 ms`) y `dt` en `bottle` (`1.121 ms`) siguen perdiendo. En `B=32` es **`5.5×–12.3×` más lento que C99**. |
| **`c_single_pass` (`C99 x85`) + `lr_dot` (`x94`)** | **`0.3799`** (`2,633 img/s`) | **`0.0271`** (`36,913/s`) | **`0.3002`** (`3,331 img/s`) | **`0.0320`** (`31,245/s`) | **`0.2450`** | **`0.2395`** | **`1.86–1.91 KB`** | **Domina a todas las CNNs** (`2.67×–3.64×` vs `ultra_tiny`, `6.46×–13.76×` vs `ds_cnn`/`mobilenet_like`). |
| **`ultra_tiny` (CNN PyTorch C++/AVX2)** | `0.9919–1.0623` | `0.4466–0.5373` | `0.9171–1.1888` | `0.5183–0.5197` | `0.9645–1.0783` | `0.9610–1.0240` | `128 KB` activaciones | Base CNN más rápida en CPU; vence al baseline en puro Python. |

### 3.2 Diagnóstico Interno del Nuevo Cuello de Botella (Por Qué C99 Cambia el Mapa)

Al medir etapa por etapa ([results/profiling/](results/profiling/kolektorsdd2/descriptor_stage_times.csv) vs `results/comparison_cpu_python/profiling/`), se descubre dónde quedó realmente el coste computacional:

1. **Con `numpy_finest_pyramid` (puro Python + NumPy, `B=1` y `B=32`):**
   - La suma de etapas en `B=1` es `0.4746 ms` (`kolektorsdd2`) / `0.4687 ms` (`bottle`): `quantize (0.049/0.047 ms)` + `fine_grid_histograms (0.120/0.119 ms, 25.2%–25.4%)` + `pyramid_sum (0.024 ms)` + `ent_g1..8 (0.077/0.078 ms)` + `asymmetry_g2/4/8 (0.194/0.190 ms, ~40.7%)` + `concat (~0.011 ms)`.
   - Al subir a `B=32`, `fine_grid_histograms` con `np.bincount` explota al **`73.5%–74.0%`** del descriptor (`0.2039–0.2161 ms/img`) y **`6,408.6 KB`** en `tracemalloc`, haciéndolo ~`10×–12×` más lento que C99.
2. **Con `c_single_pass` (C99 para $\mathbf{x}_{85}$, Python/NumPy aún para $\mathbf{x}_9$):**
   - Toda la extracción C99 de las 85 entropías (`c_fine_hist 0.0330–0.0336 ms + c_pyramid_sum 0.0110–0.0116 ms + c_entropy_lut 0.0114–0.0125 ms`) cuesta solo **`0.0554–0.0577 ms`** (`22.8%–23.0%` de las etapas del descriptor) y **`1.9–2.1 KB`** en pila estática sin `malloc`.
   - El **`74.2%–74.4%` restante** del descriptor actual (`0.1807–0.1859 ms`) corresponde por completo a **`asymmetry_g2/4/8` ejecutándose aún en Python/NumPy** (`0.0677–0.0703 + 0.0572–0.0589 + 0.0558–0.0568 ms`), que en C equivale apenas a ~100 restas `fabsf()` en pila (`< 0.003 ms`). De hecho, en [ablation_quality_vs_cost.csv](results/profiling/mvtec-ad_bottle/ablation_quality_vs_cost.csv), **`lr_dot (x85_only)` ya alcanza `0.0317–0.0326 ms` (`~31,000 img/s`)**.

### 3.3 Conclusión Concreta: Qué Sí y Qué No Puedes Validar en Puro Python Antes del MCU

- **SÍ puedes cerrar en puro Python (`numpy_finest_pyramid` + `lr_dot`):**
  1. Calidad predictiva (`F1`, `ROC-AUC`, `PR-AUC`, matriz de confusión), calibración de umbrales, ablaciones $\mathbf{x}_{85}$ vs $\mathbf{x}_9$, explicabilidad e invarianza fotométrica $255 - I$ — la paridad matemática frente a C99 es **`0.000e+00`** (diferencia $\le 1.89 \times 10^{-15}$).
  2. Exportación de parámetros finales entrenados (`w_eff[94], b_eff`) ya plegados sin depender de `StandardScaler` ni `scikit-learn`.
- **NO debes usar puro Python para:**
  1. **Comparar latencia CPU contra CNNs:** PyTorch corre convoluciones en C++/AVX2 (`ATen`). Comparar contra Python `V0` penaliza injustamente al método clásico por `~7.5×–9.4×`, y aun `numpy_finest_pyramid` queda frenado en `B=1` por el despacho Python (`asymmetry`) y en `B=32` por `np.bincount` (`9.65×–12.30×` vs C99). **Para la comparativa justa antes del MCU ya dispones de C99 (`c_single_pass`), no necesitas limitarte a Python.**
  2. **Estimar SRAM, Flash, ciclos reales o energía de MCU:** Un Core i7 x86-64 oculta el tamaño de LUTs, colisiones L1, ausencia de FPU/SIMD o uso real de Flash/SRAM en un Cortex-M o RISC-V.

### 3.4 Lista Exacta de Portabilidad Pre-MCU (4 Pasos de Desktop C99 $\to$ Firmware C)

1. **Reutilizar el núcleo sin heap de [`entropy_pyramid.c`](src/features/native/entropy_pyramid.c#L26-L31):** Su ruta `small` ya opera íntegramente en pila (`1,024 × uint16_t` finos + `384 × uint32_t` gruesos = **`3,584 B ≈ 3.5 KiB`** para `[1,2,4,8], bins=16`), cero `malloc/free` y cero dependencias del SO. Incluso `qlut[256]` desaparece usando `px >> 4` inline cuando `bins=16`. Si el MCU no tiene `16 KiB` libres para el frame `128x128`, se pueden acumular filas por tiras de `16` píxeles (`2,048 B`, una fila de celdas $8\times8$) acumulando el nivel fino al vuelo.
2. **Mover `asymmetry_g2/4/8` a C99:** Portar las 9 diferencias `LR/TB/R180` sobre `float E[85]` dentro del mismo kernel C elimina el `74%` restante de latencia Python en desktop (`0.24 ms → ~0.06–0.08 ms`) y deja el extractor $\mathbf{x}_{94}$ 100% autocontenido en C.
3. **Exportar `lr_dot` como array estático `const`:** `LinearDotCoreClassifier` reduce todo el clasificador + normalización a **`94 + 1` pesos (`380 B` en `float32`, o `~190 B` en `int16_t` Q15)**.
4. **Compactar `nlogn_lut` para Flash embebido:** En PC, [`prepare_native_buffers`](src/features/native_entropy.py#L355-L364) asigna `(128/1)^2 + 1 = 16,385` entradas `float64` (**`131,080 B ≈ 128 KiB`**) solo para cubrir la celda `1x1` (`16,384` píxeles). En firmware MCU, el nivel `8x8` (`64` celdas, `1,024` cuentas) solo necesita **`257` entradas `float32` (`1,028 B` en Flash)**, y las **`21` celdas gruesas** (`4x4 + 2x2 + 1x1` $\to$ `336` iteraciones totales) pueden evaluarse con `c * log2f((float)c)` o aritmética punto-fijo sin ocupar `128 KiB` en Flash.

---

## 4. Brechas Abiertas Antes del Artículo Final y Benchmark Real en MCU

1. **`M2` (Calibración y Guarda No-Degenerada en CNNs + Doble Umbral):** Añadir una guarda en [`find_optimal_threshold`](src/models/cnn/evaluate.py#L54-L93) cuando una predicción degenera a todo-positivo en validación pequeña (`bottle` `N_val=35`) y reportar ambos regímenes ($\tau = 0.50$ fijo y $\tau^*$ validado / OOB) tanto en ML como en CNNs.
2. **`M3` (Cobertura Multi-Categoría MVTec-AD):** Completar entrenamiento CNN y benchmark CPU `486c111` en `mvtec-ad_tile` y categorías complementarias (p. ej. `zipper`, `pill`, `transistor`) para reforzar el análisis de regímenes estructurales vs textura.
3. **Determinismo de `DecisionTreeClassifier`:** Añadir `random_state=cfg.seed` en [`build_classifier`](src/models/descriptor_classifier.py#L931-L937) para eliminar variaciones menores por empates entre corridas (`0.6364` vs `0.5714` en `bottle_dt`).
4. **Completar Port C99 de `asymmetry_g2/4/8` + `lr_dot` y Medir en MCU Real:** Añadir la medición física en microcontrolador (Flash, SRAM pico, ciclos/latencia y energía por fotograma) y explorar grillas más finas (`16×16`) o selectores dinámicos top-$k$/umbral (`T_total = T_selector + T_model`).
