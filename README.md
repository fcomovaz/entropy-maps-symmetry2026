# Entorno, guía de reproducción y ejecución

La síntesis científica vigente (veredictos de hipótesis $H_{\text{sym}}, H_{\text{inv}}, H_{\text{pareto}}, H_{\text{univ}}, H_{\text{opt-1}}, H_{\text{opt-2}}$, tablas 100 % CPU y análisis pre-MCU Python vs. C99 vs. CNNs) vive en [HYPOTHESIS.md](HYPOTHESIS.md). El mapa técnico del código, grafo de dependencias y resumen de laboratorios vive en [RESUME.md](RESUME.md), y las reglas de trabajo en [AGENTS.md](AGENTS.md).

> [!IMPORTANT]
> - Todos los comandos formales y tests (`-m src.*`, `-m experiments.*`, `-m unittest`) se ejecutan en PowerShell desde la raíz del repositorio (`code/`, localmente `D:\Publications\2026-10-Symmetry-Entropy\code`) usando `.\.venv\Scripts\python.exe`. Si el entorno ya existe y pasa las pruebas, no reinstalarlo.
> - Cifras de referencia medidas en: Windows 10, Python 3.11.9, PyTorch 2.14.0+cu126, NVIDIA GTX 1050 (`sm_61`), Intel Core i7-7700HQ @ 2.80 GHz, `seed = 42`. Las métricas de ML clásico deben coincidir estrechamente; las métricas CNN y todas las latencias dependen de hardware y driver (comparar órdenes, tendencias y razones, no dígitos exactos).
> - Nota sobre `dt`: `DecisionTreeClassifier` se construye sin `random_state` en `build_classifier`, por lo que `dt` puede variar ligeramente entre ejecuciones (p. ej. `bottle_dt` `0.6364` vs `0.5714`).

---

## Comprobación al retomar

```powershell
Set-Location D:\Publications\2026-10-Symmetry-Entropy\code
.\.venv\Scripts\python.exe tests/test_environment.py
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover tests -v
.\.venv\Scripts\python.exe understanding/03_entropy_benchmark/test_entropy.py
.\.venv\Scripts\python.exe understanding/04_lightweight_cnns/tests/test_models.py
.\.venv\Scripts\python.exe understanding/05_symmetry_features/tests/test_symmetry_features.py
.\.venv\Scripts\python.exe understanding/06_entropy_pyramid_descriptor/tests/test_descriptor.py
$env:PYTHONPATH = (Get-Location).Path; .\.venv\Scripts\python.exe understanding/07_descriptor_behavior/tests/test_behavior_helpers.py
.\.venv\Scripts\python.exe -m unittest understanding/13_entropy_maps_optimization/tests/test_entropy_variants.py -v
```

Resultado esperado: `Environment verification: PASS`, `No broken requirements found.`, **208 tests formales** en 22 módulos de `tests/` ([tests/README.md](tests/README.md)) y **86 tests de laboratorio** (`5` en `03`, `8` suites/`140` subcasos en `04`, `20` en `05`, `25` en `06`, `15`/`279` escenas en `07`, `13` en `13`). En este equipo los tests CUDA deben ejecutarse, no omitirse; `test_environment.py` exige CUDA real.

Los tests de entropía `03` necesitan las salidas de `01`, ya conservadas; si faltan, ejecutar primero `understanding/01_patch_partition/patch_partition.py --patch-size 64`.

---

## Entorno registrado

Entorno verificado originalmente el 2026-09-26 mediante operaciones reales en GPU:

| Componente | Valor |
| --- | --- |
| OS | Windows 10, build 19045, x86-64 |
| Python | 3.11.9, 64 bits |
| PyTorch | 2.14.0+cu126 |
| torchvision | 0.29.0+cu126 |
| CUDA runtime de PyTorch | 12.6 |
| CUDA disponible | True |
| GPU | NVIDIA GeForce GTX 1050 |
| Compute capability | (6, 1), Pascal |
| VRAM total | 4096 MiB; PyTorch informa ~4.00 GiB |
| Driver NVIDIA | 582.66 |

---

## Instalación reproducible y backend nativo C99

**Por qué importa el backend C99 para reproducibilidad.** El parámetro `entropy_backend = "auto"` (predeterminado en los TOML) resuelve a **`c_single_pass` si existe la DLL compilada; en caso contrario cae a `numpy_finest_pyramid`**. Ambos son matemáticamente equivalentes (paridad $\le 1.89 \times 10^{-15}$), pero las afirmaciones de latencia de [HYPOTHESIS.md](HYPOTHESIS.md) requieren la DLL compilada (`numpy_fused_entropy` nunca es elegido por `auto`).

Desde la raíz `code/`, crear `.venv` **solo si no existe**, instalar PyTorch `cu126` primero, después `requirements.txt`, validar CUDA y compilar `entropy_pyramid.c` (MSVC x64):

```powershell
Set-Location D:\Publications\2026-10-Symmetry-Entropy\code
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install --no-cache-dir --default-timeout=1000 --progress-bar off --log results/pytorch_cuda_install.log torch==2.14.0+cu126 torchvision==0.29.0+cu126 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe tests/test_environment.py

# Compilar kernel C99 de una sola pasada (src/features/native/entropy_pyramid.c)
.\.venv\Scripts\python.exe -m src.features.build_native --config configs/features/native_entropy.toml --clean
.\.venv\Scripts\python.exe -m unittest tests/test_native_entropy.py -v
```

**Expectativa:** `src/features/native/build/entropy_pyramid.dll` existe y los 5 tests de `test_native_entropy.py` pasan.

**Notas del entorno:**
* Instalar PyTorch **antes** de `requirements.txt`: los sufijos `+cu126` fijan la variante CUDA 12.6 (que incluye `sm_61` para GTX 1050 y actualizó `sympy==1.14.0`), mientras que PyPI estándar no distribuye esas wheels.
* No usar builds CUDA 13.x en esta Pascal (la versión 13.0 de `nvidia-smi` indica capacidad del driver, no el runtime 12.6 de PyTorch). No hace falta CUDA Toolkit global.
* No lanzar instalaciones `pip` concurrentes ni interrumpir la descarga (la wheel `torch` ocupa ~2.6 GB). Diagnóstico original en [results/environment_debug.md](results/environment_debug.md).
* Para ejecución remota en Google Colab a partir de datos preprocesados, existe el cuaderno [notebooks/Entropy_Maps_Full_Run.ipynb](notebooks/Entropy_Maps_Full_Run.ipynb) con [configs/cnn/colab.toml](configs/cnn/colab.toml).

---

## Carpetas (relativas a `code/`)

* [configs/](configs/base.toml): configuraciones TOML declarativas (`base.toml`, `preprocessing/`, `csv/`, `entropy/`, `asymmetry/`, `classification/`, `cnn/`, `profiling/`, `features/`, `optimization/`) y `logging.py`. Ningún módulo de `src/` hardcodea parámetros o rutas.
* [src/](src/README.md): módulos formales reutilizables (`preprocessing/`, `datasets/`, `features/` incl. C99 `entropy_pyramid`, `models/` incl. `LinearDotCoreClassifier` y `cnn/`, `profiling/`).
* [experiments/](experiments/README.md): puntos de entrada CLI para todas las corridas reproducibles.
* [understanding/](understanding/README.md): trece laboratorios pedagógicos (`01_patch_partition/` a `13_entropy_maps_optimization/`, más `07_entropy_maps_vectorized/`).
* [tests/](tests/README.md): verificación CUDA (`test_environment.py`) y 22 suites formales (`208` casos `unittest`).
* [results/](results/README.md): diagnósticos, logs, clasificación (`results/classification/`), checkpoints CNN (`results/cnn/`), comparación 100 % CPU justa con C99 (`results/comparison_cpu/`), comparativa Python puro (`results/comparison_cpu_python/`, ignorada por Git), profiling científico (`results/profiling/`) y legado `comparison/` (*deprecated*).
* `notebooks/`: cuaderno de ejecución completa en Colab ([Entropy_Maps_Full_Run.ipynb](notebooks/Entropy_Maps_Full_Run.ipynb)).
* [data/](data/README.md): `raw/` (no incluido en Git), `preprocessed/` y `csv/`.
* `memory/`: exportador de `README*.md` a `memory/` para sincronización con NotebookLM (`export_readmes.ps1`).

---

## Reproducción end-to-end del pipeline formal (`HYPOTHESIS.md`)

Esta sección recorre las corridas formales mínimas desde los **datasets crudos** hasta cada cifra y tabla reportada en [HYPOTHESIS.md](HYPOTHESIS.md). Todos los CLI admiten `--dry-run` para validar conteos y rutas sin escribir en disco.

### 0. Entradas crudas y layout esperado

Colocar los datasets originales (no distribuidos en este repositorio) en:

```text
data/raw/mvtec-ad/<category>/{train/good, test/<defect>, ground_truth/<defect>}   # 15 categorías
data/raw/KolektorSDD2/{train,test}/N.png + N_GT.png                              # stems numéricos; se ignoran copias "(copy)"
```

Rutas, lista de objetos y salidas se derivan de [configs/base.toml](configs/base.toml).

### 1. Preprocesamiento (imágenes crudas → `128×128` escala de grises)

**Qué hace.**
1. Convierte a grises (luma ITU-R 601): $Y = 0.299R + 0.587G + 0.114B$.
2. *Solo variante invertida (`--inverted`):* $I' = 255 - I$ aplicado **antes** del padding para que el relleno conserve `0`.
3. Reescalado preservando aspecto para que el lado mayor mida `128` (imágenes: `LANCZOS`; máscaras MVTec: `NEAREST`, manteniendo valores en $\{0, 255\}$).
4. *Letterbox* centrado sobre lienzo `128×128` con `pad_value = 0`.

Orden fijo: **grises → inversión → reescalado → letterbox.** Las etiquetas de KolektorSDD2 provienen de la máscara nativa: $\text{defect} \iff \max(\text{mask}) > 0$; las máscaras no se guardan en la salida.

```powershell
.\.venv\Scripts\python.exe -m experiments.preprocess_mvtec_ad
.\.venv\Scripts\python.exe -m experiments.preprocess_mvtec_ad --inverted
.\.venv\Scripts\python.exe -m experiments.preprocess_kolektorsdd2
.\.venv\Scripts\python.exe -m experiments.preprocess_kolektorsdd2 --inverted
```

**Qué esperar:** `data/preprocessed/{mvtec-ad, mvtec-ad-inverted, KolektorSDD2, KolektorSDD2-inverted}/`.
KolektorSDD2: **3,335 pares** → train `2,085 good / 246 defect`, test `894 good / 110 defect`, 2 archivos huérfanos/copia omitidos, sin máscaras en salida. MVTec-AD: todas las imágenes `128×128` modo `'L'`, máscaras estrictamente en $\{0, 255\}$.

Verificación opcional (genera matrices visuales comparativas):
`understanding/08_verify_mvtec_preprocessing/verify.py` y `understanding/10_verify_kolektorsdd2/verify.py` (`+ --inverted`).

### 2. Exportación tabular CSV de píxeles

**Qué hace.** Aplana cada imagen `128×128` por filas en `px_00000 … px_16383`, junto con metadatos y `width,height`.

```powershell
.\.venv\Scripts\python.exe -m experiments.export_mvtec_ad_csv
.\.venv\Scripts\python.exe -m experiments.export_mvtec_ad_csv --inverted
.\.venv\Scripts\python.exe -m experiments.export_kolektorsdd2_csv
.\.venv\Scripts\python.exe -m experiments.export_kolektorsdd2_csv --inverted
```

| Archivo (`data/csv/`) | Columnas | Filas | Columnas de metadatos |
|---|---:|---:|---|
| `mvtec-ad[-inverted].csv` | 16,391 | 6,612 *(incluye filas de máscara `ground_truth`)* | `object_name, split, type, defect_name, file_name` |
| `kolektorsdd2[-inverted].csv` | 16,389 | 3,335 | `split, type, file_name` |

Comprobación *round-trip* opcional (debe reportar diferencia `0`): `understanding/09_verify_mvtec_csv/verify.py` y `understanding/11_verify_kolektorsdd2_csv/verify.py` (`+ --inverted`).

### 3. Mapas de entropía multiescala ($\mathbf{x}_{85}$)

**Qué hace.** Para cada nivel de grilla $g \in \{1, 2, 4, 8\}$ divide la imagen en $g \times g$ celdas ($1 + 4 + 16 + 64 = 85$ celdas). Por celda con $N$ píxeles:
* Cuantiza: $b = \lfloor I \cdot B / 256 \rfloor$, con $B = 16$ bins.
* Probabilidades: $p_b = n_b / N$.
* Entropía de Shannon normalizada:
  $$E = \frac{-\sum_b p_b \log_2 p_b}{\log_2 B} \in [0, 1]$$

Implementación: un único histograma fino $8 \times 8$, niveles gruesos por suma *bottom-up* y entropía mediante tablas LUT $c \log_2 c$.

```powershell
.\.venv\Scripts\python.exe -m experiments.export_entropy_maps --config configs/entropy/mvtec_ad.toml
.\.venv\Scripts\python.exe -m experiments.export_entropy_maps --config configs/entropy/mvtec_ad.toml --inverted
.\.venv\Scripts\python.exe -m experiments.export_entropy_maps --config configs/entropy/kolektorsdd2.toml
.\.venv\Scripts\python.exe -m experiments.export_entropy_maps --config configs/entropy/kolektorsdd2.toml --inverted
```

**Qué esperar:** `data/csv/*-entropy.csv` + sidecar `.json` (verificar que `resolved_backend` diga `c_single_pass`). Columnas: metadatos originales, `bins`, `grid_levels` (`1|2|4|8`) y `E_1x1_r0_c0 … E_8x8_r7_c7` (85 valores en $[0, 1]$ a 8 decimales; celdas de solo padding tienen entropía `0`).

### 4. Residuos de asimetría espacial → $\mathbf{x}_{94}$ (`schema x94-v1`)

**Qué hace.** Reconstruye las entropías de cada nivel $g \in \{2, 4, 8\}$ como matriz $E^{(g)} \in \mathbb{R}^{g \times g}$ y mide su desviación respecto a tres simetrías involutivas $T \in \{\text{LR}, \text{TB}, \text{R180}\}$:

$$A_T^{(g)} = \frac{1}{g^2} \sum_{r,c} \left| E^{(g)}_{r,c} - T(E^{(g)})_{r,c} \right|$$

donde $T_{\text{LR}}: c \mapsto g-1-c$, $T_{\text{TB}}: r \mapsto g-1-r$ y $T_{\text{R180}}: (r,c) \mapsto (g-1-r, g-1-c)$. Anexa 9 columnas en orden `asym_{lr,tb,rot180}_{2x2,4x4,8x8}`, conformando $\mathbf{x}_{94} = [\mathbf{x}_{85}, \mathbf{x}_9]$ (el nivel $1\times 1$ no tiene asimetría).

```powershell
.\.venv\Scripts\python.exe -m experiments.append_asymmetry_features --all
```

**Qué esperar:** cuatro archivos `data/csv/*-entropy-full.csv` (+ `.json`): `mvtec-ad`, `mvtec-ad-inverted`, `kolektorsdd2` y `kolektorsdd2-inverted`. Metadatos y 85 entropías idénticas byte a byte a los `-entropy.csv`; 94 columnas de características en total.

### 5. Clasificadores clásicos, ablación y explicabilidad

**Qué hace.** Entrena `dt`, `knn`, `lr`, `lr_dot` y `rf` sobre $\mathbf{x}_{94}$ (`good = 0`, `defect = 1`) y ejecuta ablación automática sobre `x85_only`, `x9_only`, `x94_full`, `E_1x1`, `E_2x2`, `E_4x4` y `E_8x8`.

* **Splits:** KolektorSDD2 usa el split oficial `train`/`test` (`presplit`). MVTec-AD descarta en memoria filas `split == "ground_truth"` y aplica partición estratificada `80/20` en memoria por objeto (`seed = 42`), sin modificar los CSV en disco (el `train` oficial de MVTec solo contiene piezas `good`).
* **Predicción:** umbral fijo $\hat{p} \ge 0.50$. **`lr_dot`** (`LinearDotCoreClassifier`) pliega algebraicamente `StandardScaler` + `LogisticRegression` en un único producto punto: $w'_i = w_i / \sigma_i$, $b' = b - \sum_i w_i \mu_i / \sigma_i$ (numéricamente idéntico a `lr` $< 10^{-12}$, `95` parámetros).
* **Ganancia de simetría y azar:** $\Delta F_1 = F_1(\mathbf{x}_{94}) - F_1(\mathbf{x}_{85})$. La cuota por azar de $\mathbf{x}_9$ es $9 / 94 = 9.57\,\%$.

```powershell
# KolektorSDD2
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/kolektorsdd2.toml --classifier all --overwrite
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/kolektorsdd2.toml --classifier all --inverted --overwrite

# MVTec-AD: objeto canónico bottle (requerido para benchmark CPU)
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --object bottle --classifier all --overwrite
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --object bottle --classifier all --inverted --overwrite

# MVTec-AD: barrido completo de las 15 categorías (RF)
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --all-objects --classifier rf
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --all-objects --classifier rf --inverted
```

**Salidas en `results/classification/<dataset>[_inverted]/<object|full>_<clf>/`:** `metrics_summary.json`, `confusion_matrix_and_curves.png`, `descriptor_ablation_x85_x9_x94.png`, `ablation_table.csv`, `explainability_importance_breakdown.png`, `feature_importance.csv`, `scale_and_family_breakdown.csv`, `spatial_importance_heatmaps.png` y (`--all-objects`) `mvtec_all_objects_rf_summary.csv/.png`.

**Valores de referencia (post-merge, `seed = 42`; [HYPOTHESIS.md §2.4](HYPOTHESIS.md#24-multi-scale-ablation--explainability-mathbfx_85-vs-mathbfx_9-at-486c111)):**

| Dataset | Clf | `x85` F1 / AUC | `x9` F1 / AUC | `x94` F1 / AUC | $\Delta F_1$ |
|---|---|---|---|---|---:|
| `kolektorsdd2` | `lr` / `lr_dot` | 0.2687 / 0.6179 | 0.3591 / 0.7371 | 0.4265 / 0.8061 | +0.1579 |
| `kolektorsdd2` | `rf` (200 árboles) | 0.4286 / 0.8255 | 0.3765 / 0.7429 | 0.4298 / 0.7904 | +0.0013 |
| `mvtec bottle` | `lr` / `lr_dot` | 0.7333 / 0.9181 | 0.7333 / 0.9130 | 0.8000 / 0.9231 | +0.0667 |
| `mvtec bottle` | `rf` (100 árboles) | 0.5263 / 0.8696 | 0.6923 / 0.8980 | 0.7273 / 0.9030 | +0.2010 |

En MVTec invertido, la entropía sin padding es invariante por conmutatividad de $-\sum p_b \log_2 p_b$; en KolektorSDD2 invertido el relleno negro fijo (`pad=0`) introduce una deriva menor (`lr_dot` F1 `0.4265 -> 0.4153`).

### 6. Zoo de CNN ligeras (mismos splits que etapa 5)

**Qué hace.** Entrena 7 arquitecturas sobre PNGs `128×128`: `ultra_tiny` (`1,154` params), `ds_cnn` (`12,866`), `tiny_cnn` (`23,538`), `mobilenet_like` (`70,786`), `tiny_resnet` (`77,234`), `small_cnn` (`97,650`), `torchvision_mobilenet` (`1,519,618`, init aleatoria). Configuración: AdamW, `lr = 1e-3`, scheduler coseno, 40 épocas, early stopping 8, pesos balanceados, `hflip`, `seed = 42` (MVTec añade split de validación 15 %).

**Calibración de umbral (`val_f1`).** La pérdida balanceada desplaza el logit por $\log(w_1/w_0)$. Por defecto `val_f1` elige $\tau^* = \arg\max_\tau F_1^{\text{val}}$ solo sobre validación (el JSON también reporta $\tau = 0.50$). Con validaciones muy pequeñas (`bottle`, $N_{\text{val}} = 35$) `val_f1` puede colapsar a "todo defecto" (efecto documentado en 5 de las 7 CNNs en `bottle`).

> [!WARNING]
> Pasar siempre `--seeds 42` para escribir en la carpeta canónica (corridas multi-seed sobrescriben la misma carpeta). Usa CUDA automáticamente si está disponible (`device = "auto"`).

```powershell
# KolektorSDD2: 7 CNNs, normal e invertido
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/kolektorsdd2.toml --model all --seeds 42 --overwrite
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/kolektorsdd2.toml --model all --seeds 42 --inverted --overwrite

# MVTec-AD bottle (requerido para etapa 7), normal e invertido
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/mvtec_ad.toml --object bottle --model all --seeds 42 --overwrite
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/mvtec_ad.toml --object bottle --model all --seeds 42 --inverted --overwrite

# Opcional: las 15 categorías de MVTec-AD
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/mvtec_ad.toml --all-objects --model all --seeds 42 --overwrite
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/mvtec_ad.toml --all-objects --model all --seeds 42 --inverted --overwrite
```

**Referencia (`seed = 42`, [HYPOTHESIS.md §2.1](HYPOTHESIS.md#21-official-100-cpu-comparison-c_single_pass--ml-vs-7-cnns-threads1)):**
* `kolektorsdd2`: mejor `tiny_resnet` (`F1 = 0.7130`, `ROC-AUC = 0.9213`; las 7 CNNs superan aquí a `lr_dot` `F1 = 0.4265`).
* `bottle`: solo `mobilenet_like` evita colapso de umbral (`F1 = 0.7586`, `ROC-AUC = 0.9415`); las demás quedan en `F1 ≈ 0.36 / BalAcc ≈ 0.50`.
* `inverted`: `mobilenet_like` en `bottle` colapsa (`F1 = 0.7586 -> 0.3611`), mientras `lr_dot` conserva `F1 = 0.8000`.

### 7. Benchmark 100 % CPU justo (Fase A) y profiling profundo (Fase B)

**Qué hace.** Cronometra el pipeline completo en CPU, `threads = 1`, sobre **200 imágenes reales de test** (`seed = 42`), con `gc` deshabilitado, 20 warmups, 200 iteraciones cronometradas, 3 repeticiones e IC 95 % bootstrap (1000 remuestreos):

$$T_{\text{total}} = T_{\text{descriptor}} + T_{\text{model}}\;(\text{ML}), \qquad T_{\text{total}} = T_{\text{preprocess}} + T_{\text{model}}\;(\text{CNN})$$

Las CNNs **no se reentrenan**: se cargan los checkpoints de la etapa 6 en CPU y deben coincidir con `metrics_summary.json` dentro de `1e-5` (si difieren, aborta). Fase B añade tiempos por etapa de `x94`, capas PyTorch/ops ATen, pruebas Mann-Whitney U y ablación $\Delta F_1 / \Delta\text{ms}$.

```powershell
# Fase A — Benchmark 100 % CPU justo (backend auto -> c_single_pass)
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/kolektorsdd2.toml
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/kolektorsdd2.toml --inverted
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/mvtec_ad.toml --objects bottle
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/mvtec_ad.toml --objects bottle --inverted

# Fase B — Profiling granular, significancia Mann-Whitney U y ablación costo-calidad
.\.venv\Scripts\python.exe -m experiments.profile_pipelines --config configs/profiling/kolektorsdd2.toml
.\.venv\Scripts\python.exe -m experiments.profile_pipelines --config configs/profiling/mvtec_ad.toml --objects bottle
.\.venv\Scripts\python.exe -m experiments.profile_pipelines --config configs/profiling/kolektorsdd2.toml --inverted
.\.venv\Scripts\python.exe -m experiments.profile_pipelines --config configs/profiling/mvtec_ad.toml --objects bottle --inverted
```

**Qué esperar:**
* `results/comparison_cpu/`: `<tag>_baseline_vs_cnn_cpu.csv`, `<tag>_cuda_reference.csv` (separado, nunca mezclado), figuras Pareto y `summary_all_datasets.csv`.
* `results/profiling/<tag>/`: `environment.json`, `descriptor_stage_times.csv`, `classifier_times.csv`, `cnn_layer_times_<model>.csv`, `statistical_tests.csv`, `ablation_quality_vs_cost.csv`, 6 figuras y `README.md` autogenerado.

**Referencia (i7-7700HQ, `threads = 1`, `B = 1`; [HYPOTHESIS.md §2.1](HYPOTHESIS.md#21-official-100-cpu-comparison-c_single_pass--ml-vs-7-cnns-threads1)):**

| Pipeline | `kolektorsdd2` $T_{\text{total}}$ (ms) | `bottle` $T_{\text{total}}$ (ms) |
|---|---:|---:|
| `x94` + `lr_dot` (C99 `x85`) | **0.3799** | **0.3002** |
| `ultra_tiny` | 0.9919 | 0.9171 |
| `mobilenet_like` | 4.0077 | 4.1313 |
| `rf` | 10.9062 | 5.7947 |

Esperar el mismo ordenamiento (`lr_dot` el más rápido, `2.67×–3.64×` más rápido que `ultra_tiny` en profiling con `p < 1e-50`; `lr_dot` `33×–51×` más rápido que `sklearn lr`), no milisegundos idénticos.

> [!NOTE]
> `results/comparison/` (`compare_baseline_vs_cnn.py`) mezcla latencia CNN en CUDA con ML en CPU y está **deprecated**; no usarlo para comparaciones de latencia.

### 8. Opcional: comparativa pre-MCU en Python puro

Aísla lo que se pierde al no usar la DLL C99 (salida local ignorada por Git):

```powershell
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/kolektorsdd2.toml --entropy-backend numpy_finest_pyramid --output-dir results/comparison_cpu_python --no-cuda-reference
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/mvtec_ad.toml --objects bottle --entropy-backend numpy_finest_pyramid --output-dir results/comparison_cpu_python --no-cuda-reference
```

**Referencia esperada:** `lr_dot` total $\approx 0.795\text{ ms}$ (`kolektorsdd2`) / $0.972\text{ ms}$ (`bottle`) con `numpy_finest_pyramid`, y $\approx 2.81\text{–}2.87\text{ ms}$ con `numpy_baseline` (más lento que `ultra_tiny`). La calidad predictiva es idéntica al backend C99.

### 9. Dónde leer resultados y límites conocidos

| Consulta | Documento |
|---|---|
| Veredictos de hipótesis, tablas y checklist MCU | [HYPOTHESIS.md](HYPOTHESIS.md) |
| Explicación de figuras y tablas de clasificación | [results/classification/README.md](results/classification/README.md) |
| Hallazgos de profiling por dataset | `results/profiling/<tag>/README.md` |
| Todos los flags CLI por script | [experiments/README.md](experiments/README.md) |
| Mapa de código, grafo y reglas que muerden | [RESUME.md](RESUME.md) |
| Reglas de desarrollo del repositorio | [AGENTS.md](AGENTS.md) |

**Límites abiertos:** los tiempos corresponden a x86-64 de escritorio (no MCU físico); $\mathbf{x}_{94}$ pierde en F1 frente a CNNs en microfisuras sub-celda (`kolektorsdd2`, $H_{\text{univ}}$ refutada); quedan pendientes el port C99 de `asymmetry_g2/4/8` + `lr_dot`, barridos multi-seed / multi-categoría con guarda de no-degeneración en `val_f1` y selectores dinámicos de regiones.

---

## Repetir 01-03 sin sobrescribir evidencia

Los scripts `01`–`03` guardan junto a su propia ubicación y no admiten `--output`. Crear una copia de ejecución en `%TEMP%` permite repetirlos conservando intactos los resultados canónicos y los hashes del código:

```powershell
Set-Location D:\Publications\2026-10-Symmetry-Entropy\code
$projectPython = (Resolve-Path .\.venv\Scripts\python.exe).Path
$replayRoot = Join-Path $env:TEMP ('symmetry-replay-' + [guid]::NewGuid().ToString('N'))
$replayLabs = Join-Path $replayRoot 'code\understanding'
New-Item -ItemType Directory -Path $replayLabs -Force | Out-Null
foreach ($lab in @('01_patch_partition', '02_histogram_benchmark', '03_entropy_benchmark')) {
    Copy-Item -LiteralPath (Join-Path '.\understanding' $lab) -Destination $replayLabs -Recurse
}
Write-Output "Copia de ejecución: $replayLabs"

& $projectPython "$replayLabs\01_patch_partition\patch_partition.py" --patch-size 64
if ($LASTEXITCODE -ne 0) { throw 'Falló la partición; detener la reproducción.' }
& $projectPython "$replayLabs\03_entropy_benchmark\test_entropy.py"
if ($LASTEXITCODE -ne 0) { throw 'Fallaron los tests de entropía.' }
```

Para una comprobación corta de ambos benchmarks sobre la misma copia:

```powershell
& $projectPython "$replayLabs\02_histogram_benchmark\histogram_benchmark.py" --target-pixels-per-run 10000
if ($LASTEXITCODE -ne 0) { throw 'Falló el benchmark de histogramas.' }
& $projectPython "$replayLabs\03_entropy_benchmark\entropy_benchmark.py" --rounds 1 --entropy-repeats 2 --target-pixels 1000
if ($LASTEXITCODE -ne 0) { throw 'Falló el benchmark de entropía.' }
& $projectPython "$replayLabs\03_entropy_benchmark\analyze_results.py"
if ($LASTEXITCODE -ne 0) { throw 'Falló la validación de resultados.' }
```

Para el protocolo completo, usar otra copia y omitir las banderas de duración (tarda varios minutos; la corrida corta reduce repeticiones y sus tiempos no sustituyen las tablas históricas). `analyze_results.py` valida SHA-256 contra el `run1/` copiado y falla si se alteran scripts o hashes.

La [guía de CNN de Lab 04](understanding/04_lightweight_cnns/README.md) documenta inspección, entrenamiento y benchmark del laboratorio pedagógico. Para abrir Jupyter opcionalmente:

```powershell
.\.venv\Scripts\jupyter.exe lab
```
