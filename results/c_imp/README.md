# Resultados y Verificación de Migración C99 (`results/c_imp/`)

Síntesis técnica y auditoría de resultados de la migración a **C99 portable pre-ESP32-CAM** (`float32` y punto fijo entero `Q15` / `int8_per_channel` sin `malloc` en inferencia), generados según la especificación de [../../tasks/task_11.md](../../tasks/task_11.md) mediante `experiments.export_c_pipeline`, `experiments.verify_c_pipeline` y `experiments.visualize_c_pipeline`.

Para instrucciones de compilación, opciones CLI y ejecución manual de los binarios nativos ver [../../c_imp/README.md](../../c_imp/README.md). **Estado vigente (`2026-11-26`):** los **16/16 targets** (`kolektorsdd2` + las 15 categorías de MVTec-AD) están exportados, verificados (`0` LSB raw diff en `Q15`, `f32_max_abs_diff <= 4.364e-05` bajo `1e-4`) y visualizados en disco (ver el consolidado 16-target en [../FINAL_PAPER_REPORT.md §6](../FINAL_PAPER_REPORT.md#6-c99-f32--q15-parity-latency--memory-summary)); las tablas de las secciones 3–4 abajo conservan el desglose por muestra de los targets principales `kolektorsdd2` y `mvtec_ad_bottle` (y las observaciones de formato/Pareto en §5 ya quedaron resueltas en la corrida del `2026-11-26`). Los archivos pesados bajo `results/c_imp/<target>/` (`vectors/`, `c_outputs/`, `figures/`, CSV/JSON locales) permanecen excluidos de Git; este documento consolida en texto y tablas los hallazgos verificados.

---

## 1. Resumen Ejecutivo de Auditoría (Estado en Disco)

| Aspecto Auditado | Estado | Evidencia / Hallazgo Principal |
| :--- | :--- | :--- |
| **Paridad C99 `Q15` vs espejo Python** | **PASS (Bit-exact)** | Diferencia entera máxima de **`0` LSB** (`q15_vs_py_q15_max_raw_diff = 0`) y `0` discrepancias de etiqueta C vs Python en `descriptor_x94` y los 9 modelos de `kolektorsdd2` (`N=1004`) y `mvtec_ad_bottle` (`N=59`). |
| **Paridad C99 `float32` vs Python** | **PASS (Bajo gate)** | Error máximo de `descriptor_x94`: `2.97e-08` (`kolektorsdd2`) y `4.729e-08` (`mvtec_ad_bottle`), bajo el gate `1e-05`. Error máximo de logits en modelos: `2.122e-05` (`kolektorsdd2`) y `2.682e-05` (`mvtec_ad_bottle`), bajo el gate `1e-04`. |
| **Presupuesto Flash (`<= 2048 KB`)** | **PASS (9/9 modelos)** | Todos caben en Flash configurada (`configs/c_imp/*.toml`): `0.19–6.07 KB` en modelos tabulares; `3.08–192.17 KB` en 6 CNNs `Q15`; `1595.42 KB` en `torchvision_mobilenet` mediante cuantización automática `int8_per_channel`. |
| **Presupuesto SRAM / PSRAM ESP32-CAM** | **Selectivo SRAM; 9/9 PSRAM** | Solo `lr_dot`, `mlp_16_dot` (sin arena dinámica de tensores) y `ultra_tiny` (`160.00 KB` en `Q15`, la mitad que en `float32`) caben dentro de SRAM interna (`200000 B ≈ 195.3 KB`). Las 6 CNNs restantes (`320–1056 KB` en `Q15`) requieren PSRAM (`<= 3000000 B ≈ 2930 KB`). |
| **LUT Shannon `log2_mantissa`** | **Verificada (`514 B`)** | Modo `CIMP_SHANNON_LUT_MODE = 2` (`257` entradas, `8` bits de mantisa en `lut_shannon.h`). Mantiene residual `Q0.15 + LUT vs float64` alrededor de `~7.6e-6` (media) y `<= 3.8e-5` (máx. en asimetrías `85..93`), reduciendo >100× frente a `nlogn_full` (`~64 KB`). |
| **Colapso `F1 = 0.361` en 5 CNN de `bottle`** | **Baseline PyTorch (no bug C)** | El comportamiento todo-positivo (`TP=13, FP=46, TN=0, FN=0`) ya existe en los checkpoints `float32` originales por calibración `val_f1` sobre `N_val=35` (brecha `M2` de [../../RESUME.md](../../RESUME.md)). |
| **Deriva de `torchvision_mobilenet`** | **Esperable (`int8`)** | Único modelo con pesos `int8_per_channel`: concentra la mayor deriva cuantizada en `kolektorsdd2` (`max|Δ| = 2.5122`, `3` flips) y en `mvtec_ad_bottle` genera `r = -0.054066` al colapsar todos sus logits `f32` cerca de `-0.747`. |
| **Observaciones menores de reporte / gráficos** | **Documentadas (§5)** | (1) `visualize_data.py` etiqueta `weight_format = q15` en `lr_dot` y `mlp_16_dot` aunque en `kolektorsdd2` se exportaron pesos de `32` bits (`weight_bits = 32`); (2) En `kolektorsdd2` `fig5` (Pareto), el offset de `tiny_cnn` `(+34, -14)` acerca su etiqueta al triángulo de `small_cnn`. |

---

## 2. Protocolo y Parámetros Comunes (`configs/c_imp/*.toml`)

- **Compilador desktop verificado:** `Microsoft (R) C/C++ Optimizing Compiler Version 19.51.36244 for x64` (Windows 10, MSVC single-thread `QueryPerformanceCounter`).
- **Aritmética fija:** `feature_frac_bits = 15` (`Q0.15` para `x94`), acumuladores `accumulator_bits = 64` (`int64_t`) con redondeo `half_up`, y LUT Shannon compacta `"log2_mantissa"` (`log2_mantissa_bits = 8`, `lut_shannon.h` = `514 B`).
- **Tolerancias y presupuestos MCU (`configs/c_imp/kolektorsdd2.toml` y `mvtec_ad.toml`):**
  - Gates de paridad `float32`: `f32_descriptor_abs_tol = 1e-5`, `f32_logit_abs_tol = 1e-4`.
  - Presupuestos ESP32-CAM objetivo (`520 KB SRAM / 4 MB Flash / 4 MB PSRAM` físicos; límites de trabajo en TOML): `flash_budget_bytes = 2097152` (`2048 KB`), `sram_budget_bytes = 200000` (`195.3 KB`), `psram_budget_bytes = 3000000` (`~2930 KB`).
- **Convención de latencias:** `T_total = T_selector + T_model` medida en **CPU x86 de escritorio**, no en Xtensa LX6 físico (Etapa 12 pendiente en hardware real). En x86, `Q15` usa aritmética escalar `int64` frente a instrucciones vectoriales SSE/AVX en `float32`, por lo que `bench_q15_total_ms` suele superar a `bench_f32_total_ms` en CNNs de escritorio sin reflejar la ganancia esperada en un MCU sin unidad SIMD.

---

## 3. Benchmark Principal: `kolektorsdd2` (`N = 1,004` test, `110` defectos / `894` normales)

### 3.1 Tabla Consolidada (`summary_models_metrics.csv` + `parity_report.json`)

| Modelo | Familia | Formato Pesos (Manifest) | Flash (KB) | Arena `Q15` (KB) | Cabe SRAM Int. | `F1` (`F32` → `Q15`) | `ΔF1` | `PR-AUC` `Q15` | `ROC-AUC` `Q15` | Flips (`Q15` vs `F32`) | Máx `|Δscore|` | Pearson `r` | `C F32` Máx `|diff|` | `C Q15` Raw Diff | `T_f32` Desk (ms) | `T_q15` Desk (ms) |
| :--- | :--- | :--- | ---: | ---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lr_dot` | tabular | `q15` *(pesos `w32`)* | 0.38 | — | Si | `0.4265 → 0.4253` | -0.0012 | 0.4361 | 0.8063 | 1 | 0.007416 | 0.999998 | 1.216e-05 | 0 | 0.0175 | 0.0182 |
| `mlp_16_dot` | tabular | `q15` *(pesos `w32`)* | 6.07 | — | Si | `0.4904 → 0.4889` | -0.0016 | 0.5935 | 0.8335 | 1 | 0.004668 | 1.000000 | 5.840e-06 | 0 | 0.0188 | 0.0191 |
| `ultra_tiny` | cnn | `q15` | 3.08 | 160.00 | **Si** | `0.6304 → 0.6304` | +0.0000 | 0.7060 | 0.8815 | 0 | 0.020553 | 0.999989 | 1.282e-06 | 0 | 0.6645 | 1.1999 |
| `ds_cnn` | cnn | `q15` | 27.02 | 1056.00 | No *(PSRAM)* | `0.6562 → 0.6562` | +0.0000 | 0.7142 | 0.8778 | 0 | 0.030366 | 0.999998 | 1.490e-06 | 0 | 7.4295 | 11.4150 |
| `tiny_cnn` | cnn | `q15` | 46.92 | 1056.00 | No *(PSRAM)* | `0.6154 → 0.6190` | +0.0037 | 0.6653 | 0.8713 | 1 | 0.013746 | 0.999993 | 2.428e-06 | 0 | 30.7080 | 37.9074 |
| `mobilenet_like` | cnn | `q15` | 143.14 | 512.00 | No *(PSRAM)* | `0.6250 → 0.6215` | -0.0035 | 0.6849 | 0.8623 | 1 | 0.020412 | 0.999997 | 1.900e-06 | 0 | 11.9738 | 16.7537 |
| `tiny_resnet` | cnn | `q15` | 152.67 | 416.00 | No *(PSRAM)* | `0.7130 → 0.7100` | -0.0031 | 0.7809 | 0.9213 | 1 | 0.043619 | 0.999998 | 2.951e-06 | 0 | 45.1884 | 45.3980 |
| `small_cnn` | cnn | `q15` | 192.17 | 1056.00 | No *(PSRAM)* | `0.5965 → 0.5965` | +0.0000 | 0.7108 | 0.8925 | 0 | 0.017221 | 1.000000 | 2.150e-06 | 0 | 61.0415 | 52.2334 |
| `torchvision_mobilenet` | cnn | `int8_per_channel` | 1595.42 | 320.00 | No *(PSRAM)* | `0.5976 → 0.5890` | -0.0086 | 0.5797 | 0.7411 | 3 | 2.512167 | 0.997133 | 2.122e-05 | 0 | 16.2058 | 19.3541 |

- **`descriptor_x94` (`parity_report.json`):** `f32_max_abs_diff = 2.97e-08`, `q15_vs_py_q15_max_raw_diff = 0`, `f32_label_mismatches = 0`, `q15_vs_py_q15_label_mismatches = 0`, `runtime_tests_passed = true`, `all_gates_passed = true`.
- **Matrices de confusión `Q15` (`fig3`, formato `[TP, FP, TN, FN]`):** `lr_dot` `[74, 164, 730, 36]`; `mlp_16_dot` `[77, 128, 766, 33]`; `ultra_tiny` `[58, 16, 878, 52]`; `ds_cnn` `[63, 19, 875, 47]`; `tiny_cnn` `[52, 6, 888, 58]`; `mobilenet_like` `[55, 12, 882, 55]`; `tiny_resnet` `[82, 39, 855, 28]`; `small_cnn` `[51, 10, 884, 59]`; `torchvision_mobilenet` `[48, 5, 889, 62]`.

### 3.2 Hallazgos Clave Verificados en `kolektorsdd2` (`fig1`–`fig6`)

1. **Gates `float32` y flips cercanos al umbral (`fig4`, `fig2`):**
   - La línea punteada `1e-05` de `fig4` evalúa únicamente a `descriptor_x94` (`2.97e-08`). Todos los clasificadores y CNNs se validan contra la línea roja de `1e-04`, incluyendo `lr_dot` (`1.22e-05`) y `torchvision_mobilenet` (`2.12e-05`).
   - Sobre `9,036` evaluaciones (`9 × 1,004`), hay exactamente **`8` flips `Q15/int8` vs `float32`**. Los `5` flips en modelos `Q15` puros están pegados al umbral calibrado (`<= 0.024`):
     - `lr_dot`: `20678.png` (`true=0`, `f32 = -0.001984 → q15 = +0.005432`).
     - `mlp_16_dot`: `20990.png` (`true=0`, `f32 = -0.001260 → q15 = +0.000229`).
     - `tiny_cnn`: `20728.png` (`true=0`, `f32 = -0.794251 → q15 = -0.801086`).
     - `mobilenet_like`: `21000.png` (`true=0`, `f32 = +0.159351 → q15 = +0.166931`).
     - `tiny_resnet`: `20576.png` (`true=0`, `f32 = +0.780221 → q15 = +0.804108`).
   - Los otros **`3` flips** pertenecen a `torchvision_mobilenet [int8_per_channel]` (`20821.png`: `true=1`, `+0.585587 → -1.597275`; `20192.png`: `true=0`, `+0.178302 → -0.466019`; `21000.png`: `true=0`, `-0.983352 → +1.054062`). La muestra `21000.png` con `thr = 0.0` es la responsable única de la marca vertical `Max flipped |score - thr| = 0.9834` en el panel D de `fig4`.
2. **Desglose de latencia y frontera de Pareto (`fig5`, `timing_benchmark.csv`):**
   - `x94_only`: `f32 = 0.017409 ms`, `q15 = 0.018302 ms`.
   - `lr_dot`: `f32` (`sel=0.017396 + mod=0.000148 = 0.017544 ms`), `q15` (`sel=0.018127 + mod=0.000082 = 0.018209 ms`).
   - `mlp_16_dot`: `f32` (`sel=0.017187 + mod=0.001637 = 0.018824 ms`), `q15` (`sel=0.018389 + mod=0.000717 = 0.019106 ms`).
   - La frontera de Pareto `Q15` consta de `5` modelos: `lr_dot` (`0.018 ms, F1=0.4253`), `mlp_16_dot` (`0.019 ms, F1=0.4889`), `ultra_tiny` (`1.200 ms, F1=0.6304`), `ds_cnn` (`11.415 ms, F1=0.6562`) y `tiny_resnet` (`45.398 ms, F1=0.7100`).
   - El tamaño en disco de las fuentes C99 generadas en `c_imp/generated/kolektorsdd2/` ronda `~37 MB` debido principalmente a `model_torchvision_mobilenet.c` (`~30 MB`), mientras sus pesos cuantizados en Flash ocupan `1595.42 KB`.
3. **Verificación física del descriptor `x94` (`fig6`):**
   - En los mapas `8×8`, las columnas laterales de relleno negro (`cols 0, 1, 6, 7`) permanecen en `~0.00`, los bordes de la tira colectora (`cols 2, 5`) marcan `~0.33–0.35`, y el incremento de entropía en defectos (`Defect - Normal`) se concentra dentro de la región central activa (máximo en fila `3`, columna `4`, `~+0.018`).
   - Las `9` características de asimetría muestran valores sistemáticamente superiores en defectos y ocupan los `9` primeros lugares por tamaño de efecto (`Cohen's d`), encabezados por `asym_tb_8x8` (`d ~ 0.88`), `asym_tb_4x4` (`~0.80`) y `asym_rot180_8x8` (`~0.74`).
   - La LUT `log2_mantissa` (`514 B`) produce residual nulo (`1e-8` piso de gráfico) en las columnas uniformes y error medio `~7.6e-6` / máx. `~1.5e-5` en entropías `0..84` (`<= 3.8e-5` máx. en asimetrías `85..93`).

---

## 4. Benchmark de Contraste: `mvtec_ad_bottle` (`N = 59` test, `13` defectos / `46` normales)

### 4.1 Tabla Consolidada (`summary_models_metrics.csv` + `parity_report.json`)

| Modelo | Familia | Formato Pesos (Manifest) | Flash (KB) | Arena `Q15` (KB) | Cabe SRAM Int. | `F1` (`F32` → `Q15`) | `ΔF1` | `PR-AUC` `Q15` | `ROC-AUC` `Q15` | Flips (`Q15` vs `F32`) | Máx `|Δscore|` | Pearson `r` | `C F32` Máx `|diff|` | `C Q15` Raw Diff | `T_f32` Desk (ms) | `T_q15` Desk (ms) |
| :--- | :--- | :--- | ---: | ---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lr_dot` | tabular | `q15` (`w16`) | 0.19 | — | Si | `0.8000 → 0.8000` | +0.0000 | 0.8963 | 0.9231 | 0 | 0.009050 | 1.000000 | 1.764e-05 | 0 | 0.0235 | 0.0216 |
| `mlp_16_dot` | tabular | `q15` (`w16`) | 3.10 | — | Si | `0.8000 → 0.8000` | +0.0000 | 0.9123 | 0.9498 | 0 | 0.035511 | 0.999998 | 2.682e-05 | 0 | 0.0261 | 0.0217 |
| `ultra_tiny` | cnn | `q15` | 3.08 | 160.00 | **Si** | `0.3611 → 0.3611` | +0.0000 | 0.3543 | 0.5301 | 0 | 0.000377 | 0.999930 | 1.870e-08 | 0 | 0.6768 | 1.1268 |
| `ds_cnn` | cnn | `q15` | 27.02 | 1056.00 | No *(PSRAM)* | `0.3611 → 0.3611` | +0.0000 | 0.4025 | 0.5552 | 0 | 0.000039 | 0.999729 | 1.500e-08 | 0 | 6.4281 | 9.5510 |
| `tiny_cnn` | cnn | `q15` | 46.92 | 1056.00 | No *(PSRAM)* | `0.3571 → 0.3571` | +0.0000 | 0.4107 | 0.6463 | 0 | 0.000024 | 0.999997 | 1.500e-08 | 0 | 30.4402 | 28.6309 |
| `mobilenet_like` | cnn | `q15` | 143.14 | 512.00 | No *(PSRAM)* | `0.7586 → 0.7857` | +0.0271 | 0.8901 | 0.9415 | 1 | 0.027777 | 0.999984 | 2.101e-06 | 0 | 13.1264 | 17.6962 |
| `tiny_resnet` | cnn | `q15` | 152.67 | 416.00 | No *(PSRAM)* | `0.3662 → 0.3611` | -0.0051 | 0.6310 | 0.7383 | 1 | 0.000104 | 0.999999 | 3.000e-08 | 0 | 42.2502 | 41.6804 |
| `small_cnn` | cnn | `q15` | 192.17 | 1056.00 | No *(PSRAM)* | `0.3611 → 0.3611` | +0.0000 | 0.4250 | 0.6639 | 0 | 0.000036 | 0.999998 | 3.000e-08 | 0 | 45.9538 | 42.8495 |
| `torchvision_mobilenet` | cnn | `int8_per_channel` | 1595.42 | 320.00 | No *(PSRAM)* | `0.3611 → 0.3611` | +0.0000 | 0.2182 | 0.4590 | 0 | 0.000163 | -0.054066 | 3.000e-08 | 0 | 13.1997 | 15.1847 |

- **`descriptor_x94` (`parity_report.json`):** `f32_max_abs_diff = 4.729e-08`, `q15_vs_py_q15_max_raw_diff = 0`, `f32_label_mismatches = 0`, `q15_vs_py_q15_label_mismatches = 0`, `runtime_tests_passed = true`, `all_gates_passed = true`.
- **Tiempos tabulares adicionales (`timing_benchmark.csv`):** `x94_only` (`f32 = 0.024093 ms`, `q15 = 0.022027 ms`), `lr_dot` (`f32 = 0.023521 ms`, `q15 = 0.021589 ms`), `mlp_16_dot` (`f32 = 0.026109 ms`, `q15 = 0.021742 ms`).

### 4.2 Hallazgos Clave Verificados en `mvtec_ad_bottle`

1. **Por qué 5 CNNs colapsan en `F1 = 0.3611` (`fig1`, `fig2`, `fig3`):**
   - Verificado directamente contra `results/cnn/mvtec-ad/bottle/*/metrics_summary.json`: `ultra_tiny`, `ds_cnn`, `small_cnn` y `torchvision_mobilenet` ya predicen `100%` positivos en PyTorch `float32` (`TP=13, FP=46, TN=0, FN=0`, `BalAcc = 0.5000`, `Accuracy = 0.2203`), causado por la calibración `val_f1` sobre `N_val = 35` (brecha `M2` de [../../RESUME.md](../../RESUME.md)).
   - En `tiny_resnet`, solo una muestra límite (`202.png`, `true=0`, `f32 = +0.663278 → q15 = +0.663376`) cambia de lado respecto al umbral (`~+0.66`), pasando de `TN=1` (`F1 = 0.3662`) a `TN=0` (`F1 = 0.3611`), mientras su `ROC-AUC = 0.7383` confirma señal ordenable ignorada por el umbral.
   - En sentido opuesto, `mobilenet_like` mejora en `Q15` (`0.7586 → 0.7857`) por `1` flip beneficioso (`157.png`, `true=0`, `f32 = -1.378421 → q15 = -1.394501`, quedando en `[TP=11, FP=4, TN=42, FN=2]`).
2. **Explicación de `score_pearson_r = -0.054066` en `torchvision_mobilenet` (`fig2`):**
   - Todos los `59` logits `float32` colapsan prácticamente en un punto constante alrededor de `-0.747`. Como la varianza real entre imágenes es menor que la perturbación de cuantización `int8_per_channel` (`max|Δ| = 0.000163`), la covarianza mide ruido numérico puro (`r = -0.054`), pese a que la paridad C99-vs-Python es exacta (`3.0e-08`, `0` flips).
3. **Superioridad de `x94` en `bottle` (`fig6`):**
   - Los mapas `8×8` reconstruyen la geometría radial de la botella (esquinas `0.00`, hombro/borde `~0.65–0.70`, interior `~0.05–0.15`), con desplazamiento claro en defectos internos (`~+0.08`). Las asimetrías alcanzan `Cohen's d ~ +2.0` (`asym_lr_8x8`) y entropías en borde/esquina alcanzan `d ~ -1.4` (`E_2x2_r0_c0`, `E_8x8_r3_c6`), explicando por qué `lr_dot` (`0.19 KB` Flash) y `mlp_16_dot` (`3.10 KB` Flash) alcanzan `F1 = 0.8000` (`ROC-AUC = 0.9231` y `0.9498`).

---

## 5. Detalles Menores de Reporte / Visualización Detectados para Futuro Refinamiento

1. **Etiqueta `weight_format` en tablas tabulares (`src/c_export/visualize_data.py`):** La columna `weight_format` en `summary_models_metrics.csv` y los rótulos de `fig1`/`fig5` muestran `q15` para `lr_dot` y `mlp_16_dot`, pero en `kolektorsdd2` su `export_manifest.json` seleccionó `weight_bits = 32` (`q31`, lo que duplica su tamaño Flash a `0.38 KB` y `6.07 KB` frente a `0.19 KB` y `3.10 KB` en `mvtec_ad_bottle`, donde sí usaron `weight_bits = 16`). Las entradas permanecen correctamente reportadas en bytes/KB reales; solo la etiqueta textual asume `"q15"`.
2. **Cruces de anotación en Pareto `kolektorsdd2` (`fig5` + `src/c_export/visualize_plots.py`):** `_annotate_pareto_labels` aplica un desplazamiento `(+34, -14)` pt a `tiny_cnn` (`37.907 ms, F1 = 0.6190`) que sitúa su etiqueta hacia la derecha-abajo junto al marcador de `small_cnn` (`52.233 ms, F1 = 0.5965`, offset `(+7, +8)`), cruzando visualmente la línea guía cerca del triángulo vecino.

---

## 6. Inventario de Artefactos Generados por Target (`results/c_imp/<target>/`)

| Ruta relativa | Contenido |
| :--- | :--- |
| `parity_report.json`, `parity_per_sample.csv` | Resumen y detalle por imagen de diferencias C99 vs Python (`f32` y `Q15` bit-exact) y aceptación contra gates. |
| `memory_report.json` | Contabilidad estática de Flash, arena de activaciones por análisis de *liveness* y cumplimiento SRAM/PSRAM. |
| `timing_benchmark.csv` | Latencias en escritorio (`T_total = T_selector + T_model`) en `f32` y `Q15`. |
| `vectors/` (`manifest.csv`, `images/`, `golden_x94_*.csv`) | Entradas y salidas de referencia exportadas por `export_c_pipeline`. |
| `c_outputs/` | Salidas crudas emitidas por `cimp_pipeline.exe`. |
| `figures/` | `fig1_classification_f32_vs_q15.png` .. `fig6_x94_descriptor_insights.png`, `summary_models_metrics.csv` y `summary_report.md`. |
