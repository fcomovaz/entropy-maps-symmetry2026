# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad_tile`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `99bc026`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.8649 | 0.9678 | 0.3529 (desc) | 0.0080 | **0.3632** | 0.0220 | [0.3616, 0.3643] | 2753.3 | - |
| `mlp_16_dot (x94)` | ml | cpu | 1537 / 94 | 0.8108 | 0.9767 | 0.3529 (desc) | 0.0123 | **0.3651** | 0.0160 | [0.3630, 0.3661] | 2738.6 | - |
| `dt (x94)` | ml | cpu | 15 / 94 | 0.8387 | 0.9623 | 0.3529 (desc) | 0.1327 | **0.4867** | 0.0232 | [0.4854, 0.4884] | 2054.7 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.8649 | 0.9678 | 0.3529 (desc) | 0.2003 | **0.5536** | 0.0258 | [0.5518, 0.5554] | 1806.2 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.6667 | 0.8768 | 0.0454 (prep) | 0.7812 | **0.8363** | 0.2047 | [0.8156, 0.8485] | 1195.7 | 4.48e-07 |
| `knn (x94)` | ml | cpu | 26038 / 94 | 0.3810 | 0.8724 | 0.3529 (desc) | 0.5020 | **0.8577** | 0.1312 | [0.8494, 0.8648] | 1166.0 | - |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.6452 | 0.9212 | 0.0454 (prep) | 1.9774 | **2.0272** | 0.5784 | [1.9868, 2.1428] | 493.3 | 4.25e-07 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.5455 | 0.8147 | 0.0454 (prep) | 2.9622 | **3.0136** | 0.8544 | [2.9805, 3.1829] | 331.8 | 4.55e-07 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.4800 | 0.8346 | 0.0454 (prep) | 3.2202 | **3.2657** | 0.9776 | [3.1807, 3.4540] | 306.2 | 3.36e-07 |
| `rf (x94)` | ml | cpu | 3518 / 94 | 0.8750 | 0.9756 | 0.3529 (desc) | 4.2611 | **4.6281** | 0.7675 | [4.5438, 4.7524] | 216.1 | - |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.3908 | 0.8968 | 0.0454 (prep) | 5.0324 | **5.0772** | 1.1551 | [4.9384, 5.3511] | 197.0 | 4.02e-07 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.3908 | 0.7314 | 0.0454 (prep) | 6.8533 | **6.9122** | 1.6189 | [6.6170, 7.0743] | 144.7 | 5.81e-02 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.8824 | 0.9889 | 0.0454 (prep) | 6.9748 | **7.0357** | 1.2826 | [6.7637, 7.2147] | 142.1 | 4.54e-07 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **7.6146 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2289 ms** vs llamada formal E2E: **0.1958 ms** (gap de dispatch/asignaciones Python = **+0.0331 ms**, +14.5%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0647 | 28.3% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.0535 | 23.4% | 2.1 | `O(g^2*K) [16x3]` |
| `asymmetry_g8` | 8x8 | 0.0525 | 22.9% | 2.6 | `O(g^2*K) [64x3]` |
| `c_fine_hist` | 8x8 | 0.0290 | 12.6% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0114 | 5.0% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 4. Desglose Medido por Capas CNN en CPU (`threads=1, batch=1`)

| Arquitectura | Params | Act Peak (KB) | FLOPs (PyTorch) | Hookless (ms) | Hook overhead (%) | Cold 1st (ms) | Familias dominantes (% hooks) |
|---|---|---|---|---|---|---|---|
| `ultra_tiny` | 1,154 | 128.0 | 1,335,424 | 0.9575 | +24.5% | 1.354 | dwconv:40.4%, bn:21.6%, conv:14.8% |
| `ds_cnn` | 12,866 | 1024.0 | 19,366,400 | 2.5417 | -5.2% | 4.234 | dwconv:37.9%, bn:20.8%, pwconv:15.2% |
| `tiny_cnn` | 23,538 | 1024.0 | 80,216,320 | 5.4161 | +2.1% | 5.387 | pool:62.1%, conv:28.9%, bn:5.6% |
| `mobilenet_like` | 70,786 | 512.0 | 38,716,416 | 3.0941 | +9.8% | 3.432 | dwconv:41.0%, pwconv:22.4%, bn:20.4% |
| `tiny_resnet` | 77,234 | 256.0 | 97,648,896 | 3.5789 | +4.4% | 6.361 | conv:70.3%, bn:11.9%, pwconv:11.3% |
| `small_cnn` | 97,650 | 1024.0 | 117,965,312 | 7.2266 | -1.0% | 7.039 | pool:60.5%, conv:31.6%, bn:5.0% |
| `torchvision_mobilenet` | 1,519,618 | 288.0 | 35,293,184 | 6.8492 | +36.7% | 7.735 | dwconv:31.4%, pwconv:27.4%, bn:20.5% |

## 5. Pruebas de Significancia Estadistica (Mann-Whitney U, 1 hilo, batch=1)

| Comparacion | Mediana A (ms) | Mediana B (ms) | Ratio A/B | Estadistico U | p-valor | Significativo |
|---|---|---|---|---|---|---|
| `dt (x94)` vs `ds_cnn` | 0.4867 | 2.0272 | 0.240x | 0.0 | `4.828e-67` | True |
| `dt (x94)` vs `ultra_tiny` | 0.4867 | 0.8363 | 0.582x | 0.0 | `4.827e-67` | True |
| `lr (x94)` vs `ds_cnn` | 0.5536 | 2.0272 | 0.273x | 0.0 | `4.827e-67` | True |
| `lr (x94)` vs `ultra_tiny` | 0.5536 | 0.8363 | 0.662x | 0.0 | `4.825e-67` | True |
| `lr_dot (x94)` vs `ds_cnn` | 0.3632 | 2.0272 | 0.179x | 0.0 | `4.828e-67` | True |
| `lr_dot (x94)` vs `ultra_tiny` | 0.3632 | 0.8363 | 0.434x | 0.0 | `4.827e-67` | True |
| `mlp_16_dot (x94)` vs `ds_cnn` | 0.3651 | 2.0272 | 0.180x | 0.0 | `4.824e-67` | True |
| `mlp_16_dot (x94)` vs `ultra_tiny` | 0.3651 | 0.8363 | 0.437x | 0.0 | `4.822e-67` | True |
| `rf (x94)` vs `ds_cnn` | 4.6281 | 2.0272 | 2.283x | 39615.0 | `1.479e-64` | True |
| `rf (x94)` vs `ultra_tiny` | 4.6281 | 0.8363 | 5.534x | 40000.0 | `4.829e-67` | True |

## 6. Ablacion Calidad vs Coste (`x85` vs `x9` vs `x94`)

| Clasificador | Subconjunto | Features | F1 | ROC-AUC | T_total (ms) | dF1 vs x85 | d_ms vs x85 | dF1 / d_ms |
|---|---|---|---|---|---|---|---|---|
| `dt` | `x85_only` | 85 | 0.5098 | 0.7492 | 0.0998 | +0.0000 | +0.0000 | +0.0000 |
| `dt` | `x9_only` | 9 | 0.8235 | 0.9606 | 0.2514 | +0.3137 | +0.1517 | +2.0687 |
| `dt` | `x94_full` | 94 | 0.8387 | 0.9623 | 0.3064 | +0.3289 | +0.2067 | +1.5916 |
| `knn` | `x85_only` | 85 | 0.1111 | 0.7869 | 0.4940 | +0.0000 | +0.0000 | +0.0000 |
| `knn` | `x9_only` | 9 | 0.6667 | 0.9046 | 0.7628 | +0.5556 | +0.2687 | +2.0676 |
| `knn` | `x94_full` | 94 | 0.3810 | 0.8724 | 0.7155 | +0.2698 | +0.2215 | +1.2185 |
| `lr` | `x85_only` | 85 | 0.7333 | 0.8590 | 0.2518 | +0.0000 | +0.0000 | +0.0000 |
| `lr` | `x9_only` | 9 | 0.8333 | 0.9556 | 0.3980 | +0.1000 | +0.1462 | +0.6840 |
| `lr` | `x94_full` | 94 | 0.8649 | 0.9678 | 0.4193 | +0.1315 | +0.1675 | +0.7853 |
| `lr_dot` | `x85_only` | 85 | 0.7333 | 0.8590 | 0.0279 | +0.0000 | +0.0000 | +0.0000 |
| `lr_dot` | `x9_only` | 9 | 0.8333 | 0.9556 | 0.1839 | +0.1000 | +0.1560 | +0.6410 |
| `lr_dot` | `x94_full` | 94 | 0.8649 | 0.9678 | 0.2271 | +0.1315 | +0.1992 | +0.6603 |
| `mlp_16_dot` | `x85_only` | 85 | 0.7500 | 0.9467 | 0.0326 | +0.0000 | +0.0000 | +0.0000 |
| `mlp_16_dot` | `x9_only` | 9 | 0.7895 | 0.9534 | 0.1893 | +0.0395 | +0.1568 | +0.2518 |
| `mlp_16_dot` | `x94_full` | 94 | 0.8108 | 0.9767 | 0.2313 | +0.0608 | +0.1988 | +0.3060 |
| `rf` | `x85_only` | 85 | 0.4545 | 0.9356 | 4.2268 | +0.0000 | +0.0000 | +0.0000 |
| `rf` | `x9_only` | 9 | 0.8125 | 0.9578 | 4.2419 | +0.3580 | +0.0151 | +23.7056 |
| `rf` | `x94_full` | 94 | 0.8750 | 0.9756 | 4.4399 | +0.4205 | +0.2131 | +1.9726 |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.1409 | 876.5 |
| `ds_cnn` | cuda | 1 | 1.2388 | 807.3 |
| `tiny_cnn` | cuda | 1 | 0.7364 | 1358.0 |
| `mobilenet_like` | cuda | 1 | 2.1863 | 457.4 |
| `tiny_resnet` | cuda | 1 | 1.7018 | 587.6 |
| `small_cnn` | cuda | 1 | 0.9499 | 1052.7 |
| `torchvision_mobilenet` | cuda | 1 | 9.0639 | 110.3 |
| `ultra_tiny` | cuda | 32 | 0.0386 | 25906.7 |
| `ds_cnn` | cuda | 32 | 0.2043 | 4894.6 |
| `tiny_cnn` | cuda | 32 | 0.2054 | 4867.4 |
| `mobilenet_like` | cuda | 32 | 0.2294 | 4359.3 |
| `tiny_resnet` | cuda | 32 | 0.1777 | 5628.9 |
| `small_cnn` | cuda | 32 | 0.2312 | 4325.6 |
| `torchvision_mobilenet` | cuda | 32 | 0.2769 | 3611.9 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **28.3%** (0.0647 ms) del tiempo por etapas (E2E=0.1958 ms, gap de despacho Python/ctypes=+0.0331 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.3632 ms**, F1=0.8649), mientras el maximo F1 lo obtiene `small_cnn` (**F1=0.8824**, 7.0357 ms).
