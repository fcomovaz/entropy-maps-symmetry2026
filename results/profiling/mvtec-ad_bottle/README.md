# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad_bottle`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `486c111`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.8000 | 0.9231 | 0.3749 (desc) | 0.0081 | **0.3873** | 0.0745 | [0.3830, 0.3969] | 2582.0 | - |
| `dt (x94)` | ml | cpu | 15 / 94 | 0.5833 | 0.6614 | 0.3749 (desc) | 0.1392 | **0.5446** | 0.1263 | [0.5275, 0.5573] | 1836.0 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.8000 | 0.9231 | 0.3749 (desc) | 0.2351 | **0.6558** | 0.1500 | [0.6336, 0.6754] | 1525.0 | - |
| `knn (x94)` | ml | cpu | 21902 / 94 | 0.3750 | 0.7057 | 0.3749 (desc) | 0.5826 | **0.9931** | 0.4561 | [0.9550, 1.0500] | 1006.9 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.3611 | 0.5268 | 0.0422 (prep) | 0.9766 | **1.0328** | 0.2566 | [1.0132, 1.0621] | 968.3 | 1.47e-07 |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.3611 | 0.5535 | 0.0422 (prep) | 2.8537 | **2.9051** | 2.0450 | [2.6569, 3.1387] | 344.2 | 2.94e-07 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.7586 | 0.9415 | 0.0422 (prep) | 4.0537 | **4.1020** | 2.0757 | [3.8280, 4.3827] | 243.8 | 4.28e-07 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.3662 | 0.7391 | 0.0422 (prep) | 4.7443 | **4.7858** | 1.6543 | [4.5659, 5.0681] | 209.0 | 4.35e-07 |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.3571 | 0.6472 | 0.0422 (prep) | 5.2477 | **5.2897** | 1.6419 | [5.1666, 5.4625] | 189.0 | 4.07e-07 |
| `rf (x94)` | ml | cpu | 3496 / 94 | 0.7273 | 0.9030 | 0.3749 (desc) | 5.0093 | **5.4283** | 0.9584 | [5.2753, 5.5876] | 184.2 | - |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.3611 | 0.6639 | 0.0422 (prep) | 7.7559 | **7.8073** | 1.9319 | [7.4706, 8.1091] | 128.1 | 4.01e-07 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.3611 | 0.5000 | 0.0422 (prep) | 9.5651 | **9.6074** | 2.4740 | [9.1771, 9.7534] | 104.1 | 2.01e-02 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **11.3750 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2506 ms** vs llamada formal E2E: **0.2402 ms** (gap de dispatch/asignaciones Python = **+0.0105 ms**, +4.2%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0703 | 28.0% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.0588 | 23.5% | 2.1 | `O(g^2*K) [16x3]` |
| `asymmetry_g8` | 8x8 | 0.0568 | 22.6% | 2.6 | `O(g^2*K) [64x3]` |
| `c_fine_hist` | 8x8 | 0.0336 | 13.4% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0124 | 5.0% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 4. Desglose Medido por Capas CNN en CPU (`threads=1, batch=1`)

| Arquitectura | Params | Act Peak (KB) | FLOPs (PyTorch) | Hookless (ms) | Hook overhead (%) | Cold 1st (ms) | Familias dominantes (% hooks) |
|---|---|---|---|---|---|---|---|
| `ultra_tiny` | 1,154 | 128.0 | 1,335,424 | 1.3950 | -15.7% | 1.918 | dwconv:40.0%, bn:22.1%, conv:14.6% |
| `ds_cnn` | 12,866 | 1024.0 | 19,366,400 | 3.7210 | -31.8% | 5.033 | dwconv:39.2%, bn:20.9%, pwconv:15.2% |
| `tiny_cnn` | 23,538 | 1024.0 | 80,216,320 | 4.8464 | +41.0% | 4.847 | pool:51.9%, conv:34.4%, bn:8.6% |
| `mobilenet_like` | 70,786 | 512.0 | 38,716,416 | 3.5177 | +19.6% | 4.301 | dwconv:43.5%, pwconv:20.5%, bn:20.2% |
| `tiny_resnet` | 77,234 | 256.0 | 97,648,896 | 4.4618 | -8.3% | 7.571 | conv:70.8%, bn:11.8%, pwconv:11.1% |
| `small_cnn` | 97,650 | 1024.0 | 117,965,312 | 7.3628 | -5.8% | 11.358 | pool:54.7%, conv:36.2%, bn:5.6% |
| `torchvision_mobilenet` | 1,519,618 | 288.0 | 35,293,184 | 8.9993 | +18.3% | 14.566 | dwconv:32.4%, pwconv:26.4%, bn:20.4% |

## 5. Pruebas de Significancia Estadistica (Mann-Whitney U, 1 hilo, batch=1)

| Comparacion | Mediana A (ms) | Mediana B (ms) | Ratio A/B | Estadistico U | p-valor | Significativo |
|---|---|---|---|---|---|---|
| `dt (x94)` vs `ds_cnn` | 0.5446 | 2.9051 | 0.187x | 0.0 | `4.830e-67` | True |
| `dt (x94)` vs `ultra_tiny` | 0.5446 | 1.0328 | 0.527x | 672.0 | `9.817e-63` | True |
| `lr (x94)` vs `ds_cnn` | 0.6558 | 2.9051 | 0.226x | 0.0 | `4.830e-67` | True |
| `lr (x94)` vs `ultra_tiny` | 0.6558 | 1.0328 | 0.635x | 1765.0 | `4.861e-56` | True |
| `lr_dot (x94)` vs `ds_cnn` | 0.3873 | 2.9051 | 0.133x | 0.0 | `4.829e-67` | True |
| `lr_dot (x94)` vs `ultra_tiny` | 0.3873 | 1.0328 | 0.375x | 60.0 | `1.187e-66` | True |
| `rf (x94)` vs `ds_cnn` | 5.4283 | 2.9051 | 1.868x | 35524.5 | `4.178e-41` | True |
| `rf (x94)` vs `ultra_tiny` | 5.4283 | 1.0328 | 5.256x | 40000.0 | `4.830e-67` | True |

## 6. Ablacion Calidad vs Coste (`x85` vs `x9` vs `x94`)

| Clasificador | Subconjunto | Features | F1 | ROC-AUC | T_total (ms) | dF1 vs x85 | d_ms vs x85 | dF1 / d_ms |
|---|---|---|---|---|---|---|---|---|
| `dt` | `x85_only` | 85 | 0.5714 | 0.6781 | 0.1075 | +0.0000 | +0.0000 | +0.0000 |
| `dt` | `x9_only` | 9 | 0.5882 | 0.7584 | 0.2797 | +0.0168 | +0.1722 | +0.0976 |
| `dt` | `x94_full` | 94 | 0.5833 | 0.6614 | 0.2960 | +0.0119 | +0.1885 | +0.0632 |
| `knn` | `x85_only` | 85 | 0.1429 | 0.6605 | 0.8146 | +0.0000 | +0.0000 | +0.0000 |
| `knn` | `x9_only` | 9 | 0.6667 | 0.8997 | 1.0542 | +0.5238 | +0.2396 | +2.1857 |
| `knn` | `x94_full` | 94 | 0.3750 | 0.7057 | 0.9302 | +0.2321 | +0.1156 | +2.0073 |
| `lr` | `x85_only` | 85 | 0.7333 | 0.9181 | 0.2931 | +0.0000 | +0.0000 | +0.0000 |
| `lr` | `x9_only` | 9 | 0.7333 | 0.9130 | 0.4353 | +0.0000 | +0.1421 | +0.0000 |
| `lr` | `x94_full` | 94 | 0.8000 | 0.9231 | 0.5707 | +0.0667 | +0.2776 | +0.2402 |
| `lr_dot` | `x85_only` | 85 | 0.7333 | 0.9181 | 0.0317 | +0.0000 | +0.0000 | +0.0000 |
| `lr_dot` | `x9_only` | 9 | 0.7333 | 0.9130 | 0.2046 | +0.0000 | +0.1729 | +0.0000 |
| `lr_dot` | `x94_full` | 94 | 0.8000 | 0.9231 | 0.2216 | +0.0667 | +0.1899 | +0.3511 |
| `rf` | `x85_only` | 85 | 0.5263 | 0.8696 | 5.4427 | +0.0000 | +0.0000 | +0.0000 |
| `rf` | `x9_only` | 9 | 0.6923 | 0.8980 | 5.3873 | +0.1660 | -0.0553 | -2.9990 |
| `rf` | `x94_full` | 94 | 0.7273 | 0.9030 | 5.6531 | +0.2010 | +0.2104 | +0.9551 |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 0.9708 | 1030.1 |
| `ds_cnn` | cuda | 1 | 1.3662 | 732.0 |
| `tiny_cnn` | cuda | 1 | 1.1241 | 889.6 |
| `mobilenet_like` | cuda | 1 | 2.7999 | 357.2 |
| `tiny_resnet` | cuda | 1 | 1.7768 | 562.8 |
| `small_cnn` | cuda | 1 | 1.0537 | 949.0 |
| `torchvision_mobilenet` | cuda | 1 | 11.3987 | 87.7 |
| `ultra_tiny` | cuda | 32 | 0.0482 | 20744.9 |
| `ds_cnn` | cuda | 32 | 0.2027 | 4933.6 |
| `tiny_cnn` | cuda | 32 | 0.2053 | 4869.8 |
| `mobilenet_like` | cuda | 32 | 0.2565 | 3898.6 |
| `tiny_resnet` | cuda | 32 | 0.1813 | 5514.8 |
| `small_cnn` | cuda | 32 | 0.2315 | 4320.0 |
| `torchvision_mobilenet` | cuda | 32 | 0.3571 | 2800.3 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **28.0%** (0.0703 ms) del tiempo por etapas (E2E=0.2402 ms, gap de despacho Python/ctypes=+0.0105 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.3873 ms**, F1=0.8000), mientras el maximo F1 lo obtiene `lr (x94)` (**F1=0.8000**, 0.6558 ms).
