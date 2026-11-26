# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad_metal_nut`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `99bc026`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.4762 | 0.7237 | 0.3478 (desc) | 0.0068 | **0.3546** | 0.0153 | [0.3543, 0.3556] | 2820.1 | - |
| `mlp_16_dot (x94)` | ml | cpu | 1537 / 94 | 0.6111 | 0.7884 | 0.3478 (desc) | 0.0118 | **0.3596** | 0.0151 | [0.3592, 0.3606] | 2780.9 | - |
| `dt (x94)` | ml | cpu | 11 / 94 | 0.5882 | 0.7621 | 0.3478 (desc) | 0.0825 | **0.4319** | 0.0291 | [0.4308, 0.4378] | 2315.4 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.4762 | 0.7237 | 0.3478 (desc) | 0.2036 | **0.5528** | 0.0414 | [0.5511, 0.5672] | 1808.8 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.4000 | 0.7950 | 0.0409 (prep) | 0.7987 | **0.8399** | 0.2096 | [0.8364, 0.8434] | 1190.5 | 4.03e-07 |
| `knn (x94)` | ml | cpu | 25192 / 94 | 0.4800 | 0.7971 | 0.3478 (desc) | 0.4959 | **0.8469** | 0.1146 | [0.8418, 0.8568] | 1180.7 | - |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.7600 | 0.9627 | 0.0409 (prep) | 1.9280 | **1.9689** | 0.8345 | [1.9573, 2.0320] | 507.9 | 4.78e-07 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.7826 | 0.9507 | 0.0409 (prep) | 2.9683 | **3.0096** | 1.0226 | [2.9372, 3.2443] | 332.3 | 4.28e-07 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.6207 | 0.9211 | 0.0409 (prep) | 3.2781 | **3.3188** | 0.8247 | [3.2752, 3.5080] | 301.3 | 4.39e-07 |
| `rf (x94)` | ml | cpu | 6042 / 94 | 0.6429 | 0.8465 | 0.3478 (desc) | 4.2393 | **4.6164** | 0.7600 | [4.4992, 4.6807] | 216.6 | - |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.7778 | 0.9638 | 0.0409 (prep) | 4.9292 | **4.9705** | 1.1666 | [4.7332, 5.1134] | 201.2 | 2.22e-07 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.4419 | 0.9550 | 0.0409 (prep) | 6.4069 | **6.4476** | 1.6109 | [6.1910, 6.7401] | 155.1 | 4.65e-07 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.4419 | 0.6014 | 0.0409 (prep) | 6.4528 | **6.4938** | 1.0104 | [6.3906, 6.6390] | 154.0 | 1.43e-02 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **8.9215 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2297 ms** vs llamada formal E2E: **0.1960 ms** (gap de dispatch/asignaciones Python = **+0.0338 ms**, +14.7%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0639 | 27.8% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.0537 | 23.4% | 2.1 | `O(g^2*K) [16x3]` |
| `asymmetry_g8` | 8x8 | 0.0523 | 22.8% | 2.6 | `O(g^2*K) [64x3]` |
| `c_fine_hist` | 8x8 | 0.0312 | 13.6% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0112 | 4.9% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 4. Desglose Medido por Capas CNN en CPU (`threads=1, batch=1`)

| Arquitectura | Params | Act Peak (KB) | FLOPs (PyTorch) | Hookless (ms) | Hook overhead (%) | Cold 1st (ms) | Familias dominantes (% hooks) |
|---|---|---|---|---|---|---|---|
| `ultra_tiny` | 1,154 | 128.0 | 1,335,424 | 0.8974 | +33.0% | 1.086 | dwconv:40.6%, bn:21.6%, conv:14.8% |
| `ds_cnn` | 12,866 | 1024.0 | 19,366,400 | 2.1307 | +18.7% | 4.483 | dwconv:37.8%, bn:21.1%, pwconv:15.7% |
| `tiny_cnn` | 23,538 | 1024.0 | 80,216,320 | 5.0514 | +12.4% | 5.660 | pool:58.8%, conv:31.7%, bn:5.8% |
| `mobilenet_like` | 70,786 | 512.0 | 38,716,416 | 3.2881 | +15.6% | 5.097 | dwconv:40.6%, pwconv:22.4%, bn:20.4% |
| `tiny_resnet` | 77,234 | 256.0 | 97,648,896 | 4.0822 | -1.5% | 5.618 | conv:70.5%, bn:11.8%, pwconv:11.3% |
| `small_cnn` | 97,650 | 1024.0 | 117,965,312 | 5.7246 | +13.1% | 8.925 | pool:55.9%, conv:35.5%, bn:5.3% |
| `torchvision_mobilenet` | 1,519,618 | 288.0 | 35,293,184 | 6.6279 | +44.1% | 7.081 | dwconv:31.5%, pwconv:27.5%, bn:20.4% |

## 5. Pruebas de Significancia Estadistica (Mann-Whitney U, 1 hilo, batch=1)

| Comparacion | Mediana A (ms) | Mediana B (ms) | Ratio A/B | Estadistico U | p-valor | Significativo |
|---|---|---|---|---|---|---|
| `dt (x94)` vs `ds_cnn` | 0.4319 | 1.9689 | 0.219x | 0.0 | `4.827e-67` | True |
| `dt (x94)` vs `ultra_tiny` | 0.4319 | 0.8399 | 0.514x | 0.0 | `4.825e-67` | True |
| `lr (x94)` vs `ds_cnn` | 0.5528 | 1.9689 | 0.281x | 0.0 | `4.827e-67` | True |
| `lr (x94)` vs `ultra_tiny` | 0.5528 | 0.8399 | 0.658x | 261.0 | `2.367e-65` | True |
| `lr_dot (x94)` vs `ds_cnn` | 0.3546 | 1.9689 | 0.180x | 0.0 | `4.821e-67` | True |
| `lr_dot (x94)` vs `ultra_tiny` | 0.3546 | 0.8399 | 0.422x | 0.0 | `4.819e-67` | True |
| `mlp_16_dot (x94)` vs `ds_cnn` | 0.3596 | 1.9689 | 0.183x | 0.0 | `4.824e-67` | True |
| `mlp_16_dot (x94)` vs `ultra_tiny` | 0.3596 | 0.8399 | 0.428x | 0.0 | `4.823e-67` | True |
| `rf (x94)` vs `ds_cnn` | 4.6164 | 1.9689 | 2.345x | 39868.0 | `3.481e-66` | True |
| `rf (x94)` vs `ultra_tiny` | 4.6164 | 0.8399 | 5.496x | 40000.0 | `4.829e-67` | True |

## 6. Ablacion Calidad vs Coste (`x85` vs `x9` vs `x94`)

| Clasificador | Subconjunto | Features | F1 | ROC-AUC | T_total (ms) | dF1 vs x85 | d_ms vs x85 | dF1 / d_ms |
|---|---|---|---|---|---|---|---|---|
| `dt` | `x85_only` | 85 | 0.5882 | 0.7621 | 0.0975 | +0.0000 | +0.0000 | +0.0000 |
| `dt` | `x9_only` | 9 | 0.4286 | 0.5757 | 0.2680 | -0.1597 | +0.1705 | -0.9367 |
| `dt` | `x94_full` | 94 | 0.5882 | 0.7621 | 0.2802 | +0.0000 | +0.1826 | +0.0000 |
| `knn` | `x85_only` | 85 | 0.4800 | 0.7829 | 0.5123 | +0.0000 | +0.0000 | +0.0000 |
| `knn` | `x9_only` | 9 | 0.2222 | 0.5927 | 0.7688 | -0.2578 | +0.2565 | -1.0052 |
| `knn` | `x94_full` | 94 | 0.4800 | 0.7971 | 0.6673 | +0.0000 | +0.1551 | +0.0000 |
| `lr` | `x85_only` | 85 | 0.5116 | 0.7226 | 0.2241 | +0.0000 | +0.0000 | +0.0000 |
| `lr` | `x9_only` | 9 | 0.4167 | 0.6184 | 0.3939 | -0.0950 | +0.1698 | -0.5593 |
| `lr` | `x94_full` | 94 | 0.4762 | 0.7237 | 0.3871 | -0.0354 | +0.1630 | -0.2174 |
| `lr_dot` | `x85_only` | 85 | 0.5116 | 0.7226 | 0.0297 | +0.0000 | +0.0000 | +0.0000 |
| `lr_dot` | `x9_only` | 9 | 0.4167 | 0.6184 | 0.2000 | -0.0950 | +0.1703 | -0.5576 |
| `lr_dot` | `x94_full` | 94 | 0.4762 | 0.7237 | 0.1950 | -0.0354 | +0.1653 | -0.2144 |
| `mlp_16_dot` | `x85_only` | 85 | 0.5366 | 0.7730 | 0.0340 | +0.0000 | +0.0000 | +0.0000 |
| `mlp_16_dot` | `x9_only` | 9 | 0.4255 | 0.6140 | 0.2048 | -0.1111 | +0.1708 | -0.6504 |
| `mlp_16_dot` | `x94_full` | 94 | 0.6111 | 0.7884 | 0.1997 | +0.0745 | +0.1657 | +0.4498 |
| `rf` | `x85_only` | 85 | 0.5926 | 0.8421 | 4.2066 | +0.0000 | +0.0000 | +0.0000 |
| `rf` | `x9_only` | 9 | 0.2963 | 0.5735 | 4.3935 | -0.2963 | +0.1870 | -1.5849 |
| `rf` | `x94_full` | 94 | 0.6429 | 0.8465 | 4.5466 | +0.0503 | +0.3400 | +0.1478 |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.2306 | 812.6 |
| `ds_cnn` | cuda | 1 | 1.6105 | 620.9 |
| `tiny_cnn` | cuda | 1 | 0.9283 | 1077.2 |
| `mobilenet_like` | cuda | 1 | 2.5060 | 399.1 |
| `tiny_resnet` | cuda | 1 | 1.6806 | 595.0 |
| `small_cnn` | cuda | 1 | 1.0048 | 995.2 |
| `torchvision_mobilenet` | cuda | 1 | 10.4113 | 96.0 |
| `ultra_tiny` | cuda | 32 | 0.0412 | 24281.1 |
| `ds_cnn` | cuda | 32 | 0.2059 | 4857.1 |
| `tiny_cnn` | cuda | 32 | 0.2067 | 4837.5 |
| `mobilenet_like` | cuda | 32 | 0.2305 | 4338.4 |
| `tiny_resnet` | cuda | 32 | 0.1784 | 5606.9 |
| `small_cnn` | cuda | 32 | 0.2318 | 4314.6 |
| `torchvision_mobilenet` | cuda | 32 | 0.3174 | 3150.5 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **27.8%** (0.0639 ms) del tiempo por etapas (E2E=0.1960 ms, gap de despacho Python/ctypes=+0.0338 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.3546 ms**, F1=0.4762), mientras el maximo F1 lo obtiene `mobilenet_like` (**F1=0.7826**, 3.0096 ms).
