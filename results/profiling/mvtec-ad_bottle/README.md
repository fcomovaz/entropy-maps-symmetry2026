# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad_bottle`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `99bc026`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.8000 | 0.9231 | 0.3451 (desc) | 0.0069 | **0.3519** | 0.1560 | [0.3507, 0.3529] | 2841.3 | - |
| `mlp_16_dot (x94)` | ml | cpu | 1537 / 94 | 0.8000 | 0.9515 | 0.3451 (desc) | 0.0124 | **0.3577** | 0.1592 | [0.3562, 0.3585] | 2796.0 | - |
| `dt (x94)` | ml | cpu | 15 / 94 | 0.5833 | 0.6614 | 0.3451 (desc) | 0.0836 | **0.4285** | 0.0996 | [0.4270, 0.4290] | 2333.7 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.8000 | 0.9231 | 0.3451 (desc) | 0.2120 | **0.5570** | 0.1360 | [0.5558, 0.5580] | 1795.3 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.3611 | 0.5268 | 0.0375 (prep) | 0.7629 | **0.8068** | 0.1821 | [0.7936, 0.8268] | 1239.5 | 1.47e-07 |
| `knn (x94)` | ml | cpu | 21902 / 94 | 0.3750 | 0.7057 | 0.3451 (desc) | 0.4537 | **0.8116** | 0.1636 | [0.7974, 0.8165] | 1232.1 | - |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.3611 | 0.5535 | 0.0375 (prep) | 1.7926 | **1.8299** | 0.2207 | [1.8123, 1.8940] | 546.5 | 2.94e-07 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.7586 | 0.9415 | 0.0375 (prep) | 2.9270 | **2.9645** | 0.9275 | [2.8983, 3.1356] | 337.3 | 4.28e-07 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.3662 | 0.7391 | 0.0375 (prep) | 3.2418 | **3.2792** | 0.8519 | [3.2072, 3.4270] | 305.0 | 4.35e-07 |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.3571 | 0.6472 | 0.0375 (prep) | 4.3327 | **4.3704** | 0.9371 | [4.3317, 4.4847] | 228.8 | 4.07e-07 |
| `rf (x94)` | ml | cpu | 3496 / 94 | 0.7273 | 0.9030 | 0.3451 (desc) | 4.1096 | **4.4407** | 0.6608 | [4.3701, 4.5389] | 225.2 | - |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.3611 | 0.6639 | 0.0375 (prep) | 5.8067 | **5.8443** | 1.2722 | [5.5631, 6.0338] | 171.1 | 4.01e-07 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.3611 | 0.5000 | 0.0375 (prep) | 6.4681 | **6.5056** | 1.1998 | [6.3380, 6.6900] | 153.7 | 2.01e-02 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **7.2580 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2183 ms** vs llamada formal E2E: **0.1958 ms** (gap de dispatch/asignaciones Python = **+0.0225 ms**, +10.3%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0605 | 27.7% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.0506 | 23.2% | 2.1 | `O(g^2*K) [16x3]` |
| `asymmetry_g8` | 8x8 | 0.0492 | 22.5% | 2.6 | `O(g^2*K) [64x3]` |
| `c_fine_hist` | 8x8 | 0.0302 | 13.8% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0109 | 5.0% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 4. Desglose Medido por Capas CNN en CPU (`threads=1, batch=1`)

| Arquitectura | Params | Act Peak (KB) | FLOPs (PyTorch) | Hookless (ms) | Hook overhead (%) | Cold 1st (ms) | Familias dominantes (% hooks) |
|---|---|---|---|---|---|---|---|
| `ultra_tiny` | 1,154 | 128.0 | 1,335,424 | 0.8497 | +28.7% | 1.152 | dwconv:40.0%, bn:21.8%, conv:14.8% |
| `ds_cnn` | 12,866 | 1024.0 | 19,366,400 | 2.0840 | +7.2% | 4.682 | dwconv:37.5%, bn:21.2%, pwconv:16.0% |
| `tiny_cnn` | 23,538 | 1024.0 | 80,216,320 | 4.8003 | +1.5% | 7.758 | pool:58.1%, conv:32.1%, bn:5.9% |
| `mobilenet_like` | 70,786 | 512.0 | 38,716,416 | 2.9135 | +14.8% | 3.351 | dwconv:41.1%, pwconv:22.2%, bn:20.4% |
| `tiny_resnet` | 77,234 | 256.0 | 97,648,896 | 3.3750 | +10.9% | 5.476 | conv:70.3%, bn:12.1%, pwconv:11.3% |
| `small_cnn` | 97,650 | 1024.0 | 117,965,312 | 6.1001 | -0.3% | 6.607 | pool:55.3%, conv:36.0%, bn:5.4% |
| `torchvision_mobilenet` | 1,519,618 | 288.0 | 35,293,184 | 6.8028 | +35.2% | 9.947 | dwconv:31.4%, pwconv:27.4%, bn:20.5% |

## 5. Pruebas de Significancia Estadistica (Mann-Whitney U, 1 hilo, batch=1)

| Comparacion | Mediana A (ms) | Mediana B (ms) | Ratio A/B | Estadistico U | p-valor | Significativo |
|---|---|---|---|---|---|---|
| `dt (x94)` vs `ds_cnn` | 0.4285 | 1.8299 | 0.234x | 0.0 | `4.828e-67` | True |
| `dt (x94)` vs `ultra_tiny` | 0.4285 | 0.8068 | 0.531x | 0.0 | `4.826e-67` | True |
| `lr (x94)` vs `ds_cnn` | 0.5570 | 1.8299 | 0.304x | 0.0 | `4.828e-67` | True |
| `lr (x94)` vs `ultra_tiny` | 0.5570 | 0.8068 | 0.690x | 1018.0 | `1.422e-60` | True |
| `lr_dot (x94)` vs `ds_cnn` | 0.3519 | 1.8299 | 0.192x | 0.0 | `4.825e-67` | True |
| `lr_dot (x94)` vs `ultra_tiny` | 0.3519 | 0.8068 | 0.436x | 0.0 | `4.823e-67` | True |
| `mlp_16_dot (x94)` vs `ds_cnn` | 0.3577 | 1.8299 | 0.195x | 0.0 | `4.826e-67` | True |
| `mlp_16_dot (x94)` vs `ultra_tiny` | 0.3577 | 0.8068 | 0.443x | 0.0 | `4.825e-67` | True |
| `rf (x94)` vs `ds_cnn` | 4.4407 | 1.8299 | 2.427x | 39668.0 | `6.770e-65` | True |
| `rf (x94)` vs `ultra_tiny` | 4.4407 | 0.8068 | 5.504x | 40000.0 | `4.828e-67` | True |

## 6. Ablacion Calidad vs Coste (`x85` vs `x9` vs `x94`)

| Clasificador | Subconjunto | Features | F1 | ROC-AUC | T_total (ms) | dF1 vs x85 | d_ms vs x85 | dF1 / d_ms |
|---|---|---|---|---|---|---|---|---|
| `dt` | `x85_only` | 85 | 0.5714 | 0.6781 | 0.0989 | +0.0000 | +0.0000 | +0.0000 |
| `dt` | `x9_only` | 9 | 0.5882 | 0.7584 | 0.3125 | +0.0168 | +0.2136 | +0.0787 |
| `dt` | `x94_full` | 94 | 0.5833 | 0.6614 | 0.2604 | +0.0119 | +0.1615 | +0.0737 |
| `knn` | `x85_only` | 85 | 0.1429 | 0.6605 | 0.4879 | +0.0000 | +0.0000 | +0.0000 |
| `knn` | `x9_only` | 9 | 0.6667 | 0.8997 | 0.8132 | +0.5238 | +0.3254 | +1.6100 |
| `knn` | `x94_full` | 94 | 0.3750 | 0.7057 | 0.6566 | +0.2321 | +0.1688 | +1.3753 |
| `lr` | `x85_only` | 85 | 0.7333 | 0.9181 | 0.2382 | +0.0000 | +0.0000 | +0.0000 |
| `lr` | `x9_only` | 9 | 0.7333 | 0.9130 | 0.4360 | +0.0000 | +0.1978 | +0.0000 |
| `lr` | `x94_full` | 94 | 0.8000 | 0.9231 | 0.3942 | +0.0667 | +0.1561 | +0.4272 |
| `lr_dot` | `x85_only` | 85 | 0.7333 | 0.9181 | 0.0310 | +0.0000 | +0.0000 | +0.0000 |
| `lr_dot` | `x9_only` | 9 | 0.7333 | 0.9130 | 0.2444 | +0.0000 | +0.2134 | +0.0000 |
| `lr_dot` | `x94_full` | 94 | 0.8000 | 0.9231 | 0.1919 | +0.0667 | +0.1609 | +0.4143 |
| `mlp_16_dot` | `x85_only` | 85 | 0.8148 | 0.9699 | 0.0358 | +0.0000 | +0.0000 | +0.0000 |
| `mlp_16_dot` | `x9_only` | 9 | 0.7097 | 0.8963 | 0.2494 | -0.1051 | +0.2136 | -0.4922 |
| `mlp_16_dot` | `x94_full` | 94 | 0.8000 | 0.9515 | 0.1980 | -0.0148 | +0.1622 | -0.0914 |
| `rf` | `x85_only` | 85 | 0.5263 | 0.8696 | 4.0431 | +0.0000 | +0.0000 | +0.0000 |
| `rf` | `x9_only` | 9 | 0.6923 | 0.8980 | 4.4425 | +0.1660 | +0.3994 | +0.4156 |
| `rf` | `x94_full` | 94 | 0.7273 | 0.9030 | 4.3612 | +0.2010 | +0.3181 | +0.6317 |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 0.9608 | 1040.7 |
| `ds_cnn` | cuda | 1 | 1.5514 | 644.6 |
| `tiny_cnn` | cuda | 1 | 0.9218 | 1084.8 |
| `mobilenet_like` | cuda | 1 | 2.2051 | 453.5 |
| `tiny_resnet` | cuda | 1 | 1.6866 | 592.9 |
| `small_cnn` | cuda | 1 | 0.9487 | 1054.0 |
| `torchvision_mobilenet` | cuda | 1 | 8.8103 | 113.5 |
| `ultra_tiny` | cuda | 32 | 0.0386 | 25939.3 |
| `ds_cnn` | cuda | 32 | 0.2029 | 4929.8 |
| `tiny_cnn` | cuda | 32 | 0.2082 | 4802.7 |
| `mobilenet_like` | cuda | 32 | 0.2301 | 4346.4 |
| `tiny_resnet` | cuda | 32 | 0.1767 | 5660.3 |
| `small_cnn` | cuda | 32 | 0.2316 | 4317.8 |
| `torchvision_mobilenet` | cuda | 32 | 0.4649 | 2151.1 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **27.7%** (0.0605 ms) del tiempo por etapas (E2E=0.1958 ms, gap de despacho Python/ctypes=+0.0225 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.3519 ms**, F1=0.8000), mientras el maximo F1 lo obtiene `lr (x94)` (**F1=0.8000**, 0.5570 ms).
