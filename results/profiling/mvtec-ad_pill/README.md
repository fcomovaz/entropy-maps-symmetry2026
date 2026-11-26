# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad_pill`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `99bc026`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.7937 | 0.9146 | 0.3493 (desc) | 0.0069 | **0.3561** | 0.0375 | [0.3550, 0.3580] | 2807.8 | - |
| `mlp_16_dot (x94)` | ml | cpu | 1537 / 94 | 0.8519 | 0.9516 | 0.3493 (desc) | 0.0118 | **0.3610** | 0.0374 | [0.3600, 0.3626] | 2769.7 | - |
| `dt (x94)` | ml | cpu | 13 / 94 | 0.7143 | 0.8465 | 0.3493 (desc) | 0.1012 | **0.4804** | 0.0713 | [0.4784, 0.4817] | 2081.6 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.7937 | 0.9146 | 0.3493 (desc) | 0.2180 | **0.6039** | 0.1129 | [0.5937, 0.6155] | 1655.8 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.5952 | 0.6992 | 0.0453 (prep) | 0.7957 | **0.8418** | 0.2688 | [0.8377, 0.8510] | 1187.9 | 4.58e-07 |
| `knn (x94)` | ml | cpu | 32618 / 94 | 0.7660 | 0.9594 | 0.3493 (desc) | 0.4873 | **0.8579** | 0.1217 | [0.8433, 0.8743] | 1165.6 | - |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.6667 | 0.8650 | 0.0453 (prep) | 1.9249 | **1.9753** | 0.4486 | [1.9626, 1.9910] | 506.3 | 3.99e-07 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.7164 | 0.8680 | 0.0453 (prep) | 2.9112 | **2.9566** | 0.8498 | [2.8909, 3.0842] | 338.2 | 2.59e-07 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.5946 | 0.8172 | 0.0453 (prep) | 3.5302 | **3.5753** | 1.0590 | [3.3523, 3.8083] | 279.7 | 4.14e-07 |
| `rf (x94)` | ml | cpu | 6310 / 94 | 0.7556 | 0.9455 | 0.3493 (desc) | 4.2300 | **4.5811** | 0.7910 | [4.5018, 4.6948] | 218.3 | - |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.6849 | 0.8814 | 0.0453 (prep) | 4.7915 | **4.8432** | 1.1297 | [4.6605, 4.9580] | 206.5 | 4.93e-07 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.8070 | 0.9407 | 0.0453 (prep) | 6.2593 | **6.3233** | 1.7366 | [6.0890, 6.5014] | 158.1 | 4.56e-07 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.4870 | 0.5000 | 0.0453 (prep) | 6.4041 | **6.4492** | 1.2370 | [6.3259, 6.6831] | 155.1 | 4.78e-07 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **7.6400 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2266 ms** vs llamada formal E2E: **0.2042 ms** (gap de dispatch/asignaciones Python = **+0.0224 ms**, +9.9%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0627 | 27.7% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.0524 | 23.1% | 2.1 | `O(g^2*K) [16x3]` |
| `asymmetry_g8` | 8x8 | 0.0513 | 22.6% | 2.6 | `O(g^2*K) [64x3]` |
| `c_fine_hist` | 8x8 | 0.0319 | 14.1% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0111 | 4.9% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 4. Desglose Medido por Capas CNN en CPU (`threads=1, batch=1`)

| Arquitectura | Params | Act Peak (KB) | FLOPs (PyTorch) | Hookless (ms) | Hook overhead (%) | Cold 1st (ms) | Familias dominantes (% hooks) |
|---|---|---|---|---|---|---|---|
| `ultra_tiny` | 1,154 | 128.0 | 1,335,424 | 1.0454 | +6.4% | 1.116 | dwconv:39.8%, bn:21.7%, conv:15.1% |
| `ds_cnn` | 12,866 | 1024.0 | 19,366,400 | 1.9915 | +12.5% | 4.201 | dwconv:37.8%, bn:21.1%, pwconv:16.3% |
| `tiny_cnn` | 23,538 | 1024.0 | 80,216,320 | 4.6686 | +2.2% | 7.791 | pool:59.4%, conv:31.2%, bn:5.8% |
| `mobilenet_like` | 70,786 | 512.0 | 38,716,416 | 3.3455 | +12.4% | 4.260 | dwconv:41.1%, pwconv:21.8%, bn:20.3% |
| `tiny_resnet` | 77,234 | 256.0 | 97,648,896 | 3.5810 | +4.8% | 3.656 | conv:70.1%, bn:11.9%, pwconv:11.4% |
| `small_cnn` | 97,650 | 1024.0 | 117,965,312 | 6.1093 | +0.3% | 7.274 | pool:54.9%, conv:36.2%, bn:5.6% |
| `torchvision_mobilenet` | 1,519,618 | 288.0 | 35,293,184 | 7.1446 | +22.4% | 8.662 | dwconv:31.6%, pwconv:27.5%, bn:20.5% |

## 5. Pruebas de Significancia Estadistica (Mann-Whitney U, 1 hilo, batch=1)

| Comparacion | Mediana A (ms) | Mediana B (ms) | Ratio A/B | Estadistico U | p-valor | Significativo |
|---|---|---|---|---|---|---|
| `dt (x94)` vs `ds_cnn` | 0.4804 | 1.9753 | 0.243x | 0.0 | `4.828e-67` | True |
| `dt (x94)` vs `ultra_tiny` | 0.4804 | 0.8418 | 0.571x | 0.0 | `4.827e-67` | True |
| `lr (x94)` vs `ds_cnn` | 0.6039 | 1.9753 | 0.306x | 0.0 | `4.829e-67` | True |
| `lr (x94)` vs `ultra_tiny` | 0.6039 | 0.8418 | 0.717x | 554.5 | `1.775e-63` | True |
| `lr_dot (x94)` vs `ds_cnn` | 0.3561 | 1.9753 | 0.180x | 0.0 | `4.829e-67` | True |
| `lr_dot (x94)` vs `ultra_tiny` | 0.3561 | 0.8418 | 0.423x | 0.0 | `4.827e-67` | True |
| `mlp_16_dot (x94)` vs `ds_cnn` | 0.3610 | 1.9753 | 0.183x | 0.0 | `4.828e-67` | True |
| `mlp_16_dot (x94)` vs `ultra_tiny` | 0.3610 | 0.8418 | 0.429x | 0.0 | `4.827e-67` | True |
| `rf (x94)` vs `ds_cnn` | 4.5811 | 1.9753 | 2.319x | 40000.0 | `4.831e-67` | True |
| `rf (x94)` vs `ultra_tiny` | 4.5811 | 0.8418 | 5.442x | 40000.0 | `4.830e-67` | True |

## 6. Ablacion Calidad vs Coste (`x85` vs `x9` vs `x94`)

| Clasificador | Subconjunto | Features | F1 | ROC-AUC | T_total (ms) | dF1 vs x85 | d_ms vs x85 | dF1 / d_ms |
|---|---|---|---|---|---|---|---|---|
| `dt` | `x85_only` | 85 | 0.7143 | 0.8465 | 0.1166 | +0.0000 | +0.0000 | +0.0000 |
| `dt` | `x9_only` | 9 | 0.5195 | 0.6683 | 0.2742 | -0.1948 | +0.1577 | -1.2357 |
| `dt` | `x94_full` | 94 | 0.7143 | 0.8465 | 0.2665 | +0.0000 | +0.1499 | +0.0000 |
| `knn` | `x85_only` | 85 | 0.6818 | 0.9631 | 0.5074 | +0.0000 | +0.0000 | +0.0000 |
| `knn` | `x9_only` | 9 | 0.4444 | 0.6217 | 0.8002 | -0.2374 | +0.2928 | -0.8107 |
| `knn` | `x94_full` | 94 | 0.7660 | 0.9594 | 0.6908 | +0.0841 | +0.1834 | +0.4586 |
| `lr` | `x85_only` | 85 | 0.8197 | 0.9310 | 0.2268 | +0.0000 | +0.0000 | +0.0000 |
| `lr` | `x9_only` | 9 | 0.4375 | 0.5854 | 0.3940 | -0.3822 | +0.1672 | -2.2857 |
| `lr` | `x94_full` | 94 | 0.7937 | 0.9146 | 0.3845 | -0.0260 | +0.1577 | -0.1650 |
| `lr_dot` | `x85_only` | 85 | 0.8197 | 0.9310 | 0.0314 | +0.0000 | +0.0000 | +0.0000 |
| `lr_dot` | `x9_only` | 9 | 0.4375 | 0.5854 | 0.2009 | -0.3822 | +0.1695 | -2.2547 |
| `lr_dot` | `x94_full` | 94 | 0.7937 | 0.9146 | 0.1945 | -0.0260 | +0.1631 | -0.1596 |
| `mlp_16_dot` | `x85_only` | 85 | 0.8421 | 0.9419 | 0.0369 | +0.0000 | +0.0000 | +0.0000 |
| `mlp_16_dot` | `x9_only` | 9 | 0.4000 | 0.7088 | 0.2054 | -0.4421 | +0.1686 | -2.6222 |
| `mlp_16_dot` | `x94_full` | 94 | 0.8519 | 0.9516 | 0.1994 | +0.0097 | +0.1625 | +0.0600 |
| `rf` | `x85_only` | 85 | 0.7273 | 0.9709 | 4.5931 | +0.0000 | +0.0000 | +0.0000 |
| `rf` | `x9_only` | 9 | 0.4103 | 0.6634 | 4.4653 | -0.3170 | -0.1278 | +2.4806 |
| `rf` | `x94_full` | 94 | 0.7556 | 0.9455 | 4.2863 | +0.0283 | -0.3067 | -0.0922 |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.0061 | 994.0 |
| `ds_cnn` | cuda | 1 | 1.2662 | 789.8 |
| `tiny_cnn` | cuda | 1 | 0.7361 | 1358.5 |
| `mobilenet_like` | cuda | 1 | 2.4769 | 403.7 |
| `tiny_resnet` | cuda | 1 | 1.6786 | 595.7 |
| `small_cnn` | cuda | 1 | 0.9398 | 1064.0 |
| `torchvision_mobilenet` | cuda | 1 | 9.0404 | 110.6 |
| `ultra_tiny` | cuda | 32 | 0.0387 | 25818.9 |
| `ds_cnn` | cuda | 32 | 0.2030 | 4926.3 |
| `tiny_cnn` | cuda | 32 | 0.2058 | 4858.8 |
| `mobilenet_like` | cuda | 32 | 0.2289 | 4368.9 |
| `tiny_resnet` | cuda | 32 | 0.1786 | 5597.7 |
| `small_cnn` | cuda | 32 | 0.2328 | 4295.9 |
| `torchvision_mobilenet` | cuda | 32 | 0.2789 | 3585.1 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **27.7%** (0.0627 ms) del tiempo por etapas (E2E=0.2042 ms, gap de despacho Python/ctypes=+0.0224 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.3561 ms**, F1=0.7937), mientras el maximo F1 lo obtiene `mlp_16_dot (x94)` (**F1=0.8519**, 0.3610 ms).
