# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad_transistor`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `99bc026`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.5217 | 0.9250 | 0.3344 (desc) | 0.0072 | **0.3416** | 0.1348 | [0.3409, 0.3426] | 2927.4 | - |
| `mlp_16_dot (x94)` | ml | cpu | 1537 / 94 | 0.7368 | 0.9659 | 0.3344 (desc) | 0.0117 | **0.3461** | 0.1360 | [0.3448, 0.3469] | 2889.3 | - |
| `dt (x94)` | ml | cpu | 15 / 94 | 0.3571 | 0.6898 | 0.3344 (desc) | 0.1329 | **0.4667** | 0.1396 | [0.4649, 0.4688] | 2142.7 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.5217 | 0.9250 | 0.3344 (desc) | 0.2017 | **0.5374** | 0.1535 | [0.5356, 0.5390] | 1860.6 | - |
| `knn (x94)` | ml | cpu | 23500 / 94 | 0.0000 | 0.8318 | 0.3344 (desc) | 0.4690 | **0.8061** | 0.1950 | [0.7935, 0.8136] | 1240.5 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.3158 | 0.6864 | 0.0438 (prep) | 0.8256 | **0.8742** | 0.3014 | [0.8543, 0.9602] | 1143.9 | 4.74e-07 |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.2254 | 0.8727 | 0.0438 (prep) | 1.9257 | **1.9713** | 0.7583 | [1.9528, 2.0156] | 507.3 | 3.64e-07 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.6667 | 0.8795 | 0.0438 (prep) | 3.3030 | **3.3506** | 0.8437 | [3.3051, 3.5365] | 298.5 | 4.55e-07 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.2254 | 0.6932 | 0.0438 (prep) | 3.8399 | **3.8857** | 0.4243 | [3.7972, 3.9772] | 257.4 | 1.82e-07 |
| `rf (x94)` | ml | cpu | 3236 / 94 | 0.0000 | 0.9227 | 0.3344 (desc) | 4.0492 | **4.3612** | 0.5133 | [4.3202, 4.4127] | 229.3 | - |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.3333 | 0.7227 | 0.0438 (prep) | 4.7242 | **4.7750** | 1.0958 | [4.6534, 4.9990] | 209.4 | 3.33e-07 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.8750 | 0.9659 | 0.0438 (prep) | 6.3747 | **6.4182** | 1.2424 | [6.1961, 6.6150] | 155.8 | 4.55e-07 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.2254 | 0.2943 | 0.0438 (prep) | 6.5816 | **6.6254** | 1.2527 | [6.4056, 6.7795] | 150.9 | 2.24e-03 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **7.7859 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2160 ms** vs llamada formal E2E: **0.1869 ms** (gap de dispatch/asignaciones Python = **+0.0290 ms**, +13.5%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0605 | 28.0% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.0504 | 23.3% | 2.1 | `O(g^2*K) [16x3]` |
| `asymmetry_g8` | 8x8 | 0.0490 | 22.7% | 2.6 | `O(g^2*K) [64x3]` |
| `c_fine_hist` | 8x8 | 0.0289 | 13.4% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0106 | 4.9% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 4. Desglose Medido por Capas CNN en CPU (`threads=1, batch=1`)

| Arquitectura | Params | Act Peak (KB) | FLOPs (PyTorch) | Hookless (ms) | Hook overhead (%) | Cold 1st (ms) | Familias dominantes (% hooks) |
|---|---|---|---|---|---|---|---|
| `ultra_tiny` | 1,154 | 128.0 | 1,335,424 | 0.8644 | +24.3% | 1.743 | dwconv:40.1%, bn:21.6%, conv:15.1% |
| `ds_cnn` | 12,866 | 1024.0 | 19,366,400 | 2.8555 | -18.1% | 2.427 | dwconv:37.2%, bn:20.8%, pwconv:16.5% |
| `tiny_cnn` | 23,538 | 1024.0 | 80,216,320 | 5.0726 | +3.8% | 6.220 | pool:60.9%, conv:30.3%, bn:5.3% |
| `mobilenet_like` | 70,786 | 512.0 | 38,716,416 | 3.1410 | +12.3% | 5.393 | dwconv:40.9%, pwconv:22.4%, bn:20.4% |
| `tiny_resnet` | 77,234 | 256.0 | 97,648,896 | 3.6596 | +4.6% | 3.636 | conv:70.0%, bn:12.0%, pwconv:11.4% |
| `small_cnn` | 97,650 | 1024.0 | 117,965,312 | 6.3948 | -0.3% | 10.972 | pool:58.5%, conv:33.6%, bn:4.9% |
| `torchvision_mobilenet` | 1,519,618 | 288.0 | 35,293,184 | 6.4821 | +48.1% | 10.639 | dwconv:31.6%, pwconv:27.3%, bn:20.5% |

## 5. Pruebas de Significancia Estadistica (Mann-Whitney U, 1 hilo, batch=1)

| Comparacion | Mediana A (ms) | Mediana B (ms) | Ratio A/B | Estadistico U | p-valor | Significativo |
|---|---|---|---|---|---|---|
| `dt (x94)` vs `ds_cnn` | 0.4667 | 1.9713 | 0.237x | 0.0 | `4.828e-67` | True |
| `dt (x94)` vs `ultra_tiny` | 0.4667 | 0.8742 | 0.534x | 0.0 | `4.828e-67` | True |
| `lr (x94)` vs `ds_cnn` | 0.5374 | 1.9713 | 0.273x | 0.0 | `4.828e-67` | True |
| `lr (x94)` vs `ultra_tiny` | 0.5374 | 0.8742 | 0.615x | 95.0 | `2.003e-66` | True |
| `lr_dot (x94)` vs `ds_cnn` | 0.3416 | 1.9713 | 0.173x | 0.0 | `4.823e-67` | True |
| `lr_dot (x94)` vs `ultra_tiny` | 0.3416 | 0.8742 | 0.391x | 0.0 | `4.823e-67` | True |
| `mlp_16_dot (x94)` vs `ds_cnn` | 0.3461 | 1.9713 | 0.176x | 0.0 | `4.825e-67` | True |
| `mlp_16_dot (x94)` vs `ultra_tiny` | 0.3461 | 0.8742 | 0.396x | 0.0 | `4.825e-67` | True |
| `rf (x94)` vs `ds_cnn` | 4.3612 | 1.9713 | 2.212x | 40000.0 | `4.831e-67` | True |
| `rf (x94)` vs `ultra_tiny` | 4.3612 | 0.8742 | 4.989x | 40000.0 | `4.831e-67` | True |

## 6. Ablacion Calidad vs Coste (`x85` vs `x9` vs `x94`)

| Clasificador | Subconjunto | Features | F1 | ROC-AUC | T_total (ms) | dF1 vs x85 | d_ms vs x85 | dF1 / d_ms |
|---|---|---|---|---|---|---|---|---|
| `dt` | `x85_only` | 85 | 0.3333 | 0.6693 | 0.0979 | +0.0000 | +0.0000 | +0.0000 |
| `dt` | `x9_only` | 9 | 0.1538 | 0.2841 | 0.2553 | -0.1795 | +0.1574 | -1.1400 |
| `dt` | `x94_full` | 94 | 0.3571 | 0.6898 | 0.2585 | +0.0238 | +0.1607 | +0.1482 |
| `knn` | `x85_only` | 85 | 0.0000 | 0.8670 | 0.4727 | +0.0000 | +0.0000 | +0.0000 |
| `knn` | `x9_only` | 9 | 0.3636 | 0.6886 | 0.7664 | +0.3636 | +0.2938 | +1.2379 |
| `knn` | `x94_full` | 94 | 0.0000 | 0.8318 | 0.6819 | +0.0000 | +0.2092 | +0.0000 |
| `lr` | `x85_only` | 85 | 0.5217 | 0.9295 | 0.2323 | +0.0000 | +0.0000 | +0.0000 |
| `lr` | `x9_only` | 9 | 0.3590 | 0.7273 | 0.3831 | -0.1628 | +0.1508 | -1.0790 |
| `lr` | `x94_full` | 94 | 0.5217 | 0.9250 | 0.3872 | +0.0000 | +0.1549 | +0.0000 |
| `lr_dot` | `x85_only` | 85 | 0.5217 | 0.9295 | 0.0296 | +0.0000 | +0.0000 | +0.0000 |
| `lr_dot` | `x9_only` | 9 | 0.3590 | 0.7273 | 0.1872 | -0.1628 | +0.1577 | -1.0324 |
| `lr_dot` | `x94_full` | 94 | 0.5217 | 0.9250 | 0.1904 | +0.0000 | +0.1608 | +0.0000 |
| `mlp_16_dot` | `x85_only` | 85 | 0.6667 | 0.9500 | 0.0346 | +0.0000 | +0.0000 | +0.0000 |
| `mlp_16_dot` | `x9_only` | 9 | 0.3333 | 0.8295 | 0.1923 | -0.3333 | +0.1578 | -2.1130 |
| `mlp_16_dot` | `x94_full` | 94 | 0.7368 | 0.9659 | 0.1961 | +0.0702 | +0.1615 | +0.4345 |
| `rf` | `x85_only` | 85 | 0.0000 | 0.9045 | 4.1367 | +0.0000 | +0.0000 | +0.0000 |
| `rf` | `x9_only` | 9 | 0.4000 | 0.7227 | 4.3456 | +0.4000 | +0.2089 | +1.9148 |
| `rf` | `x94_full` | 94 | 0.0000 | 0.9227 | 4.4312 | +0.0000 | +0.2945 | +0.0000 |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.2909 | 774.7 |
| `ds_cnn` | cuda | 1 | 1.5454 | 647.1 |
| `tiny_cnn` | cuda | 1 | 0.7481 | 1336.7 |
| `mobilenet_like` | cuda | 1 | 2.1907 | 456.5 |
| `tiny_resnet` | cuda | 1 | 1.6757 | 596.7 |
| `small_cnn` | cuda | 1 | 0.9462 | 1056.9 |
| `torchvision_mobilenet` | cuda | 1 | 12.4601 | 80.3 |
| `ultra_tiny` | cuda | 32 | 0.0553 | 18097.5 |
| `ds_cnn` | cuda | 32 | 0.2047 | 4886.2 |
| `tiny_cnn` | cuda | 32 | 0.2057 | 4862.2 |
| `mobilenet_like` | cuda | 32 | 0.2293 | 4360.3 |
| `tiny_resnet` | cuda | 32 | 0.1780 | 5619.1 |
| `small_cnn` | cuda | 32 | 0.2311 | 4328.1 |
| `torchvision_mobilenet` | cuda | 32 | 0.3476 | 2877.1 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **28.0%** (0.0605 ms) del tiempo por etapas (E2E=0.1869 ms, gap de despacho Python/ctypes=+0.0290 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.3416 ms**, F1=0.5217), mientras el maximo F1 lo obtiene `small_cnn` (**F1=0.8750**, 6.4182 ms).
