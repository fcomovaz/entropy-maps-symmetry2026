# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad_bottle`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `bf4a29f`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.4651 | 0.7709 | 0.0401 (prep) | 1.0294 | **1.0753** | 0.4999 | [1.0172, 1.1685] | 930.0 | 4.92e-07 |
| `dt (x94)` | ml | cpu | 15 / 94 | 0.5833 | 0.6614 | 2.4089 (desc) | 0.0903 | **2.5014** | 0.8678 | [2.4285, 2.6323] | 399.8 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.8000 | 0.9231 | 2.4089 (desc) | 0.2094 | **2.6319** | 0.8485 | [2.5328, 2.7659] | 380.0 | - |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.5714 | 0.8980 | 0.0401 (prep) | 2.7598 | **2.7988** | 1.3685 | [2.6392, 3.1841] | 357.3 | 4.78e-07 |
| `knn (x94)` | ml | cpu | 21902 / 94 | 0.3750 | 0.7057 | 2.4089 (desc) | 0.5266 | **3.0646** | 0.8343 | [2.9565, 3.1872] | 326.3 | - |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.3529 | 0.9047 | 0.0401 (prep) | 3.2877 | **3.3307** | 1.1427 | [3.2675, 3.4259] | 300.2 | 3.22e-07 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.3611 | 0.8512 | 0.0401 (prep) | 4.0658 | **4.1059** | 1.2694 | [3.9852, 4.2472] | 243.6 | 4.31e-07 |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.6400 | 0.8896 | 0.0401 (prep) | 4.8760 | **4.9230** | 1.0162 | [4.7780, 5.1019] | 203.1 | 3.73e-07 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.8571 | 0.9816 | 0.0401 (prep) | 6.4611 | **6.5010** | 1.5335 | [6.1951, 6.8064] | 153.8 | 3.90e-07 |
| `rf (x94)` | ml | cpu | 3496 / 94 | 0.7273 | 0.9030 | 2.4089 (desc) | 4.3149 | **7.0108** | 1.2360 | [6.8322, 7.1931] | 142.6 | - |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.3611 | 0.5385 | 0.0401 (prep) | 7.3395 | **7.3829** | 1.7803 | [7.2285, 7.6467] | 135.4 | 8.28e-02 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E)

- Inicializacion amortizable (`build_entropy_plan` + LUTs): **0.7643 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **1.5873 ms** vs llamada formal E2E: **2.3266 ms** (gap de dispatch/asignaciones Python = **-0.7393 ms**, -46.6%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `region_histograms_g8` | 8x8 | 0.6141 | 38.7% | 13.4 | `O(H*W) [2*H*W=32768]` |
| `region_histograms_g4` | 4x4 | 0.2371 | 14.9% | 16.3 | `O(H*W) [2*H*W=32768]` |
| `region_histograms_g2` | 2x2 | 0.1318 | 8.3% | 50.8 | `O(H*W) [2*H*W=32768]` |
| `region_histograms_g1` | 1x1 | 0.1074 | 6.8% | 193.2 | `O(H*W) [2*H*W=32768]` |
| `asymmetry_g2` | 2x2 | 0.0934 | 5.9% | 2.1 | `O(g^2*K) [4x3]` |

## 4. Desglose Medido por Capas CNN en CPU (`threads=1, batch=1`)

| Arquitectura | Params | Act Peak (KB) | FLOPs (PyTorch) | Hookless (ms) | Hook overhead (%) | Cold 1st (ms) | Familias dominantes (% hooks) |
|---|---|---|---|---|---|---|---|
| `ultra_tiny` | 1,154 | 128.0 | 1,335,424 | 1.3624 | +1.1% | 1.951 | dwconv:40.8%, bn:20.2%, conv:13.7% |
| `ds_cnn` | 12,866 | 1024.0 | 19,366,400 | 2.3875 | +5.1% | 4.042 | dwconv:37.7%, bn:20.3%, pwconv:17.4% |
| `tiny_cnn` | 23,538 | 1024.0 | 80,216,320 | 4.7439 | +5.3% | 8.028 | pool:58.1%, conv:32.5%, bn:5.6% |
| `mobilenet_like` | 70,786 | 512.0 | 38,716,416 | 3.2306 | +16.0% | 5.562 | dwconv:41.6%, pwconv:22.3%, bn:19.8% |
| `tiny_resnet` | 77,234 | 256.0 | 97,648,896 | 3.4223 | +14.8% | 3.861 | conv:70.6%, bn:11.6%, pwconv:11.3% |
| `small_cnn` | 97,650 | 1024.0 | 117,965,312 | 6.6194 | +20.0% | 9.491 | pool:55.0%, conv:36.1%, bn:5.6% |
| `torchvision_mobilenet` | 1,519,618 | 288.0 | 35,293,184 | 7.4872 | +41.0% | 8.133 | dwconv:32.8%, pwconv:26.7%, bn:19.6% |

## 5. Pruebas de Significancia Estadistica (Mann-Whitney U, 1 hilo, batch=1)

| Comparacion | Mediana A (ms) | Mediana B (ms) | Ratio A/B | Estadistico U | p-valor | Significativo |
|---|---|---|---|---|---|---|
| `dt (x94)` vs `ds_cnn` | 2.5014 | 2.7988 | 0.894x | 16852.0 | `6.481e-03` | True |
| `dt (x94)` vs `ultra_tiny` | 2.5014 | 1.0753 | 2.326x | 39195.0 | `6.720e-62` | True |
| `lr (x94)` vs `ds_cnn` | 2.6319 | 2.7988 | 0.940x | 19056.0 | `4.145e-01` | False |
| `lr (x94)` vs `ultra_tiny` | 2.6319 | 1.0753 | 2.448x | 39385.0 | `4.289e-63` | True |
| `rf (x94)` vs `ds_cnn` | 7.0108 | 2.7988 | 2.505x | 39912.0 | `1.805e-66` | True |
| `rf (x94)` vs `ultra_tiny` | 7.0108 | 1.0753 | 6.520x | 40000.0 | `4.831e-67` | True |

## 6. Ablacion Calidad vs Coste (`x85` vs `x9` vs `x94`)

| Clasificador | Subconjunto | Features | F1 | ROC-AUC | T_total (ms) | dF1 vs x85 | d_ms vs x85 | dF1 / d_ms |
|---|---|---|---|---|---|---|---|---|
| `dt` | `x85_only` | 85 | 0.5714 | 0.6781 | 2.1850 | +0.0000 | +0.0000 | +0.0000 |
| `dt` | `x9_only` | 9 | 0.5882 | 0.7584 | 2.5115 | +0.0168 | +0.3264 | +0.0515 |
| `dt` | `x94_full` | 94 | 0.5833 | 0.6614 | 2.9958 | +0.0119 | +0.8108 | +0.0147 |
| `knn` | `x85_only` | 85 | 0.1429 | 0.6605 | 2.7007 | +0.0000 | +0.0000 | +0.0000 |
| `knn` | `x9_only` | 9 | 0.6667 | 0.8997 | 3.1860 | +0.5238 | +0.4853 | +1.0795 |
| `knn` | `x94_full` | 94 | 0.3750 | 0.7057 | 3.7382 | +0.2321 | +1.0374 | +0.2238 |
| `lr` | `x85_only` | 85 | 0.7333 | 0.9181 | 2.3105 | +0.0000 | +0.0000 | +0.0000 |
| `lr` | `x9_only` | 9 | 0.7333 | 0.9130 | 2.6288 | +0.0000 | +0.3182 | +0.0000 |
| `lr` | `x94_full` | 94 | 0.8000 | 0.9231 | 3.1635 | +0.0667 | +0.8529 | +0.0782 |
| `rf` | `x85_only` | 85 | 0.5263 | 0.8696 | 7.1631 | +0.0000 | +0.0000 | +0.0000 |
| `rf` | `x9_only` | 9 | 0.6923 | 0.8980 | 7.0624 | +0.1660 | -0.1006 | -1.6500 |
| `rf` | `x94_full` | 94 | 0.7273 | 0.9030 | 9.2192 | +0.2010 | +2.0562 | +0.0977 |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.1010 | 908.3 |
| `ds_cnn` | cuda | 1 | 2.1704 | 460.7 |
| `tiny_cnn` | cuda | 1 | 1.0122 | 988.0 |
| `mobilenet_like` | cuda | 1 | 2.6860 | 372.3 |
| `tiny_resnet` | cuda | 1 | 3.2312 | 309.5 |
| `small_cnn` | cuda | 1 | 1.5575 | 642.1 |
| `torchvision_mobilenet` | cuda | 1 | 11.3613 | 88.0 |
| `ultra_tiny` | cuda | 32 | 0.0449 | 22260.9 |
| `ds_cnn` | cuda | 32 | 0.2062 | 4849.6 |
| `tiny_cnn` | cuda | 32 | 0.2097 | 4769.5 |
| `mobilenet_like` | cuda | 32 | 0.2331 | 4289.3 |
| `tiny_resnet` | cuda | 32 | 0.1791 | 5584.5 |
| `small_cnn` | cuda | 32 | 0.2327 | 4297.3 |
| `torchvision_mobilenet` | cuda | 32 | 0.5513 | 1814.0 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Cuello de botella principal de x94:** `region_histograms_g8` consume **38.7%** (0.6141 ms) del tiempo por etapas; cualquier optimizacion futura en C/Cython debe priorizar el bucle de histogramas multi-escala antes que cuantizacion o asimetria.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `ultra_tiny` (**1.0753 ms**, F1=0.4651), mientras el maximo F1 lo obtiene `small_cnn` (**F1=0.8571**, 6.5010 ms).
