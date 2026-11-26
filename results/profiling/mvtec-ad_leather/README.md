# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad_leather`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `99bc026`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.7500 | 0.9296 | 0.3468 (desc) | 0.0069 | **0.3537** | 0.0297 | [0.3513, 0.3550] | 2827.3 | - |
| `mlp_16_dot (x94)` | ml | cpu | 1537 / 94 | 0.8000 | 0.9603 | 0.3468 (desc) | 0.0124 | **0.3592** | 0.0295 | [0.3567, 0.3607] | 2784.0 | - |
| `dt (x94)` | ml | cpu | 15 / 94 | 0.6667 | 0.8591 | 0.3468 (desc) | 0.0955 | **0.4438** | 0.0538 | [0.4409, 0.4533] | 2253.0 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.7500 | 0.9296 | 0.3468 (desc) | 0.1998 | **0.5497** | 0.0470 | [0.5479, 0.5553] | 1819.0 | - |
| `knn (x94)` | ml | cpu | 27730 / 94 | 0.6897 | 0.9430 | 0.3468 (desc) | 0.4718 | **0.8235** | 0.0766 | [0.8165, 0.8311] | 1214.4 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.3913 | 0.7222 | 0.0459 (prep) | 0.8049 | **0.8619** | 0.2163 | [0.8484, 0.8810] | 1160.2 | 3.48e-07 |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.7273 | 0.9415 | 0.0459 (prep) | 1.8994 | **1.9548** | 0.8050 | [1.9377, 2.0283] | 511.5 | 4.84e-07 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.0000 | 0.9554 | 0.0459 (prep) | 2.9611 | **3.0120** | 0.5513 | [2.9775, 3.1043] | 332.0 | 2.43e-07 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.6000 | 0.9335 | 0.0459 (prep) | 3.2298 | **3.2770** | 0.8757 | [3.2511, 3.3888] | 305.2 | 2.86e-07 |
| `rf (x94)` | ml | cpu | 3288 / 94 | 0.7222 | 0.9355 | 0.3468 (desc) | 4.2358 | **4.5710** | 0.8792 | [4.4997, 4.6936] | 218.8 | - |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.6977 | 0.9147 | 0.0459 (prep) | 5.4041 | **5.4489** | 1.1959 | [5.2810, 5.5486] | 183.5 | 4.60e-07 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.3913 | 0.5868 | 0.0459 (prep) | 6.5372 | **6.5896** | 1.1651 | [6.4341, 6.7523] | 151.8 | 4.61e-02 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.8571 | 0.9673 | 0.0459 (prep) | 7.8574 | **7.9246** | 1.4012 | [7.8470, 8.0332] | 126.2 | 4.76e-07 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **7.2698 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2180 ms** vs llamada formal E2E: **0.1888 ms** (gap de dispatch/asignaciones Python = **+0.0293 ms**, +13.4%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0610 | 28.0% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.0504 | 23.1% | 2.1 | `O(g^2*K) [16x3]` |
| `asymmetry_g8` | 8x8 | 0.0491 | 22.5% | 2.6 | `O(g^2*K) [64x3]` |
| `c_fine_hist` | 8x8 | 0.0299 | 13.7% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0109 | 5.0% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 4. Desglose Medido por Capas CNN en CPU (`threads=1, batch=1`)

| Arquitectura | Params | Act Peak (KB) | FLOPs (PyTorch) | Hookless (ms) | Hook overhead (%) | Cold 1st (ms) | Familias dominantes (% hooks) |
|---|---|---|---|---|---|---|---|
| `ultra_tiny` | 1,154 | 128.0 | 1,335,424 | 0.9301 | +16.2% | 1.664 | dwconv:40.1%, bn:21.4%, conv:15.3% |
| `ds_cnn` | 12,866 | 1024.0 | 19,366,400 | 2.2292 | -0.7% | 2.820 | dwconv:36.9%, bn:20.6%, pwconv:16.4% |
| `tiny_cnn` | 23,538 | 1024.0 | 80,216,320 | 5.8318 | -6.4% | 6.666 | pool:63.2%, conv:28.5%, bn:4.9% |
| `mobilenet_like` | 70,786 | 512.0 | 38,716,416 | 3.5293 | +2.0% | 4.015 | dwconv:40.8%, pwconv:22.4%, bn:20.5% |
| `tiny_resnet` | 77,234 | 256.0 | 97,648,896 | 3.4055 | +17.5% | 3.723 | conv:70.5%, bn:11.8%, pwconv:11.2% |
| `small_cnn` | 97,650 | 1024.0 | 117,965,312 | 6.7660 | -1.4% | 8.755 | pool:59.7%, conv:32.4%, bn:4.8% |
| `torchvision_mobilenet` | 1,519,618 | 288.0 | 35,293,184 | 6.8919 | +24.7% | 7.056 | dwconv:31.2%, pwconv:27.5%, bn:20.6% |

## 5. Pruebas de Significancia Estadistica (Mann-Whitney U, 1 hilo, batch=1)

| Comparacion | Mediana A (ms) | Mediana B (ms) | Ratio A/B | Estadistico U | p-valor | Significativo |
|---|---|---|---|---|---|---|
| `dt (x94)` vs `ds_cnn` | 0.4438 | 1.9548 | 0.227x | 0.0 | `4.830e-67` | True |
| `dt (x94)` vs `ultra_tiny` | 0.4438 | 0.8619 | 0.515x | 0.0 | `4.829e-67` | True |
| `lr (x94)` vs `ds_cnn` | 0.5497 | 1.9548 | 0.281x | 0.0 | `4.828e-67` | True |
| `lr (x94)` vs `ultra_tiny` | 0.5497 | 0.8619 | 0.638x | 65.0 | `1.279e-66` | True |
| `lr_dot (x94)` vs `ds_cnn` | 0.3537 | 1.9548 | 0.181x | 0.0 | `4.829e-67` | True |
| `lr_dot (x94)` vs `ultra_tiny` | 0.3537 | 0.8619 | 0.410x | 0.0 | `4.828e-67` | True |
| `mlp_16_dot (x94)` vs `ds_cnn` | 0.3592 | 1.9548 | 0.184x | 0.0 | `4.828e-67` | True |
| `mlp_16_dot (x94)` vs `ultra_tiny` | 0.3592 | 0.8619 | 0.417x | 0.0 | `4.827e-67` | True |
| `rf (x94)` vs `ds_cnn` | 4.5710 | 1.9548 | 2.338x | 39995.0 | `5.207e-67` | True |
| `rf (x94)` vs `ultra_tiny` | 4.5710 | 0.8619 | 5.303x | 40000.0 | `4.830e-67` | True |

## 6. Ablacion Calidad vs Coste (`x85` vs `x9` vs `x94`)

| Clasificador | Subconjunto | Features | F1 | ROC-AUC | T_total (ms) | dF1 vs x85 | d_ms vs x85 | dF1 / d_ms |
|---|---|---|---|---|---|---|---|---|
| `dt` | `x85_only` | 85 | 0.6667 | 0.7847 | 0.0970 | +0.0000 | +0.0000 | +0.0000 |
| `dt` | `x9_only` | 9 | 0.4727 | 0.7822 | 0.2598 | -0.1939 | +0.1628 | -1.1916 |
| `dt` | `x94_full` | 94 | 0.6667 | 0.8591 | 0.2652 | +0.0000 | +0.1682 | +0.0000 |
| `knn` | `x85_only` | 85 | 0.6667 | 0.8730 | 0.5137 | +0.0000 | +0.0000 | +0.0000 |
| `knn` | `x9_only` | 9 | 0.5600 | 0.8323 | 0.7838 | -0.1067 | +0.2701 | -0.3948 |
| `knn` | `x94_full` | 94 | 0.6897 | 0.9430 | 0.8745 | +0.0230 | +0.3609 | +0.0637 |
| `lr` | `x85_only` | 85 | 0.7368 | 0.8929 | 0.6372 | +0.0000 | +0.0000 | +0.0000 |
| `lr` | `x9_only` | 9 | 0.6667 | 0.8938 | 0.7452 | -0.0702 | +0.1080 | -0.6498 |
| `lr` | `x94_full` | 94 | 0.7500 | 0.9296 | 0.4480 | +0.0132 | -0.1892 | -0.0695 |
| `lr_dot` | `x85_only` | 85 | 0.7368 | 0.8929 | 0.0447 | +0.0000 | +0.0000 | +0.0000 |
| `lr_dot` | `x9_only` | 9 | 0.6667 | 0.8938 | 0.1882 | -0.0702 | +0.1435 | -0.4892 |
| `lr_dot` | `x94_full` | 94 | 0.7500 | 0.9296 | 0.1940 | +0.0132 | +0.1493 | +0.0881 |
| `mlp_16_dot` | `x85_only` | 85 | 0.7273 | 0.9405 | 0.0343 | +0.0000 | +0.0000 | +0.0000 |
| `mlp_16_dot` | `x9_only` | 9 | 0.6383 | 0.8720 | 0.1946 | -0.0890 | +0.1603 | -0.5551 |
| `mlp_16_dot` | `x94_full` | 94 | 0.8000 | 0.9603 | 0.1977 | +0.0727 | +0.1634 | +0.4450 |
| `rf` | `x85_only` | 85 | 0.6316 | 0.9137 | 4.2247 | +0.0000 | +0.0000 | +0.0000 |
| `rf` | `x9_only` | 9 | 0.6207 | 0.8056 | 5.6175 | -0.0109 | +1.3928 | -0.0078 |
| `rf` | `x94_full` | 94 | 0.7222 | 0.9355 | 4.6212 | +0.0906 | +0.3965 | +0.2286 |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.2581 | 794.8 |
| `ds_cnn` | cuda | 1 | 1.2503 | 799.8 |
| `tiny_cnn` | cuda | 1 | 0.7302 | 1369.4 |
| `mobilenet_like` | cuda | 1 | 2.2170 | 451.0 |
| `tiny_resnet` | cuda | 1 | 1.6939 | 590.4 |
| `small_cnn` | cuda | 1 | 0.9455 | 1057.7 |
| `torchvision_mobilenet` | cuda | 1 | 8.7107 | 114.8 |
| `ultra_tiny` | cuda | 32 | 0.0439 | 22801.0 |
| `ds_cnn` | cuda | 32 | 0.2099 | 4764.4 |
| `tiny_cnn` | cuda | 32 | 0.2052 | 4874.0 |
| `mobilenet_like` | cuda | 32 | 0.2295 | 4357.7 |
| `tiny_resnet` | cuda | 32 | 0.1783 | 5608.7 |
| `small_cnn` | cuda | 32 | 0.2322 | 4306.1 |
| `torchvision_mobilenet` | cuda | 32 | 0.2816 | 3551.0 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **28.0%** (0.0610 ms) del tiempo por etapas (E2E=0.1888 ms, gap de despacho Python/ctypes=+0.0293 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.3537 ms**, F1=0.7500), mientras el maximo F1 lo obtiene `small_cnn` (**F1=0.8571**, 7.9246 ms).
