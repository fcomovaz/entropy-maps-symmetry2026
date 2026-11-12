# Profiling Cientifico y Comparacion 100% CPU: `kolektorsdd2`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `37d8114`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.4265 | 0.8061 | 0.3626 (desc) | 0.0076 | **0.3709** | 0.1050 | [0.3662, 0.3797] | 2696.5 | - |
| `dt (x94)` | ml | cpu | 285 / 94 | 0.2798 | 0.5962 | 0.3626 (desc) | 0.1401 | **0.5112** | 0.0929 | [0.5036, 0.5167] | 1956.2 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.4265 | 0.8061 | 0.3626 (desc) | 0.2377 | **0.6074** | 0.1076 | [0.5934, 0.6178] | 1646.4 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.6304 | 0.8815 | 0.0424 (prep) | 1.0008 | **1.0459** | 0.2632 | [1.0120, 1.0815] | 956.1 | 2.85e-07 |
| `knn (x94)` | ml | cpu | 219114 / 94 | 0.1500 | 0.6652 | 0.3626 (desc) | 1.0589 | **1.3904** | 0.7237 | [1.2623, 1.5624] | 719.2 | - |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.6562 | 0.8779 | 0.0424 (prep) | 2.8555 | **2.9092** | 1.4988 | [2.7600, 3.1804] | 343.7 | 2.38e-07 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.6250 | 0.8622 | 0.0424 (prep) | 4.7383 | **4.7843** | 2.0238 | [4.5389, 5.2490] | 209.0 | 3.61e-07 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.7130 | 0.9213 | 0.0424 (prep) | 4.8425 | **4.8865** | 1.6102 | [4.6053, 5.0198] | 204.6 | 4.78e-07 |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.6154 | 0.8714 | 0.0424 (prep) | 5.6405 | **5.6821** | 1.5790 | [5.3690, 5.9712] | 176.0 | 3.85e-07 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.5976 | 0.7332 | 0.0424 (prep) | 8.4275 | **8.4735** | 0.7378 | [8.3749, 8.5754] | 118.0 | 3.55e-07 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.5965 | 0.8927 | 0.0424 (prep) | 9.0527 | **9.0939** | 3.5100 | [8.5946, 9.6021] | 110.0 | 3.41e-07 |
| `rf (x94)` | ml | cpu | 13208 / 94 | 0.4298 | 0.7904 | 0.3626 (desc) | 9.8989 | **10.2285** | 1.6193 | [10.0881, 10.4453] | 97.8 | - |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **8.4853 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2442 ms** vs llamada formal E2E: **0.2142 ms** (gap de dispatch/asignaciones Python = **+0.0300 ms**, +12.3%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0685 | 28.1% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.0571 | 23.4% | 2.1 | `O(g^2*K) [16x3]` |
| `asymmetry_g8` | 8x8 | 0.0553 | 22.7% | 2.6 | `O(g^2*K) [64x3]` |
| `c_fine_hist` | 8x8 | 0.0333 | 13.6% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0117 | 4.8% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 4. Desglose Medido por Capas CNN en CPU (`threads=1, batch=1`)

| Arquitectura | Params | Act Peak (KB) | FLOPs (PyTorch) | Hookless (ms) | Hook overhead (%) | Cold 1st (ms) | Familias dominantes (% hooks) |
|---|---|---|---|---|---|---|---|
| `ultra_tiny` | 1,154 | 128.0 | 1,335,424 | 1.2358 | -7.8% | 1.487 | dwconv:39.6%, bn:21.9%, conv:15.0% |
| `ds_cnn` | 12,866 | 1024.0 | 19,366,400 | 3.0665 | -4.1% | 6.828 | dwconv:38.6%, bn:21.5%, pwconv:14.6% |
| `tiny_cnn` | 23,538 | 1024.0 | 80,216,320 | 4.8001 | +13.2% | 8.951 | pool:54.0%, conv:34.2%, bn:7.2% |
| `mobilenet_like` | 70,786 | 512.0 | 38,716,416 | 4.5240 | +9.0% | 5.479 | dwconv:42.5%, pwconv:20.9%, bn:20.6% |
| `tiny_resnet` | 77,234 | 256.0 | 97,648,896 | 4.2241 | +19.3% | 6.319 | conv:69.0%, bn:12.8%, pwconv:11.6% |
| `small_cnn` | 97,650 | 1024.0 | 117,965,312 | 7.0314 | +2.1% | 9.771 | pool:50.1%, conv:39.0%, bn:6.9% |
| `torchvision_mobilenet` | 1,519,618 | 288.0 | 35,293,184 | 8.4207 | +20.6% | 11.841 | dwconv:33.5%, pwconv:26.1%, bn:19.7% |

## 5. Pruebas de Significancia Estadistica (Mann-Whitney U, 1 hilo, batch=1)

| Comparacion | Mediana A (ms) | Mediana B (ms) | Ratio A/B | Estadistico U | p-valor | Significativo |
|---|---|---|---|---|---|---|
| `dt (x94)` vs `ds_cnn` | 0.5112 | 2.9092 | 0.176x | 0.0 | `4.830e-67` | True |
| `dt (x94)` vs `ultra_tiny` | 0.5112 | 1.0459 | 0.489x | 0.0 | `4.830e-67` | True |
| `lr (x94)` vs `ds_cnn` | 0.6074 | 2.9092 | 0.209x | 0.0 | `4.830e-67` | True |
| `lr (x94)` vs `ultra_tiny` | 0.6074 | 1.0459 | 0.581x | 0.0 | `4.830e-67` | True |
| `lr_dot (x94)` vs `ds_cnn` | 0.3709 | 2.9092 | 0.127x | 0.0 | `4.830e-67` | True |
| `lr_dot (x94)` vs `ultra_tiny` | 0.3709 | 1.0459 | 0.355x | 0.0 | `4.830e-67` | True |
| `rf (x94)` vs `ds_cnn` | 10.2285 | 2.9092 | 3.516x | 40000.0 | `4.831e-67` | True |
| `rf (x94)` vs `ultra_tiny` | 10.2285 | 1.0459 | 9.780x | 40000.0 | `4.831e-67` | True |

## 6. Ablacion Calidad vs Coste (`x85` vs `x9` vs `x94`)

| Clasificador | Subconjunto | Features | F1 | ROC-AUC | T_total (ms) | dF1 vs x85 | d_ms vs x85 | dF1 / d_ms |
|---|---|---|---|---|---|---|---|---|
| `dt` | `x85_only` | 85 | 0.3550 | 0.6117 | 0.1098 | +0.0000 | +0.0000 | +0.0000 |
| `dt` | `x9_only` | 9 | 0.2326 | 0.5787 | 0.3019 | -0.1224 | +0.1921 | -0.6371 |
| `dt` | `x94_full` | 94 | 0.2798 | 0.5962 | 0.3125 | -0.0751 | +0.2027 | -0.3707 |
| `knn` | `x85_only` | 85 | 0.1513 | 0.6442 | 1.0634 | +0.0000 | +0.0000 | +0.0000 |
| `knn` | `x9_only` | 9 | 0.2208 | 0.6328 | 0.8821 | +0.0695 | -0.1813 | -0.3836 |
| `knn` | `x94_full` | 94 | 0.1500 | 0.6652 | 1.0693 | -0.0013 | +0.0059 | -0.2118 |
| `lr` | `x85_only` | 85 | 0.2687 | 0.6179 | 0.2767 | +0.0000 | +0.0000 | +0.0000 |
| `lr` | `x9_only` | 9 | 0.3591 | 0.7371 | 0.4709 | +0.0905 | +0.1942 | +0.4657 |
| `lr` | `x94_full` | 94 | 0.4265 | 0.8061 | 0.4762 | +0.1579 | +0.1995 | +0.7915 |
| `lr_dot` | `x85_only` | 85 | 0.2687 | 0.6179 | 0.0348 | +0.0000 | +0.0000 | +0.0000 |
| `lr_dot` | `x9_only` | 9 | 0.3591 | 0.7371 | 0.2273 | +0.0905 | +0.1925 | +0.4699 |
| `lr_dot` | `x94_full` | 94 | 0.4265 | 0.8061 | 0.2361 | +0.1579 | +0.2013 | +0.7842 |
| `rf` | `x85_only` | 85 | 0.4286 | 0.8255 | 9.4624 | +0.0000 | +0.0000 | +0.0000 |
| `rf` | `x9_only` | 9 | 0.3765 | 0.7429 | 9.7220 | -0.0521 | +0.2596 | -0.2007 |
| `rf` | `x94_full` | 94 | 0.4298 | 0.7904 | 9.8575 | +0.0013 | +0.3951 | +0.0032 |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.1984 | 834.4 |
| `ds_cnn` | cuda | 1 | 1.5200 | 657.9 |
| `tiny_cnn` | cuda | 1 | 0.7828 | 1277.5 |
| `mobilenet_like` | cuda | 1 | 2.8194 | 354.7 |
| `tiny_resnet` | cuda | 1 | 2.0274 | 493.2 |
| `small_cnn` | cuda | 1 | 1.1338 | 882.0 |
| `torchvision_mobilenet` | cuda | 1 | 10.0429 | 99.6 |
| `ultra_tiny` | cuda | 32 | 0.0515 | 19418.7 |
| `ds_cnn` | cuda | 32 | 0.2056 | 4863.3 |
| `tiny_cnn` | cuda | 32 | 0.2080 | 4806.6 |
| `mobilenet_like` | cuda | 32 | 0.2304 | 4340.0 |
| `tiny_resnet` | cuda | 32 | 0.1796 | 5568.0 |
| `small_cnn` | cuda | 32 | 0.2336 | 4281.0 |
| `torchvision_mobilenet` | cuda | 32 | 0.3222 | 3103.8 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **28.1%** (0.0685 ms) del tiempo por etapas (E2E=0.2142 ms, gap de despacho Python/ctypes=+0.0300 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.3709 ms**, F1=0.4265), mientras el maximo F1 lo obtiene `tiny_resnet` (**F1=0.7130**, 4.8865 ms).
