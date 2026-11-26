# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad_wood`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `99bc026`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.4828 | 0.7392 | 0.2465 (desc) | 0.0069 | **0.2534** | 0.1066 | [0.2523, 0.2600] | 3946.3 | - |
| `mlp_16_dot (x94)` | ml | cpu | 1537 / 94 | 0.5333 | 0.7238 | 0.2465 (desc) | 0.0117 | **0.2582** | 0.1067 | [0.2571, 0.2647] | 3873.0 | - |
| `dt (x94)` | ml | cpu | 13 / 94 | 0.3333 | 0.6343 | 0.2465 (desc) | 0.0841 | **0.3481** | 0.1065 | [0.3408, 0.3589] | 2872.7 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.4828 | 0.7392 | 0.2465 (desc) | 0.2037 | **0.4730** | 0.1101 | [0.4615, 0.5007] | 2114.2 | - |
| `knn (x94)` | ml | cpu | 24440 / 94 | 0.0000 | 0.7207 | 0.2465 (desc) | 0.4817 | **0.7863** | 0.1888 | [0.7480, 0.8090] | 1271.8 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.3000 | 0.6127 | 0.0468 (prep) | 0.8239 | **0.8864** | 0.2229 | [0.8654, 0.9030] | 1128.2 | 3.21e-07 |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.4211 | 0.8117 | 0.0468 (prep) | 1.9318 | **1.9827** | 0.8879 | [1.9245, 2.0319] | 504.4 | 3.95e-07 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.3077 | 0.4367 | 0.0468 (prep) | 2.9584 | **3.0099** | 0.7397 | [2.9610, 3.0627] | 332.2 | 7.71e-04 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.0000 | 0.5910 | 0.0468 (prep) | 3.2736 | **3.3208** | 0.8033 | [3.2666, 3.4461] | 301.1 | 3.83e-07 |
| `rf (x94)` | ml | cpu | 4332 / 94 | 0.2667 | 0.7562 | 0.2465 (desc) | 4.2252 | **4.5227** | 0.7231 | [4.4435, 4.6502] | 221.1 | - |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.5000 | 0.8009 | 0.0468 (prep) | 5.0644 | **5.1104** | 0.9587 | [5.0072, 5.2900] | 195.7 | 4.44e-07 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.1538 | 0.7469 | 0.0468 (prep) | 6.0347 | **6.0884** | 1.1504 | [6.0099, 6.2501] | 164.2 | 4.20e-07 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.3077 | 0.6667 | 0.0468 (prep) | 6.5307 | **6.5759** | 1.1750 | [6.4398, 6.7661] | 152.1 | 5.25e-02 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **7.7335 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2244 ms** vs llamada formal E2E: **0.1927 ms** (gap de dispatch/asignaciones Python = **+0.0317 ms**, +14.1%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0632 | 28.2% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.0527 | 23.5% | 2.1 | `O(g^2*K) [16x3]` |
| `asymmetry_g8` | 8x8 | 0.0511 | 22.8% | 2.6 | `O(g^2*K) [64x3]` |
| `c_fine_hist` | 8x8 | 0.0292 | 13.0% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0110 | 4.9% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 4. Desglose Medido por Capas CNN en CPU (`threads=1, batch=1`)

| Arquitectura | Params | Act Peak (KB) | FLOPs (PyTorch) | Hookless (ms) | Hook overhead (%) | Cold 1st (ms) | Familias dominantes (% hooks) |
|---|---|---|---|---|---|---|---|
| `ultra_tiny` | 1,154 | 128.0 | 1,335,424 | 0.8476 | +27.1% | 1.149 | dwconv:39.5%, bn:21.7%, conv:15.2% |
| `ds_cnn` | 12,866 | 1024.0 | 19,366,400 | 2.4236 | +2.7% | 7.482 | dwconv:37.4%, bn:20.9%, pwconv:16.2% |
| `tiny_cnn` | 23,538 | 1024.0 | 80,216,320 | 8.5207 | -29.5% | 12.193 | pool:62.7%, conv:27.9%, bn:5.3% |
| `mobilenet_like` | 70,786 | 512.0 | 38,716,416 | 3.5857 | +14.4% | 5.433 | dwconv:41.3%, pwconv:21.9%, bn:20.5% |
| `tiny_resnet` | 77,234 | 256.0 | 97,648,896 | 3.7230 | +16.8% | 3.976 | conv:69.4%, bn:12.1%, pwconv:11.7% |
| `small_cnn` | 97,650 | 1024.0 | 117,965,312 | 8.5885 | +6.2% | 8.547 | pool:55.6%, conv:33.7%, bn:6.8% |
| `torchvision_mobilenet` | 1,519,618 | 288.0 | 35,293,184 | 6.8907 | +47.5% | 8.543 | dwconv:32.5%, pwconv:26.7%, bn:20.1% |

## 5. Pruebas de Significancia Estadistica (Mann-Whitney U, 1 hilo, batch=1)

| Comparacion | Mediana A (ms) | Mediana B (ms) | Ratio A/B | Estadistico U | p-valor | Significativo |
|---|---|---|---|---|---|---|
| `dt (x94)` vs `ds_cnn` | 0.3481 | 1.9827 | 0.176x | 0.0 | `4.829e-67` | True |
| `dt (x94)` vs `ultra_tiny` | 0.3481 | 0.8864 | 0.393x | 0.0 | `4.828e-67` | True |
| `lr (x94)` vs `ds_cnn` | 0.4730 | 1.9827 | 0.239x | 0.0 | `4.829e-67` | True |
| `lr (x94)` vs `ultra_tiny` | 0.4730 | 0.8864 | 0.534x | 0.0 | `4.829e-67` | True |
| `lr_dot (x94)` vs `ds_cnn` | 0.2534 | 1.9827 | 0.128x | 0.0 | `4.827e-67` | True |
| `lr_dot (x94)` vs `ultra_tiny` | 0.2534 | 0.8864 | 0.286x | 0.0 | `4.827e-67` | True |
| `mlp_16_dot (x94)` vs `ds_cnn` | 0.2582 | 1.9827 | 0.130x | 0.0 | `4.827e-67` | True |
| `mlp_16_dot (x94)` vs `ultra_tiny` | 0.2582 | 0.8864 | 0.291x | 0.0 | `4.827e-67` | True |
| `rf (x94)` vs `ds_cnn` | 4.5227 | 1.9827 | 2.281x | 39930.0 | `1.379e-66` | True |
| `rf (x94)` vs `ultra_tiny` | 4.5227 | 0.8864 | 5.102x | 40000.0 | `4.830e-67` | True |

## 6. Ablacion Calidad vs Coste (`x85` vs `x9` vs `x94`)

| Clasificador | Subconjunto | Features | F1 | ROC-AUC | T_total (ms) | dF1 vs x85 | d_ms vs x85 | dF1 / d_ms |
|---|---|---|---|---|---|---|---|---|
| `dt` | `x85_only` | 85 | 0.4242 | 0.6728 | 0.1062 | +0.0000 | +0.0000 | +0.0000 |
| `dt` | `x9_only` | 9 | 0.2667 | 0.5448 | 0.3539 | -0.1576 | +0.2478 | -0.6359 |
| `dt` | `x94_full` | 94 | 0.3333 | 0.6343 | 0.2819 | -0.0909 | +0.1758 | -0.5173 |
| `knn` | `x85_only` | 85 | 0.1538 | 0.7122 | 0.7750 | +0.0000 | +0.0000 | +0.0000 |
| `knn` | `x9_only` | 9 | 0.1333 | 0.4884 | 1.0623 | -0.0205 | +0.2873 | -0.0714 |
| `knn` | `x94_full` | 94 | 0.0000 | 0.7207 | 1.1129 | -0.1538 | +0.3380 | -0.4552 |
| `lr` | `x85_only` | 85 | 0.4242 | 0.7315 | 0.2681 | +0.0000 | +0.0000 | +0.0000 |
| `lr` | `x9_only` | 9 | 0.3684 | 0.6728 | 0.6664 | -0.0558 | +0.3983 | -0.1401 |
| `lr` | `x94_full` | 94 | 0.4828 | 0.7392 | 0.5172 | +0.0585 | +0.2491 | +0.2349 |
| `lr_dot` | `x85_only` | 85 | 0.4242 | 0.7315 | 0.0346 | +0.0000 | +0.0000 | +0.0000 |
| `lr_dot` | `x9_only` | 9 | 0.3684 | 0.6728 | 0.2735 | -0.0558 | +0.2389 | -0.2337 |
| `lr_dot` | `x94_full` | 94 | 0.4828 | 0.7392 | 0.2087 | +0.0585 | +0.1742 | +0.3360 |
| `mlp_16_dot` | `x85_only` | 85 | 0.4848 | 0.7315 | 0.0389 | +0.0000 | +0.0000 | +0.0000 |
| `mlp_16_dot` | `x9_only` | 9 | 0.3902 | 0.6605 | 0.2781 | -0.0946 | +0.2391 | -0.3957 |
| `mlp_16_dot` | `x94_full` | 94 | 0.5333 | 0.7238 | 0.2145 | +0.0485 | +0.1755 | +0.2763 |
| `rf` | `x85_only` | 85 | 0.2500 | 0.7531 | 4.2304 | +0.0000 | +0.0000 | +0.0000 |
| `rf` | `x9_only` | 9 | 0.1053 | 0.5787 | 4.4626 | -0.1447 | +0.2322 | -0.6233 |
| `rf` | `x94_full` | 94 | 0.2667 | 0.7562 | 4.3942 | +0.0167 | +0.1638 | +0.1018 |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.1420 | 875.7 |
| `ds_cnn` | cuda | 1 | 1.2744 | 784.7 |
| `tiny_cnn` | cuda | 1 | 1.1524 | 867.7 |
| `mobilenet_like` | cuda | 1 | 2.7140 | 368.5 |
| `tiny_resnet` | cuda | 1 | 2.0900 | 478.5 |
| `small_cnn` | cuda | 1 | 1.5349 | 651.5 |
| `torchvision_mobilenet` | cuda | 1 | 10.8718 | 92.0 |
| `ultra_tiny` | cuda | 32 | 0.0459 | 21806.5 |
| `ds_cnn` | cuda | 32 | 0.2085 | 4795.1 |
| `tiny_cnn` | cuda | 32 | 0.2051 | 4875.0 |
| `mobilenet_like` | cuda | 32 | 0.2297 | 4352.9 |
| `tiny_resnet` | cuda | 32 | 0.1769 | 5654.5 |
| `small_cnn` | cuda | 32 | 0.2322 | 4307.5 |
| `torchvision_mobilenet` | cuda | 32 | 0.3012 | 3319.6 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **28.2%** (0.0632 ms) del tiempo por etapas (E2E=0.1927 ms, gap de despacho Python/ctypes=+0.0317 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.2534 ms**, F1=0.4828), mientras el maximo F1 lo obtiene `mlp_16_dot (x94)` (**F1=0.5333**, 0.2582 ms).
