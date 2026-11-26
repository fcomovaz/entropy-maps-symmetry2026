# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad_grid`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `99bc026`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.3704 | 0.5893 | 0.2477 (desc) | 0.0077 | **0.2565** | 0.1437 | [0.2541, 0.2754] | 3898.6 | - |
| `mlp_16_dot (x94)` | ml | cpu | 1537 / 94 | 0.3571 | 0.7335 | 0.2477 (desc) | 0.0129 | **0.2613** | 0.1458 | [0.2593, 0.2852] | 3827.0 | - |
| `dt (x94)` | ml | cpu | 13 / 94 | 0.2979 | 0.5893 | 0.2477 (desc) | 0.0842 | **0.3326** | 0.1476 | [0.3310, 0.3443] | 3006.6 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.3704 | 0.5893 | 0.2477 (desc) | 0.2129 | **0.4904** | 0.1491 | [0.4649, 0.5568] | 2038.9 | - |
| `knn (x94)` | ml | cpu | 25662 / 94 | 0.0000 | 0.5180 | 0.2477 (desc) | 0.4756 | **0.7627** | 0.1405 | [0.7335, 0.8075] | 1311.2 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.3704 | 0.7696 | 0.0417 (prep) | 0.7654 | **0.8077** | 0.1340 | [0.8062, 0.8088] | 1238.1 | 4.76e-07 |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.2750 | 0.4420 | 0.0417 (prep) | 1.8870 | **1.9286** | 0.4426 | [1.9089, 1.9711] | 518.5 | 2.90e-07 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.3636 | 0.7696 | 0.0417 (prep) | 2.8713 | **2.9166** | 0.8568 | [2.8920, 2.9917] | 342.9 | 4.76e-07 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.3056 | 0.6708 | 0.0417 (prep) | 3.3083 | **3.3506** | 0.8456 | [3.2763, 3.5251] | 298.5 | 4.83e-07 |
| `rf (x94)` | ml | cpu | 5048 / 94 | 0.1538 | 0.6105 | 0.2477 (desc) | 4.1173 | **4.4106** | 0.5656 | [4.3584, 4.4584] | 226.7 | - |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.3529 | 0.7351 | 0.0417 (prep) | 4.5948 | **4.6366** | 0.7705 | [4.5641, 4.7339] | 215.7 | 4.70e-07 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.4545 | 0.7947 | 0.0417 (prep) | 6.3865 | **6.4281** | 1.3884 | [6.2386, 6.6436] | 155.6 | 4.55e-07 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.2750 | 0.6270 | 0.0417 (prep) | 6.7497 | **6.7915** | 1.3800 | [6.6078, 7.0268] | 147.2 | 2.35e-02 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **7.1130 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2124 ms** vs llamada formal E2E: **0.1917 ms** (gap de dispatch/asignaciones Python = **+0.0207 ms**, +9.8%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0597 | 28.1% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.0499 | 23.5% | 2.1 | `O(g^2*K) [16x3]` |
| `asymmetry_g8` | 8x8 | 0.0486 | 22.9% | 2.6 | `O(g^2*K) [64x3]` |
| `c_fine_hist` | 8x8 | 0.0272 | 12.8% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0105 | 4.9% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 4. Desglose Medido por Capas CNN en CPU (`threads=1, batch=1`)

| Arquitectura | Params | Act Peak (KB) | FLOPs (PyTorch) | Hookless (ms) | Hook overhead (%) | Cold 1st (ms) | Familias dominantes (% hooks) |
|---|---|---|---|---|---|---|---|
| `ultra_tiny` | 1,154 | 128.0 | 1,335,424 | 0.8498 | +30.8% | 1.162 | dwconv:39.5%, bn:21.7%, conv:15.4% |
| `ds_cnn` | 12,866 | 1024.0 | 19,366,400 | 2.8737 | -8.1% | 3.262 | dwconv:38.0%, bn:21.2%, pwconv:15.1% |
| `tiny_cnn` | 23,538 | 1024.0 | 80,216,320 | 5.0022 | +7.9% | 7.537 | pool:60.1%, conv:30.5%, bn:5.8% |
| `mobilenet_like` | 70,786 | 512.0 | 38,716,416 | 3.1804 | +16.8% | 6.282 | dwconv:41.1%, pwconv:22.2%, bn:20.3% |
| `tiny_resnet` | 77,234 | 256.0 | 97,648,896 | 3.8229 | -5.2% | 3.733 | conv:70.2%, bn:12.0%, pwconv:11.5% |
| `small_cnn` | 97,650 | 1024.0 | 117,965,312 | 6.4947 | +0.2% | 8.277 | pool:58.8%, conv:33.1%, bn:5.1% |
| `torchvision_mobilenet` | 1,519,618 | 288.0 | 35,293,184 | 6.8912 | +35.6% | 7.122 | dwconv:31.5%, pwconv:27.4%, bn:20.5% |

## 5. Pruebas de Significancia Estadistica (Mann-Whitney U, 1 hilo, batch=1)

| Comparacion | Mediana A (ms) | Mediana B (ms) | Ratio A/B | Estadistico U | p-valor | Significativo |
|---|---|---|---|---|---|---|
| `dt (x94)` vs `ds_cnn` | 0.3326 | 1.9286 | 0.172x | 0.0 | `4.828e-67` | True |
| `dt (x94)` vs `ultra_tiny` | 0.3326 | 0.8077 | 0.412x | 0.0 | `4.825e-67` | True |
| `lr (x94)` vs `ds_cnn` | 0.4904 | 1.9286 | 0.254x | 0.0 | `4.829e-67` | True |
| `lr (x94)` vs `ultra_tiny` | 0.4904 | 0.8077 | 0.607x | 460.0 | `4.450e-64` | True |
| `lr_dot (x94)` vs `ds_cnn` | 0.2565 | 1.9286 | 0.133x | 0.0 | `4.828e-67` | True |
| `lr_dot (x94)` vs `ultra_tiny` | 0.2565 | 0.8077 | 0.318x | 0.0 | `4.825e-67` | True |
| `mlp_16_dot (x94)` vs `ds_cnn` | 0.2613 | 1.9286 | 0.135x | 0.0 | `4.827e-67` | True |
| `mlp_16_dot (x94)` vs `ultra_tiny` | 0.2613 | 0.8077 | 0.324x | 0.0 | `4.824e-67` | True |
| `rf (x94)` vs `ds_cnn` | 4.4106 | 1.9286 | 2.287x | 40000.0 | `4.830e-67` | True |
| `rf (x94)` vs `ultra_tiny` | 4.4106 | 0.8077 | 5.461x | 40000.0 | `4.827e-67` | True |

## 6. Ablacion Calidad vs Coste (`x85` vs `x9` vs `x94`)

| Clasificador | Subconjunto | Features | F1 | ROC-AUC | T_total (ms) | dF1 vs x85 | d_ms vs x85 | dF1 / d_ms |
|---|---|---|---|---|---|---|---|---|
| `dt` | `x85_only` | 85 | 0.0714 | 0.5313 | 0.0955 | +0.0000 | +0.0000 | +0.0000 |
| `dt` | `x9_only` | 9 | 0.2909 | 0.7069 | 0.2565 | +0.2195 | +0.1609 | +1.3641 |
| `dt` | `x94_full` | 94 | 0.2979 | 0.5893 | 0.2560 | +0.2264 | +0.1604 | +1.4117 |
| `knn` | `x85_only` | 85 | 0.0000 | 0.5768 | 0.4836 | +0.0000 | +0.0000 | +0.0000 |
| `knn` | `x9_only` | 9 | 0.2667 | 0.6755 | 0.7668 | +0.2667 | +0.2832 | +0.9418 |
| `knn` | `x94_full` | 94 | 0.0000 | 0.5180 | 0.6590 | +0.0000 | +0.1754 | +0.0000 |
| `lr` | `x85_only` | 85 | 0.3333 | 0.5721 | 0.2228 | +0.0000 | +0.0000 | +0.0000 |
| `lr` | `x9_only` | 9 | 0.2703 | 0.6223 | 0.3812 | -0.0631 | +0.1584 | -0.3981 |
| `lr` | `x94_full` | 94 | 0.3704 | 0.5893 | 0.3852 | +0.0370 | +0.1624 | +0.2281 |
| `lr_dot` | `x85_only` | 85 | 0.3333 | 0.5721 | 0.0278 | +0.0000 | +0.0000 | +0.0000 |
| `lr_dot` | `x9_only` | 9 | 0.2703 | 0.6223 | 0.1842 | -0.0631 | +0.1563 | -0.4033 |
| `lr_dot` | `x94_full` | 94 | 0.3704 | 0.5893 | 0.1888 | +0.0370 | +0.1609 | +0.2302 |
| `mlp_16_dot` | `x85_only` | 85 | 0.4167 | 0.7085 | 0.0336 | +0.0000 | +0.0000 | +0.0000 |
| `mlp_16_dot` | `x9_only` | 9 | 0.3721 | 0.6740 | 0.1888 | -0.0446 | +0.1552 | -0.2873 |
| `mlp_16_dot` | `x94_full` | 94 | 0.3571 | 0.7335 | 0.1963 | -0.0595 | +0.1627 | -0.3660 |
| `rf` | `x85_only` | 85 | 0.0000 | 0.5486 | 4.2155 | +0.0000 | +0.0000 | +0.0000 |
| `rf` | `x9_only` | 9 | 0.3529 | 0.6638 | 4.3061 | +0.3529 | +0.0907 | +3.8934 |
| `rf` | `x94_full` | 94 | 0.1538 | 0.6105 | 4.2568 | +0.1538 | +0.0413 | +3.7206 |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.4360 | 696.4 |
| `ds_cnn` | cuda | 1 | 1.9119 | 523.0 |
| `tiny_cnn` | cuda | 1 | 0.8561 | 1168.1 |
| `mobilenet_like` | cuda | 1 | 2.2321 | 448.0 |
| `tiny_resnet` | cuda | 1 | 1.7091 | 585.1 |
| `small_cnn` | cuda | 1 | 1.2256 | 815.9 |
| `torchvision_mobilenet` | cuda | 1 | 11.9370 | 83.8 |
| `ultra_tiny` | cuda | 32 | 0.0428 | 23382.4 |
| `ds_cnn` | cuda | 32 | 0.2092 | 4779.3 |
| `tiny_cnn` | cuda | 32 | 0.2058 | 4860.0 |
| `mobilenet_like` | cuda | 32 | 0.2299 | 4350.3 |
| `tiny_resnet` | cuda | 32 | 0.1774 | 5635.5 |
| `small_cnn` | cuda | 32 | 0.2300 | 4347.6 |
| `torchvision_mobilenet` | cuda | 32 | 0.2808 | 3561.2 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **28.1%** (0.0597 ms) del tiempo por etapas (E2E=0.1917 ms, gap de despacho Python/ctypes=+0.0207 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.2565 ms**, F1=0.3704), mientras el maximo F1 lo obtiene `small_cnn` (**F1=0.4545**, 6.4281 ms).
