# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad_hazelnut`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `99bc026`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.7429 | 0.9573 | 0.2492 (desc) | 0.0069 | **0.2561** | 0.0088 | [0.2522, 0.2576] | 3904.0 | - |
| `mlp_16_dot (x94)` | ml | cpu | 1537 / 94 | 0.7500 | 0.9557 | 0.2492 (desc) | 0.0126 | **0.2625** | 0.0124 | [0.2602, 0.2631] | 3809.5 | - |
| `dt (x94)` | ml | cpu | 15 / 94 | 0.5714 | 0.8194 | 0.2492 (desc) | 0.0958 | **0.3418** | 0.0360 | [0.3403, 0.3452] | 2925.3 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.7429 | 0.9573 | 0.2492 (desc) | 0.1983 | **0.4505** | 0.0738 | [0.4492, 0.4606] | 2219.8 | - |
| `knn (x94)` | ml | cpu | 37600 / 94 | 0.3333 | 0.8654 | 0.2492 (desc) | 0.4894 | **0.7439** | 0.1835 | [0.7341, 0.7603] | 1344.3 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.4444 | 0.9376 | 0.0448 (prep) | 0.7543 | **0.8087** | 0.1847 | [0.8006, 0.8189] | 1236.6 | 4.44e-07 |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.8000 | 0.9606 | 0.0448 (prep) | 1.9464 | **1.9924** | 0.8044 | [1.9322, 2.0474] | 501.9 | 2.69e-07 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.9231 | 1.0000 | 0.0448 (prep) | 3.2902 | **3.3350** | 0.9873 | [3.2547, 3.6064] | 299.9 | 4.29e-07 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.2435 | 0.4663 | 0.0448 (prep) | 3.7032 | **3.7541** | 0.3855 | [3.7294, 3.8126] | 266.4 | 8.42e-03 |
| `rf (x94)` | ml | cpu | 5452 / 94 | 0.6000 | 0.9278 | 0.2492 (desc) | 4.0786 | **4.3781** | 0.6788 | [4.2848, 4.4682] | 228.4 | - |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.7826 | 0.9910 | 0.0448 (prep) | 4.8249 | **4.8770** | 1.2554 | [4.7626, 5.0731] | 205.0 | 4.29e-07 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.8800 | 0.9910 | 0.0448 (prep) | 5.5838 | **5.6290** | 1.0060 | [5.5505, 5.6891] | 177.7 | 4.90e-07 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.2435 | 0.5000 | 0.0448 (prep) | 6.9991 | **7.0521** | 1.3483 | [6.7961, 7.1718] | 141.8 | 1.27e-02 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **8.4559 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2259 ms** vs llamada formal E2E: **0.1883 ms** (gap de dispatch/asignaciones Python = **+0.0376 ms**, +16.6%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0627 | 27.8% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.0523 | 23.2% | 2.1 | `O(g^2*K) [16x3]` |
| `asymmetry_g8` | 8x8 | 0.0510 | 22.6% | 2.6 | `O(g^2*K) [64x3]` |
| `c_fine_hist` | 8x8 | 0.0317 | 14.0% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0112 | 5.0% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 4. Desglose Medido por Capas CNN en CPU (`threads=1, batch=1`)

| Arquitectura | Params | Act Peak (KB) | FLOPs (PyTorch) | Hookless (ms) | Hook overhead (%) | Cold 1st (ms) | Familias dominantes (% hooks) |
|---|---|---|---|---|---|---|---|
| `ultra_tiny` | 1,154 | 128.0 | 1,335,424 | 0.9049 | +19.4% | 1.079 | dwconv:40.1%, bn:21.9%, conv:14.8% |
| `ds_cnn` | 12,866 | 1024.0 | 19,366,400 | 2.1000 | +7.1% | 5.589 | dwconv:37.6%, bn:21.4%, pwconv:15.9% |
| `tiny_cnn` | 23,538 | 1024.0 | 80,216,320 | 4.6259 | +14.7% | 7.842 | pool:59.8%, conv:32.2%, bn:4.8% |
| `mobilenet_like` | 70,786 | 512.0 | 38,716,416 | 3.2435 | +19.5% | 5.023 | dwconv:40.9%, pwconv:22.1%, bn:20.4% |
| `tiny_resnet` | 77,234 | 256.0 | 97,648,896 | 3.6307 | +1.5% | 3.691 | conv:70.1%, bn:12.1%, pwconv:11.4% |
| `small_cnn` | 97,650 | 1024.0 | 117,965,312 | 6.1536 | +1.8% | 6.324 | pool:55.5%, conv:36.9%, bn:4.8% |
| `torchvision_mobilenet` | 1,519,618 | 288.0 | 35,293,184 | 8.6119 | +34.4% | 7.080 | dwconv:32.0%, pwconv:27.1%, bn:20.7% |

## 5. Pruebas de Significancia Estadistica (Mann-Whitney U, 1 hilo, batch=1)

| Comparacion | Mediana A (ms) | Mediana B (ms) | Ratio A/B | Estadistico U | p-valor | Significativo |
|---|---|---|---|---|---|---|
| `dt (x94)` vs `ds_cnn` | 0.3418 | 1.9924 | 0.172x | 0.0 | `4.829e-67` | True |
| `dt (x94)` vs `ultra_tiny` | 0.3418 | 0.8087 | 0.423x | 0.0 | `4.827e-67` | True |
| `lr (x94)` vs `ds_cnn` | 0.4505 | 1.9924 | 0.226x | 0.0 | `4.829e-67` | True |
| `lr (x94)` vs `ultra_tiny` | 0.4505 | 0.8087 | 0.557x | 0.0 | `4.827e-67` | True |
| `lr_dot (x94)` vs `ds_cnn` | 0.2561 | 1.9924 | 0.129x | 0.0 | `4.822e-67` | True |
| `lr_dot (x94)` vs `ultra_tiny` | 0.2561 | 0.8087 | 0.317x | 0.0 | `4.820e-67` | True |
| `mlp_16_dot (x94)` vs `ds_cnn` | 0.2625 | 1.9924 | 0.132x | 0.0 | `4.823e-67` | True |
| `mlp_16_dot (x94)` vs `ultra_tiny` | 0.2625 | 0.8087 | 0.325x | 0.0 | `4.821e-67` | True |
| `rf (x94)` vs `ds_cnn` | 4.3781 | 1.9924 | 2.197x | 39858.0 | `4.041e-66` | True |
| `rf (x94)` vs `ultra_tiny` | 4.3781 | 0.8087 | 5.414x | 39902.0 | `2.095e-66` | True |

## 6. Ablacion Calidad vs Coste (`x85` vs `x9` vs `x94`)

| Clasificador | Subconjunto | Features | F1 | ROC-AUC | T_total (ms) | dF1 vs x85 | d_ms vs x85 | dF1 / d_ms |
|---|---|---|---|---|---|---|---|---|
| `dt` | `x85_only` | 85 | 0.5714 | 0.8194 | 0.1040 | +0.0000 | +0.0000 | +0.0000 |
| `dt` | `x9_only` | 9 | 0.3636 | 0.6802 | 0.2575 | -0.2078 | +0.1534 | -1.3541 |
| `dt` | `x94_full` | 94 | 0.5714 | 0.8194 | 0.2737 | +0.0000 | +0.1697 | +0.0000 |
| `knn` | `x85_only` | 85 | 0.3333 | 0.8604 | 0.5016 | +0.0000 | +0.0000 | +0.0000 |
| `knn` | `x9_only` | 9 | 0.1053 | 0.5862 | 0.7648 | -0.2281 | +0.2632 | -0.8665 |
| `knn` | `x94_full` | 94 | 0.3333 | 0.8654 | 0.6801 | +0.0000 | +0.1785 | +0.0000 |
| `lr` | `x85_only` | 85 | 0.5500 | 0.9204 | 0.2585 | +0.0000 | +0.0000 | +0.0000 |
| `lr` | `x9_only` | 9 | 0.3333 | 0.6757 | 0.4057 | -0.2167 | +0.1473 | -1.4709 |
| `lr` | `x94_full` | 94 | 0.7429 | 0.9573 | 0.4144 | +0.1929 | +0.1560 | +1.2363 |
| `lr_dot` | `x85_only` | 85 | 0.5500 | 0.9204 | 0.0313 | +0.0000 | +0.0000 | +0.0000 |
| `lr_dot` | `x9_only` | 9 | 0.3333 | 0.6757 | 0.1886 | -0.2167 | +0.1573 | -1.3770 |
| `lr_dot` | `x94_full` | 94 | 0.7429 | 0.9573 | 0.2049 | +0.1929 | +0.1736 | +1.1109 |
| `mlp_16_dot` | `x85_only` | 85 | 0.6111 | 0.9146 | 0.0361 | +0.0000 | +0.0000 | +0.0000 |
| `mlp_16_dot` | `x9_only` | 9 | 0.3529 | 0.6987 | 0.1939 | -0.2582 | +0.1578 | -1.6355 |
| `mlp_16_dot` | `x94_full` | 94 | 0.7500 | 0.9557 | 0.2094 | +0.1389 | +0.1733 | +0.8014 |
| `rf` | `x85_only` | 85 | 0.3529 | 0.9113 | 4.0865 | +0.0000 | +0.0000 | +0.0000 |
| `rf` | `x9_only` | 9 | 0.0000 | 0.6437 | 4.4451 | -0.3529 | +0.3585 | -0.9845 |
| `rf` | `x94_full` | 94 | 0.6000 | 0.9278 | 4.2149 | +0.2471 | +0.1283 | +1.9256 |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.0629 | 940.9 |
| `ds_cnn` | cuda | 1 | 1.2604 | 793.4 |
| `tiny_cnn` | cuda | 1 | 0.7330 | 1364.2 |
| `mobilenet_like` | cuda | 1 | 2.2019 | 454.2 |
| `tiny_resnet` | cuda | 1 | 1.9089 | 523.9 |
| `small_cnn` | cuda | 1 | 1.1965 | 835.8 |
| `torchvision_mobilenet` | cuda | 1 | 9.0646 | 110.3 |
| `ultra_tiny` | cuda | 32 | 0.0458 | 21825.1 |
| `ds_cnn` | cuda | 32 | 0.2092 | 4779.5 |
| `tiny_cnn` | cuda | 32 | 0.2051 | 4874.8 |
| `mobilenet_like` | cuda | 32 | 0.2295 | 4356.4 |
| `tiny_resnet` | cuda | 32 | 0.1793 | 5576.5 |
| `small_cnn` | cuda | 32 | 0.2327 | 4297.8 |
| `torchvision_mobilenet` | cuda | 32 | 0.2785 | 3590.8 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **27.8%** (0.0627 ms) del tiempo por etapas (E2E=0.1883 ms, gap de despacho Python/ctypes=+0.0376 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.2561 ms**, F1=0.7429), mientras el maximo F1 lo obtiene `tiny_resnet` (**F1=0.9231**, 3.3350 ms).
