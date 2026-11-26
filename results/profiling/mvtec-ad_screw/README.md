# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad_screw`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `99bc026`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.6452 | 0.8721 | 0.3448 (desc) | 0.0099 | **0.3554** | 0.1567 | [0.3536, 0.3580] | 2813.7 | - |
| `mlp_16_dot (x94)` | ml | cpu | 1537 / 94 | 0.6452 | 0.8513 | 0.3448 (desc) | 0.0118 | **0.3565** | 0.1481 | [0.3558, 0.3577] | 2805.0 | - |
| `dt (x94)` | ml | cpu | 15 / 94 | 0.5556 | 0.7969 | 0.3448 (desc) | 0.1336 | **0.4777** | 0.1595 | [0.4760, 0.4786] | 2093.4 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.6452 | 0.8721 | 0.3448 (desc) | 0.2021 | **0.5473** | 0.1411 | [0.5456, 0.5480] | 1827.3 | - |
| `knn (x94)` | ml | cpu | 36096 / 94 | 0.5128 | 0.8247 | 0.3448 (desc) | 0.5053 | **0.8276** | 0.1760 | [0.8243, 0.8402] | 1208.3 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.6349 | 0.8050 | 0.0433 (prep) | 0.7903 | **0.8469** | 0.2293 | [0.8278, 0.8553] | 1180.7 | 3.65e-07 |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.6667 | 0.9196 | 0.0433 (prep) | 1.9607 | **2.0119** | 0.5938 | [1.9884, 2.0701] | 497.0 | 3.33e-07 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.7200 | 0.9178 | 0.0433 (prep) | 3.5565 | **3.6010** | 0.9883 | [3.4924, 3.8568] | 277.7 | 4.44e-07 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.7000 | 0.8843 | 0.0433 (prep) | 3.7805 | **3.8282** | 1.0810 | [3.6695, 3.9163] | 261.2 | 2.59e-07 |
| `rf (x94)` | ml | cpu | 7678 / 94 | 0.6829 | 0.8594 | 0.3448 (desc) | 4.1969 | **4.4950** | 0.7666 | [4.4203, 4.5761] | 222.5 | - |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.5714 | 0.8605 | 0.0433 (prep) | 5.0019 | **5.0467** | 1.3928 | [4.8570, 5.3056] | 198.1 | 4.71e-07 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.6341 | 0.8785 | 0.0433 (prep) | 6.3353 | **6.3783** | 1.3664 | [6.2250, 6.5070] | 156.8 | 4.44e-07 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.4000 | 0.6230 | 0.0433 (prep) | 7.5671 | **7.6211** | 1.7107 | [7.4496, 7.7805] | 131.2 | 4.66e-02 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **8.6860 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2202 ms** vs llamada formal E2E: **0.1882 ms** (gap de dispatch/asignaciones Python = **+0.0320 ms**, +14.5%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0610 | 27.7% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.0506 | 23.0% | 2.1 | `O(g^2*K) [16x3]` |
| `asymmetry_g8` | 8x8 | 0.0495 | 22.5% | 2.6 | `O(g^2*K) [64x3]` |
| `c_fine_hist` | 8x8 | 0.0309 | 14.1% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0111 | 5.0% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 4. Desglose Medido por Capas CNN en CPU (`threads=1, batch=1`)

| Arquitectura | Params | Act Peak (KB) | FLOPs (PyTorch) | Hookless (ms) | Hook overhead (%) | Cold 1st (ms) | Familias dominantes (% hooks) |
|---|---|---|---|---|---|---|---|
| `ultra_tiny` | 1,154 | 128.0 | 1,335,424 | 0.8619 | +36.7% | 1.407 | dwconv:41.4%, bn:21.7%, conv:14.0% |
| `ds_cnn` | 12,866 | 1024.0 | 19,366,400 | 2.1521 | +8.3% | 3.697 | dwconv:37.3%, bn:20.9%, pwconv:15.7% |
| `tiny_cnn` | 23,538 | 1024.0 | 80,216,320 | 4.5599 | +1.9% | 10.211 | pool:59.0%, conv:31.9%, bn:5.5% |
| `mobilenet_like` | 70,786 | 512.0 | 38,716,416 | 3.0656 | +28.9% | 4.683 | dwconv:41.3%, pwconv:22.0%, bn:20.4% |
| `tiny_resnet` | 77,234 | 256.0 | 97,648,896 | 3.4059 | +8.3% | 3.667 | conv:70.2%, bn:12.1%, pwconv:11.4% |
| `small_cnn` | 97,650 | 1024.0 | 117,965,312 | 6.1082 | -0.3% | 6.435 | pool:56.1%, conv:35.6%, bn:5.3% |
| `torchvision_mobilenet` | 1,519,618 | 288.0 | 35,293,184 | 6.9655 | +23.0% | 10.936 | dwconv:31.5%, pwconv:27.5%, bn:20.5% |

## 5. Pruebas de Significancia Estadistica (Mann-Whitney U, 1 hilo, batch=1)

| Comparacion | Mediana A (ms) | Mediana B (ms) | Ratio A/B | Estadistico U | p-valor | Significativo |
|---|---|---|---|---|---|---|
| `dt (x94)` vs `ds_cnn` | 0.4777 | 2.0119 | 0.237x | 0.0 | `4.827e-67` | True |
| `dt (x94)` vs `ultra_tiny` | 0.4777 | 0.8469 | 0.564x | 0.0 | `4.827e-67` | True |
| `lr (x94)` vs `ds_cnn` | 0.5473 | 2.0119 | 0.272x | 0.0 | `4.826e-67` | True |
| `lr (x94)` vs `ultra_tiny` | 0.5473 | 0.8469 | 0.646x | 0.0 | `4.826e-67` | True |
| `lr_dot (x94)` vs `ds_cnn` | 0.3554 | 2.0119 | 0.177x | 0.0 | `4.829e-67` | True |
| `lr_dot (x94)` vs `ultra_tiny` | 0.3554 | 0.8469 | 0.420x | 0.0 | `4.828e-67` | True |
| `mlp_16_dot (x94)` vs `ds_cnn` | 0.3565 | 2.0119 | 0.177x | 0.0 | `4.825e-67` | True |
| `mlp_16_dot (x94)` vs `ultra_tiny` | 0.3565 | 0.8469 | 0.421x | 0.0 | `4.825e-67` | True |
| `rf (x94)` vs `ds_cnn` | 4.4950 | 2.0119 | 2.234x | 39758.0 | `1.787e-65` | True |
| `rf (x94)` vs `ultra_tiny` | 4.4950 | 0.8469 | 5.307x | 40000.0 | `4.830e-67` | True |

## 6. Ablacion Calidad vs Coste (`x85` vs `x9` vs `x94`)

| Clasificador | Subconjunto | Features | F1 | ROC-AUC | T_total (ms) | dF1 vs x85 | d_ms vs x85 | dF1 / d_ms |
|---|---|---|---|---|---|---|---|---|
| `dt` | `x85_only` | 85 | 0.5556 | 0.7969 | 0.1033 | +0.0000 | +0.0000 | +0.0000 |
| `dt` | `x9_only` | 9 | 0.5000 | 0.7104 | 0.2793 | -0.0556 | +0.1759 | -0.3157 |
| `dt` | `x94_full` | 94 | 0.5556 | 0.7969 | 0.2717 | +0.0000 | +0.1683 | +0.0000 |
| `knn` | `x85_only` | 85 | 0.5128 | 0.8038 | 0.5067 | +0.0000 | +0.0000 | +0.0000 |
| `knn` | `x9_only` | 9 | 0.2857 | 0.6956 | 0.7719 | -0.2271 | +0.2652 | -0.8564 |
| `knn` | `x94_full` | 94 | 0.5128 | 0.8247 | 0.6806 | +0.0000 | +0.1739 | +0.0000 |
| `lr` | `x85_only` | 85 | 0.6667 | 0.8559 | 0.2356 | +0.0000 | +0.0000 | +0.0000 |
| `lr` | `x9_only` | 9 | 0.5152 | 0.7147 | 0.4019 | -0.1515 | +0.1663 | -0.9111 |
| `lr` | `x94_full` | 94 | 0.6452 | 0.8721 | 0.4074 | -0.0215 | +0.1718 | -0.1252 |
| `lr_dot` | `x85_only` | 85 | 0.6667 | 0.8559 | 0.0312 | +0.0000 | +0.0000 | +0.0000 |
| `lr_dot` | `x9_only` | 9 | 0.5152 | 0.7147 | 0.1984 | -0.1515 | +0.1671 | -0.9065 |
| `lr_dot` | `x94_full` | 94 | 0.6452 | 0.8721 | 0.2026 | -0.0215 | +0.1714 | -0.1255 |
| `mlp_16_dot` | `x85_only` | 85 | 0.6102 | 0.8380 | 0.0362 | +0.0000 | +0.0000 | +0.0000 |
| `mlp_16_dot` | `x9_only` | 9 | 0.4762 | 0.7193 | 0.2029 | -0.1340 | +0.1668 | -0.8035 |
| `mlp_16_dot` | `x94_full` | 94 | 0.6452 | 0.8513 | 0.2075 | +0.0350 | +0.1714 | +0.2042 |
| `rf` | `x85_only` | 85 | 0.6316 | 0.8623 | 4.2544 | +0.0000 | +0.0000 | +0.0000 |
| `rf` | `x9_only` | 9 | 0.5000 | 0.7454 | 4.4367 | -0.1316 | +0.1823 | -0.7218 |
| `rf` | `x94_full` | 94 | 0.6829 | 0.8594 | 4.1798 | +0.0513 | -0.0746 | -0.6878 |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.2380 | 807.8 |
| `ds_cnn` | cuda | 1 | 1.2598 | 793.8 |
| `tiny_cnn` | cuda | 1 | 1.1942 | 837.4 |
| `mobilenet_like` | cuda | 1 | 2.4240 | 412.5 |
| `tiny_resnet` | cuda | 1 | 1.6764 | 596.5 |
| `small_cnn` | cuda | 1 | 1.2003 | 833.2 |
| `torchvision_mobilenet` | cuda | 1 | 9.7452 | 102.6 |
| `ultra_tiny` | cuda | 32 | 0.0436 | 22934.1 |
| `ds_cnn` | cuda | 32 | 0.2104 | 4753.7 |
| `tiny_cnn` | cuda | 32 | 0.2037 | 4908.2 |
| `mobilenet_like` | cuda | 32 | 0.2304 | 4341.0 |
| `tiny_resnet` | cuda | 32 | 0.1787 | 5597.5 |
| `small_cnn` | cuda | 32 | 0.2326 | 4298.4 |
| `torchvision_mobilenet` | cuda | 32 | 0.2777 | 3601.6 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **27.7%** (0.0610 ms) del tiempo por etapas (E2E=0.1882 ms, gap de despacho Python/ctypes=+0.0320 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.3554 ms**, F1=0.6452), mientras el maximo F1 lo obtiene `mobilenet_like` (**F1=0.7200**, 3.6010 ms).
