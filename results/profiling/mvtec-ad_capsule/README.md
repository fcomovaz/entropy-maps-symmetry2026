# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad_capsule`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `99bc026`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.7600 | 0.9091 | 0.2502 (desc) | 0.0067 | **0.2570** | 0.0384 | [0.2565, 0.2575] | 3891.1 | - |
| `mlp_16_dot (x94)` | ml | cpu | 1537 / 94 | 0.8696 | 0.9397 | 0.2502 (desc) | 0.0119 | **0.2624** | 0.0383 | [0.2620, 0.2630] | 3811.0 | - |
| `dt (x94)` | ml | cpu | 15 / 94 | 0.6667 | 0.7175 | 0.2502 (desc) | 0.0977 | **0.3494** | 0.0781 | [0.3480, 0.3623] | 2861.6 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.7600 | 0.9091 | 0.2502 (desc) | 0.2063 | **0.4641** | 0.0996 | [0.4627, 0.4672] | 2154.5 | - |
| `knn (x94)` | ml | cpu | 26320 / 94 | 0.6286 | 0.8711 | 0.2502 (desc) | 0.4776 | **0.8276** | 0.1926 | [0.7631, 0.8686] | 1208.3 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.4731 | 0.6568 | 0.0422 (prep) | 0.7930 | **0.8362** | 0.2171 | [0.8230, 0.8395] | 1195.9 | 2.80e-07 |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.6809 | 0.8553 | 0.0422 (prep) | 1.8752 | **1.9206** | 0.7925 | [1.9046, 1.9510] | 520.7 | 4.30e-07 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.4731 | 0.4833 | 0.0422 (prep) | 2.8491 | **2.8909** | 0.8053 | [2.8331, 3.0296] | 345.9 | 4.67e-07 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.4828 | 0.7922 | 0.0422 (prep) | 3.1739 | **3.2248** | 0.6141 | [3.1810, 3.2563] | 310.1 | 4.82e-07 |
| `rf (x94)` | ml | cpu | 5906 / 94 | 0.7027 | 0.8989 | 0.2502 (desc) | 4.1294 | **4.3903** | 0.6531 | [4.3568, 4.4643] | 227.8 | - |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.5176 | 0.8163 | 0.0422 (prep) | 4.7137 | **4.7549** | 1.2317 | [4.5469, 4.9133] | 210.3 | 4.84e-07 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.4731 | 0.5000 | 0.0422 (prep) | 6.5559 | **6.6101** | 1.2605 | [6.4935, 6.7999] | 151.3 | 2.80e-07 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.7556 | 0.9295 | 0.0422 (prep) | 7.4489 | **7.4924** | 0.4891 | [7.4026, 7.5579] | 133.5 | 4.44e-07 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **7.1538 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2248 ms** vs llamada formal E2E: **0.1878 ms** (gap de dispatch/asignaciones Python = **+0.0371 ms**, +16.5%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0623 | 27.7% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.0520 | 23.1% | 2.1 | `O(g^2*K) [16x3]` |
| `asymmetry_g8` | 8x8 | 0.0508 | 22.6% | 2.6 | `O(g^2*K) [64x3]` |
| `c_fine_hist` | 8x8 | 0.0316 | 14.1% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0111 | 4.9% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 4. Desglose Medido por Capas CNN en CPU (`threads=1, batch=1`)

| Arquitectura | Params | Act Peak (KB) | FLOPs (PyTorch) | Hookless (ms) | Hook overhead (%) | Cold 1st (ms) | Familias dominantes (% hooks) |
|---|---|---|---|---|---|---|---|
| `ultra_tiny` | 1,154 | 128.0 | 1,335,424 | 0.9833 | +8.9% | 1.068 | dwconv:39.6%, bn:21.8%, conv:15.2% |
| `ds_cnn` | 12,866 | 1024.0 | 19,366,400 | 2.5480 | -10.0% | 5.175 | dwconv:36.6%, bn:21.1%, pwconv:15.8% |
| `tiny_cnn` | 23,538 | 1024.0 | 80,216,320 | 4.4693 | +8.8% | 10.011 | pool:57.7%, conv:32.8%, bn:5.7% |
| `mobilenet_like` | 70,786 | 512.0 | 38,716,416 | 3.1292 | +18.4% | 3.390 | dwconv:40.9%, pwconv:22.4%, bn:20.3% |
| `tiny_resnet` | 77,234 | 256.0 | 97,648,896 | 3.7329 | +14.0% | 5.496 | conv:70.8%, bn:11.8%, pwconv:11.0% |
| `small_cnn` | 97,650 | 1024.0 | 117,965,312 | 6.2619 | +1.2% | 8.132 | pool:55.2%, conv:36.1%, bn:5.3% |
| `torchvision_mobilenet` | 1,519,618 | 288.0 | 35,293,184 | 6.8982 | +34.5% | 6.846 | dwconv:31.2%, pwconv:27.6%, bn:20.5% |

## 5. Pruebas de Significancia Estadistica (Mann-Whitney U, 1 hilo, batch=1)

| Comparacion | Mediana A (ms) | Mediana B (ms) | Ratio A/B | Estadistico U | p-valor | Significativo |
|---|---|---|---|---|---|---|
| `dt (x94)` vs `ds_cnn` | 0.3494 | 1.9206 | 0.182x | 0.0 | `4.826e-67` | True |
| `dt (x94)` vs `ultra_tiny` | 0.3494 | 0.8362 | 0.418x | 0.0 | `4.824e-67` | True |
| `lr (x94)` vs `ds_cnn` | 0.4641 | 1.9206 | 0.242x | 0.0 | `4.827e-67` | True |
| `lr (x94)` vs `ultra_tiny` | 0.4641 | 0.8362 | 0.555x | 0.0 | `4.826e-67` | True |
| `lr_dot (x94)` vs `ds_cnn` | 0.2570 | 1.9206 | 0.134x | 0.0 | `4.809e-67` | True |
| `lr_dot (x94)` vs `ultra_tiny` | 0.2570 | 0.8362 | 0.307x | 0.0 | `4.808e-67` | True |
| `mlp_16_dot (x94)` vs `ds_cnn` | 0.2624 | 1.9206 | 0.137x | 0.0 | `4.814e-67` | True |
| `mlp_16_dot (x94)` vs `ultra_tiny` | 0.2624 | 0.8362 | 0.314x | 0.0 | `4.812e-67` | True |
| `rf (x94)` vs `ds_cnn` | 4.3903 | 1.9206 | 2.286x | 39857.0 | `4.102e-66` | True |
| `rf (x94)` vs `ultra_tiny` | 4.3903 | 0.8362 | 5.250x | 40000.0 | `4.829e-67` | True |

## 6. Ablacion Calidad vs Coste (`x85` vs `x9` vs `x94`)

| Clasificador | Subconjunto | Features | F1 | ROC-AUC | T_total (ms) | dF1 vs x85 | d_ms vs x85 | dF1 / d_ms |
|---|---|---|---|---|---|---|---|---|
| `dt` | `x85_only` | 85 | 0.6800 | 0.7440 | 0.1011 | +0.0000 | +0.0000 | +0.0000 |
| `dt` | `x9_only` | 9 | 0.5238 | 0.7435 | 0.2716 | -0.1562 | +0.1705 | -0.9161 |
| `dt` | `x94_full` | 94 | 0.6667 | 0.7175 | 0.2642 | -0.0133 | +0.1631 | -0.0817 |
| `knn` | `x85_only` | 85 | 0.5556 | 0.8609 | 0.4964 | +0.0000 | +0.0000 | +0.0000 |
| `knn` | `x9_only` | 9 | 0.5946 | 0.8071 | 0.7729 | +0.0390 | +0.2766 | +0.1411 |
| `knn` | `x94_full` | 94 | 0.6286 | 0.8711 | 0.6829 | +0.0730 | +0.1866 | +0.3913 |
| `lr` | `x85_only` | 85 | 0.7234 | 0.9017 | 0.2271 | +0.0000 | +0.0000 | +0.0000 |
| `lr` | `x9_only` | 9 | 0.5000 | 0.7365 | 0.3824 | -0.2234 | +0.1553 | -1.4385 |
| `lr` | `x94_full` | 94 | 0.7600 | 0.9091 | 0.4172 | +0.0366 | +0.1901 | +0.1925 |
| `lr_dot` | `x85_only` | 85 | 0.7234 | 0.9017 | 0.0315 | +0.0000 | +0.0000 | +0.0000 |
| `lr_dot` | `x9_only` | 9 | 0.5000 | 0.7365 | 0.1892 | -0.2234 | +0.1577 | -1.4166 |
| `lr_dot` | `x94_full` | 94 | 0.7600 | 0.9091 | 0.1945 | +0.0366 | +0.1630 | +0.2245 |
| `mlp_16_dot` | `x85_only` | 85 | 0.8511 | 0.9378 | 0.0368 | +0.0000 | +0.0000 | +0.0000 |
| `mlp_16_dot` | `x9_only` | 9 | 0.6939 | 0.8302 | 0.1941 | -0.1572 | +0.1573 | -0.9993 |
| `mlp_16_dot` | `x94_full` | 94 | 0.8696 | 0.9397 | 0.1991 | +0.0185 | +0.1623 | +0.1140 |
| `rf` | `x85_only` | 85 | 0.6857 | 0.9109 | 4.3520 | +0.0000 | +0.0000 | +0.0000 |
| `rf` | `x9_only` | 9 | 0.5714 | 0.7913 | 4.5129 | -0.1143 | +0.1609 | -0.7105 |
| `rf` | `x94_full` | 94 | 0.7027 | 0.8989 | 4.3266 | +0.0170 | -0.0254 | -0.6688 |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.0568 | 946.3 |
| `ds_cnn` | cuda | 1 | 1.4825 | 674.5 |
| `tiny_cnn` | cuda | 1 | 1.1582 | 863.4 |
| `mobilenet_like` | cuda | 1 | 3.3931 | 294.7 |
| `tiny_resnet` | cuda | 1 | 2.2780 | 439.0 |
| `small_cnn` | cuda | 1 | 1.2119 | 825.2 |
| `torchvision_mobilenet` | cuda | 1 | 13.1993 | 75.8 |
| `ultra_tiny` | cuda | 32 | 0.0462 | 21651.6 |
| `ds_cnn` | cuda | 32 | 0.2103 | 4754.8 |
| `tiny_cnn` | cuda | 32 | 0.2073 | 4824.4 |
| `mobilenet_like` | cuda | 32 | 0.2306 | 4336.2 |
| `tiny_resnet` | cuda | 32 | 0.1786 | 5597.8 |
| `small_cnn` | cuda | 32 | 0.2338 | 4277.3 |
| `torchvision_mobilenet` | cuda | 32 | 0.3406 | 2935.8 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **27.7%** (0.0623 ms) del tiempo por etapas (E2E=0.1878 ms, gap de despacho Python/ctypes=+0.0371 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.2570 ms**, F1=0.7600), mientras el maximo F1 lo obtiene `mlp_16_dot (x94)` (**F1=0.8696**, 0.2624 ms).
