# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad-inverted_metal_nut`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `99bc026`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.4762 | 0.7237 | 0.3720 (desc) | 0.0088 | **0.3806** | 0.1865 | [0.3710, 0.3861] | 2627.1 | - |
| `mlp_16_dot (x94)` | ml | cpu | 1537 / 94 | 0.6111 | 0.7884 | 0.3720 (desc) | 0.0118 | **0.3857** | 0.1908 | [0.3772, 0.3904] | 2592.4 | - |
| `dt (x94)` | ml | cpu | 11 / 94 | 0.5882 | 0.7621 | 0.3720 (desc) | 0.0877 | **0.4682** | 0.1814 | [0.4528, 0.4807] | 2135.8 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.4762 | 0.7237 | 0.3720 (desc) | 0.2257 | **0.6105** | 0.2182 | [0.5963, 0.6194] | 1638.1 | - |
| `knn (x94)` | ml | cpu | 25192 / 94 | 0.4800 | 0.7971 | 0.3720 (desc) | 0.5450 | **0.9107** | 0.2228 | [0.8902, 0.9337] | 1098.0 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.2581 | 0.6140 | 0.0422 (prep) | 0.8791 | **0.9253** | 0.3246 | [0.8996, 0.9512] | 1080.7 | 0.00e+00 |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.8500 | 0.9627 | 0.0422 (prep) | 2.4499 | **2.4920** | 1.2805 | [2.3690, 2.7392] | 401.3 | 0.00e+00 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.7600 | 0.9715 | 0.0422 (prep) | 3.2790 | **3.3208** | 1.5515 | [3.2614, 3.4768] | 301.1 | 0.00e+00 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.4419 | 0.9441 | 0.0422 (prep) | 4.3605 | **4.4025** | 2.1587 | [4.1235, 4.7824] | 227.1 | 0.00e+00 |
| `rf (x94)` | ml | cpu | 6042 / 94 | 0.6429 | 0.8476 | 0.3720 (desc) | 4.9652 | **5.3125** | 0.9129 | [5.1694, 5.4064] | 188.2 | - |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.4419 | 0.9589 | 0.0422 (prep) | 7.1567 | **7.2071** | 1.2958 | [7.1022, 7.3757] | 138.8 | 0.00e+00 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.4419 | 0.6502 | 0.0422 (prep) | 8.5318 | **8.5741** | 2.0668 | [8.2396, 8.7776] | 116.6 | 1.10e-03 |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.8261 | 0.9638 | 0.0422 (prep) | 10.8384 | **10.8811** | 3.1278 | [10.2585, 11.2035] | 91.9 | 0.00e+00 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **12.0489 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.3828 ms** vs llamada formal E2E: **0.2191 ms** (gap de dispatch/asignaciones Python = **+0.1637 ms**, +42.8%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.1004 | 26.2% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g8` | 8x8 | 0.0940 | 24.6% | 2.6 | `O(g^2*K) [64x3]` |
| `asymmetry_g4` | 4x4 | 0.0935 | 24.4% | 2.1 | `O(g^2*K) [16x3]` |
| `c_fine_hist` | 8x8 | 0.0466 | 12.2% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0201 | 5.3% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.3850 | 722.0 |
| `ds_cnn` | cuda | 1 | 1.6898 | 591.8 |
| `tiny_cnn` | cuda | 1 | 0.9913 | 1008.8 |
| `mobilenet_like` | cuda | 1 | 2.8894 | 346.1 |
| `tiny_resnet` | cuda | 1 | 2.2630 | 441.9 |
| `small_cnn` | cuda | 1 | 1.3221 | 756.4 |
| `torchvision_mobilenet` | cuda | 1 | 11.1410 | 89.8 |
| `ultra_tiny` | cuda | 32 | 0.0519 | 19273.6 |
| `ds_cnn` | cuda | 32 | 0.2142 | 4668.5 |
| `tiny_cnn` | cuda | 32 | 0.2088 | 4789.3 |
| `mobilenet_like` | cuda | 32 | 0.2337 | 4279.3 |
| `tiny_resnet` | cuda | 32 | 0.1802 | 5548.0 |
| `small_cnn` | cuda | 32 | 0.2356 | 4244.9 |
| `torchvision_mobilenet` | cuda | 32 | 0.3971 | 2518.4 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **26.2%** (0.1004 ms) del tiempo por etapas (E2E=0.2191 ms, gap de despacho Python/ctypes=+0.1637 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.3806 ms**, F1=0.4762), mientras el maximo F1 lo obtiene `ds_cnn` (**F1=0.8500**, 2.4920 ms).

## Nota Metodologica (Variante Invertida — Short Descriptor Check)

Siguiendo el protocolo §10, las variantes invertidas ejecutan comparacion CPU completa (Fase A) y esta verificacion corta del descriptor x94 para confirmar empiricamente que la inversion fotometrica (`255 - x`) conserva el mismo perfil computacional (E2E mediana batch=1 = **0.2191 ms**, staged sum = **0.3828 ms**).
