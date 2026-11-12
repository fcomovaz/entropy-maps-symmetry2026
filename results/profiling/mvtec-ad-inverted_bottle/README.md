# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad-inverted_bottle`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `486c111`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.8000 | 0.9214 | 0.2316 (desc) | 0.0077 | **0.2395** | 0.1368 | [0.2315, 0.2577] | 4175.4 | - |
| `dt (x94)` | ml | cpu | 15 / 94 | 0.5833 | 0.6614 | 0.2316 (desc) | 0.1318 | **0.3767** | 0.0992 | [0.3649, 0.3975] | 2654.6 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.8000 | 0.9214 | 0.2316 (desc) | 0.2561 | **0.5222** | 0.1613 | [0.4958, 0.5474] | 1915.2 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.4138 | 0.5987 | 0.0471 (prep) | 0.9058 | **0.9609** | 0.2178 | [0.9507, 0.9852] | 1040.6 | 0.00e+00 |
| `knn (x94)` | ml | cpu | 21902 / 94 | 0.3750 | 0.7057 | 0.2316 (desc) | 0.7225 | **1.0112** | 0.4252 | [0.9646, 1.0694] | 988.9 | - |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.6190 | 0.9365 | 0.0471 (prep) | 3.4620 | **3.5351** | 2.0643 | [3.2372, 3.6962] | 282.9 | 0.00e+00 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.3611 | 0.6739 | 0.0471 (prep) | 4.2831 | **4.3317** | 2.7644 | [4.0553, 4.8641] | 230.9 | 6.69e-03 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.5625 | 0.8127 | 0.0471 (prep) | 4.6106 | **4.6738** | 1.6924 | [4.4002, 4.9533] | 214.0 | 0.00e+00 |
| `rf (x94)` | ml | cpu | 3506 / 94 | 0.7273 | 0.9013 | 0.2316 (desc) | 5.2123 | **5.4762** | 1.1635 | [5.3671, 5.6331] | 182.6 | - |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.3636 | 0.6187 | 0.0471 (prep) | 5.8533 | **5.8984** | 1.8581 | [5.5389, 6.2956] | 169.5 | 0.00e+00 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.8000 | 0.9916 | 0.0471 (prep) | 7.2008 | **7.2747** | 1.9346 | [7.0319, 7.6148] | 137.5 | 0.00e+00 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.3611 | 0.5109 | 0.0471 (prep) | 8.9516 | **8.9965** | 3.2300 | [8.3948, 9.8369] | 111.2 | 5.69e-02 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **9.6808 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2319 ms** vs llamada formal E2E: **0.2643 ms** (gap de dispatch/asignaciones Python = **-0.0323 ms**, -13.9%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0617 | 26.6% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.0557 | 24.0% | 2.1 | `O(g^2*K) [16x3]` |
| `asymmetry_g8` | 8x8 | 0.0557 | 24.0% | 2.6 | `O(g^2*K) [64x3]` |
| `c_fine_hist` | 8x8 | 0.0297 | 12.8% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0120 | 5.2% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 0.9818 | 1018.5 |
| `ds_cnn` | cuda | 1 | 1.3355 | 748.8 |
| `tiny_cnn` | cuda | 1 | 1.1031 | 906.5 |
| `mobilenet_like` | cuda | 1 | 3.0783 | 324.9 |
| `tiny_resnet` | cuda | 1 | 2.1570 | 463.6 |
| `small_cnn` | cuda | 1 | 0.9911 | 1008.9 |
| `torchvision_mobilenet` | cuda | 1 | 11.7344 | 85.2 |
| `ultra_tiny` | cuda | 32 | 0.0583 | 17155.9 |
| `ds_cnn` | cuda | 32 | 0.2045 | 4889.2 |
| `tiny_cnn` | cuda | 32 | 0.2070 | 4831.2 |
| `mobilenet_like` | cuda | 32 | 0.2555 | 3913.2 |
| `tiny_resnet` | cuda | 32 | 0.1845 | 5420.4 |
| `small_cnn` | cuda | 32 | 0.2330 | 4291.8 |
| `torchvision_mobilenet` | cuda | 32 | 0.3727 | 2683.0 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **26.6%** (0.0617 ms) del tiempo por etapas (E2E=0.2643 ms, gap de despacho Python/ctypes=-0.0323 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.2395 ms**, F1=0.8000), mientras el maximo F1 lo obtiene `lr (x94)` (**F1=0.8000**, 0.5222 ms).

## Nota Metodologica (Variante Invertida — Short Descriptor Check)

Siguiendo el protocolo §10, las variantes invertidas ejecutan comparacion CPU completa (Fase A) y esta verificacion corta del descriptor x94 para confirmar empiricamente que la inversion fotometrica (`255 - x`) conserva el mismo perfil computacional (E2E mediana batch=1 = **0.2643 ms**, staged sum = **0.2319 ms**).
