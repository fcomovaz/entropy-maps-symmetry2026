# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad-inverted_cable`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `99bc026`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.5116 | 0.7612 | 0.3718 (desc) | 0.0120 | **0.3860** | 0.2319 | [0.3738, 0.4186] | 2591.0 | - |
| `mlp_16_dot (x94)` | ml | cpu | 1537 / 94 | 0.5625 | 0.8411 | 0.3718 (desc) | 0.0255 | **0.4138** | 0.2286 | [0.3976, 0.4402] | 2416.3 | - |
| `dt (x94)` | ml | cpu | 13 / 94 | 0.4000 | 0.6014 | 0.3718 (desc) | 0.1368 | **0.5384** | 0.2649 | [0.5162, 0.5527] | 1857.4 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.5116 | 0.7612 | 0.3718 (desc) | 0.3652 | **0.7694** | 0.4376 | [0.7314, 0.8256] | 1299.6 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.5490 | 0.7329 | 0.0493 (prep) | 1.2439 | **1.2991** | 1.0465 | [1.1878, 1.5015] | 769.8 | 0.00e+00 |
| `knn (x94)` | ml | cpu | 28106 / 94 | 0.3478 | 0.6296 | 0.3718 (desc) | 0.8690 | **1.3234** | 0.9418 | [1.1676, 1.4690] | 755.7 | - |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.5357 | 0.7817 | 0.0493 (prep) | 3.9698 | **4.0213** | 2.5351 | [3.7144, 4.3715] | 248.7 | 0.00e+00 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.5556 | 0.8021 | 0.0493 (prep) | 4.1991 | **4.2611** | 1.4295 | [4.1402, 4.4903] | 234.7 | 0.00e+00 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.5714 | 0.8645 | 0.0493 (prep) | 6.5309 | **6.5858** | 4.0419 | [6.2673, 7.1273] | 151.8 | 0.00e+00 |
| `rf (x94)` | ml | cpu | 6020 / 94 | 0.4348 | 0.7700 | 0.3718 (desc) | 6.5121 | **7.1466** | 3.3582 | [6.6452, 7.5732] | 139.9 | - |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.6897 | 0.9172 | 0.0493 (prep) | 8.7602 | **8.8112** | 1.6141 | [8.6356, 8.9351] | 113.5 | 0.00e+00 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.3871 | 0.5000 | 0.0493 (prep) | 9.8225 | **9.8661** | 3.8725 | [9.1181, 10.4137] | 101.4 | 9.80e-02 |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.5714 | 0.8285 | 0.0493 (prep) | 10.0964 | **10.1641** | 3.0978 | [9.9390, 10.5413] | 98.4 | 0.00e+00 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **8.8461 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.3095 ms** vs llamada formal E2E: **0.2314 ms** (gap de dispatch/asignaciones Python = **+0.0780 ms**, +25.2%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g8` | 8x8 | 0.0813 | 26.3% | 2.6 | `O(g^2*K) [64x3]` |
| `asymmetry_g2` | 2x2 | 0.0791 | 25.6% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.0696 | 22.5% | 2.1 | `O(g^2*K) [16x3]` |
| `c_fine_hist` | 8x8 | 0.0361 | 11.7% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0186 | 6.0% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.2080 | 827.8 |
| `ds_cnn` | cuda | 1 | 1.6939 | 590.4 |
| `tiny_cnn` | cuda | 1 | 0.9294 | 1076.0 |
| `mobilenet_like` | cuda | 1 | 3.1294 | 319.6 |
| `tiny_resnet` | cuda | 1 | 1.9955 | 501.1 |
| `small_cnn` | cuda | 1 | 1.0341 | 967.0 |
| `torchvision_mobilenet` | cuda | 1 | 10.9510 | 91.3 |
| `ultra_tiny` | cuda | 32 | 0.0508 | 19697.8 |
| `ds_cnn` | cuda | 32 | 0.2064 | 4844.4 |
| `tiny_cnn` | cuda | 32 | 0.2085 | 4795.0 |
| `mobilenet_like` | cuda | 32 | 0.2328 | 4294.9 |
| `tiny_resnet` | cuda | 32 | 0.1817 | 5505.0 |
| `small_cnn` | cuda | 32 | 0.2345 | 4265.3 |
| `torchvision_mobilenet` | cuda | 32 | 0.3788 | 2639.6 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g8` concentra **26.3%** (0.0813 ms) del tiempo por etapas (E2E=0.2314 ms, gap de despacho Python/ctypes=+0.0780 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.3860 ms**, F1=0.5116), mientras el maximo F1 lo obtiene `small_cnn` (**F1=0.6897**, 8.8112 ms).

## Nota Metodologica (Variante Invertida — Short Descriptor Check)

Siguiendo el protocolo §10, las variantes invertidas ejecutan comparacion CPU completa (Fase A) y esta verificacion corta del descriptor x94 para confirmar empiricamente que la inversion fotometrica (`255 - x`) conserva el mismo perfil computacional (E2E mediana batch=1 = **0.2314 ms**, staged sum = **0.3095 ms**).
