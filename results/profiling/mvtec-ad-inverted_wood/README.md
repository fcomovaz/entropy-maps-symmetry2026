# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad-inverted_wood`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `99bc026`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.4828 | 0.7392 | 0.3573 (desc) | 0.0073 | **0.3647** | 0.0105 | [0.3640, 0.3659] | 2742.0 | - |
| `mlp_16_dot (x94)` | ml | cpu | 1537 / 94 | 0.5333 | 0.7238 | 0.3573 (desc) | 0.0117 | **0.3691** | 0.0096 | [0.3683, 0.3704] | 2708.9 | - |
| `dt (x94)` | ml | cpu | 13 / 94 | 0.3333 | 0.6343 | 0.3573 (desc) | 0.1387 | **0.4972** | 0.0120 | [0.4965, 0.4988] | 2011.3 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.4828 | 0.7392 | 0.3573 (desc) | 0.2135 | **0.5719** | 0.0294 | [0.5704, 0.5748] | 1748.4 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.1111 | 0.6744 | 0.0414 (prep) | 0.8127 | **0.8561** | 0.1961 | [0.8462, 0.8650] | 1168.0 | 0.00e+00 |
| `knn (x94)` | ml | cpu | 24440 / 94 | 0.0000 | 0.7207 | 0.3573 (desc) | 0.5093 | **0.8672** | 0.1028 | [0.8566, 0.8841] | 1153.1 | - |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.3396 | 0.5201 | 0.0414 (prep) | 1.9844 | **2.0339** | 0.7757 | [2.0200, 2.0854] | 491.7 | 0.00e+00 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.6923 | 0.8627 | 0.0414 (prep) | 2.9443 | **2.9916** | 0.8119 | [2.9302, 3.1746] | 334.3 | 0.00e+00 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.7619 | 0.9336 | 0.0414 (prep) | 3.4024 | **3.4440** | 0.9454 | [3.3576, 3.6738] | 290.4 | 0.00e+00 |
| `rf (x94)` | ml | cpu | 4352 / 94 | 0.2667 | 0.7623 | 0.3573 (desc) | 4.0724 | **4.4347** | 0.7669 | [4.3472, 4.5573] | 225.5 | - |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.1176 | 0.5756 | 0.0414 (prep) | 5.0621 | **5.1152** | 1.0811 | [4.9179, 5.3075] | 195.5 | 0.00e+00 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.0000 | 0.7299 | 0.0414 (prep) | 6.3438 | **6.3851** | 1.2961 | [6.2325, 6.6689] | 156.6 | 0.00e+00 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.3077 | 0.5262 | 0.0414 (prep) | 6.5485 | **6.5896** | 1.2851 | [6.4381, 6.8203] | 151.8 | 1.43e-01 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **12.6376 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.4528 ms** vs llamada formal E2E: **0.3822 ms** (gap de dispatch/asignaciones Python = **+0.0706 ms**, +15.6%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.1199 | 26.5% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.1107 | 24.4% | 2.1 | `O(g^2*K) [16x3]` |
| `asymmetry_g8` | 8x8 | 0.1101 | 24.3% | 2.6 | `O(g^2*K) [64x3]` |
| `c_fine_hist` | 8x8 | 0.0524 | 11.6% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0244 | 5.4% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.3059 | 765.8 |
| `ds_cnn` | cuda | 1 | 1.2508 | 799.5 |
| `tiny_cnn` | cuda | 1 | 0.7344 | 1361.7 |
| `mobilenet_like` | cuda | 1 | 2.1904 | 456.5 |
| `tiny_resnet` | cuda | 1 | 1.8218 | 548.9 |
| `small_cnn` | cuda | 1 | 0.9471 | 1055.9 |
| `torchvision_mobilenet` | cuda | 1 | 9.4657 | 105.6 |
| `ultra_tiny` | cuda | 32 | 0.0427 | 23422.6 |
| `ds_cnn` | cuda | 32 | 0.2092 | 4779.9 |
| `tiny_cnn` | cuda | 32 | 0.2055 | 4866.5 |
| `mobilenet_like` | cuda | 32 | 0.2303 | 4341.7 |
| `tiny_resnet` | cuda | 32 | 0.1778 | 5623.9 |
| `small_cnn` | cuda | 32 | 0.2308 | 4333.3 |
| `torchvision_mobilenet` | cuda | 32 | 0.2928 | 3414.8 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **26.5%** (0.1199 ms) del tiempo por etapas (E2E=0.3822 ms, gap de despacho Python/ctypes=+0.0706 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.3647 ms**, F1=0.4828), mientras el maximo F1 lo obtiene `tiny_resnet` (**F1=0.7619**, 3.4440 ms).

## Nota Metodologica (Variante Invertida — Short Descriptor Check)

Siguiendo el protocolo §10, las variantes invertidas ejecutan comparacion CPU completa (Fase A) y esta verificacion corta del descriptor x94 para confirmar empiricamente que la inversion fotometrica (`255 - x`) conserva el mismo perfil computacional (E2E mediana batch=1 = **0.3822 ms**, staged sum = **0.4528 ms**).
