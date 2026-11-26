# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad-inverted_carpet`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `99bc026`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.4167 | 0.6452 | 0.3461 (desc) | 0.0092 | **0.3541** | 0.1444 | [0.3510, 0.3643] | 2824.1 | - |
| `mlp_16_dot (x94)` | ml | cpu | 1537 / 94 | 0.4211 | 0.6935 | 0.3461 (desc) | 0.0123 | **0.3584** | 0.1512 | [0.3545, 0.3666] | 2789.8 | - |
| `dt (x94)` | ml | cpu | 13 / 94 | 0.4262 | 0.7003 | 0.3461 (desc) | 0.1019 | **0.4485** | 0.1391 | [0.4420, 0.4605] | 2229.7 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.4167 | 0.6452 | 0.3461 (desc) | 0.2364 | **0.6019** | 0.1939 | [0.5884, 0.6113] | 1661.4 | - |
| `knn (x94)` | ml | cpu | 29798 / 94 | 0.2500 | 0.5582 | 0.3461 (desc) | 0.5969 | **0.9423** | 0.3235 | [0.9132, 0.9758] | 1061.3 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.0952 | 0.6971 | 0.0478 (prep) | 1.2529 | **1.3034** | 0.7404 | [1.2242, 1.4006] | 767.3 | 0.00e+00 |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.6977 | 0.8683 | 0.0478 (prep) | 3.3760 | **3.4263** | 1.4436 | [3.2710, 3.5907] | 291.9 | 0.00e+00 |
| `rf (x94)` | ml | cpu | 6262 / 94 | 0.1905 | 0.7858 | 0.3461 (desc) | 4.9988 | **5.3525** | 0.9101 | [5.2279, 5.4244] | 186.8 | - |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.3855 | 0.7706 | 0.0478 (prep) | 6.0221 | **6.1163** | 3.5064 | [5.4976, 6.6639] | 163.5 | 0.00e+00 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.4681 | 0.7312 | 0.0478 (prep) | 6.9636 | **7.0225** | 4.7921 | [6.4271, 7.5606] | 142.4 | 0.00e+00 |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.6667 | 0.9104 | 0.0478 (prep) | 7.1139 | **7.1723** | 1.8052 | [6.9246, 7.4203] | 139.4 | 0.00e+00 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.6667 | 0.9095 | 0.0478 (prep) | 8.9360 | **8.9809** | 2.5878 | [8.6603, 9.1959] | 111.3 | 0.00e+00 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.3673 | 0.5179 | 0.0478 (prep) | 9.9235 | **9.9833** | 4.5857 | [9.6114, 10.8838] | 100.2 | 7.89e-02 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **8.1220 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.3789 ms** vs llamada formal E2E: **0.2511 ms** (gap de dispatch/asignaciones Python = **+0.1278 ms**, +33.7%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.1008 | 26.6% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g8` | 8x8 | 0.0937 | 24.7% | 2.6 | `O(g^2*K) [64x3]` |
| `asymmetry_g4` | 4x4 | 0.0936 | 24.7% | 2.1 | `O(g^2*K) [16x3]` |
| `c_fine_hist` | 8x8 | 0.0420 | 11.1% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0201 | 5.3% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.3540 | 738.5 |
| `ds_cnn` | cuda | 1 | 1.5839 | 631.4 |
| `tiny_cnn` | cuda | 1 | 0.9736 | 1027.1 |
| `mobilenet_like` | cuda | 1 | 2.9970 | 333.7 |
| `tiny_resnet` | cuda | 1 | 2.8744 | 347.9 |
| `small_cnn` | cuda | 1 | 1.6524 | 605.2 |
| `torchvision_mobilenet` | cuda | 1 | 10.4499 | 95.7 |
| `ultra_tiny` | cuda | 32 | 0.0587 | 17024.4 |
| `ds_cnn` | cuda | 32 | 0.2126 | 4703.5 |
| `tiny_cnn` | cuda | 32 | 0.2089 | 4786.2 |
| `mobilenet_like` | cuda | 32 | 0.2331 | 4290.8 |
| `tiny_resnet` | cuda | 32 | 0.1811 | 5522.3 |
| `small_cnn` | cuda | 32 | 0.2335 | 4282.3 |
| `torchvision_mobilenet` | cuda | 32 | 0.3970 | 2518.6 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **26.6%** (0.1008 ms) del tiempo por etapas (E2E=0.2511 ms, gap de despacho Python/ctypes=+0.1278 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.3541 ms**, F1=0.4167), mientras el maximo F1 lo obtiene `ds_cnn` (**F1=0.6977**, 3.4263 ms).

## Nota Metodologica (Variante Invertida — Short Descriptor Check)

Siguiendo el protocolo §10, las variantes invertidas ejecutan comparacion CPU completa (Fase A) y esta verificacion corta del descriptor x94 para confirmar empiricamente que la inversion fotometrica (`255 - x`) conserva el mismo perfil computacional (E2E mediana batch=1 = **0.2511 ms**, staged sum = **0.3789 ms**).
