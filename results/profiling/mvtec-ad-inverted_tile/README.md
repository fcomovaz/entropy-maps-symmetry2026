# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad-inverted_tile`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `99bc026`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.8649 | 0.9678 | 0.3446 (desc) | 0.0076 | **0.3521** | 0.1447 | [0.3511, 0.3546] | 2840.1 | - |
| `mlp_16_dot (x94)` | ml | cpu | 1537 / 94 | 0.8108 | 0.9767 | 0.3446 (desc) | 0.0123 | **0.3569** | 0.1513 | [0.3559, 0.3584] | 2801.9 | - |
| `dt (x94)` | ml | cpu | 15 / 94 | 0.8387 | 0.9623 | 0.3446 (desc) | 0.0978 | **0.4465** | 0.1891 | [0.4419, 0.4578] | 2239.9 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.8649 | 0.9678 | 0.3446 (desc) | 0.2182 | **0.5678** | 0.1497 | [0.5608, 0.5745] | 1761.2 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.6250 | 0.8579 | 0.0399 (prep) | 0.8003 | **0.8403** | 0.1581 | [0.8162, 0.8528] | 1190.1 | 0.00e+00 |
| `knn (x94)` | ml | cpu | 26038 / 94 | 0.3810 | 0.8724 | 0.3446 (desc) | 0.6421 | **0.9624** | 0.3354 | [0.9306, 0.9866] | 1039.1 | - |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.6667 | 0.9512 | 0.0399 (prep) | 2.7355 | **2.7753** | 1.3957 | [2.5620, 3.0550] | 360.3 | 0.00e+00 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.6207 | 0.9367 | 0.0399 (prep) | 2.9334 | **2.9737** | 0.6980 | [2.9291, 3.0392] | 336.3 | 0.00e+00 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.4416 | 0.8468 | 0.0399 (prep) | 3.2969 | **3.3367** | 0.8322 | [3.2898, 3.4413] | 299.7 | 0.00e+00 |
| `rf (x94)` | ml | cpu | 3526 / 94 | 0.8000 | 0.9734 | 0.3446 (desc) | 4.3651 | **4.7206** | 0.7170 | [4.5634, 4.7972] | 211.8 | - |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.3908 | 0.8801 | 0.0399 (prep) | 5.9936 | **6.0336** | 1.4188 | [5.8650, 6.3213] | 165.7 | 0.00e+00 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.3908 | 0.8002 | 0.0399 (prep) | 6.5967 | **6.6364** | 1.3369 | [6.4451, 6.9600] | 150.7 | 0.00e+00 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.3908 | 0.6854 | 0.0399 (prep) | 8.0381 | **8.0783** | 1.3683 | [8.0060, 8.1523] | 123.8 | 1.53e-01 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **9.2191 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2334 ms** vs llamada formal E2E: **0.2602 ms** (gap de dispatch/asignaciones Python = **-0.0268 ms**, -11.5%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0628 | 26.9% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.0573 | 24.5% | 2.1 | `O(g^2*K) [16x3]` |
| `asymmetry_g8` | 8x8 | 0.0570 | 24.4% | 2.6 | `O(g^2*K) [64x3]` |
| `c_fine_hist` | 8x8 | 0.0271 | 11.6% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0118 | 5.0% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.1024 | 907.1 |
| `ds_cnn` | cuda | 1 | 1.2757 | 783.9 |
| `tiny_cnn` | cuda | 1 | 0.7362 | 1358.3 |
| `mobilenet_like` | cuda | 1 | 2.2184 | 450.8 |
| `tiny_resnet` | cuda | 1 | 1.6973 | 589.2 |
| `small_cnn` | cuda | 1 | 0.9551 | 1047.0 |
| `torchvision_mobilenet` | cuda | 1 | 9.0727 | 110.2 |
| `ultra_tiny` | cuda | 32 | 0.0475 | 21054.0 |
| `ds_cnn` | cuda | 32 | 0.2094 | 4776.3 |
| `tiny_cnn` | cuda | 32 | 0.2049 | 4880.0 |
| `mobilenet_like` | cuda | 32 | 0.2286 | 4375.4 |
| `tiny_resnet` | cuda | 32 | 0.1793 | 5578.5 |
| `small_cnn` | cuda | 32 | 0.2311 | 4326.3 |
| `torchvision_mobilenet` | cuda | 32 | 0.2781 | 3596.1 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **26.9%** (0.0628 ms) del tiempo por etapas (E2E=0.2602 ms, gap de despacho Python/ctypes=-0.0268 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.3521 ms**, F1=0.8649), mientras el maximo F1 lo obtiene `lr (x94)` (**F1=0.8649**, 0.5678 ms).

## Nota Metodologica (Variante Invertida — Short Descriptor Check)

Siguiendo el protocolo §10, las variantes invertidas ejecutan comparacion CPU completa (Fase A) y esta verificacion corta del descriptor x94 para confirmar empiricamente que la inversion fotometrica (`255 - x`) conserva el mismo perfil computacional (E2E mediana batch=1 = **0.2602 ms**, staged sum = **0.2334 ms**).
