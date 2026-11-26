# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad-inverted_bottle`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `99bc026`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `mlp_16_dot (x94)` | ml | cpu | 1537 / 94 | 0.8000 | 0.9515 | 0.3660 (desc) | 0.0136 | **0.3817** | 0.1043 | [0.3764, 0.3961] | 2619.9 | - |
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.8000 | 0.9214 | 0.3660 (desc) | 0.0156 | **0.3845** | 0.1058 | [0.3788, 0.3978] | 2600.8 | - |
| `dt (x94)` | ml | cpu | 15 / 94 | 0.5833 | 0.6614 | 0.3660 (desc) | 0.1411 | **0.5131** | 0.1231 | [0.4997, 0.5232] | 1948.9 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.8000 | 0.9214 | 0.3660 (desc) | 0.2525 | **0.6432** | 0.1651 | [0.6037, 0.6637] | 1554.7 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.4138 | 0.5987 | 0.0403 (prep) | 0.9159 | **0.9562** | 0.5349 | [0.9082, 1.0576] | 1045.9 | 0.00e+00 |
| `knn (x94)` | ml | cpu | 21902 / 94 | 0.3750 | 0.7057 | 0.3660 (desc) | 0.5640 | **0.9575** | 0.2404 | [0.9274, 0.9865] | 1044.4 | - |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.6190 | 0.9365 | 0.0403 (prep) | 2.5105 | **2.5526** | 1.1894 | [2.4055, 2.6684] | 391.8 | 0.00e+00 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.3611 | 0.6739 | 0.0403 (prep) | 3.8226 | **3.8620** | 1.8659 | [3.6385, 4.2477] | 258.9 | 6.69e-03 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.5625 | 0.8127 | 0.0403 (prep) | 4.4764 | **4.5172** | 1.9150 | [4.2184, 4.9420] | 221.4 | 0.00e+00 |
| `rf (x94)` | ml | cpu | 3506 / 94 | 0.7273 | 0.9013 | 0.3660 (desc) | 5.6665 | **6.0682** | 1.6246 | [5.7179, 6.2068] | 164.8 | - |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.3636 | 0.6187 | 0.0403 (prep) | 6.5640 | **6.6036** | 2.6410 | [6.1724, 6.7862] | 151.4 | 0.00e+00 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.8000 | 0.9916 | 0.0403 (prep) | 8.0616 | **8.1017** | 1.7165 | [7.7815, 8.3633] | 123.4 | 0.00e+00 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.3611 | 0.5109 | 0.0403 (prep) | 9.3214 | **9.3614** | 2.6376 | [9.1470, 9.7261] | 106.8 | 5.69e-02 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **38.5422 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2348 ms** vs llamada formal E2E: **0.2180 ms** (gap de dispatch/asignaciones Python = **+0.0168 ms**, +7.2%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0626 | 26.7% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.0566 | 24.1% | 2.1 | `O(g^2*K) [16x3]` |
| `asymmetry_g8` | 8x8 | 0.0566 | 24.1% | 2.6 | `O(g^2*K) [64x3]` |
| `c_fine_hist` | 8x8 | 0.0300 | 12.8% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0117 | 5.0% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 2.3030 | 434.2 |
| `ds_cnn` | cuda | 1 | 3.0589 | 326.9 |
| `tiny_cnn` | cuda | 1 | 1.6521 | 605.3 |
| `mobilenet_like` | cuda | 1 | 3.9501 | 253.2 |
| `tiny_resnet` | cuda | 1 | 2.3119 | 432.6 |
| `small_cnn` | cuda | 1 | 1.4647 | 682.7 |
| `torchvision_mobilenet` | cuda | 1 | 16.1737 | 61.8 |
| `ultra_tiny` | cuda | 32 | 0.0649 | 15413.9 |
| `ds_cnn` | cuda | 32 | 0.2183 | 4580.4 |
| `tiny_cnn` | cuda | 32 | 0.2210 | 4524.5 |
| `mobilenet_like` | cuda | 32 | 0.2454 | 4075.6 |
| `tiny_resnet` | cuda | 32 | 0.1924 | 5197.7 |
| `small_cnn` | cuda | 32 | 0.2615 | 3824.3 |
| `torchvision_mobilenet` | cuda | 32 | 0.4347 | 2300.4 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **26.7%** (0.0626 ms) del tiempo por etapas (E2E=0.2180 ms, gap de despacho Python/ctypes=+0.0168 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `mlp_16_dot (x94)` (**0.3817 ms**, F1=0.8000), mientras el maximo F1 lo obtiene `lr (x94)` (**F1=0.8000**, 0.6432 ms).

## Nota Metodologica (Variante Invertida — Short Descriptor Check)

Siguiendo el protocolo §10, las variantes invertidas ejecutan comparacion CPU completa (Fase A) y esta verificacion corta del descriptor x94 para confirmar empiricamente que la inversion fotometrica (`255 - x`) conserva el mismo perfil computacional (E2E mediana batch=1 = **0.2180 ms**, staged sum = **0.2348 ms**).
