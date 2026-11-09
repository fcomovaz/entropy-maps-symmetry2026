# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad-inverted_bottle`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `bf4a29f`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.4444 | 0.8946 | 0.0392 (prep) | 0.8038 | **0.8452** | 0.0528 | [0.8421, 0.8483] | 1183.1 | 0.00e+00 |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.4444 | 0.8813 | 0.0392 (prep) | 1.8692 | **1.9100** | 0.1946 | [1.9000, 1.9273] | 523.6 | 0.00e+00 |
| `dt (x94)` | ml | cpu | 15 / 94 | 0.5833 | 0.6614 | 2.2022 (desc) | 0.0816 | **2.2964** | 0.5817 | [2.2752, 2.3643] | 435.5 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.8000 | 0.9214 | 2.2022 (desc) | 0.2170 | **2.4703** | 0.5540 | [2.4239, 2.5267] | 404.8 | - |
| `knn (x94)` | ml | cpu | 21902 / 94 | 0.3750 | 0.7057 | 2.2022 (desc) | 0.4885 | **2.7054** | 0.5898 | [2.6678, 2.7630] | 369.6 | - |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.3611 | 0.4967 | 0.0392 (prep) | 3.1082 | **3.1474** | 0.6557 | [3.1026, 3.2851] | 317.7 | 6.69e-02 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.6452 | 0.8562 | 0.0392 (prep) | 3.4759 | **3.5189** | 0.6755 | [3.4248, 3.5783] | 284.2 | 0.00e+00 |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.6000 | 0.8846 | 0.0392 (prep) | 5.1602 | **5.1998** | 0.9754 | [5.0539, 5.2841] | 192.3 | 0.00e+00 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.8000 | 0.9799 | 0.0392 (prep) | 5.7896 | **5.8288** | 0.8910 | [5.7416, 5.9084] | 171.6 | 0.00e+00 |
| `rf (x94)` | ml | cpu | 3506 / 94 | 0.7273 | 0.9013 | 2.2022 (desc) | 4.1327 | **6.4997** | 0.6899 | [6.4108, 6.6468] | 153.9 | - |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.3611 | 0.5000 | 0.0392 (prep) | 6.5008 | **6.5397** | 0.6368 | [6.4465, 6.6257] | 152.9 | 0.00e+00 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E)

- Inicializacion amortizable (`build_entropy_plan` + LUTs): **0.6809 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **1.2712 ms** vs llamada formal E2E: **2.2074 ms** (gap de dispatch/asignaciones Python = **-0.9362 ms**, -73.7%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `region_histograms_g8` | 8x8 | 0.4951 | 38.9% | 13.4 | `O(H*W) [2*H*W=32768]` |
| `region_histograms_g4` | 4x4 | 0.1953 | 15.4% | 16.3 | `O(H*W) [2*H*W=32768]` |
| `region_histograms_g2` | 2x2 | 0.1122 | 8.8% | 50.8 | `O(H*W) [2*H*W=32768]` |
| `region_histograms_g1` | 1x1 | 0.0880 | 6.9% | 193.2 | `O(H*W) [2*H*W=32768]` |
| `asymmetry_g2` | 2x2 | 0.0650 | 5.1% | 2.1 | `O(g^2*K) [4x3]` |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 0.9905 | 1009.6 |
| `ds_cnn` | cuda | 1 | 1.3353 | 748.9 |
| `tiny_cnn` | cuda | 1 | 0.7813 | 1279.8 |
| `mobilenet_like` | cuda | 1 | 2.7517 | 363.4 |
| `tiny_resnet` | cuda | 1 | 1.8148 | 551.0 |
| `small_cnn` | cuda | 1 | 0.9966 | 1003.5 |
| `torchvision_mobilenet` | cuda | 1 | 10.3805 | 96.3 |
| `ultra_tiny` | cuda | 32 | 0.0388 | 25797.1 |
| `ds_cnn` | cuda | 32 | 0.2011 | 4972.8 |
| `tiny_cnn` | cuda | 32 | 0.2067 | 4836.8 |
| `mobilenet_like` | cuda | 32 | 0.2288 | 4370.8 |
| `tiny_resnet` | cuda | 32 | 0.1766 | 5663.6 |
| `small_cnn` | cuda | 32 | 0.2314 | 4322.0 |
| `torchvision_mobilenet` | cuda | 32 | 0.3087 | 3239.4 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Cuello de botella principal de x94:** `region_histograms_g8` consume **38.9%** (0.4951 ms) del tiempo por etapas; cualquier optimizacion futura en C/Cython debe priorizar el bucle de histogramas multi-escala antes que cuantizacion o asimetria.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `ultra_tiny` (**0.8452 ms**, F1=0.4444), mientras el maximo F1 lo obtiene `lr (x94)` (**F1=0.8000**, 2.4703 ms).

## Nota Metodologica (Variante Invertida — Short Descriptor Check)

Siguiendo el protocolo §10, las variantes invertidas ejecutan comparacion CPU completa (Fase A) y esta verificacion corta del descriptor x94 para confirmar empiricamente que la inversion fotometrica (`255 - x`) conserva el mismo perfil computacional (E2E mediana batch=1 = **2.2074 ms**, staged sum = **1.2712 ms**).
