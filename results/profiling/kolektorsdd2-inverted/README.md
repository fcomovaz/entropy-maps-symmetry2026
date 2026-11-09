# Profiling Cientifico y Comparacion 100% CPU: `kolektorsdd2-inverted`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `bf4a29f`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.3889 | 0.7440 | 0.0417 (prep) | 0.7903 | **0.8409** | 0.0441 | [0.8263, 0.8447] | 1189.2 | 0.00e+00 |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.6355 | 0.8744 | 0.0417 (prep) | 1.8708 | **1.9151** | 0.1032 | [1.9040, 1.9268] | 522.2 | 0.00e+00 |
| `dt (x94)` | ml | cpu | 305 / 94 | 0.3140 | 0.6228 | 2.2113 (desc) | 0.0767 | **2.3104** | 0.4334 | [2.2491, 2.3484] | 432.8 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.4153 | 0.8151 | 2.2113 (desc) | 0.2125 | **2.4343** | 0.4420 | [2.3788, 2.4844] | 410.8 | - |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.6000 | 0.8405 | 0.0417 (prep) | 2.9028 | **2.9451** | 0.1659 | [2.9234, 2.9632] | 339.6 | 0.00e+00 |
| `knn (x94)` | ml | cpu | 219114 / 94 | 0.1500 | 0.6607 | 2.2113 (desc) | 0.7263 | **3.0417** | 0.4669 | [2.9775, 3.1043] | 328.8 | - |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.7030 | 0.8743 | 0.0417 (prep) | 3.2186 | **3.2626** | 0.1575 | [3.2474, 3.2783] | 306.5 | 0.00e+00 |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.5625 | 0.8170 | 0.0417 (prep) | 4.0334 | **4.0750** | 0.4256 | [4.0389, 4.1412] | 245.4 | 0.00e+00 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.5949 | 0.8612 | 0.0417 (prep) | 5.0760 | **5.1232** | 0.1984 | [5.0960, 5.1424] | 195.2 | 0.00e+00 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.1975 | 0.5206 | 0.0417 (prep) | 6.4200 | **6.4673** | 0.4917 | [6.4084, 6.5246] | 154.6 | 7.25e-03 |
| `rf (x94)` | ml | cpu | 13728 / 94 | 0.4237 | 0.7978 | 2.2113 (desc) | 7.9494 | **10.2474** | 0.5604 | [10.1553, 10.2875] | 97.6 | - |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E)

- Inicializacion amortizable (`build_entropy_plan` + LUTs): **1.1118 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **1.1986 ms** vs llamada formal E2E: **2.0591 ms** (gap de dispatch/asignaciones Python = **-0.8605 ms**, -71.8%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `region_histograms_g8` | 8x8 | 0.4662 | 38.9% | 13.4 | `O(H*W) [2*H*W=32768]` |
| `region_histograms_g4` | 4x4 | 0.1879 | 15.7% | 16.3 | `O(H*W) [2*H*W=32768]` |
| `region_histograms_g2` | 2x2 | 0.1086 | 9.1% | 50.8 | `O(H*W) [2*H*W=32768]` |
| `region_histograms_g1` | 1x1 | 0.0847 | 7.1% | 193.2 | `O(H*W) [2*H*W=32768]` |
| `asymmetry_g2` | 2x2 | 0.0605 | 5.0% | 2.1 | `O(g^2*K) [4x3]` |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.2279 | 814.4 |
| `ds_cnn` | cuda | 1 | 1.3010 | 768.6 |
| `tiny_cnn` | cuda | 1 | 1.2059 | 829.3 |
| `mobilenet_like` | cuda | 1 | 2.4246 | 412.4 |
| `tiny_resnet` | cuda | 1 | 1.7513 | 571.0 |
| `small_cnn` | cuda | 1 | 0.9893 | 1010.8 |
| `torchvision_mobilenet` | cuda | 1 | 8.9101 | 112.2 |
| `ultra_tiny` | cuda | 32 | 0.0450 | 22229.9 |
| `ds_cnn` | cuda | 32 | 0.2043 | 4895.2 |
| `tiny_cnn` | cuda | 32 | 0.2043 | 4894.0 |
| `mobilenet_like` | cuda | 32 | 0.2275 | 4394.8 |
| `tiny_resnet` | cuda | 32 | 0.1764 | 5670.2 |
| `small_cnn` | cuda | 32 | 0.2308 | 4332.8 |
| `torchvision_mobilenet` | cuda | 32 | 0.2783 | 3592.7 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Cuello de botella principal de x94:** `region_histograms_g8` consume **38.9%** (0.4662 ms) del tiempo por etapas; cualquier optimizacion futura en C/Cython debe priorizar el bucle de histogramas multi-escala antes que cuantizacion o asimetria.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `ultra_tiny` (**0.8409 ms**, F1=0.3889), mientras el maximo F1 lo obtiene `tiny_resnet` (**F1=0.7030**, 3.2626 ms).

## Nota Metodologica (Variante Invertida — Short Descriptor Check)

Siguiendo el protocolo §10, las variantes invertidas ejecutan comparacion CPU completa (Fase A) y esta verificacion corta del descriptor x94 para confirmar empiricamente que la inversion fotometrica (`255 - x`) conserva el mismo perfil computacional (E2E mediana batch=1 = **2.0591 ms**, staged sum = **1.1986 ms**).
