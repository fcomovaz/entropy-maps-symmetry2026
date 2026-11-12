# Profiling Cientifico y Comparacion 100% CPU: `kolektorsdd2-inverted`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `37d8114`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.4153 | 0.8151 | 0.3866 (desc) | 0.0077 | **0.3948** | 0.1473 | [0.3872, 0.4034] | 2532.6 | - |
| `dt (x94)` | ml | cpu | 305 / 94 | 0.3140 | 0.6228 | 0.3866 (desc) | 0.0879 | **0.4800** | 0.1630 | [0.4716, 0.4909] | 2083.1 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.4153 | 0.8151 | 0.3866 (desc) | 0.2232 | **0.6226** | 0.1322 | [0.6120, 0.6424] | 1606.2 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.5185 | 0.8179 | 0.0430 (prep) | 1.0725 | **1.1187** | 0.1949 | [1.1001, 1.1328] | 893.9 | 0.00e+00 |
| `knn (x94)` | ml | cpu | 219114 / 94 | 0.1500 | 0.6607 | 0.3866 (desc) | 1.2199 | **1.6123** | 0.3417 | [1.5741, 1.6529] | 620.2 | - |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.5665 | 0.8673 | 0.0430 (prep) | 2.5924 | **2.6429** | 1.0069 | [2.5273, 2.7171] | 378.4 | 0.00e+00 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.4845 | 0.7900 | 0.0430 (prep) | 4.0084 | **4.0717** | 0.7079 | [3.9426, 4.1661] | 245.6 | 0.00e+00 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.6509 | 0.9131 | 0.0430 (prep) | 4.2787 | **4.3338** | 0.7964 | [4.2546, 4.4405] | 230.7 | 0.00e+00 |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.2098 | 0.8281 | 0.0430 (prep) | 5.1671 | **5.2203** | 0.6665 | [5.1397, 5.2783] | 191.6 | 0.00e+00 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.5682 | 0.8684 | 0.0430 (prep) | 6.7672 | **6.8091** | 0.6638 | [6.7352, 6.9152] | 146.9 | 0.00e+00 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.1975 | 0.5369 | 0.0430 (prep) | 8.1141 | **8.1720** | 0.8477 | [8.0739, 8.3183] | 122.4 | 1.01e-03 |
| `rf (x94)` | ml | cpu | 13728 / 94 | 0.4237 | 0.7978 | 0.3866 (desc) | 9.6724 | **10.0285** | 0.5775 | [9.9622, 10.1065] | 99.7 | - |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **9.2964 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2160 ms** vs llamada formal E2E: **0.1938 ms** (gap de dispatch/asignaciones Python = **+0.0222 ms**, +10.3%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0570 | 26.4% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.0513 | 23.7% | 2.1 | `O(g^2*K) [16x3]` |
| `asymmetry_g8` | 8x8 | 0.0508 | 23.5% | 2.6 | `O(g^2*K) [64x3]` |
| `c_fine_hist` | 8x8 | 0.0298 | 13.8% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0111 | 5.1% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.1785 | 848.5 |
| `ds_cnn` | cuda | 1 | 1.3329 | 750.2 |
| `tiny_cnn` | cuda | 1 | 0.7848 | 1274.2 |
| `mobilenet_like` | cuda | 1 | 2.8864 | 346.4 |
| `tiny_resnet` | cuda | 1 | 2.1749 | 459.8 |
| `small_cnn` | cuda | 1 | 1.1950 | 836.8 |
| `torchvision_mobilenet` | cuda | 1 | 10.1202 | 98.8 |
| `ultra_tiny` | cuda | 32 | 0.0489 | 20432.9 |
| `ds_cnn` | cuda | 32 | 0.2067 | 4837.4 |
| `tiny_cnn` | cuda | 32 | 0.2086 | 4794.5 |
| `mobilenet_like` | cuda | 32 | 0.2305 | 4338.4 |
| `tiny_resnet` | cuda | 32 | 0.1781 | 5615.2 |
| `small_cnn` | cuda | 32 | 0.2336 | 4281.2 |
| `torchvision_mobilenet` | cuda | 32 | 0.3586 | 2789.0 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **26.4%** (0.0570 ms) del tiempo por etapas (E2E=0.1938 ms, gap de despacho Python/ctypes=+0.0222 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.3948 ms**, F1=0.4153), mientras el maximo F1 lo obtiene `tiny_resnet` (**F1=0.6509**, 4.3338 ms).

## Nota Metodologica (Variante Invertida — Short Descriptor Check)

Siguiendo el protocolo §10, las variantes invertidas ejecutan comparacion CPU completa (Fase A) y esta verificacion corta del descriptor x94 para confirmar empiricamente que la inversion fotometrica (`255 - x`) conserva el mismo perfil computacional (E2E mediana batch=1 = **0.1938 ms**, staged sum = **0.2160 ms**).
