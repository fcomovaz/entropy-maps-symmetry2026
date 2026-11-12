# Profiling Cientifico y Comparacion 100% CPU: `kolektorsdd2-inverted`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `486c111`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.4153 | 0.8151 | 0.2326 (desc) | 0.0092 | **0.2450** | 0.1444 | [0.2407, 0.2625] | 4081.6 | - |
| `dt (x94)` | ml | cpu | 305 / 94 | 0.3140 | 0.6228 | 0.2326 (desc) | 0.0856 | **0.3207** | 0.1587 | [0.3171, 0.3421] | 3117.7 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.4153 | 0.8151 | 0.2326 (desc) | 0.2848 | **0.6035** | 0.2457 | [0.5561, 0.6405] | 1657.0 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.5185 | 0.8179 | 0.0431 (prep) | 1.0359 | **1.0783** | 0.3679 | [1.0570, 1.1281] | 927.4 | 0.00e+00 |
| `knn (x94)` | ml | cpu | 219114 / 94 | 0.1500 | 0.6607 | 0.2326 (desc) | 0.8062 | **1.0897** | 0.1880 | [1.0490, 1.1290] | 917.7 | - |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.4845 | 0.7900 | 0.0431 (prep) | 3.5319 | **3.5752** | 1.0493 | [3.4330, 3.7492] | 279.7 | 0.00e+00 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.6509 | 0.9131 | 0.0431 (prep) | 3.8072 | **3.8522** | 0.7189 | [3.7383, 3.9935] | 259.6 | 0.00e+00 |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.5665 | 0.8673 | 0.0431 (prep) | 4.0871 | **4.1302** | 1.9298 | [3.8878, 4.4239] | 242.1 | 0.00e+00 |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.2098 | 0.8281 | 0.0431 (prep) | 4.8556 | **4.8980** | 0.9334 | [4.7115, 4.9965] | 204.2 | 0.00e+00 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.5682 | 0.8684 | 0.0431 (prep) | 6.3673 | **6.4104** | 1.3382 | [6.3056, 6.5613] | 156.0 | 0.00e+00 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.1975 | 0.5369 | 0.0431 (prep) | 7.2463 | **7.2960** | 0.8709 | [7.2027, 7.4138] | 137.1 | 1.01e-03 |
| `rf (x94)` | ml | cpu | 13728 / 94 | 0.4237 | 0.7978 | 0.2326 (desc) | 9.2464 | **9.4816** | 1.5181 | [9.4027, 9.6516] | 105.5 | - |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **11.8748 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2281 ms** vs llamada formal E2E: **0.2391 ms** (gap de dispatch/asignaciones Python = **-0.0110 ms**, -4.8%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0623 | 27.3% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.0544 | 23.8% | 2.1 | `O(g^2*K) [16x3]` |
| `asymmetry_g8` | 8x8 | 0.0542 | 23.7% | 2.6 | `O(g^2*K) [64x3]` |
| `c_fine_hist` | 8x8 | 0.0294 | 12.9% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0113 | 5.0% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.3305 | 751.6 |
| `ds_cnn` | cuda | 1 | 1.5111 | 661.7 |
| `tiny_cnn` | cuda | 1 | 1.1522 | 867.9 |
| `mobilenet_like` | cuda | 1 | 2.9535 | 338.6 |
| `tiny_resnet` | cuda | 1 | 2.1825 | 458.2 |
| `small_cnn` | cuda | 1 | 1.0835 | 922.9 |
| `torchvision_mobilenet` | cuda | 1 | 10.9656 | 91.2 |
| `ultra_tiny` | cuda | 32 | 0.0624 | 16028.0 |
| `ds_cnn` | cuda | 32 | 0.2119 | 4719.6 |
| `tiny_cnn` | cuda | 32 | 0.2065 | 4841.6 |
| `mobilenet_like` | cuda | 32 | 0.2347 | 4261.6 |
| `tiny_resnet` | cuda | 32 | 0.1807 | 5533.3 |
| `small_cnn` | cuda | 32 | 0.2516 | 3974.9 |
| `torchvision_mobilenet` | cuda | 32 | 0.3695 | 2706.6 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **27.3%** (0.0623 ms) del tiempo por etapas (E2E=0.2391 ms, gap de despacho Python/ctypes=-0.0110 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.2450 ms**, F1=0.4153), mientras el maximo F1 lo obtiene `tiny_resnet` (**F1=0.6509**, 3.8522 ms).

## Nota Metodologica (Variante Invertida — Short Descriptor Check)

Siguiendo el protocolo §10, las variantes invertidas ejecutan comparacion CPU completa (Fase A) y esta verificacion corta del descriptor x94 para confirmar empiricamente que la inversion fotometrica (`255 - x`) conserva el mismo perfil computacional (E2E mediana batch=1 = **0.2391 ms**, staged sum = **0.2281 ms**).
