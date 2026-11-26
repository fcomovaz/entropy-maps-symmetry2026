# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad-inverted_hazelnut`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `99bc026`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.7429 | 0.9573 | 0.4412 (desc) | 0.0081 | **0.4503** | 0.1274 | [0.4481, 0.4532] | 2220.7 | - |
| `mlp_16_dot (x94)` | ml | cpu | 1537 / 94 | 0.7500 | 0.9548 | 0.4412 (desc) | 0.0134 | **0.4545** | 0.1237 | [0.4518, 0.4571] | 2200.5 | - |
| `dt (x94)` | ml | cpu | 15 / 94 | 0.5714 | 0.8194 | 0.4412 (desc) | 0.1247 | **0.5724** | 0.1292 | [0.5641, 0.5914] | 1747.2 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.7429 | 0.9573 | 0.4412 (desc) | 0.2396 | **0.6931** | 0.1362 | [0.6788, 0.7089] | 1442.8 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.6667 | 0.9819 | 0.0406 (prep) | 0.9147 | **0.9579** | 0.1671 | [0.9326, 0.9754] | 1044.0 | 0.00e+00 |
| `knn (x94)` | ml | cpu | 37600 / 94 | 0.3333 | 0.8654 | 0.4412 (desc) | 0.8117 | **1.2529** | 0.4381 | [1.2159, 1.3142] | 798.2 | - |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.8333 | 0.9655 | 0.0406 (prep) | 2.2647 | **2.3116** | 0.3539 | [2.2721, 2.3446] | 432.6 | 0.00e+00 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.7273 | 0.9992 | 0.0406 (prep) | 3.4419 | **3.4855** | 0.7813 | [3.4378, 3.5670] | 286.9 | 0.00e+00 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.8889 | 0.9967 | 0.0406 (prep) | 3.7540 | **3.7954** | 0.5235 | [3.7161, 3.8491] | 263.5 | 0.00e+00 |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.8800 | 0.9984 | 0.0406 (prep) | 5.3254 | **5.3722** | 1.2132 | [5.2524, 5.5212] | 186.1 | 0.00e+00 |
| `rf (x94)` | ml | cpu | 5458 / 94 | 0.6000 | 0.9212 | 0.4412 (desc) | 5.1063 | **5.4950** | 1.2225 | [5.3552, 5.6920] | 182.0 | - |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.8800 | 0.9951 | 0.0406 (prep) | 6.5897 | **6.6406** | 0.9187 | [6.5263, 6.7783] | 150.6 | 0.00e+00 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.0000 | 0.6720 | 0.0406 (prep) | 10.2886 | **10.3292** | 3.3241 | [9.9435, 10.7467] | 96.8 | 2.55e-02 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **10.9592 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2742 ms** vs llamada formal E2E: **0.2168 ms** (gap de dispatch/asignaciones Python = **+0.0575 ms**, +20.9%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0718 | 26.2% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g8` | 8x8 | 0.0689 | 25.1% | 2.6 | `O(g^2*K) [64x3]` |
| `asymmetry_g4` | 4x4 | 0.0683 | 24.9% | 2.1 | `O(g^2*K) [16x3]` |
| `c_fine_hist` | 8x8 | 0.0314 | 11.4% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0142 | 5.2% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.4203 | 704.1 |
| `ds_cnn` | cuda | 1 | 1.6246 | 615.5 |
| `tiny_cnn` | cuda | 1 | 0.8464 | 1181.5 |
| `mobilenet_like` | cuda | 1 | 3.0212 | 331.0 |
| `tiny_resnet` | cuda | 1 | 2.1353 | 468.3 |
| `small_cnn` | cuda | 1 | 1.2381 | 807.7 |
| `torchvision_mobilenet` | cuda | 1 | 10.9505 | 91.3 |
| `ultra_tiny` | cuda | 32 | 0.0497 | 20113.1 |
| `ds_cnn` | cuda | 32 | 0.2107 | 4746.0 |
| `tiny_cnn` | cuda | 32 | 0.2087 | 4792.3 |
| `mobilenet_like` | cuda | 32 | 0.2325 | 4301.4 |
| `tiny_resnet` | cuda | 32 | 0.1814 | 5512.5 |
| `small_cnn` | cuda | 32 | 0.2360 | 4237.0 |
| `torchvision_mobilenet` | cuda | 32 | 0.4005 | 2496.6 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **26.2%** (0.0718 ms) del tiempo por etapas (E2E=0.2168 ms, gap de despacho Python/ctypes=+0.0575 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.4503 ms**, F1=0.7429), mientras el maximo F1 lo obtiene `tiny_resnet` (**F1=0.8889**, 3.7954 ms).

## Nota Metodologica (Variante Invertida — Short Descriptor Check)

Siguiendo el protocolo §10, las variantes invertidas ejecutan comparacion CPU completa (Fase A) y esta verificacion corta del descriptor x94 para confirmar empiricamente que la inversion fotometrica (`255 - x`) conserva el mismo perfil computacional (E2E mediana batch=1 = **0.2168 ms**, staged sum = **0.2742 ms**).
