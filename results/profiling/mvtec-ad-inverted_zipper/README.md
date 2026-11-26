# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad-inverted_zipper`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `99bc026`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.7925 | 0.9485 | 0.3441 (desc) | 0.0076 | **0.3518** | 0.0403 | [0.3512, 0.3532] | 2842.1 | - |
| `mlp_16_dot (x94)` | ml | cpu | 1537 / 94 | 0.8571 | 0.9644 | 0.3441 (desc) | 0.0123 | **0.3563** | 0.0416 | [0.3556, 0.3571] | 2806.6 | - |
| `dt (x94)` | ml | cpu | 15 / 94 | 0.6792 | 0.8420 | 0.3441 (desc) | 0.0997 | **0.4445** | 0.0340 | [0.4435, 0.4486] | 2250.0 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.7925 | 0.9485 | 0.3441 (desc) | 0.2144 | **0.5580** | 0.0610 | [0.5558, 0.5620] | 1792.1 | - |
| `knn (x94)` | ml | cpu | 29328 / 94 | 0.2857 | 0.7924 | 0.3441 (desc) | 0.4938 | **0.8519** | 0.1195 | [0.8341, 0.8608] | 1173.8 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.6222 | 0.8258 | 0.0468 (prep) | 0.8119 | **0.8596** | 0.1893 | [0.8457, 0.8621] | 1163.4 | 0.00e+00 |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.6364 | 0.8659 | 0.0468 (prep) | 1.9089 | **1.9567** | 0.5766 | [1.9331, 2.0167] | 511.1 | 0.00e+00 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.5789 | 0.8530 | 0.0468 (prep) | 2.9469 | **2.9935** | 1.0212 | [2.9326, 3.2122] | 334.1 | 0.00e+00 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.4660 | 0.8902 | 0.0468 (prep) | 3.3223 | **3.3686** | 0.8736 | [3.3283, 3.4426] | 296.9 | 0.00e+00 |
| `rf (x94)` | ml | cpu | 5510 / 94 | 0.7556 | 0.9394 | 0.3441 (desc) | 4.1723 | **4.5045** | 0.7312 | [4.4224, 4.5661] | 222.0 | - |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.5405 | 0.8818 | 0.0468 (prep) | 5.0172 | **5.0640** | 1.4506 | [4.7585, 5.1697] | 197.5 | 0.00e+00 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.4660 | 0.8902 | 0.0468 (prep) | 5.6579 | **5.7063** | 1.0697 | [5.6149, 5.7918] | 175.2 | 0.00e+00 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.4660 | 0.5326 | 0.0468 (prep) | 6.8868 | **6.9341** | 1.4246 | [6.6767, 7.1834] | 144.2 | 1.44e-02 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **11.3612 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.3795 ms** vs llamada formal E2E: **0.3360 ms** (gap de dispatch/asignaciones Python = **+0.0435 ms**, +11.5%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0998 | 26.3% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g8` | 8x8 | 0.0933 | 24.6% | 2.6 | `O(g^2*K) [64x3]` |
| `asymmetry_g4` | 4x4 | 0.0928 | 24.5% | 2.1 | `O(g^2*K) [16x3]` |
| `c_fine_hist` | 8x8 | 0.0457 | 12.1% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0200 | 5.3% | 2.0 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.1376 | 879.1 |
| `ds_cnn` | cuda | 1 | 1.2691 | 788.0 |
| `tiny_cnn` | cuda | 1 | 0.7521 | 1329.6 |
| `mobilenet_like` | cuda | 1 | 2.2152 | 451.4 |
| `tiny_resnet` | cuda | 1 | 2.1385 | 467.6 |
| `small_cnn` | cuda | 1 | 0.9492 | 1053.6 |
| `torchvision_mobilenet` | cuda | 1 | 8.9451 | 111.8 |
| `ultra_tiny` | cuda | 32 | 0.0436 | 22911.1 |
| `ds_cnn` | cuda | 32 | 0.2041 | 4900.8 |
| `tiny_cnn` | cuda | 32 | 0.2064 | 4845.8 |
| `mobilenet_like` | cuda | 32 | 0.2297 | 4353.0 |
| `tiny_resnet` | cuda | 32 | 0.1786 | 5599.7 |
| `small_cnn` | cuda | 32 | 0.2320 | 4310.2 |
| `torchvision_mobilenet` | cuda | 32 | 0.2763 | 3619.5 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **26.3%** (0.0998 ms) del tiempo por etapas (E2E=0.3360 ms, gap de despacho Python/ctypes=+0.0435 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.3518 ms**, F1=0.7925), mientras el maximo F1 lo obtiene `mlp_16_dot (x94)` (**F1=0.8571**, 0.3563 ms).

## Nota Metodologica (Variante Invertida — Short Descriptor Check)

Siguiendo el protocolo §10, las variantes invertidas ejecutan comparacion CPU completa (Fase A) y esta verificacion corta del descriptor x94 para confirmar empiricamente que la inversion fotometrica (`255 - x`) conserva el mismo perfil computacional (E2E mediana batch=1 = **0.3360 ms**, staged sum = **0.3795 ms**).
