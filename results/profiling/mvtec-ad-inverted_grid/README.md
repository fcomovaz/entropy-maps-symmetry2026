# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad-inverted_grid`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `99bc026`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.3704 | 0.5893 | 0.3316 (desc) | 0.0079 | **0.3395** | 0.1498 | [0.3070, 0.3513] | 2945.5 | - |
| `mlp_16_dot (x94)` | ml | cpu | 1537 / 94 | 0.3571 | 0.7320 | 0.3316 (desc) | 0.0158 | **0.3487** | 0.1467 | [0.3152, 0.3609] | 2868.2 | - |
| `dt (x94)` | ml | cpu | 13 / 94 | 0.2979 | 0.5893 | 0.3316 (desc) | 0.0859 | **0.4269** | 0.1488 | [0.3993, 0.4338] | 2342.2 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.3704 | 0.5893 | 0.3316 (desc) | 0.2260 | **0.5826** | 0.1964 | [0.5546, 0.5985] | 1716.6 | - |
| `knn (x94)` | ml | cpu | 25662 / 94 | 0.0000 | 0.5180 | 0.3316 (desc) | 0.5458 | **0.8744** | 0.1808 | [0.8553, 0.9062] | 1143.6 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.2750 | 0.3636 | 0.0618 (prep) | 1.2854 | **1.3470** | 0.6205 | [1.2873, 1.4523] | 742.4 | 0.00e+00 |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.4390 | 0.7226 | 0.0618 (prep) | 2.8100 | **2.9011** | 1.1914 | [2.6341, 3.0346] | 344.7 | 0.00e+00 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.4242 | 0.7994 | 0.0618 (prep) | 4.0183 | **4.0784** | 1.2682 | [3.8805, 4.2286] | 245.2 | 0.00e+00 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.4737 | 0.7994 | 0.0618 (prep) | 4.3951 | **4.4600** | 1.3263 | [4.2872, 4.6152] | 224.2 | 0.00e+00 |
| `rf (x94)` | ml | cpu | 5038 / 94 | 0.0000 | 0.5925 | 0.3316 (desc) | 5.1797 | **5.5359** | 1.1054 | [5.3872, 5.6203] | 180.6 | - |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.4348 | 0.8103 | 0.0618 (prep) | 6.3767 | **6.4394** | 1.2611 | [6.3277, 6.7203] | 155.3 | 0.00e+00 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.3478 | 0.7085 | 0.0618 (prep) | 7.5138 | **7.5812** | 1.0274 | [7.4566, 7.6714] | 131.9 | 0.00e+00 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.2750 | 0.5313 | 0.0618 (prep) | 9.0120 | **9.1024** | 0.9962 | [9.0267, 9.2057] | 109.9 | 9.40e-03 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **12.5680 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.4669 ms** vs llamada formal E2E: **0.4435 ms** (gap de dispatch/asignaciones Python = **+0.0233 ms**, +5.0%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.1245 | 26.7% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.1156 | 24.8% | 2.1 | `O(g^2*K) [16x3]` |
| `asymmetry_g8` | 8x8 | 0.1135 | 24.3% | 2.6 | `O(g^2*K) [64x3]` |
| `c_fine_hist` | 8x8 | 0.0520 | 11.1% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0246 | 5.3% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.1629 | 859.9 |
| `ds_cnn` | cuda | 1 | 1.4691 | 680.7 |
| `tiny_cnn` | cuda | 1 | 0.7944 | 1258.8 |
| `mobilenet_like` | cuda | 1 | 2.5715 | 388.9 |
| `tiny_resnet` | cuda | 1 | 1.9144 | 522.4 |
| `small_cnn` | cuda | 1 | 1.4222 | 703.1 |
| `torchvision_mobilenet` | cuda | 1 | 10.2510 | 97.6 |
| `ultra_tiny` | cuda | 32 | 0.0455 | 22000.7 |
| `ds_cnn` | cuda | 32 | 0.2127 | 4702.5 |
| `tiny_cnn` | cuda | 32 | 0.2069 | 4832.6 |
| `mobilenet_like` | cuda | 32 | 0.2303 | 4341.6 |
| `tiny_resnet` | cuda | 32 | 0.1790 | 5586.3 |
| `small_cnn` | cuda | 32 | 0.2325 | 4301.2 |
| `torchvision_mobilenet` | cuda | 32 | 0.2960 | 3378.1 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **26.7%** (0.1245 ms) del tiempo por etapas (E2E=0.4435 ms, gap de despacho Python/ctypes=+0.0233 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.3395 ms**, F1=0.3704), mientras el maximo F1 lo obtiene `tiny_resnet` (**F1=0.4737**, 4.4600 ms).

## Nota Metodologica (Variante Invertida — Short Descriptor Check)

Siguiendo el protocolo §10, las variantes invertidas ejecutan comparacion CPU completa (Fase A) y esta verificacion corta del descriptor x94 para confirmar empiricamente que la inversion fotometrica (`255 - x`) conserva el mismo perfil computacional (E2E mediana batch=1 = **0.4435 ms**, staged sum = **0.4669 ms**).
