# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad-inverted_leather`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `99bc026`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.7500 | 0.9296 | 0.4503 (desc) | 0.0079 | **0.4585** | 0.1006 | [0.4479, 0.4669] | 2181.3 | - |
| `mlp_16_dot (x94)` | ml | cpu | 1537 / 94 | 0.8000 | 0.9603 | 0.4503 (desc) | 0.0130 | **0.4634** | 0.1006 | [0.4523, 0.4719] | 2158.0 | - |
| `dt (x94)` | ml | cpu | 15 / 94 | 0.7143 | 0.8343 | 0.4503 (desc) | 0.1850 | **0.6266** | 0.1441 | [0.6144, 0.6576] | 1595.9 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.7500 | 0.9296 | 0.4503 (desc) | 0.2306 | **0.7032** | 0.1298 | [0.6875, 0.7164] | 1422.1 | - |
| `knn (x94)` | ml | cpu | 27730 / 94 | 0.6897 | 0.9420 | 0.4503 (desc) | 0.9950 | **1.4369** | 0.7222 | [1.3809, 1.5465] | 695.9 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.4783 | 0.7599 | 0.0566 (prep) | 1.9268 | **1.9833** | 1.4200 | [1.8130, 2.1863] | 504.2 | 0.00e+00 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.7027 | 0.9157 | 0.0566 (prep) | 4.4054 | **4.4625** | 1.5071 | [4.2794, 4.6964] | 224.1 | 0.00e+00 |
| `rf (x94)` | ml | cpu | 3308 / 94 | 0.7429 | 0.9345 | 0.4503 (desc) | 5.4742 | **5.9455** | 1.8073 | [5.6633, 6.1402] | 168.2 | - |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.3913 | 0.7232 | 0.0566 (prep) | 5.9067 | **5.9637** | 2.1020 | [5.6925, 6.3444] | 167.7 | 0.00e+00 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.4583 | 0.7391 | 0.0566 (prep) | 7.9825 | **8.0390** | 2.3713 | [7.6030, 8.3309] | 124.4 | 0.00e+00 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.7895 | 0.9514 | 0.0566 (prep) | 9.2300 | **9.2862** | 2.1378 | [9.0016, 9.5504] | 107.7 | 0.00e+00 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.3913 | 0.6086 | 0.0566 (prep) | 9.2680 | **9.3256** | 2.1696 | [9.0440, 9.6633] | 107.2 | 3.57e-02 |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.6667 | 0.8661 | 0.0566 (prep) | 10.3023 | **10.3588** | 5.5410 | [9.5695, 10.7823] | 96.5 | 0.00e+00 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **8.1701 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2366 ms** vs llamada formal E2E: **0.3571 ms** (gap de dispatch/asignaciones Python = **-0.1204 ms**, -50.9%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0630 | 26.6% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.0572 | 24.2% | 2.1 | `O(g^2*K) [16x3]` |
| `asymmetry_g8` | 8x8 | 0.0567 | 24.0% | 2.6 | `O(g^2*K) [64x3]` |
| `c_fine_hist` | 8x8 | 0.0300 | 12.7% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0121 | 5.1% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.1564 | 864.7 |
| `ds_cnn` | cuda | 1 | 1.6062 | 622.6 |
| `tiny_cnn` | cuda | 1 | 0.7743 | 1291.4 |
| `mobilenet_like` | cuda | 1 | 2.4190 | 413.4 |
| `tiny_resnet` | cuda | 1 | 2.1926 | 456.1 |
| `small_cnn` | cuda | 1 | 1.0033 | 996.7 |
| `torchvision_mobilenet` | cuda | 1 | 10.4454 | 95.7 |
| `ultra_tiny` | cuda | 32 | 0.0436 | 22951.4 |
| `ds_cnn` | cuda | 32 | 0.2075 | 4819.7 |
| `tiny_cnn` | cuda | 32 | 0.2072 | 4827.1 |
| `mobilenet_like` | cuda | 32 | 0.2305 | 4338.7 |
| `tiny_resnet` | cuda | 32 | 0.1782 | 5612.1 |
| `small_cnn` | cuda | 32 | 0.2325 | 4301.9 |
| `torchvision_mobilenet` | cuda | 32 | 0.3009 | 3323.9 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **26.6%** (0.0630 ms) del tiempo por etapas (E2E=0.3571 ms, gap de despacho Python/ctypes=-0.1204 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.4585 ms**, F1=0.7500), mientras el maximo F1 lo obtiene `mlp_16_dot (x94)` (**F1=0.8000**, 0.4634 ms).

## Nota Metodologica (Variante Invertida — Short Descriptor Check)

Siguiendo el protocolo §10, las variantes invertidas ejecutan comparacion CPU completa (Fase A) y esta verificacion corta del descriptor x94 para confirmar empiricamente que la inversion fotometrica (`255 - x`) conserva el mismo perfil computacional (E2E mediana batch=1 = **0.3571 ms**, staged sum = **0.2366 ms**).
