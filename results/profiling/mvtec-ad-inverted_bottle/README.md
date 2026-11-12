# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad-inverted_bottle`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `37d8114`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.8000 | 0.9214 | 0.2756 (desc) | 0.0073 | **0.2828** | 0.1644 | [0.2606, 0.3527] | 3535.4 | - |
| `dt (x94)` | ml | cpu | 15 / 94 | 0.5833 | 0.6614 | 0.2756 (desc) | 0.1245 | **0.4266** | 0.1613 | [0.3866, 0.4632] | 2343.8 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.8000 | 0.9214 | 0.2756 (desc) | 0.2197 | **0.5411** | 0.1763 | [0.4996, 0.5682] | 1848.1 | - |
| `knn (x94)` | ml | cpu | 21902 / 94 | 0.3750 | 0.7057 | 0.2756 (desc) | 0.5927 | **0.9196** | 0.2466 | [0.8913, 0.9553] | 1087.4 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.4138 | 0.5987 | 0.0430 (prep) | 0.8944 | **0.9514** | 0.1890 | [0.9277, 0.9931] | 1051.1 | 0.00e+00 |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.6190 | 0.9365 | 0.0430 (prep) | 2.9693 | **3.0147** | 0.6904 | [2.9215, 3.0543] | 331.7 | 0.00e+00 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.5625 | 0.8127 | 0.0430 (prep) | 3.9857 | **4.0286** | 1.4375 | [3.9081, 4.2934] | 248.2 | 0.00e+00 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.3611 | 0.6739 | 0.0430 (prep) | 5.1150 | **5.1596** | 2.5462 | [4.8951, 5.4655] | 193.8 | 6.69e-03 |
| `rf (x94)` | ml | cpu | 3506 / 94 | 0.7273 | 0.9013 | 0.2756 (desc) | 5.0683 | **5.4054** | 0.4730 | [5.3276, 5.4753] | 185.0 | - |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.3636 | 0.6187 | 0.0430 (prep) | 5.5308 | **5.5746** | 0.5314 | [5.5206, 5.6395] | 179.4 | 0.00e+00 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.8000 | 0.9916 | 0.0430 (prep) | 7.1322 | **7.1799** | 0.8475 | [7.0651, 7.2633] | 139.3 | 0.00e+00 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.3611 | 0.5109 | 0.0430 (prep) | 8.1885 | **8.2355** | 1.1258 | [8.0802, 8.3963] | 121.4 | 5.69e-02 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **9.4542 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2235 ms** vs llamada formal E2E: **0.2085 ms** (gap de dispatch/asignaciones Python = **+0.0150 ms**, +6.7%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0590 | 26.4% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.0529 | 23.7% | 2.1 | `O(g^2*K) [16x3]` |
| `asymmetry_g8` | 8x8 | 0.0529 | 23.7% | 2.6 | `O(g^2*K) [64x3]` |
| `c_fine_hist` | 8x8 | 0.0298 | 13.3% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0120 | 5.3% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.2489 | 800.7 |
| `ds_cnn` | cuda | 1 | 1.6564 | 603.7 |
| `tiny_cnn` | cuda | 1 | 0.9397 | 1064.1 |
| `mobilenet_like` | cuda | 1 | 2.3925 | 418.0 |
| `tiny_resnet` | cuda | 1 | 2.0185 | 495.4 |
| `small_cnn` | cuda | 1 | 1.1575 | 864.0 |
| `torchvision_mobilenet` | cuda | 1 | 10.2299 | 97.8 |
| `ultra_tiny` | cuda | 32 | 0.0439 | 22795.3 |
| `ds_cnn` | cuda | 32 | 0.2021 | 4948.6 |
| `tiny_cnn` | cuda | 32 | 0.2080 | 4808.7 |
| `mobilenet_like` | cuda | 32 | 0.2322 | 4307.1 |
| `tiny_resnet` | cuda | 32 | 0.1807 | 5535.2 |
| `small_cnn` | cuda | 32 | 0.2316 | 4318.5 |
| `torchvision_mobilenet` | cuda | 32 | 0.3359 | 2977.3 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **26.4%** (0.0590 ms) del tiempo por etapas (E2E=0.2085 ms, gap de despacho Python/ctypes=+0.0150 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.2828 ms**, F1=0.8000), mientras el maximo F1 lo obtiene `lr (x94)` (**F1=0.8000**, 0.5411 ms).

## Nota Metodologica (Variante Invertida — Short Descriptor Check)

Siguiendo el protocolo §10, las variantes invertidas ejecutan comparacion CPU completa (Fase A) y esta verificacion corta del descriptor x94 para confirmar empiricamente que la inversion fotometrica (`255 - x`) conserva el mismo perfil computacional (E2E mediana batch=1 = **0.2085 ms**, staged sum = **0.2235 ms**).
