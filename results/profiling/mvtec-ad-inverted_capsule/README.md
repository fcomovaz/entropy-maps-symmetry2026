# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad-inverted_capsule`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `99bc026`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.7600 | 0.9091 | 0.3656 (desc) | 0.0074 | **0.3730** | 0.2301 | [0.3662, 0.3937] | 2680.6 | - |
| `mlp_16_dot (x94)` | ml | cpu | 1537 / 94 | 0.8696 | 0.9369 | 0.3656 (desc) | 0.0123 | **0.3783** | 0.2301 | [0.3711, 0.3987] | 2643.4 | - |
| `dt (x94)` | ml | cpu | 15 / 94 | 0.6667 | 0.7175 | 0.3656 (desc) | 0.1168 | **0.4956** | 0.2147 | [0.4757, 0.5085] | 2017.8 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.7600 | 0.9091 | 0.3656 (desc) | 0.2450 | **0.6288** | 0.2665 | [0.5923, 0.6590] | 1590.5 | - |
| `knn (x94)` | ml | cpu | 26320 / 94 | 0.6286 | 0.8711 | 0.3656 (desc) | 0.5478 | **0.9423** | 0.2605 | [0.9143, 0.9779] | 1061.3 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.4731 | 0.5603 | 0.0445 (prep) | 1.5650 | **1.6084** | 1.5035 | [1.3148, 1.7358] | 621.7 | 0.00e+00 |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.6129 | 0.8655 | 0.0445 (prep) | 3.7275 | **3.7854** | 3.8693 | [3.2645, 4.1721] | 264.2 | 0.00e+00 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.6190 | 0.8182 | 0.0445 (prep) | 4.2268 | **4.2908** | 0.7092 | [4.2029, 4.4021] | 233.1 | 0.00e+00 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.6818 | 0.8701 | 0.0445 (prep) | 4.2571 | **4.3143** | 1.6141 | [4.0827, 4.5464] | 231.8 | 0.00e+00 |
| `rf (x94)` | ml | cpu | 5916 / 94 | 0.7027 | 0.8998 | 0.3656 (desc) | 5.9939 | **6.2924** | 2.3920 | [6.0009, 6.6454] | 158.9 | - |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.5316 | 0.8228 | 0.0445 (prep) | 6.8926 | **6.9360** | 2.7890 | [6.6231, 7.3328] | 144.2 | 0.00e+00 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.8500 | 0.9620 | 0.0445 (prep) | 7.6534 | **7.7138** | 2.2098 | [7.3115, 7.9904] | 129.6 | 0.00e+00 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.0000 | 0.4499 | 0.0445 (prep) | 8.6096 | **8.6693** | 2.6794 | [8.2970, 9.0850] | 115.3 | 4.64e-03 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **11.5034 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2504 ms** vs llamada formal E2E: **0.2273 ms** (gap de dispatch/asignaciones Python = **+0.0231 ms**, +9.2%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0665 | 26.6% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g8` | 8x8 | 0.0595 | 23.8% | 2.6 | `O(g^2*K) [64x3]` |
| `asymmetry_g4` | 4x4 | 0.0589 | 23.5% | 2.1 | `O(g^2*K) [16x3]` |
| `c_fine_hist` | 8x8 | 0.0318 | 12.7% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0140 | 5.6% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.5911 | 628.5 |
| `ds_cnn` | cuda | 1 | 2.2389 | 446.6 |
| `tiny_cnn` | cuda | 1 | 0.8287 | 1206.7 |
| `mobilenet_like` | cuda | 1 | 2.7490 | 363.8 |
| `tiny_resnet` | cuda | 1 | 2.9455 | 339.5 |
| `small_cnn` | cuda | 1 | 1.1636 | 859.4 |
| `torchvision_mobilenet` | cuda | 1 | 11.0040 | 90.9 |
| `ultra_tiny` | cuda | 32 | 0.0444 | 22536.8 |
| `ds_cnn` | cuda | 32 | 0.2083 | 4801.7 |
| `tiny_cnn` | cuda | 32 | 0.2105 | 4750.9 |
| `mobilenet_like` | cuda | 32 | 0.2333 | 4285.6 |
| `tiny_resnet` | cuda | 32 | 0.1802 | 5548.0 |
| `small_cnn` | cuda | 32 | 0.2357 | 4242.0 |
| `torchvision_mobilenet` | cuda | 32 | 0.3538 | 2826.5 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **26.6%** (0.0665 ms) del tiempo por etapas (E2E=0.2273 ms, gap de despacho Python/ctypes=+0.0231 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.3730 ms**, F1=0.7600), mientras el maximo F1 lo obtiene `mlp_16_dot (x94)` (**F1=0.8696**, 0.3783 ms).

## Nota Metodologica (Variante Invertida — Short Descriptor Check)

Siguiendo el protocolo §10, las variantes invertidas ejecutan comparacion CPU completa (Fase A) y esta verificacion corta del descriptor x94 para confirmar empiricamente que la inversion fotometrica (`255 - x`) conserva el mismo perfil computacional (E2E mediana batch=1 = **0.2273 ms**, staged sum = **0.2504 ms**).
