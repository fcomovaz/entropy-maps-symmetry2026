# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad-inverted_pill`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `99bc026`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.7742 | 0.9140 | 0.2994 (desc) | 0.0095 | **0.3089** | 0.1835 | [0.2871, 0.3126] | 3237.8 | - |
| `mlp_16_dot (x94)` | ml | cpu | 1537 / 94 | 0.8519 | 0.9516 | 0.2994 (desc) | 0.0168 | **0.3174** | 0.1855 | [0.3063, 0.3306] | 3151.1 | - |
| `dt (x94)` | ml | cpu | 13 / 94 | 0.7143 | 0.8466 | 0.2994 (desc) | 0.1386 | **0.4506** | 0.1720 | [0.4327, 0.4765] | 2219.3 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.7742 | 0.9140 | 0.2994 (desc) | 0.4812 | **0.8273** | 0.3040 | [0.7838, 0.8569] | 1208.8 | - |
| `knn (x94)` | ml | cpu | 32618 / 94 | 0.7660 | 0.9588 | 0.2994 (desc) | 0.7145 | **1.1298** | 0.3136 | [1.1002, 1.1593] | 885.1 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.5538 | 0.7548 | 0.0641 (prep) | 1.1355 | **1.1993** | 0.2792 | [1.1681, 1.2382] | 833.8 | 0.00e+00 |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.6842 | 0.9249 | 0.0641 (prep) | 3.5207 | **3.5816** | 1.5195 | [3.3621, 3.7689] | 279.2 | 0.00e+00 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.4000 | 0.7542 | 0.0641 (prep) | 4.1924 | **4.2674** | 1.0955 | [4.1348, 4.4182] | 234.3 | 0.00e+00 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.6571 | 0.8729 | 0.0641 (prep) | 4.3609 | **4.4223** | 1.5016 | [4.2559, 4.6125] | 226.1 | 0.00e+00 |
| `rf (x94)` | ml | cpu | 6312 / 94 | 0.7826 | 0.9467 | 0.2994 (desc) | 5.4314 | **5.7873** | 1.6767 | [5.5271, 6.1076] | 172.8 | - |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.4870 | 0.5000 | 0.0641 (prep) | 8.1099 | **8.1740** | 2.1083 | [7.9787, 8.3066] | 122.3 | 0.00e+00 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.7937 | 0.9637 | 0.0641 (prep) | 8.1131 | **8.1802** | 2.8613 | [7.8657, 8.5071] | 122.2 | 0.00e+00 |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.6957 | 0.9128 | 0.0641 (prep) | 9.1335 | **9.1987** | 3.4457 | [8.6417, 9.6287] | 108.7 | 0.00e+00 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **7.5084 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2381 ms** vs llamada formal E2E: **0.3669 ms** (gap de dispatch/asignaciones Python = **-0.1288 ms**, -54.1%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0628 | 26.4% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.0575 | 24.1% | 2.1 | `O(g^2*K) [16x3]` |
| `asymmetry_g8` | 8x8 | 0.0572 | 24.0% | 2.6 | `O(g^2*K) [64x3]` |
| `c_fine_hist` | 8x8 | 0.0307 | 12.9% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0120 | 5.1% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.0852 | 921.5 |
| `ds_cnn` | cuda | 1 | 1.2511 | 799.3 |
| `tiny_cnn` | cuda | 1 | 0.7320 | 1366.1 |
| `mobilenet_like` | cuda | 1 | 2.1651 | 461.9 |
| `tiny_resnet` | cuda | 1 | 1.7129 | 583.8 |
| `small_cnn` | cuda | 1 | 0.9480 | 1054.8 |
| `torchvision_mobilenet` | cuda | 1 | 8.7230 | 114.6 |
| `ultra_tiny` | cuda | 32 | 0.0378 | 26427.7 |
| `ds_cnn` | cuda | 32 | 0.2031 | 4924.1 |
| `tiny_cnn` | cuda | 32 | 0.2059 | 4856.5 |
| `mobilenet_like` | cuda | 32 | 0.2295 | 4356.9 |
| `tiny_resnet` | cuda | 32 | 0.1778 | 5625.5 |
| `small_cnn` | cuda | 32 | 0.2312 | 4325.0 |
| `torchvision_mobilenet` | cuda | 32 | 0.2818 | 3548.2 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **26.4%** (0.0628 ms) del tiempo por etapas (E2E=0.3669 ms, gap de despacho Python/ctypes=-0.1288 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.3089 ms**, F1=0.7742), mientras el maximo F1 lo obtiene `mlp_16_dot (x94)` (**F1=0.8519**, 0.3174 ms).

## Nota Metodologica (Variante Invertida — Short Descriptor Check)

Siguiendo el protocolo §10, las variantes invertidas ejecutan comparacion CPU completa (Fase A) y esta verificacion corta del descriptor x94 para confirmar empiricamente que la inversion fotometrica (`255 - x`) conserva el mismo perfil computacional (E2E mediana batch=1 = **0.3669 ms**, staged sum = **0.2381 ms**).
