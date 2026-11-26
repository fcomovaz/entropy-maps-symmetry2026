# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad-inverted_transistor`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `99bc026`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.5217 | 0.9250 | 0.3509 (desc) | 0.0076 | **0.3589** | 0.1462 | [0.3569, 0.3595] | 2786.3 | - |
| `mlp_16_dot (x94)` | ml | cpu | 1537 / 94 | 0.7368 | 0.9659 | 0.3509 (desc) | 0.0123 | **0.3634** | 0.1535 | [0.3608, 0.3641] | 2752.2 | - |
| `dt (x94)` | ml | cpu | 15 / 94 | 0.3571 | 0.6898 | 0.3509 (desc) | 0.0853 | **0.4477** | 0.1699 | [0.4389, 0.4491] | 2233.6 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.5217 | 0.9250 | 0.3509 (desc) | 0.2213 | **0.5688** | 0.0756 | [0.5665, 0.5707] | 1757.9 | - |
| `knn (x94)` | ml | cpu | 23500 / 94 | 0.0000 | 0.8295 | 0.3509 (desc) | 0.4864 | **0.8357** | 0.1168 | [0.8293, 0.8510] | 1196.6 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.2254 | 0.2955 | 0.0401 (prep) | 0.7783 | **0.8372** | 0.2164 | [0.8217, 0.8492] | 1194.5 | 0.00e+00 |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.2254 | 0.6705 | 0.0401 (prep) | 1.8929 | **1.9327** | 0.7591 | [1.9180, 1.9914] | 517.4 | 0.00e+00 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.6087 | 0.9364 | 0.0401 (prep) | 2.9438 | **2.9884** | 0.8722 | [2.8908, 3.1469] | 334.6 | 0.00e+00 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.7778 | 0.9636 | 0.0401 (prep) | 3.2753 | **3.3204** | 0.8975 | [3.2794, 3.4839] | 301.2 | 0.00e+00 |
| `rf (x94)` | ml | cpu | 3238 / 94 | 0.0000 | 0.9273 | 0.3509 (desc) | 4.0888 | **4.4215** | 0.7166 | [4.3806, 4.5115] | 226.2 | - |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.6667 | 0.9227 | 0.0401 (prep) | 4.9532 | **4.9912** | 1.1695 | [4.6932, 5.1737] | 200.4 | 0.00e+00 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.0000 | 0.9273 | 0.0401 (prep) | 6.0957 | **6.1349** | 1.2545 | [6.0331, 6.3732] | 163.0 | 0.00e+00 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.2254 | 0.4398 | 0.0401 (prep) | 6.8542 | **6.8940** | 1.3595 | [6.7667, 7.0540] | 145.1 | 6.78e-03 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **10.8811 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2439 ms** vs llamada formal E2E: **0.4034 ms** (gap de dispatch/asignaciones Python = **-0.1595 ms**, -65.4%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0649 | 26.6% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g8` | 8x8 | 0.0588 | 24.1% | 2.6 | `O(g^2*K) [64x3]` |
| `asymmetry_g4` | 4x4 | 0.0587 | 24.0% | 2.1 | `O(g^2*K) [16x3]` |
| `c_fine_hist` | 8x8 | 0.0304 | 12.5% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0122 | 5.0% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.1929 | 838.3 |
| `ds_cnn` | cuda | 1 | 1.2985 | 770.1 |
| `tiny_cnn` | cuda | 1 | 0.8078 | 1237.9 |
| `mobilenet_like` | cuda | 1 | 2.2133 | 451.8 |
| `tiny_resnet` | cuda | 1 | 1.7126 | 583.9 |
| `small_cnn` | cuda | 1 | 0.9456 | 1057.6 |
| `torchvision_mobilenet` | cuda | 1 | 9.1054 | 109.8 |
| `ultra_tiny` | cuda | 32 | 0.0382 | 26180.2 |
| `ds_cnn` | cuda | 32 | 0.2072 | 4826.8 |
| `tiny_cnn` | cuda | 32 | 0.2055 | 4865.2 |
| `mobilenet_like` | cuda | 32 | 0.2271 | 4402.4 |
| `tiny_resnet` | cuda | 32 | 0.1773 | 5639.4 |
| `small_cnn` | cuda | 32 | 0.2312 | 4324.7 |
| `torchvision_mobilenet` | cuda | 32 | 0.2792 | 3581.8 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **26.6%** (0.0649 ms) del tiempo por etapas (E2E=0.4034 ms, gap de despacho Python/ctypes=-0.1595 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.3589 ms**, F1=0.5217), mientras el maximo F1 lo obtiene `tiny_resnet` (**F1=0.7778**, 3.3204 ms).

## Nota Metodologica (Variante Invertida — Short Descriptor Check)

Siguiendo el protocolo §10, las variantes invertidas ejecutan comparacion CPU completa (Fase A) y esta verificacion corta del descriptor x94 para confirmar empiricamente que la inversion fotometrica (`255 - x`) conserva el mismo perfil computacional (E2E mediana batch=1 = **0.4034 ms**, staged sum = **0.2439 ms**).
