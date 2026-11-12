# Profiling Cientifico y Comparacion 100% CPU: `kolektorsdd2`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `486c111`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.4265 | 0.8061 | 0.3551 (desc) | 0.0077 | **0.3629** | 0.1746 | [0.3556, 0.3875] | 2756.0 | - |
| `dt (x94)` | ml | cpu | 285 / 94 | 0.2798 | 0.5962 | 0.3551 (desc) | 0.1329 | **0.4759** | 0.2011 | [0.4396, 0.5114] | 2101.1 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.4265 | 0.8061 | 0.3551 (desc) | 0.2220 | **0.6018** | 0.2012 | [0.5892, 0.6269] | 1661.7 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.6304 | 0.8815 | 0.0436 (prep) | 1.2692 | **1.3211** | 0.6098 | [1.1849, 1.4127] | 757.0 | 2.85e-07 |
| `knn (x94)` | ml | cpu | 219114 / 94 | 0.1500 | 0.6652 | 0.3551 (desc) | 1.1180 | **1.4735** | 0.7277 | [1.3323, 1.6039] | 678.6 | - |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.6562 | 0.8779 | 0.0436 (prep) | 2.2927 | **2.3431** | 0.7377 | [2.2977, 2.4660] | 426.8 | 2.38e-07 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.6250 | 0.8622 | 0.0436 (prep) | 3.9241 | **3.9680** | 1.7831 | [3.7243, 4.1679] | 252.0 | 3.61e-07 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.7130 | 0.9213 | 0.0436 (prep) | 4.8677 | **4.9185** | 1.7588 | [4.6846, 5.1059] | 203.3 | 4.78e-07 |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.6154 | 0.8714 | 0.0436 (prep) | 5.1099 | **5.1567** | 2.0476 | [4.9356, 5.7975] | 193.9 | 3.85e-07 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.5965 | 0.8927 | 0.0436 (prep) | 7.9342 | **7.9777** | 2.3923 | [7.7564, 8.3015] | 125.4 | 3.41e-07 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.5976 | 0.7332 | 0.0436 (prep) | 9.7691 | **9.8125** | 2.3422 | [9.3012, 10.0767] | 101.9 | 3.55e-07 |
| `rf (x94)` | ml | cpu | 13208 / 94 | 0.4298 | 0.7904 | 0.3551 (desc) | 10.3708 | **10.7286** | 2.7330 | [10.4938, 10.9499] | 93.2 | - |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **7.3973 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2427 ms** vs llamada formal E2E: **0.2054 ms** (gap de dispatch/asignaciones Python = **+0.0373 ms**, +15.4%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0677 | 27.9% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.0571 | 23.5% | 2.1 | `O(g^2*K) [16x3]` |
| `asymmetry_g8` | 8x8 | 0.0558 | 23.0% | 2.6 | `O(g^2*K) [64x3]` |
| `c_fine_hist` | 8x8 | 0.0330 | 13.6% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0114 | 4.7% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 4. Desglose Medido por Capas CNN en CPU (`threads=1, batch=1`)

| Arquitectura | Params | Act Peak (KB) | FLOPs (PyTorch) | Hookless (ms) | Hook overhead (%) | Cold 1st (ms) | Familias dominantes (% hooks) |
|---|---|---|---|---|---|---|---|
| `ultra_tiny` | 1,154 | 128.0 | 1,335,424 | 1.4429 | -3.9% | 2.108 | dwconv:40.4%, bn:21.3%, conv:13.6% |
| `ds_cnn` | 12,866 | 1024.0 | 19,366,400 | 2.4239 | +4.5% | 3.345 | dwconv:39.2%, bn:21.0%, pwconv:15.3% |
| `tiny_cnn` | 23,538 | 1024.0 | 80,216,320 | 5.9139 | +1.2% | 7.221 | pool:54.6%, conv:34.0%, bn:6.9% |
| `mobilenet_like` | 70,786 | 512.0 | 38,716,416 | 4.4533 | +12.9% | 3.983 | dwconv:42.4%, pwconv:21.7%, bn:20.0% |
| `tiny_resnet` | 77,234 | 256.0 | 97,648,896 | 4.9877 | +5.7% | 3.768 | conv:69.0%, bn:12.8%, pwconv:11.4% |
| `small_cnn` | 97,650 | 1024.0 | 117,965,312 | 8.0530 | -3.0% | 9.336 | pool:48.9%, conv:39.5%, bn:7.4% |
| `torchvision_mobilenet` | 1,519,618 | 288.0 | 35,293,184 | 7.7399 | +61.3% | 14.083 | dwconv:34.0%, pwconv:25.4%, bn:19.8% |

## 5. Pruebas de Significancia Estadistica (Mann-Whitney U, 1 hilo, batch=1)

| Comparacion | Mediana A (ms) | Mediana B (ms) | Ratio A/B | Estadistico U | p-valor | Significativo |
|---|---|---|---|---|---|---|
| `dt (x94)` vs `ds_cnn` | 0.4759 | 2.3431 | 0.203x | 0.0 | `4.831e-67` | True |
| `dt (x94)` vs `ultra_tiny` | 0.4759 | 1.3211 | 0.360x | 0.0 | `4.831e-67` | True |
| `lr (x94)` vs `ds_cnn` | 0.6018 | 2.3431 | 0.257x | 0.0 | `4.830e-67` | True |
| `lr (x94)` vs `ultra_tiny` | 0.6018 | 1.3211 | 0.456x | 152.0 | `4.690e-66` | True |
| `lr_dot (x94)` vs `ds_cnn` | 0.3629 | 2.3431 | 0.155x | 0.0 | `4.830e-67` | True |
| `lr_dot (x94)` vs `ultra_tiny` | 0.3629 | 1.3211 | 0.275x | 0.0 | `4.830e-67` | True |
| `rf (x94)` vs `ds_cnn` | 10.7286 | 2.3431 | 4.579x | 40000.0 | `4.831e-67` | True |
| `rf (x94)` vs `ultra_tiny` | 10.7286 | 1.3211 | 8.121x | 40000.0 | `4.831e-67` | True |

## 6. Ablacion Calidad vs Coste (`x85` vs `x9` vs `x94`)

| Clasificador | Subconjunto | Features | F1 | ROC-AUC | T_total (ms) | dF1 vs x85 | d_ms vs x85 | dF1 / d_ms |
|---|---|---|---|---|---|---|---|---|
| `dt` | `x85_only` | 85 | 0.3550 | 0.6117 | 0.1066 | +0.0000 | +0.0000 | +0.0000 |
| `dt` | `x9_only` | 9 | 0.2326 | 0.5787 | 0.2829 | -0.1224 | +0.1763 | -0.6944 |
| `dt` | `x94_full` | 94 | 0.2798 | 0.5962 | 0.3722 | -0.0751 | +0.2656 | -0.2829 |
| `knn` | `x85_only` | 85 | 0.1513 | 0.6442 | 1.2647 | +0.0000 | +0.0000 | +0.0000 |
| `knn` | `x9_only` | 9 | 0.2208 | 0.6328 | 1.1931 | +0.0695 | -0.0716 | -0.9716 |
| `knn` | `x94_full` | 94 | 0.1500 | 0.6652 | 1.6008 | -0.0013 | +0.3361 | -0.0038 |
| `lr` | `x85_only` | 85 | 0.2687 | 0.6179 | 0.2805 | +0.0000 | +0.0000 | +0.0000 |
| `lr` | `x9_only` | 9 | 0.3591 | 0.7371 | 0.4353 | +0.0905 | +0.1548 | +0.5844 |
| `lr` | `x94_full` | 94 | 0.4265 | 0.8061 | 0.5451 | +0.1579 | +0.2646 | +0.5965 |
| `lr_dot` | `x85_only` | 85 | 0.2687 | 0.6179 | 0.0326 | +0.0000 | +0.0000 | +0.0000 |
| `lr_dot` | `x9_only` | 9 | 0.3591 | 0.7371 | 0.2096 | +0.0905 | +0.1770 | +0.5111 |
| `lr_dot` | `x94_full` | 94 | 0.4265 | 0.8061 | 0.2923 | +0.1579 | +0.2597 | +0.6078 |
| `rf` | `x85_only` | 85 | 0.4286 | 0.8255 | 10.5590 | +0.0000 | +0.0000 | +0.0000 |
| `rf` | `x9_only` | 9 | 0.3765 | 0.7429 | 10.5923 | -0.0521 | +0.0333 | -1.5669 |
| `rf` | `x94_full` | 94 | 0.4298 | 0.7904 | 10.0897 | +0.0013 | -0.4694 | -0.0027 |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.1172 | 895.1 |
| `ds_cnn` | cuda | 1 | 1.8880 | 529.6 |
| `tiny_cnn` | cuda | 1 | 1.2685 | 788.3 |
| `mobilenet_like` | cuda | 1 | 2.8105 | 355.8 |
| `tiny_resnet` | cuda | 1 | 2.3216 | 430.7 |
| `small_cnn` | cuda | 1 | 1.5205 | 657.7 |
| `torchvision_mobilenet` | cuda | 1 | 11.0107 | 90.8 |
| `ultra_tiny` | cuda | 32 | 0.0383 | 26142.7 |
| `ds_cnn` | cuda | 32 | 0.2125 | 4705.4 |
| `tiny_cnn` | cuda | 32 | 0.2101 | 4760.0 |
| `mobilenet_like` | cuda | 32 | 0.2356 | 4245.2 |
| `tiny_resnet` | cuda | 32 | 0.1810 | 5525.0 |
| `small_cnn` | cuda | 32 | 0.2547 | 3926.9 |
| `torchvision_mobilenet` | cuda | 32 | 0.3238 | 3087.9 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **27.9%** (0.0677 ms) del tiempo por etapas (E2E=0.2054 ms, gap de despacho Python/ctypes=+0.0373 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.3629 ms**, F1=0.4265), mientras el maximo F1 lo obtiene `tiny_resnet` (**F1=0.7130**, 4.9185 ms).
