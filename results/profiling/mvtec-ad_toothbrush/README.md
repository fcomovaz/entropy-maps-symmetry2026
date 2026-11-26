# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad_toothbrush`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `99bc026`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.8000 | 0.9333 | 0.2393 (desc) | 0.0068 | **0.2460** | 0.0389 | [0.2126, 0.2464] | 4065.0 | - |
| `mlp_16_dot (x94)` | ml | cpu | 1537 / 94 | 0.6667 | 0.9000 | 0.2393 (desc) | 0.0118 | **0.2509** | 0.0382 | [0.2208, 0.2513] | 3985.7 | - |
| `dt (x94)` | ml | cpu | 9 / 94 | 0.3750 | 0.4944 | 0.2393 (desc) | 0.0974 | **0.3366** | 0.0495 | [0.3012, 0.3372] | 2970.9 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.8000 | 0.9333 | 0.2393 (desc) | 0.2341 | **0.4574** | 0.0563 | [0.4559, 0.4705] | 2186.3 | - |
| `knn (x94)` | ml | cpu | 7614 / 94 | 0.2857 | 0.8333 | 0.2393 (desc) | 0.4367 | **0.6757** | 0.0608 | [0.6745, 0.6769] | 1479.9 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.0000 | 0.4222 | 0.0433 (prep) | 0.7928 | **0.8362** | 0.1988 | [0.8162, 0.8448] | 1195.8 | 3.33e-07 |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.4000 | 0.6778 | 0.0433 (prep) | 1.9518 | **1.9949** | 0.3661 | [1.9653, 2.0513] | 501.3 | 3.64e-07 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.4444 | 0.4500 | 0.0433 (prep) | 3.0042 | **3.0475** | 0.9713 | [2.9533, 3.2395] | 328.1 | 5.56e-03 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.6000 | 0.7667 | 0.0433 (prep) | 3.2783 | **3.3214** | 0.6871 | [3.2776, 3.4112] | 301.1 | 3.33e-07 |
| `rf (x94)` | ml | cpu | 1508 / 94 | 0.2857 | 0.9889 | 0.2393 (desc) | 4.0373 | **4.2682** | 0.5500 | [4.1868, 4.3096] | 234.3 | - |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.4444 | 0.7889 | 0.0433 (prep) | 4.5146 | **4.5579** | 0.7181 | [4.5016, 4.6236] | 219.4 | 4.44e-07 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.4444 | 0.7778 | 0.0433 (prep) | 6.1110 | **6.1544** | 1.3689 | [5.9474, 6.3988] | 162.5 | 4.44e-07 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.4444 | 0.5000 | 0.0433 (prep) | 6.3905 | **6.4337** | 1.0242 | [6.3721, 6.5782] | 155.4 | 4.44e-07 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **6.8231 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2157 ms** vs llamada formal E2E: **0.1993 ms** (gap de dispatch/asignaciones Python = **+0.0164 ms**, +7.6%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0601 | 27.9% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.0503 | 23.3% | 2.1 | `O(g^2*K) [16x3]` |
| `asymmetry_g8` | 8x8 | 0.0492 | 22.8% | 2.6 | `O(g^2*K) [64x3]` |
| `c_fine_hist` | 8x8 | 0.0294 | 13.6% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0104 | 4.8% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 4. Desglose Medido por Capas CNN en CPU (`threads=1, batch=1`)

| Arquitectura | Params | Act Peak (KB) | FLOPs (PyTorch) | Hookless (ms) | Hook overhead (%) | Cold 1st (ms) | Familias dominantes (% hooks) |
|---|---|---|---|---|---|---|---|
| `ultra_tiny` | 1,154 | 128.0 | 1,335,424 | 0.9099 | +18.5% | 3.177 | dwconv:39.6%, bn:21.8%, conv:15.1% |
| `ds_cnn` | 12,866 | 1024.0 | 19,366,400 | 2.5835 | -14.0% | 6.290 | dwconv:37.2%, bn:21.4%, pwconv:16.0% |
| `tiny_cnn` | 23,538 | 1024.0 | 80,216,320 | 5.1068 | -2.0% | 6.536 | pool:60.1%, conv:30.8%, bn:5.5% |
| `mobilenet_like` | 70,786 | 512.0 | 38,716,416 | 3.4905 | +7.6% | 6.800 | dwconv:41.3%, pwconv:22.0%, bn:20.3% |
| `tiny_resnet` | 77,234 | 256.0 | 97,648,896 | 3.3710 | +39.6% | 7.342 | conv:71.0%, bn:11.8%, pwconv:11.2% |
| `small_cnn` | 97,650 | 1024.0 | 117,965,312 | 6.0819 | +3.3% | 9.131 | pool:56.4%, conv:35.2%, bn:5.3% |
| `torchvision_mobilenet` | 1,519,618 | 288.0 | 35,293,184 | 7.0445 | +20.9% | 17.426 | dwconv:31.2%, pwconv:27.6%, bn:20.6% |

## 5. Pruebas de Significancia Estadistica (Mann-Whitney U, 1 hilo, batch=1)

| Comparacion | Mediana A (ms) | Mediana B (ms) | Ratio A/B | Estadistico U | p-valor | Significativo |
|---|---|---|---|---|---|---|
| `dt (x94)` vs `ds_cnn` | 0.3366 | 1.9949 | 0.169x | 0.0 | `4.822e-67` | True |
| `dt (x94)` vs `ultra_tiny` | 0.3366 | 0.8362 | 0.403x | 0.0 | `4.821e-67` | True |
| `lr (x94)` vs `ds_cnn` | 0.4574 | 1.9949 | 0.229x | 0.0 | `4.829e-67` | True |
| `lr (x94)` vs `ultra_tiny` | 0.4574 | 0.8362 | 0.547x | 173.0 | `6.410e-66` | True |
| `lr_dot (x94)` vs `ds_cnn` | 0.2460 | 1.9949 | 0.123x | 0.0 | `4.817e-67` | True |
| `lr_dot (x94)` vs `ultra_tiny` | 0.2460 | 0.8362 | 0.294x | 0.0 | `4.816e-67` | True |
| `mlp_16_dot (x94)` vs `ds_cnn` | 0.2509 | 1.9949 | 0.126x | 0.0 | `4.810e-67` | True |
| `mlp_16_dot (x94)` vs `ultra_tiny` | 0.2509 | 0.8362 | 0.300x | 0.0 | `4.809e-67` | True |
| `rf (x94)` vs `ds_cnn` | 4.2682 | 1.9949 | 2.139x | 39923.0 | `1.531e-66` | True |
| `rf (x94)` vs `ultra_tiny` | 4.2682 | 0.8362 | 5.104x | 40000.0 | `4.829e-67` | True |

## 6. Ablacion Calidad vs Coste (`x85` vs `x9` vs `x94`)

| Clasificador | Subconjunto | Features | F1 | ROC-AUC | T_total (ms) | dF1 vs x85 | d_ms vs x85 | dF1 / d_ms |
|---|---|---|---|---|---|---|---|---|
| `dt` | `x85_only` | 85 | 0.2857 | 0.4667 | 0.0996 | +0.0000 | +0.0000 | +0.0000 |
| `dt` | `x9_only` | 9 | 0.5333 | 0.7111 | 0.2722 | +0.2476 | +0.1726 | +1.4342 |
| `dt` | `x94_full` | 94 | 0.3750 | 0.4944 | 0.2704 | +0.0893 | +0.1708 | +0.5228 |
| `knn` | `x85_only` | 85 | 0.2857 | 0.8333 | 0.4831 | +0.0000 | +0.0000 | +0.0000 |
| `knn` | `x9_only` | 9 | 0.6000 | 0.9111 | 0.7659 | +0.3143 | +0.2828 | +1.1115 |
| `knn` | `x94_full` | 94 | 0.2857 | 0.8333 | 0.6512 | +0.0000 | +0.1681 | +0.0000 |
| `lr` | `x85_only` | 85 | 0.8000 | 0.9333 | 0.2241 | +0.0000 | +0.0000 | +0.0000 |
| `lr` | `x9_only` | 9 | 0.3750 | 0.6444 | 0.3822 | -0.4250 | +0.1581 | -2.6873 |
| `lr` | `x94_full` | 94 | 0.8000 | 0.9333 | 0.3968 | +0.0000 | +0.1727 | +0.0000 |
| `lr_dot` | `x85_only` | 85 | 0.8000 | 0.9333 | 0.0329 | +0.0000 | +0.0000 | +0.0000 |
| `lr_dot` | `x9_only` | 9 | 0.3750 | 0.6444 | 0.1911 | -0.4250 | +0.1583 | -2.6856 |
| `lr_dot` | `x94_full` | 94 | 0.8000 | 0.9333 | 0.2034 | +0.0000 | +0.1705 | +0.0000 |
| `mlp_16_dot` | `x85_only` | 85 | 0.6667 | 0.9111 | 0.0359 | +0.0000 | +0.0000 | +0.0000 |
| `mlp_16_dot` | `x9_only` | 9 | 0.5556 | 0.8111 | 0.1993 | -0.1111 | +0.1634 | -0.6798 |
| `mlp_16_dot` | `x94_full` | 94 | 0.6667 | 0.9000 | 0.2080 | +0.0000 | +0.1721 | +0.0000 |
| `rf` | `x85_only` | 85 | 0.2857 | 0.9667 | 4.1910 | +0.0000 | +0.0000 | +0.0000 |
| `rf` | `x9_only` | 9 | 0.8333 | 0.9000 | 4.4297 | +0.5476 | +0.2387 | +2.2942 |
| `rf` | `x94_full` | 94 | 0.2857 | 0.9889 | 4.3713 | +0.0000 | +0.1803 | +0.0000 |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.2326 | 811.3 |
| `ds_cnn` | cuda | 1 | 1.2655 | 790.2 |
| `tiny_cnn` | cuda | 1 | 0.9510 | 1051.5 |
| `mobilenet_like` | cuda | 1 | 2.8275 | 353.7 |
| `tiny_resnet` | cuda | 1 | 1.6831 | 594.1 |
| `small_cnn` | cuda | 1 | 0.9324 | 1072.6 |
| `torchvision_mobilenet` | cuda | 1 | 10.6205 | 94.2 |
| `ultra_tiny` | cuda | 32 | 0.0385 | 25944.5 |
| `ds_cnn` | cuda | 32 | 0.2054 | 4869.1 |
| `tiny_cnn` | cuda | 32 | 0.2059 | 4856.3 |
| `mobilenet_like` | cuda | 32 | 0.2296 | 4354.7 |
| `tiny_resnet` | cuda | 32 | 0.1781 | 5614.2 |
| `small_cnn` | cuda | 32 | 0.2319 | 4311.3 |
| `torchvision_mobilenet` | cuda | 32 | 0.2757 | 3627.3 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **27.9%** (0.0601 ms) del tiempo por etapas (E2E=0.1993 ms, gap de despacho Python/ctypes=+0.0164 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.2460 ms**, F1=0.8000), mientras el maximo F1 lo obtiene `lr (x94)` (**F1=0.8000**, 0.4574 ms).
