# Profiling Cientifico y Comparacion 100% CPU: `kolektorsdd2`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `bf4a29f`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.5914 | 0.8793 | 0.0399 (prep) | 0.8494 | **0.8891** | 0.1401 | [0.8715, 0.9154] | 1124.7 | 2.11e-07 |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.6102 | 0.8875 | 0.0399 (prep) | 2.0372 | **2.0805** | 0.2145 | [2.0580, 2.1069] | 480.7 | 4.92e-07 |
| `dt (x94)` | ml | cpu | 285 / 94 | 0.2798 | 0.5962 | 2.4324 (desc) | 0.0757 | **2.5126** | 0.6610 | [2.4271, 2.6227] | 398.0 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.4265 | 0.8061 | 2.4324 (desc) | 0.2165 | **2.7124** | 0.6714 | [2.5805, 2.7838] | 368.7 | - |
| `knn (x94)` | ml | cpu | 219114 / 94 | 0.1500 | 0.6652 | 2.4324 (desc) | 0.8425 | **3.3743** | 0.7769 | [3.2974, 3.5192] | 296.4 | - |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.6224 | 0.8894 | 0.0399 (prep) | 3.3477 | **3.3875** | 0.6113 | [3.3311, 3.4558] | 295.2 | 4.33e-07 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.7389 | 0.9219 | 0.0399 (prep) | 3.5784 | **3.6254** | 0.7541 | [3.5709, 3.7244] | 275.8 | 2.62e-07 |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.5677 | 0.8625 | 0.0399 (prep) | 4.3639 | **4.4114** | 0.2689 | [4.3705, 4.4363] | 226.7 | 4.22e-07 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.6193 | 0.9093 | 0.0399 (prep) | 5.5349 | **5.5807** | 0.2884 | [5.5437, 5.6046] | 179.2 | 3.40e-07 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.6049 | 0.7816 | 0.0399 (prep) | 8.1400 | **8.1839** | 1.9975 | [7.9473, 8.5606] | 122.2 | 9.69e-06 |
| `rf (x94)` | ml | cpu | 13208 / 94 | 0.4298 | 0.7904 | 2.4324 (desc) | 9.5815 | **12.2171** | 2.7413 | [11.9378, 12.7179] | 81.9 | - |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E)

- Inicializacion amortizable (`build_entropy_plan` + LUTs): **0.6253 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **1.2865 ms** vs llamada formal E2E: **2.0918 ms** (gap de dispatch/asignaciones Python = **-0.8053 ms**, -62.6%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `region_histograms_g8` | 8x8 | 0.4827 | 37.5% | 13.4 | `O(H*W) [2*H*W=32768]` |
| `region_histograms_g4` | 4x4 | 0.1963 | 15.3% | 16.3 | `O(H*W) [2*H*W=32768]` |
| `region_histograms_g2` | 2x2 | 0.1159 | 9.0% | 50.8 | `O(H*W) [2*H*W=32768]` |
| `region_histograms_g1` | 1x1 | 0.0963 | 7.5% | 193.2 | `O(H*W) [2*H*W=32768]` |
| `asymmetry_g2` | 2x2 | 0.0680 | 5.3% | 2.1 | `O(g^2*K) [4x3]` |

## 4. Desglose Medido por Capas CNN en CPU (`threads=1, batch=1`)

| Arquitectura | Params | Act Peak (KB) | FLOPs (PyTorch) | Hookless (ms) | Hook overhead (%) | Cold 1st (ms) | Familias dominantes (% hooks) |
|---|---|---|---|---|---|---|---|
| `ultra_tiny` | 1,154 | 128.0 | 1,335,424 | 1.5857 | -18.0% | 1.475 | dwconv:40.5%, bn:20.7%, conv:13.9% |
| `ds_cnn` | 12,866 | 1024.0 | 19,366,400 | 2.3721 | +18.9% | 5.907 | dwconv:37.9%, bn:19.5%, pwconv:15.7% |
| `tiny_cnn` | 23,538 | 1024.0 | 80,216,320 | 4.9356 | -3.7% | 5.683 | pool:57.1%, conv:33.7%, bn:5.4% |
| `mobilenet_like` | 70,786 | 512.0 | 38,716,416 | 3.1098 | +72.0% | 5.109 | dwconv:43.3%, pwconv:21.1%, bn:20.3% |
| `tiny_resnet` | 77,234 | 256.0 | 97,648,896 | 3.4394 | +18.4% | 3.848 | conv:70.2%, pwconv:11.9%, bn:11.5% |
| `small_cnn` | 97,650 | 1024.0 | 117,965,312 | 6.0601 | +21.2% | 6.542 | pool:52.3%, conv:38.1%, bn:6.0% |
| `torchvision_mobilenet` | 1,519,618 | 288.0 | 35,293,184 | 7.4935 | +29.4% | 8.488 | dwconv:31.8%, pwconv:27.1%, bn:20.0% |

## 5. Pruebas de Significancia Estadistica (Mann-Whitney U, 1 hilo, batch=1)

| Comparacion | Mediana A (ms) | Mediana B (ms) | Ratio A/B | Estadistico U | p-valor | Significativo |
|---|---|---|---|---|---|---|
| `dt (x94)` vs `ds_cnn` | 2.5126 | 2.0805 | 1.208x | 36246.0 | `7.545e-45` | True |
| `dt (x94)` vs `ultra_tiny` | 2.5126 | 0.8891 | 2.826x | 39641.0 | `1.008e-64` | True |
| `lr (x94)` vs `ds_cnn` | 2.7124 | 2.0805 | 1.304x | 38199.0 | `7.955e-56` | True |
| `lr (x94)` vs `ultra_tiny` | 2.7124 | 0.8891 | 3.051x | 39875.0 | `3.136e-66` | True |
| `rf (x94)` vs `ds_cnn` | 12.2171 | 2.0805 | 5.872x | 40000.0 | `4.830e-67` | True |
| `rf (x94)` vs `ultra_tiny` | 12.2171 | 0.8891 | 13.740x | 40000.0 | `4.830e-67` | True |

## 6. Ablacion Calidad vs Coste (`x85` vs `x9` vs `x94`)

| Clasificador | Subconjunto | Features | F1 | ROC-AUC | T_total (ms) | dF1 vs x85 | d_ms vs x85 | dF1 / d_ms |
|---|---|---|---|---|---|---|---|---|
| `dt` | `x85_only` | 85 | 0.3550 | 0.6117 | 2.1532 | +0.0000 | +0.0000 | +0.0000 |
| `dt` | `x9_only` | 9 | 0.2326 | 0.5787 | 2.3104 | -0.1224 | +0.1572 | -0.7790 |
| `dt` | `x94_full` | 94 | 0.2798 | 0.5962 | 2.4214 | -0.0751 | +0.2682 | -0.2802 |
| `knn` | `x85_only` | 85 | 0.1513 | 0.6442 | 2.8607 | +0.0000 | +0.0000 | +0.0000 |
| `knn` | `x9_only` | 9 | 0.2208 | 0.6328 | 2.8888 | +0.0695 | +0.0281 | +2.4784 |
| `knn` | `x94_full` | 94 | 0.1500 | 0.6652 | 3.1249 | -0.0013 | +0.2642 | -0.0048 |
| `lr` | `x85_only` | 85 | 0.2687 | 0.6179 | 2.2955 | +0.0000 | +0.0000 | +0.0000 |
| `lr` | `x9_only` | 9 | 0.3591 | 0.7371 | 2.4534 | +0.0905 | +0.1579 | +0.5731 |
| `lr` | `x94_full` | 94 | 0.4265 | 0.8061 | 2.5694 | +0.1579 | +0.2738 | +0.5764 |
| `rf` | `x85_only` | 85 | 0.4286 | 0.8255 | 11.1941 | +0.0000 | +0.0000 | +0.0000 |
| `rf` | `x9_only` | 9 | 0.3765 | 0.7429 | 14.3706 | -0.0521 | +3.1766 | -0.0164 |
| `rf` | `x94_full` | 94 | 0.4298 | 0.7904 | 11.7668 | +0.0013 | +0.5728 | +0.0022 |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 0.9775 | 1023.0 |
| `ds_cnn` | cuda | 1 | 1.5389 | 649.8 |
| `tiny_cnn` | cuda | 1 | 0.8043 | 1243.2 |
| `mobilenet_like` | cuda | 1 | 2.3397 | 427.4 |
| `tiny_resnet` | cuda | 1 | 2.0147 | 496.4 |
| `small_cnn` | cuda | 1 | 0.9970 | 1003.0 |
| `torchvision_mobilenet` | cuda | 1 | 10.7124 | 93.4 |
| `ultra_tiny` | cuda | 32 | 0.0428 | 23369.6 |
| `ds_cnn` | cuda | 32 | 0.2078 | 4812.7 |
| `tiny_cnn` | cuda | 32 | 0.2071 | 4829.4 |
| `mobilenet_like` | cuda | 32 | 0.2297 | 4353.4 |
| `tiny_resnet` | cuda | 32 | 0.1767 | 5658.8 |
| `small_cnn` | cuda | 32 | 0.2343 | 4267.7 |
| `torchvision_mobilenet` | cuda | 32 | 0.3139 | 3185.4 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Cuello de botella principal de x94:** `region_histograms_g8` consume **37.5%** (0.4827 ms) del tiempo por etapas; cualquier optimizacion futura en C/Cython debe priorizar el bucle de histogramas multi-escala antes que cuantizacion o asimetria.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `ultra_tiny` (**0.8891 ms**, F1=0.5914), mientras el maximo F1 lo obtiene `tiny_resnet` (**F1=0.7389**, 3.6254 ms).
