# Profiling Cientifico y Comparacion 100% CPU: `kolektorsdd2`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `99bc026`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.4265 | 0.8061 | 0.2051 (desc) | 0.0068 | **0.2119** | 0.0072 | [0.2115, 0.2126] | 4719.2 | - |
| `mlp_16_dot (x94)` | ml | cpu | 1537 / 94 | 0.4904 | 0.8334 | 0.2051 (desc) | 0.0128 | **0.2189** | 0.0145 | [0.2183, 0.2203] | 4567.3 | - |
| `dt (x94)` | ml | cpu | 285 / 94 | 0.2798 | 0.5962 | 0.2051 (desc) | 0.0849 | **0.2906** | 0.0111 | [0.2899, 0.2923] | 3441.2 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.4265 | 0.8061 | 0.2051 (desc) | 0.2155 | **0.4323** | 0.0624 | [0.4246, 0.4337] | 2313.2 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.6304 | 0.8815 | 0.0392 (prep) | 0.7664 | **0.8061** | 0.0562 | [0.8050, 0.8081] | 1240.5 | 2.85e-07 |
| `knn (x94)` | ml | cpu | 219114 / 94 | 0.1500 | 0.6652 | 0.2051 (desc) | 0.6824 | **0.8992** | 0.1728 | [0.8882, 0.9078] | 1112.0 | - |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.6562 | 0.8779 | 0.0392 (prep) | 1.9729 | **2.0122** | 0.8413 | [1.9867, 2.0790] | 497.0 | 2.38e-07 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.6250 | 0.8622 | 0.0392 (prep) | 2.8674 | **2.9079** | 0.5297 | [2.8913, 2.9522] | 343.9 | 3.61e-07 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.7130 | 0.9213 | 0.0392 (prep) | 3.3609 | **3.4002** | 0.5902 | [3.2731, 3.4528] | 294.1 | 4.78e-07 |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.6154 | 0.8714 | 0.0392 (prep) | 4.0947 | **4.1380** | 0.9497 | [4.0583, 4.2895] | 241.7 | 3.85e-07 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.5965 | 0.8927 | 0.0392 (prep) | 5.7326 | **5.7735** | 1.4819 | [5.5113, 6.0507] | 173.2 | 3.41e-07 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.5976 | 0.7332 | 0.0392 (prep) | 6.4279 | **6.4673** | 1.2959 | [6.3601, 6.6753] | 154.6 | 3.55e-07 |
| `rf (x94)` | ml | cpu | 13208 / 94 | 0.4298 | 0.7904 | 0.2051 (desc) | 8.2677 | **8.4736** | 1.1310 | [8.3251, 8.6990] | 118.0 | - |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **8.7252 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2344 ms** vs llamada formal E2E: **0.1928 ms** (gap de dispatch/asignaciones Python = **+0.0415 ms**, +17.7%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0654 | 27.9% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.0551 | 23.5% | 2.1 | `O(g^2*K) [16x3]` |
| `asymmetry_g8` | 8x8 | 0.0539 | 23.0% | 2.6 | `O(g^2*K) [64x3]` |
| `c_fine_hist` | 8x8 | 0.0319 | 13.6% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0109 | 4.7% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 4. Desglose Medido por Capas CNN en CPU (`threads=1, batch=1`)

| Arquitectura | Params | Act Peak (KB) | FLOPs (PyTorch) | Hookless (ms) | Hook overhead (%) | Cold 1st (ms) | Familias dominantes (% hooks) |
|---|---|---|---|---|---|---|---|
| `ultra_tiny` | 1,154 | 128.0 | 1,335,424 | 0.9111 | +18.7% | 1.717 | dwconv:39.6%, bn:22.0%, conv:15.2% |
| `ds_cnn` | 12,866 | 1024.0 | 19,366,400 | 2.0469 | +5.9% | 4.633 | dwconv:37.6%, bn:21.3%, pwconv:15.8% |
| `tiny_cnn` | 23,538 | 1024.0 | 80,216,320 | 4.0929 | +6.2% | 4.519 | pool:55.7%, conv:33.6%, bn:6.4% |
| `mobilenet_like` | 70,786 | 512.0 | 38,716,416 | 3.1111 | +14.5% | 3.247 | dwconv:40.9%, pwconv:22.0%, bn:20.7% |
| `tiny_resnet` | 77,234 | 256.0 | 97,648,896 | 3.6502 | +10.6% | 4.206 | conv:69.9%, bn:12.1%, pwconv:11.4% |
| `small_cnn` | 97,650 | 1024.0 | 117,965,312 | 6.1072 | -7.4% | 7.655 | pool:53.3%, conv:37.8%, bn:5.5% |
| `torchvision_mobilenet` | 1,519,618 | 288.0 | 35,293,184 | 6.8902 | +32.3% | 7.105 | dwconv:31.3%, pwconv:27.2%, bn:20.8% |

## 5. Pruebas de Significancia Estadistica (Mann-Whitney U, 1 hilo, batch=1)

| Comparacion | Mediana A (ms) | Mediana B (ms) | Ratio A/B | Estadistico U | p-valor | Significativo |
|---|---|---|---|---|---|---|
| `dt (x94)` vs `ds_cnn` | 0.2906 | 2.0122 | 0.144x | 0.0 | `4.815e-67` | True |
| `dt (x94)` vs `ultra_tiny` | 0.2906 | 0.8061 | 0.360x | 0.0 | `4.811e-67` | True |
| `lr (x94)` vs `ds_cnn` | 0.4323 | 2.0122 | 0.215x | 0.0 | `4.827e-67` | True |
| `lr (x94)` vs `ultra_tiny` | 0.4323 | 0.8061 | 0.536x | 0.0 | `4.823e-67` | True |
| `lr_dot (x94)` vs `ds_cnn` | 0.2119 | 2.0122 | 0.105x | 0.0 | `4.816e-67` | True |
| `lr_dot (x94)` vs `ultra_tiny` | 0.2119 | 0.8061 | 0.263x | 0.0 | `4.812e-67` | True |
| `mlp_16_dot (x94)` vs `ds_cnn` | 0.2189 | 2.0122 | 0.109x | 0.0 | `4.820e-67` | True |
| `mlp_16_dot (x94)` vs `ultra_tiny` | 0.2189 | 0.8061 | 0.272x | 0.0 | `4.816e-67` | True |
| `rf (x94)` vs `ds_cnn` | 8.4736 | 2.0122 | 4.211x | 40000.0 | `4.831e-67` | True |
| `rf (x94)` vs `ultra_tiny` | 8.4736 | 0.8061 | 10.511x | 40000.0 | `4.826e-67` | True |

## 6. Ablacion Calidad vs Coste (`x85` vs `x9` vs `x94`)

| Clasificador | Subconjunto | Features | F1 | ROC-AUC | T_total (ms) | dF1 vs x85 | d_ms vs x85 | dF1 / d_ms |
|---|---|---|---|---|---|---|---|---|
| `dt` | `x85_only` | 85 | 0.3550 | 0.6117 | 0.1135 | +0.0000 | +0.0000 | +0.0000 |
| `dt` | `x9_only` | 9 | 0.2326 | 0.5787 | 0.2847 | -0.1224 | +0.1712 | -0.7153 |
| `dt` | `x94_full` | 94 | 0.2798 | 0.5962 | 0.2778 | -0.0751 | +0.1643 | -0.4574 |
| `knn` | `x85_only` | 85 | 0.1513 | 0.6442 | 0.6900 | +0.0000 | +0.0000 | +0.0000 |
| `knn` | `x9_only` | 9 | 0.2208 | 0.6328 | 0.7984 | +0.0695 | +0.1084 | +0.6413 |
| `knn` | `x94_full` | 94 | 0.1500 | 0.6652 | 1.0514 | -0.0013 | +0.3614 | -0.0035 |
| `lr` | `x85_only` | 85 | 0.2687 | 0.6179 | 0.2343 | +0.0000 | +0.0000 | +0.0000 |
| `lr` | `x9_only` | 9 | 0.3591 | 0.7371 | 0.4088 | +0.0905 | +0.1745 | +0.5185 |
| `lr` | `x94_full` | 94 | 0.4265 | 0.8061 | 0.4116 | +0.1579 | +0.1773 | +0.8903 |
| `lr_dot` | `x85_only` | 85 | 0.2687 | 0.6179 | 0.0312 | +0.0000 | +0.0000 | +0.0000 |
| `lr_dot` | `x9_only` | 9 | 0.3591 | 0.7371 | 0.2053 | +0.0905 | +0.1741 | +0.5197 |
| `lr_dot` | `x94_full` | 94 | 0.4265 | 0.8061 | 0.2046 | +0.1579 | +0.1734 | +0.9104 |
| `mlp_16_dot` | `x85_only` | 85 | 0.3548 | 0.7008 | 0.0364 | +0.0000 | +0.0000 | +0.0000 |
| `mlp_16_dot` | `x9_only` | 9 | 0.3416 | 0.7441 | 0.2077 | -0.0133 | +0.1713 | -0.0774 |
| `mlp_16_dot` | `x94_full` | 94 | 0.4904 | 0.8334 | 0.2098 | +0.1356 | +0.1734 | +0.7820 |
| `rf` | `x85_only` | 85 | 0.4286 | 0.8255 | 8.7233 | +0.0000 | +0.0000 | +0.0000 |
| `rf` | `x9_only` | 9 | 0.3765 | 0.7429 | 8.5482 | -0.0521 | -0.1751 | +0.2975 |
| `rf` | `x94_full` | 94 | 0.4298 | 0.7904 | 8.0870 | +0.0013 | -0.6364 | -0.0020 |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 0.9655 | 1035.7 |
| `ds_cnn` | cuda | 1 | 1.2679 | 788.7 |
| `tiny_cnn` | cuda | 1 | 0.7449 | 1342.5 |
| `mobilenet_like` | cuda | 1 | 2.3271 | 429.7 |
| `tiny_resnet` | cuda | 1 | 1.6910 | 591.4 |
| `small_cnn` | cuda | 1 | 0.9544 | 1047.8 |
| `torchvision_mobilenet` | cuda | 1 | 9.0762 | 110.2 |
| `ultra_tiny` | cuda | 32 | 0.0448 | 22341.7 |
| `ds_cnn` | cuda | 32 | 0.2008 | 4980.6 |
| `tiny_cnn` | cuda | 32 | 0.2066 | 4839.5 |
| `mobilenet_like` | cuda | 32 | 0.2298 | 4350.7 |
| `tiny_resnet` | cuda | 32 | 0.1787 | 5597.3 |
| `small_cnn` | cuda | 32 | 0.2321 | 4309.2 |
| `torchvision_mobilenet` | cuda | 32 | 0.2903 | 3444.8 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **27.9%** (0.0654 ms) del tiempo por etapas (E2E=0.1928 ms, gap de despacho Python/ctypes=+0.0415 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.2119 ms**, F1=0.4265), mientras el maximo F1 lo obtiene `tiny_resnet` (**F1=0.7130**, 3.4002 ms).
