# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad_cable`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `99bc026`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.5116 | 0.7632 | 0.3462 (desc) | 0.0068 | **0.3530** | 0.0208 | [0.3523, 0.3536] | 2832.9 | - |
| `mlp_16_dot (x94)` | ml | cpu | 1537 / 94 | 0.5625 | 0.8460 | 0.3462 (desc) | 0.0118 | **0.3579** | 0.0208 | [0.3573, 0.3586] | 2793.7 | - |
| `dt (x94)` | ml | cpu | 13 / 94 | 0.4000 | 0.6014 | 0.3462 (desc) | 0.1277 | **0.4754** | 0.0320 | [0.4738, 0.4789] | 2103.5 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.5116 | 0.7632 | 0.3462 (desc) | 0.1998 | **0.5572** | 0.0530 | [0.5523, 0.5608] | 1794.7 | - |
| `knn (x94)` | ml | cpu | 28106 / 94 | 0.3478 | 0.6296 | 0.3462 (desc) | 0.4832 | **0.8482** | 0.0766 | [0.8436, 0.8522] | 1179.0 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.5000 | 0.7096 | 0.0405 (prep) | 0.8255 | **0.8663** | 0.2932 | [0.8439, 0.8881] | 1154.3 | 4.27e-07 |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.5769 | 0.8226 | 0.0405 (prep) | 1.9196 | **1.9600** | 0.6675 | [1.9263, 2.0053] | 510.2 | 4.75e-07 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.5294 | 0.7827 | 0.0405 (prep) | 2.7934 | **2.8341** | 0.4157 | [2.7698, 2.8781] | 352.8 | 4.91e-07 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.7000 | 0.8967 | 0.0405 (prep) | 3.2340 | **3.2744** | 0.8167 | [3.2088, 3.5513] | 305.4 | 4.50e-07 |
| `rf (x94)` | ml | cpu | 6014 / 94 | 0.4348 | 0.7836 | 0.3462 (desc) | 4.2050 | **4.5642** | 0.7660 | [4.4686, 4.7331] | 219.1 | - |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.6667 | 0.8626 | 0.0405 (prep) | 5.1025 | **5.1433** | 1.1748 | [4.9997, 5.2953] | 194.4 | 4.91e-07 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.3636 | 0.9561 | 0.0405 (prep) | 6.3055 | **6.3458** | 1.1799 | [6.1983, 6.5306] | 157.6 | 3.64e-07 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.3871 | 0.5000 | 0.0405 (prep) | 6.3937 | **6.4342** | 1.0287 | [6.2891, 6.6013] | 155.4 | 1.17e-02 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **7.0924 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2418 ms** vs llamada formal E2E: **0.1966 ms** (gap de dispatch/asignaciones Python = **+0.0452 ms**, +18.7%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0686 | 28.4% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.0559 | 23.1% | 2.1 | `O(g^2*K) [16x3]` |
| `asymmetry_g8` | 8x8 | 0.0542 | 22.4% | 2.6 | `O(g^2*K) [64x3]` |
| `c_fine_hist` | 8x8 | 0.0323 | 13.4% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0120 | 5.0% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 4. Desglose Medido por Capas CNN en CPU (`threads=1, batch=1`)

| Arquitectura | Params | Act Peak (KB) | FLOPs (PyTorch) | Hookless (ms) | Hook overhead (%) | Cold 1st (ms) | Familias dominantes (% hooks) |
|---|---|---|---|---|---|---|---|
| `ultra_tiny` | 1,154 | 128.0 | 1,335,424 | 0.9616 | +13.4% | 1.351 | dwconv:39.6%, bn:21.7%, conv:15.1% |
| `ds_cnn` | 12,866 | 1024.0 | 19,366,400 | 2.2363 | +16.3% | 4.386 | dwconv:36.9%, bn:21.3%, pwconv:15.9% |
| `tiny_cnn` | 23,538 | 1024.0 | 80,216,320 | 5.2977 | +4.0% | 8.988 | pool:61.9%, conv:29.3%, bn:5.3% |
| `mobilenet_like` | 70,786 | 512.0 | 38,716,416 | 3.2917 | +5.7% | 3.295 | dwconv:40.9%, pwconv:22.5%, bn:20.3% |
| `tiny_resnet` | 77,234 | 256.0 | 97,648,896 | 3.6796 | -1.4% | 5.802 | conv:70.4%, bn:12.0%, pwconv:11.3% |
| `small_cnn` | 97,650 | 1024.0 | 117,965,312 | 6.6900 | +0.8% | 6.550 | pool:58.3%, conv:33.6%, bn:5.0% |
| `torchvision_mobilenet` | 1,519,618 | 288.0 | 35,293,184 | 6.9048 | +33.2% | 9.112 | dwconv:31.5%, pwconv:27.4%, bn:20.5% |

## 5. Pruebas de Significancia Estadistica (Mann-Whitney U, 1 hilo, batch=1)

| Comparacion | Mediana A (ms) | Mediana B (ms) | Ratio A/B | Estadistico U | p-valor | Significativo |
|---|---|---|---|---|---|---|
| `dt (x94)` vs `ds_cnn` | 0.4754 | 1.9600 | 0.243x | 0.0 | `4.825e-67` | True |
| `dt (x94)` vs `ultra_tiny` | 0.4754 | 0.8663 | 0.549x | 0.0 | `4.825e-67` | True |
| `lr (x94)` vs `ds_cnn` | 0.5572 | 1.9600 | 0.284x | 0.0 | `4.825e-67` | True |
| `lr (x94)` vs `ultra_tiny` | 0.5572 | 0.8663 | 0.643x | 0.0 | `4.825e-67` | True |
| `lr_dot (x94)` vs `ds_cnn` | 0.3530 | 1.9600 | 0.180x | 0.0 | `4.818e-67` | True |
| `lr_dot (x94)` vs `ultra_tiny` | 0.3530 | 0.8663 | 0.407x | 0.0 | `4.817e-67` | True |
| `mlp_16_dot (x94)` vs `ds_cnn` | 0.3579 | 1.9600 | 0.183x | 0.0 | `4.816e-67` | True |
| `mlp_16_dot (x94)` vs `ultra_tiny` | 0.3579 | 0.8663 | 0.413x | 0.0 | `4.815e-67` | True |
| `rf (x94)` vs `ds_cnn` | 4.5642 | 1.9600 | 2.329x | 39886.0 | `2.661e-66` | True |
| `rf (x94)` vs `ultra_tiny` | 4.5642 | 0.8663 | 5.268x | 40000.0 | `4.830e-67` | True |

## 6. Ablacion Calidad vs Coste (`x85` vs `x9` vs `x94`)

| Clasificador | Subconjunto | Features | F1 | ROC-AUC | T_total (ms) | dF1 vs x85 | d_ms vs x85 | dF1 / d_ms |
|---|---|---|---|---|---|---|---|---|
| `dt` | `x85_only` | 85 | 0.4000 | 0.6014 | 0.0974 | +0.0000 | +0.0000 | +0.0000 |
| `dt` | `x9_only` | 9 | 0.3396 | 0.5692 | 0.2596 | -0.0604 | +0.1623 | -0.3721 |
| `dt` | `x94_full` | 94 | 0.4000 | 0.6014 | 0.2619 | +0.0000 | +0.1645 | +0.0000 |
| `knn` | `x85_only` | 85 | 0.3478 | 0.7651 | 0.4900 | +0.0000 | +0.0000 | +0.0000 |
| `knn` | `x9_only` | 9 | 0.4000 | 0.4981 | 0.7831 | +0.0522 | +0.2930 | +0.1781 |
| `knn` | `x94_full` | 94 | 0.3478 | 0.6296 | 0.6793 | +0.0000 | +0.1892 | +0.0000 |
| `lr` | `x85_only` | 85 | 0.6190 | 0.8411 | 0.2253 | +0.0000 | +0.0000 | +0.0000 |
| `lr` | `x9_only` | 9 | 0.4348 | 0.6111 | 0.3795 | -0.1843 | +0.1542 | -1.1950 |
| `lr` | `x94_full` | 94 | 0.5116 | 0.7632 | 0.3813 | -0.1074 | +0.1560 | -0.6886 |
| `lr_dot` | `x85_only` | 85 | 0.6190 | 0.8411 | 0.0289 | +0.0000 | +0.0000 | +0.0000 |
| `lr_dot` | `x9_only` | 9 | 0.4348 | 0.6111 | 0.1870 | -0.1843 | +0.1581 | -1.1651 |
| `lr_dot` | `x94_full` | 94 | 0.5116 | 0.7632 | 0.1902 | -0.1074 | +0.1613 | -0.6660 |
| `mlp_16_dot` | `x85_only` | 85 | 0.6857 | 0.8967 | 0.0337 | +0.0000 | +0.0000 | +0.0000 |
| `mlp_16_dot` | `x9_only` | 9 | 0.3830 | 0.6442 | 0.1925 | -0.3027 | +0.1588 | -1.9058 |
| `mlp_16_dot` | `x94_full` | 94 | 0.5625 | 0.8460 | 0.1949 | -0.1232 | +0.1612 | -0.7644 |
| `rf` | `x85_only` | 85 | 0.2727 | 0.8558 | 4.1317 | +0.0000 | +0.0000 | +0.0000 |
| `rf` | `x9_only` | 9 | 0.3478 | 0.5897 | 4.4262 | +0.0751 | +0.2946 | +0.2550 |
| `rf` | `x94_full` | 94 | 0.4348 | 0.7836 | 4.4556 | +0.1621 | +0.3239 | +0.5003 |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.1173 | 895.0 |
| `ds_cnn` | cuda | 1 | 1.5464 | 646.7 |
| `tiny_cnn` | cuda | 1 | 0.9511 | 1051.4 |
| `mobilenet_like` | cuda | 1 | 2.6257 | 380.9 |
| `tiny_resnet` | cuda | 1 | 2.6210 | 381.5 |
| `small_cnn` | cuda | 1 | 1.4914 | 670.5 |
| `torchvision_mobilenet` | cuda | 1 | 11.9343 | 83.8 |
| `ultra_tiny` | cuda | 32 | 0.0434 | 23041.5 |
| `ds_cnn` | cuda | 32 | 0.2137 | 4679.7 |
| `tiny_cnn` | cuda | 32 | 0.2067 | 4837.2 |
| `mobilenet_like` | cuda | 32 | 0.2296 | 4355.6 |
| `tiny_resnet` | cuda | 32 | 0.1784 | 5604.5 |
| `small_cnn` | cuda | 32 | 0.2331 | 4290.7 |
| `torchvision_mobilenet` | cuda | 32 | 0.4582 | 2182.6 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **28.4%** (0.0686 ms) del tiempo por etapas (E2E=0.1966 ms, gap de despacho Python/ctypes=+0.0452 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.3530 ms**, F1=0.5116), mientras el maximo F1 lo obtiene `tiny_resnet` (**F1=0.7000**, 3.2744 ms).
