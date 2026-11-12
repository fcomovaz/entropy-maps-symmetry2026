# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad_bottle`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `37d8114`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.8000 | 0.9231 | 0.4762 (desc) | 0.0097 | **0.4863** | 0.1208 | [0.4749, 0.5029] | 2056.3 | - |
| `dt (x94)` | ml | cpu | 15 / 94 | 0.5833 | 0.6614 | 0.4762 (desc) | 0.1673 | **0.6517** | 0.1980 | [0.6322, 0.6830] | 1534.6 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.8000 | 0.9231 | 0.4762 (desc) | 0.2402 | **0.7367** | 0.1315 | [0.7180, 0.7627] | 1357.4 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.3611 | 0.5268 | 0.0444 (prep) | 0.9680 | **1.0171** | 0.0982 | [1.0114, 1.0275] | 983.1 | 1.47e-07 |
| `knn (x94)` | ml | cpu | 21902 / 94 | 0.3750 | 0.7057 | 0.4762 (desc) | 1.1622 | **1.6410** | 0.6810 | [1.5378, 1.7404] | 609.4 | - |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.3611 | 0.5535 | 0.0444 (prep) | 2.9489 | **2.9935** | 0.6020 | [2.9303, 3.0908] | 334.1 | 2.94e-07 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.7586 | 0.9415 | 0.0444 (prep) | 4.0777 | **4.1231** | 0.7047 | [4.0608, 4.2537] | 242.5 | 4.28e-07 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.3662 | 0.7391 | 0.0444 (prep) | 4.2957 | **4.3398** | 0.7069 | [4.2651, 4.4441] | 230.4 | 4.35e-07 |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.3571 | 0.6472 | 0.0444 (prep) | 5.2470 | **5.2910** | 1.1028 | [5.2416, 5.4434] | 189.0 | 4.07e-07 |
| `rf (x94)` | ml | cpu | 3496 / 94 | 0.7273 | 0.9030 | 0.4762 (desc) | 5.1308 | **5.6139** | 1.3421 | [5.5176, 5.8256] | 178.1 | - |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.3611 | 0.6639 | 0.0444 (prep) | 7.2579 | **7.3024** | 0.7135 | [7.2018, 7.3641] | 136.9 | 4.01e-07 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.3611 | 0.5000 | 0.0444 (prep) | 8.3512 | **8.4119** | 1.3748 | [8.2571, 8.5131] | 118.9 | 2.01e-02 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **8.2548 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2418 ms** vs llamada formal E2E: **0.2054 ms** (gap de dispatch/asignaciones Python = **+0.0364 ms**, +15.0%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0673 | 27.8% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.0556 | 23.0% | 2.1 | `O(g^2*K) [16x3]` |
| `asymmetry_g8` | 8x8 | 0.0540 | 22.3% | 2.6 | `O(g^2*K) [64x3]` |
| `c_fine_hist` | 8x8 | 0.0338 | 14.0% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0125 | 5.2% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 4. Desglose Medido por Capas CNN en CPU (`threads=1, batch=1`)

| Arquitectura | Params | Act Peak (KB) | FLOPs (PyTorch) | Hookless (ms) | Hook overhead (%) | Cold 1st (ms) | Familias dominantes (% hooks) |
|---|---|---|---|---|---|---|---|
| `ultra_tiny` | 1,154 | 128.0 | 1,335,424 | 1.5281 | -24.3% | 1.873 | dwconv:39.9%, bn:21.6%, conv:15.1% |
| `ds_cnn` | 12,866 | 1024.0 | 19,366,400 | 2.9713 | +2.8% | 6.220 | dwconv:39.0%, bn:21.4%, pwconv:14.5% |
| `tiny_cnn` | 23,538 | 1024.0 | 80,216,320 | 5.0871 | +12.7% | 8.374 | pool:54.6%, conv:34.3%, bn:6.9% |
| `mobilenet_like` | 70,786 | 512.0 | 38,716,416 | 4.0325 | +16.7% | 6.649 | dwconv:43.0%, pwconv:20.7%, bn:20.3% |
| `tiny_resnet` | 77,234 | 256.0 | 97,648,896 | 4.2401 | +18.3% | 7.356 | conv:69.1%, bn:12.6%, pwconv:11.9% |
| `small_cnn` | 97,650 | 1024.0 | 117,965,312 | 7.5450 | +3.5% | 8.668 | pool:50.7%, conv:38.3%, bn:7.0% |
| `torchvision_mobilenet` | 1,519,618 | 288.0 | 35,293,184 | 8.4278 | +21.5% | 12.041 | dwconv:33.9%, pwconv:25.7%, bn:19.5% |

## 5. Pruebas de Significancia Estadistica (Mann-Whitney U, 1 hilo, batch=1)

| Comparacion | Mediana A (ms) | Mediana B (ms) | Ratio A/B | Estadistico U | p-valor | Significativo |
|---|---|---|---|---|---|---|
| `dt (x94)` vs `ds_cnn` | 0.6517 | 2.9935 | 0.218x | 677.0 | `1.056e-62` | True |
| `dt (x94)` vs `ultra_tiny` | 0.6517 | 1.0171 | 0.641x | 2446.0 | `4.595e-52` | True |
| `lr (x94)` vs `ds_cnn` | 0.7367 | 2.9935 | 0.246x | 140.0 | `3.923e-66` | True |
| `lr (x94)` vs `ultra_tiny` | 0.7367 | 1.0171 | 0.724x | 3547.5 | `5.961e-46` | True |
| `lr_dot (x94)` vs `ds_cnn` | 0.4863 | 2.9935 | 0.162x | 115.0 | `2.702e-66` | True |
| `lr_dot (x94)` vs `ultra_tiny` | 0.4863 | 1.0171 | 0.478x | 771.0 | `4.114e-62` | True |
| `rf (x94)` vs `ds_cnn` | 5.6139 | 2.9935 | 1.875x | 39827.0 | `6.413e-66` | True |
| `rf (x94)` vs `ultra_tiny` | 5.6139 | 1.0171 | 5.519x | 40000.0 | `4.830e-67` | True |

## 6. Ablacion Calidad vs Coste (`x85` vs `x9` vs `x94`)

| Clasificador | Subconjunto | Features | F1 | ROC-AUC | T_total (ms) | dF1 vs x85 | d_ms vs x85 | dF1 / d_ms |
|---|---|---|---|---|---|---|---|---|
| `dt` | `x85_only` | 85 | 0.5714 | 0.6781 | 0.1069 | +0.0000 | +0.0000 | +0.0000 |
| `dt` | `x9_only` | 9 | 0.5882 | 0.7584 | 0.2774 | +0.0168 | +0.1705 | +0.0986 |
| `dt` | `x94_full` | 94 | 0.5833 | 0.6614 | 0.2910 | +0.0119 | +0.1841 | +0.0646 |
| `knn` | `x85_only` | 85 | 0.1429 | 0.6605 | 0.6635 | +0.0000 | +0.0000 | +0.0000 |
| `knn` | `x9_only` | 9 | 0.6667 | 0.8997 | 0.9866 | +0.5238 | +0.3231 | +1.6209 |
| `knn` | `x94_full` | 94 | 0.3750 | 0.7057 | 0.9036 | +0.2321 | +0.2401 | +0.9669 |
| `lr` | `x85_only` | 85 | 0.7333 | 0.9181 | 0.2687 | +0.0000 | +0.0000 | +0.0000 |
| `lr` | `x9_only` | 9 | 0.7333 | 0.9130 | 0.4461 | +0.0000 | +0.1774 | +0.0000 |
| `lr` | `x94_full` | 94 | 0.8000 | 0.9231 | 0.4305 | +0.0667 | +0.1618 | +0.4120 |
| `lr_dot` | `x85_only` | 85 | 0.7333 | 0.9181 | 0.0322 | +0.0000 | +0.0000 | +0.0000 |
| `lr_dot` | `x9_only` | 9 | 0.7333 | 0.9130 | 0.2021 | +0.0000 | +0.1699 | +0.0000 |
| `lr_dot` | `x94_full` | 94 | 0.8000 | 0.9231 | 0.2171 | +0.0667 | +0.1849 | +0.3605 |
| `rf` | `x85_only` | 85 | 0.5263 | 0.8696 | 5.1937 | +0.0000 | +0.0000 | +0.0000 |
| `rf` | `x9_only` | 9 | 0.6923 | 0.8980 | 4.9977 | +0.1660 | -0.1961 | -0.8467 |
| `rf` | `x94_full` | 94 | 0.7273 | 0.9030 | 5.2716 | +0.2010 | +0.0779 | +2.5780 |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.1476 | 871.4 |
| `ds_cnn` | cuda | 1 | 1.7485 | 571.9 |
| `tiny_cnn` | cuda | 1 | 0.8548 | 1169.8 |
| `mobilenet_like` | cuda | 1 | 2.5569 | 391.1 |
| `tiny_resnet` | cuda | 1 | 2.2761 | 439.4 |
| `small_cnn` | cuda | 1 | 1.3247 | 754.9 |
| `torchvision_mobilenet` | cuda | 1 | 10.1573 | 98.5 |
| `ultra_tiny` | cuda | 32 | 0.0443 | 22548.7 |
| `ds_cnn` | cuda | 32 | 0.2025 | 4937.9 |
| `tiny_cnn` | cuda | 32 | 0.2078 | 4811.2 |
| `mobilenet_like` | cuda | 32 | 0.2316 | 4318.7 |
| `tiny_resnet` | cuda | 32 | 0.1796 | 5568.3 |
| `small_cnn` | cuda | 32 | 0.2330 | 4291.9 |
| `torchvision_mobilenet` | cuda | 32 | 0.3577 | 2795.9 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **27.8%** (0.0673 ms) del tiempo por etapas (E2E=0.2054 ms, gap de despacho Python/ctypes=+0.0364 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.4863 ms**, F1=0.8000), mientras el maximo F1 lo obtiene `lr (x94)` (**F1=0.8000**, 0.7367 ms).
