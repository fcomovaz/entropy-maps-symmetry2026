# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad_carpet`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `99bc026`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.4167 | 0.6434 | 0.2486 (desc) | 0.0069 | **0.2561** | 0.1371 | [0.2536, 0.3482] | 3904.0 | - |
| `mlp_16_dot (x94)` | ml | cpu | 1537 / 94 | 0.4211 | 0.6953 | 0.2486 (desc) | 0.0123 | **0.2617** | 0.1368 | [0.2592, 0.3536] | 3820.4 | - |
| `dt (x94)` | ml | cpu | 13 / 94 | 0.4262 | 0.7003 | 0.2486 (desc) | 0.1328 | **0.3840** | 0.1240 | [0.3808, 0.4728] | 2604.2 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.4167 | 0.6434 | 0.2486 (desc) | 0.2091 | **0.5356** | 0.1392 | [0.4742, 0.5451] | 1867.1 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.5000 | 0.7697 | 0.0438 (prep) | 0.7585 | **0.8083** | 0.0705 | [0.7971, 0.8334] | 1237.1 | 2.62e-07 |
| `knn (x94)` | ml | cpu | 29798 / 94 | 0.2500 | 0.5573 | 0.2486 (desc) | 0.5636 | **0.8893** | 0.2020 | [0.8661, 0.8960] | 1124.4 | - |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.3333 | 0.4561 | 0.0438 (prep) | 2.0063 | **2.0772** | 0.8553 | [1.9602, 2.2711] | 481.4 | 3.44e-07 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.3750 | 0.6711 | 0.0438 (prep) | 2.8796 | **2.9251** | 0.7494 | [2.8860, 3.0312] | 341.9 | 1.65e-07 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.6452 | 0.8495 | 0.0438 (prep) | 3.2915 | **3.3353** | 0.8680 | [3.2475, 3.5349] | 299.8 | 4.38e-07 |
| `rf (x94)` | ml | cpu | 6252 / 94 | 0.1905 | 0.7643 | 0.2486 (desc) | 4.1864 | **4.4628** | 0.7087 | [4.3809, 4.5534] | 224.1 | - |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.3956 | 0.9158 | 0.0438 (prep) | 5.3609 | **5.4192** | 1.3515 | [5.2971, 5.6009] | 184.5 | 4.50e-07 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.3673 | 0.5175 | 0.0438 (prep) | 6.7637 | **6.8077** | 1.3408 | [6.7083, 7.0095] | 146.9 | 5.19e-02 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.6667 | 0.9113 | 0.0438 (prep) | 8.6754 | **8.7191** | 0.4682 | [8.6564, 8.7808] | 114.7 | 4.22e-07 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **9.0419 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2833 ms** vs llamada formal E2E: **0.2693 ms** (gap de dispatch/asignaciones Python = **+0.0140 ms**, +4.9%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0778 | 27.5% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.0678 | 23.9% | 2.1 | `O(g^2*K) [16x3]` |
| `asymmetry_g8` | 8x8 | 0.0673 | 23.8% | 2.6 | `O(g^2*K) [64x3]` |
| `c_fine_hist` | 8x8 | 0.0338 | 11.9% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0149 | 5.2% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 4. Desglose Medido por Capas CNN en CPU (`threads=1, batch=1`)

| Arquitectura | Params | Act Peak (KB) | FLOPs (PyTorch) | Hookless (ms) | Hook overhead (%) | Cold 1st (ms) | Familias dominantes (% hooks) |
|---|---|---|---|---|---|---|---|
| `ultra_tiny` | 1,154 | 128.0 | 1,335,424 | 1.2612 | -14.7% | 1.620 | dwconv:39.6%, bn:21.8%, conv:15.2% |
| `ds_cnn` | 12,866 | 1024.0 | 19,366,400 | 2.6603 | -9.5% | 3.979 | dwconv:36.5%, bn:21.2%, pwconv:16.2% |
| `tiny_cnn` | 23,538 | 1024.0 | 80,216,320 | 8.0689 | -27.4% | 6.282 | pool:63.9%, conv:28.0%, bn:4.9% |
| `mobilenet_like` | 70,786 | 512.0 | 38,716,416 | 3.8887 | -5.8% | 5.093 | dwconv:41.2%, pwconv:22.0%, bn:20.3% |
| `tiny_resnet` | 77,234 | 256.0 | 97,648,896 | 3.7813 | -5.8% | 5.533 | conv:70.4%, bn:12.1%, pwconv:11.4% |
| `small_cnn` | 97,650 | 1024.0 | 117,965,312 | 6.9053 | +7.8% | 8.406 | pool:60.9%, conv:31.6%, bn:4.6% |
| `torchvision_mobilenet` | 1,519,618 | 288.0 | 35,293,184 | 6.7944 | +26.8% | 6.974 | dwconv:31.2%, pwconv:27.7%, bn:20.5% |

## 5. Pruebas de Significancia Estadistica (Mann-Whitney U, 1 hilo, batch=1)

| Comparacion | Mediana A (ms) | Mediana B (ms) | Ratio A/B | Estadistico U | p-valor | Significativo |
|---|---|---|---|---|---|---|
| `dt (x94)` vs `ds_cnn` | 0.3840 | 2.0772 | 0.185x | 0.0 | `4.828e-67` | True |
| `dt (x94)` vs `ultra_tiny` | 0.3840 | 0.8083 | 0.475x | 0.0 | `4.826e-67` | True |
| `lr (x94)` vs `ds_cnn` | 0.5356 | 2.0772 | 0.258x | 0.0 | `4.829e-67` | True |
| `lr (x94)` vs `ultra_tiny` | 0.5356 | 0.8083 | 0.663x | 0.0 | `4.826e-67` | True |
| `lr_dot (x94)` vs `ds_cnn` | 0.2561 | 2.0772 | 0.123x | 0.0 | `4.826e-67` | True |
| `lr_dot (x94)` vs `ultra_tiny` | 0.2561 | 0.8083 | 0.317x | 0.0 | `4.823e-67` | True |
| `mlp_16_dot (x94)` vs `ds_cnn` | 0.2617 | 2.0772 | 0.126x | 0.0 | `4.826e-67` | True |
| `mlp_16_dot (x94)` vs `ultra_tiny` | 0.2617 | 0.8083 | 0.324x | 0.0 | `4.823e-67` | True |
| `rf (x94)` vs `ds_cnn` | 4.4628 | 2.0772 | 2.149x | 39573.0 | `2.744e-64` | True |
| `rf (x94)` vs `ultra_tiny` | 4.4628 | 0.8083 | 5.521x | 40000.0 | `4.827e-67` | True |

## 6. Ablacion Calidad vs Coste (`x85` vs `x9` vs `x94`)

| Clasificador | Subconjunto | Features | F1 | ROC-AUC | T_total (ms) | dF1 vs x85 | d_ms vs x85 | dF1 / d_ms |
|---|---|---|---|---|---|---|---|---|
| `dt` | `x85_only` | 85 | 0.4583 | 0.6989 | 0.0959 | +0.0000 | +0.0000 | +0.0000 |
| `dt` | `x9_only` | 9 | 0.4681 | 0.7186 | 0.2633 | +0.0098 | +0.1674 | +0.0583 |
| `dt` | `x94_full` | 94 | 0.4262 | 0.7003 | 0.2704 | -0.0321 | +0.1745 | -0.1840 |
| `knn` | `x85_only` | 85 | 0.2400 | 0.5690 | 0.4885 | +0.0000 | +0.0000 | +0.0000 |
| `knn` | `x9_only` | 9 | 0.3846 | 0.6711 | 0.7860 | +0.1446 | +0.2974 | +0.4863 |
| `knn` | `x94_full` | 94 | 0.2500 | 0.5573 | 0.6973 | +0.0100 | +0.2088 | +0.0479 |
| `lr` | `x85_only` | 85 | 0.3182 | 0.6147 | 0.2326 | +0.0000 | +0.0000 | +0.0000 |
| `lr` | `x9_only` | 9 | 0.4583 | 0.7159 | 0.3906 | +0.1402 | +0.1580 | +0.8868 |
| `lr` | `x94_full` | 94 | 0.4167 | 0.6434 | 0.3816 | +0.0985 | +0.1489 | +0.6612 |
| `lr_dot` | `x85_only` | 85 | 0.3182 | 0.6147 | 0.0283 | +0.0000 | +0.0000 | +0.0000 |
| `lr_dot` | `x9_only` | 9 | 0.4583 | 0.7159 | 0.1952 | +0.1402 | +0.1669 | +0.8397 |
| `lr_dot` | `x94_full` | 94 | 0.4167 | 0.6434 | 0.1883 | +0.0985 | +0.1600 | +0.6155 |
| `mlp_16_dot` | `x85_only` | 85 | 0.4865 | 0.6676 | 0.0326 | +0.0000 | +0.0000 | +0.0000 |
| `mlp_16_dot` | `x9_only` | 9 | 0.5000 | 0.7204 | 0.2003 | +0.0135 | +0.1677 | +0.0806 |
| `mlp_16_dot` | `x94_full` | 94 | 0.4211 | 0.6953 | 0.1932 | -0.0654 | +0.1606 | -0.4074 |
| `rf` | `x85_only` | 85 | 0.0000 | 0.7375 | 3.9498 | +0.0000 | +0.0000 | +0.0000 |
| `rf` | `x9_only` | 9 | 0.3846 | 0.7249 | 4.3746 | +0.3846 | +0.4248 | +0.9053 |
| `rf` | `x94_full` | 94 | 0.1905 | 0.7643 | 4.3287 | +0.1905 | +0.3789 | +0.5028 |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.0534 | 949.4 |
| `ds_cnn` | cuda | 1 | 1.2601 | 793.6 |
| `tiny_cnn` | cuda | 1 | 0.7361 | 1358.4 |
| `mobilenet_like` | cuda | 1 | 2.1902 | 456.6 |
| `tiny_resnet` | cuda | 1 | 1.6742 | 597.3 |
| `small_cnn` | cuda | 1 | 0.9382 | 1065.9 |
| `torchvision_mobilenet` | cuda | 1 | 8.5884 | 116.4 |
| `ultra_tiny` | cuda | 32 | 0.0434 | 23035.7 |
| `ds_cnn` | cuda | 32 | 0.2026 | 4936.8 |
| `tiny_cnn` | cuda | 32 | 0.2055 | 4867.1 |
| `mobilenet_like` | cuda | 32 | 0.2300 | 4348.5 |
| `tiny_resnet` | cuda | 32 | 0.1793 | 5578.3 |
| `small_cnn` | cuda | 32 | 0.2323 | 4304.0 |
| `torchvision_mobilenet` | cuda | 32 | 0.3323 | 3009.7 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **27.5%** (0.0778 ms) del tiempo por etapas (E2E=0.2693 ms, gap de despacho Python/ctypes=+0.0140 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.2561 ms**, F1=0.4167), mientras el maximo F1 lo obtiene `small_cnn` (**F1=0.6667**, 8.7191 ms).
