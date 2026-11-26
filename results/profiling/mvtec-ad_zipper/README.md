# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad_zipper`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `99bc026`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.7925 | 0.9492 | 0.2183 (desc) | 0.0067 | **0.2251** | 0.1107 | [0.2170, 0.2514] | 4442.5 | - |
| `mlp_16_dot (x94)` | ml | cpu | 1537 / 94 | 0.8571 | 0.9636 | 0.2183 (desc) | 0.0117 | **0.2299** | 0.1110 | [0.2221, 0.2563] | 4348.8 | - |
| `dt (x94)` | ml | cpu | 15 / 94 | 0.7037 | 0.8633 | 0.2183 (desc) | 0.0952 | **0.3509** | 0.1316 | [0.3378, 0.3702] | 2849.4 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.7925 | 0.9492 | 0.2183 (desc) | 0.2194 | **0.4630** | 0.1180 | [0.4614, 0.4692] | 2159.8 | - |
| `knn (x94)` | ml | cpu | 29328 / 94 | 0.2857 | 0.7924 | 0.2183 (desc) | 0.5002 | **0.7795** | 0.1739 | [0.7449, 0.8182] | 1282.9 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.5352 | 0.6553 | 0.0501 (prep) | 0.7768 | **0.8439** | 0.1758 | [0.8213, 0.8513] | 1184.9 | 4.81e-07 |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.6341 | 0.8758 | 0.0501 (prep) | 2.0381 | **2.0890** | 0.8106 | [2.0712, 2.1856] | 478.7 | 4.24e-07 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.6667 | 0.8803 | 0.0501 (prep) | 3.0431 | **3.1007** | 0.7842 | [3.0419, 3.2057] | 322.5 | 3.33e-07 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.7797 | 0.9409 | 0.0501 (prep) | 3.2087 | **3.2643** | 0.7130 | [3.2340, 3.3314] | 306.3 | 2.42e-07 |
| `rf (x94)` | ml | cpu | 5538 / 94 | 0.7273 | 0.9409 | 0.2183 (desc) | 4.3469 | **4.6162** | 0.7983 | [4.5234, 4.7684] | 216.6 | - |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.6286 | 0.8902 | 0.0501 (prep) | 4.6119 | **4.6685** | 0.8198 | [4.5601, 4.7868] | 214.2 | 4.85e-07 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.8400 | 0.9780 | 0.0501 (prep) | 6.4104 | **6.4583** | 1.4619 | [6.2551, 6.6807] | 154.8 | 4.55e-07 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.4660 | 0.5451 | 0.0501 (prep) | 6.8849 | **6.9459** | 1.3725 | [6.7558, 7.1637] | 144.0 | 9.09e-03 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **8.2227 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2260 ms** vs llamada formal E2E: **0.1946 ms** (gap de dispatch/asignaciones Python = **+0.0314 ms**, +13.9%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0629 | 27.8% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.0524 | 23.2% | 2.1 | `O(g^2*K) [16x3]` |
| `asymmetry_g8` | 8x8 | 0.0511 | 22.6% | 2.6 | `O(g^2*K) [64x3]` |
| `c_fine_hist` | 8x8 | 0.0310 | 13.7% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0112 | 5.0% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 4. Desglose Medido por Capas CNN en CPU (`threads=1, batch=1`)

| Arquitectura | Params | Act Peak (KB) | FLOPs (PyTorch) | Hookless (ms) | Hook overhead (%) | Cold 1st (ms) | Familias dominantes (% hooks) |
|---|---|---|---|---|---|---|---|
| `ultra_tiny` | 1,154 | 128.0 | 1,335,424 | 1.2435 | -4.3% | 1.182 | dwconv:40.5%, bn:22.2%, conv:14.1% |
| `ds_cnn` | 12,866 | 1024.0 | 19,366,400 | 2.4710 | -3.2% | 5.131 | dwconv:36.6%, bn:20.2%, pwconv:15.9% |
| `tiny_cnn` | 23,538 | 1024.0 | 80,216,320 | 5.3239 | -1.0% | 6.141 | pool:60.2%, conv:30.7%, bn:5.5% |
| `mobilenet_like` | 70,786 | 512.0 | 38,716,416 | 3.2749 | +10.6% | 7.141 | dwconv:40.7%, pwconv:22.1%, bn:20.4% |
| `tiny_resnet` | 77,234 | 256.0 | 97,648,896 | 3.7089 | +7.4% | 3.748 | conv:70.0%, bn:12.0%, pwconv:11.5% |
| `small_cnn` | 97,650 | 1024.0 | 117,965,312 | 5.7263 | +28.3% | 8.256 | pool:55.9%, conv:35.4%, bn:5.5% |
| `torchvision_mobilenet` | 1,519,618 | 288.0 | 35,293,184 | 7.2594 | +37.1% | 10.167 | dwconv:31.5%, pwconv:27.3%, bn:20.5% |

## 5. Pruebas de Significancia Estadistica (Mann-Whitney U, 1 hilo, batch=1)

| Comparacion | Mediana A (ms) | Mediana B (ms) | Ratio A/B | Estadistico U | p-valor | Significativo |
|---|---|---|---|---|---|---|
| `dt (x94)` vs `ds_cnn` | 0.3509 | 2.0890 | 0.168x | 0.0 | `4.829e-67` | True |
| `dt (x94)` vs `ultra_tiny` | 0.3509 | 0.8439 | 0.416x | 0.0 | `4.828e-67` | True |
| `lr (x94)` vs `ds_cnn` | 0.4630 | 2.0890 | 0.222x | 0.0 | `4.829e-67` | True |
| `lr (x94)` vs `ultra_tiny` | 0.4630 | 0.8439 | 0.549x | 0.0 | `4.828e-67` | True |
| `lr_dot (x94)` vs `ds_cnn` | 0.2251 | 2.0890 | 0.108x | 0.0 | `4.822e-67` | True |
| `lr_dot (x94)` vs `ultra_tiny` | 0.2251 | 0.8439 | 0.267x | 0.0 | `4.822e-67` | True |
| `mlp_16_dot (x94)` vs `ds_cnn` | 0.2299 | 2.0890 | 0.110x | 0.0 | `4.825e-67` | True |
| `mlp_16_dot (x94)` vs `ultra_tiny` | 0.2299 | 0.8439 | 0.272x | 0.0 | `4.824e-67` | True |
| `rf (x94)` vs `ds_cnn` | 4.6162 | 2.0890 | 2.210x | 39886.0 | `2.661e-66` | True |
| `rf (x94)` vs `ultra_tiny` | 4.6162 | 0.8439 | 5.470x | 40000.0 | `4.830e-67` | True |

## 6. Ablacion Calidad vs Coste (`x85` vs `x9` vs `x94`)

| Clasificador | Subconjunto | Features | F1 | ROC-AUC | T_total (ms) | dF1 vs x85 | d_ms vs x85 | dF1 / d_ms |
|---|---|---|---|---|---|---|---|---|
| `dt` | `x85_only` | 85 | 0.5417 | 0.7572 | 0.1079 | +0.0000 | +0.0000 | +0.0000 |
| `dt` | `x9_only` | 9 | 0.5116 | 0.7481 | 0.3750 | -0.0300 | +0.2671 | -0.1124 |
| `dt` | `x94_full` | 94 | 0.7037 | 0.8633 | 0.3115 | +0.1620 | +0.2037 | +0.7955 |
| `knn` | `x85_only` | 85 | 0.2222 | 0.8057 | 0.5360 | +0.0000 | +0.0000 | +0.0000 |
| `knn` | `x9_only` | 9 | 0.5116 | 0.6902 | 0.8639 | +0.2894 | +0.3280 | +0.8825 |
| `knn` | `x94_full` | 94 | 0.2857 | 0.7924 | 0.7268 | +0.0635 | +0.1908 | +0.3328 |
| `lr` | `x85_only` | 85 | 0.7200 | 0.9311 | 0.2303 | +0.0000 | +0.0000 | +0.0000 |
| `lr` | `x9_only` | 9 | 0.5185 | 0.7826 | 0.5230 | -0.2015 | +0.2927 | -0.6885 |
| `lr` | `x94_full` | 94 | 0.7925 | 0.9492 | 0.4435 | +0.0725 | +0.2132 | +0.3398 |
| `lr_dot` | `x85_only` | 85 | 0.7200 | 0.9311 | 0.0353 | +0.0000 | +0.0000 | +0.0000 |
| `lr_dot` | `x9_only` | 9 | 0.5185 | 0.7826 | 0.2534 | -0.2015 | +0.2181 | -0.9240 |
| `lr_dot` | `x94_full` | 94 | 0.7925 | 0.9492 | 0.2425 | +0.0725 | +0.2072 | +0.3497 |
| `mlp_16_dot` | `x85_only` | 85 | 0.8462 | 0.9583 | 0.0444 | +0.0000 | +0.0000 | +0.0000 |
| `mlp_16_dot` | `x9_only` | 9 | 0.5763 | 0.7856 | 0.2615 | -0.2699 | +0.2172 | -1.2428 |
| `mlp_16_dot` | `x94_full` | 94 | 0.8571 | 0.9636 | 0.2515 | +0.0110 | +0.2071 | +0.0531 |
| `rf` | `x85_only` | 85 | 0.7727 | 0.9295 | 5.4319 | +0.0000 | +0.0000 | +0.0000 |
| `rf` | `x9_only` | 9 | 0.4762 | 0.7970 | 5.6978 | -0.2965 | +0.2659 | -1.1152 |
| `rf` | `x94_full` | 94 | 0.7273 | 0.9409 | 4.5107 | -0.0455 | -0.9212 | +0.0493 |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.0684 | 936.0 |
| `ds_cnn` | cuda | 1 | 1.2554 | 796.6 |
| `tiny_cnn` | cuda | 1 | 0.8057 | 1241.1 |
| `mobilenet_like` | cuda | 1 | 3.3303 | 300.3 |
| `tiny_resnet` | cuda | 1 | 1.6781 | 595.9 |
| `small_cnn` | cuda | 1 | 0.9459 | 1057.2 |
| `torchvision_mobilenet` | cuda | 1 | 11.1048 | 90.1 |
| `ultra_tiny` | cuda | 32 | 0.0445 | 22484.5 |
| `ds_cnn` | cuda | 32 | 0.2110 | 4738.5 |
| `tiny_cnn` | cuda | 32 | 0.2050 | 4877.6 |
| `mobilenet_like` | cuda | 32 | 0.2299 | 4348.9 |
| `tiny_resnet` | cuda | 32 | 0.1785 | 5602.9 |
| `small_cnn` | cuda | 32 | 0.2323 | 4304.1 |
| `torchvision_mobilenet` | cuda | 32 | 0.2897 | 3452.2 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **27.8%** (0.0629 ms) del tiempo por etapas (E2E=0.1946 ms, gap de despacho Python/ctypes=+0.0314 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.2251 ms**, F1=0.7925), mientras el maximo F1 lo obtiene `mlp_16_dot (x94)` (**F1=0.8571**, 0.2299 ms).
