# Profiling Cientifico y Comparacion 100% CPU: `kolektorsdd2-inverted`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `99bc026`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.4153 | 0.8151 | 0.2086 (desc) | 0.0079 | **0.2190** | 0.0435 | [0.2162, 0.2225] | 4566.2 | - |
| `mlp_16_dot (x94)` | ml | cpu | 1537 / 94 | 0.4906 | 0.8311 | 0.2086 (desc) | 0.0125 | **0.2213** | 0.0433 | [0.2205, 0.2247] | 4518.8 | - |
| `dt (x94)` | ml | cpu | 305 / 94 | 0.3140 | 0.6228 | 0.2086 (desc) | 0.1371 | **0.3490** | 0.0462 | [0.3470, 0.3511] | 2865.3 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.4153 | 0.8151 | 0.2086 (desc) | 0.2207 | **0.4550** | 0.1109 | [0.4400, 0.4647] | 2198.0 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.5185 | 0.8179 | 0.0417 (prep) | 0.8498 | **0.8959** | 0.2173 | [0.8837, 0.9229] | 1116.3 | 0.00e+00 |
| `knn (x94)` | ml | cpu | 219114 / 94 | 0.1500 | 0.6607 | 0.2086 (desc) | 0.7382 | **0.9895** | 0.2980 | [0.9517, 1.0309] | 1010.7 | - |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.5665 | 0.8673 | 0.0417 (prep) | 2.2441 | **2.2859** | 0.7234 | [2.2058, 2.3739] | 437.5 | 0.00e+00 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.4845 | 0.7900 | 0.0417 (prep) | 3.3510 | **3.3925** | 0.8487 | [3.2086, 3.5299] | 294.8 | 0.00e+00 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.6509 | 0.9131 | 0.0417 (prep) | 3.4267 | **3.4682** | 0.4628 | [3.4016, 3.5330] | 288.3 | 0.00e+00 |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.2098 | 0.8281 | 0.0417 (prep) | 4.2828 | **4.3285** | 0.7351 | [4.2526, 4.3979] | 231.0 | 0.00e+00 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.5682 | 0.8684 | 0.0417 (prep) | 6.7857 | **6.8308** | 0.4126 | [6.7647, 6.9069] | 146.4 | 0.00e+00 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.1975 | 0.5369 | 0.0417 (prep) | 7.3063 | **7.3481** | 1.1935 | [7.1701, 7.5822] | 136.1 | 1.01e-03 |
| `rf (x94)` | ml | cpu | 13728 / 94 | 0.4237 | 0.7978 | 0.2086 (desc) | 8.4956 | **8.7788** | 0.7558 | [8.6469, 8.9017] | 113.9 | - |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **8.5964 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2154 ms** vs llamada formal E2E: **0.1849 ms** (gap de dispatch/asignaciones Python = **+0.0305 ms**, +14.1%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0564 | 26.2% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g4` | 4x4 | 0.0513 | 23.8% | 2.1 | `O(g^2*K) [16x3]` |
| `asymmetry_g8` | 8x8 | 0.0509 | 23.6% | 2.6 | `O(g^2*K) [64x3]` |
| `c_fine_hist` | 8x8 | 0.0292 | 13.6% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0113 | 5.3% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 0.9560 | 1046.0 |
| `ds_cnn` | cuda | 1 | 1.3233 | 755.7 |
| `tiny_cnn` | cuda | 1 | 0.7804 | 1281.4 |
| `mobilenet_like` | cuda | 1 | 2.3062 | 433.6 |
| `tiny_resnet` | cuda | 1 | 1.7562 | 569.4 |
| `small_cnn` | cuda | 1 | 0.9855 | 1014.8 |
| `torchvision_mobilenet` | cuda | 1 | 9.1937 | 108.8 |
| `ultra_tiny` | cuda | 32 | 0.0432 | 23138.1 |
| `ds_cnn` | cuda | 32 | 0.2114 | 4730.3 |
| `tiny_cnn` | cuda | 32 | 0.2059 | 4856.0 |
| `mobilenet_like` | cuda | 32 | 0.2278 | 4390.1 |
| `tiny_resnet` | cuda | 32 | 0.1782 | 5610.6 |
| `small_cnn` | cuda | 32 | 0.2323 | 4305.4 |
| `torchvision_mobilenet` | cuda | 32 | 0.3008 | 3324.1 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **26.2%** (0.0564 ms) del tiempo por etapas (E2E=0.1849 ms, gap de despacho Python/ctypes=+0.0305 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.2190 ms**, F1=0.4153), mientras el maximo F1 lo obtiene `tiny_resnet` (**F1=0.6509**, 3.4682 ms).

## Nota Metodologica (Variante Invertida — Short Descriptor Check)

Siguiendo el protocolo §10, las variantes invertidas ejecutan comparacion CPU completa (Fase A) y esta verificacion corta del descriptor x94 para confirmar empiricamente que la inversion fotometrica (`255 - x`) conserva el mismo perfil computacional (E2E mediana batch=1 = **0.1849 ms**, staged sum = **0.2154 ms**).
