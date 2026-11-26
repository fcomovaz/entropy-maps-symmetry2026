# Profiling Cientifico y Comparacion 100% CPU: `mvtec-ad-inverted_screw`

## 1. Objetivo y Protocolo Medido

- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads=1` principal), tamano de lote (`batch_size=1` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.
- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).
- **Entorno registrado:** CPU `Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz` (4 nucleos fisicos / 8 logicos), RAM 15.9 GB, SO `Windows-10-10.0.19045-SP0`, PyTorch `2.14.0+cu126`, NumPy `2.2.6`, scikit-learn `1.7.2`, seed `42`, commit `99bc026`.
- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.

## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)

| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr_dot (x94)` | ml | cpu | 95 / 94 | 0.6452 | 0.8721 | 0.3375 (desc) | 0.0070 | **0.3445** | 0.1196 | [0.3434, 0.3451] | 2902.8 | - |
| `mlp_16_dot (x94)` | ml | cpu | 1537 / 94 | 0.6333 | 0.8484 | 0.3375 (desc) | 0.0150 | **0.3528** | 0.1242 | [0.3515, 0.3550] | 2834.5 | - |
| `dt (x94)` | ml | cpu | 15 / 94 | 0.5556 | 0.7969 | 0.3375 (desc) | 0.0837 | **0.4395** | 0.1711 | [0.4211, 0.4717] | 2275.6 | - |
| `lr (x94)` | ml | cpu | 95 / 94 | 0.6452 | 0.8721 | 0.3375 (desc) | 0.2057 | **0.5424** | 0.1024 | [0.5415, 0.5433] | 1843.5 | - |
| `ultra_tiny` | cnn | cpu | 1154 / 0 | 0.5581 | 0.8576 | 0.0371 (prep) | 0.7620 | **0.8004** | 0.0838 | [0.7984, 0.8021] | 1249.4 | 0.00e+00 |
| `knn (x94)` | ml | cpu | 36096 / 94 | 0.5128 | 0.8247 | 0.3375 (desc) | 0.4981 | **0.8445** | 0.1230 | [0.8357, 0.8548] | 1184.1 | - |
| `ds_cnn` | cnn | cpu | 12866 / 0 | 0.4000 | 0.8027 | 0.0371 (prep) | 1.8734 | **1.9108** | 0.7192 | [1.8678, 1.9434] | 523.3 | 0.00e+00 |
| `mobilenet_like` | cnn | cpu | 70786 / 0 | 0.7200 | 0.9300 | 0.0371 (prep) | 3.2327 | **3.2698** | 0.6147 | [3.2045, 3.3465] | 305.8 | 0.00e+00 |
| `tiny_resnet` | cnn | cpu | 77234 / 0 | 0.5128 | 0.8733 | 0.0371 (prep) | 3.5443 | **3.5813** | 1.0148 | [3.4024, 3.7687] | 279.2 | 0.00e+00 |
| `rf (x94)` | ml | cpu | 7680 / 94 | 0.6829 | 0.8611 | 0.3375 (desc) | 4.2520 | **4.5972** | 0.7237 | [4.5055, 4.7168] | 217.5 | - |
| `tiny_cnn` | cnn | cpu | 23538 / 0 | 0.6531 | 0.8918 | 0.0371 (prep) | 4.6704 | **4.7074** | 1.2402 | [4.5489, 4.8807] | 212.4 | 0.00e+00 |
| `small_cnn` | cnn | cpu | 97650 / 0 | 0.6818 | 0.8947 | 0.0371 (prep) | 5.8915 | **5.9285** | 1.3499 | [5.7632, 6.1212] | 168.7 | 0.00e+00 |
| `torchvision_mobilenet` | cnn | cpu | 1519618 / 0 | 0.4000 | 0.8252 | 0.0371 (prep) | 6.5371 | **6.5742** | 1.1219 | [6.4083, 6.7241] | 152.1 | 1.33e-02 |

## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E, backend `c_single_pass`)

- Backend de entropia activo: **`c_single_pass`** (solicitado: `auto`).
- Inicializacion amortizable (`build_entropy_plan` + LUTs): **7.4104 ms** (se computa una sola vez).
- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **0.000e+00**.
- Suma de medianas por etapa: **0.2709 ms** vs llamada formal E2E: **0.2646 ms** (gap de dispatch/asignaciones Python = **+0.0063 ms**, +2.3%).

| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |
|---|---|---|---|---|---|
| `asymmetry_g2` | 2x2 | 0.0708 | 26.1% | 2.1 | `O(g^2*K) [4x3]` |
| `asymmetry_g8` | 8x8 | 0.0679 | 25.1% | 2.6 | `O(g^2*K) [64x3]` |
| `asymmetry_g4` | 4x4 | 0.0673 | 24.8% | 2.1 | `O(g^2*K) [16x3]` |
| `c_fine_hist` | 8x8 | 0.0311 | 11.5% | 1.9 | `O(H*W) [C99 fused quant+hist, H=W=128]` |
| `c_entropy_lut` | all | 0.0142 | 5.2% | 2.1 | `O(F_e*B) [C99 nlogn LUT, F_e=85]` |

## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)

| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |
|---|---|---|---|---|
| `ultra_tiny` | cuda | 1 | 1.0051 | 994.9 |
| `ds_cnn` | cuda | 1 | 1.2498 | 800.1 |
| `tiny_cnn` | cuda | 1 | 0.7345 | 1361.5 |
| `mobilenet_like` | cuda | 1 | 2.4853 | 402.4 |
| `tiny_resnet` | cuda | 1 | 2.0762 | 481.7 |
| `small_cnn` | cuda | 1 | 0.9353 | 1069.2 |
| `torchvision_mobilenet` | cuda | 1 | 9.1205 | 109.6 |
| `ultra_tiny` | cuda | 32 | 0.0410 | 24382.8 |
| `ds_cnn` | cuda | 32 | 0.2086 | 4792.9 |
| `tiny_cnn` | cuda | 32 | 0.2061 | 4852.0 |
| `mobilenet_like` | cuda | 32 | 0.2298 | 4351.4 |
| `tiny_resnet` | cuda | 32 | 0.1778 | 5623.4 |
| `small_cnn` | cuda | 32 | 0.2320 | 4310.6 |
| `torchvision_mobilenet` | cuda | 32 | 0.2784 | 3591.6 |

## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones

- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.
- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.
- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.

## 9. Recomendaciones Derivadas de los Datos

- **Perfil actual de x94 (`c_single_pass`):** `asymmetry_g2` concentra **26.1%** (0.0708 ms) del tiempo por etapas (E2E=0.2646 ms, gap de despacho Python/ctypes=+0.0063 ms); la piramide de histogramas ya opera en C99 en un solo paso.
- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `lr_dot (x94)` (**0.3445 ms**, F1=0.6452), mientras el maximo F1 lo obtiene `mobilenet_like` (**F1=0.7200**, 3.2698 ms).

## Nota Metodologica (Variante Invertida — Short Descriptor Check)

Siguiendo el protocolo §10, las variantes invertidas ejecutan comparacion CPU completa (Fase A) y esta verificacion corta del descriptor x94 para confirmar empiricamente que la inversion fotometrica (`255 - x`) conserva el mismo perfil computacional (E2E mediana batch=1 = **0.2646 ms**, staged sum = **0.2709 ms**).
