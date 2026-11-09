# Complete profiling and CPU fair comparison

## Issue

* All the experiments have been done and we have some important results as: 1. the x9 symmetry descriptors increase the performance in classification, so adding symmetry metrics is a real improvement, not only an speculation; 2. the x94 descriptor takes a little bit longer in get the descriptor but maybe the computational behind is less since the parameters is almost 100x less than a cnn.
* That is the problem, we don't really know exactly what part is taking more time so, we need to create a detailed profiling, rather a scientifically based profiling in python to see in each stage of the entropy and the cnn what process is taking longer, create a report and from there, later, take decisions in the research.
* y mucho cuidado, leyendo hallazgos de los experimentos veo que **la CNN ligera (ds_cnn, 12.8k parámetros) en GPU clasifica directamente desde píxeles en solo 0.21ms** lo cual hace ver como que todo tarda mucho pero realmente esto es  en GPU y a mi me interesa comparar el ML clásico en CPU y el cnn en cpu para después tomar todo lo demás en cuanta en el performance.
* so, lo primero sería coregir las ejecuciones porque todo lo que tengo en results comparison es entonces cnn gpu vs ml clasico que es obviamente cpu, entonces hacer nuevos experimentos en cpu y luego hacer el profiling bien hecho, bonito y cientifico.

## Folder Scructure

The results of each experiments are located in `D:\Publications\2026-10-Symmetry-Entropy\code\results` and `D:\Publications\2026-10-Symmetry-Entropy\code\results\comparison` and each folder has a readme well done that can be read in `D:\Publications\2026-10-Symmetry-Entropy\code\memory` were all the readmes are located.

## Plan

> Don't execute anything, just plan it

- [x] Read AGENTS.md file to know the general rules of the project.
- [x] Create a detailed plan for this task and update it here to implement it later for an agent, so it needs to be detailed.
- [x] Update this file with the plan, stages and everything to resolve and implement the task.



---

# PLAN DETALLADO DE IMPLEMENTACIÓN (para agente ejecutor)

> Estado: **solo planificación**. Nada de lo siguiente se ha ejecutado. Respetar `AGENTS.md`: TOML en `configs/`, `src/` sin valores hardcodeados, `pathlib`, `from __future__ import annotations`, `@dataclass(frozen=True, slots=True)`, CLI con `argparse` + `main(argv) -> int` + `python -m`, `logging` en vez de `print`, `.venv\Scripts\python.exe`, seeds explícitas, no fabricar resultados, tests antes de dar por terminado.

## 0. Diagnóstico (hallazgos de la inspección)

| Hallazgo | Evidencia | Consecuencia |
|---|---|---|
| La comparación actual mezcla dispositivos: CNN medida con `device = "auto"` (→ CUDA) y ML clásico en CPU | `configs/cnn/*.toml` `[cnn.training] device="auto"`; `results/comparison/*.csv` (`ds_cnn` 0.21 ms vs `dt (x94)` 0.48 ms) | La comparación `t_total_ms` no es justa. |
| `t_inf_model_ms` de la CNN viene de la evaluación de test (según `device` de entrenamiento), no del benchmark CPU aislado | `experiments/compare_baseline_vs_cnn.py::_collect_metrics` (`timing.get(...)`) | Hay que usar un benchmark CPU explícito. |
| `src/models/cnn/benchmark.py` ya existe (CPU/CUDA, `threads`, warmup, mediana/IQR, FLOPs, VRAM) y la config ya trae `devices=["cpu","cuda"]`, `threads=1`, `batch_sizes=[1,32]` | `src/models/cnn/benchmark.py`, `[cnn.benchmark]` | Reutilizar, no reescribir. |
| La latencia del descriptor x94 se mide como **una sola tirada** sobre imágenes aleatorias (`np.random`), todo el lote junto, sin desglose | `measure_descriptor_latency_ms` | Sin distribución, sin etapas, y datos aleatorios pueden sesgar (p. ej. histogramas). |
| `t_inf_model_ms` del ML clásico se mide con `perf_counter` sobre el lote de test completo / N | `descriptor_classifier.py::train_and_evaluate` (L864-871) | Mezcla batch con per-sample; RF usa `n_jobs=-1` (multihilo) mientras CNN usa 1 hilo. |
| `flops = 0` para ML clásico y `params = 94` (cuenta de features, no de parámetros del modelo) | CSV comparison | Parámetros/FLOPs de ML no comparables; hay que definirlos bien. |

Pipelines a perfilar:
- **ML clásico (x94)**: imagen uint8 128×128 → cuantización LUT (`quantize_pixels`) → histogramas por región (`region_histograms`, grids 1/2/4/8) → entropía normalizada (`normalized_entropy`) → [x85] → asimetría (`compute_asymmetry_features`, escalas 2/4/8 → x9) → vector x94 → clasificador (dt/knn/lr/rf).
- **CNN**: imagen → tensor `[B,1,128,128]` (normalización) → forward por capa → logits. Modelos: ds_cnn, mobilenet_like, small_cnn, tiny_cnn, tiny_resnet, torchvision_mobilenet, ultra_tiny.

## 1. Estructura de entregables

```
configs/profiling/kolektorsdd2.toml        # NUEVO: parámetros de perfilado
configs/profiling/mvtec_ad.toml            # NUEVO (mismo esquema)
src/profiling/__init__.py                  # NUEVO
src/profiling/timing.py                    # temporizador robusto + estadísticas
src/profiling/environment.py               # captura de HW/SW/threads (reproducibilidad)
src/profiling/descriptor_stages.py         # perfil por etapas del x94
src/profiling/classifier_profile.py        # latencia/params/FLOPs del ML clásico
src/profiling/cnn_layers.py                # perfil por capa de la CNN en CPU
src/profiling/report.py                    # tablas CSV/JSON + figuras + README
experiments/benchmark_cpu_fair.py          # NUEVO: Fase A (comparación CPU justa)
experiments/profile_pipelines.py           # NUEVO: Fase B (profiling detallado)
tests/test_profiling_*.py                  # NUEVO
results/profiling/<dataset>/...            # salidas
understanding/12_profiling_stages/         # laboratorio: cronometraje manual por etapa + matplotlib
```
Actualizar: `experiments/README.md`, `src/README.md`, `tests/README.md`, `results/README.md`, `requirements.txt` (si se añade `psutil`; preferir librería estándar + `torch`/`numpy` ya instalados) y exportar readmes a `memory/` (con `memory/export_readmes.ps1`).

## 2. Fase 0 — Entorno y prerequisitos (verificación, sin cambios)

- [ ] Leer `AGENTS.md`, `experiments/README.md`, `src/models/cnn/benchmark.py`, `compare_baseline_vs_cnn.py`, `descriptor_classifier.py`.
- [ ] Confirmar existencia de modelos CNN entrenados (`results/cnn/kolektorsdd2/full/...` checkpoints) y CSVs de descriptores (`data/csv/*`). Si faltan checkpoints, el perfil de latencia puede usar pesos aleatorios con seed fija (la latencia no depende de los pesos), pero **las métricas de calidad deben reutilizar checkpoints existentes o reentrenar en CPU** (ver Fase A).
- [ ] Ejecutar `tests/test_environment.py` y los tests CNN/clasificación actuales como línea base (`.\.venv\Scripts\python.exe -m unittest discover tests`).

## 3. Fase A — Corregir la comparación: todo en CPU

Objetivo: tabla en la que ML clásico y CNN se miden **en CPU, mismas condiciones, mismos datos**.

### A.1 Protocolo de medición justo (definir en TOML `[profiling.protocol]`)
- `device = "cpu"` forzado para CNN (nunca `auto`). La GPU queda como columna secundaria *separada* y etiquetada (`device=cuda`) solo como referencia, nunca mezclada en el mismo ranking.
- `threads_list = [1, <n_cores_físicos>]`: reportar **1 hilo (comparable a embebido)** como principal y multihilo como secundario. Aplicar el mismo número de hilos a: `torch.set_num_threads`, `OMP_NUM_THREADS`/`MKL_NUM_THREADS` vía `threadpoolctl` (scikit-learn ya lo trae como dependencia) y `n_jobs` de RF/kNN (`n_jobs=1` en modo principal; no tocar el TOML de entrenamiento: sobreescribir en tiempo de perfilado con `set_params`).
- `batch_sizes = [1, 32]`: **batch=1 (latencia, caso embebido) es la métrica principal**; batch=32 para throughput.
- `warmup_iters ≥ 20`, `timed_iters ≥ 200` (batch 1), `repeats`/`seeds` ≥ 5 procesos o bloques; reportar **mediana, IQR, p95, media, std, min, max** (no solo media).
- Temporizador: `time.perf_counter_ns()`; desactivar GC durante la medición (`gc.disable()` en contexto, re-habilitar al final); `torch.inference_mode()`; para CUDA `torch.cuda.synchronize()` (solo referencia).
- Datos de entrada **reales**: muestrear N imágenes del split test (seed fija, p. ej. 200) en lugar de `np.random`. Pre-cargar fuera del cronómetro; cronometrar solo la inferencia (excluir I/O de disco, indicando explícitamente que se excluye).
- Estado térmico/ruido: cerrar apps pesadas, perfil de energía "Alto rendimiento", registrar en `environment.json`; repetir la corrida completa 3 veces y reportar variabilidad inter-corrida.
- Definir latencias con precisión (alineado con AGENTS.md):
  - `T_selector`/`T_descriptor`: imagen → vector x94 (por muestra, batch=1 y batch=N amortizado).
  - `T_model`: vector → predicción (ML) o tensor → logits (CNN).
  - `T_preprocess` CNN: conversión a tensor + normalización (incluirla en CNN para ser justos; el ML paga su "preprocess" en el descriptor).
  - `T_total = T_descriptor + T_model` (ML) y `T_total = T_preprocess + T_model` (CNN).

### A.2 Nuevo experimento `experiments/benchmark_cpu_fair.py`
- CLI: `--config`, `--dataset`, `--object`, `--inverted`, `--threads`, `--batch-sizes`, `--models`, `--output-dir`, `--dry-run`.
- Para ML clásico (dt, knn, lr, rf): entrenar (o cargar pickles si existen; si no, entrenar con la config de `configs/classification/*.toml`) y medir `predict` **muestra a muestra (batch=1) y por lote**, con un clasificador con `n_jobs` coherente con `threads`.
- Para CNN: reutilizar `benchmark_model_latency(..., device=cpu, threads=...)` pero alimentando **muestras reales**; si `benchmark.py` solo admite `torch.randn`, extenderlo de forma retrocompatible con un parámetro opcional `inputs: torch.Tensor | None`.
- **Re-evaluación de calidad en CPU**: ejecutar `evaluate` de las CNN con `--device cpu` desde los checkpoints existentes para confirmar que las métricas (F1, ROC-AUC) no cambian (deben coincidir salvo ruido numérico ~1e-6; si difieren, reportarlo). Si no hay checkpoints, reentrenar con `device="cpu"` solo si el usuario lo aprueba (puede tardar mucho; el entrenamiento en GPU es válido porque la calidad no depende del device — documentar esta decisión metodológica).
- Salida: `results/comparison_cpu/<dataset>_baseline_vs_cnn_cpu.csv` con columnas: `model, family, device, threads, batch_size, params_model, params_descriptor, macs, flops, t_preprocess_ms_median, t_descriptor_ms_median, t_model_ms_median, t_total_ms_{median,iqr,p95,mean,std}, throughput_samples_s, peak_ram_mb, accuracy, balanced_accuracy, f1, roc_auc, pr_auc, seed, n_samples, timestamp, git_commit`.
- **No sobrescribir** `results/comparison/` (GPU vs CPU antiguo): conservarlo como histórico y marcarlo en su README como `mixed-device (deprecated)`. Opcionalmente, añadir una columna `device` retroactiva solo en el nuevo directorio.
- Figuras (matplotlib): F1 vs `t_total` (mediana, con barras IQR), ROC-AUC vs `t_total`, F1 vs params, throughput; mismas escalas log; una figura por `threads`.
- Parámetros y FLOPs de ML clásico:
  - `params_model`: contar reales (LR: coef+bias=95; DT: nodos/hojas; RF: suma de nodos de todos los árboles; kNN: tamaño de la memoria de entrenamiento `N_train × 94`). Documentar la definición — el "94" actual son features, no parámetros.
  - `macs/flops` aproximados: DT = profundidad media (comparaciones); RF = Σ profundidad media; LR = 94 MAC; kNN = `N_train × 94`. Marcar como **estimación analítica** y no mezclarla con FLOPs medidos de PyTorch (`FlopCounterMode`).
  - FLOPs del descriptor: estimación analítica por etapa (ver B.1) o conteo de operaciones documentado.

## 4. Fase B — Profiling científico detallado

Objetivo: saber **qué etapa consume el tiempo** en el descriptor x94 y en cada CNN.

### B.1 Perfil por etapas del descriptor x94 (`src/profiling/descriptor_stages.py`)
Descomponer (sin alterar la lógica de `src/features`; **importar y llamar** las funciones existentes `quantize_pixels`, `region_histograms`, `normalized_entropy`, `compute_asymmetry_features`, `reconstruct_grids`) y cronometrar cada una **por escala de grid (1,2,4,8)** y por muestra con batch=1:
1. Plan/LUT precomputados (`build_entropy_plan`, `build_quantization_lut`, `build_shannon_term_lut`) → medir **una vez** (costo de inicialización, amortizable) y reportar por separado.
2. `quantize_pixels` (uint8 → bins).
3. `region_histograms` por `g`.
4. `normalized_entropy` por `g`.
5. Concatenación a x85.
6. `reconstruct_grids` + `compute_asymmetry_for_grid` por escala (x9).
7. Concatenación final x94 + posible scaling/`StandardScaler` (para lr/knn).
8. `predict` del clasificador.
- Si `compute_entropy_features` no expone etapas intermedias con granularidad suficiente, **no modificar `src/features`**: replicar la secuencia en el módulo de profiling y **verificar con test** que `descriptor_stages.run(img)` == `compute_entropy_features + compute_asymmetry_features` (`np.allclose`) — mecanismo de validación obligatorio.
- Medición: `perf_counter_ns` por etapa con warmup, N muestras reales, reportar % del total y mediana/IQR. Detectar overhead de Python: comparar suma de etapas vs medición end-to-end (el gap = overhead de llamadas, debe reportarse).
- Complementar con profilers estándar (documentando que **solo localizan**, el tiempo oficial viene del cronómetro manual porque los profilers distorsionan):
  - `cProfile` + `pstats` (top funciones acumuladas) → `results/profiling/.../cprofile_descriptor.txt`.
  - `line_profiler` solo si el usuario lo aprueba (dependencia extra; si se usa, instalarlo en `.venv` y fijar versión en `requirements.txt`).
  - `tracemalloc` + `psutil`(RSS) o `resource` para pico de RAM por etapa (en Windows usar `psutil`; si no se quiere dependencia, `tracemalloc` para asignaciones NumPy).
- Análisis de escalabilidad: variar `image_size` (64/128/256, si el descriptor lo admite) y `grid_levels`/`bins` (p. ej. bins 8/16/32) → curva tiempo vs configuración (ablation de costo). Esto sustenta decisiones de investigación (qué parte del descriptor merece optimización).
- Análisis de complejidad esperado vs medido (marcar claramente "esperado" vs "medido"): histogramas O(HW) por nivel, entropía O(g²·bins), asimetría O(g²).

### B.2 Perfil por capa de la CNN en CPU (`src/profiling/cnn_layers.py`)
- Para cada arquitectura del registry (`architectures/registry.py`), batch=1, `threads=1` (y multihilo secundario), CPU:
  - **Hooks de forward** (`register_forward_pre_hook`/`forward_hook`) con `perf_counter_ns` por módulo hoja → tiempo por capa, % del total, tipo de capa (Conv, DWConv, BN, ReLU, Pool, Linear).
  - **`torch.profiler`** (`ProfilerActivity.CPU`, `record_shapes=True`, `profile_memory=True`, `with_flops=True`) → tabla por operador (`aten::conv2d`, `aten::batch_norm`...), exportar `key_averages().table(sort_by="cpu_time_total")` y chrome trace (`.json`) a `results/profiling/...`.
  - MACs/FLOPs y parámetros por capa (reutilizar `estimate_flops` / `torchinfo`) → ratio **tiempo/FLOP** por capa (intensidad: capas memory-bound vs compute-bound; p. ej. depthwise suele ser lenta respecto a sus FLOPs).
  - Memoria: tamaño de activaciones máximo por capa (`[B,C,H,W]`), tamaño del modelo en MB (fp32), relevante para hardware embebido.
- Validar: suma de tiempos por capa ≈ forward completo (reportar overhead de hooks; aceptar ≤ ~15 %, si no, preferir `torch.profiler` como fuente de verdad).
- Además perfilar: carga de checkpoint y primera inferencia (cold start) vs régimen estacionario; separar "primera llamada" del warmup para reportar cold-start.

### B.3 Perfil del clasificador clásico (`src/profiling/classifier_profile.py`)
- Latencia `predict`/`predict_proba` batch=1 vs batch=N por modelo, con `n_jobs=1`; para kNN separar costo de scaler + búsqueda. Destacar que en batch=1 RF/sklearn tiene overhead fijo de ~decenas de µs a ms (joblib/validación de entrada); medirlo y, opcionalmente, comparar con `predict` sobre `estimator.tree_` directo o `check_input=False` **solo como análisis aparte** (no cambiar el baseline).
- Tamaño en memoria (pickle bytes) y conteo real de parámetros (ver A.2).

### B.4 Análisis comparativo y estadística
- Intervalos de confianza por bootstrap (p. ej. 1000 remuestreos) sobre las latencias por muestra; usar **mediana + IC95 %**.
- Test no paramétrico (Mann-Whitney U, `scipy.stats`) entre `x94+dt/lr` y `ds_cnn/ultra_tiny` para afirmar diferencias de latencia solo si son significativas.
- Ablación de aporte de calidad vs costo: x85 vs x9 vs x94 (ya existen subsets) → `ΔF1 / Δms` (eficiencia marginal de los descriptores de simetría). Esta es la conexión directa con el hallazgo 1 del issue.
- Puntos de equilibrio (Pareto): marcar frontera F1 vs `t_total` y F1 vs FLOPs/params, **solo en CPU, 1 hilo, batch 1**.
- Sensibilidad a hilos: curva latencia vs `threads ∈ {1,2,4,N}` para CNN y RF.

### B.5 Reporte (`src/profiling/report.py` + `results/profiling/<dataset>/README.md`)
- Archivos: `environment.json` (CPU modelo, núcleos, frecuencia, RAM, SO, versiones de Python/torch/numpy/sklearn, `torch.get_num_threads`, `torch.__config__.parallel_info()`, commit git, seed), `descriptor_stage_times.csv`, `cnn_layer_times_<model>.csv`, `classifier_times.csv`, `comparison_cpu.csv`, figuras PNG:
  1. Barras apiladas: tiempo por etapa del x94 (% y ms).
  2. Heatmap/treemap de tiempo por capa por arquitectura.
  3. Pareto F1 vs latencia CPU.
  4. Latencia vs threads.
  5. Escalabilidad del descriptor (bins/grid/image_size).
  6. Throughput batch=32.
- README del reporte con: objetivo, protocolo, entorno, tablas, **hallazgos medidos** (separados de hipótesis/esperado), limitaciones (CPU de escritorio ≠ MCU/embebido; sin medir energía; Python overhead; etc.) y **recomendaciones para siguientes decisiones** (p. ej. ¿vale la pena vectorizar/portar a C/NumPy la etapa X?). No escribir conclusiones antes de tener los números.

## 5. Fase C — `understanding/` (laboratorio, regla *understanding-first*)
`understanding/12_profiling_stages/`: script que, para **una** imagen, imprime tiempos manuales por etapa (imagen → cuantización → histograma → entropía → asimetría) y los grafica con matplotlib (barras apiladas), y para una CNN muestra tiempo por capa; comparar `perf_counter` manual vs `cProfile` vs `torch.profiler`, y explicar por qué difieren. Con README propio. No mover nada a `src/` hasta entenderlo (el módulo `src/profiling` se escribe después de validar esto).

## 6. Fase D — Tests (`tests/`)
- `test_profiling_timing.py`: estadísticas (mediana/IQR/p95) con datos sintéticos conocidos; el cronómetro devuelve ≥ 0 y N muestras; GC restaurado.
- `test_profiling_descriptor_stages.py`: **equivalencia numérica** de etapas vs `compute_entropy_features`+`compute_asymmetry_features`; suma de % ≈ 100; formas `[N, 94]`.
- `test_profiling_cnn_layers.py`: modelo pequeño sintético → todas las capas hoja reportadas; suma por capa ≈ total (tolerancia); funciona en CPU sin CUDA.
- `test_profiling_config.py`: carga TOML, fail-fast ante claves faltantes, rutas con `pathlib` ancladas al project root.
- `test_benchmark_cpu_fair_cli.py`: `main([...,"--dry-run"])` devuelve 0; fuerza `device=cpu`; rechaza `cuda` en modo principal.
- Ejecutar toda la suite: `.\.venv\Scripts\python.exe -m unittest discover tests`.

## 7. Orden de ejecución recomendado (checklist del agente)

- [ ] 1. Fase 0: baseline de tests y lectura de código.
- [ ] 2. Crear `configs/profiling/*.toml` (sin hardcodear en `src/`).
- [ ] 3. `understanding/12_profiling_stages/` (explorar y validar tiempos a mano).
- [ ] 4. `src/profiling/{timing,environment}.py` + tests.
- [ ] 5. `descriptor_stages.py` + test de equivalencia.
- [ ] 6. `cnn_layers.py`, `classifier_profile.py` + tests.
- [ ] 7. Extender `benchmark.py` de forma retrocompatible (entrada real opcional) + test existente de benchmark sigue pasando.
- [ ] 8. `experiments/benchmark_cpu_fair.py` (Fase A) → ejecutar en CPU, 1 hilo y multi-hilo, `kolektorsdd2` primero, luego `kolektorsdd2-inverted` y `mvtec-ad` (bottle) si el tiempo lo permite.
- [ ] 9. `experiments/profile_pipelines.py` (Fase B) → ejecutar y generar `results/profiling/...`.
- [ ] 10. Reporte `README.md` con hallazgos reales; marcar `results/comparison/` como mixto (deprecated) sin borrarlo.
- [ ] 11. Actualizar READMEs (`experiments`, `src`, `tests`, `results`), `requirements.txt` si hubo nuevas dependencias, y ejecutar `memory/export_readmes.ps1`.
- [ ] 12. Suite completa de tests + verificación final (`implementation → execution → tests → expected output`), indicando explícitamente lo no verificado.

## 8. Criterios de aceptación (Definition of Done)
- Existe una tabla de comparación **100 % CPU** con mismo protocolo (hilos, batch, muestras reales) para x94+{dt,knn,lr,rf} y todas las CNN, con mediana/IQR/IC95 %, params reales, FLOPs/MACs (marcando medidos vs estimados), RAM y throughput.
- Ninguna fila mezcla dispositivos; GPU solo aparece en tabla aparte claramente etiquetada.
- Existe desglose por etapa del x94 y por capa de cada CNN, con validación de que la suma ≈ total.
- Los tests nuevos y existentes pasan; la ejecución real genera los CSV/PNG/README en `results/profiling/` y `results/comparison_cpu/`.
- Entorno y seeds registrados; resultados reproducibles (re-ejecución dentro de la variabilidad reportada).
- Documentación actualizada; sin cambios en la lógica de `src/features` ni borrado de resultados previos.

## 9. Riesgos y decisiones abiertas (confirmar con el usuario antes de ejecutar)
1. ¿Reentrenar CNNs en CPU o reutilizar checkpoints entrenados en GPU y solo re-evaluar/medir en CPU? (recomendado: reutilizar; la calidad no depende del device; documentar.)
2. ¿Añadir dependencias (`psutil`, `line_profiler`, `threadpoolctl`)? Preferencia: usar `tracemalloc`, `cProfile`, `torch.profiler` y `threadpoolctl` (ya viene con scikit-learn) para minimizar.
3. ¿Hilos principales: 1 (embebido) como referencia, multihilo secundario? (recomendado.)
4. ¿Datasets a cubrir: KolektorSDD2 (+ inverted) y MVTec-AD `bottle` como en los comparisons existentes, o más objetos?
5. Limitación: una CPU de escritorio no representa un MCU; para claims de hardware embebido se necesitarían FLOPs/RAM y, a futuro, medición en el dispositivo objetivo.


---

## 10. Decisiones confirmadas por el usuario (resuelven la sección 9)

1. **Checkpoints:** reutilizar los checkpoints entrenados en GPU; no reentrenar en CPU. La calidad (F1, ROC-AUC, etc.) no depende del device. Solo se re-evalúa en CPU para confirmar que las métricas coinciden (tolerancia ~1e-6) y se mide la latencia en CPU. Decisión metodológica a documentar en el README del reporte. Si faltara algún checkpoint, reportarlo antes de reentrenar.
2. **Dependencias:**
   - `psutil`: **sí**. Aporta RSS/pico de RAM por proceso y número de núcleos físicos en Windows, que `tracemalloc` no cubre (solo ve asignaciones de Python/NumPy). Instalar en `.venv` con `.venv/Scripts/python.exe -m pip install psutil` y fijar la versión en `requirements.txt`.
   - `line_profiler`: **no**. El cronómetro manual por etapa más `cProfile` y `torch.profiler` ya dan la granularidad necesaria. `line_profiler` añade overhead y una dependencia sin aportar información nueva. Eliminar de la sección B.1 la mención a `line_profiler`.
   - `threadpoolctl`: ya viene con scikit-learn, se usa sin añadirlo a `requirements.txt` salvo que el import falle.
3. **Hilos:** 1 hilo es la referencia principal. Multihilo (`{2, 4, núcleos físicos}`) es secundario y va en tablas y figuras separadas.
4. **Datasets (configurables):**
   - Obligatorios: `kolektorsdd2`, `kolektorsdd2-inverted`, `mvtec-ad` objeto `bottle`, `mvtec-ad-inverted` objeto `bottle`.
   - Extensible: `configs/profiling/mvtec_ad.toml` define una lista `objects = ["bottle"]`. Para añadir `metal_nut`, `tile` u otros basta editar esa lista, sin tocar código.
   - Cada combinación `(dataset, object, inverted)` genera su propio subdirectorio de resultados: `results/profiling/<tag>/` y `results/comparison_cpu/<tag>_baseline_vs_cnn_cpu.csv`.
   - El experimento admite `--all-objects` y `--objects a b c` para barridos. Validar al inicio que existan los CSV de descriptores y los checkpoints de cada objeto (fail fast con mensaje claro).
   - Añadir un CSV agregado `results/comparison_cpu/summary_all_datasets.csv` con la mediana de `t_total` y el F1 por modelo y dataset.

### Ajustes al orden de ejecución (sección 7)
- Paso 4 pasa a incluir instalar `psutil` y actualizar `requirements.txt`.
- Paso 8 ejecuta en este orden: `kolektorsdd2` → `kolektorsdd2-inverted` → `mvtec-ad bottle` → `mvtec-ad-inverted bottle`. Se corre el perfil completo (Fase B) solo en `kolektorsdd2` y `mvtec-ad bottle`. En las variantes invertidas basta con la comparación CPU (Fase A), porque el coste computacional no depende de la inversión de intensidades. Verificarlo con una corrida corta del descriptor en `kolektorsdd2-inverted`.
- Las preguntas abiertas 1 a 4 de la sección 9 quedan cerradas. Sigue vigente la limitación 5 (CPU de escritorio ≠ hardware embebido).
