# Resultados Comparativos Históricos — **mixed-device (deprecated)**

> [!WARNING]
> **Estado: `mixed-device (deprecated)` — No utilizar para afirmaciones de velocidad en CPU.**
> Los archivos CSV y figuras PNG de esta carpeta se conservan intactos únicamente como registro histórico del Experimento 08 (`experiments/compare_baseline_vs_cnn.py`).

## Por qué esta carpeta está obsoleta para comparación de latencia

En la versión original de `experiments/compare_baseline_vs_cnn.py`, las métricas de latencia se recolectaron en condiciones heterogéneas (*mixed-device*):

1. **Dispositivos mezclados:** Las CNN se evaluaron usando `[cnn.training] device = "auto"` (**CUDA GPU**, GTX 1050) con lotes grandes de test, mientras que el descriptor $\mathbf{x}_{94}$ y los clasificadores clásicos (`dt`, `knn`, `lr`, `rf`) corrieron en **CPU**. Por ejemplo, en `kolektorsdd2_baseline_vs_cnn.csv`, `ds_cnn` reporta `0.2109 ms` (GPU por lote en test) frente a `0.4815 ms` de `lr (x94)` (CPU).
2. **Hilos y protocolo desalineados:** `RandomForestClassifier` corría con `n_jobs = -1` (todos los hilos lógicos) mientras otras etapas usaban configuración por defecto; `measure_descriptor_latency_ms` medía un único lote sintético con `np.random` sin warmup robusto ni control de GC/intervalos de confianza.
3. **Conteo de parámetros/FLOPs no homogéneo:** Los clasificadores clásicos registraban `params = 94` (cardinalidad del vector $\mathbf{x}_{94}$, no parámetros del modelo) y `flops = 0`.

## Reemplazo oficial (100% CPU Justo + Profiling Científico)

Utilizar los directorios generados por la **Fase A y Fase B (`task_08`)**:

- **`../comparison_cpu/`** (`experiments/benchmark_cpu_fair.py`): Comparación estricta **100% CPU**, mismos hilos (`threads=1` principal comparable a embebido + escalado multihilo), mismos tamaños de lote (`batch=1` latencia + `batch=32` throughput), muestras reales del split de prueba, inclusión de preprocesado CNN vs $\mathbf{x}_{94}$, `perf_counter_ns()` con GC deshabilitado, mediana/IQR/IC95% bootstrap, `psutil` RSS, parámetros reales de ML y verificación de paridad de métricas CPU tras cargar checkpoints GPU. Cualquier referencia CUDA opcional queda aislada en `*_cuda_reference.csv`.
- **`../profiling/<dataset_tag>/`** (`experiments/profile_pipelines.py`): Desglose por etapas de $\mathbf{x}_{94}$, análisis `cProfile`, perfiles por capa CNN (`perf_counter_ns` hooks + `torch.profiler` ATen), ablación calidad-vs-coste ($\Delta\text{F1}/\Delta\text{ms}$), pruebas estadísticas Mann-Whitney U, 6 figuras científicas y reporte consolidado.

## Inventario histórico conservado

- `kolektorsdd2_baseline_vs_cnn.csv` y `kolektorsdd2_baseline_vs_cnn.png`
- `kolektorsdd2-inverted_baseline_vs_cnn.csv` y `kolektorsdd2-inverted_baseline_vs_cnn.png`
- `mvtec-ad_bottle_baseline_vs_cnn.csv` y `mvtec-ad_bottle_baseline_vs_cnn.png`
