# Resultados de Comparación 100 % CPU Justa (`results/comparison_cpu/`)

Este directorio almacena la comparación oficial **100 % CPU** entre los clasificadores clásicos sobre $\mathbf{x}_{94}$ (`dt`, `knn`, `lr`, `lr_dot`, `rf`, con extractor C99 `c_single_pass`) y las **7 arquitecturas CNN ligeras**, generada por `experiments.benchmark_cpu_fair`.

Para el análisis científico completo y los veredictos de hipótesis ver [../../HYPOTHESIS.md](../../HYPOTHESIS.md); para los desgloses por etapa, capa PyTorch y pruebas U de Mann-Whitney ver [../profiling/](../README.md).

---

## Protocolo aplicado

- **Igualdad estricta en CPU:** Toda comparación principal corre con `device = "cpu"` sobre **200 muestras reales del split de prueba** (`seed = 42`, sin tensores aleatorios, `gc.disable()`, `time.perf_counter_ns()`, `20` warmups, `200` iteraciones en `B=1`, `50` en `B=32`, `3` repeticiones e IC 95 % *bootstrap* con 1000 remuestreos).
- **Control simétrico de hilos:** Comparación primaria con `threads = 1` (fijado en PyTorch, `threadpoolctl` BLAS/OpenMP y `n_jobs`), más barridos con `threads = 2` y `threads = 4`.
- **Reutilización sin reentrenamiento:** Las CNNs cargan los checkpoints en `results/cnn/` y verifican paridad numérica en CPU (`metric_parity_max_diff <= 1e-5`) antes de medir latencia.
- **Separación de tiempos:** `t_total = t_descriptor + t_model` para ML ($\mathbf{x}_{94}$) y `t_total = t_preprocess + t_model` para CNNs.

---

## Inventario de artefactos en disco (generados por CLI, ignorados en Git salvo este README)

| Archivo | Contenido |
|---|---|
| `summary_all_datasets.csv` | Tabla consolidada multi-dataset (`kolektorsdd2`, `kolektorsdd2-inverted`, `mvtec-ad_bottle`, `mvtec-ad-inverted_bottle`) a través de `threads ∈ {1, 2, 4}` y `batch_size ∈ {1, 32}` |
| `<tag>_baseline_vs_cnn_cpu.csv` | Métricas completas, conteo real de parámetros (`params_model`, `params_descriptor`), `macs`/`flops`/`ops_basis`, desglose de latencia (`t_preprocess`, `t_descriptor`, `t_model`, `t_total`, IQR, p95, IC95%, `CV%`), RAM pico (`peak_ram_mb`) y paridad CPU |
| `<tag>_cuda_reference.csv` | Referencia secundaria opcional en GPU CUDA, **aislada** y nunca mezclada en las tablas o gráficas CPU principales |
| `<tag>_pareto_threads{1,2,4}.png` | Fronteras de Pareto en CPU (`F1`, `ROC-AUC`, `Throughput`) para `threads = 1, 2, 4` |

*(Nota: la comparativa local en Python/NumPy puro `numpy_finest_pyramid` y `numpy_baseline` sin la DLL C99 se genera por separado en `results/comparison_cpu_python/`.)*

## Reproducción

```powershell
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/kolektorsdd2.toml
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/kolektorsdd2.toml --inverted
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/mvtec_ad.toml --objects bottle
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/mvtec_ad.toml --objects bottle --inverted
```
