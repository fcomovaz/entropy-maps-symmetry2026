# Laboratorio 12: Profiling paso a paso (`understanding-first`)

Explora de manera explícita e inspeccionable qué etapa computacional consume más tiempo en CPU (`threads=1`, `batch=1`) dentro de ambos paradigmas antes de consolidar la infraestructura formal en `src/profiling/`:

1. **Descriptor $\mathbf{x}_{94}$ etapa por etapa**:
   - Separa el coste amortizable de inicialización (`EntropyPlan` + LUTs) del cómputo por imagen.
   - Mide con `time.perf_counter_ns()` cada paso (`quantize_pixels`, `region_histograms` por escala `1/2/4/8`, `normalized_entropy` por escala, concatenación $\mathbf{x}_{85}$, reconstrucción + asimetrías $\mathbf{x}_9$, ensamblado $\mathbf{x}_{94}$).
   - Verifica equivalencia numérica exacta (`< 1e-12`) frente a `compute_entropy_features` + `compute_asymmetry_features` y cuantifica el gap de Python (`Suma de etapas - End-to-End`).
   - Contrasta tiempos manuales frente a `cProfile` (`pstats`).
2. **CNN ligera (`ds_cnn`) por capa**:
   - Compara forward limpio sin instrumentación, hooks por módulo hoja (agrupando módulos compartidos como `ReLU`) y operadores ATen mediante `torch.profiler`.
   - Explica por qué difieren los números absolutos entre cada profiler.

## Ejecución manual (PowerShell desde `code/`)

```powershell
# Imagen sintética estructurada 128x128
.\.venv\Scripts\python.exe understanding/12_profiling_stages/explain_profiling_stages.py

# Opcional: sobre una imagen real preprocesada
.\.venv\Scripts\python.exe understanding/12_profiling_stages/explain_profiling_stages.py --image data/preprocessed/KolektorSDD2/test/defect/10002.png
```

## Salida

- Reporte detallado en consola con tabla de etapas, verificación numérica y tabla ATen de `torch.profiler`.
- Figura `understanding/12_profiling_stages/outputs/profiling_stages_breakdown.png` (barras apiladas del descriptor $\mathbf{x}_{94}$ y familias de capas de `ds_cnn`).
