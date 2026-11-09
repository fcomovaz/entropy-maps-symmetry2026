# Evidencia y registro de revisión

Esta carpeta conserva diagnósticos generales del proyecto. Los resultados
científicos/pedagógicos actuales siguen junto a cada laboratorio: no se movieron
para evitar romper rutas, procedencia o hashes.

## Dónde está cada evidencia

| Ruta | Contenido y lectura recomendada |
|---|---|
| [environment_debug.md](environment_debug.md) | Diagnóstico histórico del 2026-09-26: reparación CUDA y verificaciones; no es un benchmark |
| `pytorch_cuda_install.log`, `cuda_verification.log`, `environment_verification.log` | Logs locales de instalación y verificación original |
| [01/outputs/run_summary.json](../understanding/01_patch_partition/outputs/run_summary.json) | Seis casos de partición; cada carpeta de caso incluye rejilla, reconstrucción, coordenadas y métricas |
| [01/assets/source_metadata.json](../understanding/01_patch_partition/assets/source_metadata.json) | Procedencia y hash del JPEG; conservar junto al asset y al PNG derivado |
| [02/results/summary_table.md](../understanding/02_histogram_benchmark/results/summary_table.md) | Resumen histórico de histogramas; CSV canónico, dos snapshots, end-to-end y metadatos en la misma carpeta |
| [03/results/summary.md](../understanding/03_entropy_benchmark/results/summary.md) | Entropía, pipeline, errores y ranking; `run1/` conserva primera corrida, `validation.json` registra comparación |
| [04/outputs/validation.md](../understanding/04_lightweight_cnns/outputs/validation.md) | Validación original de CNN y benchmark; `.pt`, `example_rgb.png` y `folder_fixture/` son demostraciones artificiales |
| [05_symmetry_features](../understanding/05_symmetry_features/README.md) | Formulación y validación analítica de 9 residuos espaciales de asimetría |
| [06_entropy_pyramid_descriptor](../understanding/06_entropy_pyramid_descriptor/README.md) | Especificación inmutable del esquema canónico `x94-v1` y tests formales |
| [07/results](../understanding/07_descriptor_behavior/README.md) | 15 experimentos controlados, 279 filas de medición y curvas de respuesta de `x94` |
| [08_verify_mvtec_preprocessing](../understanding/08_verify_mvtec_preprocessing/README.md) | Control de integridad de MVTec-AD preprocesado (128x128, grises, `{0, 255}` en máscaras) y figura comparativa |
| [09_verify_mvtec_csv](../understanding/09_verify_mvtec_csv/README.md) | Control de integridad tabular MVTec-AD (16391 cols, 6612 filas, round-trip bit a bit con diff=0) |
| [10_verify_kolektorsdd2](../understanding/10_verify_kolektorsdd2/README.md) | Verificación de KolektorSDD2 (clasificación por máscara nativa, conteo 3335 muestras) y matriz visual |
| [11_verify_kolektorsdd2_csv](../understanding/11_verify_kolektorsdd2_csv/README.md) | Control de integridad tabular KolektorSDD2 (16389 cols, 3335 filas, round-trip bit a bit con diff=0) |
| [classification/README.md](classification/README.md) | Resultados medidos de clasificación supervisada (`RF` y `k-NN`), ablación $\mathbf{x}_{85}$ vs $\mathbf{x}_9$, análisis de matrices de confusión, umbrales y comparación `normal` vs `inverted` |
| [comparison/README.md](comparison/README.md) | **Legado `mixed-device (deprecated)`:** CSVs/PNGs históricos del Experimento 08 que mezclaban latencia CNN en CUDA con ML en CPU; conservado sin borrar |
| `comparison_cpu/` | Comparación **100% CPU justa** (`experiments/benchmark_cpu_fair.py`): CSVs `<tag>_baseline_vs_cnn_cpu.csv`, referencia CUDA aislada `<tag>_cuda_reference.csv`, Pareto PNGs por hilos y tabla agregada `summary_all_datasets.csv` |
| `profiling/` | Reportes de profiling científico profundo (`experiments/profile_pipelines.py`): subcarpetas `<tag>/` con `environment.json`, desgloses por etapa (`x94`) y capa (`cnn`), ATen ops, `cProfile`, pruebas Mann-Whitney U, ablación $\Delta\text{F1}/\Delta\text{ms}$, 6 figuras y `README.md` de hallazgos |

Los checkpoints sintéticos y el fixture no son resultados de detección real.
Los CSV/JSON de tiempos describen el equipo y protocolo registrados, no hardware
embebido. Los logs y artefactos grandes se mantienen locales; actualmente no hay
un repositorio Git inicializado en la raíz. `.gitignore` expresa una política
futura, no constituye un respaldo. Conservar las imágenes cacheadas para poder
repetir sin depender de una descarga externa.

## Revisión de estructura y documentación · 2026-09-28

Se revisaron reglas, árbol de archivos, scripts, tests, dependencias directas y
evidencia de los cuatro laboratorios. Se actualizaron los índices raíz/code/
understanding, las cuatro guías de ejecución y la responsabilidad de src/tests/
experiments/results. Se documentaron entradas, salidas esperadas, rutas, opciones,
sobrescrituras y diferencia entre prueba corta y benchmark completo.

Se quitaron **18 carpetas que solo contenían un `.gitkeep` vacío**, comprobando
su contenido y su ubicación dentro del proyecto antes de eliminarlas:

- `code/notebooks/`.
- En `code/src/`: `entropy`, `metrics`, `models`, `selectors`, `utils`.
- En `code/experiments/`: `cnn`, `mlp`, `entropy_threshold`, `patch_size`, `selector_comparison`.
- En `code/understanding/`: `edges`, `entropy`, `histograms`, `models`, `patches`, `selectors`, `visualizations`.

Se conservaron src, tests, experiments, los cuatro laboratorios, sus modelos y
tests locales, el entorno, datos, checkpoints y resultados históricos. No se
cambió código Python ni se instalaron paquetes. El registro histórico de CNN
se corrigió en una frase: el test del entorno ejecuta un forward, no entrenamiento.

## Verificación realizada en esta revisión

Se utilizó `code/.venv/Scripts/python.exe` (Python 3.11.9), PyTorch 2.14.0+cu126,
torchvision 0.29.0+cu126 y GTX 1050. Comandos reproducibles en
[code/README.md](../README.md) y las guías de cada laboratorio.

| Comprobación | Resultado |
|---|---|
| `test_environment.py` | PASS: imports, CUDA real, transferencias, matmul y forward |
| `pip check` | Sin dependencias rotas |
| 01, `--patch-size 64` | Seis casos validados con reconstrucción exacta en copia temporal |
| 03, `test_entropy.py` | Cinco tests correctos |
| 04, `tests/test_models.py` | Ocho tests correctos, sin omisiones; 140 forwards y 28 pasos backward/Adam en CPU/CUDA |
| 02, `--target-pixels-per-run 10000` | Corrida corta: 35 entradas validadas, 280 filas |
| 03, análisis de CSV históricos | Correcto sobre una copia, incluidos hash y coincidencia entre corridas |
| 03, `--rounds 1 --entropy-repeats 2 --target-pixels 1000` | Corrida corta: 7,668 comparaciones; error máximo exacto 1.60e-14 |
| 03, análisis de la corrida corta | Correcto; datos/errores/puntuaciones/rankings coinciden con el run1 conservado |
| 04, inspección de modelos y DS-CNN `--leaf` | Siete redes con `[4, 2]` y traza de operaciones con logits finales `[1, 2]` |
| 04, TinyCNN sintética, CUDA, tres épocas, seed 42 | Entrenamiento y checkpoint correctos; validación 0.500 en las tres épocas, sin evidencia de generalización |
| 04, DS-CNN con fixture por carpetas, CPU, una época | Entrenamiento/carga correctos; mapeo `defective=0`, `normal=1` |
| 04, predicción de imagen | Ejecutada con pesos aleatorios y con ambos checkpoints; advertencias y preprocesado correctos |
| 04, benchmark completo | 14 filas CPU/CUDA, input 128, batch 1, diez warmups y 50 mediciones; resultado separado del histórico |

Los originales de 01–03 se preservaron usando una copia temporal con la misma
estructura de carpetas. Logs de esta revisión: `review_20260928.log`
(entorno y CNN), `review_benchmarks_20260928.log` y `review_cnn_20260928.log`.
Se conserva la validación de la corrida corta en `review_entropy_20260928.json`
y las muestras del benchmark CNN en `review_cnn_20260928.json`. Los checkpoints
de esta comprobación se generaron temporalmente; los originales se conservaron.
Los tiempos de las corridas cortas no reemplazan
los resultados canónicos ni fundamentan afirmaciones nuevas de rendimiento.

No se reinstaló el entorno desde cero, no se volvió a descargar la imagen y no
se repitieron los protocolos temporales completos de 02/03. Dataset real,
calidad predictiva, selector integrado y ejecución embebida siguen pendientes.

Se comprobaron los enlaces locales de los 12 documentos principales y la
sintaxis de sus 20 bloques PowerShell. No hubo cambios en `requirements.txt`
porque no cambió la ejecución ni se añadieron dependencias.

La eliminación final de la copia de pruebas en TEMP fue bloqueada por la
revisión automática (`blocked by policy`, sin motivo más específico). La copia
permanece fuera del proyecto y su ruta está en `review_workspace.tmp`; no afecta
a las 18 carpetas vacías que sí se eliminaron del proyecto. Los checkpoints
de esta comprobación siguen en esa copia temporal.
