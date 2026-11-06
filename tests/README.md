# Pruebas del proyecto (`tests/`)

Suites de pruebas automáticas, unitarias y de integración del proyecto.

---

## Comandos de ejecución (PowerShell desde carpeta `code/`)

```powershell
# 1. Verificación del entorno y aceleración CUDA real
.\.venv\Scripts\python.exe tests/test_environment.py

# 2. Pruebas de preprocesamiento (letterboxing, máscaras e inversión)
.\.venv\Scripts\python.exe -m unittest tests/test_preprocessing.py
.\.venv\Scripts\python.exe -m unittest tests/test_kolektorsdd2.py

# 3. Pruebas de exportación CSV (esquema tabular, streaming y atomicidad)
.\.venv\Scripts\python.exe -m unittest tests/test_csv_export.py
.\.venv\Scripts\python.exe -m unittest tests/test_kolektorsdd2_csv_export.py

# 4. Pruebas de mapas de entropía y características regionales (x85)
.\.venv\Scripts\python.exe -m unittest tests/test_entropy_maps.py
.\.venv\Scripts\python.exe -m unittest tests/test_entropy_csv.py

# 5. Pruebas de anexado de asimetrías de entropía (x85 -> x94)
.\.venv\Scripts\python.exe -m unittest tests/test_asymmetry_maps.py
.\.venv\Scripts\python.exe -m unittest tests/test_asymmetry_csv.py

# 6. Pruebas de clasificación supervisada, ablación y explicabilidad (x94)
.\.venv\Scripts\python.exe -m unittest tests/test_descriptor_classifier.py

# 7. Pruebas de redes convolucionales ligeras (CNNs) y comparación
.\.venv\Scripts\python.exe -m unittest tests/test_cnn_architectures.py
.\.venv\Scripts\python.exe -m unittest tests/test_cnn_config.py
.\.venv\Scripts\python.exe -m unittest tests/test_cnn_data.py
.\.venv\Scripts\python.exe -m unittest tests/test_cnn_train.py
.\.venv\Scripts\python.exe -m unittest tests/test_cnn_evaluate.py
.\.venv\Scripts\python.exe -m unittest tests/test_cnn_cli.py

# 8. Pruebas pedagógicas de laboratorios locales
.\.venv\Scripts\python.exe understanding/03_entropy_benchmark/test_entropy.py
.\.venv\Scripts\python.exe understanding/04_lightweight_cnns/tests/test_models.py
.\.venv\Scripts\python.exe understanding/05_symmetry_features/test_symmetry.py
.\.venv\Scripts\python.exe understanding/06_entropy_pyramid_descriptor/test_descriptor.py
.\.venv\Scripts\python.exe understanding/07_descriptor_behavior/test_behavior.py
.\.venv\Scripts\python.exe understanding/07_entropy_maps_vectorized/explain_entropy_maps.py
```

---

## Inventario de suites de pruebas

| Suite | Ámbito / Qué demuestra | Requisitos / Notas |
|---|---|---|
| `test_environment.py` | Imports, asignación/transferencia CUDA, matmul GPU y forward de red mínima | Entorno local y GPU NVIDIA (Pascal/GTX 1050); falla sin CUDA real |
| `test_preprocessing.py` | Letterbox centrado, conversión a grises, preservación binaria de máscaras (`{0, 255}`), inversión y orquestación CLI | Cobertura integral de `src/preprocessing/mvtec_ad.py` |
| `test_kolektorsdd2.py` | Emparejamiento `_GT`, descarte de huérfanos/copias, clasificación binaria, letterbox 128x128, grises, inversión y CLI | Cobertura integral de `src/preprocessing/kolektorsdd2.py` |
| `test_csv_export.py` | Lectura de TOML, nombres de columnas (16391 cols), aplanado row-major, streaming temporal seguro, conteos y CLI | Cobertura integral de `src/datasets/mvtec_ad_csv.py` |
| `test_kolektorsdd2_csv_export.py` | Lectura de TOML, nombres de columnas (16389 cols), aplanado row-major, streaming temporal seguro, conteos y CLI | Cobertura integral de `src/datasets/kolektorsdd2_csv.py` |
| `test_entropy_maps.py` | Validación de niveles/bins, cotas $[0, 1]$, determinismo, partición espacial y paridad exacta con oráculo `06` ($< 10^{-12}$) | Cobertura matemática de `src/features/entropy_maps.py` |
| `test_entropy_csv.py` | Carga TOML, validación de cabecera, preservación de metadatos, escritura atómica `.tmp`, sidecar y CLI | Cobertura integral de `src/features/entropy_csv.py` y CLI |
| `test_asymmetry_maps.py` | Validación de escalas/tipos, cotas $[0, 1]$, invariancia simétrica, desigualdades triangulares y paridad con oráculo `05` ($< 10^{-12}$) | Cobertura matemática de `src/features/asymmetry_maps.py` |
| `test_asymmetry_csv.py` | Carga TOML, validación de cabecera, copia byte a byte de metadatos, atomicidad `.tmp`, sidecar y CLI | Cobertura integral de `src/features/asymmetry_csv.py` y CLI |
| `test_descriptor_classifier.py` | Carga TOML, exclusión de `ground_truth`, partición `presplit` vs `stratified` sin mutar CSVs, RF y k-NN (`StandardScaler`), ablación `x85/x9/x94`, explicabilidad 2D y CLI | Cobertura integral de `src/models/` y `experiments/evaluate_descriptors.py` |
| `test_cnn_architectures.py` | Verificación exacta de parámetros en 7 CNNs, salida `[B, 2]`, resolución multiescala (32/64/128) y backward | Cobertura de `src/models/cnn/architectures/` |
| `test_cnn_config.py` | Carga TOML de KolektorSDD2 y MVTec-AD, validación de claves requeridas y resolución dinámica de rutas | Cobertura de `src/models/cnn/config.py` |
| `test_cnn_data.py` | Ingesta directa de PNGs, exclusión de `ground_truth`, paridad de split con baseline y validación de 128x128 | Cobertura de `src/models/cnn/data.py` |
| `test_cnn_train.py` | Smoke test de entrenamiento (convergencia de pérdida), determinismo de semillas y early stopping | Cobertura de `src/models/cnn/train.py` |
| `test_cnn_evaluate.py` | Cálculo de métricas compatibles con `ClassificationMetrics` (Acc, BalAcc, F1, ROC, PR) y probabilidades | Cobertura de `src/models/cnn/evaluate.py` |
| `test_cnn_cli.py` | Validación CLI dry-run para ambos datasets, manejo de errores y reporte comparativo | Cobertura de `experiments/evaluate_cnn.py` y `compare_baseline_vs_cnn.py` |
| `03/test_entropy.py` | 5 tests: formulaciones Shannon E1–E5, cotas, LUTs y rankings | Utiliza artefactos cacheados de 01 |
| `04/tests/test_models.py` | 8 tests: shapes, backward/Adam, parámetros y capas depthwise en 7 CNNs | CPU obligatorio, CUDA probado en hardware compatible |
| `05/test_symmetry.py` | 20 tests: simetrías LR/TB/R180, desigualdades y rangos de asimetría | Verificación matemática de residuos espaciales |
| `06/test_descriptor.py` | 25 tests: vector canónico x94, pirámide espacial 1/2/4/8 e inmutabilidad | Valida esquema `x94-v1` |
| `07/test_behavior.py` | 15 tests: respuesta de x94 ante ruido, desenfoque, contraste y defectos sintéticos | Caracterización de límites y sensibilidad |
