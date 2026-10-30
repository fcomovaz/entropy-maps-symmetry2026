# Experimentos científicos

Registro y puntos de entrada CLI para los experimentos reproducibles del proyecto.

---

## Experimento 01: Preprocesamiento y estandarización de MVTec-AD

Prepara el dataset base estandarizando imágenes y máscaras de defectos a escala de grises (`mode 'L'`) y canvas cuadrado de `128x128` píxeles mediante letterboxing centrado con valor de relleno `0` (fondo negro), preservando la relación de aspecto original sin deformaciones geométricas.

### Qué hace el proceso

1. **Filtrado determinista**: recorre únicamente las 15 categorías configuradas en `configs/base.toml` (`subdatasets`), ignorando artefactos externos (`.cache/`, `results/`, archivos de texto).
2. **Conversión a grises**: aplica ITU-R 601 luma a imágenes en color. Las máscaras de defectos no se modifican en niveles.
3. **Inversión opcional (`--inverted`)**:
   - Aplica `255 - x` a las imágenes de `train` y `test`.
   - **Orden crítico de operaciones**: `grises -> inversión -> letterboxing`. De este modo el padding permanece en `0` (negro) integrándose con fondos claros invertidos, sin crear bordes artificiales blancos de valor 255.
   - **Máscaras intactas**: las máscaras en `ground_truth` **nunca se invierten** (preservan las etiquetas binarias del defecto y del fondo).
   - **Directorio independiente**: la variante invertida se almacena de forma aislada en `data/preprocessed/mvtec-ad-inverted/`.
4. **Reescalado proporcional (128 px máx)**:
   - Imágenes: interpolación `LANCZOS` (anti-aliasing para reducción de alta resolución).
   - Máscaras (`ground_truth`): interpolación `NEAREST` para garantizar que los valores binarios se mantengan estrictamente en `{0, 255}` y alineados píxel a píxel con la imagen correspondiente.
5. **Letterboxing/Padding centrado**: calcula márgenes simétricos para centrar el contenido y rellena el lienzo restante con `pad_value = 0`.
6. **Estructura réplica**: genera la salida en el directorio destino correspondiente (`data/preprocessed/mvtec-ad/` o `data/preprocessed/mvtec-ad-inverted/`) preservando la jerarquía original (`<categoría>/{train/good, test/<defecto>, ground_truth/<defecto>}`).

### Comandos de ejecución manual (PowerShell desde carpeta `code/`)

```powershell
# 1. Ejecutar tests unitarios y de integración (pruebas de letterbox, máscaras e inversión)
.\.venv\Scripts\python.exe -m unittest tests/test_preprocessing.py

# 2. Conteo previo estándar sin escribir archivos (dry-run)
.\.venv\Scripts\python.exe -m experiments.preprocess_mvtec_ad --dry-run

# 3. Conteo previo de variante invertida (dry-run)
.\.venv\Scripts\python.exe -m experiments.preprocess_mvtec_ad --inverted --dry-run

# 4. Ejecutar preprocesamiento estándar completo (salta archivos ya existentes)
.\.venv\Scripts\python.exe -m experiments.preprocess_mvtec_ad

# 5. Ejecutar preprocesamiento con intensidad invertida (salta archivos ya existentes en mvtec-ad-inverted)
.\.venv\Scripts\python.exe -m experiments.preprocess_mvtec_ad --inverted

# 6. Forzar reprocesamiento sobrescribiendo existentes (--overwrite)
.\.venv\Scripts\python.exe -m experiments.preprocess_mvtec_ad --overwrite
.\.venv\Scripts\python.exe -m experiments.preprocess_mvtec_ad --inverted --overwrite

# 7. Verificar integridad, reducción en MB y generar figura comparativa
.\.venv\Scripts\python.exe understanding/08_verify_mvtec_preprocessing/verify.py
```

---

## Experimento 02: Exportación CSV de MVTec-AD

Serializa cada imagen preprocesada de MVTec-AD (`128x128` en escala de grises) en una fila dentro de un archivo CSV plano (`data/csv/mvtec-ad.csv` o `data/csv/mvtec-ad-inverted.csv`), conteniendo metadatos de trazabilidad, vector de intensidades de píxeles (`px_00000...px_16383`) y dimensiones espaciales.

### Esquema de columnas (16391 columnas)

1. `object_name`: nombre de la categoría (1 de las 15 en `configs/base.toml`).
2. `split`: partición (`train`, `test`, `ground_truth`).
3. `type`: clasificación binaria de la muestra (`good` si la carpeta es `good`, `defect` para cualquier carpeta de anomalía).
4. `defect_name`: nombre original de la carpeta de defecto o `good`.
5. `file_name`: nombre del archivo PNG original (ej. `000.png` o `000_mask.png`).
6. `px_00000` a `px_16383`: intensidades de píxel uint8 en orden row-major (`row * 128 + col`).
7. `width`: ancho en píxeles (`128`).
8. `height`: alto en píxeles (`128`).

### Comandos de ejecución manual (PowerShell desde carpeta `code/`)

```powershell
# 1. Ejecutar tests unitarios y de integración de exportación CSV
.\.venv\Scripts\python.exe -m unittest tests/test_csv_export.py

# 2. Conteo previo estándar sin escribir archivos (dry-run)
.\.venv\Scripts\python.exe -m experiments.export_mvtec_ad_csv --dry-run

# 3. Conteo previo de variante invertida (dry-run)
.\.venv\Scripts\python.exe -m experiments.export_mvtec_ad_csv --inverted --dry-run

# 4. Exportar dataset estándar a CSV (salta si ya existe)
.\.venv\Scripts\python.exe -m experiments.export_mvtec_ad_csv

# 5. Exportar variante invertida a CSV (salta si ya existe)
.\.venv\Scripts\python.exe -m experiments.export_mvtec_ad_csv --inverted

# 6. Forzar re-exportación sobrescribiendo archivo existente (--overwrite)
.\.venv\Scripts\python.exe -m experiments.export_mvtec_ad_csv --overwrite
.\.venv\Scripts\python.exe -m experiments.export_mvtec_ad_csv --inverted --overwrite

# 7. Verificar integridad, esquema y reconstrucción píxel a píxel
.\.venv\Scripts\python.exe understanding/09_verify_mvtec_csv/verify.py
.\.venv\Scripts\python.exe understanding/09_verify_mvtec_csv/verify.py --inverted
```

