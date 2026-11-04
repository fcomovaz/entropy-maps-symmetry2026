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

---

## Experimento 03: Preprocesamiento y clasificación de KolektorSDD2

Prepara el dataset KolektorSDD2 inspeccionando cada máscara binaria `_GT` a resolución nativa para clasificar la muestra en `good` o `defect`, estandarizando las imágenes a escala de grises (`mode 'L'`) y canvas cuadrado de `128x128` mediante letterboxing centrado con valor de relleno `0`. Las máscaras se descartan de la jerarquía procesada final.

### Qué hace el proceso

1. **Emparejamiento determinista (`N.png` con `N_GT.png`)**:
   - Valida stems contra el patrón configurado `^\d+$`.
   - Ignora y registra advertencias para archivos de copia u huérfanos (ej. `10301 (copy).png` y `10301_GT (copy).png`).
2. **Clasificación binaria por máscara nativa**:
   - Muestra evaluada como `defect` si `mask.max() > defect_threshold` (por defecto `0`). De lo contrario se asigna a `good`.
   - **Las máscaras no se guardan en la salida**: sólo se leen para clasificar la muestra.
3. **Conversión a grises**: convierte imágenes RGB a `mode 'L'` mediante ITU-R 601 luma.
4. **Letterboxing centrado a 128x128**:
   - Escala imágenes de aspecto ~1:2.8 centrando el contenido y rellenando los laterales con `0` (interpolación `LANCZOS`).
5. **Inversión opcional (`--inverted`)**:
   - Invierte intensidad de imagen (`255 - x`) antes del letterbox para que el padding permanezca en `0`.
   - Guarda los resultados en directorio independiente `data/processed/KolektorSDD2-inverted/`.

### Comandos de ejecución manual (PowerShell desde carpeta `code/`)

```powershell
Set-Location D:\Publications\2026-10-Symmetry-Entropy\code

# 1. Ejecutar tests unitarios (debe pasar 9/9)
.\.venv\Scripts\python.exe -m unittest tests/test_kolektorsdd2.py

# 2. Conteo previo sin escribir archivos (dry-run estándar)
# Esperado: 3335 pares (train: 2085 good, 246 defect; test: 894 good, 110 defect), 2 huérfanos omitidos
.\.venv\Scripts\python.exe -m experiments.preprocess_kolektorsdd2 --dry-run

# 3. Conteo previo sin escribir archivos (dry-run invertido)
.\.venv\Scripts\python.exe -m experiments.preprocess_kolektorsdd2 --inverted --dry-run

# 4. Ejecutar preprocesamiento estándar completo
.\.venv\Scripts\python.exe -m experiments.preprocess_kolektorsdd2

# 5. Ejecutar preprocesamiento con intensidad invertida (255 - x)
.\.venv\Scripts\python.exe -m experiments.preprocess_kolektorsdd2 --inverted

# 6. Opcional: forzar reprocesamiento sobrescribiendo existentes
.\.venv\Scripts\python.exe -m experiments.preprocess_kolektorsdd2 --overwrite
.\.venv\Scripts\python.exe -m experiments.preprocess_kolektorsdd2 --inverted --overwrite

# 7. Verificar integridad, conteos exactos y generar figura visual comparativa
.\.venv\Scripts\python.exe understanding/10_verify_kolektorsdd2/verify.py
.\.venv\Scripts\python.exe understanding/10_verify_kolektorsdd2/verify.py --inverted
```

---

## Experimento 04: Exportación CSV de KolektorSDD2

Serializa cada imagen preprocesada de KolektorSDD2 (`128x128` en escala de grises) en una fila dentro de un archivo CSV plano (`data/csv/kolektorsdd2.csv` o `data/csv/kolektorsdd2-inverted.csv`), conteniendo metadatos de partición y tipo (`split, type, file_name`), vector de intensidades de píxeles (`px_00000...px_16383`) y dimensiones espaciales (`width, height`).

### Esquema de columnas (16389 columnas)

1. `split`: partición (`train` o `test`).
2. `type`: clasificación binaria de la muestra (`good` o `defect`).
3. `file_name`: nombre del archivo PNG original (ej. `10021.png`).
4. `px_00000` a `px_16383`: intensidades de píxel uint8 en orden row-major (`row * 128 + col`).
5. `width`: ancho en píxeles (`128`).
6. `height`: alto en píxeles (`128`).

Total: 3 metadatos + 16384 intensidades de píxeles + 2 dimensiones = 16389 columnas.

### Comandos de ejecución manual (PowerShell desde carpeta `code/`)

```powershell
# 1. Ejecutar tests unitarios y de integración de exportación CSV KolektorSDD2
.\.venv\Scripts\python.exe -m unittest tests/test_kolektorsdd2_csv_export.py

# 2. Conteo previo estándar sin escribir archivos (dry-run)
.\.venv\Scripts\python.exe -m experiments.export_kolektorsdd2_csv --dry-run

# 3. Conteo previo de variante invertida (dry-run)
.\.venv\Scripts\python.exe -m experiments.export_kolektorsdd2_csv --inverted --dry-run

# 4. Exportar dataset estándar a CSV (salta si ya existe)
.\.venv\Scripts\python.exe -m experiments.export_kolektorsdd2_csv

# 5. Exportar variante invertida a CSV (salta si ya existe)
.\.venv\Scripts\python.exe -m experiments.export_kolektorsdd2_csv --inverted

# 6. Forzar re-exportación sobrescribiendo archivo existente (--overwrite)
.\.venv\Scripts\python.exe -m experiments.export_kolektorsdd2_csv --overwrite
.\.venv\Scripts\python.exe -m experiments.export_kolektorsdd2_csv --inverted --overwrite

# 7. Verificar integridad, esquema y reconstrucción píxel a píxel
.\.venv\Scripts\python.exe understanding/11_verify_kolektorsdd2_csv/verify.py
.\.venv\Scripts\python.exe understanding/11_verify_kolektorsdd2_csv/verify.py --inverted
```

---

## Experimento 05: Extracción de mapas de entropía visual por regiones ($\mathbf{x}_{85}$)

Extrae descriptores de entropía visual espacial multiescala a partir de los archivos CSV con píxeles aplanados (`mvtec-ad.csv`, `kolektorsdd2.csv` o sus variantes invertidas), dividiendo cada imagen en particiones regulares de $g \times g$ regiones para cada escala configurada (por defecto $g \in \{1, 2, 4, 8\}$, sumando $1 + 4 + 16 + 64 = 85$ características normalizadas de Shannon).

### Esquema de salida

Las columnas de píxeles (`px_*`) y dimensiones (`width`, `height`) son reemplazadas por:
1. **Metadatos originales preservados intactos**:
   - MVTec-AD: `object_name, split, type, defect_name, file_name` (5 columnas).
   - KolektorSDD2: `split, type, file_name` (3 columnas).
2. **Columnas de trazabilidad de configuración**:
   - `bins`: número entero de contenedores de histograma (ej. `16`).
   - `grid_levels`: niveles de partición concatenados con separador configurable (ej. `1|2|4|8`).
3. **Características de entropía regional normalizada**:
   - Columnas `E_{g}x{g}_r{r}_c{c}` (ej. `E_1x1_r0_c0` ... `E_8x8_r7_c7`) con valores en $[0.0, 1.0]$ formateados a la precisión decimal configurada.
4. **Sidecar de reproducibilidad**:
   - Archivo `.json` adyacente que registra la ruta origen, niveles, contenedores, tiempo medido de cómputo y versión de NumPy.

### Comandos de ejecución manual (PowerShell desde carpeta `code/`)

```powershell
# 1. Ejecutar tests matemáticos y de integración del pipeline de entropía
.\.venv\Scripts\python.exe -m unittest tests/test_entropy_maps.py
.\.venv\Scripts\python.exe -m unittest tests/test_entropy_csv.py

# 2. Ejecutar script pedagógico de comprensión paso a paso con gráfico
.\.venv\Scripts\python.exe understanding/07_entropy_maps_vectorized/explain_entropy_maps.py

# 3. Dry-run de MVTec-AD (valida esquema y dimensiones sin escribir)
.\.venv\Scripts\python.exe -m experiments.export_entropy_maps --config configs/entropy/mvtec_ad.toml --dry-run

# 4. Dry-run de KolektorSDD2 (valida esquema y dimensiones sin escribir)
.\.venv\Scripts\python.exe -m experiments.export_entropy_maps --config configs/entropy/kolektorsdd2.toml --dry-run

# 5. Exportar mapas de entropía MVTec-AD (genera data/csv/mvtec-ad-entropy.csv)
.\.venv\Scripts\python.exe -m experiments.export_entropy_maps --config configs/entropy/mvtec_ad.toml

# 6. Exportar mapas de entropía MVTec-AD invertido (genera data/csv/mvtec-ad-inverted-entropy.csv)
.\.venv\Scripts\python.exe -m experiments.export_entropy_maps --config configs/entropy/mvtec_ad.toml --inverted

# 7. Exportar mapas de entropía KolektorSDD2 (genera data/csv/kolektorsdd2-entropy.csv)
.\.venv\Scripts\python.exe -m experiments.export_entropy_maps --config configs/entropy/kolektorsdd2.toml

# 8. Exportar mapas de entropía KolektorSDD2 invertido (genera data/csv/kolektorsdd2-inverted-entropy.csv)
.\.venv\Scripts\python.exe -m experiments.export_entropy_maps --config configs/entropy/kolektorsdd2.toml --inverted

# 9. Forzar sobrescritura en caso de existir previamente (--overwrite)
.\.venv\Scripts\python.exe -m experiments.export_entropy_maps --config configs/entropy/mvtec_ad.toml --overwrite
```


