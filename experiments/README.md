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
   - Guarda los resultados en directorio independiente `data/preprocessed/KolektorSDD2-inverted/`.

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
   - Archivo `.json` adyacente que registra la ruta origen, niveles, contenedores, backend utilizado (`auto` -> `c_single_pass` / `numpy_finest_pyramid`, etc.), tiempo medido de cómputo y versión de NumPy.

### Comandos de ejecución manual (PowerShell desde carpeta `code/`)

```powershell
# 0. Opcional: compilar DLL nativa C99 (Windows MSVC x64) si no existe aún
.\.venv\Scripts\python.exe -m src.features.build_native --config configs/features/native_entropy.toml

# 1. Ejecutar tests matemáticos y de integración del pipeline de entropía (NumPy + C99)
.\.venv\Scripts\python.exe -m unittest tests/test_entropy_maps.py
.\.venv\Scripts\python.exe -m unittest tests/test_native_entropy.py
.\.venv\Scripts\python.exe -m unittest tests/test_entropy_csv.py

# 2. Ejecutar script pedagógico de comprensión paso a paso con gráfico
.\.venv\Scripts\python.exe understanding/07_entropy_maps_vectorized/explain_entropy_maps.py

# 3. Dry-run de MVTec-AD y KolektorSDD2 (valida esquema, backend y dimensiones sin escribir)
.\.venv\Scripts\python.exe -m experiments.export_entropy_maps --config configs/entropy/mvtec_ad.toml --dry-run
.\.venv\Scripts\python.exe -m experiments.export_entropy_maps --config configs/entropy/kolektorsdd2.toml --dry-run

# 4. Exportar mapas de entropía (auto = c_single_pass; admite --backend numpy_finest_pyramid, numpy_fused_entropy o numpy_baseline)
.\.venv\Scripts\python.exe -m experiments.export_entropy_maps --config configs/entropy/mvtec_ad.toml
.\.venv\Scripts\python.exe -m experiments.export_entropy_maps --config configs/entropy/mvtec_ad.toml --inverted
.\.venv\Scripts\python.exe -m experiments.export_entropy_maps --config configs/entropy/kolektorsdd2.toml
.\.venv\Scripts\python.exe -m experiments.export_entropy_maps --config configs/entropy/kolektorsdd2.toml --inverted

# 5. Forzar backend 100 % NumPy puro o sobrescritura (--backend / --overwrite)
.\.venv\Scripts\python.exe -m experiments.export_entropy_maps --config configs/entropy/mvtec_ad.toml --backend numpy_finest_pyramid --overwrite
```

---

## Experimento 06: Anexado de asimetrías de entropía multiescala ($\mathbf{x}_{85} \rightarrow \mathbf{x}_{94}$)

Calcula las 9 medidas de asimetría espacial multiescala ($\mathbf{x}_9$) directamente a partir de las columnas de entropía de los archivos CSV planos existentes (sin releer imágenes ni recalcular histogramas) y las anexa al final de la tabla para producir el descriptor completo $\mathbf{x}_{94}$ bajo el contrato determinista `schema x94-v1`.

### Esquema y estructura matemática

1. **Reconstrucción espacial**:
   - Para cada escala $g \in \{2, 4, 8\}$, reconstruye la matriz $2\text{D}$ row-major $E^{(g)} \in \mathbb{R}^{g \times g}$.
   - La celda $1 \times 1$ global se preserva en los metadatos pero se excluye del cálculo de asimetría (residuos idénticos a 0).
2. **Residuos absolutos de asimetría**:
   - $A_{\mathrm{LR}}^{(g)} = \frac{1}{g^2} \sum_{r,c} |E_{r,c}^{(g)} - E_{r,g-1-c}^{(g)}|$ (reflexión izquierda-derecha)
   - $A_{\mathrm{TB}}^{(g)} = \frac{1}{g^2} \sum_{r,c} |E_{r,c}^{(g)} - E_{g-1-r,c}^{(g)}|$ (reflexión arriba-abajo)
   - $A_{\mathrm{R180}}^{(g)} = \frac{1}{g^2} \sum_{r,c} |E_{r,c}^{(g)} - E_{g-1-r,g-1-c}^{(g)}|$ (rotación central 180°)
3. **Orden canónico de columnas anexadas (9 columnas)**:
   - `asym_lr_2x2, asym_tb_2x2, asym_rot180_2x2`
   - `asym_lr_4x4, asym_tb_4x4, asym_rot180_4x4`
   - `asym_lr_8x8, asym_tb_8x8, asym_rot180_8x8`
4. **Archivos de salida**:
   - Genera `<nombre>-full.csv` y su sidecar `<nombre>-full.json` junto al archivo de entrada.
   - Las columnas de metadatos y las 85 características de entropía originales se copian byte a byte sin alteración de formato.

### Comandos de ejecución manual (PowerShell desde carpeta `code/`)

```powershell
# 1. Ejecutar pruebas unitarias y de integración de asimetría
.\.venv\Scripts\python.exe -m unittest tests/test_asymmetry_maps.py
.\.venv\Scripts\python.exe -m unittest tests/test_asymmetry_csv.py

# 2. Dry-run de validación sobre KolektorSDD2 (no escribe archivos)
.\.venv\Scripts\python.exe -m experiments.append_asymmetry_features --config configs/asymmetry/kolektorsdd2.toml --dry-run

# 3. Anexar asimetrías a KolektorSDD2 estándar (genera data/csv/kolektorsdd2-entropy-full.csv)
.\.venv\Scripts\python.exe -m experiments.append_asymmetry_features --config configs/asymmetry/kolektorsdd2.toml

# 4. Anexar asimetrías a KolektorSDD2 invertido (genera data/csv/kolektorsdd2-inverted-entropy-full.csv)
.\.venv\Scripts\python.exe -m experiments.append_asymmetry_features --config configs/asymmetry/kolektorsdd2.toml --inverted

# 5. Dry-run de validación sobre MVTec-AD (no escribe archivos)
.\.venv\Scripts\python.exe -m experiments.append_asymmetry_features --config configs/asymmetry/mvtec_ad.toml --dry-run

# 6. Anexar asimetrías a MVTec-AD estándar (genera data/csv/mvtec-ad-entropy-full.csv)
.\.venv\Scripts\python.exe -m experiments.append_asymmetry_features --config configs/asymmetry/mvtec_ad.toml

# 7. Anexar asimetrías a MVTec-AD invertido (genera data/csv/mvtec-ad-inverted-entropy-full.csv)
.\.venv\Scripts\python.exe -m experiments.append_asymmetry_features --config configs/asymmetry/mvtec_ad.toml --inverted

# 8. Ejecución masiva de todas las variantes configuradas
.\.venv\Scripts\python.exe -m experiments.append_asymmetry_features --all

# 9. Forzar sobrescritura en caso de existir previamente (--overwrite)
.\.venv\Scripts\python.exe -m experiments.append_asymmetry_features --config configs/asymmetry/mvtec_ad.toml --overwrite
```

---

## Experimento 07: Clasificación supervisada ligera, ablación de subconjuntos y explicabilidad espacial sobre descriptores $\mathbf{x}_{94}$

Entrena y evalúa clasificadores supervisados ligeros (`RandomForestClassifier`, `KNeighborsClassifier`, `DecisionTreeClassifier`, `LogisticRegression` con `StandardScaler` y el núcleo plegado `lr_dot` vía `LinearDotCoreClassifier`) sobre el descriptor tabular completo $\mathbf{x}_{94} = [\mathbf{x}_{85}, \mathbf{x}_9]$, ejecutando automáticamente un estudio de ablación por subconjuntos (`x85_only`, `x9_only`, `x94_full` y cada escala individual $E^{(1)}, E^{(2)}, E^{(4)}, E^{(8)}$) y un análisis de explicabilidad espacial para cuantificar cuánto aportan las 9 medidas de asimetría ($\mathbf{x}_9$) y cuál mapa de entropía es más discriminativo.

### Rol científico de cada dataset en la investigación (*Entropy Maps vs. Convoluciones*)

1. **KolektorSDD2 (`split_mode = "presplit"` — Benchmark principal de comparación supervisada directa):**
   - **Por qué es el benchmark primario:** Sus particiones oficiales `train` (`2,085 good`, `246 defect`) y `test` (`894 good`, `110 defect`) ya incluyen ambas clases de fábrica. Esto permite comparar directamente el descriptor $\mathbf{x}_{94}$ + clasificador ligero frente a redes convolucionales ligeras (CNNs) sobre exactamente el mismo split estándar sin ninguna modificación del protocolo original.
   - **Física visual:** Las muestras son tiras metálicas verticales centradas mediante *letterboxing* (`pad_value = 0` en los márgenes izquierdo y derecho) donde los defectos son fisuras o manchas locales. Aquí se evalúa la capacidad de las rejillas finas ($4\times4$ y $8\times8$) y de la asimetría vertical/horizontal (`asym_tb_8x8`, `asym_tb_4x4`, `asym_lr_8x8`) para detectar anomalías locales sin convoluciones.

2. **MVTec-AD (`split_mode = "stratified"` en memoria por objeto — Banco de pruebas multi-objeto y de simetría espacial):**
   - **Por qué requiere partición estratificada en memoria:** MVTec-AD fue concebido originalmente para *One-Class / Unsupervised Anomaly Detection*, por lo que su carpeta `train` oficial contiene **únicamente muestras `good` (`0 defect`)** en las 15 categorías, además de incluir una partición `ground_truth` con máscaras binarias. Si un clasificador supervisado binario (`RandomForest` o `k-NN`) se entrenara con el `train` nativo, solo vería una clase.
   - **Protocolo reproducible sin tocar los CSV:** El cargador descarta estrictamente las filas `split == "ground_truth"` (evitando fuga de máscaras), filtra el objeto elegido (`--object bottle`, `tile`, `metal_nut`, etc.) y aplica en RAM una partición estratificada `train`/`test` (`test_size = 0.20`, `seed = 42` configurables en TOML). Los archivos `.csv` en `data/csv/` permanecen **100% intactos** bit a bit. Cuando se comparan $\mathbf{x}_{94}$ y CNNs ligeras bajo este mismo `seed`, ambos modelos ven exactamente las mismas imágenes de entrenamiento y prueba.
   - **Por qué complementa a KolektorSDD2:** Permite demostrar dónde brilla cada componente del descriptor $\mathbf{x}_{94}$ a través de 15 dominios visuales divididos en **objetos estructurados/simétricos** (`bottle`, `cable`, `capsule`, `hazelnut`, `metal_nut`, `pill`, `screw`, `toothbrush`, `transistor`, `zipper`) y **texturas regulares/estocásticas** (`carpet`, `grid`, `leather`, `tile`, `wood`).

### Qué buscar e interpretar en cada archivo y gráfica generada (`results/classification/`)

Para cada corrida se crea una subcarpeta en `results/classification/<dataset>[_inverted]/<objeto_o_full>_<dt|knn|lr|lr_dot|rf>/` con **4 archivos de datos** y **4 figuras PNG**:

1. **`confusion_matrix_and_curves.png` (Rendimiento predictivo global):**
   - **Panel izquierdo (Matriz de Confusión):** Muestra verdaderos negativos (`TN`, piezas buenas bien clasificadas), falsos positivos (`FP`, falsas alarmas), falsos negativos (`FN`, defectos no detectados) y verdaderos positivos (`TP`, defectos detectados), tanto en conteo absoluto como en porcentaje por fila (`Row %`).
   - **Panel central (Curva ROC):** Área bajo la curva (`ROC-AUC`). Mide la capacidad global del descriptor $\mathbf{x}_{94}$ para separar las distribuciones de `good` y `defect` independientemente del umbral de decisión (`> 0.80` indica buena separabilidad; `> 0.90` excelente).
   - **Panel derecho (Curva Precision-Recall):** Crítica cuando hay desbalance de clases (como en KolektorSDD2 donde solo ~10% son defectos). `PR-AUC (AP)` refleja el compromiso entre no dejar escapar defectos (`Recall`) y no generar falsas alarmas (`Precision`).

2. **`descriptor_ablation_x85_x9_x94.png` y `ablation_table.csv` (¿Funcionó agregar las 9 medidas de simetría $\mathbf{x}_9$ y qué escala de entropía es mejor?):**
   - Compara en barras agrupadas (`F1-Score`, `Balanced Accuracy`, `ROC-AUC`) el desempeño sobre la misma partición usando:
     - `x85_only` (solo los 85 valores de entropía multiescala).
     - `x9_only` (solo los 9 residuos de asimetría espacial).
     - `x94_full` (descriptor completo $\mathbf{x}_{85} + \mathbf{x}_9$).
     - Escalas individuales: `E_1x1` (1 col), `E_2x2` (4 cols), `E_4x4` (16 cols) y `E_8x8` (64 cols).
   - **Qué buscar en el título y barras:**
     - `Symmetry Gain (x94 - x85): dF1`: Si `dF1 > 0` (ej. `+0.6235` en `tile`, `+0.2010` en `bottle`, `+0.1734` en `hazelnut`, `+0.0388` en `KolektorSDD2`), demuestra cuantitativamente que las 9 características de asimetría aportan información geométrica ortogonal que la entropía regional sola no captura.
     - Comparando `E_1x1` vs `E_2x2` vs `E_4x4` vs `E_8x8` se identifica qué resolución espacial aisla mejor el defecto (en la mayoría de objetos y en KolektorSDD2, `E_4x4` logra el mayor `F1` individual con solo 16 características).

3. **`explainability_importance_breakdown.png`, `feature_importance.csv` y `scale_and_family_breakdown.csv` (Explicabilidad de $\mathbf{x}_{85}$ vs $\mathbf{x}_9$):**
   - **Panel izquierdo (Top-15 características individuales):** Barras horizontales coloreadas en **azul** para entropía (`E_*`) y **naranja** para asimetría (`asym_*`). Si aparecen barras naranjas en el Top-5 (como ocurre en `KolektorSDD2`, `bottle`, `tile`, `carpet`, `capsule`, `zipper`), confirma que las variables de simetría son nodos primarios de decisión.
   - **Panel central (`Total Contribution by Scale`):** Suma del porcentaje de importancia total aportado por cada nivel (`E_1x1`, `E_2x2`, `E_4x4`, `E_8x8`, `asym_2x2`, `asym_4x4`, `asym_8x8`) y el resumen `x85_entropy` vs `x9_asymmetry`. *(Nota de lectura: como $\mathbf{x}_9$ son solo 9 de 94 columnas, una expectativa uniforme al azar sería $9/94 = 9.57\%$; valores de $15\%$ a $58\%$ indican una concentración de información muy superior al azar).*
   - **Panel derecho (`Efficiency per Feature` — Normalizado por cardinalidad):** Divide la importancia de cada grupo entre su número de características ($1, 4, 16, 64, 3, 3, 3$). Este gráfico elimina el sesgo de tamaño (64 columnas de `E_8x8` vs 3 columnas de `asym_8x8`) y revela la **densidad de información por característica**, además de detallar en el subtítulo qué eje de simetría dominó (`asym_lr`, `asym_tb` o `asym_rot180`).

4. **`spatial_importance_heatmaps.png` (Mapa de calor espacial 2D sobre la imagen):**
   - Reconstruye en matrices 2D ($2\times2$, $4\times4$ y $8\times8$) la importancia asignada a cada región espacial `E_{g}x{g}_r{r}_c{c}` sobre el lienzo de $128\times128$ píxeles.
   - **Qué buscar:** Muestra físicamente qué zonas de la pieza inspecciona el modelo. Por ejemplo, en **KolektorSDD2**, las columnas exteriores `c0` y `c3` ($4\times4$) y `c0, c1, c6, c7` ($8\times8$) aparecen con **`0.00%` de importancia** (amarillo claro) mientras toda la atención se concentra en la franja central `c1-c2` / `c2-c5`, demostrando que el clasificador ignora automáticamente el *letterbox padding* negro y focaliza el cómputo en el centro metálico.

5. **`metrics_summary.json` y reporte multi-objeto (`mvtec_all_objects_<rf|knn>_summary.csv` / `.png`):**
   - `metrics_summary.json` guarda todos los hiperparámetros, conteos, métricas y tiempos desglosados (`T_data_load_s`, `T_model_train_s`, `T_model_inference_s`, `inference_ms_per_sample`), cumpliendo la regla de separación temporal de `AGENTS.md`.
   - Al usar `--all-objects` en MVTec-AD, se genera `mvtec_all_objects_rf_comparison.png`, que contrasta en un solo lienzo las 15 categorías (`F1(x85)` vs `F1(x94)` y el porcentaje de aporte de $\mathbf{x}_9$ frente al umbral uniforme del $9.6\%$).

### Notas metodológicas para el artículo (`normal` vs `--inverted`, probabilidades y umbral realista `0.50`)

1. **Hiperparámetros configurados por dataset y efecto del umbral realista `0.50`:**
   - En los TOML actuales (`configs/classification/`), `kolektorsdd2.toml` fija `rf` en `n_estimators = 200, max_depth = 6` (`dt max_depth = 16`), mientras que `mvtec_ad.toml` fija `rf` en `n_estimators = 100, max_depth = 12` (`dt max_depth = 3`). Las corridas históricas iniciales de `mvtec_ad` usaron `200` árboles (`F1 = 0.7619` en `bottle`); la configuración vigente da `F1 = 0.7273` (`ROC-AUC = 0.9030`).
   - En un `RandomForestClassifier` de $T$ árboles, la probabilidad del modelo (`predict_proba`) es el promedio aritmético $\hat{p} = \frac{1}{T}\sum_{t=1}^{T} p_t$ y `.predict(X_test)` aplica el umbral realista **$\hat{p} \ge 0.50 \Rightarrow \text{defect (1)}$**. En objetos con muy pocos defectos y alta correlación entre celdas, `max_features="sqrt"` diluye $\hat{p}$ por debajo de `0.50` aunque el ranking `ROC-AUC` sea alto.
   - `lr_dot` pliega algebraicamente `StandardScaler` + `LogisticRegression` ($w'_i = w_i / \sigma_i$, $b' = b - \sum_i w_i \mu_i / \sigma_i$) logrando equivalencia numérica exacta frente a `lr` (`< 1e-12` en probabilidades/métricas) sin wrapper `sklearn`.

2. **Qué variante elegir para la investigación (`normal` vs `--inverted`) y cómo reportar ambas en el paper:**
   - **El letterboxing siempre permanece en `0` (negro puro):** Tanto en MVTec-AD como en KolektorSDD2, la inversión $255 - I$ se aplica a la imagen **antes** de `letterbox_to_square(..., pad_value=0)`. Así, las franjas de relleno conservan siempre intensidad `0` y entropía `0.0`.
   - **Para maximizar la detección real de defectos (`TP` y `F1` con el corte realista de `0.50`), usar `normal` como configuración primaria:**
     - En **KolektorSDD2**, `normal` detecta un **33% más de defectos verdaderos** en la matriz de confusión (`TP = 24/110`, `F1 = 0.3333`, `BalAcc = 0.6035`) frente a `inverted` (`TP = 18/110`, `F1 = 0.2647`).
     - En **MVTec-AD**, como las 15 categorías tienen resolución nativa cuadrada (sin píxeles de letterboxing), invertir $255 - I$ solo invierte el orden de los 256 bins del histograma ($p'_k = p_{255-k}$). Como la entropía de Shannon $H = -\sum p_k \log_2 p_k$ es conmutativa, `normal` e `inverted` producen exactamente la misma matriz de confusión en 12 de los 15 objetos.
   - **Valor científico de reportar ambos (`normal` e `inverted`) en el artículo:**
     - En **MVTec-AD**, reportar la tabla comparativa `normal` vs `inverted` demuestra empíricamente una ventaja teórica clave frente a las CNNs: **el descriptor $\mathbf{x}_{94}$ es intrínsecamente invariante a la inversión de polaridad de iluminación ($255 - I$)** en imágenes sin padding, mientras que los filtros convolucionales cambian de signo de activación si no se reentrenan con aumento de datos.
     - En **KolektorSDD2**, reportar ambos muestra cómo `normal` maximiza el conteo de verdaderos positivos (`TP = 24`) en operación estándar (`0.50`), mientras que `inverted` eleva ligeramente el área bajo la curva (`ROC-AUC = 0.8383` vs `0.8219`) al separar las sombras oscuras del metal respecto al `pad_value = 0` del letterbox.

### Comandos de ejecución manual (PowerShell desde carpeta `code/`)

```powershell
# 1. Ejecutar suite de pruebas unitarias y de integración (8 tests)
.\.venv\Scripts\python.exe -m unittest tests/test_descriptor_classifier.py -v

# 2. Dry-run de validación (verifica TOML, exclusión de ground_truth y conteo de clases sin entrenar)
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/kolektorsdd2.toml --dry-run
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --object bottle --dry-run

# 3. KolektorSDD2 estándar e invertido con clasificadores ligeros (RF, k-NN, Decision Tree, Logistic Regression)
# Evaluar todos los modelos ML disponibles en una sola pasada:
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/kolektorsdd2.toml --classifier all
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/kolektorsdd2.toml --classifier all --inverted

# O evaluar clasificadores individuales (`dt`, `knn`, `lr`, `lr_dot`, `rf`):
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/kolektorsdd2.toml --classifier rf
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/kolektorsdd2.toml --classifier knn
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/kolektorsdd2.toml --classifier dt
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/kolektorsdd2.toml --classifier lr
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/kolektorsdd2.toml --classifier lr_dot

# 4. MVTec-AD por objeto individual (ej. bottle, tile, metal_nut, zipper; normal o --inverted)
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --object bottle --classifier rf
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --object bottle --classifier dt
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --object tile --classifier rf
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --object bottle --classifier knn --inverted

# 5. MVTec-AD barrido completo de los 15 objetos + tabla y gráfica comparativa global
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --all-objects --classifier rf
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --all-objects --classifier rf --inverted
```

---

## Experimento 08: Entrenamiento, evaluación y benchmarking de CNNs ligeras y comparación contra el baseline

Entrena y evalúa las 7 arquitecturas convolucionales ligeras (`ultra_tiny`, `ds_cnn`, `tiny_cnn`, `mobilenet_like`, `tiny_resnet`, `small_cnn`, `torchvision_mobilenet`) sobre imágenes en escala de grises de $128\times128$ (`mode 'L'`) procedentes directamente de `data/preprocessed/`, respetando estrictamente el protocolo y particiones idénticas a la clasificación de descriptores (task_06A). Además, el módulo de comparación cruzada mide la latencia de extracción del descriptor ($T_{\text{descriptor}}$) para contrastar el tiempo total de extremo a extremo ($T_{\text{total}} = T_{\text{descriptor}} + T_{\text{model}}$) con la inferencia de la red convolucional sobre píxeles directos.

> [!WARNING]
> **Nota metodológica sobre `results/comparison/` (`mixed-device (deprecated)`):** La comparación cruzada histórica generada por `experiments.compare_baseline_vs_cnn` mezclaba inferencia CNN en GPU CUDA (`device = "auto"`) con descriptores y modelos clásicos ejecutados en CPU (además de RF con `n_jobs = -1` y `params = 94` como número de características). **No usar sus latencias para comparaciones de velocidad en CPU.** Para una comparación justa **100 % CPU** y profiling granular, utilizar el **Experimento 09** (`benchmark_cpu_fair.py` y `profile_pipelines.py`). Los archivos en `results/comparison/` se conservan intactos como registro histórico.

### Modelos evaluados

| Modelo | Parámetros | Tamaño MB | Características |
|---|---|---|---|
| `ultra_tiny` | 1,154 | 0.0046 | Bloques separables con reducción temprana (TinyML) |
| `ds_cnn` | 12,866 | 0.0515 | Depthwise-Separable Convolutions (MobileNetV1 style) |
| `tiny_cnn` | 23,538 | 0.0942 | 3 etapas Conv-BN-ReLU-MaxPool con global average pooling |
| `mobilenet_like` | 70,786 | 0.2831 | Bloques separables escalonados sin expansión invertida |
| `tiny_resnet` | 77,234 | 0.3089 | 3 bloques residuales con shortcuts identidad y proyección |
| `small_cnn` | 97,650 | 0.3906 | 4 etapas convolucionales con mayor capacidad |
| `torchvision_mobilenet` | 1,519,618 | 6.0785 | MobileNetV3-Small oficial (pesos aleatorios, sin descargas) |

### Calibración del umbral de decisión (*Threshold Calibration*) y desbalance de clases

Al entrenar redes neuronales con pérdida ponderada (`class_weighting = "balanced"`), el cociente de pesos $w_1 / w_0$ introduce un sesgo aditivo en los logits:
$$z_1 - z_0 \approx \log \left(\frac{P(Y=1 \mid X)}{P(Y=0 \mid X)}\right) + \log \left(\frac{w_1}{w_0}\right)$$

En datasets con fuerte desbalance como KolektorSDD2 ($\sim 11\,\%$ defectos, $w_1 / w_0 \approx 8.47$), aplicar a ciegas el umbral $\tau = 0.50$ sobre la probabilidad Softmax equivale en el espacio posterior a clasificar como defecto cualquier muestra con $P(Y=1 \mid X) \ge 10.5\,\%$. Esto puede inflar drásticamente los falsos positivos en semillas con distribuciones suaves, desplomando la Accuracy y el F1 a pesar de que el ranking del modelo sea óptimo ($\text{ROC-AUC} \ge 0.90$, $\text{PR-AUC} \ge 0.75$).

Para garantizar rigor científico y evitar fuga de datos (*zero data leakage*):
1. **`--threshold-mode val_f1` (Por defecto)**: Calibra el umbral óptimo $\tau^*$ evaluando exclusivamente la partición de validación (`X_val, y_val`), maximizando F1 sin ver el conjunto de prueba.
2. **`--threshold-mode fixed_05`**: Aplica el corte rígido clásico $\tau = 0.50$ para comparabilidad con modelos previos.
3. **`--threshold-mode val_bal_acc`**: Calibra el umbral optimizando la exactitud balanceada (estadístico J de Youden) en validación.
4. **`--threshold <float>`**: Permite fijar un umbral explícito manual (ej. `--threshold 0.72`).

El reporte generado en `metrics_summary.json` registra tanto el desempeño en el umbral calibrado como en $\tau = 0.50$ y el óptimo oráculo de prueba.

### Comandos de ejecución manual (PowerShell desde carpeta `code/`)

```powershell
# 1. Pruebas unitarias y de integración de CNNs
.\.venv\Scripts\python.exe -m unittest tests/test_cnn_architectures.py -v
.\.venv\Scripts\python.exe -m unittest tests/test_cnn_config.py -v
.\.venv\Scripts\python.exe -m unittest tests/test_cnn_data.py -v
.\.venv\Scripts\python.exe -m unittest tests/test_cnn_train.py -v
.\.venv\Scripts\python.exe -m unittest tests/test_cnn_evaluate.py -v
.\.venv\Scripts\python.exe -m unittest tests/test_cnn_cli.py -v

# O ejecutar toda la suite de tests del proyecto
.\.venv\Scripts\python.exe -m unittest discover tests -v

# 2. Dry-run de validación (verifica TOML, split counts y forward pass sin entrenar)
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/kolektorsdd2.toml --dry-run
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/mvtec_ad.toml --object bottle --dry-run

# 3. KolektorSDD2 con umbral calibrado en validación (por defecto, F1 recuperado)
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/kolektorsdd2.toml --model ds_cnn --overwrite
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/kolektorsdd2.toml --model ds_cnn --inverted --overwrite

# O evaluar todas las CNNs del zoo (7 arquitecturas) en una sola corrida:
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/kolektorsdd2.toml --model all --overwrite
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/kolektorsdd2.toml --model all --inverted --overwrite

# Opcional: Ejecutar con umbral fijo 0.50 para observar el efecto del sesgo de pesos
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/kolektorsdd2.toml --model ds_cnn --threshold-mode fixed_05 --overwrite

# Otras arquitecturas representativas en KolektorSDD2
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/kolektorsdd2.toml --model tiny_cnn --overwrite
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/kolektorsdd2.toml --model tiny_resnet --overwrite

# 4. MVTec-AD por objeto individual (ej. bottle, tile, metal_nut, screw)
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/mvtec_ad.toml --object bottle --model ds_cnn --overwrite
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/mvtec_ad.toml --object tile --model ds_cnn --overwrite

# 5. MVTec-AD barrido multi-objeto
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/mvtec_ad.toml --all-objects --model ds_cnn --overwrite

# 6. Comparación cruzada histórica (OBSOLETA para latencia en CPU: mixed-device GPU CNN vs CPU ML)
.\.venv\Scripts\python.exe -m experiments.compare_baseline_vs_cnn --dataset kolektorsdd2
.\.venv\Scripts\python.exe -m experiments.compare_baseline_vs_cnn --dataset kolektorsdd2 --inverted
.\.venv\Scripts\python.exe -m experiments.compare_baseline_vs_cnn --dataset mvtecad --object bottle
```

---

## Experimento 09: Benchmarking 100 % CPU Justo (`benchmark_cpu_fair.py`) y Profiling Científico Detallado (`profile_pipelines.py`)

Corrige el sesgo *mixed-device* de `results/comparison/` midiendo el pipeline tabular **$\mathbf{x}_{94}$ + clasificadores clásicos** (`dt`, `knn`, `lr`, `lr_dot`, `rf`) y las **7 arquitecturas CNN ligeras** bajo un protocolo homogéneo **100 % CPU** (`task_08` + `task_10`), además de descomponer dónde se invierte el tiempo y la memoria en cada etapa y capa y permitir conmutar el *backend* de entropía (`auto` -> `c_single_pass` C99, `numpy_finest_pyramid` 100 % Python/NumPy puro, `numpy_fused_entropy` o `numpy_baseline`).

### Protocolo científico y decisiones metodológicas

1. **Paridad estricta de hardware (`--device cpu`):**
   - `benchmark_cpu_fair.py` impone `device = "cpu"` como comparación primaria (pasar `--device cuda` falla inmediatamente). Si hay GPU disponible, cualquier referencia CUDA opcional se guarda **aislada** en `<tag>_cuda_reference.csv` sin mezclarse en la tabla ni en las gráficas principales.
2. **Hilos y tamaño de lote (`threads=1`, `batch=1` como referencia primaria):**
   - El perfil principal fija **`threads = 1`** (comparable a ejecución embebida/mono-hilo) imponiendo simétricamente `torch.set_num_threads(threads)`, `threadpoolctl` (OMP/MKL/BLAS) y `n_jobs = threads` en Scikit-learn (`rf`, `knn`). El escalado multihilo (`threads ∈ {2, 4, physical_cores}`) y `batch = 32` (*throughput*) se reportan como análisis secundario separado.
   - `batch = 1` sobre **200 imágenes reales del split `test`** (semilla fija `42`) mide latencia real extremo a extremo; nunca se usa `np.random` ni tensores aleatorios para el reporte principal.
3. **Separación de tiempos (`T_total = T_prep_or_desc + T_model`) y cronómetro robusto:**
   - **ML clásico ($\mathbf{x}_{94}$):** `T_total = T_descriptor + T_model` (`t_preprocess = 0`, el preprocesamiento vive en `T_descriptor`).
   - **CNNs:** `T_total = T_preprocess + T_model`, incluyendo explícitamente la normalización `uint8 -> [B, 1, 128, 128]` en `T_preprocess`.
   - Cronómetro `time.perf_counter_ns()` con recolector de basura deshabilitado (`gc.disable()`), `warmup_iters = 20`, `timed_iters = 200` (`50` para `batch = 32`), `repeats = 3` (variabilidad inter-corrida `CV%`) y resumen por **mediana, IQR, p95, media, std e IC 95 % por *bootstrap* (1000 remuestreos)**.
4. **Reutilización de checkpoints GPU y verificación de paridad CPU:**
   - Por decisión metodológica (`task_08` §10), los modelos CNN **no se reentrenan en CPU**: se cargan los `checkpoint.pt` ya entrenados en GPU (`results/cnn/.../checkpoint.pt`), se re-evalúan en CPU para verificar equivalencia numérica de métricas (`metrics_tolerance = 1e-5` respecto a `metrics_summary.json`) y se cronometran exclusivamente en CPU.
   - Si faltan checkpoints en algún objeto objetivo (p. ej., `mvtec-ad/bottle` o `mvtec-ad-inverted/bottle`), el CLI falla rápido indicando el comando exacto de `experiments.evaluate_cnn` a ejecutar antes.
5. **Parámetros reales y operaciones analíticas separadas de PyTorch `FlopCounterMode`:**
   - `params_model` registra parámetros reales del clasificador (`dt`: número de nodos, `rf`: suma de nodos de los árboles, `lr`: coeficientes + intercepto, `knn`: `N_train * n_features`), separados de `params_descriptor` (94 características).
   - `ops_basis` distingue explícitamente `analytical_estimate` (operaciones analíticas del descriptor + profunidades/productos escalares de ML) frente a `torch_flop_counter_mode` en CNNs.
6. **Variantes invertidas (`--inverted`):**
   - `benchmark_cpu_fair.py` ejecuta la comparación 100 % CPU completa en variantes normales e invertidas (`kolektorsdd2`, `kolektorsdd2-inverted`, `mvtec-ad_bottle`, `mvtec-ad-bottle-inverted`), consolidando todas las corridas en `results/comparison_cpu/summary_all_datasets.csv`.
   - En `profile_pipelines.py`, por configuración (`run_full_profiling_on_inverted = false`), las variantes `--inverted` ejecutan automáticamente la verificación corta del descriptor (`--short-descriptor-check`) sin repetir el profiling redundante de todas las capas CNN (salvo que se pase `--full-inverted`).

### Salidas generadas

- **Fase A (`results/comparison_cpu/`):**
  - `<tag>_baseline_vs_cnn_cpu.csv` y (opcional) `<tag>_cuda_reference.csv`.
  - `summary_all_datasets.csv` (tabla agregada de mediana `t_total` y `F1` entre datasets/objetos).
  - `pareto_cpu_t1.png` (y por cada conteo de hilos): frontera de Pareto (`F1`, `ROC-AUC`, `Throughput`).
- **Fase B (`results/profiling/<tag>/`):**
  - `environment.json` (CPU, núcleos físicos/lógicos, frecuencia, plan de energía Windows, RAM, Git commit, versiones, `parallel_info`).
  - `descriptor_stage_times.csv`, `cprofile_descriptor.txt`, `descriptor_scalability.csv` (barrido `image_size`, `bins`, `grid_levels`).
  - `classifier_times.csv`, `classifier_thread_scaling.csv`.
  - `cnn_layer_times_<model>.csv`, `cnn_aten_ops_<model>.csv`, `chrome_trace_<model>.json`, `cnn_architectures_summary.csv`, `cnn_thread_scaling.csv` (incluyendo *cold start* de checkpoint + primera inferencia y verificación de overhead de *hooks* `<= 15 %`).
  - `ablation_quality_vs_cost.csv` ($\Delta\text{F1} / \Delta\text{ms}$ entre `x85_only`, `x9_only` y `x94_full`) y `statistical_tests.csv` (test U de Mann-Whitney).
  - 6 figuras PNG (`01_x94_stage_breakdown.png` ... `06_throughput_batch32.png`) y reporte `README.md` poblado automáticamente con los números medidos.

### Comandos de ejecución manual (PowerShell desde carpeta `code/`)

```powershell
# 1. Instalar/sincronizar psutil==7.2.2 si falta y verificar tests de profiling + C99
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest tests/test_profiling_config.py tests/test_profiling_timing.py tests/test_profiling_descriptor_stages.py tests/test_profiling_classifier.py tests/test_profiling_cnn_layers.py tests/test_benchmark_cpu_fair_cli.py tests/test_native_entropy.py -v
.\.venv\Scripts\python.exe -m unittest discover tests -v

# 2. Laboratorios previos (understanding-first) en CPU
.\.venv\Scripts\python.exe understanding/12_profiling_stages/explain_profiling_stages.py
.\.venv\Scripts\python.exe understanding/12_profiling_stages/explain_profiling_stages.py --image data/preprocessed/KolektorSDD2/test/defect/10002.png
.\.venv\Scripts\python.exe understanding/13_entropy_maps_optimization/explain_single_pass.py

# 3. Recompilar DLL C99 si no existe o tras editar src/features/native/
.\.venv\Scripts\python.exe -m src.features.build_native --config configs/features/native_entropy.toml

# 4. Validación previa (--dry-run) de Fase A y Fase B sin ejecutar benchmarks
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/kolektorsdd2.toml --dry-run
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/mvtec_ad.toml --objects bottle --dry-run
.\.venv\Scripts\python.exe -m experiments.profile_pipelines --config configs/profiling/kolektorsdd2.toml --dry-run
.\.venv\Scripts\python.exe -m experiments.profile_pipelines --config configs/profiling/mvtec_ad.toml --objects bottle --dry-run

# 5. Fase A (por defecto backend='auto' -> 'c_single_pass' C99): Comparación 100 % CPU justa
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/kolektorsdd2.toml
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/kolektorsdd2.toml --inverted
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/mvtec_ad.toml --objects bottle
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/mvtec_ad.toml --objects bottle --inverted

# 5B. Comparativa 100 % Python/NumPy puro pre-MCU (sin DLL C99):
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/kolektorsdd2.toml --entropy-backend numpy_finest_pyramid --output-dir results/comparison_cpu_python --no-cuda-reference
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/mvtec_ad.toml --objects bottle --entropy-backend numpy_finest_pyramid --output-dir results/comparison_cpu_python --no-cuda-reference

# 6. Fase B: Profiling científico profundo (normales completo; invertidos verificación corta)
.\.venv\Scripts\python.exe -m experiments.profile_pipelines --config configs/profiling/kolektorsdd2.toml
.\.venv\Scripts\python.exe -m experiments.profile_pipelines --config configs/profiling/mvtec_ad.toml --objects bottle
.\.venv\Scripts\python.exe -m experiments.profile_pipelines --config configs/profiling/kolektorsdd2.toml --inverted
.\.venv\Scripts\python.exe -m experiments.profile_pipelines --config configs/profiling/mvtec_ad.toml --objects bottle --inverted
```


