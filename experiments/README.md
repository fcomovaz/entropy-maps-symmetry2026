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
