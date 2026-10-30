# Verificación y validación de integridad de KolektorSDD2 CSV

## 1. Propósito

Validar de manera exhaustiva y sin cargar el archivo completo en memoria la estructura tabular, esquema de columnas, rangos de intensidad y reconstrucción exacta píxel a píxel de los datasets exportados en CSV para KolektorSDD2 (`data/csv/kolektorsdd2.csv` y `data/csv/kolektorsdd2-inverted.csv`).

Este laboratorio actúa como control de calidad e integridad matemática tras la ejecución de `experiments.export_kolektorsdd2_csv`.

## 2. Esquema tabular (16387 columnas)

A diferencia de MVTec-AD (que incluye `object_name` y `defect_name`), KolektorSDD2 es un dataset de objeto único sin máscaras en el árbol procesado final, por lo que su esquema optimizado consta de:

1. `split`: partición (`train` o `test`).
2. `type`: clasificación binaria (`good` o `defect`).
3. `file_name`: nombre del archivo PNG original (ej. `10021.png`).
4. `px_00000` a `px_16383`: vector plano de intensidades en escala de grises uint8 en orden row-major (`row * 128 + col`).
5. `width`: ancho en píxeles (`128`).
6. `height`: alto en píxeles (`128`).

## 3. Qué mide y valida

1. **Esquema e integridad de cabecera**: comprueba que los nombres y el orden exacto de las 16387 columnas coincidan con la especificación.
2. **Dimensiones y formato**: verifica que cada fila contenga exactamente 16387 campos y que `width == height == 128`.
3. **Rangos fotométricos**: comprueba que los valores numéricos de píxeles pertenezcan estrictamente al rango uint8 `[0, 255]`.
4. **Conteos exactos de registros**:
   - `train/good`: 2085 muestras
   - `train/defect`: 246 muestras
   - `test/good`: 894 muestras
   - `test/defect`: 110 muestras
   - **Total**: 3335 filas
5. **Reconstrucción píxel a píxel (round-trip)**:
   - Selecciona $N$ muestras pseudoaleatorias con semilla controlable.
   - Reconstruye la matriz 2D $[128, 128]$ a partir del vector tabular.
   - Comprueba la igualdad estricta con la imagen PNG original en disco (`max(abs(orig - recon)) == 0`).
   - Genera una figura comparativa visual con mapas de calor de diferencia guardada en `csv_roundtrip.png` (o `csv_roundtrip_inverted.png`).
6. **Nota metodológica sobre inversión y letterboxing**:
   - En el pipeline de preprocesamiento, la inversión fotométrica (`255 - x`) se aplicó **antes** del letterbox centrado con `pad_value = 0` (negro).
   - Por diseño, las barras de padding lateral tienen valor 0 en ambas variantes (estándar e invertida), evitando marcos blancos espurios.
   - En consecuencia, la relación `inv == 255 - std` se cumple estrictamente dentro del área activa del objeto, pero no sobre las bandas de relleno.

## 4. Comandos de ejecución manual (PowerShell en carpeta `code/`)

```powershell
# 1. Verificación de la variante estándar:
.\.venv\Scripts\python.exe understanding/11_verify_kolektorsdd2_csv/verify.py

# 2. Verificación de la variante invertida:
.\.venv\Scripts\python.exe understanding/11_verify_kolektorsdd2_csv/verify.py --inverted
```
