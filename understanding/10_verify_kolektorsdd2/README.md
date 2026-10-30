# Verificación y validación visual de preprocesamiento KolektorSDD2

## 1. Propósito

Inspeccionar, medir y validar visualmente los resultados del pipeline de preprocesamiento y clasificación binaria aplicado al dataset industrial KolektorSDD2 (`data/raw/KolektorSDD2` → `data/processed/KolektorSDD2` y `data/processed/KolektorSDD2-inverted`).

Este laboratorio actúa como herramienta de control de calidad e integridad tras la ejecución de `experiments.preprocess_kolektorsdd2`.

## 2. Qué mide y valida

1. **Integridad dimensional y tipo**: verifica que el 100% de los archivos preprocesados tengan dimensiones exactas de `128x128` píxeles y modo `'L'` (un solo canal uint8).
2. **Exclusión estricta de máscaras**: comprueba que ninguna máscara `_GT` se haya guardado en el árbol de salida procesado (las máscaras sólo se emplean en resolución nativa para clasificar).
3. **Distribución y conteos exactos por partición**:
   - `train`: 2085 imágenes `good`, 246 imágenes `defect` (total: 2331).
   - `test`: 894 imágenes `good`, 110 imágenes `defect` (total: 1004).
   - Detección y reporte de discrepancias frente a los valores esperados.
4. **Reducción de huella en disco**: calcula el tamaño total real en MB para los archivos de imagen brutos vs. preprocesados.
5. **Inspección visual cualitativa y fotométrica**: genera una matriz comparativa con Matplotlib conteniendo:
   - Muestra `train/good`, `train/defect`, `test/good` y `test/defect`.
   - Comparación visual: imagen raw vertical (~1:2.8), máscara binaria original, imagen procesada letterboxed 128x128 e histogramas de intensidad superpuestos.
   - Guarda el artefacto en `understanding/10_verify_kolektorsdd2/outputs/kolektorsdd2_verification.png` (o `_inverted.png`).

## 3. Comandos de ejecución manual (PowerShell en carpeta `code/`)

```powershell
# Verificación de la variante estándar:
.\.venv\Scripts\python.exe understanding/10_verify_kolektorsdd2/verify.py

# Verificación de la variante invertida:
.\.venv\Scripts\python.exe understanding/10_verify_kolektorsdd2/verify.py --inverted
```
