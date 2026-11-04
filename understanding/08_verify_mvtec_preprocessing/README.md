# Verificación y validación visual de preprocesamiento MVTec-AD

## 1. Propósito

Inspeccionar, medir y validar visualmente los resultados del pipeline de preprocesamiento aplicado al dataset industrial real MVTec-AD (`data/raw/mvtec-ad` → `data/preprocessed/mvtec-ad` y `data/preprocessed/mvtec-ad-inverted`).

Este laboratorio no realiza transformaciones de entrenamiento ni modelos; actúa como herramienta de control de calidad e integridad tras la ejecución de `experiments.preprocess_mvtec_ad`.

## 2. Qué mide y valida

1. **Integridad dimensional y tipo**: verifica que el 100% de los archivos preprocesados tengan dimensiones exactas de `128x128` píxeles y modo `'L'` (un solo canal uint8).
2. **Preservación estricta de máscaras de defecto**: comprueba que las máscaras de `ground_truth` contengan únicamente etiquetas binarias pertenecientes a `{0, 255}`, sin la aparición de nuevos niveles de gris espurios generados por interpolaciones continuas.
3. **Consistencia de cardinalidad**: compara el conteo de archivos descubiertos en las 15 subcarpetas de categorías entre raw y preprocessed.
4. **Reducción de huella en disco**: calcula el tamaño total real en MB (medido, no estimado) y el porcentaje de compresión volumétrica al pasar de alta resolución/RGB a `128x128` escala de grises.
5. **Inspección visual cualitativa**: genera una matriz comparativa con Matplotlib de:
   - 3 muestras reales de categorías distintas (`train/good`).
   - 1 máscara de defecto real (`ground_truth`).
   - 1 caso sintético no cuadrado para validar el padding simétrico y la relación de aspecto.
   - Guarda el artefacto en `understanding/08_verify_mvtec_preprocessing/preprocessing_verification.png` (o `preprocessing_verification_inverted.png`).

## 3. Comandos de ejecución manual

```powershell
# Verificación de la variante estándar:
.\.venv\Scripts\python.exe understanding/08_verify_mvtec_preprocessing/verify.py

# Verificación de la variante con intensidad invertida:
.\.venv\Scripts\python.exe understanding/08_verify_mvtec_preprocessing/verify.py --inverted
```
