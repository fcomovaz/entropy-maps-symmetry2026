# Verificación y validación de ida y vuelta de MVTec-AD CSV

## 1. Propósito

Validar la integridad, consistencia dimensional, fidelidad de píxeles y correspondencia unívoca del dataset serializado en formato tabular CSV (`data/csv/mvtec-ad.csv` y `data/csv/mvtec-ad-inverted.csv`) frente a las imágenes preprocesadas en disco (`data/preprocessed/mvtec-ad/` y `data/preprocessed/mvtec-ad-inverted/`).

## 2. Qué mide y valida

1. **Esquema de cabecera**: confirma que las 16391 columnas (`object_name, split, type, defect_name, file_name, px_00000...px_16383, width, height`) coincidan exactamente con la especificación definida.
2. **Cardinalidad y distribución**:
   - Total esperado: 6612 filas (sin contar la fila de encabezado).
   - Distribución por split y tipo:
     - `train/good`: 3629
     - `test/good`: 467
     - `test/defect`: 1258
     - `ground_truth/defect`: 1258
3. **Consistencia de valores**:
   - `width` y `height` coinciden con `target_size = 128`.
   - Intensidad de píxeles contenida en el rango `[0, 255]`.
   - Máscaras de `ground_truth` restringidas a `{0, 255}`.
4. **Reconstrucción exacta píxel a píxel**:
   - Muestreo aleatorio determinista de filas mediante generador pseudoaleatorio con semilla fija (`--seed 42`).
   - Reconstrucción de la matriz 2D `[128, 128]` en orden row-major (C-order).
   - Comprobación de igualdad estricta con la imagen PNG original en disco (diferencia absoluta máxima = 0).
5. **Artefacto visual**:
   - Genera matriz comparativa (PNG fuente, reconstrucción CSV, diferencia absoluta) guardada en:
     - `understanding/09_verify_mvtec_csv/csv_roundtrip.png`
     - `understanding/09_verify_mvtec_csv/csv_roundtrip_inverted.png`

## 3. Comandos de ejecución manual

```powershell
# Verificación de la exportación estándar:
.\code\.venv\Scripts\python.exe code/understanding/09_verify_mvtec_csv/verify.py

# Verificación de la exportación invertida:
.\code\.venv\Scripts\python.exe code/understanding/09_verify_mvtec_csv/verify.py --inverted
```
