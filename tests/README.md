# Pruebas del proyecto

Desde la raíz, PowerShell, sin activar el entorno:

```powershell
.\code\.venv\Scripts\python.exe .\code\tests\test_environment.py
.\code\.venv\Scripts\python.exe .\code\understanding\03_entropy_benchmark\test_entropy.py
.\code\.venv\Scripts\python.exe .\code\understanding\04_lightweight_cnns\tests\test_models.py
```

| Prueba | Qué demuestra | Requisitos |
|---|---|---|
| `test_environment.py` | Imports, asignación/transferencia CUDA, matmul y forward de una red mínima | Entorno del proyecto y GPU CUDA real; falla sin CUDA |
| `03/test_entropy.py` | Cinco tests: casos analíticos, cotas fixed-point, LUT reutilizada, histogramas y ranking; coincidencia de patches de 01 | Imagen, coordenadas y muestras conservadas en 01 |
| `04/tests/test_models.py` | Ocho tests: shapes, backward/Adam, parámetros, depthwise, residuales y datos reproducibles | PyTorch; CPU obligatorio, CUDA condicional en esta suite |

La suite CNN ejecuta 140 forwards y 28 pasos de entrenamiento cuando están
disponibles los siete modelos y CPU/CUDA. Sus dos tests CUDA omitidos no validan
la GPU: ejecutar siempre el test del entorno en este equipo.

01 incluye `self_test()` y reconstrucción exacta en cada ejecución del ejemplo.
02 valida que los ocho métodos coincidan bin a bin con H1 en sus 35 entradas.
03 también comprueba 7,668 comparaciones exactas durante el benchmark y permite
validar CSV/hash/ranking con `analyze_results.py`. Consultar sus README antes de
repetir porque esos scripts escriben archivos.

No hay pytest requerido ni una suite global que descubra automáticamente todo.
No ejecutar Python con `-O`: los laboratorios usan `assert` para validar.
Los futuros tests de `src/` se añadirán aquí cuando exista código formal.
