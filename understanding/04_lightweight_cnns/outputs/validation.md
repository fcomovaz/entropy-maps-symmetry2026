# Validación local — 2026-09-28

Implementación y resultados limitados a `code/understanding/04_lightweight_cnns/`.
No se modificó el pipeline, no se creó un entorno y no se instalaron paquetes.
El directorio raíz no contiene metadatos Git; no fue posible usar git diff.

## Entorno comprobado

- Python 3.11.9 del `code/.venv` existente.
- PyTorch 2.14.0+cu126; torchvision 0.29.0+cu126; CUDA runtime 12.6.
- GTX 1050, capability 6.1, 4 GB. Matmul CUDA real ejecutado.
- `code/tests/test_environment.py`: PASS, incluidos allocation, transfer,
  ejecución CUDA y forward de su red mínima (ese test no entrena).
- `python -m pip check`: `No broken requirements found.`
- pytest ausente: se utilizó unittest, sin instalaciones.

## Pruebas ejecutadas

`python tests/test_models.py`: **8 tests OK**, 11.183 segundos, ninguno omitido.

- 140 forwards: 7 arquitecturas × 5 tamaños (32/64/128/224/256) ×
  2 configuraciones de clases (2/5) × 2 dispositivos (CPU/CUDA).
- Todos con batch 2, salida `[2, num_classes]` y valores finitos.
- 28 pasos reales forward/backward/Adam: cada arquitectura con 2/5 clases
  en CPU/CUDA, input 32. Loss y gradientes finitos; cambio de pesos comprobado.
- Presupuestos de parámetros, estructura depthwise/pointwise, suma residual
  con identidad y proyección, y reproducibilidad del dataset sintético.
- `inspect_models.py`: las siete redes con batch 4, input 128, salida `[4,2]`.
- `inspect_architecture.py`: TinyCNN, DS-CNN con `--leaf`, TinyResNet y
  MobileNetV3-Small, incluidas trazas a resolución 32.

## Scripts y datos de prueba

Ver `smoke_runs.log` y `cli_checks.log` para comandos y salidas completas.

- TinyCNN: synthetic, CUDA, 3 épocas, 64 train/16 val, batch 8, input 64.
  Train loss: 0.5130 → 0.3764 → 0.2846. Val accuracy: 0.500 en las tres épocas.
  **No se demostró generalización ni calidad para defectos reales.**
- TinyCNN: synthetic, CPU, 1 época, mismos tamaños.
- Otras seis redes: synthetic, CUDA, 1 época, 16 train/16 val.
- DS-CNN: modo folder, CPU, 1 época; fixture local artificial de 16 train/8 val.
  Se verificó el mapeo alfabético `defective=0`, `normal=1`.
- `predict_image.py`: imagen RGB artificial convertida a grayscale; pesos
  aleatorios con warning; checkpoint sintético TinyCNN en CUDA; checkpoint
  folder DS-CNN en CPU. Logits, probabilidades y nombres de clases impresos.
- Checkpoints y PNG de prueba permanecen locales bajo outputs y están
  excluidos mediante el `.gitignore` de este módulo.

## Benchmark medido

`python benchmark_inference.py`: batch 1, input 128, float32, CPU 1 hilo,
10 warmups, 50 mediciones por modelo/dispositivo. JSON completo en
`benchmark.json`, con muestras, mediana/media/mínimo y entorno.

| Modelo | Parámetros (2 clases) | Mediana CPU ms | Mediana CUDA ms |
|---|---:|---:|---:|
| tiny_cnn | 23,538 | 5.1664 | 0.7082 |
| small_cnn | 97,650 | 7.0047 | 0.9061 |
| ds_cnn | 12,866 | 1.8075 | 1.2247 |
| mobilenet_like | 70,786 | 2.7151 | 2.7839 |
| tiny_resnet | 77,234 | 3.0645 | 1.6294 |
| ultra_tiny | 1,154 | 0.7414 | 0.9053 |
| torchvision_mobilenet | 1,519,618 | 6.5137 | 8.6134 |

Una sola ejecución local; no prueba superioridad estadística. Se excluyen
transferencias y preprocesado. Se incluyen overhead Python y sincronización
CUDA. No se midió selector, pipeline completo, FLOPs, RAM/VRAM pico ni energía.

**GTX 1050 timing is NOT representative of ESP32/microcontroller performance.**

No se encontraron fallos de ejecución. Datos reales, calidad predictiva y
hardware embebido quedan fuera del alcance de esta validación.
