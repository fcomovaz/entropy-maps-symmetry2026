# CNN ligeras: laboratorio educativo

Siete clasificadores grayscale para leer, ejecutar y comparar. No son modelos
entrenados para defectos reales ni forman parte del pipeline del paper.
Todo el código, tests y resultados de este laboratorio vive aquí.

| Model | Main idea | Approx params (2 clases) | Intended use |
|---|---|---:|---|
| `tiny_cnn` / TinyCNN | 3 Conv–BN–ReLU, 2 MaxPool | 23,538 | Baseline clásico |
| `small_cnn` / SmallCNN | 4 bloques, canales 16→32→64→128 | 97,650 | CNN convencional más profunda |
| `ds_cnn` / DepthwiseSeparableCNN | Depthwise 3×3 + pointwise 1×1 | 12,866 | Estudiar separación espacial/canales |
| `mobilenet_like` / MobileNetLikeCNN | 6 bloques separables, reducción progresiva | 70,786 | Backbone ligero propio |
| `tiny_resnet` / TinyResidualCNN | 3 bloques con identidad/proyección | 77,234 | Comprender conexiones residuales |
| `ultra_tiny` / UltraTinyCNN | 8→16→32 canales, 2 bloques separables | 1,154 | Explorar capacidad mínima conceptual TinyML |
| `torchvision_mobilenet` / TorchvisionMobileNet | MobileNetV3-Small de torchvision | 1,519,618 | Referencia conocida, sin pesos descargados |

Los conteos corresponden a `num_classes=2`, incluidos parámetros de BatchNorm.
Todos devuelven **logits** `[B, num_classes]`, reciben `[B, 1, H, W]` y usan
`AdaptiveAvgPool2d(1)` antes de la clasificación. No hay softmax en las redes.
Los tests cubren 32, 64, 128, 224 y 256 píxeles por lado, con 2 y 5 clases.
La independencia de resolución es estructural: no garantiza igual calidad al
cambiar tamaño. La profundidad de downsampling también cambia cuánto detalle queda.

## Ejecutar en Windows PowerShell

### Receta desde la raíz, con salidas separadas

La siguiente secuencia usa el Python exacto del proyecto y un directorio nuevo
por corrida. Ejecutarla desde `D:\Publications\2026-10-Symmetry-Entropy`, en la
misma ventana de PowerShell. No necesita activar el entorno ni descargar datos:

```powershell
$cnnLab = '.\code\understanding\04_lightweight_cnns'
$cnnRun = Join-Path $cnnLab ('outputs\replay-' + [guid]::NewGuid().ToString('N'))
.\code\.venv\Scripts\python.exe "$cnnLab\inspect_models.py"
.\code\.venv\Scripts\python.exe "$cnnLab\inspect_architecture.py" --model ds_cnn --input-size 128 --leaf
.\code\.venv\Scripts\python.exe "$cnnLab\train_demo.py" --model tiny_cnn --synthetic --epochs 3 --seed 42 --device cuda --output "$cnnRun\tiny_cnn.pt"
if ($LASTEXITCODE -ne 0) { throw 'Falló el entrenamiento de prueba.' }
.\code\.venv\Scripts\python.exe "$cnnLab\predict_image.py" --model tiny_cnn --image "$cnnLab\outputs\example_rgb.png" --checkpoint "$cnnRun\tiny_cnn.pt" --num-classes 2 --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Falló la inferencia.' }
.\code\.venv\Scripts\python.exe "$cnnLab\benchmark_inference.py" --input-size 128 --output "$cnnRun\benchmark.json"
if ($LASTEXITCODE -ne 0) { throw 'Falló el benchmark.' }
.\code\.venv\Scripts\python.exe "$cnnLab\tests\test_models.py"
```

Esperar siete filas de modelos con salida `[4, 2]`, traza de arquitectura,
tres épocas con loss/accuracy, checkpoint, logits/probabilidades y un JSON
con 14 filas de benchmark cuando CPU/CUDA están disponibles. La predicción
usa el tamaño 64 guardado por el entrenamiento; el benchmark usa 128.
`example_rgb.png` es una imagen artificial ya guardada, no una muestra validada
de defectos. No se espera una clase «correcta» ni una accuracy objetivo.

Para comprobar rápidamente el benchmark, agregar `--warmup 1 --iterations 3`;
para entrenamiento corto usar `--epochs 1 --samples 16 --input-size 32`.
Estos ajustes solo prueban ejecución; el protocolo de benchmark completo usa
10 warmups y 50 mediciones. El benchmark no carga el checkpoint entrenado:
mide forwards de arquitecturas recién inicializadas.

Para probar el modo carpetas con el fixture artificial ya conservado:

```powershell
.\code\.venv\Scripts\python.exe "$cnnLab\train_demo.py" --model ds_cnn --data "$cnnLab\outputs\folder_fixture" --epochs 1 --device cpu --output "$cnnRun\ds_cnn_folder.pt"
.\code\.venv\Scripts\python.exe "$cnnLab\predict_image.py" --model ds_cnn --image "$cnnLab\outputs\example_rgb.png" --checkpoint "$cnnRun\ds_cnn_folder.pt" --num-classes 2 --device cpu
```

El fixture contiene 16 imágenes train/8 val y dos clases, `defective=0` y
`normal=1`. Solo valida la lectura de carpetas. Para datos propios sustituir
`--data` por el directorio con `train/` y `val/`; no hay división automática.
Para comprobar la advertencia de pesos aleatorios, ejecutar `predict_image.py`
con la misma imagen omitiendo `--checkpoint`.

### Trabajar dentro de la carpeta del laboratorio

Usar el entorno **existente**, sin instalar ni actualizar paquetes:

```powershell
cd D:\Publications\2026-10-Symmetry-Entropy\code
cd understanding\04_lightweight_cnns

..\..\.venv\Scripts\python.exe inspect_models.py
..\..\.venv\Scripts\python.exe inspect_models.py --input-size 256 --num-classes 5 --device cuda
..\..\.venv\Scripts\python.exe inspect_architecture.py --model tiny_cnn --input-size 128
..\..\.venv\Scripts\python.exe inspect_architecture.py --model ds_cnn --input-size 128
..\..\.venv\Scripts\python.exe inspect_architecture.py --model ds_cnn --input-size 128 --leaf
..\..\.venv\Scripts\python.exe inspect_architecture.py --model tiny_resnet --input-size 32
..\..\.venv\Scripts\python.exe benchmark_inference.py --input-size 128
..\..\.venv\Scripts\python.exe train_demo.py --model tiny_cnn --synthetic --epochs 3
..\..\.venv\Scripts\python.exe train_demo.py --model ds_cnn --synthetic --epochs 3 --device cpu
..\..\.venv\Scripts\python.exe tests\test_models.py
..\..\.venv\Scripts\python.exe -m unittest discover -s tests -v
..\..\.venv\Scripts\python.exe -m pip check
```

Estos comandos tampoco requieren activar el entorno ni cambiar políticas de PowerShell.

`pytest` no está instalado en el entorno inspeccionado. Los tests usan únicamente
`unittest` de la biblioteca estándar; no es necesario instalar pytest.
CUDA se prueba cuando está disponible; en otro equipo esos dos tests se omiten
explícitamente. En este equipo se exige además la verificación real del entorno:

```powershell
..\..\.venv\Scripts\python.exe ..\..\tests\test_environment.py
```

`--device cuda` falla si CUDA no está disponible; `--device auto` utiliza CUDA
si existe y CPU en caso contrario. Inspección usa CPU por defecto; entrenamiento
e inferencia usan auto. Scripts fijan un hilo CPU para evitar overhead en redes
pequeñas; benchmark y entrenamiento admiten `--threads`.

## Leer las arquitecturas

Abrir primero `models/tiny_cnn.py`, después `models/ds_cnn.py` y
`models/tiny_resnet.py`. En un bloque separable, `groups=in_channels` aplica
un filtro independiente a cada canal. La convolución 1×1 posterior mezcla
canales. Sin contar BN ni bias, una convolución convencional 3×3 usa
`9 * Cin * Cout` pesos, frente a `9 * Cin + Cin * Cout` en el bloque separable.
Esto reduce operaciones teóricas, pero no asegura menor latencia en esta GPU.

En el bloque residual se ve `output = fx + residual`. La rama residual es
identidad cuando coinciden dimensiones; si cambian usa proyección 1×1 con stride.
La ReLU se aplica después de sumar.

`inspect_architecture.py` muestra arquitectura completa, shapes y parámetros
por bloque principal. `--leaf` muestra operaciones individuales. El flatten
funcional está escrito en forward y no genera un hook. Una ReLU compartida
puede aparecer varias veces en la traza.

MobileNet-like reutiliza el bloque de `ds_cnn.py`; no copia MobileNetV2/V3.
La referencia de torchvision conserva sus componentes originales (incluidos
SE y activaciones originales); solo se cambia la primera convolución a un canal
y el número de clases. No se añade atención a las seis CNN propias.
El wrapper acepta exclusivamente `weights=None`: no descarga nada.

Uso desde Python con esta carpeta como directorio actual:

```python
import torch
from models import TinyCNN, create_model

model = TinyCNN(num_classes=2)
x = torch.randn(4, 1, 128, 128)
logits = model(x)
print(logits.shape)  # torch.Size([4, 2])
model = create_model(name="ds_cnn", num_classes=5)
print(model(x).shape)  # torch.Size([4, 5])
```

Para inferencia usar `model.eval()` y `torch.inference_mode()`.
Durante entrenamiento BatchNorm necesita más de un valor por canal; con mapas
1×1 (referencia a 32 píxeles) usar batch >= 2. El demo exige batch >= 2 y omite
solo un batch final de una imagen. Inferencia admite batch 1 en todos los modelos.

## Imagen propia y checkpoint

```powershell
..\..\.venv\Scripts\python.exe predict_image.py --model tiny_cnn --image "D:\datos\defecto.png" --num-classes 2 --input-size 128
..\..\.venv\Scripts\python.exe train_demo.py --model tiny_cnn --synthetic --epochs 3 --device cuda
..\..\.venv\Scripts\python.exe predict_image.py --model tiny_cnn --image "D:\datos\defecto.png" --num-classes 2 --checkpoint outputs\tiny_cnn_synthetic.pt
```

El primer comando imprime literalmente:
`WARNING: model uses random/untrained weights. Predictions are not meaningful.`
También muestra logits, probabilidades calculadas con softmax fuera del modelo
y clase predicha. Un checkpoint sintético sigue sin validar defectos reales.

Preprocesado común en `common.py`: convertir con Pillow a grayscale de 8 bits,
resize cuadrado bilinear y float32 en [0,1]. El resize no conserva la relación
de aspecto; no hay normalización ImageNet. Los checkpoints del demo guardan
resolución y mapeo de clases, que inferencia reutiliza. `--input-size` permite
sobrescribir la resolución. Se acepta también un state_dict puro; en ese caso
el usuario debe asegurar mismo preprocesado, resolución y orden de clases.
Carga estricta con `weights_only=True` y validación de arquitectura/clases
cuando hay metadatos.

## Entrenamiento de prueba o carpetas

Sintético: 64 imágenes train y 16 val por defecto, 64×64, batch 8, Adam,
CrossEntropyLoss, learning rate 0.001 y seed 42. Clase 0 contiene franjas suaves
y ruido; clase 1 añade un cuadrado claro. Train y val usan seeds distintas.
Las métricas solo verifican ejecución, no son evidencia científica ni calidad
de detección. `--samples`, `--seed`, `--input-size`, `--batch-size` y `--lr`
son configurables. El modo sintético siempre tiene dos clases.

Para carpetas locales (sin descargas):

```text
dataset/
  train/
    normal/
    defective/
  val/
    normal/
    defective/
```

```powershell
..\..\.venv\Scripts\python.exe train_demo.py --model ds_cnn --data "D:\datos\dataset" --epochs 10
..\..\.venv\Scripts\python.exe predict_image.py --model ds_cnn --image "D:\datos\defecto.png" --checkpoint outputs\ds_cnn_folder.pt --num-classes 2
```

ImageFolder asigna índices alfabéticamente (`defective=0`, `normal=1` en ese
ejemplo). Train y val deben contener las mismas clases; se verifica el mapeo.
El número de clases se infiere de carpetas; `--num-classes` opcional lo valida.
No se divide automáticamente un dataset ni se añade augmentation.
Se usa `num_workers=0` para simplificar Windows.
Checkpoint final: `outputs/<model>_<synthetic|folder>.pt`; ejecuciones posteriores
sobrescriben esa ruta. `--output` permite elegir otra. No es un checkpoint para
reanudar optimizer: guarda pesos y configuración para inferencia.

## Benchmark y límites

Batch 1, grayscale 128×128, float32, 10 warmups y 50 forwards medidos por
modelo/dispositivo. `time.perf_counter()` y sincronización CUDA antes/después
de cada medida. Las transferencias CPU→GPU y la carga del modelo quedan fuera.
Se reportan mediana, media y mínimo en ms por batch. Los tiempos GPU incluyen
despacho Python y sincronización; no son tiempos de kernel puro.

`outputs/benchmark.json` conserva muestras individuales, versiones, dispositivo,
fecha UTC, seed y configuración. Se sobrescribe al repetir; usar `--output`
para conservar varias ejecuciones. Los MB son decimales: parámetros × 4 / 1e6
para float32; excluyen buffers BatchNorm, activaciones, gradientes, optimizer
y overhead del archivo. No equivalen a RAM/VRAM total.

**GTX 1050 timing is NOT representative of ESP32/microcontroller performance.**

No se mide selector, preprocesado ni pipeline completo. En estudios futuros
habrá que medir por separado `T_selector`, `T_model` y
`T_total = T_selector + T_model`. Repetir con el equipo sin otras cargas y no
interpretar pequeñas diferencias como superioridad estadística.

## Archivos y validación

- `models/`: seis arquitecturas propias, referencia opcional y registro explícito.
- `inspect_models.py`: tabla de parámetros y forward de todas las redes.
- `inspect_architecture.py`: arquitectura y shapes por bloque o por operación.
- `train_demo.py`: datos sintéticos/ImageFolder, épocas y guardado de pesos.
- `predict_image.py`: preprocesado, carga estricta, logits y softmax externo.
- `benchmark_inference.py`: latencia CPU/CUDA de forward con sincronización.
- `common.py`: seeds, dispositivo, preprocesado y conteo; `models/registry.py`
  conecta los nombres de CLI con constructores explícitos.
- `tests/test_models.py`: 140 combinaciones de forward (7 modelos × 5 tamaños ×
  2 cantidades de clases × CPU/CUDA), 28 pasos de backward/Adam, presupuestos
  de parámetros, conexiones residuales, depthwise y reproducibilidad sintética.
- `outputs/`: resultados y checkpoints locales de demostración; no versionar
  archivos grandes. `outputs/validation.md` registra la ejecución realizada.

Entorno inspeccionado: Python 3.11.9, PyTorch 2.14.0+cu126, torchvision
0.29.0+cu126, Pillow 12.3.0, GTX 1050 (4 GB, capability 6.1).
No se creó otro entorno, no se instalaron paquetes y no se cambiaron versiones.
