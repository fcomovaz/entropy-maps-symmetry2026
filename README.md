# Entorno y guía de ejecución

Todos los comandos de esta página se ejecutan desde `D:\Publications\2026-10-Symmetry-Entropy` en PowerShell. Usar siempre `.\code\.venv\Scripts\python.exe`; la activación es opcional. Si el entorno ya existe y pasa las pruebas, no reinstalarlo.

## Comprobación al retomar

```powershell
Set-Location D:\Publications\2026-10-Symmetry-Entropy
.\code\.venv\Scripts\python.exe .\code\tests\test_environment.py
.\code\.venv\Scripts\python.exe -m pip check
.\code\.venv\Scripts\python.exe .\code\understanding\03_entropy_benchmark\test_entropy.py
.\code\.venv\Scripts\python.exe .\code\understanding\04_lightweight_cnns\tests\test_models.py
```

Resultado esperado: `Environment verification: PASS`, `No broken requirements found.`, cinco tests de entropía y ocho tests de CNN correctos. En este equipo los tests CUDA deben ejecutarse, no omitirse. La suite de CNN incluye CPU y CUDA, pero por sí sola permite omitir CUDA en otros equipos; el test del entorno lo exige. Los tests locales se invocan por ruta porque un único discovery en `code/tests/` no recorre los laboratorios.

Los tests de entropía necesitan las salidas de 01, ya conservadas. Si faltan, ejecutar primero `patch_partition.py --patch-size 64`. Los comandos no necesitan pytest, Jupyter ni descargas de pesos.

## Entorno registrado

Entorno verificado el 2026-09-26 mediante operaciones reales en GPU:

| Componente | Valor |
| --- | --- |
| OS | Windows 10, build 19045, x86-64 |
| Python | 3.11.9, 64 bits |
| PyTorch | 2.14.0+cu126 |
| torchvision | 0.29.0+cu126 |
| CUDA runtime de PyTorch | 12.6 |
| CUDA disponible | True |
| GPU | NVIDIA GeForce GTX 1050 |
| Compute capability | (6, 1), Pascal |
| VRAM total | 4096 MiB; PyTorch informa aproximadamente 4.00 GiB |
| Driver NVIDIA | 582.66 |

## Instalación reproducible

Desde la raíz del proyecto, crear el venv **solo si no existe**:

```powershell
py -3.11 -m venv code\.venv
```

La activación es opcional; los comandos siguientes usan siempre el intérprete exacto:

```powershell
.\code\.venv\Scripts\Activate.ps1
.\code\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\code\.venv\Scripts\python.exe -m pip install --no-cache-dir --default-timeout=1000 --progress-bar off --log code/results/pytorch_cuda_install.log torch==2.14.0+cu126 torchvision==0.29.0+cu126 --index-url https://download.pytorch.org/whl/cu126
.\code\.venv\Scripts\python.exe -m pip install -r code\requirements.txt
.\code\.venv\Scripts\python.exe -m pip check
.\code\.venv\Scripts\python.exe .\code\tests\test_environment.py
```

El comando cu126 anterior es el que completó la reparación. Instalar PyTorch **antes** de `requirements.txt`: los sufijos `+cu126` fijan la variante, pero PyPI por sí solo no proporciona esas wheels. El archivo fija las dependencias directas; no es un lock completo de dependencias transitivas. La reparación actualizó también sympy a 1.14.0, requerido por PyTorch.

La wheel local verificada incluye `sm_61`; el diagnóstico histórico está en [results/environment_debug.md](results/environment_debug.md). No usar builds CUDA 13.x para esta GTX 1050. La versión 13.0 registrada por `nvidia-smi` expresa la capacidad del driver; el runtime realmente utilizado por PyTorch es 12.6. Estas versiones describen el entorno instalado, no una búsqueda de las versiones más recientes.

No se necesita CUDA Toolkit global: las wheels oficiales contienen las librerías runtime. No se instaló torchaudio. No lanzar instalaciones pip concurrentes ni interrumpir una descarga porque tarde varios minutos: la wheel torch ocupa aproximadamente 2.6 GB. El intento exitoso duró unos seis minutos, incluyendo instalación. `--no-cache-dir` evita reutilizar la caché y el timeout ampliado se aplica al socket, no limita la duración total.

El test exige CUDA y devuelve código 1 ante fallos; nunca hace fallback a CPU. Comprueba importaciones, GPU/capability/VRAM, asignación CUDA, transferencia de ida y vuelta, multiplicación pequeña con resultado conocido y una red mínima. La verificación inicial adicional ejecutó una multiplicación de 2048×2048 en `cuda:0` y transferencia de 1000×1000. No se entrenaron modelos ni se interpretó esa ejecución como benchmark.

El diagnóstico, la evidencia del intento original y las limitaciones están en [results/environment_debug.md](results/environment_debug.md). Los logs completos permanecen locales e ignorados por Git.

## Carpetas

Las rutas de esta lista son relativas a `code/`. Los comandos de este documento se ejecutan desde la raíz del proyecto.

- [src/](src/README.md): módulos formales reutilizables (`src/preprocessing/` para letterbox y MVTec-AD; `src/datasets/` para exportación tabular CSV).
- [experiments/](experiments/README.md): puntos de entrada CLI para experimentos del pipeline (`preprocess_mvtec_ad.py`, `export_mvtec_ad_csv.py`).
- [understanding/](understanding/README.md): nueve laboratorios pedagógicos (`01_patch_partition/` a `09_verify_mvtec_csv/`).
- [tests/](tests/README.md): verificación del entorno CUDA, suites de preprocesamiento, exportación CSV y tests de laboratorios.
- [results/](results/README.md): diagnóstico, logs y registro de revisión. Los resultados de cada laboratorio viven dentro de él.

`notebooks/` se retiró porque solo contenía `.gitkeep`. Jupyter permanece instalado, pero no hay notebooks que ejecutar. No se cambiaron paquetes por esta limpieza.

## Repetir 01-03 sin sobrescribir evidencia

Los tres scripts guardan junto a su propia ubicación y no tienen `--output`.
Crear una copia de ejecución fuera del proyecto permite conservar intactos los
resultados canónicos y los hashes del código. Ejecutar este bloque una vez por
sesión y continuar en la misma ventana de PowerShell:

```powershell
$projectPython = (Resolve-Path .\code\.venv\Scripts\python.exe).Path
$replayRoot = Join-Path $env:TEMP ('symmetry-replay-' + [guid]::NewGuid().ToString('N'))
$replayLabs = Join-Path $replayRoot 'code\understanding'
New-Item -ItemType Directory -Path $replayLabs -Force | Out-Null
foreach ($lab in @('01_patch_partition', '02_histogram_benchmark', '03_entropy_benchmark')) {
    Copy-Item -LiteralPath (Join-Path '.\code\understanding' $lab) -Destination $replayLabs -Recurse
}
Write-Output "Copia de ejecución: $replayLabs"

& $projectPython "$replayLabs\01_patch_partition\patch_partition.py" --patch-size 64
if ($LASTEXITCODE -ne 0) { throw 'Falló la partición; detener la reproducción.' }
& $projectPython "$replayLabs\03_entropy_benchmark\test_entropy.py"
if ($LASTEXITCODE -ne 0) { throw 'Fallaron los tests de entropía.' }
```

La copia incluye la imagen cacheada y los resultados históricos; no descarga
nada mientras el asset exista. Mantener 01, 02 y 03 como carpetas hermanas.
Para una comprobación corta de ambos benchmarks:

```powershell
& $projectPython "$replayLabs\02_histogram_benchmark\histogram_benchmark.py" --target-pixels-per-run 10000
if ($LASTEXITCODE -ne 0) { throw 'Falló el benchmark de histogramas.' }
& $projectPython "$replayLabs\03_entropy_benchmark\entropy_benchmark.py" --rounds 1 --entropy-repeats 2 --target-pixels 1000
if ($LASTEXITCODE -ne 0) { throw 'Falló el benchmark de entropía.' }
& $projectPython "$replayLabs\03_entropy_benchmark\analyze_results.py"
if ($LASTEXITCODE -ne 0) { throw 'Falló la validación de resultados.' }
```

Para repetir el protocolo completo, usar otra copia y omitir las opciones de
duración en ambos benchmarks. La ejecución completa puede tardar varios minutos.
La corta conserva casos y verificaciones, pero reduce repeticiones: **sus tiempos
no sustituyen las tablas históricas**. `analyze_results.py` lee los CSV de la copia,
comprueba su hash y compara datos/rankings contra el `run1/` copiado; luego genera
`summary.md` y `validation.json`. Sus rótulos históricos «Run2» se refieren a la
corrida canónica original; después de repetir, la carpeta contiene la nueva corrida.

Las salidas nuevas están en `$replayLabs\<laboratorio>\outputs` o `results`.
Guardar los artefactos que interesen fuera de TEMP antes de una limpieza del
sistema. Invocar los scripts originales directamente también funciona, pero
sobrescribe sus archivos fijos; los README históricos no se actualizan solos.

## Preprocesamiento de datasets (MVTec-AD)

El preprocesamiento estandariza las imágenes a escala de grises (`mode 'L'`) y tamaño `128x128` mediante letterboxing/padding centrado con valor `0` (negro), conservando la relación de aspecto original sin distorsiones geométricas.

### Comandos de ejecución

```powershell
# 1. Pruebas unitarias y de integración
.\code\.venv\Scripts\python.exe -m unittest tests/test_preprocessing.py

# 2. Conteo previo (dry-run, sin escribir archivos)
.\code\.venv\Scripts\python.exe -m experiments.preprocess_mvtec_ad --dry-run

# 3. Procesamiento completo
.\code\.venv\Scripts\python.exe -m experiments.preprocess_mvtec_ad

# 4. Verificación de integridad y visualización comparativa
.\code\.venv\Scripts\python.exe understanding/08_verify_mvtec_preprocessing/verify.py
```

### Configuración y salida

- Configuración de dataset y rutas: `configs/base.toml` (`[datasets.mvtecad]`).
- Parámetros de preprocesamiento: `configs/preprocessing/mvtec_ad.toml`.
  - `target_size = 128`, `pad_value = 0`, `padding_anchor = "center"`.
  - Imágenes: interpolación `LANCZOS`, conversión a escala de grises.
  - Máscaras (`ground_truth`): interpolación `NEAREST` para preservar valores discretos `{0, 255}`, sin alteración de niveles de gris.
- Directorio de salida: `data/preprocessed/mvtec-ad/` (refleja la estructura de subcarpetas de `data/raw/mvtec-ad/`).
- Nota metodológica: el reescalado a 128 px con interpolación de vecinos cercanos en máscaras puede atenuar o eliminar defectos minúsculos de pocos píxeles presentes en resoluciones nativas de 1024 px.

## Exportación tabular CSV (MVTec-AD)

Aplana las imágenes preprocesadas en formato tabular (un registro por fila con 16391 columnas) almacenado en `data/csv/mvtec-ad.csv` (y `data/csv/mvtec-ad-inverted.csv`), con metadatos de procedencia y píxeles en orden row-major (`px_00000...px_16383`).

### Comandos de ejecución

```powershell
# 1. Pruebas unitarias y de integración de exportación CSV
.\code\.venv\Scripts\python.exe -m unittest tests/test_csv_export.py

# 2. Conteo previo (dry-run, sin escribir archivos)
.\code\.venv\Scripts\python.exe -m experiments.export_mvtec_ad_csv --dry-run
.\code\.venv\Scripts\python.exe -m experiments.export_mvtec_ad_csv --inverted --dry-run

# 3. Exportación completa estándar e invertida
.\code\.venv\Scripts\python.exe -m experiments.export_mvtec_ad_csv
.\code\.venv\Scripts\python.exe -m experiments.export_mvtec_ad_csv --inverted

# 4. Verificación de integridad y reconstrucción píxel a píxel
.\code\.venv\Scripts\python.exe code/understanding/09_verify_mvtec_csv/verify.py
.\code\.venv\Scripts\python.exe code/understanding/09_verify_mvtec_csv/verify.py --inverted
```



La [guía de CNN](understanding/04_lightweight_cnns/README.md) incluye inspección,
entrenamiento, inferencia y benchmark desde la raíz, con salidas por corrida.
Para abrir Jupyter opcionalmente:

```powershell
.\code\.venv\Scripts\jupyter.exe lab
```
