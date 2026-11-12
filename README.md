# Entorno y guía de ejecución

La documentación de síntesis científica e hipótesis vive en [HYPOTHESIS.md](HYPOTHESIS.md) y la auditoría metodológica profunda en [REVIEW_01.md](REVIEW_01.md).

Salvo los bloques de instalación inicial y réplica temporal desde `D:\Publications\2026-10-Symmetry-Entropy`, los comandos de módulos (`-m src.*`, `-m experiments.*`, `-m unittest`) se ejecutan desde `D:\Publications\2026-10-Symmetry-Entropy\code` en PowerShell usando `.\.venv\Scripts\python.exe`; la activación es opcional. Si el entorno ya existe y pasa las pruebas, no reinstalarlo.

## Comprobación al retomar

```powershell
Set-Location D:\Publications\2026-10-Symmetry-Entropy\code
.\.venv\Scripts\python.exe tests/test_environment.py
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover tests -v
.\.venv\Scripts\python.exe understanding/03_entropy_benchmark/test_entropy.py
.\.venv\Scripts\python.exe understanding/04_lightweight_cnns/tests/test_models.py
.\.venv\Scripts\python.exe -m unittest understanding/13_entropy_maps_optimization/tests/test_entropy_variants.py -v
```

Resultado esperado: `Environment verification: PASS`, `No broken requirements found.`, 208 tests formales en `tests/` (`tests/README.md`), cinco tests de entropía (`03`), ocho suites de CNN (`04`) y trece tests de variantes piramidales (`13`) correctos. En este equipo los tests CUDA deben ejecutarse, no omitirse. La suite de CNN incluye CPU y CUDA, pero por sí sola permite omitir CUDA en otros equipos; el test del entorno lo exige. Los tests de laboratorio se invocan explícitamente porque un único `discover tests` no recorre `understanding/` (ver detalles en [tests/README.md](tests/README.md), incluyendo el ajuste `PYTHONPATH` para el Lab 07).

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

Las rutas de esta lista son relativas a `code/`. Documentos clave de síntesis en `code/`: [HYPOTHESIS.md](HYPOTHESIS.md) (verificación de hipótesis y análisis pre-MCU Python vs C99 vs CNNs), [REVIEW_01.md](REVIEW_01.md) (auditoría forense) y [RESUME.md](RESUME.md) (estado técnico actual).

- [src/](src/README.md): módulos formales reutilizables (`src/preprocessing/`, `src/datasets/`, `src/features/` con backend C99 `entropy_pyramid` y `native_entropy.py`, `src/models/` incl. `LinearDotCoreClassifier` y `src/models/cnn/`, y `src/profiling/` para benchmarking 100 % CPU justo y perfilado granular).
- [experiments/](experiments/README.md): puntos de entrada CLI para experimentos reproducibles (`preprocess_*.py`, `export_*.py`, `export_entropy_maps.py`, `append_asymmetry_features.py`, `evaluate_descriptors.py`, `evaluate_cnn.py`, `compare_baseline_vs_cnn.py`, `benchmark_cpu_fair.py`, `profile_pipelines.py`).
- [understanding/](understanding/README.md): trece laboratorios pedagógicos (`01_patch_partition/` a `13_entropy_maps_optimization/`).
- [tests/](tests/README.md): verificación del entorno CUDA, suites de preprocesamiento, CSV, mapas/asimetría, backend nativo C99 (`test_native_entropy.py`), clasificación, CNNs, profiling CPU (`test_profiling_*.py`, `test_benchmark_cpu_fair_cli.py`) y tests de laboratorios.
- [results/](results/README.md): diagnóstico, logs, clasificación (`results/classification/`), checkpoints CNN (`results/cnn/`), comparación 100 % CPU con C99 (`results/comparison_cpu/`), comparación pre-MCU puro Python (`results/comparison_cpu_python/`, local ignorado por Git), profiling científico (`results/profiling/`) y registro histórico (`results/comparison/` marcado *deprecated*).

`notebooks/` se retiró porque solo contenía `.gitkeep`. Jupyter permanece instalado, pero no hay notebooks que ejecutar en esta rama. No se cambiaron paquetes por esta limpieza.

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

### Comandos de ejecución (PowerShell desde `code/`)

```powershell
# 1. Pruebas unitarias y de integración
.\.venv\Scripts\python.exe -m unittest tests/test_preprocessing.py

# 2. Conteo previo (dry-run, sin escribir archivos)
.\.venv\Scripts\python.exe -m experiments.preprocess_mvtec_ad --dry-run

# 3. Procesamiento completo
.\.venv\Scripts\python.exe -m experiments.preprocess_mvtec_ad

# 4. Verificación de integridad y visualización comparativa
.\.venv\Scripts\python.exe understanding/08_verify_mvtec_preprocessing/verify.py
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

### Comandos de ejecución (PowerShell desde `code/`)

```powershell
# 1. Pruebas unitarias y de integración de exportación CSV
.\.venv\Scripts\python.exe -m unittest tests/test_csv_export.py

# 2. Conteo previo (dry-run, sin escribir archivos)
.\.venv\Scripts\python.exe -m experiments.export_mvtec_ad_csv --dry-run
.\.venv\Scripts\python.exe -m experiments.export_mvtec_ad_csv --inverted --dry-run

# 3. Exportación completa estándar e invertida
.\.venv\Scripts\python.exe -m experiments.export_mvtec_ad_csv
.\.venv\Scripts\python.exe -m experiments.export_mvtec_ad_csv --inverted

# 4. Verificación de integridad y reconstrucción píxel a píxel
.\.venv\Scripts\python.exe understanding/09_verify_mvtec_csv/verify.py
.\.venv\Scripts\python.exe understanding/09_verify_mvtec_csv/verify.py --inverted
```

## Preprocesamiento y clasificación de datasets (KolektorSDD2)

Clasifica las muestras en `train/good`, `train/defect`, `test/good` y `test/defect` inspeccionando la máscara binaria nativa `_GT` (`mask.max() > 0`), estandarizando a escala de grises (`mode 'L'`) y tamaño cuadrado `128x128` mediante letterboxing centrado con valor `0`. Las máscaras de defectos no se almacenan en la salida procesada.

### Comandos de ejecución (PowerShell desde `code/`)

```powershell
# 1. Pruebas unitarias y de integración de KolektorSDD2
.\.venv\Scripts\python.exe -m unittest tests/test_kolektorsdd2.py

# 2. Conteo previo (dry-run, sin escribir archivos)
.\.venv\Scripts\python.exe -m experiments.preprocess_kolektorsdd2 --dry-run
.\.venv\Scripts\python.exe -m experiments.preprocess_kolektorsdd2 --inverted --dry-run

# 3. Procesamiento completo estándar e invertido
.\.venv\Scripts\python.exe -m experiments.preprocess_kolektorsdd2
.\.venv\Scripts\python.exe -m experiments.preprocess_kolektorsdd2 --inverted

# 4. Verificación de integridad, conteos y visualización comparativa
.\.venv\Scripts\python.exe understanding/10_verify_kolektorsdd2/verify.py
.\.venv\Scripts\python.exe understanding/10_verify_kolektorsdd2/verify.py --inverted
```

### Configuración y salida

- Configuración de dataset y rutas: `configs/base.toml` (`[datasets.kolektorsdd2]`).
- Parámetros de preprocesamiento: `configs/preprocessing/kolektorsdd2.toml`.
  - `target_size = 128`, `pad_value = 0`, `padding_anchor = "center"`.
  - `resampling = "LANCZOS"`, conversión a escala de grises ITU-R 601 luma.
  - Clasificación: `defect_threshold = 0`, carpetas `good` y `defect`.
  - Emparejamiento: regex `^\d+$` que ignora copias y archivos huérfanos.
- Directorios de salida: `data/preprocessed/KolektorSDD2/` y `data/preprocessed/KolektorSDD2-inverted/`.

## Exportación tabular CSV (KolektorSDD2)

Aplana las imágenes preprocesadas de KolektorSDD2 en formato tabular (un registro por fila con 16389 columnas) almacenado en `data/csv/kolektorsdd2.csv` (y `data/csv/kolektorsdd2-inverted.csv`), con metadatos de procedencia (`split, type, file_name`), vector de intensidades de píxeles en orden row-major (`px_00000...px_16383`) y dimensiones espaciales (`width, height`).

### Comandos de ejecución (PowerShell desde `code/`)

```powershell
# 1. Pruebas unitarias y de integración de exportación CSV KolektorSDD2
.\.venv\Scripts\python.exe -m unittest tests/test_kolektorsdd2_csv_export.py

# 2. Conteo previo (dry-run, sin escribir archivos)
.\.venv\Scripts\python.exe -m experiments.export_kolektorsdd2_csv --dry-run
.\.venv\Scripts\python.exe -m experiments.export_kolektorsdd2_csv --inverted --dry-run

# 3. Exportación completa estándar e invertida
.\.venv\Scripts\python.exe -m experiments.export_kolektorsdd2_csv
.\.venv\Scripts\python.exe -m experiments.export_kolektorsdd2_csv --inverted

# 4. Verificación de integridad y reconstrucción píxel a píxel
.\.venv\Scripts\python.exe understanding/11_verify_kolektorsdd2_csv/verify.py
.\.venv\Scripts\python.exe understanding/11_verify_kolektorsdd2_csv/verify.py --inverted
```

### Configuración y salida

- Configuración de dataset y rutas: `configs/base.toml` (`[datasets.kolektorsdd2]`).
- Parámetros de exportación: `configs/csv/kolektorsdd2.toml`.
- Archivos de salida: `data/csv/kolektorsdd2.csv` y `data/csv/kolektorsdd2-inverted.csv`.

## Anexado de asimetrías de entropía multiescala ($\mathbf{x}_{85} \rightarrow \mathbf{x}_{94}$)

Calcula los 9 residuos espaciales de asimetría multiescala (`asym_lr_2x2`, `asym_tb_2x2`, `asym_rot180_2x2`, `asym_lr_4x4`, `asym_tb_4x4`, `asym_rot180_4x4`, `asym_lr_8x8`, `asym_tb_8x8`, `asym_rot180_8x8`) directamente sobre las columnas de entropía de los archivos CSV planos (`data/csv/*-entropy.csv`) y los anexa al final de la tabla para generar los descriptores completos `x94-v1` (`data/csv/*-entropy-full.csv`), preservando metadatos y valores originales byte a byte y actualizando el sidecar JSON.

### Comandos de ejecución manual (PowerShell desde carpeta `code/`)

```powershell
# 1. Pruebas unitarias y de integración de asimetría
.\.venv\Scripts\python.exe -m unittest tests/test_asymmetry_maps.py
.\.venv\Scripts\python.exe -m unittest tests/test_asymmetry_csv.py

# 2. Conteo previo (dry-run, sin escribir archivos)
.\.venv\Scripts\python.exe -m experiments.append_asymmetry_features --config configs/asymmetry/kolektorsdd2.toml --dry-run
.\.venv\Scripts\python.exe -m experiments.append_asymmetry_features --config configs/asymmetry/mvtec_ad.toml --dry-run

# 3. Procesamiento completo estándar e invertido
.\.venv\Scripts\python.exe -m experiments.append_asymmetry_features --config configs/asymmetry/kolektorsdd2.toml
.\.venv\Scripts\python.exe -m experiments.append_asymmetry_features --config configs/asymmetry/kolektorsdd2.toml --inverted
.\.venv\Scripts\python.exe -m experiments.append_asymmetry_features --config configs/asymmetry/mvtec_ad.toml
.\.venv\Scripts\python.exe -m experiments.append_asymmetry_features --config configs/asymmetry/mvtec_ad.toml --inverted

# 4. Procesamiento masivo de todos los datasets y variantes
.\.venv\Scripts\python.exe -m experiments.append_asymmetry_features --all
```

## Clasificación supervisada, ablación y explicabilidad de descriptores ($\mathbf{x}_{94} = [\mathbf{x}_{85}, \mathbf{x}_9]$)

Evalúa la capacidad discriminativa supervisada (`good = 0`, `defect = 1`) del descriptor tabular compacto de 94 dimensiones mediante clasificadores ligeros (`dt`, `knn`, `lr`, `lr_dot` con kernel lineal plegado `LinearDotCoreClassifier`, y `rf`), comparando la contribución de las 85 características de entropía multiescala ($\mathbf{x}_{85}$) frente a las 9 de asimetría espacial ($\mathbf{x}_9$) y desglosando la importancia por escala piramidal ($1\times1$, $2\times2$, $4\times4$, $8\times8$) y celda espacial 2D.

- En **KolektorSDD2** (`configs/classification/kolektorsdd2.toml`), utiliza el split oficial `train`/`test` (`split_mode = "presplit"`).
- En **MVTec-AD** (`configs/classification/mvtec_ad.toml`), filtra en memoria las filas `split == "ground_truth"` y aplica una partición estratificada en memoria (`split_mode = "stratified"`, `test_size = 0.20`, `seed = 42`) por categoría de objeto, **sin modificar los CSV en disco**.

### Comandos de ejecución manual (PowerShell desde carpeta `code/`)

```powershell
# 1. Pruebas unitarias y de integración del clasificador y motor de explicabilidad
.\.venv\Scripts\python.exe -m unittest tests/test_descriptor_classifier.py -v

# 2. Validación previa (dry-run, sin entrenar ni escribir artefactos)
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/kolektorsdd2.toml --dry-run
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --object bottle --dry-run

# 3. Evaluación completa de los 5 clasificadores (dt, knn, lr, lr_dot, rf)
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/kolektorsdd2.toml --classifier all --overwrite
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/kolektorsdd2.toml --classifier all --inverted --overwrite
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --object bottle --classifier all --overwrite
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --object bottle --classifier all --inverted --overwrite

# 4. MVTec-AD barrido completo de las 15 categorías de objetos
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --all-objects --classifier rf
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --all-objects --classifier rf --inverted
```

## Backend nativo C99, benchmark 100 % CPU y profiling (`x94` vs CNNs)

Para compilar la librería compartida C99 de pirámide de entropía (`src/features/native/build/entropy_pyramid.dll`) y ejecutar las comparaciones justas 100 % CPU (`benchmark_cpu_fair` y `profile_pipelines`) tanto con backend `auto` (`c_single_pass`) como en modo puro Python (`--entropy-backend numpy_finest_pyramid` / `numpy_baseline`):

```powershell
Set-Location D:\Publications\2026-10-Symmetry-Entropy\code
.\.venv\Scripts\python.exe -m src.features.build_native --config configs/features/native_entropy.toml --clean
.\.venv\Scripts\python.exe -m unittest tests/test_native_entropy.py -v
```

La guía detallada de parámetros CLI para `export_entropy_maps.py`, `evaluate_descriptors.py`, `evaluate_cnn.py`, `benchmark_cpu_fair.py` y `profile_pipelines.py` se encuentra en [experiments/README.md](experiments/README.md), y los resultados consolidados en [HYPOTHESIS.md](HYPOTHESIS.md).

La [guía de CNN](understanding/04_lightweight_cnns/README.md) incluye inspección,
entrenamiento, inferencia y benchmark desde la raíz, con salidas por corrida.
Para abrir Jupyter opcionalmente (desde `D:\Publications\2026-10-Symmetry-Entropy`):

```powershell
.\code\.venv\Scripts\jupyter.exe lab
```
