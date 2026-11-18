# Implementación C99 portable pre-ESP32 (`c_imp/`)

Implementación C99 pura (escritorio Windows x64 vía MSVC `cl.exe` y lista para portación a MCU ESP32-CAM) en **doble modo aritmético** (`float32` y punto fijo entero `Q15` sin `<math.h>` ni asignación dinámica `malloc`/`free` en inferencia) para:

1. **Descriptor multiescala + asimetría $\mathbf{x}_{94}$ (`runtime/cimp_descriptor.{h,c}`):**
   - Cuantización de píxeles vía LUT de 256 entradas (`lut_bin_quant.h`), histograma en grilla fina $8\times 8$, suma piramidal ascendente ($8\times 8 \to 4\times 4 \to 2\times 2 \to 1\times 1$), cálculo de entropía Shannon vía LUT (`nlogn_full`, `term_per_level` o `log2_mantissa` seleccionable en TOML) y 9 residuos de asimetría LR/TB/R180.
2. **Clasificadores tabulares plegados (`runtime/infra_sklearn.{h,c}`):**
   - `lr_dot`: núcleo lineal con `StandardScaler` plegado (`cimp_dense_dot_f32` y `cimp_dense_dot_q15` / `_w32` con acumulador `int64_t`).
   - `mlp_16_dot`: MLP de 1 capa oculta ReLU con `StandardScaler` plegado (`cimp_mlp_1hidden_f32` y `cimp_mlp_1hidden_q15`).
3. **Zoo de 7 CNNs ligeras (`runtime/infra_pytorch.{h,c}`):**
   - Normalización de entrada por LUT `uint8 -> Qm.n`, `conv2d` (`k=1/3/5`), `dwconv2d`, `relu`, `hardswish`, `hardsigmoid`, `maxpool2d_2x2`, `global_avgpool2d`, `linear`, `add` residual y `scale_channels` (Squeeze-and-Excitation).
   - Pesos en `q15` (`int16_t`) cuando caben en presupuesto Flash o `int8_per_channel` automático (`torchvision_mobilenet`) usando arena estática precalculada por análisis de liveness.
4. **Despacho unificado y binario de verificación (`runtime/cimp_pipeline.{h,c}`, `app/desktop_main.c`, `tests/test_runtime.c`):**
   - Cronometraje separado de selector y modelo vía `QueryPerformanceCounter` (`cimp_platform.{h,c}`).

---

## Estructura de `c_imp/`

```text
c_imp/
├── runtime/
│   ├── cimp_fixed_point.h     # Primitivas q7/q15/q31/q63, redondeo half-up y saturación
│   ├── cimp_tensor.h          # Vistas NCHW [C, H, W] y códigos de estado CIMP_OK / < 0
│   ├── cimp_platform.{h,c}    # Reloj monotónico en nanosegundos (Win32 QPC / POSIX / MCU hook)
│   ├── cimp_descriptor.{h,c}  # Extractor x94-v1 f32 y q15 zero-malloc
│   ├── infra_sklearn.{h,c}    # Kernels f32/q15 para lr_dot y mlp_16_dot
│   ├── infra_pytorch.{h,c}    # Kernels CNN f32/q15 y q15_w8 (int8 per-channel)
│   └── cimp_pipeline.{h,c}    # Dispatcher unificado descriptor + ML + 7 CNNs
├── app/
│   └── desktop_main.c         # CLI desktop de paridad y benchmarking
├── tests/
│   └── test_runtime.c         # Tests unitarios C99 independientes de código generado
├── generated/                 # Headers/fuentes C99 autogenerados por src/c_export por target (ignorado por Git)
└── build/                     # Binarios .exe/.obj/.pdb compilados en escritorio (ignorado por Git)
```

---

## Comandos manuales (PowerShell desde `code/`)

### 1. Flujo estándar por dataset

#### KolektorSDD2
```powershell
# Validación rápida sin generar archivos (revisa config, imágenes y checkpoints)
.\.venv\Scripts\python.exe -m experiments.export_c_pipeline --config configs/c_imp/kolektorsdd2.toml --dry-run

# Exportar LUTs, vectores golden, lr_dot, mlp_16_dot y las 7 CNNs
.\.venv\Scripts\python.exe -m experiments.export_c_pipeline --config configs/c_imp/kolektorsdd2.toml

# Compilar binarios C99 y verificar paridad exacta Python-vs-C
.\.venv\Scripts\python.exe -m experiments.verify_c_pipeline --config configs/c_imp/kolektorsdd2.toml

# Compilar, verificar paridad y medir latencias en escritorio (timing_benchmark.csv)
.\.venv\Scripts\python.exe -m experiments.verify_c_pipeline --config configs/c_imp/kolektorsdd2.toml --benchmark

# Validar inputs y generar figuras diagnósticas y tablas resumen desde c_imp/generated + results/c_imp
.\.venv\Scripts\python.exe -m experiments.visualize_c_pipeline --config configs/c_imp/kolektorsdd2.toml --dry-run
.\.venv\Scripts\python.exe -m experiments.visualize_c_pipeline --config configs/c_imp/kolektorsdd2.toml
```

#### MVTec-AD (ej. `bottle`, o sustituir `--object` por otro subdataset y usar `--inverted` si aplica)
```powershell
.\.venv\Scripts\python.exe -m experiments.export_c_pipeline --config configs/c_imp/mvtec_ad.toml --object bottle
.\.venv\Scripts\python.exe -m experiments.verify_c_pipeline --config configs/c_imp/mvtec_ad.toml --object bottle --benchmark --visualize
.\.venv\Scripts\python.exe -m experiments.visualize_c_pipeline --config configs/c_imp/mvtec_ad.toml --object bottle

# Variante con imágenes invertidas
.\.venv\Scripts\python.exe -m experiments.export_c_pipeline --config configs/c_imp/mvtec_ad.toml --object bottle --inverted
.\.venv\Scripts\python.exe -m experiments.verify_c_pipeline --config configs/c_imp/mvtec_ad.toml --object bottle --inverted --benchmark
.\.venv\Scripts\python.exe -m experiments.visualize_c_pipeline --config configs/c_imp/mvtec_ad.toml --object bottle --inverted
```

### 2. Variantes y subconjuntos de `export_c_pipeline` y visualización

`experiments.export_c_pipeline` acepta opciones para exportar subconjuntos, limitar imágenes de prueba, compilar o renderizar las figuras justo al terminar (`--visualize`):

```powershell
# Exportar y compilar cimp_pipeline.exe en el mismo paso (sin correr toda la verificación)
.\.venv\Scripts\python.exe -m experiments.export_c_pipeline --config configs/c_imp/kolektorsdd2.toml --compile

# Exportar o verificar añadiendo --visualize para generar el resumen visual al finalizar
.\.venv\Scripts\python.exe -m experiments.export_c_pipeline --config configs/c_imp/kolektorsdd2.toml --visualize
.\.venv\Scripts\python.exe -m experiments.verify_c_pipeline --config configs/c_imp/kolektorsdd2.toml --benchmark --visualize

# Validación dry-run del visualizador sin escribir PNG ni tablas resumen
.\.venv\Scripts\python.exe -m experiments.visualize_c_pipeline --config configs/c_imp/kolektorsdd2.toml --dry-run

# Exportar solo un subconjunto de CNNs o clasificadores tabulares
.\.venv\Scripts\python.exe -m experiments.export_c_pipeline --config configs/c_imp/kolektorsdd2.toml --cnn-models ultra_tiny ds_cnn --classifier-models lr_dot

# Saltar CNNs (útil si solo quieres descriptor x94 + lr_dot + mlp_16_dot, evitando el .c grande de MobileNet)
.\.venv\Scripts\python.exe -m experiments.export_c_pipeline --config configs/c_imp/kolektorsdd2.toml --skip-cnn

# Saltar modelos clásicos tabulares o limitar el número de vectores golden exportados
.\.venv\Scripts\python.exe -m experiments.export_c_pipeline --config configs/c_imp/kolektorsdd2.toml --skip-tabular --max-images 50

# Ejecutar únicamente etapas concretas: luts | descriptor | classifier | cnn | vectors
.\.venv\Scripts\python.exe -m experiments.export_c_pipeline --config configs/c_imp/kolektorsdd2.toml --only luts vectors
```

Para ver todas las opciones disponibles de los tres CLIs:
```powershell
.\.venv\Scripts\python.exe -m experiments.export_c_pipeline --help
.\.venv\Scripts\python.exe -m experiments.verify_c_pipeline --help
.\.venv\Scripts\python.exe -m experiments.visualize_c_pipeline --help
```

### 3. Ejecución directa de los binarios C99 compilados

Tras compilar (`verify_c_pipeline` o `export_c_pipeline --compile`), puedes invocar los ejecutables C99 directamente desde PowerShell (ejemplo sobre `<target> = kolektorsdd2` o `mvtec_ad_bottle`):

```powershell
# Tests unitarios C99 standalone del runtime (fixed-point, sklearn, pytorch kernels)
.\c_imp\build\_runtime_tests\cimp_test_runtime.exe

# Descriptor x94 + clasificador tabular (ej. lr_dot en Q15 con benchmark)
.\c_imp\build\kolektorsdd2\cimp_pipeline.exe `
  --manifest results/c_imp/kolektorsdd2/vectors/manifest.csv `
  --images-dir results/c_imp/kolektorsdd2/vectors/images `
  --mode descriptor --model lr_dot --arith q15 `
  --out-csv results/c_imp/kolektorsdd2/c_outputs/manual_lr_q15.csv `
  --x94-csv results/c_imp/kolektorsdd2/c_outputs/manual_x94_q15.csv `
  --bench-warmup 50 --bench-reps 300 `
  --bench-csv results/c_imp/kolektorsdd2/c_outputs/manual_bench_lr_q15.csv

# Inferencia CNN en punto fijo Q15 o float32 (ultra_tiny, ds_cnn, tiny_cnn, mobilenet_like, tiny_resnet, small_cnn, torchvision_mobilenet)
.\c_imp\build\kolektorsdd2\cimp_pipeline.exe `
  --manifest results/c_imp/kolektorsdd2/vectors/manifest.csv `
  --images-dir results/c_imp/kolektorsdd2/vectors/images `
  --mode cnn --model ultra_tiny --arith q15 `
  --out-csv results/c_imp/kolektorsdd2/c_outputs/manual_ultra_tiny_q15.csv `
  --bench-warmup 50 --bench-reps 300 `
  --bench-csv results/c_imp/kolektorsdd2/c_outputs/manual_bench_ultra_tiny_q15.csv
```

En `--mode descriptor`, `--model` acepta `x94_only`, `lr_dot` o `mlp_16_dot`.

### 4. Pruebas automáticas y laboratorio exploratorio (Lab 15)

```powershell
# Verificación del entorno Python + PyTorch CUDA
.\.venv\Scripts\python.exe tests/test_environment.py

# Suite unitaria y de paridad C99 (cubre test_c_export_*.py y test_c_pipeline_parity.py)
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_c_*.py" -v

# Suite específica de visualización C99
.\.venv\Scripts\python.exe -m unittest tests/test_c_export_visualize.py -v

# Suite completa del repositorio
.\.venv\Scripts\python.exe -m unittest discover tests -q

# Pruebas y scripts exploratorios de Lab 15 (aritmética Q15, LUTs Shannon, BN folding, recuantización)
.\.venv\Scripts\python.exe -m unittest discover understanding/15_c_migration_fixed_point/tests -v
.\.venv\Scripts\python.exe understanding/15_c_migration_fixed_point/01_q_formats.py --config configs/understanding/c_migration_fixed_point.toml
.\.venv\Scripts\python.exe understanding/15_c_migration_fixed_point/02_entropy_fixed_point.py --config configs/understanding/c_migration_fixed_point.toml
.\.venv\Scripts\python.exe understanding/15_c_migration_fixed_point/03_bn_folding.py --config configs/understanding/c_migration_fixed_point.toml
.\.venv\Scripts\python.exe understanding/15_c_migration_fixed_point/04_requantization.py --config configs/understanding/c_migration_fixed_point.toml
```

### 5. Salidas generadas y limpieza

- **Código C99 generado**: `c_imp/generated/<target>/` (~37 MB si se exportan las 7 CNNs completas, debido principalmente a `model_torchvision_mobilenet.c` ~30 MB).
- **Binarios compilados**: `c_imp/build/<target>/cimp_pipeline.exe` y `c_imp/build/_runtime_tests/cimp_test_runtime.exe`.
- **Reportes y vectores**: `results/c_imp/<target>/` (`memory_report.json`, `parity_report.json`, `timing_benchmark.csv`, `vectors/`, `c_outputs/`). Ver la auditoría consolidada de resultados `f32`/`Q15`, presupuestos ESP32-CAM y hallazgos diagnósticos en [`../results/c_imp/README.md`](../results/c_imp/README.md).
- **Figuras diagnósticas y resumen legible**: `results/c_imp/<target>/figures/` (`fig1_classification_f32_vs_q15.png` .. `fig6_x94_descriptor_insights.png`, `summary_models_metrics.csv`, `summary_report.md`; configurables en `[c_imp.visualization]`).

Para limpiar los artefactos generados en PowerShell:
```powershell
foreach ($d in @("c_imp/generated", "c_imp/build", "results/c_imp", "understanding/15_c_migration_fixed_point/outputs")) {
  if (Test-Path $d) { Remove-Item -Recurse -Force $d }
}
```
