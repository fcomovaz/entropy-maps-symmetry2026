# Laboratorio de comprensión

Este espacio existe para **comprender antes que optimizar** (`understanding > optimization`). Contiene implementaciones explícitas, visualizaciones, inspección de valores intermedios, formulaciones matemáticas directas y pruebas unitarias de comportamiento. 

No es código de producción: una implementación migra a `src/` únicamente cuando sus propiedades están matemáticamente entendidas, empíricamente caracterizadas y son requeridas por el pipeline formal.

---

## Estado general: ¿En qué fase estamos?

El proyecto ha completado la exploración por componentes, la experimentación formal en datasets reales, la optimización del descriptor en Python/NumPy puro y C99 nativo, y la migración a C99 portable `f32`/`Q15` verificada en escritorio, preparándose para la medición final en microcontrolador físico (ESP32-CAM):

```
[ Fase 1-5: Primitivas, CNNs, x94, Datasets, Profiling ] -> [ Fase 6: Opt. CPU / ML MCU ] -> [ Fase 7: Migración C99 f32/Q15 ]
              (01..12 + experiments/)                          (13, 14 + src/)                 (15 + src/c_export/ + c_imp/)
                    COMPLETADA                                    COMPLETADA                     COMPLETADA (Desktop Pre-MCU)
```

1. **Fase 1 — Primitivas matemáticas y computacionales (Completada):**
   - Geometría de patches, bordes DROP/PARTIAL/PAD y reconstrucción exacta ([01](01_patch_partition/README.md)).
   - Conteo directo de histogramas 256-bin como referencia $\Omega(N)$ ([02](02_histogram_benchmark/README.md)).
   - Equivalencia numérica de Shannon E1–E5, cuantización de tablas LUT y fidelidad de rankings ([03](03_entropy_benchmark/README.md)).
2. **Fase 2 — Modelos de referencia neuronal (Completada):**
   - Siete arquitecturas CNN ligeras (1.1k a 1.5M parámetros), perfiles de memoria y latencia CPU/CUDA aislada ([04](04_lightweight_cnns/README.md)).
3. **Fase 3 — Extractor manual de características x94 (Completada):**
   - 9 residuos de asimetría espacial de distribución de entropía (LR, TB, R180) ([05](05_symmetry_features/README.md)).
   - Extractor canónico multiescala `image -> x94` bajo esquema congelado `x94-v1` (85 entropías + 9 asimetrías) ([06](06_entropy_pyramid_descriptor/README.md)).
   - Caracterización científica del comportamiento de `x94` ante 15 perturbaciones controladas: ruido, desenfoque, contraste, defectos locales y colisiones estructurales ([07](07_descriptor_behavior/README.md)).
4. **Fase 4 — Validación y control de datasets reales (Completada):**
   - MVTec-AD estandarizado a 128x128 en grises e invertido ([08](08_verify_mvtec_preprocessing/README.md)), y su serialización tabular de 16391 columnas con reconstrucción exacta ([09](09_verify_mvtec_csv/README.md)).
   - KolektorSDD2 preprocesado mediante clasificación por máscara nativa a 128x128 ([10](10_verify_kolektorsdd2/README.md)), y su serialización tabular de 16389 columnas con reconstrucción exacta ([11](11_verify_kolektorsdd2_csv/README.md)).
5. **Fase 5 — Experimentación formal y profiling en CPU (Completada):**
   - Clasificación supervisada ligera (`dt`, `knn`, `lr`, `lr_dot`, `rf`), ablaciones $\mathbf{x}_{85}$ vs $\mathbf{x}_9$ vs $\mathbf{x}_{94}$ y comparación 100 % CPU justa frente a las 7 CNNs ligeras en `experiments/`.
   - Descomposición granular por etapas de $\mathbf{x}_{94}$ y capas PyTorch en [12](12_profiling_stages/README.md), identificando el cuello de botella de las 85 rebanadas Python/NumPy y el wrapper de `scikit-learn`.
6. **Fase 6 — Optimización algorítmica y clasificadores ML embebibles (Completada):**
   - Validación matemática y experimental de paso único en grilla más fina con pirámide aditiva en NumPy puro (`numpy_finest_pyramid`) y kernel C99 portable (`c_single_pass`), más el núcleo lineal plegado `lr_dot` ([13](13_entropy_maps_optimization/README.md)), promovidos al pipeline formal `src/`.
   - Comparativa exploratoria (sin modificar `src/`) de 9 clasificadores ML portables a C99 frente a `lr_dot`, `dt` y `rf` en [14](14_ml_classifiers_mcu/README.md), promoviendo `mlp_16_dot` (`MlpDotCoreClassifier`) a `src/models/mlp_dot.py`.
7. **Fase 7 — Aritmética fija `Q15`, LUTs compactas y migración C99 portable pre-ESP32 (Completada en escritorio):**
   - Estudio previo de formatos `Qm.n`, modos de LUT Shannon (`nlogn_full` ~64 KB, `term_per_level` ~44 KB, `log2_mantissa` 514 B), plegado `Conv2d + BatchNorm2d` y acumuladores `int64_t` en [15](15_c_migration_fixed_point/README.md), materializado en `src/c_export/` y `c_imp/` (resultados auditados en [../results/c_imp/README.md](../results/c_imp/README.md)).

---

## Mapa de los laboratorios

Todos los laboratorios residen en carpetas hermanas dentro de `code/understanding/`:

| Laboratorio | Enfoque central | Entrada / Datos | Salida principal |
|---|---|---|---|
| [01_patch_partition/](01_patch_partition/README.md) | Geometría de recorte y reconstrucción | Imagen de prueba en `assets/` | Rejillas visuales y `outputs/run_summary.json` |
| [02_histogram_benchmark/](02_histogram_benchmark/README.md) | 8 métodos de conteo de 256 niveles | Patches generados por 01 | Medidas de coste en `results/summary_table.md` |
| [03_entropy_benchmark/](03_entropy_benchmark/README.md) | Shannon E1–E5, LUTs y cuantización | Patches generados por 01 | `results/summary.md` y `validation.json` |
| [04_lightweight_cnns/](04_lightweight_cnns/README.md) | 7 arquitecturas CNN para visión embebida | Sintético / fixtures de prueba | Checkpoints, `validation.md` y benchmarks |
| [05_symmetry_features/](05_symmetry_features/README.md) | Residuos espaciales de asimetría | Reutiliza E1 de 03; imágenes sintéticas | Formulaciones acotadas, matrices 2x2/4x4/8x8 |
| [06_entropy_pyramid_descriptor/](06_entropy_pyramid_descriptor/README.md) | Extractor canónico `image -> x94` | Sintético / imágenes arbitrarias | API `extract_descriptor`, schema `x94-v1` |
| [07_descriptor_behavior/](07_descriptor_behavior/README.md) | Pruebas unitarias científicas sobre `x94` | Reutiliza 06; escenas analíticas | 279 filas medidas, curvas de respuesta y CSVs |
| [08_verify_mvtec_preprocessing/](08_verify_mvtec_preprocessing/README.md) | Control de integridad de MVTec-AD | `data/preprocessed/mvtec-ad(-inverted)` | Métricas MB y `preprocessing_verification.png` |
| [09_verify_mvtec_csv/](09_verify_mvtec_csv/README.md) | Control de integridad de CSV MVTec-AD | `data/csv/mvtec-ad(-inverted).csv` | Validación 16391 cols y `csv_roundtrip.png` |
| [10_verify_kolektorsdd2/](10_verify_kolektorsdd2/README.md) | Verificación e integridad KolektorSDD2 | `data/preprocessed/KolektorSDD2(-inverted)` | Conteos good/defect y `kolektorsdd2_verification.png` |
| [11_verify_kolektorsdd2_csv/](11_verify_kolektorsdd2_csv/README.md) | Control de integridad CSV KolektorSDD2 | `data/csv/kolektorsdd2(-inverted).csv` | Validación 16389 cols y `csv_roundtrip.png` |
| [12_profiling_stages/](12_profiling_stages/README.md) | Profiling manual por etapa ($\mathbf{x}_{94}$) y capa (`ds_cnn`) en CPU | Sintético o PNG `128x128` | Desglose `perf_counter_ns`, `cProfile`, `torch.profiler` y figura apilada |
| [13_entropy_maps_optimization/](13_entropy_maps_optimization/README.md) | Paso único en grilla más fina, pirámide aditiva NumPy (`V1/V2`) y kernel C99 (`V3`) + `lr_dot` | `real:mvtec-ad` (`n=200`, `128x128`) | 13 tests de paridad ($\le 1.89\times10^{-15}$), `benchmark_*.csv` y 7+1 figuras |
| [14_ml_classifiers_mcu/](14_ml_classifiers_mcu/README.md) | Clasificadores ML ligeros plegados y kernels C99 embebibles (`0 malloc`, MLPs, L1, LDA, SVC, GNB cuad., GBDT) sin tocar `src/` | Particiones canónicas `seed=42` (`kolektorsdd2`, `bottle`, `tile`, normal/inv. + 15 objetos MVTec-AD) | 6 tests de paridad/Flash/C99, CSVs comparativos, `sample_weights_mvtec_bottle.h` y 4 figuras |
| [15_c_migration_fixed_point/](15_c_migration_fixed_point/README.md) | Comprensión pre-C de formatos `Qm.n`, modos de LUT Shannon en entero, plegado Conv+BN y recuantización | Config TOML + muestras `128x128` (`KolektorSDD2` o sintético) | 4 scripts interactivos, tablas de error `Qm.n`, figuras PNG y 4 tests unitarios |

Además existe [`07_entropy_maps_vectorized/explain_entropy_maps.py`](07_entropy_maps_vectorized/explain_entropy_maps.py) como guion explicativo visual de la vectorización de mapas $\mathbf{x}_{85}$.

---

## Ruta metodológica de comprensión

```
Píxeles crudos
     │
     ├──> [01] Geometría de partición ──> [02] Histogramas ──> [03] Entropía Shannon
     │                                                               │
     │                                                               ├──> [05] Asimetrías espaciales
     │                                                               │           │
     │                                                               └──> [06] Descriptor canónico x94
     │                                                                           │
     ├──> [04] Modelos CNN de referencia (GTX 1050)                              └──> [07] Comportamiento x94
     │                                                                                       │
     ├──> [08] MVTec-AD estandarizado (128x128, grises, letterbox) <────────────────────────┤
     │           │                                                                           │
     │           └───> [09] MVTec-AD serializado en CSV (16391 cols, row-major)              │
     │                                                                                       │
     └──> [10] KolektorSDD2 (clasificación por máscara nativa, 128x128, grises) <────────────┘
                 │
                 └───> [11] KolektorSDD2 serializado en CSV (16389 cols, row-major)
                             │
                             └───> [12] Profiling por etapas  ──>  [13] Optimización single-pass (Py + C99)  ──>  [14] Clasificadores MCU  ──>  [15] Punto fijo Q15 / LUTs
                                         │                                  │                                   │                             │
                                         └──────────────────────────────> Pipeline formal en src/, experiments/ y c_imp/ <──────────────────────────┘
```

---

## Qué hemos entendido y validado empíricamente

- **Geometría y reconstrucción (01):** las políticas DROP, PARTIAL y PAD imponen compromisos estrictos entre cobertura completa y distorsión por padding artificial. La reconstrucción píxel a píxel está demostrada.
- **Histogramas (02):** el conteo por indexación directa es óptimo ($\Theta(N + 256)$). Diferencias temporales en Python reflejan sobrecostes del runtime y no ventajas algorítmicas trasladables a C embebido.
- **Entropía de Shannon (03):** E1 define la referencia canónica $- \sum p \log_2(p)$. Las tablas LUT ahorran tiempo pero exigen memoria (evaluada por tamaño de patch). Cuantizaciones bajas (Q16) alteran el orden relativo de los parches (rankings top-k); Q30 preserva el orden sin discrepancias en los casos medidos.
- **Modelos neuronales (04):** parámetros y latencia son magnitudes desacopladas (modelos con menos parámetros pueden ser más lentos si carecen de núcleos optimizados).
- **Asimetrías espaciales (05):** los 9 residuos $A_{\text{LR}}, A_{\text{TB}}, A_{\text{R180}}$ capturan desbalances macroscópicos de distribución de entropía y satisfacen desigualdades triangulares estrictas. No añaden información matemática nueva sobre los 85 valores de entropía, pero hacen explícitas relaciones estructurales espaciales.
- **Descriptor canónico x94 (06):** vector de 94 floats acotados en $[0, 1]$ con esquema estricto e inmutable (`x94-v1`): nivel 1x1 (1), 2x2 (4), 4x4 (16), 8x8 (64) + 9 asimetrías. Extracción determinista y desacoplada de librerías externas.
- **Comportamiento y límites del descriptor (07):**
  - *Monotonía ante defectos:* la respuesta crece de manera monótona con el tamaño y contraste del defecto en escenas controladas, localizándose en las celdas afectadas.
  - *Efecto del ruido:* el ruido aditivo $\sigma$ eleva falsamente la entropía promedio ($0.661 \to 0.830$) y aplana las asimetrías ($0.0154 \to 0.0095$).
  - *Colisión estructural:* distribuciones espaciales distintas con idéntico histograma global arrojan **exactamente el mismo vector x94** (la entropía de Shannon ignora la disposición interna de píxeles dentro de la celda).
  - *Fronteras de celda:* el descriptor responde con máxima sensibilidad cuando el frente de un defecto cruza el borde de una celda de la rejilla.
- **Estandarización de datos industriales (08):**
  - MVTec-AD requiere letterboxing centrado para preservar la relación de aspecto original sin distorsiones geométricas.
  - Las máscaras de defectos deben interpolarse con `NEAREST` para no introducir niveles espurios fuera de $\{0, 255\}$ y permanecer perfectamente alineadas con las imágenes.
  - La inversión de intensidad (`255 - x`) debe aplicarse antes del letterbox para que el padding conserve el valor $0$ (negro).
- **Serialización tabular CSV (09):**
  - Exportación en streaming atómica (`.tmp` -> final) de 16391 columnas con reconstrucción exacta píxel a píxel (diferencia absoluta 0).
- **Preprocesamiento y clasificación de KolektorSDD2 (10):**
  - Clasificación binaria (`good`/`defect`) mediante inspección de la máscara binaria en su resolución nativa (`mask.max() > 0`) antes del reescalado para evitar atenuar defectos microscópicos.
  - Letterbox centrado a 128x128 preservando el aspecto vertical (~1:2.8).
  - Exclusión total de archivos de máscara en el directorio procesado final.
- **Serialización tabular CSV de KolektorSDD2 (11):**
  - Exportación en streaming atómica (`.tmp` -> final) de 16389 columnas (`split`, `type`, `file_name`, 16384 intensidades de píxel en row-major, `width`, `height`).
  - Total de 3335 muestras balanceadas por split (`train`: 2085 good, 246 defect; `test`: 894 good, 110 defect) y reconstrucción exacta píxel a píxel (diferencia absoluta 0).
- **Profiling granular en CPU (12):**
  - En la implementación baseline, la extracción de $\mathbf{x}_{94}$ invertía la mayor parte del tiempo en 85 iteraciones Python (`slice` + `np.bincount`, con `8x8` dominando), el *gap* de despacho inter-etapa y el wrapper `Pipeline.predict` de `scikit-learn` (`StandardScaler` + validación), ocultando el costo real de un producto punto de 94 dimensiones.
- **Optimización single-pass y núcleo C99 pre-MCU (13):**
  - Al calcular histogramas únicamente sobre la grilla más fina ($8\times8$) en una sola lectura y obtener $4\times4$, $2\times2$ y $1\times1$ por suma exacta de bloques hijo, **NumPy puro (`numpy_finest_pyramid`)** acelera $\mathbf{x}_{85}$ en `9.40x` y el kernel **C99 (`c_single_pass`, sin `malloc` en batch=1)** alcanza `92.65x` (`0.024 ms` vs `2.23 ms`) manteniendo paridad numérica $\le 1.89\times10^{-15}$. El plegado algebraico de `StandardScaler` en un producto punto directo (`lr_dot`) elimina el overhead de `scikit-learn`.
- **Clasificadores ML alternativos portables a C99 sin modificar `src/` (14):**
  - Plegar `StandardScaler` en MLPs ReLU (`6.0–14.0 KB` Flash, `64–192 B` SRAM, `0 <math.h>` en umbral) eleva el F1 medio en los 15 objetos MVTec-AD a `0.6905` (`mlp_16_skip_dot`) frente a `0.6486` (`lr_dot`) y `0.5178` (`rf`), y en `kolektorsdd2` a `F1=0.5000` / `PR-AUC=0.5936` frente a `lr_dot` (`0.4265` / `0.4360`). Todos los núcleos C99 logran `100%` de acuerdo de etiqueta y corren en `0.019–4.02 us` (`< 1.5%` de `T_desc`). `lr_l1_dot` recorta los pesos activos a `41.1/94` en promedio (`0.32 KB`), y `gnb_quad` revela cancelación de ULP `float32` cuando el offset log-ratio alcanza `~1.1e4` pese a conservar `100%` de acuerdo binario.

- **Aritmética fija Q15, LUTs Shannon y BN-folding pre-ESP32 (15):**
  - La cuantización `Q0.15` satura `1.0` en `32767` con error `< 5e-5`; la LUT `log2_mantissa` (`257` entradas, `~514 B` Flash) reduce >100× la memoria frente a `nlogn_full` (`~64 KB`) manteniendo residuales medios `~7.6e-6` frente a `float64`. El plegado analítico `Conv2d + BatchNorm2d` logra equivalencia `< 1e-6`, y los acumuladores `int64_t` evitan desbordamientos de `int32_t` cuando el *fan-in* `Cin*k^2` crece.

---

## Qué está verificado y qué sigue abierto hacia el paper y el MCU

1. **Capacidad predictiva real y ablación (`Completado` — ver [`../HYPOTHESIS.md`](../HYPOTHESIS.md)):** evaluado en MVTec-AD y KolektorSDD2 con clasificadores ligeros (`dt`, `knn`, `lr`, `lr_dot`, `rf`) frente a 7 CNNs ligeras. Confirmado el aporte de $\mathbf{x}_9$ sobre $\mathbf{x}_{85}$ y la invarianza fotométrica en imágenes sin padding; refutada la superioridad universal frente a CNNs en micro-fisuras sub-celda (`KolektorSDD2`).
2. **Evaluación justa de coste total ($T_{\text{total}}$) en CPU (`Completado`):** medido extremo a extremo (`T_descriptor + T_model`) en Python puro (`numpy_baseline`, `numpy_finest_pyramid`) y con entropía en C99 (`c_single_pass`).
3. **Migración completa a C99 portable `f32`/`Q15` en escritorio (`Completado` — ver [`../results/c_imp/README.md`](../results/c_imp/README.md)):** portados `x94` completo (incluyendo `asymmetry_g2/4/8`), `lr_dot`, `mlp_16_dot` y las 7 CNNs en `c_imp/` con cero `malloc` y paridad `Q15` bit-exacta (`0` LSB de diferencia C vs Python). Queda pendiente la **Etapa 12**: medición de ciclos, latencia Flash vs PSRAM y consumo energético en **hardware físico ESP32-CAM**.
4. **Ablación de grilla fina ($16\times16$) y ruteo adaptativo por entropía:** estudiar si extender la pirámide a `g=16` (`8x8` px por celda) o usar $\mathbf{x}_{85}$ como selector rápido de parches (`top-k / threshold`) antes de una CNN reduce falsos negativos en fisuras de 1–2 px.

---

## Reproducibilidad y ejecución

Los scripts de comprensión usan rutas relativas al proyecto y fijan semillas reproducibles (20260928 en 02/03; 20261001 en 05/06/07; 42 en 04, 12, 13, 14 y 15).

Para ejecutar los laboratorios individuales o sus tests manuales (PowerShell desde carpeta `code/`):
```powershell
# Pruebas unitarias de componentes (invocar Lab 07 con PYTHONPATH=code/)
.\.venv\Scripts\python.exe understanding/03_entropy_benchmark/test_entropy.py
.\.venv\Scripts\python.exe -m unittest understanding/04_lightweight_cnns/tests/test_models.py
.\.venv\Scripts\python.exe -m unittest understanding/05_symmetry_features/tests/test_symmetry_features.py
.\.venv\Scripts\python.exe -m unittest understanding/06_entropy_pyramid_descriptor/tests/test_descriptor.py
$env:PYTHONPATH = (Get-Location).Path; .\.venv\Scripts\python.exe -m unittest understanding/07_descriptor_behavior/tests/test_behavior_helpers.py
.\.venv\Scripts\python.exe -m unittest understanding/13_entropy_maps_optimization/tests/test_entropy_variants.py
.\.venv\Scripts\python.exe -m unittest understanding/14_ml_classifiers_mcu/tests/test_ml_classifiers_mcu.py
.\.venv\Scripts\python.exe -m unittest discover understanding/15_c_migration_fixed_point/tests -v

# Demos y benchmarks de comprensión
.\.venv\Scripts\python.exe understanding/06_entropy_pyramid_descriptor/run_demo.py
$env:PYTHONPATH = (Get-Location).Path; .\.venv\Scripts\python.exe understanding/07_descriptor_behavior/run_all.py
.\.venv\Scripts\python.exe understanding/07_entropy_maps_vectorized/explain_entropy_maps.py

# Verificación de preprocesamiento y exportación CSV
.\.venv\Scripts\python.exe understanding/08_verify_mvtec_preprocessing/verify.py
.\.venv\Scripts\python.exe understanding/08_verify_mvtec_preprocessing/verify.py --inverted
.\.venv\Scripts\python.exe understanding/09_verify_mvtec_csv/verify.py
.\.venv\Scripts\python.exe understanding/09_verify_mvtec_csv/verify.py --inverted
.\.venv\Scripts\python.exe understanding/10_verify_kolektorsdd2/verify.py
.\.venv\Scripts\python.exe understanding/10_verify_kolektorsdd2/verify.py --inverted
.\.venv\Scripts\python.exe understanding/11_verify_kolektorsdd2_csv/verify.py
.\.venv\Scripts\python.exe understanding/11_verify_kolektorsdd2_csv/verify.py --inverted

# Lab 12: Profiling manual paso a paso en CPU (x94 + ds_cnn)
.\.venv\Scripts\python.exe understanding/12_profiling_stages/explain_profiling_stages.py
.\.venv\Scripts\python.exe understanding/12_profiling_stages/explain_profiling_stages.py --image data/preprocessed/KolektorSDD2/test/defect/10002.png

# Lab 13: Compilación C99, explicación single-pass, benchmark CPU-justo y gráficas
.\.venv\Scripts\python.exe understanding/13_entropy_maps_optimization/build_native.py
.\.venv\Scripts\python.exe understanding/13_entropy_maps_optimization/explain_single_pass.py
.\.venv\Scripts\python.exe understanding/13_entropy_maps_optimization/benchmark_entropy_variants.py --config configs/optimization/entropy_maps.toml
.\.venv\Scripts\python.exe understanding/13_entropy_maps_optimization/plot_entropy_optimization.py --config configs/optimization/entropy_maps.toml

# Lab 14: Tests, benchmark exploratorio de clasificadores ML portables a C99 (sin tocar src/) y figuras
.\.venv\Scripts\python.exe -m unittest understanding/14_ml_classifiers_mcu/tests/test_ml_classifiers_mcu.py -v
.\.venv\Scripts\python.exe understanding/14_ml_classifiers_mcu/benchmark_classifiers.py --config configs/understanding/ml_classifiers_mcu.toml --include-inverted
.\.venv\Scripts\python.exe understanding/14_ml_classifiers_mcu/plot_results.py --config configs/understanding/ml_classifiers_mcu.toml

# Lab 15: Tests y estudio de punto fijo Q15, LUTs Shannon, BN folding y recuantización int64
.\.venv\Scripts\python.exe -m unittest discover understanding/15_c_migration_fixed_point/tests -v
.\.venv\Scripts\python.exe understanding/15_c_migration_fixed_point/01_q_formats.py --config configs/understanding/c_migration_fixed_point.toml
.\.venv\Scripts\python.exe understanding/15_c_migration_fixed_point/02_entropy_fixed_point.py --config configs/understanding/c_migration_fixed_point.toml
.\.venv\Scripts\python.exe understanding/15_c_migration_fixed_point/03_bn_folding.py --config configs/understanding/c_migration_fixed_point.toml
.\.venv\Scripts\python.exe understanding/15_c_migration_fixed_point/04_requantization.py --config configs/understanding/c_migration_fixed_point.toml
```
