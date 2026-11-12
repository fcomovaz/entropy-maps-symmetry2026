# Pipeline formal (`src/`)

Módulos modulares, reutilizables y testeables del pipeline formal del proyecto.

---

## Módulos implementados

### 1. `src/preprocessing/`

Preprocesamiento geométrico y fotométrico de imágenes:

- `letterbox.py`:
  - `letterbox_to_square(image, size, pad_value=0, interpolation=LANCZOS)`: escala proporcional preservando aspecto con padding simétrico centrado.
  - `to_grayscale(image)`: conversión estandarizada a un solo canal uint8 (luma ITU-R 601).
  - `invert_intensity(image)`: inversión lineal de luminancia (`255 - x`).
- `mvtec_ad.py`:
  - `load_config(config_path, project_root)`: lectura y validación estricta de parámetros TOML.
  - `run(cfg, dry_run=False)`: recorrido batch de MVTec-AD con skip de archivos existentes, interpolación `LANCZOS` para imágenes, `NEAREST` para máscaras de defectos y soporte para variante invertida (`mvtec-ad-inverted`).
- `kolektorsdd2.py`:
  - `load_config(config_path, project_root)`: carga y validación estricta de parámetros TOML para KolektorSDD2.
  - `is_defective(mask, threshold=0)`: detección binaria de defectos sobre máscara a resolución nativa (`mask.max() > threshold`).
  - `discover_pairs(split_dir, cfg)` / `iter_pairs(split_dir, cfg)`: emparejamiento determinista de imágenes y máscaras `_GT` con filtrado de stems inválidos y huérfanos.
  - `process_pair(pair, split, cfg, dry_run=False)`: clasificación a `good`/`defect`, conversión a grises, letterboxing centrado 128x128 e inversión opcional.
  - `run(cfg, dry_run=False)`: procesamiento batch por particiones con conteo de métricas y exclusión de máscaras en la salida.

### 2. `src/datasets/`

Serialización, acceso y manipulación de datos tabulares y colecciones:

- `mvtec_ad_csv.py`:
  - `load_csv_config(config_path, project_root)`: carga y validación de configuración de exportación TOML.
  - `classify_type(defect_name, cfg)`: asignación determinista de clase binaria (`good` vs. `defect`).
  - `iter_records(cfg)`: descubrimiento estructurado y ordenado de rutas y procedencia.
  - `load_pixels(path, image_size)`: lectura defensiva uint8 validando modo (`'L'`) y resolución (`image_size x image_size`).
  - `record_to_row(record, pixels)`: aplanado row-major (C-order) con metadatos y dimensiones.
  - `export_csv(cfg, dry_run=False)`: escritor streaming mediante archivo temporal atómico (`<dest>.tmp` -> `<dest>`) con reporte de métricas (16391 columnas).
- `kolektorsdd2_csv.py`:
  - `load_kolektor_csv_config(config_path, project_root)`: carga y validación de configuración de exportación TOML para KolektorSDD2.
  - `classify_type(class_folder, cfg)`: mapeo determinista a clase binaria (`good` vs. `defect`).
  - `iter_records(cfg)`: descubrimiento determinista ordenado por partición (`train`, `test`), clase y nombre de archivo.
  - `build_header(cfg)`: construcción del esquema de 16389 columnas (`split, type, file_name, px_00000...px_16383, width, height`).
  - `record_to_row(record, pixels)`: ensamblado de fila plana en orden row-major.
  - `export_kolektor_csv(cfg, dry_run=False)`: escritor en streaming con archivo temporal atómico (`.tmp` -> `<dest>`) y métricas de ejecución.

### 3. `src/features/`

Extracción modular y vectorizada de características visuales y descriptores de entropía:

- `entropy_maps.py`:
  - `validate_grid_levels(levels, image_size)`: validación defensiva de escalas $g$ sin duplicados ni sobrepaso de tamaño.
  - `validate_bins(bins)`: comprobación estricta de contenedores de histograma $B \in [2, 256]$.
  - `build_quantization_lut(bins)` y `build_shannon_term_lut(n_pixels)` / `build_nlogn_lut(n_pixels)`: precomputación O(1) de cuantización y tablas LUT de Shannon ($-p\log_2 p$ y $n\log_2 n$).
  - `build_entropy_plan(image_size, grid_levels, bins, backend="auto")`: plan inmutable (`EntropyPlan`) que congela geometrías, buffers y tablas LUT para ejecución multiescala con cero recálculos.
  - `resolve_entropy_backend(backend, image_size, grid_levels, bins)`: despacho multi-backend (`"auto"`, `"c_single_pass"`, `"numpy_finest_pyramid"`, `"numpy_fused_entropy"`, `"numpy_baseline"`) con selección automática en pirámide anidada y validación *fail-fast* en modos explícitos.
  - `region_edges(size, g)`: cálculo proporcional de límites $\lfloor k \cdot \text{size} / g \rfloor$ en división entera C-order.
  - `quantize_pixels(images, bins, lut=None)`: cuantización acelerada por LUT vectorizada para lotes.
  - `region_histograms(quantized, g, bins)`: histogramas discretos por región mediante truco de offset vectorizado sobre `np.bincount` en `int32`.
  - `normalized_entropy(counts, bins, term_lut=None)`: entropía de Shannon normalizada $H / \log_2(B)$, acelerada por LUT con valor analítico $1.0$ exacto para distribuciones equiprobables.
  - `compute_entropy_features(images, grid_levels, bins, plan=None, backend=None)`: extracción de vector concatenado $\sum g^2$ (85 características para escalas $[1, 2, 4, 8]$) con paridad numérica $< 10^{-12}$ entre backends.
  - `feature_names(grid_levels, template)`: generación determinista de nombres de columna (ej. `E_1x1_r0_c0` ... `E_8x8_r7_c7`).
- `native_entropy.py` / `build_native.py` / `native/entropy_pyramid.{h,c}`:
  - Kernel nativo C99 compilado a `src/features/native/build/entropy_pyramid.dll` (configurado en `configs/features/native_entropy.toml`) y envoltura `ctypes` con buffers preasignados cero-asignaciones en caliente (`NativeEntropyPlan`, `compute_entropy_features_c`).
- `entropy_csv.py`:
  - `load_entropy_config(config_path, project_root)`: carga y validación inmutable (`EntropyCsvConfig`) anclada dinámicamente al raíz del proyecto (incluyendo `entropy_backend`).
  - `validate_header(header, cfg)`: validación estricta de metadatos, dimensiones y contigüidad de columnas de píxeles.
  - `export_entropy_csv(cfg, dry_run=False)`: escritor streaming en bloques (`chunk_size`), reemplazo atómico seguro (`.tmp` -> `.csv`) y emisión de sidecar JSON con metadata de reproducibilidad, `entropy_backend`, `resolved_backend` y tiempo de extracción.
- `asymmetry_maps.py`:
  - `validate_scales(scales, global_scale=1)`: validación estricta de escalas espaciales $g \ge 2$ sin duplicados y exclusión de escala global.
  - `validate_kinds(kinds)`: comprobación defensiva de medidas canónicas (`"lr"`, `"tb"`, `"rot180"`).
  - `asymmetry_column_names(scales, kinds, templates)`: formateo ordenado según contrato `schema x94-v1`.
  - `entropy_column_names(scale, template)`: nombres de columnas de entropía en orden row-major.
  - `reconstruct_grids(flat_values, g)`: reconstrucción vectorizada de matrices 2D a partir de filas de características planas.
  - `validate_entropy_values(values, tolerance=1e-6)`: verificación de números finitos en $[0, 1 + \epsilon]$ sin truncado ni clipping.
  - `compute_asymmetry_for_grid(grids, kinds, tolerance)`: cálculo vectorizado sobre NumPy de residuos medios absolutos LR, TB y R180.
  - `compute_asymmetry_features(entropy_by_scale, scales, kinds, tolerance)`: ensamblado concatenado determinista (9 características espaciales).
- `asymmetry_csv.py`:
  - `load_asymmetry_config(config_path, project_root)`: carga y validación inmutable (`AsymmetryCsvConfig`) anclada dinámicamente al raíz del proyecto.
  - `validate_header(header, cfg)`: validación de presencia de columnas de entropía de entrada y prevención estricta de colisión / doble anexado.
  - `append_asymmetry_csv(cfg, dry_run=False)`: procesamiento streaming por bloques con copia byte a byte de metadatos originales, reemplazo atómico seguro (`.tmp` -> `<nombre>-full.csv`) y actualización de sidecar JSON.

### 4. `src/models/`

Clasificación supervisada ligera, ablación de subconjuntos y explicabilidad espacial sobre descriptores $\mathbf{x}_{94}$:

- `descriptor_classifier.py`:
  - `load_classification_config(config_path, project_root)`: carga y validación inmutable (`ClassificationConfig`) anclada a `configs/base.toml` y `configs/asymmetry/*.toml`.
  - `resolve_feature_groups(header, cfg)`: validación y resolución determinista de subconjuntos (`x94_full`, `x85_only`, `x9_only`, `E_1x1..E_8x8`, `asym_2x2..asym_8x8`, `asym_lr`, `asym_tb`, `asym_rot180`).
  - `load_and_split_descriptors(cfg, object_name=None, subset_name=None)`: filtrado en memoria de particiones permitidas (excluyendo estrictamente `ground_truth`), selección de objeto en MVTec-AD y partición `presplit` (KolektorSDD2) o `stratified` reproducible en memoria (MVTec-AD) sin mutar los archivos CSV en disco.
  - `build_classifier(classifier_name, cfg, linear_mode=None)`: construcción de `RandomForestClassifier`, `DecisionTreeClassifier`, `Pipeline(StandardScaler + KNeighborsClassifier | LogisticRegression)` o núcleo lineal plegado `LinearDotCoreClassifier` (`lr_dot`, donde escala y sesgo del `StandardScaler` se fusionan algebraicamente en `w_eff @ x + b_eff`).
  - `train_and_evaluate(partition, classifier_name, cfg, subset_name=None)`: entrenamiento, evaluación de métricas (`Accuracy`, `Balanced Accuracy`, `Precision`, `Recall`, `F1`, `ROC-AUC`, `PR-AUC`, matriz de confusión) y separación de tiempos (`T_model_train`, `T_model_inference`, `ms/sample`).
  - `run_subset_ablation(cfg, object_name, classifier_name, base_partition=None)`: evaluación comparativa de subconjuntos sobre la misma partición exacta para cuantificar la ganancia de simetría $\Delta\text{F1}(\mathbf{x}_{94} - \mathbf{x}_{85})$ y la escala de entropía óptima.
  - `compute_explainability(model, partition, cfg)`: motor de explicabilidad (impureza Gini MDI para Random Forest; importancia por permutación en test o distancia discriminante de Fisher para k-NN) con desglose por familia ($\mathbf{x}_{85}$ vs $\mathbf{x}_9$), por escala ($1\times1, 2\times2, 4\times4, 8\times8$), por eje de simetría (`lr`, `tb`, `rot180`) y reconstrucción de mapas de calor espaciales 2D ($2\times2, 4\times4, 8\times8$).
  - `run_evaluation_pipeline(cfg, classifier_name=None, object_name=None, dry_run=False)`: orquestación completa y exportación de tablas CSV, resumen JSON y figuras diagnósticas.
- `classification_plots.py`:
  - `plot_confusion_and_curves(metrics, title_prefix, out_path, dpi)`: matriz de confusión (conteos y %), curva ROC y curva Precision-Recall.
  - `plot_subset_ablation(ablation_metrics, title_prefix, out_path, dpi)`: gráfico comparativo de `x85_only`, `x9_only`, `x94_full` y escalas individuales `E_1x1..E_8x8`.
  - `plot_explainability_breakdown(report, title_prefix, out_path, dpi)`: Top-$K$ características individuales, aporte porcentual total por escala y eficiencia media por característica normalizada por tamaño de grupo.
  - `plot_spatial_importance_heatmaps(report, title_prefix, out_path, dpi)`: mapas de calor espaciales 2D ($2\times2, 4\times4, 8\times8$) de importancia regional de entropía.
  - `plot_all_objects_summary(summary_rows, title_prefix, out_path, dpi)`: comparativa multi-objeto para las 15 categorías de MVTec-AD.

### 5. `src/models/cnn/`

Entrenamiento, evaluación y benchmarking de arquitecturas convolucionales ligeras:

- `architectures/`:
  - Zoo de modelos convolucionales ligeros para imágenes grayscale: `UltraTinyCNN` (1.1k params), `DepthwiseSeparableCNN` (12.8k params), `TinyCNN` (23.5k params), `MobileNetLikeCNN` (70.7k params), `TinyResidualCNN` (77.2k params), `SmallCNN` (97.6k params) y `TorchvisionMobileNet` (1.5M params).
  - `registry.py`: catálogo centralizado `create_model(name, num_classes)` y `model_names()`.
  - `common.py`: utilidades compartidas de semillas deterministas (`seed_everything`), selección de hardware (`select_device`) y conteo de parámetros (`model_stats`).
- `config.py`:
  - `load_cnn_config(config_path, project_root)`: carga y validación inmutable de parámetros de entrenamiento, datos, benchmarking y exportación en `CnnConfig`.
- `data.py`:
  - `load_cnn_partition(cfg, object_name=None, seed=None)`: cargador streaming en memoria directamente desde imágenes PNG preprocesadas en `data/preprocessed/`, asegurando paridad de splits e índices con el baseline clásico mediante ordenamiento indexado por metadatos.
  - `InMemCnnDataset`: dataset PyTorch con soporte para aumento de datos simétrico (horizontal flip).
- `train.py`:
  - `train_model(model, partition, cfg, seed, device)`: ciclo de entrenamiento con cross-entropy ponderado por clases desbalanceadas, optimizador AdamW, programación de tasa de aprendizaje por coseno y early stopping sobre PR-AUC de validación.
- `evaluate.py`:
  - `predict_proba(model, images, device)`: cómputo vectorizado de probabilidades con softmax bajo `torch.inference_mode()`.
  - `evaluate_cnn(model, partition, cfg, model_name, train_elapsed_s, device)`: cálculo de métricas estándar compatibles bit a bit con `ClassificationMetrics` y medición aislada de latencia de inferencia `T_inf`.
- `benchmark.py`:
  - `run_benchmark(model, model_name, cfg, device, batch_size, inputs=None, threads=None, warmup_iters=None, timed_iters=None, disable_gc=False)`: protocolo de medición de latencia con `perf_counter_ns()`, soporte retrocompatible para entrada aleatoria o tensores reales (`inputs`), control opcional de hilos/GC, sincronización GPU CUDA, cálculo de mediana/IQR/media/std, estimación de FLOPs/MACs (`FlopCounterMode`) y consumo pico de VRAM.
- `report.py`:
  - `save_metrics_summary(...)`, `save_training_history(...)`, `save_checkpoint(...)`, `plot_confusion_and_curves(...)` y `plot_training_curves(...)`.

### 6. `src/profiling/`

Módulos formales para benchmarking justo **100% CPU**, perfilado granular por etapas y capas, estadística no paramétrica y generación de reportes científicos:

- `config.py`:
  - `load_profiling_config(config_path, project_root)`: carga y validación estricta (*fail-fast*) de configuraciones `configs/profiling/*.toml`, exigiendo `device = "cpu"` como protocolo primario y resolviendo rutas relativas ancladas al raíz del proyecto.
  - `resolve_target_objects(...)` y `effective_variant_tag(...)`: resolución determinista de categorías MVTec-AD (`bottle`, `--objects`, `--all-objects`) y etiquetas canónicas normal/invertido.
- `timing.py`:
  - `disabled_gc(...)` y `thread_limits(threads)`: context managers que pauses Python GC y fijan simétricamente hilos PyTorch (`torch.set_num_threads`) y BLAS/OpenMP (`threadpoolctl`) garantizando restauración ante excepciones.
  - `measure_indexed_latencies(...)`, `measure_callable_latencies(...)`, `summarize_latencies(...)` (mediana, IQR, p95, IC95% por bootstrap, throughput), `summarize_multi_run(...)` (CV% inter-corrida), `compare_latency_distributions(...)` (test U de Mann-Whitney) y `measure_memory_alloc(...)` (`psutil` RSS + `tracemalloc`).
- `environment.py`:
  - `capture_environment(project_root, seed)` y `save_environment_json(...)`: captura reproducible de CPU, núcleos físicos/lógicos, frecuencia, plan de energía, RAM, SO, commit Git, backend de threadpools y versiones exactas de PyTorch/NumPy/SciPy/sklearn/psutil.
- `descriptor_stages.py`:
  - `extract_x94_end_to_end(...)` y `extract_x94_staged(...)`: replicación exacta etapa por etapa importando directamente las primitivas de `src/features/` sin modificarlas, validando equivalencia numérica `< 1e-12`.
  - `profile_descriptor_stages(...)`, `run_descriptor_cprofile(...)`, `estimate_descriptor_ops(...)` y `profile_descriptor_scalability(...)`: medición de plan/LUT inicial, latencia y memoria por etapa (`quantize_pixels`, `region_histograms`, `normalized_entropy`, ensamblado, `reconstruct_grids` + asimetrías, `StandardScaler`), cálculo de overhead de llamadas Python y barridos de escalabilidad (`image_size`, `bins`, `grid_levels`).
- `classifier_profile.py`:
  - `configure_classifier_threads(...)`, `count_classical_model_params(...)` (parámetros reales: nodos de árbol, suma de nodos RF, coeficientes LR, vectores kNN), `estimate_classical_inference_ops(...)` (estimación analítica separada de MACs neuronales), `profile_classifier(...)` (`predict`, `predict_proba`, `StandardScaler`, núcleo directo y overhead de despacho sklearn) y `profile_classifier_thread_scaling(...)`.
- `cnn_layers.py`:
  - `measure_cnn_preprocess_latency(...)`, `measure_checkpoint_cold_start(...)` (deserialización + primera inferencia vs estacionario), `profile_cnn_layers(...)` (hooks *pre/post* por módulo hoja + `torch.profiler` ATen + trazas Chrome + FLOPs/activaciones por capa + verificación de overhead de hooks) y `profile_cnn_thread_scaling(...)`.
- `report.py`:
  - Escritores CSV deterministas (`comparison_cpu.csv`, `summary_all_datasets.csv`, `descriptor_stage_times.csv`, `descriptor_scalability.csv`, `classifier_times.csv`, `cnn_layer_times_*.csv`, `cnn_aten_ops_*.csv`, `cnn_architectures_summary.csv`, `ablation_quality_vs_cost.csv`, `statistical_tests.csv`), frontera de Pareto (`compute_pareto_frontier_indices`), renderizado de 6 figuras científicas (`generate_profiling_figures`, `plot_cpu_comparison_pareto`) y generador del reporte `write_profiling_readme(...)`.

---

## Principios de diseño

- Sin rutas, parámetros ni vocabularios hardcodeados: toda configuración externa reside en `configs/*.toml`.
- Tipado estricto (`from __future__ import annotations`), dataclasses inmutables (`@dataclass(frozen=True, slots=True)`).
- Logging estructurado centralizado (`logging.getLogger("research")`); sin `print`.
- Pruebas unitarias asociadas en `tests/`.
