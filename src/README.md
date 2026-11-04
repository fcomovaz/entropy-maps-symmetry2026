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
  - `build_quantization_lut(bins)`: precomputación de Look-Up Table de 256 bytes para cuantización directa $O(1)$.
  - `build_shannon_term_lut(n_pixels)`: precomputación y caché LRU de términos de Shannon $-p \log_2(p)$ para evitar llamadas trascendentales.
  - `build_entropy_plan(image_size, grid_levels, bins)`: plan inmutable (`EntropyPlan`) que congela geometrías y tablas para ejecución en cero recálculos.
  - `region_edges(size, g)`: cálculo proporcional de límites $\lfloor k \cdot \text{size} / g \rfloor$ en división entera C-order.
  - `quantize_pixels(images, bins, lut=None)`: cuantización acelerada por LUT vectorizada para lotes.
  - `region_histograms(quantized, g, bins)`: histogramas discretos por región mediante truco de offset vectorizado sobre `np.bincount` en `int32`.
  - `normalized_entropy(counts, bins, term_lut=None)`: entropía de Shannon normalizada $H / \log_2(B)$, acelerada por LUT con valor analítico $1.0$ exacto para distribuciones equiprobables.
  - `compute_entropy_features(images, grid_levels, bins, plan=None)`: extracción de vector concatenado $\sum g^2$ (85 características para escalas $[1, 2, 4, 8]$).
  - `feature_names(grid_levels, template)`: generación determinista de nombres de columna (ej. `E_1x1_r0_c0` ... `E_8x8_r7_c7`).
- `entropy_csv.py`:
  - `load_entropy_config(config_path, project_root)`: carga y validación inmutable (`EntropyCsvConfig`) anclada dinámicamente al raíz del proyecto.
  - `validate_header(header, cfg)`: validación estricta de metadatos, dimensiones y contigüidad de columnas de píxeles.
  - `export_entropy_csv(cfg, dry_run=False)`: escritor streaming en bloques (`chunk_size`), reemplazo atómico seguro (`.tmp` -> `.csv`) y emisión de sidecar JSON con metadata de reproducibilidad y tiempo de extracción.

---

## Principios de diseño

- Sin rutas, parámetros ni vocabularios hardcodeados: toda configuración externa reside en `configs/*.toml`.
- Tipado estricto (`from __future__ import annotations`), dataclasses inmutables (`@dataclass(frozen=True, slots=True)`).
- Logging estructurado centralizado (`logging.getLogger("research")`); sin `print`.
- Pruebas unitarias asociadas en `tests/`.
