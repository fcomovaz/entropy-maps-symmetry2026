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
  - `export_csv(cfg, dry_run=False)`: escritor streaming mediante archivo temporal atómico (`<dest>.tmp` -> `<dest>`) con reporte de métricas.

---

## Principios de diseño

- Sin rutas, parámetros ni vocabularios hardcodeados: toda configuración externa reside en `configs/*.toml`.
- Tipado estricto (`from __future__ import annotations`), dataclasses inmutables (`@dataclass(frozen=True, slots=True)`).
- Logging estructurado centralizado (`logging.getLogger("research")`); sin `print`.
- Pruebas unitarias asociadas en `tests/`.
