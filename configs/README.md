# Configuraciones declarativas (`configs/`)

Directorio único para parámetros externos, hiperparámetros, rutas de datasets y protocolos experimentales en formato **TOML**. De acuerdo con [AGENTS.md](../AGENTS.md), ningún módulo bajo `src/` hardcodea parámetros o rutas: toda configuración se lee desde aquí y se ancla dinámicamente al raíz `code/` mediante `pathlib.Path`.

---

## Núcleo común y logging

| Archivo | Rol |
|---|---|
| [`base.toml`](base.toml) | Rutas maestras de `mvtecad` y `kolektorsdd2` (`raw/`, `preprocessed/`, `csv/`, `-entropy`, `-entropy-full`, normal e invertido), lista de las 15 categorías de MVTec-AD y splits oficiales |
| [`logging.py`](logging.py) | Configuración compartida `configure_logging()` (`logger = logging.getLogger("research")`, UTC) hacia consola y [`logs/research.log`](logs/research.log) |

---

## Subdirectorios por etapa

| Subcarpeta | Archivos | Propósito | Consumidor principal |
|---|---|---|---|
| `preprocessing/` | `mvtec_ad.toml`, `kolektorsdd2.toml` | Tamaño objetivo (`128`), `pad_value = 0`, remuestreo (`LANCZOS`/`NEAREST`), emparejamiento `_GT` y `--inverted` | `experiments.preprocess_mvtec_ad`, `experiments.preprocess_kolektorsdd2` |
| `csv/` | `mvtec_ad.toml`, `kolektorsdd2.toml` | Exportación tabular row-major (`px_00000..px_16383`, metadatos y dimensiones) | `experiments.export_mvtec_ad_csv`, `experiments.export_kolektorsdd2_csv` |
| `entropy/` | `mvtec_ad.toml`, `kolektorsdd2.toml` | Grillas (`[1, 2, 4, 8]`), contenedores (`bins = 16`), `chunk_size`, sidecar JSON y `backend = "auto"` (`x85`) | `experiments.export_entropy_maps` |
| `features/` | `native_entropy.toml` | Compilación MSVC x64 (`/O2 /LD /arch:AVX2 /fp:precise`) de `src/features/native/entropy_pyramid.c` hacia `src/features/native/build/` | `src.features.build_native`, `src.features.native_entropy` |
| `asymmetry/` | `mvtec_ad.toml`, `kolektorsdd2.toml` | Escalas (`[2, 4, 8]`), simetrías (`lr`, `tb`, `rot180`) y contrato `schema x94-v1` ($\mathbf{x}_{85} \to \mathbf{x}_{94}$) | `experiments.append_asymmetry_features` |
| `classification/` | `mvtec_ad.toml`, `kolektorsdd2.toml` | Partición (`stratified` 80/20 vs `presplit`), subconjuntos de ablación, modelos (`dt`, `knn`, `lr`, `lr_dot`, `mlp_16_dot`, `rf`), explicabilidad y salida | `experiments.evaluate_descriptors` |
| `cnn/` | `mvtec_ad.toml`, `kolektorsdd2.toml`, `colab.toml` | 7 arquitecturas, `AdamW`, `cosine`, 40 épocas, calibración `val_f1` y variante `colab.toml` (`device = "cuda"`, `seeds = [42, 43, 44]`) | `experiments.evaluate_cnn`, [`notebooks/Entropy_Maps_Full_Run.ipynb`](../notebooks/Entropy_Maps_Full_Run.ipynb) |
| `profiling/` | `mvtec_ad.toml`, `kolektorsdd2.toml` | Protocolo 100 % CPU justo (`threads_list`, `batch_sizes`, warmup/repeticiones, bootstrap IC95%, Mann-Whitney U) y referencia CUDA aislada | `experiments.benchmark_cpu_fair`, `experiments.profile_pipelines` |
| `c_imp/` | `mvtec_ad.toml`, `kolektorsdd2.toml` | Presupuestos MCU ESP32-CAM (`flash_budget_bytes`, `sram_budget_bytes`, `psram_budget_bytes`), LUT Shannon `log2_mantissa`, punto fijo `Q15`/`int8_per_channel`, tolerancias de paridad `f32` y renderizado `[c_imp.visualization]` | `experiments.export_c_pipeline`, `experiments.verify_c_pipeline`, `experiments.visualize_c_pipeline` |
| `optimization/` | `entropy_maps.toml` | Configuración de Lab 13 (`V0–V3`, `bins_list`, `image_sizes`, compilación MSVC local) | [`understanding/13_entropy_maps_optimization/`](../understanding/13_entropy_maps_optimization/README.md) |
| `understanding/` | `ml_classifiers_mcu.toml`, `c_migration_fixed_point.toml` | Configuraciones de Lab 14 (clasificadores ML plegados y kernels C99) y Lab 15 (aritmética `Qm.n`, LUTs Shannon, BN folding y recuantización `int64`) | [`understanding/14_ml_classifiers_mcu/`](../understanding/14_ml_classifiers_mcu/README.md), [`understanding/15_c_migration_fixed_point/`](../understanding/15_c_migration_fixed_point/README.md) |
