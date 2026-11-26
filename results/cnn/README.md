# Checkpoints y Evaluación de CNNs Ligeras (`results/cnn/`)

Este directorio almacena los modelos convolucionales entrenados por `experiments.evaluate_cnn` (`src/models/cnn/`) sobre imágenes `128×128` en escala de grises procedentes de `data/preprocessed/`, respetando los mismos splits que la línea base clásica (`seed = 42` canónico).

Los archivos pesados (`.pt`, `.csv`, `.json`, `.png`) están excluidos de Git mediante `.gitignore`.

---

## Estructura de carpetas

```text
results/cnn/
├── kolektorsdd2/full/<modelo>/
├── kolektorsdd2-inverted/full/<modelo>/
├── mvtec-ad/<objeto>/<modelo>/              # las 15 categorías en disco (mvtec_all_objects_<modelo>_summary.csv)
└── mvtec-ad-inverted/<objeto>/<modelo>/     # las 15 categorías invertidas en disco
```

Modelos evaluados en el zoo (`<modelo>`):
`ultra_tiny` (`1,154` params), `ds_cnn` (`12,866`), `tiny_cnn` (`23,538`), `mobilenet_like` (`70,786`), `tiny_resnet` (`77,234`), `small_cnn` (`97,650`) y `torchvision_mobilenet` (`1,519,618`, inicialización aleatoria).

---

## Archivos dentro de cada `<modelo>/`

| Archivo | Contenido |
|---|---|
| `checkpoint.pt` | Pesos del mejor estado según PR-AUC de validación (`best_val_pr_auc`), metadatos del split y arquitectura; reutilizado directamente en CPU por `benchmark_cpu_fair.py` y `profile_pipelines.py` |
| `metrics_summary.json` | Métricas de clasificación (`val_f1` calibrado en validación y referencia en $\tau = 0.50$), parámetros, FLOPs y tiempos de entrenamiento/inferencia |
| `training_history.csv` | Registro por época (`loss`, `accuracy`, `pr_auc`, `f1` de entrenamiento y validación) |
| `training_curves.png` | Gráficas de convergencia de pérdida y métricas |
| `confusion_matrix_and_curves.png` | Matriz de confusión, curva ROC y curva Precision-Recall en test |

> [!WARNING]
> **Regla de semilla única:** Para conservar resultados canónicos de un experimento específico sin sobrescribir la carpeta con semillas posteriores de una lista, invocar siempre con `--seeds 42` (por defecto en TOML `seeds = [42, 43, 44]` sobreescribiría la misma ruta al iterar). Con splits de validación muy pequeños (`mvtec-ad/bottle`, `N_val = 35`), la calibración `val_f1` puede colapsar a "todo defecto" (documentado en [../../HYPOTHESIS.md §2.2](../../HYPOTHESIS.md#22-exact-diagnosis-of-cnn-collapse-on-mvtec-ad_bottle-seed42)).

## Reproducción mínima

```powershell
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/kolektorsdd2.toml --model all --seeds 42 --overwrite
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/kolektorsdd2.toml --model all --seeds 42 --inverted --overwrite
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/mvtec_ad.toml --object bottle --model all --seeds 42 --overwrite
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/mvtec_ad.toml --object bottle --model all --seeds 42 --inverted --overwrite
```
