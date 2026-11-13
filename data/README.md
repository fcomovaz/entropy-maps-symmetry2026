# Datos del proyecto (`data/`)

Los archivos pesados de imágenes y CSV están excluidos de Git en [../.gitignore](../.gitignore) (se conservan únicamente este archivo y los `.gitkeep`), con rutas y variantes configuradas en [../configs/base.toml](../configs/base.toml).

## Estructura y artefactos locales

```text
data/
├── raw/            # Datasets originales sin modificar (colocar manualmente antes de preprocesar)
│   ├── mvtec-ad/<objeto>/{train/good, test/<defecto>, ground_truth/<defecto>}
│   └── KolektorSDD2/{train, test}/N.png + N_GT.png
├── preprocessed/   # Imágenes 128x128 escala de grises ('L') con letterbox centrado (pad=0)
│   ├── mvtec-ad/                  # Conserva máscaras binarias {0, 255} (NEAREST)
│   ├── mvtec-ad-inverted/         # Imágenes 255 - I antes de padding; máscaras intactas
│   ├── KolektorSDD2/              # Separado en good/defect por máscara nativa (sin máscaras en salida)
│   └── KolektorSDD2-inverted/     # Variante 255 - I antes de padding; lateral en pad=0
└── csv/            # Exportaciones tabulares y descriptores derivados (normal e -inverted)
    ├── mvtec-ad[-inverted].csv                 # 16,391 cols x 6,612 filas (px_00000..px_16383)
    ├── mvtec-ad[-inverted]-entropy.csv/.json   # 85 características de entropía (x85, bins=16)
    ├── mvtec-ad[-inverted]-entropy-full.csv/.json      # 94 características (schema x94-v1)
    ├── kolektorsdd2[-inverted].csv             # 16,389 cols x 3,335 filas (px_00000..px_16383)
    ├── kolektorsdd2[-inverted]-entropy.csv/.json       # 85 características de entropía (x85)
    └── kolektorsdd2[-inverted]-entropy-full.csv/.json  # 94 características (schema x94-v1)
```

Para generar `preprocessed/` y `csv/` a partir de `raw/`, seguir [../README.md](../README.md) o [../experiments/README.md](../experiments/README.md).

---

# Download MVTEC-AD Dataset


The official link for some reason is broken, so you can download the full dataset from huggingface.

```powershell
python -m pip install -U huggingface_hub
hf download foersben/mvtec-ad --repo-type dataset --local-dir ".\data\raw\mvtec-ad"
```

Cite as follows:

```bibtex
@inproceedings{bergmann2019mvtec,
  title={MVTec AD--A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection},
  author={Bergmann, Paul and Fauser, Michael and Sattlegger, David and Steger, Carsten},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={9584--9592},
  year={2019}
}
@article{bergmann2021mvtec,
  title={The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection},
  author={Bergmann, Paul and Fauser, Michael and Sattlegger, David and Steger, Carsten},
  journal={International Journal of Computer Vision},
  volume={129},
  number={4},
  pages={1038--1059},
  year={2021},
  publisher={Springer}
}
```


# Download Kolektor Surface-Defect Dataset 2 (KolektorSDD2)

The official link is [here](https://www.vicos.si/resources/kolektorsdd2/). Cite as follows:

```bibtex
@article{Bozic2021COMIND,
  author = {Bo{\v{z}}i{\v{c}}, Jakob and Tabernik, Domen and 
  Sko{\v{c}}aj, Danijel},
  journal = {Computers in Industry},
  title = {{Mixed supervision for surface-defect detection:
from weakly to fully supervised learning}},
  year = {2021}
}
```