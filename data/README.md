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