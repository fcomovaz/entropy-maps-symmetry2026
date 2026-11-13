# Cuadernos exploratorios y ejecución en la nube (`notebooks/`)

Contiene el cuaderno reproducible de ejecución completa desde datasets ya preprocesados:

- [`Entropy_Maps_Full_Run.ipynb`](Entropy_Maps_Full_Run.ipynb): ejecuta extremo a extremo en Google Colab (GPU + CPU) las etapas 0–7 sobre KolektorSDD2 y **las 15 categorías de MVTec-AD** (normal + invertido), apoyándose en [`../configs/cnn/colab.toml`](../configs/cnn/colab.toml) y registrando comandos, duraciones y código de salida en `results/colab_run_manifest.json`.

> [!NOTE]
> Las mediciones de tiempo tomadas en Colab describen el hardware de la sesión registrado en el manifiesto, no la estación de trabajo local ni microcontroladores físicos. Los módulos reutilizables viven siempre en `src/`, no dentro de los notebooks.
