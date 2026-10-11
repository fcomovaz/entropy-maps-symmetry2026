# Experimentos científicos · reservado

Esta carpeta se conserva para evaluar el futuro pipeline con baselines
comparables, dataset, seeds y configuración registrados. Aún no contiene
experimentos integrados; los estudios actuales son los cuatro laboratorios de
[understanding/](../understanding/README.md).

El siguiente estudio deberá definir tarea/dataset, baseline sin selección,
selección aleatoria y selección informativa. Registrar calidad, patches
procesados y `T_selector`, `T_model`, `T_total`; no inferir ahorro de tiempo
solo por reducir regiones. Las salidas formales irán a `../results/`.

Se retiraron las reservas vacías `cnn/`, `mlp/`, `entropy_threshold/`,
`patch_size/` y `selector_comparison/` el 2026-09-28. Se crearán estudios con
nombres y protocolos concretos cuando se implementen.
