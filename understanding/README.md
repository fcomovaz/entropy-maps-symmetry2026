# Laboratorio de comprensión

Este espacio existe para comprender antes que optimizar. Contiene código explícito, visualizaciones, inspección de valores intermedios, reconstrucciones manuales de algoritmos y comparaciones contra implementaciones de librerías. No es código de producción: una implementación pasa a `src/` únicamente cuando está entendida, validada y es necesaria para el pipeline formal.

## Ubicación y experimentos

La ubicación del laboratorio es `code/understanding/`, dentro de la carpeta `code/` del proyecto. En PowerShell también se escribe `code\understanding\`. Los comandos de los README de cada experimento se ejecutan desde la raíz del proyecto con `code/.venv/Scripts/python.exe`.

- [01_patch_partition/](01_patch_partition/README.md): partición visual de imágenes, coordenadas y reconstrucción; conserva sus imágenes en `assets/` y sus salidas en `outputs/`.
- [02_histogram_benchmark/](02_histogram_benchmark/README.md): comparación de histogramas; usa imágenes del experimento 01 y guarda resultados en su propio `results/`.
- [03_entropy_benchmark/](03_entropy_benchmark/README.md): comparación y validación de entropía; usa patches del experimento 01 y guarda resultados en su propio `results/`.
- [04_lightweight_cnns/](04_lightweight_cnns/README.md): siete arquitecturas, inspección, entrenamiento artificial, inferencia y benchmark CPU/CUDA; conserva checkpoints y medidas en `outputs/`.

Los cuatro experimentos son carpetas hermanas. Los scripts resuelven las rutas de datos desde su propia ubicación mediante `Path(__file__)`. Las rutas pasadas por el usuario en `--image`, `--data` o `--output` son relativas al directorio actual. El laboratorio 04 aún no consume patches seleccionados por entropía.

## Ruta de comprensión

| Orden | Recorrido | Qué observar |
|---|---|---|
| 01 | Imagen → coordenadas → recorte → reconstrucción | Cobertura, descarte y padding; ver `outputs/run_summary.json` y las rejillas |
| 02 | Bytes del patch → 256 counts | Igualdad exacta, suma N y coste; ver `results/summary_table.md` |
| 03 | Counts → probabilidades → términos → H → ranking | Equivalencia, error y cambios top-k; ver `results/summary.md` |
| 04 | Tensor → convoluciones → pooling → logits | Shapes, parámetros, entrenamiento e inferencia; ver `outputs/validation.md` |

01 alimenta a 02 y 03. El 03 reproduce algoritmos de 02 dentro de su script, sin importarlo. El 04 funciona de forma independiente.

## Qué hemos entendido y validado

- **Geometría:** tamaño y política de borde determinan regiones, cobertura y datos artificiales. Se comprueba reconstrucción píxel a píxel.
- **Histograma:** el conteo directo es una referencia simple y exacta para 256 niveles. Diferencias de tiempo Python/librería no demuestran mejoras algorítmicas trasladables a C.
- **Shannon:** E1 define `-sum(p * log2(p))`; E2 usa counts; E3/E4 reutilizan tablas. Se valida equivalencia con tolerancia y se cuantifica memoria. Q16 puede cambiar el ranking pese a un error pequeño; Q30 no tuvo inversiones estrictas en los casos medidos.
- **CNN:** convolución convencional, depthwise/pointwise, residuales, pooling global, logits y carga de pesos tienen ejemplos ejecutables. Parámetros y latencia son magnitudes distintas.

Estas conclusiones tienen evidencia local; no equivalen a haber demostrado la hipótesis del paper. Los números históricos, seeds, configuración y límites están en cada laboratorio.

## Qué falta comprender o comprobar

1. Definir tarea, dataset y particiones de entrenamiento/validación/prueba.
2. Evaluar si entropía, varianza o bordes identifican regiones útiles: el ruido también puede producir entropía alta.
3. Definir selector top-k/threshold, desempates y agregación. Los rankings del 03 evalúan fidelidad numérica, no un selector conectado a CNN.
4. Comparar con imagen completa/todos los patches y selección aleatoria bajo condiciones equivalentes; medir calidad y coste total.
5. Solo después, elegir qué pasa a `src/` y evaluar memoria, C/C++ y hardware embebido.

Sobel, selectores del pipeline y estudios de MLP aún no están implementados. La pequeña red del test del entorno solo comprueba CUDA. Una imagen real y datos artificiales no permiten generalizar a una tarea de detección o segmentación.

## Reproducibilidad y conservación

Consultar los [comandos comunes](../README.md) y ejecutar benchmarks secuencialmente, sin cargas fuertes simultáneas. Una corrida corta comprueba funcionamiento; no reproduce las condiciones temporales de una corrida completa.

01–03 escriben en rutas fijas. Usar la [copia de trabajo](../README.md#repetir-01-03-sin-sobrescribir-evidencia) antes de repetirlos. El 04 permite `--output`. No editar CSV para ajustarlos a una conclusión.

## Limpieza de estructura · 2026-09-28

Se quitaron `edges/`, `entropy/`, `histograms/`, `models/`, `patches/`, `selectors/` y `visualizations/`: solo tenían `.gitkeep`. Los temas estudiados viven en los laboratorios numerados y los pendientes quedan descritos arriba. `models/` y `tests/` del laboratorio 04 sí contienen código y se conservan.
