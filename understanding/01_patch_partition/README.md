# Patch partition: visual study

Este experimento muestra exactamente qué regiones de una imagen grayscale se convertirían en patches antes de calcular histogramas o entropía. No implementa ningún selector ni modifica el pipeline del proyecto.

La lógica de coordenadas está escrita con bucles y aritmética entera de Python. `urllib.request` descarga la imagen si falta; Pillow decodifica, recorta coordenadas ya calculadas, dibuja y guarda imágenes. No se usan NumPy, OpenCV, scikit-image, PyTorch ni torchvision.

## Ejecución

Para conservar las salidas actuales, crear primero la [copia de ejecución](../../README.md#repetir-01-03-sin-sobrescribir-evidencia). Los comandos directos siguientes regeneran archivos en este laboratorio.

Desde la raíz del repositorio, usando el entorno existente:

```powershell
code\.venv\Scripts\python.exe code\understanding\01_patch_partition\patch_partition.py --patch-size 64
```

Esto genera siempre el experimento A completo (patches 32, 64 y 128) y el experimento B con las tres políticas. Para limitar B a una política:

```powershell
code\.venv\Scripts\python.exe code\understanding\01_patch_partition\patch_partition.py --patch-size 64 --edge-policy pad
```

Consultar todas las opciones:

```powershell
code\.venv\Scripts\python.exe code\understanding\01_patch_partition\patch_partition.py --help
```

La primera ejecución descarga una imagen fija desde Lorem Picsum y la guarda en `assets/`. Las siguientes ejecuciones reutilizan el archivo, incluso offline. `assets/source_metadata.json` registra URL, resolución y SHA256. La variante 500x333 es un recorte determinista de la imagen grayscale 512x512 descargada.

Con `--patch-size 64` y sin `--edge-policy`, la consola termina con `Validated 6 cases`:
256/64/16 patches para A (32/64/128) y 35/48/48 para B (drop/partial/pad).
Abrir primero `outputs/A_divisible_patch_64/grid.png`, después
`outputs/B_500x333_patch_64_partial/contact_sheet.png` y finalmente
`outputs/run_summary.json`. Con una sola política se validan cuatro casos.

`--patch-size` solo cambia B; A siempre ejecuta 32, 64 y 128. Repetir sobrescribe
los archivos de esos casos, el resumen y los metadatos del asset; no borra carpetas
de otros tamaños ni políticas anteriores. `run_summary.json` describe únicamente
la última ejecución. Conservar el JPEG cacheado: el script registra su hash,
pero no lo contrasta contra un hash histórico antes de reutilizarlo. Descargar
de nuevo en el futuro no garantiza los mismos bytes.

## Dónde leer el código

En `patch_partition.py`, seguir `PatchBox` → `patch_boxes` → `validate_boxes` →
`prepare_canvas` → `reconstruct` → `run_case` → `main`. `self_test` fija los
casos geométricos esperados. Las imágenes usan `(width, height)`; las cajas usan
`(x0, y0, x1, y1)`, no índices `(fila, columna)`.

## Coordenadas e indexación

Los patches se recorren por filas (row-major) y el ID crece de izquierda a derecha:

```text
          column 0     column 1     column 2
row 0     patch 0      patch 1      patch 2
row 1     patch 3      patch 4      patch 5
```

Una caja `(x0, y0, x1, y1)` usa límite final exclusivo, igual que los rangos de Python:

```text
                 x0              x1 (no incluido)
                  +---------------+
               y0 |   píxeles     |
                  |   del patch   |
y1 (no incluido)  +---------------+

width  = x1 - x0
height = y1 - y0
```

Ejemplo para el primer patch 64x64: `(0, 0, 64, 64)`. Contiene x=0..63 e y=0..63. En una imagen 512x512, el último es `(448, 448, 512, 512)`.

## Experimento A: 512x512 divisible

Para patches cuadrados no solapados:

```text
patch size    filas x columnas    total
32x32         16 x 16             256
64x64          8 x  8              64
128x128        4 x  4              16
```

Cada carpeta `outputs/A_divisible_patch_*` contiene la entrada, la rejilla, un contact sheet numerado, patches de muestra, coordenadas CSV, métricas JSON y la reconstrucción. El script compara píxel por píxel la reconstrucción con los patches que la originaron y falla si difieren. En A eso equivale a toda la entrada; en B-DROP la zona descartada aparece negra y no se presenta falsamente como reconstruida.

## Experimento B: 500x333 no divisible, patch 64x64

```text
DROP                         PARTIAL                      PAD
7 x 5 patches completos      8 x 6 patches               canvas 512x384
+---------+ resto x          +----------+                +----------+pp
|[][][][][]|                 |[][][][][]p| borde estrecho |[][][][][][]
|[][][][][]|                 |[][][][][]p|                |[][][][][][]
+---------+                  +----------+                +----------+pp
resto inferior               última fila más baja         p = padding cero
```

- `DROP`: genera 35 patches. Procesa 143,360 píxeles reales, descarta 23,140 y cubre 86.102102% de la imagen. Ventaja observable para un sistema embebido: todos los buffers de patch tienen tamaño uniforme y no se crea padding. Desventaja: se pierde información real de los bordes.
- `PARTIAL`: genera 48 patches. Procesa los 166,500 píxeles reales, no descarta ni añade píxeles. Ventaja: cobertura real completa sin trabajo artificial. Desventaja: los patches del borde tienen tamaños variables (el último mide 52x13), lo que exige manejar dimensiones distintas.
- `PAD`: genera 48 patches uniformes sobre 512x384. Procesa los 166,500 píxeles reales y añade 30,108 píxeles cero. Ventaja: buffers y bucles uniformes con cobertura completa. Desventaja: se almacenan/procesan píxeles artificiales que podrían afectar métricas posteriores.

Las cifras anteriores corresponden exactamente a 500x333 y 64x64. Para otros tamaños, consultar `metrics.json`; no deben extrapolarse sin ejecutar el caso.

## Archivos de salida

```text
outputs/<caso>/
  partition_input.png       imagen realmente particionada (incluye padding en PAD)
  grid.png                  límites de todos los patches
  contact_sheet.png         patches con ID, fila y columna
  reconstruction.png        imagen reconstruida
  patch_coordinates.csv     coordenadas y dimensiones exactas
  metrics.json              conteos, cobertura y validación
  sample_patches/           primeros, centrales y últimos patches
outputs/run_summary.json    resumen de la ejecución más reciente
```

En visualizaciones PAD, el contorno azul marca el límite de la imagen real y las zonas negras exteriores son padding cero.

## Validaciones automáticas

Antes de generar resultados, el script comprueba coordenadas positivas y dentro del canvas, ausencia de overlaps, conteos esperados, tamaños uniformes donde corresponden y casos exactos DROP/PARTIAL/PAD. Cada ejecución reconstruye también todos los patches generados y exige igualdad pixel-perfect. En DROP, los píxeles descartados se representan con negro porque, por definición, no pueden reconstruirse desde los patches.

## Implications for the paper

- El tamaño de patch cambia directamente la cantidad de regiones: en 512x512, pasar de 32 a 128 reduce el conteo de 256 a 16.
- Una resolución no divisible obliga a definir explícitamente una política de borde; DROP, PARTIAL y PAD producen regiones y costes de datos distintos.
- DROP omite píxeles reales; PARTIAL conserva todos pero introduce tamaños variables; PAD conserva todos con patches uniformes a cambio de píxeles artificiales.
- Estas observaciones describen geometría y cantidad de datos. Este experimento no mide precisión, entropía, latencia, memoria ni consumo energético, y por tanto no demuestra que una política sea superior.
