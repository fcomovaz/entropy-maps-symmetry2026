# Shannon entropy: counts, LUTs y portabilidad

Experimento de comprensión independiente del código del paper. Solo Python standard library y Pillow para decodificar la imagen grayscale ya existente. No usa NumPy, scipy, OpenCV, scikit-image, PyTorch, pandas ni numba. No se instalaron dependencias. No mide GPU, modelos, energía ni hardware embebido.

## Reproducir

Antes de repetir el benchmark, usar la [copia de ejecución](../../README.md#repetir-01-03-sin-sobrescribir-evidencia)
para no sustituir las mediciones canónicas. Los comandos siguientes operan sobre
los archivos originales. Los tests solo leen datos; benchmark y análisis escriben resultados.

Desde la raíz del repositorio, PowerShell, Python 3.11 del entorno existente:

```powershell
code/.venv/Scripts/python.exe code/understanding/03_entropy_benchmark/test_entropy.py
code/.venv/Scripts/python.exe code/understanding/03_entropy_benchmark/entropy_benchmark.py
code/.venv/Scripts/python.exe code/understanding/03_entropy_benchmark/analyze_results.py
```

Ejecutar secuencialmente, con el equipo sin otras cargas significativas. El benchmark tarda varios minutos. Parámetros por defecto: seed 20260928, 7 rondas, 5 llamadas warm-up por kernel, 300 repeticiones por ronda para entropía; pipeline/histograma `max(20, 300000//N)` repeticiones. Para ajustar duración: `--rounds`, `--entropy-repeats`, `--target-pixels`. La corrección no depende de esos parámetros. Los kernels presuponen counts enteros no negativos, 256 bins y suma N>0; un patch vacío no tiene entropía definida aquí.

Los datos, la selección y los cálculos son deterministas para este entorno; los tiempos no lo son. Se conserva el vector de tiempos de cada ronda, mínimo, máximo, mediana, versión de Python, reloj, plataforma, seed y SHA256 del script y de la imagen. No se promete identidad bit a bit entre distintas implementaciones de `libm`.

### Salida esperada y archivos que cambian

- `test_entropy.py`: cinco tests y `OK`; necesita las salidas de 01, incluidas
  coordenadas y muestras para 32/64/128.
- `entropy_benchmark.py`: cuatro mensajes `Completed`, luego `PASS: 7668 comparisons`.
  Genera 480 filas de entropía, 1200 de pipeline, 300 de histogramas, ocho de LUT
  y puntuaciones para 2156 patches. Sobrescribe los ocho CSV y `run_metadata.json`
  de `results/`; no actualiza `results/run1/` ni este README.
- `analyze_results.py`: lee esos resultados, valida cantidades/tolerancia/hash,
  contrasta manifest, puntuaciones, errores y rankings con `run1/` si existe y
  escribe `summary.md` y `validation.json`. No vuelve a medir tiempos.

Para una comprobación corta en la copia, usar `--rounds 1 --entropy-repeats 2
--target-pixels 1000` en una sola línea. Reduce mediciones, no casos ni ranking.
Los resultados temporales de esa corrida no equivalen a los del protocolo completo.

Si el análisis falla por hash, hay una discrepancia entre código y metadatos:
no cambiar el hash ni omitir la aserción para ocultarla. Conservar la corrida
original y repetir con su código, o generar una corrida nueva documentada.
Si faltan datos de 01, regenerarlos con su JPEG cacheado. Mantener los laboratorios
como hermanos aunque se ejecute desde otra carpeta.

### Dónde leer el código

En `entropy_benchmark.py`, seguir `histogram` → `e1` → `e2` →
`count_lut`/`term_lut` → `e3`/`e4` → `fixed_entropy` → `ranking_metrics`.
`datasets` construye entradas y comprueba coordenadas; `measure` cronometra;
`main` conecta validación y tablas. `sparse_entropy` es una formulación auxiliar:
los tiempos E5 se obtienen de los cierres especializados en `main`, con los
costes de copia de counts descritos abajo. No confundir la función auxiliar con
el kernel efectivamente cronometrado.

## Procedencia y alcance

Se leyeron `AGENTS.md`, el README raíz, README y resultados del experimento 01, y el código del 02. Al iniciar este trabajo el 02 aún no tenía resultados; aparecieron durante la primera ejecución y se revisaron antes de la segunda. Su tabla canónica informa H1=53.699 ns/pixel, Counter=45.141, array=92.270 y unroll4=74.630 (medianas sobre sus 35 casos), y recomienda el array directo como algoritmo portable. Este experimento vuelve a medir dentro del 03 H1 direct-address, H2 array, H5 unroll4 y H3 Counter, reproduciendo sus algoritmos. H1-local es un alias redundante; unroll8 no se incluyó. Counter es un control de hashing implementado parcialmente en C. Pillow histogram se excluye expresamente: aquí Pillow solo carga imágenes. No se modificó ni ejecutó el 02 para generar archivos fuera del alcance autorizado.

Se usa `01_patch_partition/outputs/A_divisible_patch_32/partition_input.png` (512×512, modo L). Durante la ejecución, un cambio externo movió los experimentos 01 y 02 de `Understanding/` a `code/understanding/`. Tras corregir la ubicación del experimento 03, el script busca el 01 como carpeta hermana en `code/understanding/`, sin modificarlo; se verificó que la imagen mantiene el SHA256 `712d7c327c78dcda0726c1d3070f87b31f3a9aa756b9c4b083380ce89795316d`. Todo el experimento 03 está en `code/understanding/03_entropy_benchmark/`. Los resultados históricos se conservaron sin recalcular tiempos; el hash del código ejecutado permanece en los metadatos y el ajuste posterior de rutas se registra por separado en `relocation`. Para 32, 64 y 128 se exige que las coordenadas calculadas coincidan con los CSV del 01 y los tests verifican todas sus imágenes de muestra byte por byte. Para 16 se particiona esa misma entrada sin solapamiento ni padding. Se incluyen todos sus 1024/256/64/16 patches en ranking; ocho posiciones espaciadas por tamaño se cronometran. Solo hay una imagen real: no permite generalizar a un dataset de segmentación.

Por tamaño se añaden siete casos sintéticos al benchmark: un único gris, 256 niveles exactamente equiprobables, dos niveles 50/50, cuatro niveles con pesos 90/5/3/2, gradiente horizontal con `size` niveles, aproximadamente uniforme en 256 bins con pequeño exceso en cero, y random uint8. “Uniforme” puede significar imagen constante (H=0) o distribución uniforme de intensidades (H=8); se prueban ambas, con nombres inequívocos.

Ranking añade 96 patches con alfabetos/pesos aleatorios y 96 casi equiprobables con transferencias unitarias entre bins. Los grupos `real`, `near_tie` y `all` se reportan por separado. Los tamaños nunca se mezclan en un ranking. No se evalúan políticas PARTIAL/PAD en el benchmark; tests adicionales verifican N no potencia de dos y reutilización E3. `manifest.csv` permite identificar cada patch y verificar sus bytes.

## Formulaciones y demostración

Sea `m` el número de bins no cero, `N=sum(c_i)` y `sum(p_i)=1`:

```text
H = -Σ (c_i/N) log2(c_i/N)
  = -Σ (c_i/N) [log2(c_i) - log2(N)]
  = -(1/N) Σ c_i log2(c_i) + log2(N) (Σ c_i/N)
  = log2(N) - (1/N) Σ c_i log2(c_i).
```

Las sumas de la derivación omiten ceros. Definir `0 log2(0)=0` por continuidad permite incluirlos. E1 implementa literalmente probabilidades y es la referencia. E2 usa counts y una división final. `log2(N)` se precomputa por tamaño para E2/E3, igual que se inicializan las LUTs; con N variable habría que calcularlo o recuperarlo de una tabla.

E3 almacena `c*log2(c)`, c=0..N. Una tabla para Nmax también sirve para patches menores usando su N y log2(N) correctos. E4 almacena `-(c/N)*log2(c/N)` y depende de N: no se puede reutilizar la misma tabla para tamaños distintos. E3/E4 recorren 256 bins sin branch para ceros. En Python se usan listas de floats double; “exacto” significa fórmula matemáticamente equivalente, implementada en coma flotante con error absoluto ≤1e-12 frente a E1, no aritmética real exacta. No se fuerza/clampa H a cero para ocultar error.

E5 construye una lista de niveles ocupados en orden de primer encuentro y conserva el histograma directo. Se cronometran tanto el branch/registro por pixel como la entropía posterior. E5+E3/E4 consultan directamente esos índices. E5+E1/E2 materializan los counts ocupados y reutilizan el kernel: esa asignación y copia están incluidas en el tiempo. Una implementación C podría evitarla; estos números no predicen su velocidad. Todas las variantes incluyen inicialización de 256 counters; no se experimenta con limpiar únicamente los bins de un patch anterior.

## Qué se cronometra

`time.perf_counter_ns`, warm-up y batches, con orden de métodos barajado por ronda mediante seed fijo. Se mide el mismo patch repetidamente: régimen de caché caliente, no streaming ni cold-cache. Se conserva el GC normal de CPython. No se resta el overhead del bucle/llamada; también afecta métodos rápidos. Las siete rondas pertenecen al mismo proceso y no son siete ejecuciones independientes del sistema.

1. Entropía aislada: histograma y LUT listos, solo histogram→entropy. En E5, `used_bins` ya está listo.
2. Histograma aislado: bytes listos→histograma, incluyendo inicialización, asignaciones y tracking cuando corresponde.
3. Pipeline: bytes listos→histograma→entropía, medido directamente. No incluye carga, crop, conversión, construcción de LUT ni preparación de datos.
4. Construcción E3/E4: medida por separado, 20 construcciones/ronda. `summary.md` estima cuántos patches amortizan esa construcción respecto a E1.

Las fracciones son estimaciones `T_hist/(T_hist+T_entropy)` y `T_entropy/(T_hist+T_entropy)` a partir de mediciones aisladas. No son una descomposición causal exacta de la latencia completa: el CSV conserva `component_sum_ns`, pipeline medido y `residual_ns`, que puede ser negativo por ruido/caché/llamadas. Para decidir el ganador se usa el pipeline medido. No se ha medido inferencia: en el paper seguirá siendo necesario separar `T_selector`, `T_model` y `T_total=T_selector+T_model`.

## LUT memory analysis

Cada LUT requiere exactamente N+1 entradas, incluyendo count=0 y count=N:

| Patch | N | Entradas | float64 | float32 / uint32 | uint16 |
|---|---:|---:|---:|---:|---:|
| 16×16 | 256 | 257 | 2056 B | 1028 B | 514 B |
| 32×32 | 1024 | 1025 | 8200 B | 4100 B | 2050 B |
| 64×64 | 4096 | 4097 | 32776 B | 16388 B | 8194 B |
| 128×128 | 16384 | 16385 | 131080 B | 65540 B | 32770 B |

Estos bytes describen almacenamiento contiguo C sin encabezados/alineación, no objetos Python. `lut_memory_build.csv` separa `python_deep_bytes` (lista más tamaños de sus elementos, sin asignador/interpreter) de esos valores. Float32 se analiza en memoria, **no** se validó su error; los métodos exactos medidos usan double.

Rango E3: máximo `N log2(N)` = 2048, 10240, 49152, 229376. En uint16 sin fracción solo caben los tres primeros; a 128×128 no cabe ni la parte entera. En uint32 los cuatro caben, pero el formato Q determina la precisión disponible. E3 double con Nmax=16384 ocupa 131080 B y cubre todos los N inferiores. E4 requiere tablas distintas: almacenar los cuatro tamaños simultáneamente exige 174112 B double, 87056 B float32/uint32 o 43528 B uint16. El benchmark mantiene tablas en RAM; almacenarlas en flash/ROM no consume esos mismos bytes de RAM, pero puede cambiar latencia de acceso.

Histograma C: 256 uint16 = 512 B para N≤16384 (uint8 no sirve cuando count=256); uint32 = 1024 B. E5 añade hasta 256 índices uint8 = 256 B y un contador de longitud capaz de representar 256. Los píxeles y buffers de imagen son adicionales y comunes. Hashing para un dominio exacto de 256 valores añade gestión de tabla, colisiones/metadatos; Counter puede ganar por su implementación nativa de Python, lo que no justifica trasladar hashing al MCU.

## Fixed-point: formato, rango y error

Se cuantiza E4, no la LUT de counts: `q[c]=floor(term[c]*2^F+0.5)`, redondeo al más cercano con empates hacia arriba. Se suman enteros y se convierte una sola vez para reportar H. No se mide velocidad de estos enteros Python como evidencia de rendimiento C.

`-p log2(p)` alcanza `1/(e ln 2)≈0.530738`; por eso la tabla **UQ0.16** cabe en uint16 (escala 65536), sin saturación. H completo puede llegar a 8: requiere acumulador uint32, no uint16. Con F=30 se usa uint32 para cada término (UQ2.30, aunque los valores usan <30 bits) y uint64 para la suma. El peor límite de suma es `8*2^F+128`, por 256 errores de redondeo de a lo sumo media unidad: 20 bits unsigned para F=16 y 34 para F=30 bastan. No usar acumulación uint32 para F=30.

El error absoluto satisface `|Hq-H| ≤ m/(2*2^F)`, más el diminuto error de construir la tabla en double: como máximo 0.001953125 bits con F=16 y 1.1920929e-7 con F=30. Se reportan MAE y máximo absoluto, y errores relativos solo para H>1e-12; dividir por H=0 no tiene sentido. Los tests verifican rango, acumulación y la cota. En C se pueden generar tablas offline; la política de redondeo y la escala deben quedar fijadas en el artefacto, no depender de conversiones implícitas.

## Ranking y correctitud

Los CSV contienen las puntuaciones individuales, máximos/medias de error y ranking. Una inversión estricta exige que el par E1 esté separado por >1e-12 y que el método invierta ese orden por >1e-12. Se cuentan por separado pares distintos colapsados a empate, empates E1 rotos y posiciones diferentes en el orden completo. Top-k se mide como `|top_k_ref ∩ top_k_method|/k` para k=1,5,10, floor(10%) y floor(25%) de cada grupo.

El sort usa H descendente sin tolerancia y después índice estable. Por eso redondeos de double pueden cambiar un top-k entre patches prácticamente empatados sin producir inversiones estrictas; no se oculta este efecto. La colección casi uniforme estresa precisamente esos casos. Una futura política de tolerancia para selección necesita especificarse y validarse; no se aplica aquí para maquillar agreement.

`test_entropy.py` verifica los tres resultados analíticos, reutilización de E3 para N variable, histogramas/tracking con longitudes no múltiplo de cuatro, límites fixed-point y que la métrica de ranking detecte inversiones y empates conocidos. El benchmark valida además cada patch de ranking y cada pipeline medido. Una aserción aborta si un método exacto excede 1e-12.

## Operaciones conceptuales

N píxeles, B=256 bins y m bins ocupados. Se cuentan operaciones del algoritmo escalar, no instrucciones CPU ni overhead Python. Inicializar el histograma requiere además B escrituras; acumular H requiere B o m sumas. Las condiciones de control del bucle se omiten de “branches”, que cuenta condiciones de datos. `log2(N)` está precomputado.

| Método con histograma H1 | Increments hist | Iteraciones bins | Divisiones | Multiplicaciones | log2 | LUT accesses | Branches de datos |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 | N | B | m | m | m | 0 | B |
| E2 | N | B | 1 | m | m | 0 | B |
| E3 | N | B | 1 | 0 | 0 | B | 0 |
| E4 | N | B | 0 | 0 | 0 | B | 0 |
| E5+E1 | N | m | m | m | m | 0 | N + m* |
| E5+E2 | N | m | 1 | m | m | 0 | N + m* |
| E5+E3 | N | m | 1 | 0 | 0 | m | N |
| E5+E4 | N | m | 0 | 0 | 0 | m | N |
| E4 fixed | N | B | 1 final al reportar | 0 | 0 | B | 0 |

`*` Los kernels E1/E2 reutilizados aún comprueban count!=0 y copian m counts; en C podría eliminarse esa comprobación. E5 añade m registros/append, m lecturas indirectas del histograma y control del contador de usados. Para ranking fixed puro, la conversión/división final puede omitirse y comparar sumas enteras.

H2 y H5 conservan N increments; H2 en Python añade la conversión de 256 counters a lista. H3 Counter añade hashing/lookup por pixel y 256 búsquedas para producir una lista densa: su cantidad de probes/branches depende de implementación, no se asigna un número inventado. LUT build E3 tiene N log2 y N multiplicaciones; E4 en este código tiene 2N divisiones, N log2 y N multiplicaciones. Es coste de inicialización, fuera del patch loop. El compilador C podría reemplazar división por constante con multiplicación/shift; este experimento no mide ese efecto.

## Resultados disponibles

Consultar [tablas medidas](results/summary.md), [entropía aislada](results/entropy_benchmark.csv), [pipeline](results/pipeline_benchmark.csv), [histogramas](results/histogram_benchmark.csv), [LUT build/memoria](results/lut_memory_build.csv), [errores](results/errors.csv), [ranking](results/ranking.csv), [puntuaciones](results/patch_scores.csv), [manifest](results/manifest.csv) y [metadatos](results/run_metadata.json). Los resultados deben interpretarse como evidencia de CPython en este equipo, no benchmark de C ni garantía de ahorro energético.

Se completaron **dos ejecuciones completas**, 7668 comparaciones exactas por ejecución y cinco tests finales correctos. Los resultados canónicos son la segunda ejecución; `results/run1/` conserva la primera. Datos, puntuaciones, errores y rankings resultaron **idénticos byte por byte** entre ambas, validado por `analyze_results.py` y registrado en [validation.json](results/validation.json). Los CSV canónicos incluyen 480 filas de entropía, 1200 de pipeline y 300 de histogramas; ranking sobre 2156 patches. No se oculta la variación temporal: por ejemplo, la mediana H1+E1 de 32×32 varió −23.9%. No usar diferencias pequeñas entre kernels para afirmar superioridad estadística.

Medianas entre los 15 casos de cada tamaño (ns/patch, no medias ponderadas de un dataset final):

| Patch | E1 aislado | E3 aislado | E4 aislado | H1+E1 | H1+E4 | Menor mediana pipeline en run2 |
|---|---:|---:|---:|---:|---:|---|
| 16×16 | 16684 | 7995 | 8245 | 29590 | 22308 | H1+E3: 22251 |
| 32×32 | 11238 | 8135 | 8383 | 67034 | 58111 | H1+E3: 56747 |
| 64×64 | 38798 | 8409 | 8408 | 225636 | 197747 | Counter+E3: 153719 |
| 128×128 | 48322 | 8483 | 8486 | 851670 | 755490 | Counter+E3: 502725 |

## Best exact implementation

Recomendación: **E4 double para N fijo y LUT reutilizada**, con H1 directo como histograma portable. E3 queda como alternativa equivalente cuando se necesitan distintos N con una sola tabla. E4 reproduce E1 bit a bit en los patches medidos porque usa los mismos términos y orden de suma; E2/E3 tienen error máximo 1.60e-14, muy inferior a 1e-12. E3 obtuvo la menor mediana del pipeline en run2: la preferencia E4 para N fijo se basa en su fidelidad numérica y bucle simple, no en afirmar que fue el más rápido. Evita normalización por patch; diferencias pequeñas E3/E4 no prueban un ganador temporal estable. Counter puede ganar para patches grandes, sin convertirse en recomendación de hashing para C.

## Best embedded implementation

Para un MCU con memoria limitada y varios tamaños, trasladaría primero **256 counters uint16 + E2**, manteniendo E1 como referencia: 512 B de counters, sin tabla proporcional a N y una sola división final. Es una decisión de simplicidad/memoria, no un ganador medido en MCU. Con N fijo pequeño y flash suficiente, elegiría **E4 en ROM**: double si se requiere la misma precisión validada; Q30 si se quiere aritmética entera conservadora, con suma uint64. Q16 necesita menos flash y suma uint32, pero requiere aceptar y evaluar cambios de ranking. No se ha medido C, coste de libm, FPU, accesos flash ni latencia energética; esa validación queda explícitamente pendiente para hardware real.

## Is a LUT justified?

Sí como optimización de un kernel reutilizado, condicionada al presupuesto de memoria. E3/E4 eliminan log2 del bucle y la tabla medida cuantifica tiempo de construcción/amortización. No hay una garantía de gran ahorro total: a 128×128 la entropía E4 representa alrededor del 1% del pipeline H1 en patches reales; acelerar ese 1% tiene un límite de beneficio muy pequeño. A 16×16 su fracción es mucho mayor. La tabla double crece de 2056 a 131080 B: mantenerla solo para ahorrar pocas operaciones en un MCU pequeño puede ser una mala decisión. Las latencias completas y su variabilidad tienen prioridad sobre un speedup aislado.

## Is fixed-point justified?

Como opción de implementación, sí; como sustituto sin consecuencias, no. En `all`, Q16 produjo 2/0/34/282 inversiones estrictas para tamaños 16/32/64/128. Su máximo error absoluto fue aproximadamente 0.00129 bits; el pequeño error no garantiza orden idéntico. Q30 no tuvo inversiones estrictas, con máximo absoluto ≈2.14e-8 bits; su mayor memoria y acumulación de 64 bits son el precio. Es evidencia sobre estos conjuntos, no prueba universal de ranking: para pares separados menos que la suma de cotas de error puede haber cambios. No se ha demostrado velocidad fixed-point en hardware.

## Is sparse-bin tracking justified?

Solo merece consideración como opción principal para patches pequeños con muy pocos niveles: E5+E4 ganó a H1+E4 en imagen constante 16×16 por 1.16×/1.23× (run1/run2), y en cuatro niveles por 1.20×/2.54×, con magnitud claramente inestable. E5+E4 perdió en todos los casos representativos de 32/64/128 y alta ocupación de la tabla. Hay excepciones específicas E5+E1 (random 32×32: 1.10×/1.17×), que no convierten tracking en ganador sobre el pipeline con LUT. El ahorro es como máximo evitar recorrer 256 bins, mientras el branch extra cuesta N operaciones y los registros cuestan m. Con alta ocupación casi no se ahorran lecturas; para patches grandes ese coste domina incluso con entropía baja. No se recomienda activarlo universalmente. Las variantes E1/E2 incluyen además la copia de counts ya explicada; para una decisión final C hay que medir un bucle especializado sin esa copia.

## Implication for entropy-aware patch selection

Las formulaciones exactas conservan el criterio matemático. E4 mantuvo todas las posiciones y top-k medidos; E2/E3 no tuvieron inversiones estrictas, aunque su top-10 agreement fue 0.9 en el grupo casi empatado de 32×32 por empates numéricos. Q16 sí puede cambiar el orden y top-k: en `all`, 128×128, k=21 dio 16/21=0.7619, aunque top-10 fue 1.0; en `near_tie`, k=24 dio 0.875. En los patches reales, Q16 tuvo dos inversiones a 16×16 pero agreement 1.0 para todos los k evaluados; no se generaliza este éxito a otros datos. Los CSV separan ambos grupos para no esconderlo detrás del conjunto combinado. Definir desempates deterministas, evaluar el k realmente utilizado por el selector y relacionar cambios con calidad del modelo son pasos necesarios antes de adoptar cuantización. Este experimento no mide accuracy, IoU ni la relación entre entropía y utilidad semántica.

## Recommendation for the paper

Usar E1 como definición y referencia verificable; describir la identidad de counts y documentar la variante finalmente desplegada, escala, tipos y política de empate como detalles reproducibles. Incluir latencia total del selector (histograma + entropía + ranking/selección) y luego del modelo, junto con presupuesto de flash/RAM. Si fixed-point altera la selección, merece una ablación sobre el dataset final y sus métricas; no basta el error de entropía. **La LUT es ingeniería estándar, no una contribución científica novedosa.** La contribución principal continúa siendo el esquema entropy-aware de selección/adaptación y su compromiso calidad–coste, no ahorrar una división o multiplicación.
