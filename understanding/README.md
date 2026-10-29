# Laboratorio de comprensión

Este espacio existe para **comprender antes que optimizar** (`understanding > optimization`). Contiene implementaciones explícitas, visualizaciones, inspección de valores intermedios, formulaciones matemáticas directas y pruebas unitarias de comportamiento. 

No es código de producción: una implementación migra a `src/` únicamente cuando sus propiedades están matemáticamente entendidas, empíricamente caracterizadas y son requeridas por el pipeline formal.

---

## Estado general: ¿En qué fase estamos?

El proyecto se encuentra en la transición entre el análisis exploratorio a nivel de componentes y la experimentación formal con datasets reales:

```
[ Fase 1: Primitivas ]  ──>  [ Fase 2: Baselines CNN ]  ──>  [ Fase 3: Descriptor x94 ]  ──>  [ Fase 4: Dataset Real ]  ──>  [ Fase 5: Experimentos ]
   (01, 02, 03)                     (04)                         (05, 06, 07)                       (08)                  (experiments/)
    COMPLETADA                       COMPLETADA                    COMPLETADA                    LISTA PARA VALIDAR             SIGUIENTE HITO
```

1. **Fase 1 — Primitivas matemáticas y computacionales (Completada):**
   - Geometría de patches, bordes DROP/PARTIAL/PAD y reconstrucción exacta ([01](01_patch_partition/README.md)).
   - Conteo directo de histogramas 256-bin como referencia $\Omega(N)$ ([02](02_histogram_benchmark/README.md)).
   - Equivalencia numérica de Shannon E1–E5, cuantización de tablas LUT y fidelidad de rankings ([03](03_entropy_benchmark/README.md)).
2. **Fase 2 — Modelos de referencia neuronal (Completada):**
   - Siete arquitecturas CNN ligeras (1.1k a 1.5M parámetros), perfiles de memoria y latencia CPU/CUDA aislada ([04](04_lightweight_cnns/README.md)).
3. **Fase 3 — Extractor manual de características x94 (Completada):**
   - 9 residuos de asimetría espacial de distribución de entropía (LR, TB, R180) ([05](05_symmetry_features/README.md)).
   - Extractor canónico multiescala `image -> x94` bajo esquema congelado `x94-v1` (85 entropías + 9 asimetrías) ([06](06_entropy_pyramid_descriptor/README.md)).
   - Caracterización científica del comportamiento de `x94` ante 15 perturbaciones controladas: ruido, desenfoque, contraste, defectos locales y colisiones estructurales ([07](07_descriptor_behavior/README.md)).
4. **Fase 4 — Validación y control de dataset real (Implementada / Lista para validación):**
   - Herramientas de inspección, verificación métrica y visualización del dataset industrial MVTec-AD estandarizado a 128x128 en escala de grises y variante invertida ([08](08_verify_mvtec_preprocessing/README.md)).
5. **Próxima fase — Integración científica (`code/experiments/`):**
   - Clasificación de defectos con clasificadores ligeros (SVM, RF, MLP) consumiendo `x94`.
   - Pipeline de selección de regiones informativas (top-k / umbral) conectado a las CNN de [04](04_lightweight_cnns/README.md).
   - Balance riguroso de costes: $T_{\text{total}} = T_{\text{selector}} + T_{\text{model}}$.

---

## Mapa de los ocho laboratorios

Todos los laboratorios residen en carpetas hermanas dentro de `code/understanding/`:

| Laboratorio | Enfoque central | Entrada / Datos | Salida principal |
|---|---|---|---|
| [01_patch_partition/](01_patch_partition/README.md) | Geometría de recorte y reconstrucción | Imagen de prueba en `assets/` | Rejillas visuales y `outputs/run_summary.json` |
| [02_histogram_benchmark/](02_histogram_benchmark/README.md) | 8 métodos de conteo de 256 niveles | Patches generados por 01 | Medidas de coste en `results/summary_table.md` |
| [03_entropy_benchmark/](03_entropy_benchmark/README.md) | Shannon E1–E5, LUTs y cuantización | Patches generados por 01 | `results/summary.md` y `validation.json` |
| [04_lightweight_cnns/](04_lightweight_cnns/README.md) | 7 arquitecturas CNN para visión embebida | Sintético / fixtures de prueba | Checkpoints, `validation.md` y benchmarks |
| [05_symmetry_features/](05_symmetry_features/README.md) | Residuos espaciales de asimetría | Reutiliza E1 de 03; imágenes sintéticas | Formulaciones acotadas, matrices 2x2/4x4/8x8 |
| [06_entropy_pyramid_descriptor/](06_entropy_pyramid_descriptor/README.md) | Extractor canónico `image -> x94` | Sintético / imágenes arbitrarias | API `extract_descriptor`, schema `x94-v1` |
| [07_descriptor_behavior/](07_descriptor_behavior/README.md) | Pruebas unitarias científicas sobre `x94` | Reutiliza 06; escenas analíticas | 279 filas medidas, curvas de respuesta y CSVs |
| [08_verify_mvtec_preprocessing/](08_verify_mvtec_preprocessing/README.md) | Control de integridad de MVTec-AD | `data/preprocessed/mvtec-ad(-inverted)` | Métricas MB y `preprocessing_verification.png` |

---

## Ruta metodológica de comprensión

```
Píxeles crudos
     │
     ├──> [01] Geometría de partición ──> [02] Histogramas ──> [03] Entropía Shannon
     │                                                               │
     │                                                               ├──> [05] Asimetrías espaciales
     │                                                               │           │
     │                                                               └──> [06] Descriptor canónico x94
     │                                                                           │
     ├──> [04] Modelos CNN de referencia (GTX 1050)                              └──> [07] Comportamiento x94
     │                                                                                       │
     └──> [08] MVTec-AD estandarizado (128x128, grises, letterbox) <────────────────────────┘
                 │
                 └───> Conexión formal en experiments/ (clasificación, selección y coste total)
```

---

## Qué hemos entendido y validado empíricamente

- **Geometría y reconstrucción (01):** las políticas DROP, PARTIAL y PAD imponen compromisos estrictos entre cobertura completa y distorsión por padding artificial. La reconstrucción píxel a píxel está demostrada.
- **Histogramas (02):** el conteo por indexación directa es óptimo ($\Theta(N + 256)$). Diferencias temporales en Python reflejan sobrecostes del runtime y no ventajas algorítmicas trasladables a C embebido.
- **Entropía de Shannon (03):** E1 define la referencia canónica $- \sum p \log_2(p)$. Las tablas LUT ahorran tiempo pero exigen memoria (evaluada por tamaño de patch). Cuantizaciones bajas (Q16) alteran el orden relativo de los parches (rankings top-k); Q30 preserva el orden sin discrepancias en los casos medidos.
- **Modelos neuronales (04):** parámetros y latencia son magnitudes desacopladas (modelos con menos parámetros pueden ser más lentos si carecen de núcleos optimizados).
- **Asimetrías espaciales (05):** los 9 residuos $A_{\text{LR}}, A_{\text{TB}}, A_{\text{R180}}$ capturan desbalances macroscópicos de distribución de entropía y satisfacen desigualdades triangulares estrictas. No añaden información matemática nueva sobre los 85 valores de entropía, pero hacen explícitas relaciones estructurales espaciales.
- **Descriptor canónico x94 (06):** vector de 94 floats acotados en $[0, 1]$ con esquema estricto e inmutable (`x94-v1`): nivel 1x1 (1), 2x2 (4), 4x4 (16), 8x8 (64) + 9 asimetrías. Extracción determinista y desacoplada de librerías externas.
- **Comportamiento y límites del descriptor (07):**
  - *Monotonía ante defectos:* la respuesta crece de manera monótona con el tamaño y contraste del defecto en escenas controladas, localizándose en las celdas afectadas.
  - *Efecto del ruido:* el ruido aditivo $\sigma$ eleva falsamente la entropía promedio ($0.661 \to 0.830$) y aplana las asimetrías ($0.0154 \to 0.0095$).
  - *Colisión estructural:* distribuciones espaciales distintas con idéntico histograma global arrojan **exactamente el mismo vector x94** (la entropía de Shannon ignora la disposición interna de píxeles dentro de la celda).
  - *Fronteras de celda:* el descriptor responde con máxima sensibilidad cuando el frente de un defecto cruza el borde de una celda de la rejilla.
- **Estandarización de datos industriales (08):**
  - MVTec-AD requiere letterboxing centrado para preservar la relación de aspecto original sin distorsiones geométricas.
  - Las máscaras de defectos deben interpolarse con `NEAREST` para no introducir niveles espurios fuera de $\{0, 255\}$ y permanecer perfectamente alineadas con las imágenes.
  - La inversión de intensidad (`255 - x`) debe aplicarse antes del letterbox para que el padding conserve el valor $0$ (negro).

---

## Qué falta comprender o comprobar (Gaps hacia el paper)

1. **Capacidad predictiva real:** evaluar si `x94` separa defectos industriales reales en MVTec-AD mediante clasificadores clásicos (regresión logística, Random Forest, SVM) y MLP ligeros.
2. **Ablación de componentes:** cuantificar el aporte real de las 85 entropías solas vs. las 9 asimetrías solas vs. el vector combinado 94.
3. **Mecanismos de selección:** diseñar selectores basados en entropía local (top-k / threshold) para descartar regiones homogéneas y alimentar las CNN de [04](04_lightweight_cnns/README.md).
4. **Evaluación de coste total ($T_{\text{total}}$):** verificar experimentalmente si seleccionar parches reduce el tiempo total de inferencia o si el coste del selector ($T_{\text{selector}}$) supera el ahorro del modelo ($T_{\text{model}}$).

---

## Reproducibilidad y ejecución

Los scripts de comprensión usan rutas relativas al proyecto y fijan semillas reproducibles (20260928 en 02/03; 20261001 en 05/06/07; 42 en 04).

Para ejecutar los laboratorios individuales o sus tests manuales:
```powershell
# Pruebas unitarias de componentes
.\code\.venv\Scripts\python.exe .\code\understanding\03_entropy_benchmark\test_entropy.py
.\code\.venv\Scripts\python.exe .\code\understanding\04_lightweight_cnns\tests\test_models.py
.\code\.venv\Scripts\python.exe .\code\understanding\05_symmetry_features\tests\test_symmetry_features.py
.\code\.venv\Scripts\python.exe .\code\understanding\06_entropy_pyramid_descriptor\tests\test_descriptor.py
.\code\.venv\Scripts\python.exe .\code\understanding\07_descriptor_behavior\tests\test_behavior_helpers.py

# Demos y benchmarks
.\code\.venv\Scripts\python.exe .\code\understanding\06_entropy_pyramid_descriptor\run_demo.py
.\code\.venv\Scripts\python.exe .\code\understanding\07_descriptor_behavior\run_all.py

# Verificación de preprocesamiento MVTec-AD
.\code\.venv\Scripts\python.exe .\code\understanding\08_verify_mvtec_preprocessing\verify.py
```
