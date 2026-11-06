# Resultados de Clasificación Supervisada, Ablación y Explicabilidad ($\mathbf{x}_{94} = [\mathbf{x}_{85}, \mathbf{x}_9]$)

Este directorio contiene la evidencia experimental cuantitativa (`.json`, `.csv`) y visual (`.png`) generada por `experiments/evaluate_descriptors.py` sobre los 4 datasets tabulares completos (`kolektorsdd2`, `kolektorsdd2-inverted`, `mvtec-ad` y `mvtec-ad-inverted`).

---

## 1. Estructura de Carpetas y Artefactos

| Carpeta / Archivo | Contenido |
|---|---|
| `kolektorsdd2/full_rf/`, `kolektorsdd2/full_knn/` | Resultados sobre KolektorSDD2 estándar (`split_mode = "presplit"`, Train = 2,331, Test = 1,004). |
| `kolektorsdd2-inverted/full_rf/`, `kolektorsdd2-inverted/full_knn/` | Resultados sobre KolektorSDD2 invertido ($255 - I$ antes de letterbox). |
| `mvtec-ad/<objeto>_rf/`, `mvtec-ad/bottle_knn/` | Resultados por categoría sobre las 15 clases de MVTec-AD estándar (`split_mode = "stratified"`, 80% train / 20% test en memoria, `seed = 42`, excluyendo `ground_truth`). |
| `mvtec-ad-inverted/<objeto>_rf/`, `mvtec-ad-inverted/bottle_knn/` | Resultados por categoría sobre las 15 clases de MVTec-AD invertido. |
| `mvtec-ad/mvtec_all_objects_rf_summary.csv` y `.png` | Tabla consolidada y gráfica comparativa de las 15 categorías de MVTec-AD estándar. |
| `mvtec-ad-inverted/mvtec_all_objects_rf_summary.csv` y `.png` | Tabla consolidada y gráfica comparativa de las 15 categorías de MVTec-AD invertido. |

Dentro de cada subcarpeta individual (`<objeto_o_full>_<rf|knn>/`) se encuentran 8 archivos:
1. `metrics_summary.json`: Conteos por clase, métricas del descriptor activo, hallazgos de explicabilidad y tiempos desglosados (`T_data_load_s`, `T_model_train_s`, `T_model_inference_s`, `inference_ms_per_sample`).
2. `ablation_table.csv`: Comparación de `x85_only`, `x9_only`, `x94_full`, `E_1x1`, `E_2x2`, `E_4x4` y `E_8x8`.
3. `feature_importance.csv`: Ranking completo de las 94 características con su peso bruto y porcentaje normalizado (`share_pct`).
4. `scale_and_family_breakdown.csv`: Importancia agregada y media por característica para cada familia (`x85_entropy` vs `x9_asymmetry`), escala (`1x1..8x8`) y eje de simetría (`lr`, `tb`, `rot180`).
5. `confusion_matrix_and_curves.png`: Matriz de confusión + Curva ROC + Curva Precision-Recall.
6. `descriptor_ablation_x85_x9_x94.png`: Gráfica de barras agrupadas de ablación y ganancia de simetría $\Delta\text{F1}(\mathbf{x}_{94} - \mathbf{x}_{85})$.
7. `explainability_importance_breakdown.png`: Top-15 variables individuales, contribución total por escala y eficiencia media por característica.
8. `spatial_importance_heatmaps.png`: Mapas de calor 2D ($2\times2$, $4\times4$, $8\times8$) proyectados sobre el lienzo de $128\times128$ píxeles.

---

## 2. Diagnóstico de las Matrices de Confusión: ¿Por qué la esquina inferior derecha (`defect`) a veces da `0` o valores bajos?

### 2.1 Convención de celdas en `confusion_matrix_and_curves.png`
Siguiendo el estándar de `scikit-learn` (`good = 0` clase negativa en la primera fila/columna; `defect = 1` clase positiva en la segunda fila/columna):
- **Arriba-Izquierda (`[0, 0]`, etiquetada `TN`):** Piezas **`good` clasificadas correctamente como `good`**. En todos los experimentos esta celda tiene un acierto entre **`90.4%` y `100.0%`**. *(Nota: si se lee la tabla con la convención donde la primera celda `[0, 0]` se llama "True Positive de la clase 0", entonces la esquina inferior derecha `[1, 1]` corresponde a la detección de la clase 1 `defect`).*
- **Abajo-Derecha (`[1, 1]`, etiquetada `TP`):** Piezas **`defect` detectadas correctamente como `defect`**.

### 2.2 De dónde salen las probabilidades (`predict_proba`) y por qué la celda `[1, 1]` (`defect`) cae a `0` en `transistor` y `grid` bajo el umbral realista de `0.50`
En un `RandomForestClassifier` compuesto por $T = 200$ árboles (`n_estimators = 200`), cada árbol individual $t$ evalúa la muestra $\mathbf{x}_{94}$ hasta llegar a un nodo hoja y emite una probabilidad local $p_t(\text{defect} \mid \mathbf{x})$ igual a la fracción (ponderada por `class_weight="balanced"`) de muestras defectuosas de entrenamiento en esa hoja. La probabilidad final que entrega `model.predict_proba(X_test)[:, 1]` (y con la cual se grafican las curvas ROC y Precision-Recall) es el **promedio aritmético de los 200 árboles**:
$$\hat{p}(\text{defect} \mid \mathbf{x}) = \frac{1}{200} \sum_{t=1}^{200} p_t(\text{defect} \mid \mathbf{x})$$
Por ejemplo, si 76 de los 200 árboles votan por `defect` y 124 por `good`, la probabilidad asignada a esa pieza es $\hat{p} = 76 / 200 = 0.38$. Para construir la **Matriz de Confusión**, `.predict(X_test)` aplica la regla estándar y realista de mayoría simple: **$\hat{p} \ge 0.50 \Rightarrow \text{defect (1)}$** y **$\hat{p} < 0.50 \Rightarrow \text{good (0)}$**.

En categorías con fuerte desbalance de clases y pocos ejemplos (por ejemplo, `transistor` tiene solo `32` defectos frente a `218` piezas buenas en entrenamiento = `12.8%`; `kolektorsdd2` tiene `246` defectos frente a `2,085` buenas = `10.6%`), como cada árbol del Random Forest solo puede elegir entre $\sqrt{94} \approx 9$ columnas al azar en cada nodo (`max_features="sqrt"`), muchos árboles dividen sobre parches de fondo sano y votan una probabilidad cercana a la tasa base ($\approx 0.12$). Eso diluye el promedio de los 200 árboles para las piezas defectuosas al rango **`[0.16, 0.48]`**, mientras que las piezas buenas (`good`) reciben probabilidades muy bajas en **`[0.00, 0.12]`**:
- En **`transistor`** (63 muestras de test: 55 `good` y 8 `defect`): las 55 piezas `good` obtienen probabilidades entre `0.00` y `0.43` (mediana `0.09`), y las 8 piezas `defect` obtienen probabilidades entre `0.16` y `0.45` (mediana `0.38`). El modelo **sí distingue ambas clases (`ROC-AUC = 0.9386`, `PR-AUC = 0.7569`)**, pero como ninguna supera el corte estándar `0.50`, la celda inferior derecha queda en `0/8` (`TP = 0`).
- **Nota sobre árboles de decisión individuales (`DecisionTreeClassifier`) frente a `RandomForest`:** Un solo `DecisionTreeClassifier` inspecciona las **94 columnas completas (`max_features=None`)** en cada corte sin diluir su predicción entre 200 árboles aleatorios; por ello, en objetos con pocos defectos y anomalías muy localizadas en 2–3 celdas de `E_8x8`, un árbol individual o un bosque con `max_features=None` produce decisiones binarias directas $\{0, 1\}$ que elevan los `TP` bajo el umbral realista de `0.50`.

### 2.3 Diagnóstico de Separabilidad Interna (Qué pasaría si el umbral no estuviera fijo en `0.50` — Solo como referencia diagnóstica)

El pipeline mantiene estrictamente el **umbral realista $\tau = 0.50$** en todos los reportes. Únicamente como verificación diagnóstica para confirmar que las probabilidades internas $\hat{p}$ sí contienen señal discriminativa cuando `TP = 0`, la siguiente tabla compara el corte fijo `0.50` frente al umbral $\tau^*$ de las predicciones *Out-Of-Bag* de entrenamiento:

| Dataset / Objeto | Matriz @ $\tau = 0.50$ (`TN, FP / FN, TP`) | F1 @ `0.50` | Umbral OOB $\tau^*$ | Matriz @ $\tau^*$ (`TN, FP / FN, TP`) | F1 @ $\tau^*$ | BalAcc @ $\tau^*$ | ROC-AUC |
|---|---|---:|---:|---|---:|---:|---:|
| **`kolektorsdd2`** | `TN=884, FP=10 / FN=86, TP=24` | `0.3333` | `0.37` | `TN=857, FP=37 / FN=67, TP=43` | **`0.4526`** | **`0.6748`** | `0.8219` |
| **`transistor`** | `TN=55, FP=0 / FN=8, TP=0` | `0.0000` | `0.23` | `TN=48, FP=7 / FN=2, TP=6` | **`0.5714`** | **`0.8114`** | `0.9386` |
| **`toothbrush`** | `TN=15, FP=0 / FN=5, TP=1` | `0.2857` | `0.47` (`0.30` prior) | `TN=15, FP=0 / FN=1, TP=5` (@`0.30`) | **`0.9091`** | **`0.9167`** | `0.9889` |
| **`grid`** | `TN=58, FP=0 / FN=11, TP=0` | `0.0000` | `0.20` | `TN=34, FP=24 / FN=5, TP=6` | **`0.2927`** | **`0.5658`** | `0.6332` |
| **`wood`** | `TN=53, FP=1 / FN=10, TP=2` | `0.2667` | `0.24` | `TN=39, FP=15 / FN=4, TP=8` | **`0.4571`** | **`0.6944`** | `0.7654` |
| **`carpet`** | `TN=61, FP=1 / FN=14, TP=4` | `0.3478` | `0.20` | `TN=24, FP=38 / FN=2, TP=16` | **`0.4444`** | **`0.6380`** | `0.7849` |
| **`cable`** | `TN=57, FP=0 / FN=13, TP=5` | `0.4348` | `0.29` | `TN=40, FP=17 / FN=4, TP=14` | **`0.5714`** | **`0.7398`** | `0.8255` |
| **`metal_nut`** | `TN=48, FP=0 / FN=11, TP=8` | `0.5926` | `0.32` | `TN=39, FP=9 / FN=5, TP=14` | **`0.6667`** | **`0.7747`** | `0.8596` |
| **`pill`** | `TN=59, FP=0 / FN=11, TP=17` | `0.7556` | `0.33` | `TN=51, FP=8 / FN=4, TP=24` | **`0.8000`** | **`0.8608`** | `0.9582` |
| **`capsule`** | `TN=47, FP=2 / FN=9, TP=13` | `0.7027` | `0.41` | `TN=44, FP=5 / FN=5, TP=17` | **`0.7727`** | **`0.8353`** | `0.9100` |
| **`bottle`** | `TN=45, FP=1 / FN=5, TP=8` | `0.7273` | `0.44` | `TN=45, FP=1 / FN=5, TP=8` | **`0.7273`** | **`0.7968`** | `0.8980` |
| **`leather`** | `TN=51, FP=5 / FN=5, TP=13` | `0.7222` | `0.59` | `TN=54, FP=2 / FN=6, TP=12` | **`0.7500`** | **`0.8155`** | `0.9395` |
| **`zipper`** | `TN=51, FP=4 / FN=5, TP=19` | `0.8085` | `0.37` | `TN=47, FP=8 / FN=1, TP=23` | **`0.8364`** | **`0.9064`** | `0.9424` |
| **`tile`** | `TN=50, FP=3 / FN=3, TP=14` | `0.8235` | `0.38` | `TN=49, FP=4 / FN=1, TP=16` | **`0.8649`** | **`0.9329`** | `0.9711` |

---

## 3. ¿Conviene usar `normal` o `inverted` (`--inverted`) en cada dataset?

El análisis directo de los archivos CSV (`*-entropy-full.csv` vs `*-inverted-entropy-full.csv`) y de las matrices de confusión revela un hallazgo matemático fundamental sobre la **entropía de Shannon frente a la inversión fotométrica ($I_{\text{inv}} = 255 - I$)**:

### 3.1 Por qué en MVTec-AD `normal` e `inverted` dan resultados casi idénticos (`4,770 / 6,612` filas exactamente iguales)
1. **Propiedad matemática:** Con `bins = 256` sobre intensidades enteras $[0, 255]$, invertir una imagen ($k \mapsto 255 - k$) únicamente **invierte el orden de las 256 barras del histograma** ($p'_k = p_{255-k}$). Como la suma de la entropía de Shannon es conmutativa:
   $$H(255 - I) = -\sum_{k=0}^{255} p_{255-k}\log_2 p_{255-k} = -\sum_{j=0}^{255} p_j\log_2 p_j = H(I)$$
   **la entropía de Shannon (y por tanto también $\mathbf{x}_9$) es matemáticamente invariante al negativo fotométrico** siempre que no exista relleno (*letterboxing*) mezclado con la imagen.
2. **En MVTec-AD todas las imágenes nativas son cuadradas ($1024\times1024$, $900\times900$, $840\times840$):** No se añade ni un solo píxel de relleno negro (`needs_padding = False`). La única diferencia minúscula (`máx = 0.0092`, media $= 9.1\times 10^{-6}$) proviene del redondeo `.5` del filtro `LANCZOS` al reescalar a $128\times128$.
3. **Diferencias prácticas en MVTec-AD entre `normal` e `inverted`:**
   - **12 de los 15 objetos tienen exactamente la misma matriz de confusión** en `normal` e `inverted`.
   - En **2 objetos gana ligeramente `inverted`**:
     - `zipper`: `inverted` logra `TP = 20/24` (`F1 = 0.8333`, `BalAcc = 0.8803`) vs `normal` `TP = 19/24` (`F1 = 0.8085`).
     - `leather`: `inverted` reduce falsos positivos (`FP = 4`, `F1 = 0.7429`, `ROC-AUC = 0.9415`) vs `normal` (`FP = 5`, `F1 = 0.7222`, `ROC-AUC = 0.9395`).
   - En **1 objeto gana `normal`**:
     - `carpet`: `normal` logra `TP = 4/18` (`F1 = 0.3478`) vs `inverted` `TP = 3/18` (`F1 = 0.2727`).
   - **Recomendación para MVTec-AD:** **Usar `normal` como configuración canónica** (y reportar en el paper que $\mathbf{x}_{94}$ posee **invarianza teórica y práctica a la inversión de polaridad de iluminación $255 - I$** en imágenes sin padding, una ventaja analítica que las CNNs sin aumento de datos no tienen).

### 3.2 Por qué en KolektorSDD2 `normal` e `inverted` SÍ difieren (`643 / 3,335` filas cambian hasta en `0.248`) y cuál conviene elegir
1. **Causa física (interacción entre inversión y *letterboxing*):**
   En `src/preprocessing/kolektorsdd2.py`, la inversión `255 - I` se aplica a la tira metálica rectangular **antes** de centrarla sobre el lienzo negro `pad_value = 0` de $128\times128$. Por ello, en las celdas que pisan simultáneamente borde metálico y relleno negro (`E_1x1`, `E_2x2` y las columnas de transición `c1, c6` de `E_8x8`):
   - En **`normal`**, los píxeles oscuros del metal (` intensidad 0`) se fusionan en el `bin 0` con el relleno negro (`pad_value = 0`).
   - En **`inverted`**, los píxeles brillantes de reflejo especular (`255 -> 0`) son los que se fusionan en el `bin 0` con el relleno negro, cambiando la entropía de los parches globales/fronterizos.
2. **Comparación medida en KolektorSDD2 (`Random Forest`):**
   - **`kolektorsdd2` (Normal):** Mayor sensibilidad en el corte estándar `0.50` (**`TP = 24/110`, `FP = 10`, `F1 = 0.3333`, `BalAcc = 0.6035`**), `ROC-AUC = 0.8219`, `PR-AUC = 0.4906`.
   - **`kolektorsdd2-inverted` (Invertido):** Más conservador en el corte `0.50` (`TP = 18/110`, `FP = 8`, `F1 = 0.2647`), pero logra **mejor separabilidad global de probabilidades (`ROC-AUC = 0.8383` vs `0.8219` y `PR-AUC = 0.5019` vs `0.4906`)**.
   - **Recomendación práctica para la investigación y el paper (KolektorSDD2 y MVTec-AD):**
     1. **Usar `normal` como configuración principal cuando el objetivo prioritario es detectar defectos (`TP` y `F1` bajo el umbral realista `0.50`):** En **KolektorSDD2**, `normal` detecta un **33% más de defectos verdaderos (`24/110` vs `18/110`)** y logra mayor `F1` (`0.3333` vs `0.2647`); en **MVTec-AD**, `normal` iguala o supera a `inverted` en 13 de las 15 categorías.
     2. **Reportar también `inverted` como estudio de ablación/invarianza en el paper:**
        - En **MVTec-AD** (imágenes cuadradas sin letterbox), mostrar que `normal` e `inverted` arrojan resultados prácticamente idénticos sirve como **prueba experimental de que el descriptor $\mathbf{x}_{94}$ es invariante a la inversión fotométrica ($255 - I$)**, a diferencia de las CNNs cuyos filtros dependen del signo del gradiente de intensidad.
        - En **KolektorSDD2** (imágenes rectangulares con letterbox negro conservado siempre en `0`), mostrar ambos permite discutir cómo mantener `pad_value = 0` fijo tras invertir la pieza hace que `normal` tenga mayor sensibilidad directa (`TP = 24`) mientras que `inverted` mejora ligeramente el ranking global (`ROC-AUC = 0.8383` vs `0.8219`) al evitar que las sombras oscuras del metal se mezclen en el `bin 0` con el relleno negro.

---

## 4. Resumen Ejecutivo de los Hallazgos de Explicabilidad ($\mathbf{x}_{85}$ vs $\mathbf{x}_9$)

1. **Aporte de las 9 características de simetría ($\mathbf{x}_9$):**
   - Representando solo el `9.57%` de las columnas (`9/94`), concentran el **`34.98%` de la importancia total en KolektorSDD2** (con las 7 variables Top-1 a Top-7 siendo todas `asym_*`), el **`58.63%` en `tile` ($\Delta\text{F1} = +0.6235$)**, el **`37.28%` en `bottle` ($\Delta\text{F1} = +0.2010$)**, el **`18.90%` en `carpet` ($\Delta\text{F1} = +0.3478$)** y el **`16.14%` en `zipper`**.
2. **Mejor escala individual de entropía ($\mathbf{x}_{85}$):**
   - **`E_4x4` (16 características, parches de $32\times32$ px):** Es la escala individual ganadora en **KolektorSDD2 (`F1 = 0.4000`)** y en **9 de los 15 objetos de MVTec-AD** (`bottle`, `cable`, `capsule`, `hazelnut`, `metal_nut`, `pill`, `tile`, `transistor`, `wood`).
   - **`E_8x8` (64 características, parches de $16\times16$ px):** Gana en defectos locales finos (`screw`, `zipper` y mejor `ROC-AUC = 0.8181` en `KolektorSDD2`).
3. **Descarte automático del Letterboxing (`spatial_importance_heatmaps.png`):**
   - En `kolektorsdd2/full_rf/spatial_importance_heatmaps.png`, las columnas laterales de relleno negro (`c0, c3` en $4\times4$ y `c0, c1, c6, c7` en $8\times8$) reciben exactamente **`0.00%` de importancia**, concentrando el `100%` de las decisiones en la franja metálica central (`c1–c2` / `c2–c5`).
