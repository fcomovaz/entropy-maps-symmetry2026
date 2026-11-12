# Lab 13 — Optimización de Mapas de Entropía Multiescala (`task_09`)

> **Estado actual:** Implementado, compilado con MSVC (`19.51.36244` x64), optimizado y verificado experimentalmente sobre `real:mvtec-ad` (`n=200`, `128x128`, `[1,2,4,8]`, `bins=16`, CPU `threads=1`). Todos los tests del laboratorio (`13/13`), la suite completa del repositorio (`195 tests`, `1 skipped` opcional de KolektorSDD2), la paridad numérica (`28/28` checks `<= 1.89e-15`) y las 8 figuras en `outputs/` han sido validados.

---

## 1. Diagnóstico del Problema (`src/features/entropy_maps.py`)

El descriptor formal actual (`compute_entropy_features` en `src/features/entropy_maps.py`) **ya precalcula las coordenadas geométricas** (`EntropyPlan.regions`, 85 cajas `RegionSpec` para niveles `[1, 2, 4, 8]`) y los Lookup Tables de cuantización y entropía.

Sin embargo, el perfilado previo (`REVIEW_01.md`) mostró que `compute_entropy_features` sigue iterando en Python sobre las 85 regiones individuales:
1. Cada iteración hace un *slice* no contiguo (`quantized[:, y0:y1, x0:x1]`), copia/cast a `int32`, cálculo de offsets e invocación separada a `np.bincount`, lookup `term_lut[counts]`, suma y regla equiprobable.
2. En `batch_size = 1`, el costo queda dominado por la latencia fija de despacho por llamada (~85 micro-llamadas NumPy + gap de despacho Python).
3. Cada píxel de la imagen cuantizada se lee **4 veces** (una por escala `1x1`, `2x2`, `4x4`, `8x8`), aunque los histogramas por conteo entero son **estrictamente aditivos**.

### Idea central: paso único en grilla más fina + pirámide aditiva

Cuando el tamaño de imagen $H = W$ y los niveles anidados satisfacen:
$$\text{size} \bmod g_{\max} = 0 \quad \text{y} \quad \forall g \in \text{levels},\; g_{\max} \bmod g = 0,$$
cada celda de un nivel más grueso $g$ es la unión exacta de un bloque de $k \times k$ celdas hijas de un nivel más fino ($k = g_{\text{hijo}} / g$). Por aditividad exacta de histogramas:
- Se leen los píxeles **1 sola vez** para poblar exclusivamente las celdas de la grilla más fina ($8 \times 8 = 64$ histogramas).
- Los niveles $4 \times 4$, $2 \times 2$ y $1 \times 1$ se obtienen sumando histogramas hijos en memoria ($O(\text{cells} \cdot B)$ enteros, sin volver a leer los píxeles de la imagen).
- Las variantes rápidas **validan explícitamente** la condición de anidamiento (`validate_nested_grid_levels`) y fallan rápido con `ValueError` si no se cumple.

---

## 2. Hipótesis Experimentales

- **H1 (Single-pass + pirámide en NumPy):** Calcular únicamente los histogramas de la grilla más fina en un solo `np.bincount` global y derivar los niveles gruesos mediante sumas por bloques (`V1_numpy_finest_pyramid`) produce una aceleración significativa respecto a `V0_baseline`. La variante `V2_numpy_fused_entropy` evalúa si fusionar todas las escalas en un solo tensor `[N, 85, B]` reduce llamadas adicionales en `batch = 1` lo suficiente como para justificar la complejidad del LUT `c * log2(c)`.
- **H2 (Kernel nativo en C + `ctypes`):** Mover todo el pipeline (`quantize -> hist_fina -> suma_pirámide -> entropía_LUT`) a una DLL compilada con MSVC (`/O2 /arch:AVX2 /fp:precise`) invocada por `ctypes` con buffers preasignados (`V3_c_single_pass`) agrega una mejora sustancial adicional, especialmente en `batch = 1`, al eliminar el overhead de despacho de Python y NumPy.

---

## 3. Variantes Comparadas

| ID | Estrategia | Origen / Archivo | Objetivo |
|---|---|---|---|
| `V0_baseline` | `compute_entropy_features` con `EntropyPlan` | `variants.py` (envuelve `src/features/entropy_maps.py`) | Oráculo / baseline formal |
| `V1_numpy_finest_pyramid` | Cuantización única + `bincount` único en grilla más fina + sumas `reshape(...).sum((2,4))` + entropía por nivel | `variants.py` | Eliminar loop de 85 regiones y 4 lecturas de píxel en NumPy puro |
| `V2_numpy_fused_entropy` | Igual que `V1`, pero concentra los histogramas de todos los niveles en `[N, total_cells, B]` y evalúa la entropía en una sola llamada LUT `nlogn_lut` | `variants.py` | Comprobar si reducir 4 evaluaciones de entropía a 1 ayuda en `batch = 1` |
| `V3_c_single_pass` | DLL C99 compilada con MSVC + `ctypes` (`argtypes` preconfigurados, buffer batch=1 reutilizable, paso único secuencial sin `malloc`) | `native/entropy_pyramid.c`, `native_binding.py` | Eliminar por completo el overhead Python/NumPy por región |

---

## 4. Notas de Diseño Técnico e Ingeniería

- **Cero valores mágicos en código:** Toda ruta, hiperparámetro, tolerancia (`1e-12`), flags de compilador, tamaños de barrido y parámetros de protocolo se leen desde `configs/optimization/entropy_maps.toml` (`config_loader.py`).
- **Fusión cuantización + histograma en C (`px >> shift` vs LUT):** Cuando `bins` es potencia de dos ($2^k \le 256$), $(p \cdot B) // 256 \equiv p \gg (8 - k)$ para `uint8_t`, ejecutándose inline sin memoria auxiliar; para `bins` arbitrarios (e.g., `7`, `13`) se usa `qlut[px]` (256 bytes en L1).
- **Contadores `uint16_t` escalonados en pila + promoción `uint32_t` en pirámide y respaldo fino:** Para evitar llamadas `_chkstk` cuando varía `bins`, `entropy_pyramid.c` separa tres niveles aislados con `EP_NOINLINE`: `small` (`<=1024` slots finos / `<=384` gruesos, `<3.6 KB`, cero `_chkstk`, cubre `[1,2,4,8]` con `bins <= 16`), `medium` (`<=4096` / `<=1536`, `~14 KB`, cubre `[1,2,4,8]` hasta `bins = 64`) y `general` (`256 KB`). El pase fino usa `uint16_t` mientras `npix_finest <= 65535` (p. ej. hasta `512x512` con `g_max = 8`, donde `npix_finest = 4096`), promoviendo inmediatamente a `uint32_t` en `sum_u16_to_u32` para niveles gruesos (`1x1` = `262,144` px en `512x512`) y reservando `run_batch_u32_general` como respaldo cuando `npix_finest > 65535`.
- **Entropía branchless con tabla `c * log2(c)` + reglas analíticas exactas:**
  $$H_{\text{norm}} = \frac{\log_2(N) - \frac{1}{N}\sum_{b=0}^{B-1} \text{nlogn\_lut}[c_b]}{\log_2(B)},$$
  aprovechando `nlogn_lut[0] == 0.0` para eliminar ramas en el lazo interno y preservando exactamente `N = 0 -> 0.0`, celda constante (`c == N`) -> `0.0` y distribución equiprobable (`all c == c0`) -> `1.0` dentro de `1e-12` (`max|diff| <= 1.89e-15`).
- **Alcance real de SIMD (`AVX2`):** Los flags `/arch:AVX2` ayudan principalmente en las sumas vectoriales de histogramas padre (`sum_u16_to_u32`, `sum_u32_to_u32`) y en las reducciones sobre `bins`. En contraste, el incremento aleatorio/scatter `hist[bin]++` del histograma fino está limitado por dependencias de lectura-escritura y latencia de L1, por lo que no se asume aceleración SIMD en esa subetapa.
- **Hilos y protocolo CPU justo:** Todo el benchmark principal corre fijado a `threads = 1` (`threadpoolctl` + `torch.set_num_threads(1)`), `gc.disable()`, `time.perf_counter_ns()`, `warmup_iters = 20`, `timed_iters = 200`, `repeats = 3`, intervalos de confianza bootstrap (95%) y test U de Mann-Whitney ($\alpha = 0.05$). El paralelismo multihilo sobre imágenes (`#pragma omp parallel for`) solo aplica a lotes grandes y queda fuera de la comparación `batch = 1`.
- **Alternativas como Numba/Cython:** `numba` y `cffi` no están instalados y la regla del proyecto evita añadir dependencias a `requirements.txt`. La solución con `cl.exe` + `ctypes` (stdlib) no introduce dependencias y deja código C portable a sistemas embebidos.
- **Parámetros y separación de tiempos frente a CNN PyTorch:** El sistema clásico reporta explícitamente `T_desc` (descriptor de 85 features, 0 parámetros aprendibles), `T_model` (`LogisticRegression`: 86 parámetros en el estimador lineal $+ 170$ de `StandardScaler` si `scale_features = true`, i.e., 256 totales almacenados) y `T_total = T_desc + T_model`, frente a `ds_cnn` y `ultra_tiny` (~1.1k–97k parámetros) evaluados en CPU (`device = "cpu"`, `threads = 1`, `batch = 1`).

---

## 5. Estructura del Laboratorio

```text
configs/optimization/entropy_maps.toml
understanding/13_entropy_maps_optimization/
├── README.md
├── config_loader.py
├── build_native.py
├── variants.py
├── native_binding.py
├── explain_single_pass.py
├── benchmark_entropy_variants.py
├── plot_entropy_optimization.py
├── native/
│   ├── entropy_pyramid.h
│   └── entropy_pyramid.c
├── tests/
│   └── test_entropy_variants.py
└── outputs/
```

---

## 6. Resultados Medidos (`outputs/*.csv` y `outputs/run_manifest.json`)

> Ejecutado el `2026-11-12T00:46–00:47 UTC` (commit `bdbb961`) en Intel Core i7-7700HQ @ 2.80 GHz, Windows 10, `threads = 1`, `gc.disable()`, `real:mvtec-ad` (`n = 200`, `128x128`, `[1,2,4,8]`, `bins = 16`). Paridad numérica aprobada `28/28` checks (`<= 1.89e-15` vs `tolerance_abs = 1e-12`).

### 6.1 Respuestas cuantitativas a las preguntas clave
- **(a) ¿El paso único en grilla más fina mejora respecto al baseline (`V1`/`V2` vs `V0`)?**
  - `V0_baseline` (batch=1): `2.2282 ms` (`448.8 img/s`, IQR `1.4058 ms`, CV `0.456%`, `prepare = 1.3021 ms`, alloc `209.7 KB`)
  - `V1_numpy_finest_pyramid` (batch=1): `0.2371 ms` (**`9.40x`**, IC95 `[8.75x, 9.96x]`, `p = 4.83e-67`, `4,217.6 img/s`, IQR `0.0582 ms`, CV `0.939%`, `prepare = 0.7320 ms`, alloc `216.9 KB`)
  - `V2_numpy_fused_entropy` (batch=1): `0.2125 ms` (**`10.49x` vs `V0`**, IC95 `[9.82x, 11.07x]`, `p = 4.83e-67`, `1.12x` vs `V1`, `4,705.9 img/s`, CV `3.236%`). **Matiz importante:** en lotes grandes (`batch = 8 / 32 / 256`) y en `512x512`, `V2` queda ligeramente por detrás de `V1` (`0.2330 / 0.2577 / 0.2648 ms` vs `0.2215 / 0.2530 / 0.2597 ms`; `3.800 vs 3.395 ms` en `512x512`) por mayor presión de ancho de banda del tensor `[N, 85, B]`. Por ello, para NumPy puro `V1` resulta más simple y consistente en todo el rango.
- **(b) ¿Cuánto agrega el kernel nativo en C (`V3` vs `V1` y `V0`) y cuánto cuesta `ctypes`?**
  - `V3_c_single_pass` (`run()`, batch=1): **`0.02405 ms`** (**`92.65x` vs `V0`**, IC95 `[86.00x, 98.30x]`, `p = 4.81e-67`; **`9.86x` vs `V1`**; **`41,580.0 img/s`**, IQR `0.00468 ms`, CV `1.627%`, `prepare = 0.5573 ms`, alloc `1.27 KB`)
  - `V3_c_single_pass_into` (zero-alloc `run_into()`, batch=1): `0.02555 ms` (`87.21x` vs `V0`, `39,138.9 img/s`, `repeats = 1` diagnóstico)
  - `V3_ctypes_noop_overhead`: `0.00220 ms` (**`2.20 µs`**, ~9.1% de una llamada `V3` `batch = 1`)
  - **Desglose por etapa (`benchmark_stages.csv`, `batch = 1`):** `V0` (`quant 0.0481 + hist 0.8938 + ent 1.1996 = 2.1414 ms`) $\to$ `V1` (`quant 0.0442 + hist 0.1055 + pyr 0.0201 + ent 0.0791 = 0.2489 ms`) $\to$ `V3` end-to-end fusionado (`0.02405 ms`).
  - **Escalado de lote (`batch = 8 / 32 / 256`, ms/img):** `V0 = 0.6522 / 0.5576 / 0.6254 ms`, `V1 = 0.2215 / 0.2530 / 0.2597 ms`, `V3 = 0.01704 / 0.02617 / 0.01860 ms` (hasta `58,673 img/s` en `batch = 8`).
  - **Barridos (`benchmark_sweeps.csv`, `batch = 1`):**
    - `bins = 8 / 16 / 32 / 64` en `128x128`: tras introducir pools escalonados (`medium` + `EP_NOINLINE`) y lazo LUT branchless, `V3` se mantiene plano en **`0.02560 / 0.02595 / 0.02445 / 0.02730 ms`** (`82.6x / 76.9x / 90.7x / 80.6x`; antes de esa mejora `bins = 32` sufría `_chkstk` con `0.04705 ms`).
    - `image_size = 128 / 256 / 512` con `bins = 16`: `V3` escala linealmente en píxeles (`0.02290 / 0.08410 / 0.33610 ms`, aceleración **`85.62x / 39.11x / 25.95x`** sobre `V0`).
- **(c) ¿Se supera el tiempo en CPU de PyTorch (`ds_cnn`, `ultra_tiny`) con menos parámetros?**
  - `CNN_ds_cnn_cpu` (CPU, `batch = 1`, `threads = 1`): `2.5284 ms` (IC95 `[2.3586, 2.6829]`), `12,866` parámetros, `19,366,400` FLOPs.
  - `CNN_ultra_tiny_cpu` (CPU, `batch = 1`, `threads = 1`): `0.8901 ms` (IC95 `[0.8748, 0.9222]`), `1,154` parámetros, `1,335,424` FLOPs.
  - `V3_c_single_pass + LR (dot core)`: `T_desc = 0.02405 ms` + `T_model = 0.00320 ms` = **`T_total = 0.02725 ms`** (**`92.79x` más rápido que `ds_cnn`** y **`32.66x` más rápido que `ultra_tiny`**), **`256` parámetros totales** (`86` de `LogisticRegression` + `170` de `StandardScaler`), `~19,444` operaciones analíticas (`19,104` descriptor + `340` escalado/dot: `85` mul + `85` add + `170` scalar).
  - `V3_c_single_pass + LR (sklearn)`: `T_total = 0.25080 ms` (`0.02405 + 0.22675`, `10.08x` vs `ds_cnn`), exponiendo que **`Pipeline.predict()` de scikit-learn cuesta `0.22675 ms` (9.4x más que todo el descriptor en C)**. En inferencia edge/CPU real debe evitarse el wrapper de `sklearn` y evaluar el producto punto directamente (en NumPy o fusionado en C).

### 6.2 Validación de hipótesis e ítems no verificados
- **Estado de H1:** **Confirmada** (`V1` logra `9.40x` y `V2` logra `10.49x` en `batch = 1`, `p < 1e-66`). Como matiz negativo útil, la fusión completa `V2` no aporta ventaja sostenida sobre `V1` para lotes $\ge 8$ ni para `512x512`.
- **Estado de H2:** **Confirmada** (`V3` logra `92.65x` sobre `V0`, `9.86x` sobre `V1` y `92.79x` extremo-a-extremo frente a `ds_cnn` en CPU con `50.3x` menos parámetros).
- **Ítems no verificados / notas de entorno (`unverified_items = []` en `run_manifest.json`):**
  - Todos los pasos configurados quedaron verificados con ejecución real (`cl.exe` MSVC `19.51.36244` `/O2 /LD /arch:AVX2 /fp:precise`, `compiler_stderr` vacío, `real:mvtec-ad`, `13/13` tests en [test_entropy_variants.py](file:///d:/Publications/2026-10-Symmetry-Entropy/code/understanding/13_entropy_maps_optimization/tests/test_entropy_variants.py) y `195` tests del repositorio con `OK (skipped=1)` en `test_real_workspace_dry_run_if_data_and_checkpoints_present` por ausencia local de CSV/checkpoints de KolektorSDD2).
  - En el perfilado de FLOPs PyTorch (`benchmark_vs_cnn.py`), `torch.utils.flop_counter` emite un aviso estándar `"triton not found"` en Windows; los FLOPs ATen estándar se calcularon con precisión sin afectar métricas.

---

## 7. Preparación para Migración Futura (Stage 9 — NO migrar todavía)

> No mover código a `src/` sin validación completa en pruebas/benchmarks ni aprobación explícita.

1. **Candidato principal para `src/features/entropy_maps.py`:** La lógica validada de la variante ganadora en NumPy (`V1` o `V2` según resultados medidos), integrada detrás de un guard explícito de divisibilidad anidada (`size % g_max == 0` y `g_max % g == 0`) y conservando la implementación actual como respaldo para mallas no anidadas o no divisibles.
2. **Candidato opcional para backend nativo acelerado:** Adaptador tipo `native_binding.py` + `build_native.py` como extensión opcional con carga segura y fallback transparente a NumPy si la DLL no está compilada.
3. **Núcleo C portable a firmware embebido (MCU / SoC sin SO):** Todo `entropy_pyramid.h` / `entropy_pyramid.c` (C99 puro, tipos `stdint.h`, cero recursión, cero `malloc`/`free`, cero headers del SO). En firmware basta almacenar `qlut[256]` y `nlogn_lut` como tablas constantes `const` en Flash/ROM o inicializarlas una sola vez en el arranque del dispositivo.

---

## 8. Comandos Manuales de Ejecución (PowerShell desde `code/`)

```powershell
# 1. Compilar kernel nativo en C (DLL x64 con MSVC)
.\.venv\Scripts\python.exe understanding/13_entropy_maps_optimization/build_native.py

# 2. Laboratorio pedagógico paso a paso + figura explicativa
.\.venv\Scripts\python.exe understanding/13_entropy_maps_optimization/explain_single_pass.py

# 3. Tests unitarios de paridad y contratos (V0-V3 vs oráculo)
.\.venv\Scripts\python.exe understanding/13_entropy_maps_optimization/tests/test_entropy_variants.py

# 4. Suite completa de benchmarks CPU-justos (genera CSV/JSON en outputs/)
.\.venv\Scripts\python.exe understanding/13_entropy_maps_optimization/benchmark_entropy_variants.py --config configs/optimization/entropy_maps.toml

# 5. Generar las 7 gráficas científicas a partir de los CSVs
.\.venv\Scripts\python.exe understanding/13_entropy_maps_optimization/plot_entropy_optimization.py --config configs/optimization/entropy_maps.toml

# 6. Verificar que toda la suite existente del proyecto sigue intacta
.\.venv\Scripts\python.exe -m unittest discover tests -v
```
