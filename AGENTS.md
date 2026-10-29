# Project purpose

Este proyecto académico investiga visión computacional eficiente basada en selección de regiones informativas. Estudia entropía visual y local, patches, selección de regiones, histogramas, varianza, bordes (Sobel), selectores top-k y por threshold, clasificadores ligeros, MLP, CNN ligeras, eficiencia computacional y posible ejecución futura en hardware embebido. El objetivo es producir experimentos reproducibles y potencialmente un paper académico.

## Operating environment

- Operating system: Windows 10
- Shell preference: PowerShell
- Python: 3.11
- Python environment: `code/.venv`
- Main ML framework: PyTorch

No instalar paquetes globalmente: todo paquete Python debe instalarse dentro de `code/.venv`. No cambiar la versión de Python sin razón explícita. No introducir TensorFlow salvo solicitud expresa. PyTorch es el framework principal; comprobar la disponibilidad real de GPU antes de asumir CUDA.

Este equipo usa NVIDIA GeForce GTX 1050 (Pascal, compute capability 6.1). Usar builds oficiales PyTorch CUDA 12.x compatibles con Pascal, preferentemente `cu126`; no seleccionar CUDA 13.x. Las wheels incluyen el runtime: no instalar CUDA Toolkit global para ejecutar este proyecto. Usar la ruta explícita `code/.venv/Scripts/python.exe` y ejecutar una sola instalación pip a la vez; esperar a que termine aunque la wheel tarde varios minutos. El test del entorno exige ejecución real CUDA y debe fallar si no está disponible.

## Directory responsibilities

- `code/src/`: código formal, modular, reutilizable, claro, correcto, testeable y reproducible.
- `code/experiments/`: ablaciones, benchmarks, comparaciones, barridos y estudios de selectores, patch size, thresholds, MLP y CNN. Los scripts deben producir resultados reproducibles.
- `code/understanding/`: laboratorio personal. Se permite reimplementar algoritmos, imprimir valores intermedios, visualizar matrices/kernels/histogramas, reconstruir operaciones y comparar con librerías. Aquí rige `understanding > optimization`; no es producción. No mover automáticamente código a `src/`: hacerlo solo cuando esté entendido, validado y sea necesario para el pipeline formal.
- `code/notebooks/` (opcional, actualmente ausente): crear solo cuando haya un notebook concreto de análisis exploratorio; las funciones reutilizables deben migrar a `src/` cuando estén validadas y sean necesarias.
- `code/tests/`: pruebas automáticas; ejecutar las relevantes antes de dar por terminada una modificación importante.
- `code/results/`: resultados generados; no mezclar código y evitar versionar archivos grandes innecesariamente.

## Coding principles

1. Preferir implementaciones simples y evitar abstracciones innecesarias.
2. No añadir dependencias si NumPy, PyTorch, SciPy, OpenCV o scikit-image resuelven el problema.
3. Mantener funciones pequeñas y con responsabilidades claras.
4. Documentar dimensiones no obvias, por ejemplo `[B, C, H, W]`, `[N, patch_height, patch_width]` y `[N, features]`.
5. Usar type hints y docstrings en funciones públicas o científicamente relevantes.
6. Evitar constantes mágicas y permitir controlar seeds cuando exista aleatoriedad.
7. Priorizar la reproducibilidad experimental.

## Scientific integrity

No fabricar resultados, inventar métricas o referencias, ni modificar resultados para ajustar hipótesis. Diferenciar resultados medidos de esperados, registrar parámetros, usar seeds explícitos, documentar decisiones metodológicas y mantener baselines comparables. Al optimizar, conservar una referencia o mecanismo de validación cuando sea razonable.

## Performance claims

Distinguir `selector time`, `model inference time` y `total pipeline time`: `T_total = T_selector + T_model`. Procesar menos patches no implica automáticamente menor coste total. Registrar cuando corresponda número de patches, porcentaje procesado, parámetros, FLOPs/MACs, RAM, VRAM, latencia y throughput.

## Understanding-first rule

Si no se comprende un concepto, algoritmo o transformación, crear primero una explicación o script en `code/understanding/`, no una versión sofisticada en `src/`. Para entropía, inspeccionar `image → patch → histogram → counts → probabilities → p * log2(p) → entropy` y solo después optimizar.

## Visualization rule

En `understanding/`, favorecer matplotlib para visualizar imagen original, patches, coordenadas, matrices, kernels, histogramas, valores intermedios, mapas de entropía, gradientes y regiones seleccionadas.

## Agent behavior

Inspeccionar antes de reemplazar; hacer cambios pequeños y verificables; no borrar trabajo; no reestructurar por una tarea pequeña; ejecutar pruebas tras cambios relevantes; informar y corregir fallos propios; actualizar `requirements.txt` y documentación cuando cambie la ejecución. No crear arquitectura innecesaria.

## Directrices Generales de Programación y Arquitectura

Para garantizar la calidad, mantenibilidad, reproducibilidad y escalabilidad del código en todos los proyectos, los agentes deben adherirse estrictamente a las siguientes prácticas:

**1. Gestión de Configuración y Rutas (Determinismo y Flexibilidad)**
*   **Archivos TOML:** Toda la configuración externa (rutas de datasets, hiperparámetros, definiciones de entorno) debe definirse en archivos `.toml` dentro del directorio `configs/`.
*   **Separación Estricta:** El directorio `src/` está reservado *exclusivamente* para la lógica de negocio, algoritmos y procesamiento. Ningún parámetro de configuración ni valor predeterminado debe estar *hardcodeado* en el código fuente.
*   **Resolución Dinámica de Rutas:** Queda estrictamente prohibido el uso de rutas absolutas locales o relativas dependientes del directorio de trabajo actual (CWD). Se debe utilizar la librería estándar `pathlib.Path`. La ruta raíz del proyecto (Project Root) debe calcularse dinámicamente en los puntos de entrada (ej. `Path(__file__).resolve().parents[...]`), y a partir de ahí anclar y resolver todas las rutas relativas leídas de las configuraciones.

**2. Código Limpio (Clean Code) y Tipado**
*   **Funciones de Responsabilidad Única:** Las funciones deben ser concisas, descriptivas y realizar una sola tarea bien definida. El código debe ser altamente legible, priorizando nombres que expresen intención.
*   **Tipado Estricto (Type Hints):** Uso obligatorio de `from __future__ import annotations` y anotaciones de tipos exhaustivas en todas las firmas de funciones, retornos y variables complejas para facilitar el análisis estático.
*   **Inmutabilidad (Data Classes):** Para estructuras de datos, configuraciones o estados que no deban modificarse en tiempo de ejecución, utilizar `@dataclass(frozen=True, slots=True)` para garantizar la inmutabilidad y optimizar el uso de memoria.

**3. Arquitectura y Patrones (SOLID)**
*   **Principio de Responsabilidad Única (SRP):** Cada clase, módulo o función debe tener un único propósito y una sola razón para cambiar.
*   **Patrones de Diseño:** Implementar patrones de diseño estructurados (ej. Factory, Strategy, Builder) de manera pragmática cuando la complejidad de la lógica (como carga de distintos datasets o pipelines de evaluación) lo justifique.
*   **Testabilidad:** El código debe ser inherentemente fácil de probar. Evitar acoplamientos fuertes; favorecer la inyección de dependencias (como pasar las rutas de configuración por argumento en lugar de instanciarlas globalmente).

**4. Diseño de Interfaces de Línea de Comandos (CLI) y Ejecución**
*   **Ejecución Modular:** Los scripts deben estar diseñados para ser ejecutados como módulos de Python (`python -m carpeta.modulo`).
*   **Contratos CLI Claros:** Utilizar `argparse` para definir explícitamente los argumentos, valores por defecto e instrucciones de uso. Aislar la configuración del *parser* en su propia función.
*   **Puntos de Entrada Aislados:** La ejecución debe encapsularse en una función `main(argv: list[str] | None = None) -> int`. Esto permite probar la lógica de la CLI inyectando listas de argumentos sin depender de `sys.argv`. Finalizar el archivo con el bloque `if __name__ == "__main__": raise SystemExit(main())`.

**5. Programación Defensiva y Trazabilidad**
*   **Fail Fast:** Validar entradas y configuraciones tempranamente. Verificar la existencia de claves requeridas o tipos de datos correctos antes de ejecutar procesos costosos.
*   **Salidas Limpias (Manejo de Errores):** Capturar excepciones previsibles (como `OSError` o `ValueError`) en el nivel superior (`main()`) y abortar el programa devolviendo mensajes concisos y accionables (ej. `raise SystemExit(f"error: {exc}") from exc`). No exponer *tracebacks* internos al usuario a menos que sea un error crítico no manejado.
*   **Logging vs Print:** Utilizar un sistema estructurado (ej. módulo `logging` de Python) para la trazabilidad del flujo del programa (`logger.info`, `logger.error`). El uso de `print` queda reservado estrictamente para salidas CLI interactivas o de consulta directa (ej. listar banderas de ayuda o catálogos).

## Definition of done

Una tarea no termina al escribir archivos: cuando corresponda verificar `implementation → execution → tests → expected output`. Indicar explícitamente cualquier parte no verificada.
