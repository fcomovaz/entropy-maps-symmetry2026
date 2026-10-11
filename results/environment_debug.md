# PyTorch CUDA environment debugging

Fecha: 2026-09-26. Proyecto: `D:\Publications\2026-10-Symmetry-Entropy`.

## Initial state

- Windows 10 build 19045, Python 3.11.9 AMD64 / 64 bits.
- Intérprete confirmado: `code\.venv\Scripts\python.exe`; pip 26.2.1 dentro del mismo venv.
- `where.exe python` y `where.exe pip` resolvían primero Python 3.14 global; por eso todos los comandos de reparación usan la ruta explícita del venv. `py -0p` también detectó Python 3.10 y 3.11.
- PyTorch 2.6.0+cpu, torchvision 0.21.0+cpu, `torch.version.cuda=None`, CUDA no disponible. Sin distribuciones duplicadas ni paquetes NVIDIA/CUDA parciales en site-packages; `pip check` limpio.
- NVIDIA-SMI: GTX 1050, driver 582.66, 4096 MiB totales / 113 MiB usados; proceso gráfico `ePowerButton_NB.exe` (PID 8240). Ejecutable: `C:\Windows\System32\nvidia-smi.exe`.
- El driver anuncia compatibilidad máxima CUDA 13.0; esto no es el runtime de PyTorch ni una razón para instalar CUDA 13 en Pascal.
- Espacio libre inicial aproximado: C: 221 GiB (TEMP y caché pip); D: 365 GiB (proyecto). Caché: `%LOCALAPPDATA%\pip\cache`; TEMP: `%LOCALAPPDATA%\Temp`.
- No había configuración pip.ini ni índice personalizado configurado. `pip cache list` no mostró wheels PyTorch.

## Hypotheses and evidence

1. **Build CPU: confirmado.** Explica directamente la ausencia de CUDA; no es una inferencia a partir del driver.
2. **Instalación anterior interrumpida/concurrente: evidencia histórica.** Se revisó el historial de la tarea `Code tree and venv` (01a0d927-5112-7663-86eb-ddf82ac8b039). Los intentos CUDA 2.14.0 y 2.6.0 terminaron tras aproximadamente 339 y 307 segundos, con códigos -1 y 1, respectivamente, mientras descargaban. No se conservó una excepción de red en esas salidas. Los intentos se solaparon temporalmente y el historial contiene después una terminación forzada de procesos Python.
3. **Wheels incompletas: confirmado.** Restos en TEMP: `pip-unpack-73os0dfp/torch-2.14.0+cu126-cp311-cp311-win_amd64.whl` (1,822,949,376 bytes frente a ~2,602.8 MB esperados) y `pip-unpack-lp0v60yo/torch-2.6.0+cu126-cp311-cp311-win_amd64.whl` (1,110,179,840 bytes frente a ~2,496.1 MB). Ambos devolvieron `zipfile.is_zipfile=False`; última escritura 20:55:02. No se reutilizaron ni borraron.
4. **Bloqueo de archivos: confirmado en un intento CPU posterior**, no demostrado como causa de las descargas CUDA. Error determinante recuperado:

   ```text
   ERROR: Could not install packages due to an OSError: [WinError 32] El proceso no tiene acceso al archivo porque está siendo utilizado por otro proceso: 'D:\Publications\2026-10-Symmetry-Entropy\code\.venv\Lib\site-packages\torch\distributed\pipelining\schedules.py'
   ```

5. El historial también contiene `WARNING: Cache entry deserialization failed, entry ignored` y un conflicto NumPy/OpenCV que ya estaba resuelto al comenzar esta reparación. La descarga actual evita la caché con `--no-cache-dir`; no se purgó la caché compartida.
6. Sin evidencia inicial de falta de espacio, incompatibilidad Python/arquitectura, driver roto, DLL ausente o necesidad de CUDA Toolkit global. No atribuir el fallo a Defender, SSL, permisos o red sin evidencia.

## Root cause

La causa inmediata fue instalar y fijar explícitamente una build CPU después de abandonar descargas CUDA incompletas. El test anterior imprimía PASS aun sin CUDA, por lo que ese PASS solo acreditaba importaciones. El mecanismo exacto que terminó cada descarga original no consta en las salidas disponibles; su duración no prueba un timeout de pip. Hubo además instalaciones solapadas y un bloqueo WinError 32 durante el fallback CPU.

## Fix applied

- Actualización solicitada de pip/setuptools/wheel: ya estaban actualizados (26.2.1 / 84.0.0 / 0.48.0).
- Desinstalados únicamente torch y torchvision CPU; torchaudio no estaba instalado. `find_spec('torch')` y `find_spec('torchvision')` devolvieron None; se conservó torchinfo.
- Seleccionados torch 2.14.0+cu126 y torchvision 0.29.0+cu126 desde el índice oficial, con wheels CPython 3.11 Windows AMD64. La [matriz oficial](https://github.com/pytorch/pytorch/blob/main/RELEASE.md) confirma Python 3.11 y Pascal para CUDA 12.6 en Windows; la [publicación 2.14.0](https://github.com/pytorch/pytorch/releases/tag/v2.14.0) y `pip index versions` confirman la versión estable disponible.
- Una sola instalación activa; timeout de socket aumentado preventivamente a 1000 segundos, sin caché. Esto no demuestra que el fallo original fuera un timeout de red.

```powershell
.\code\.venv\Scripts\python.exe -m pip install --no-cache-dir --default-timeout=1000 --progress-bar off --log code/results/pytorch_cuda_install.log torch==2.14.0+cu126 torchvision==0.29.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

## Final verification

- Instalación finalizó con código 0: `Successfully installed sympy-1.14.0 torch-2.14.0+cu126 torchvision-0.29.0+cu126`. No hubo fallos en el intento actual; descarga e instalación completadas en unos seis minutos, sin necesidad de wheel manual.
- Python 3.11.9, ejecutable del venv confirmado; PyTorch 2.14.0+cu126 y torchvision 0.29.0+cu126. `torch.version.cuda='12.6'`, `torch.cuda.is_available()=True`, device count 1.
- NVIDIA GeForce GTX 1050, capability (6, 1), 3.9998779296875 GiB de VRAM según PyTorch. Arquitecturas compiladas: sm_50, sm_60, sm_61, sm_70, sm_75, sm_80, sm_86, sm_90.
- Asignación de dos matrices aleatorias float32 [2048, 2048] en CUDA: PASS. Producto sincronizado en cuda:0: PASS, forma [2048, 2048], valores finitos y bloque [8, 8] contrastado con cálculo CPU (rtol=atol=1e-3). Tiempo observado de una ejecución sin calentamiento: 0.0507281 s; no es un benchmark ni evidencia de aceleración frente a CPU.
- Transferencia CPU → GPU → CPU de [1000, 1000], igualdad exacta: PASS.
- MLP Linear(128,256) → ReLU → Linear(256,10), batch [64,128]: PASS; salida [64,10] en cuda:0 y valores finitos, sin entrenamiento.
- `code/tests/test_environment.py`: PASS (código 0) tras reforzarlo para exigir CUDA y operaciones reales pequeñas. `pip check`: `No broken requirements found.` Sin distribuciones torch duplicadas ni restos `~*` en site-packages.
- Prueba negativa en un subproceso con `CUDA_VISIBLE_DEVICES=-1`: devolvió código 1, `RuntimeError: No CUDA GPUs are available` y `Environment verification: FAIL`, sin imprimir PASS. La variable se aplicó únicamente a ese subproceso, sin modificar el entorno persistente.
- `pip install --dry-run -r code/requirements.txt`: todas las dependencias satisfechas, sin sustitución de la variante CUDA.
- Logs locales: `pytorch_cuda_install.log`, `cuda_verification.log`, `environment_verification.log`. El resumen Markdown se exceptuó de `.gitignore` para poder conservarlo; los logs continúan ignorados. No hay repositorio `.git` inicializado en esta carpeta.

## System changes and limits

Ningún cambio global: sin winget, CUDA Toolkit, driver, Visual C++ Redistributable, PATH global, cambio de Python, reinicio ni recreación del venv. Ningún proceso ajeno fue terminado durante esta reparación. Se conservaron las wheels antiguas incompletas en TEMP y la caché compartida; no forman parte del venv reparado. No se modificó código científico. La reparación demuestra que el driver y las dependencias de Windows existentes bastan para estas pruebas; no prueba la causa exacta de la interrupción de cada descarga histórica.
