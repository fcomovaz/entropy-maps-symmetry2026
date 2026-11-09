"""Hardware, OS, threading backend, library version, and seed provenance capture."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import psutil
import scipy
import sklearn
import threadpoolctl
import torch


@dataclass(frozen=True, slots=True)
class EnvironmentSnapshot:
    """Reproducibility snapshot of hardware, OS, runtime libraries, and thread pool configuration."""

    timestamp_utc: str
    git_commit: str
    seed: int
    python_version: str
    os_platform: str
    cpu_model: str
    cpu_architecture: str
    physical_cores: int | None
    logical_cores: int | None
    cpu_freq_current_mhz: float | None
    cpu_freq_max_mhz: float | None
    ram_total_gb: float
    power_plan: str
    torch_version: str
    cuda_available: bool
    gpu_name: str | None
    numpy_version: str
    scipy_version: str
    sklearn_version: str
    pandas_version: str
    psutil_version: str
    threadpoolctl_version: str
    torch_num_threads: int
    torch_num_interop_threads: int
    torch_parallel_info: str
    threadpool_backends: list[dict[str, Any]]


def _detect_cpu_model() -> str:
    """Read human-readable CPU brand string on Windows when available, falling back cleanly."""
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                if isinstance(val, str) and val.strip():
                    return val.strip()
        except OSError:
            pass
    proc = platform.processor()
    return proc.strip() if proc and proc.strip() else "unknown"


def _query_power_plan() -> str:
    """Query active Windows power plan safely with short timeout, never raising."""
    if sys.platform != "win32":
        return "not_applicable_non_windows"
    try:
        completed = subprocess.run(
            ["powercfg", "/getactivescheme"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        out = (completed.stdout or "").strip()
        return out if out else "unavailable"
    except Exception:
        return "unavailable"


def query_git_commit(project_root: Path) -> str:
    """Return short git commit hash when repository exists, otherwise 'not_a_git_repo'."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            return completed.stdout.strip()
        return "not_a_git_repo"
    except Exception:
        return "not_a_git_repo"


def capture_environment(project_root: Path, seed: int) -> EnvironmentSnapshot:
    """Capture current host hardware, OS, library versions, thread pools, and seed."""
    freq = psutil.cpu_freq()
    vm = psutil.virtual_memory()
    tp_info = [dict(entry) for entry in threadpoolctl.threadpool_info()]

    return EnvironmentSnapshot(
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=query_git_commit(project_root),
        seed=int(seed),
        python_version=platform.python_version(),
        os_platform=platform.platform(),
        cpu_model=_detect_cpu_model(),
        cpu_architecture=platform.machine(),
        physical_cores=psutil.cpu_count(logical=False),
        logical_cores=psutil.cpu_count(logical=True),
        cpu_freq_current_mhz=round(float(freq.current), 2) if freq is not None else None,
        cpu_freq_max_mhz=round(float(freq.max), 2) if freq is not None and freq.max > 0 else None,
        ram_total_gb=round(float(vm.total) / (1024.0 ** 3), 3),
        power_plan=_query_power_plan(),
        torch_version=str(torch.__version__),
        cuda_available=bool(torch.cuda.is_available()),
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        numpy_version=str(np.__version__),
        scipy_version=str(scipy.__version__),
        sklearn_version=str(sklearn.__version__),
        pandas_version=str(pd.__version__),
        psutil_version=str(psutil.__version__),
        threadpoolctl_version=str(getattr(threadpoolctl, "__version__", "unknown")),
        torch_num_threads=int(torch.get_num_threads()),
        torch_num_interop_threads=int(torch.get_num_interop_threads()),
        torch_parallel_info=torch.__config__.parallel_info(),
        threadpool_backends=tp_info,
    )


def save_environment_json(snapshot: EnvironmentSnapshot, output_path: Path) -> Path:
    """Persist EnvironmentSnapshot to JSON safely creating parent directories."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(snapshot), f, indent=2)
    return output_path
