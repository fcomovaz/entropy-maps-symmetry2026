"""Build orchestration for CIMP unit-test and desktop parity C99 executables."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil

from .config import CImpConfig
from .toolchain import CExecutableBuildResult, compile_c_executable


def build_c_runtime_test(cfg: CImpConfig, clean: bool = False) -> CExecutableBuildResult:
    """Compile standalone `cimp_test_runtime.exe` exercising runtime kernels without generated headers."""
    test_build_dir = cfg.paths.build_root / "_runtime_tests"
    if clean and test_build_dir.exists():
        shutil.rmtree(test_build_dir)
    sources = [
        cfg.paths.c_tests_dir / "test_runtime.c",
        cfg.paths.runtime_dir / "cimp_platform.c",
        cfg.paths.runtime_dir / "infra_sklearn.c",
        cfg.paths.runtime_dir / "infra_pytorch.c",
    ]
    exe_path = test_build_dir / "cimp_test_runtime.exe"
    return compile_c_executable(
        sources=sources,
        include_dirs=(cfg.paths.runtime_dir,),
        exe_path=exe_path,
        build_dir=test_build_dir,
        c_standard_flag=cfg.build.c_standard_flag,
        compiler_flags=cfg.build.compiler_flags,
        vswhere_hint=cfg.build.vswhere_hint,
    )


def build_c_streaming_test(cfg: CImpConfig, clean: bool = False) -> CExecutableBuildResult:
    """Compile `cimp_test_streaming.exe` linking `test_streaming.c` + `cimp_descriptor.c` with F32 disabled."""
    required_generated = (
        "cimp_config.h",
        "lut_bin_quant.h",
        "lut_shannon.h",
        "test_image_0.h",
        "test_image_1.h",
        "test_image_2.h",
        "test_image_3.h",
    )
    missing = [name for name in required_generated if not (cfg.generated_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete generated export in {cfg.generated_dir} (missing: {', '.join(missing)}). "
            f"Run export_c_pipeline cleanly first for target '{cfg.effective_target_tag}'."
        )

    test_build_dir = cfg.build_dir / "_streaming_test"
    if clean and test_build_dir.exists():
        shutil.rmtree(test_build_dir)
    sources = [
        cfg.paths.c_tests_dir / "test_streaming.c",
        cfg.paths.runtime_dir / "cimp_descriptor.c",
    ]
    exe_path = test_build_dir / "cimp_test_streaming.exe"
    return compile_c_executable(
        sources=sources,
        include_dirs=(cfg.generated_dir, cfg.paths.runtime_dir),
        exe_path=exe_path,
        build_dir=test_build_dir,
        c_standard_flag=cfg.build.c_standard_flag,
        compiler_flags=(*cfg.build.compiler_flags, "/DCIMP_ENABLE_F32=0"),
        vswhere_hint=cfg.build.vswhere_hint,
    )


def build_desktop_runner(cfg: CImpConfig, clean: bool = False) -> CExecutableBuildResult:
    """Compile target-specific `cimp_desktop_<target_tag>.exe` linking generated models + runtime."""
    build_dir = cfg.build_dir
    if clean and build_dir.exists():
        shutil.rmtree(build_dir)

    required_generated = ("cimp_config.h", "cimp_models_registry.h", "export_manifest.json")
    missing = [name for name in required_generated if not (cfg.generated_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete generated export in {cfg.generated_dir} (missing: {', '.join(missing)}). "
            f"Run export_c_pipeline cleanly first for target '{cfg.effective_target_tag}'."
        )

    cnn_sources = sorted(cfg.generated_dir.glob("model_*.c"))
    sources = [
        cfg.paths.app_dir / "desktop_main.c",
        cfg.paths.runtime_dir / "cimp_platform.c",
        cfg.paths.runtime_dir / "cimp_descriptor.c",
        cfg.paths.runtime_dir / "infra_sklearn.c",
        cfg.paths.runtime_dir / "infra_pytorch.c",
        cfg.paths.runtime_dir / "cimp_pipeline.c",
        *cnn_sources,
    ]
    exe_path = build_dir / cfg.build.executable_name
    result = compile_c_executable(
        sources=sources,
        include_dirs=(cfg.generated_dir, cfg.paths.runtime_dir),
        exe_path=exe_path,
        build_dir=build_dir,
        c_standard_flag=cfg.build.c_standard_flag,
        compiler_flags=cfg.build.compiler_flags,
        vswhere_hint=cfg.build.vswhere_hint,
    )
    info_path = build_dir / "build_info.json"
    info_path.write_text(json.dumps(asdict(result), indent=2, default=str) + "\n", encoding="utf-8")
    return result
