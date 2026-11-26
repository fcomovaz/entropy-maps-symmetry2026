<#
.SYNOPSIS
    Complete sequential reproduction script for the paper experimental pipeline.

.DESCRIPTION
    Executes every canonical command used to generate all models, benchmarks,
    profiling tables, C99 fixed-point verification outputs, MCU fit matrices,
    publication figures, and `results/FINAL_PAPER_REPORT.md` (`seed = 42`).

    Execution is strictly sequential (single process at a time) so CPU single-thread
    latency and GPU training measurements remain uncontaminated.

    NOTE ON SECTION 0 (IMAGE TRANSFORMATIONS & PIXEL CSV SERIALIZATION):
    Section 0 (Steps 0A–0B: raw image conversion to 128x128 grayscale + centered
    letterboxing, pre-padding polarity inversion `255 - I`, and flat row-major
    pixel CSV export) is INCLUDED BELOW BUT COMMENTED OUT BY DEFAULT because
    `data/preprocessed/` and `data/csv/{kolektorsdd2,mvtec-ad}[-inverted].csv`
    are already prepared. Collaborators starting from `data/raw/` only can
    uncomment Section 0 (Steps 0A–0B). All `x85` entropy and `x9` spatial
    asymmetry (`x94-v1`) descriptor steps (Section 1B) are ACTIVE below.

.PARAMETER DryRun
    Prints every command in execution order without running Python processes.

.EXAMPLE
    # Preview all active commands without executing
    .\reproduce_paper_pipeline.ps1 -DryRun

.EXAMPLE
    # Full sequential reproduction from precomputed data/preprocessed & data/csv
    .\reproduce_paper_pipeline.ps1
#>

[CmdletBinding()]
param(
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($scriptRoot)) {
    $scriptRoot = (Get-Location).Path
} else {
    $scriptRoot = (Resolve-Path -LiteralPath $scriptRoot).Path
}
Set-Location -LiteralPath $scriptRoot

$pythonExe = Join-Path $scriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    Write-Error "Virtual environment interpreter not found at '$pythonExe'. Create `.venv` (Python 3.11 + PyTorch cu126) before running."
    exit 1
}

$mvtecCategories = @(
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
)

function Invoke-PipelineCommand {
    param(
        [Parameter(Mandatory = $true)][string]$StepLabel,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    $displayCmd = ".\.venv\Scripts\python.exe " + ($Arguments -join " ")
    Write-Host "`n[$StepLabel]" -ForegroundColor Cyan
    Write-Host "  $displayCmd" -ForegroundColor Gray

    if ($DryRun) {
        Write-Host "  -> [DRY-RUN] Skipped execution." -ForegroundColor Yellow
        return
    }

    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    & $pythonExe $Arguments
    $exitCode = $LASTEXITCODE
    $timer.Stop()

    if ($exitCode -ne 0) {
        Write-Error "Step '$StepLabel' failed with exit code $exitCode after $([math]::Round($timer.Elapsed.TotalSeconds, 2))s."
        exit $exitCode
    }
    Write-Host "  -> [OK] Finished in $([math]::Round($timer.Elapsed.TotalSeconds, 2))s." -ForegroundColor Green
}

$pipelineTimer = [System.Diagnostics.Stopwatch]::StartNew()
Write-Host "==============================================================================" -ForegroundColor White
Write-Host "  CANONICAL SEQUENTIAL REPRODUCTION PIPELINE (seed = 42)" -ForegroundColor White
Write-Host "  Working Directory : $scriptRoot" -ForegroundColor Gray
Write-Host "  Python Executable : $pythonExe" -ForegroundColor Gray
Write-Host "  Mode              : $(if ($DryRun) { 'DRY-RUN' } else { 'LIVE SEQUENTIAL EXECUTION' })" -ForegroundColor Yellow
Write-Host "==============================================================================" -ForegroundColor White


# ==============================================================================
# SECTION 0 (COMMENTED BY DEFAULT): IMAGE TRANSFORMATIONS & PIXEL CSV EXPORT
# ==============================================================================
# Uncomment Steps 0A–0B ONLY if regenerating `data/preprocessed/` (128x128 PNGs)
# and flat pixel tables `data/csv/{kolektorsdd2,mvtec-ad}[-inverted].csv`
# directly from `data/raw/`.
#
# Transformation order implemented in `src/preprocessing/letterbox.py`:
#   1. Grayscale conversion (`mode='L'`, ITU-R 601 luma: 0.299R + 0.587G + 0.114B).
#   2. Optional polarity inversion (`I' = 255 - I` in uint8) BEFORE letterboxing,
#      ensuring outer background padding remains `pad_value = 0` (black).
#   3. Aspect-preserving LANCZOS resize (`scale = 128 / max(H, W)`) + centered
#      square letterboxing onto `128x128` (`pad_value = 0`, `anchor = 'center'`).
#   4. Mask rules: KolektorSDD2 inspects full-res `_GT` masks (`mask.max() > 0`),
#      validates numeric stems (`^\d+$`), routes to `good/` or `defect/`, and
#      discards masks; MVTec-AD resizes `ground_truth` masks with `NEAREST`
#      without grayscale reweighting or inversion.
# ==============================================================================
<#
# --- 0A. Raw Image Transformations & Standardization (128x128 PNGs) ---
Invoke-PipelineCommand -StepLabel "0A.1 Preprocess KolektorSDD2 (Standard)" `
    -Arguments @("-m", "experiments.preprocess_kolektorsdd2", "--config", "configs/preprocessing/kolektorsdd2.toml", "--overwrite")

Invoke-PipelineCommand -StepLabel "0A.2 Preprocess KolektorSDD2 (Inverted: 255 - I)" `
    -Arguments @("-m", "experiments.preprocess_kolektorsdd2", "--config", "configs/preprocessing/kolektorsdd2.toml", "--inverted", "--overwrite")

Invoke-PipelineCommand -StepLabel "0A.3 Preprocess MVTec-AD All-15 (Standard)" `
    -Arguments @("-m", "experiments.preprocess_mvtec_ad", "--config", "configs/preprocessing/mvtec_ad.toml", "--overwrite")

Invoke-PipelineCommand -StepLabel "0A.4 Preprocess MVTec-AD All-15 (Inverted: 255 - I)" `
    -Arguments @("-m", "experiments.preprocess_mvtec_ad", "--config", "configs/preprocessing/mvtec_ad.toml", "--inverted", "--overwrite")

# --- 0B. Flat Row-Major Pixel CSV Export (16,384 pixels: px_00000..px_16383) ---
Invoke-PipelineCommand -StepLabel "0B.1 Export KolektorSDD2 Flat Pixel CSV (Standard)" `
    -Arguments @("-m", "experiments.export_kolektorsdd2_csv", "--config", "configs/csv/kolektorsdd2.toml", "--overwrite")

Invoke-PipelineCommand -StepLabel "0B.2 Export KolektorSDD2 Flat Pixel CSV (Inverted)" `
    -Arguments @("-m", "experiments.export_kolektorsdd2_csv", "--config", "configs/csv/kolektorsdd2.toml", "--inverted", "--overwrite")

Invoke-PipelineCommand -StepLabel "0B.3 Export MVTec-AD Flat Pixel CSV (Standard)" `
    -Arguments @("-m", "experiments.export_mvtec_ad_csv", "--config", "configs/csv/mvtec_ad.toml", "--overwrite")

Invoke-PipelineCommand -StepLabel "0B.4 Export MVTec-AD Flat Pixel CSV (Inverted)" `
    -Arguments @("-m", "experiments.export_mvtec_ad_csv", "--config", "configs/csv/mvtec_ad.toml", "--inverted", "--overwrite")
#>

# Optional: To wipe all prior generated files in `results/` before starting fresh:
# .\cleanup.ps1 -AllArtifacts


# ==============================================================================
# SECTION 1A: ENVIRONMENT VERIFICATION & NATIVE C99 KERNEL COMPILATION
# ==============================================================================
Invoke-PipelineCommand -StepLabel "1A.1 Verify Python, PyTorch & Real CUDA Execution" `
    -Arguments @("tests/test_environment.py")

Invoke-PipelineCommand -StepLabel "1A.2 Compile Native C99 Single-Pass Entropy DLL (MSVC /O2)" `
    -Arguments @("-m", "src.features.build_native", "--config", "configs/features/native_entropy.toml", "--clean")


# ==============================================================================
# SECTION 1B: MULTISCALE ENTROPY (x85) & SPATIAL ASYMMETRY (x9 -> x94-v1) EXPORT
# ==============================================================================
# Extracts the 85 normalized Shannon entropy features (B=16, grids 1x1..8x8 via
# C99 `c_single_pass`) and appends the 9 reflection/rotation asymmetry features
# (`asym_{lr,tb,rot180}_{2x2,4x4,8x8}`) forming the canonical `x94-v1` CSVs.
Invoke-PipelineCommand -StepLabel "1B.1 Extract Entropy Pyramid x85: KolektorSDD2 (Standard)" `
    -Arguments @("-m", "experiments.export_entropy_maps", "--config", "configs/entropy/kolektorsdd2.toml", "--overwrite")

Invoke-PipelineCommand -StepLabel "1B.2 Extract Entropy Pyramid x85: KolektorSDD2 (Inverted)" `
    -Arguments @("-m", "experiments.export_entropy_maps", "--config", "configs/entropy/kolektorsdd2.toml", "--inverted", "--overwrite")

Invoke-PipelineCommand -StepLabel "1B.3 Extract Entropy Pyramid x85: MVTec-AD All-15 (Standard)" `
    -Arguments @("-m", "experiments.export_entropy_maps", "--config", "configs/entropy/mvtec_ad.toml", "--overwrite")

Invoke-PipelineCommand -StepLabel "1B.4 Extract Entropy Pyramid x85: MVTec-AD All-15 (Inverted)" `
    -Arguments @("-m", "experiments.export_entropy_maps", "--config", "configs/entropy/mvtec_ad.toml", "--inverted", "--overwrite")

Invoke-PipelineCommand -StepLabel "1B.5 Append Spatial Asymmetry Features x9 -> x94-v1 (All Datasets, Std & Inv)" `
    -Arguments @("-m", "experiments.append_asymmetry_features", "--all", "--overwrite")


# ==============================================================================
# SECTION 2: CLASSICAL DESCRIPTOR EVALUATION (x85, x9, x94 — ALL 6 CLASSIFIERS)
# ==============================================================================
# Evaluates dt, knn, lr, lr_dot, mlp_16_dot, rf with 7-subset ablations and
# permutation/Gini explainability across KolektorSDD2 and all 15 MVTec-AD categories.
Invoke-PipelineCommand -StepLabel "2.1 Classical ML Evaluation: KolektorSDD2 (Standard)" `
    -Arguments @("-m", "experiments.evaluate_descriptors", "--config", "configs/classification/kolektorsdd2.toml", "--classifier", "all", "--overwrite")

Invoke-PipelineCommand -StepLabel "2.2 Classical ML Evaluation: KolektorSDD2 (Inverted)" `
    -Arguments @("-m", "experiments.evaluate_descriptors", "--config", "configs/classification/kolektorsdd2.toml", "--classifier", "all", "--inverted", "--overwrite")

Invoke-PipelineCommand -StepLabel "2.3 Classical ML Evaluation: MVTec-AD All-15 Categories (Standard)" `
    -Arguments @("-m", "experiments.evaluate_descriptors", "--config", "configs/classification/mvtec_ad.toml", "--all-objects", "--classifier", "all", "--overwrite")

Invoke-PipelineCommand -StepLabel "2.4 Classical ML Evaluation: MVTec-AD All-15 Categories (Inverted)" `
    -Arguments @("-m", "experiments.evaluate_descriptors", "--config", "configs/classification/mvtec_ad.toml", "--all-objects", "--classifier", "all", "--inverted", "--overwrite")


# ==============================================================================
# SECTION 3: LIGHTWEIGHT CNN ZOO TRAINING & EVALUATION (ALL 7 MODELS, SEED = 42)
# ==============================================================================
# Trains ultra_tiny, ds_cnn, tiny_cnn, mobilenet_like, tiny_resnet, small_cnn,
# and torchvision_mobilenet with stratified val early stopping and val_f1 calibration.
Invoke-PipelineCommand -StepLabel "3.1 CNN Zoo Training & Evaluation: KolektorSDD2 (Standard, seed=42)" `
    -Arguments @("-m", "experiments.evaluate_cnn", "--config", "configs/cnn/kolektorsdd2.toml", "--model", "all", "--seeds", "42", "--overwrite")

Invoke-PipelineCommand -StepLabel "3.2 CNN Zoo Training & Evaluation: KolektorSDD2 (Inverted, seed=42)" `
    -Arguments @("-m", "experiments.evaluate_cnn", "--config", "configs/cnn/kolektorsdd2.toml", "--model", "all", "--seeds", "42", "--inverted", "--overwrite")

Invoke-PipelineCommand -StepLabel "3.3 CNN Zoo Training & Evaluation: MVTec-AD All-15 Categories (Standard, seed=42)" `
    -Arguments @("-m", "experiments.evaluate_cnn", "--config", "configs/cnn/mvtec_ad.toml", "--all-objects", "--model", "all", "--seeds", "42", "--overwrite")

Invoke-PipelineCommand -StepLabel "3.4 CNN Zoo Training & Evaluation: MVTec-AD All-15 Categories (Inverted, seed=42)" `
    -Arguments @("-m", "experiments.evaluate_cnn", "--config", "configs/cnn/mvtec_ad.toml", "--all-objects", "--model", "all", "--seeds", "42", "--inverted", "--overwrite")


# ==============================================================================
# SECTION 4: 100% CPU-FAIR BENCHMARKING (PHASE A) & DEEP PROFILING (PHASE B)
# ==============================================================================
# Run strictly with no parallel load (`threads = 1` primary, `batch = 1` latency,
# `batch = 32` throughput; `T_total = T_descriptor + T_model` vs `T_preprocess + T_model`).
Invoke-PipelineCommand -StepLabel "4A.1 CPU-Fair Benchmark: KolektorSDD2 (Standard)" `
    -Arguments @("-m", "experiments.benchmark_cpu_fair", "--config", "configs/profiling/kolektorsdd2.toml")

Invoke-PipelineCommand -StepLabel "4A.2 CPU-Fair Benchmark: KolektorSDD2 (Inverted)" `
    -Arguments @("-m", "experiments.benchmark_cpu_fair", "--config", "configs/profiling/kolektorsdd2.toml", "--inverted")

Invoke-PipelineCommand -StepLabel "4A.3 CPU-Fair Benchmark: MVTec-AD All-15 Categories (Standard)" `
    -Arguments @("-m", "experiments.benchmark_cpu_fair", "--config", "configs/profiling/mvtec_ad.toml", "--all-objects")

Invoke-PipelineCommand -StepLabel "4A.4 CPU-Fair Benchmark: MVTec-AD All-15 Categories (Inverted)" `
    -Arguments @("-m", "experiments.benchmark_cpu_fair", "--config", "configs/profiling/mvtec_ad.toml", "--all-objects", "--inverted")

Invoke-PipelineCommand -StepLabel "4B.1 Deep CPU Profiling: KolektorSDD2 (Standard)" `
    -Arguments @("-m", "experiments.profile_pipelines", "--config", "configs/profiling/kolektorsdd2.toml")

Invoke-PipelineCommand -StepLabel "4B.2 Deep CPU Profiling: KolektorSDD2 (Inverted)" `
    -Arguments @("-m", "experiments.profile_pipelines", "--config", "configs/profiling/kolektorsdd2.toml", "--inverted")

Invoke-PipelineCommand -StepLabel "4B.3 Deep CPU Profiling: MVTec-AD All-15 Categories (Standard)" `
    -Arguments @("-m", "experiments.profile_pipelines", "--config", "configs/profiling/mvtec_ad.toml", "--all-objects")

Invoke-PipelineCommand -StepLabel "4B.4 Deep CPU Profiling: MVTec-AD All-15 Categories (Inverted)" `
    -Arguments @("-m", "experiments.profile_pipelines", "--config", "configs/profiling/mvtec_ad.toml", "--all-objects", "--inverted")


# ==============================================================================
# SECTION 5: C99 FIXED-POINT (f32 / Q15) EXPORT, PARITY, TIMING & VISUALIZATION
# ==============================================================================
Invoke-PipelineCommand -StepLabel "5.0a Export C99 Pipeline: KolektorSDD2" `
    -Arguments @("-m", "experiments.export_c_pipeline", "--config", "configs/c_imp/kolektorsdd2.toml")

Invoke-PipelineCommand -StepLabel "5.0b Verify C99 Parity & Desktop Timing: KolektorSDD2" `
    -Arguments @("-m", "experiments.verify_c_pipeline", "--config", "configs/c_imp/kolektorsdd2.toml", "--benchmark")

Invoke-PipelineCommand -StepLabel "5.0c Visualize C99 Diagnostics: KolektorSDD2" `
    -Arguments @("-m", "experiments.visualize_c_pipeline", "--config", "configs/c_imp/kolektorsdd2.toml")

$idx = 0
foreach ($cat in $mvtecCategories) {
    $idx++
    Invoke-PipelineCommand -StepLabel "5.${idx}a Export C99 Pipeline: MVTec-AD $cat" `
        -Arguments @("-m", "experiments.export_c_pipeline", "--config", "configs/c_imp/mvtec_ad.toml", "--object", $cat)

    Invoke-PipelineCommand -StepLabel "5.${idx}b Verify C99 Parity & Desktop Timing: MVTec-AD $cat" `
        -Arguments @("-m", "experiments.verify_c_pipeline", "--config", "configs/c_imp/mvtec_ad.toml", "--object", $cat, "--benchmark")

    Invoke-PipelineCommand -StepLabel "5.${idx}c Visualize C99 Diagnostics: MVTec-AD $cat" `
        -Arguments @("-m", "experiments.visualize_c_pipeline", "--config", "configs/c_imp/mvtec_ad.toml", "--object", $cat)
}


# ==============================================================================
# SECTION 6: MICROCONTROLLER MEMORY/FIT MATRIX & LIVE ESP32-CAM BENCH
# ==============================================================================
# Live MCU execution requires connecting each board over USB/UART
# (replace the Windows COM ports below if yours differ).
$esp32Port = "COM5"
$nanoPort  = "COM4"
$unoPort   = "COM6"

Invoke-PipelineCommand -StepLabel "6.1a Cross-Compile Fit Sweep: Arduino Nano (Compile-Only)" `
    -Arguments @("-m", "experiments.hw_test", "bench", "--config", "configs/hw_test/boards.toml", "--board", "nano", "--compile-only")

Invoke-PipelineCommand -StepLabel "6.1b Cross-Compile Fit Sweep: Arduino UNO (Compile-Only)" `
    -Arguments @("-m", "experiments.hw_test", "bench", "--config", "configs/hw_test/boards.toml", "--board", "uno", "--compile-only")

Invoke-PipelineCommand -StepLabel "6.1c Cross-Compile Fit Sweep: NodeMCU ESP8266 (Compile-Only)" `
    -Arguments @("-m", "experiments.hw_test", "bench", "--config", "configs/hw_test/boards.toml", "--board", "nodemcu", "--compile-only")

<#
# Uncomment to run the live hardware benchmark sweeps on ESP32-CAM, Arduino Nano, and Arduino Uno:
Invoke-PipelineCommand -StepLabel "6.2a Stage Deterministic KolektorSDD2 Test Images (seed=42)" `
    -Arguments @("-m", "experiments.hw_test", "images", "--config", "configs/hw_test/boards.toml", "--all")

Invoke-PipelineCommand -StepLabel "6.2b ESP32-CAM: Compile All f32 + Run lr_dot, mlp_16_dot, ultra_tiny (f32)" `
    -Arguments @("-m", "experiments.hw_test", "bench", "--config", "configs/hw_test/boards.toml", "--board", "esp32cam", "--transport", "uart", "--port", $esp32Port, "--models", "lr_dot", "mlp_16_dot", "ultra_tiny", "--modes", "f32", "--all", "--dump-descriptor", "--no-prompt")

Invoke-PipelineCommand -StepLabel "6.2c ESP32-CAM: Compile All q15 + Run All 9 q15 Models" `
    -Arguments @("-m", "experiments.hw_test", "bench", "--config", "configs/hw_test/boards.toml", "--board", "esp32cam", "--transport", "uart", "--port", $esp32Port, "--modes", "q15", "--all", "--dump-descriptor", "--no-prompt")

Invoke-PipelineCommand -StepLabel "6.2d Arduino Nano (8-bit ATmega328P): Run lr_dot & mlp_16_dot (q15 streaming)" `
    -Arguments @("-m", "experiments.hw_test", "bench", "--config", "configs/hw_test/boards.toml", "--board", "nano", "--transport", "uart", "--port", $nanoPort, "--models", "lr_dot", "mlp_16_dot", "--modes", "q15", "--all", "--no-prompt")

Invoke-PipelineCommand -StepLabel "6.2e Arduino Uno (8-bit ATmega328P): Run lr_dot & mlp_16_dot (q15 streaming)" `
    -Arguments @("-m", "experiments.hw_test", "bench", "--config", "configs/hw_test/boards.toml", "--board", "uno", "--transport", "uart", "--port", $unoPort, "--models", "lr_dot", "mlp_16_dot", "--modes", "q15", "--all", "--no-prompt")
#>

Invoke-PipelineCommand -StepLabel "6.3 Regenerate Consolidated 53-Row MCU Fit Report" `
    -Arguments @("-m", "experiments.hw_test", "report", "--config", "configs/hw_test/boards.toml")


# ==============================================================================
# SECTION 7: METHODOLOGY FIGURES, PAPER REPORT & FORMAL UNIT TEST VERIFICATION
# ==============================================================================
Invoke-PipelineCommand -StepLabel "7.1 Generate Methodology & Polarity-Invariance Figures (MDPI 300 DPI)" `
    -Arguments @("-m", "experiments.visualize_methodology", "--config", "configs/visualization/methodology.toml", "--overwrite")

Invoke-PipelineCommand -StepLabel "7.2 Generate Paper-Ready Consolidated Report (results/FINAL_PAPER_REPORT.md)" `
    -Arguments @("-m", "experiments.generate_paper_report", "--config", "configs/reporting/paper_report.toml", "--overwrite")

Invoke-PipelineCommand -StepLabel "7.3 Run Formal Repository Unit Test Suite" `
    -Arguments @("-B", "-m", "unittest", "discover", "tests", "-v")

$pipelineTimer.Stop()
Write-Host "`n==============================================================================" -ForegroundColor Green
Write-Host "  ALL PIPELINE STAGES COMPLETED SUCCESSFULLY IN $([math]::Round($pipelineTimer.Elapsed.TotalMinutes, 2)) MINUTES" -ForegroundColor Green
Write-Host "  Final Report: results/FINAL_PAPER_REPORT.md" -ForegroundColor White
Write-Host "==============================================================================" -ForegroundColor Green
