<#
.SYNOPSIS
    Master research pipeline orchestrator from start to finish.

.DESCRIPTION
    Centralized execution script containing all formal experiment commands for the
    research on visual entropy maps and spatial asymmetry descriptors (x94-v1) vs CNNs,
    desktop C99 migration, and microcontroller hardware test bench verification.
    
    Cross-checked and aligned with all repository tasks (Task 01 through Task 12).
    Organized into 12 clear, modular stages with colorful, styled terminal titles.
    Every command is fully documented so you can run the whole research end-to-end,
    select specific stages via parameters, or open this file and comment/uncomment
    individual lines as needed.

.PARAMETER Stage
    Specific stage to execute:
    - 'all'          : Runs stages 0 through 8 (default complete software & ML research pipeline)
    - '0_build'      : Stage 0: Environment check and C99 native kernel compilation
    - '1_preprocess' : Stage 1: Raw image preprocessing (MVTec-AD & KolektorSDD2)
    - '2_csv'        : Stage 2: Flat pixel CSV export
    - '3_entropy'    : Stage 3: Visual region multiscale entropy extraction (x85)
    - '4_asymmetry'  : Stage 4: Multiscale spatial asymmetry feature appending (x85 -> x94)
    - '5_ml'         : Stage 5: Classical supervised classification, ablation & explainability
    - '6_cnn'        : Stage 6: Lightweight CNN zoo training & evaluation
    - '7_benchmark'  : Stage 7: 100% CPU-fair benchmarking (Phase A)
    - '8_profiling'  : Stage 8: Granular pipeline profiling (Phase B)
    - '9_python_cpu' : Stage 9: (Optional) Pure-Python pre-MCU benchmark without C99 DLL
    - '10_c_embed'   : Stage 10: (Task 11) Embedded C pipeline export, parity & visualization
    - '11_hw_test'   : Stage 11: (Task 12) Microcontroller hardware test bench (PC -> MCU -> PC)

.PARAMETER DryRun
    Prints every command that would be executed with full arguments without running them.

.PARAMETER SkipCNN
    If running 'all', skips Stage 6 (heavy CNN training).

.PARAMETER SkipAllObjects
    If set, skips the multi-object 15-category MVTec sweep to focus exclusively on
    the canonical benchmark pair: KolektorSDD2 and MVTec-AD bottle.

.PARAMETER ContinueOnError
    By default, the script aborts immediately if any step returns an exit code != 0.
    Set this flag to continue running subsequent steps despite errors.

.EXAMPLE
    # Dry-run to preview all commands across all stages with colored titles
    .\pipeline.ps1 -DryRun

.EXAMPLE
    # Run only Stage 5 (classical ML classifiers on x94)
    .\pipeline.ps1 -Stage 5_ml

.EXAMPLE
    # Run full pipeline but skip heavy CNN training
    .\pipeline.ps1 -SkipCNN

.EXAMPLE
    # Run full canonical pipeline skipping the 15-object MVTec sweep
    .\pipeline.ps1 -SkipAllObjects

.EXAMPLE
    # Run Stage 10 (Task 11: C99 export & desktop parity)
    .\pipeline.ps1 -Stage 10_c_embed

.EXAMPLE
    # Run Stage 11 (Task 12: Microcontroller hardware test bench catalog & fit report)
    .\pipeline.ps1 -Stage 11_hw_test
#>

[CmdletBinding()]
param(
    [ValidateSet('all', '0_build', '1_preprocess', '2_csv', '3_entropy', '4_asymmetry', '5_ml', '6_cnn', '7_benchmark', '8_profiling', '9_python_cpu', '10_c_embed', '11_hw_test')]
    [string]$Stage = 'all',

    [switch]$DryRun,
    [switch]$SkipCNN,
    [switch]$SkipAllObjects,
    [switch]$ContinueOnError
)

Set-StrictMode -Version Latest
$ErrorActionPreference = if ($ContinueOnError) { "Continue" } else { "Stop" }

# Enable UTF-8 console output for crisp box borders across PS versions
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
} catch {}

# =====================================================================
#  0. ENVIRONMENT RESOLUTION & PATH ANCHORING
# =====================================================================
$scriptRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($scriptRoot)) {
    $scriptRoot = (Get-Location).Path
} else {
    $scriptRoot = (Resolve-Path -LiteralPath $scriptRoot).Path
}
Set-Location -LiteralPath $scriptRoot

$pythonExe = Join-Path $scriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    Write-Error "CRITICAL: Python interpreter not found at '$pythonExe'. Please initialize the virtual environment first."
    exit 1
}

# =====================================================================
#  TERMINAL STYLING & BANNER HELPERS
# =====================================================================
function Show-PipelineHeader {
    Write-Host ""
    Write-Host "╔════════════════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║                                                                            ║" -ForegroundColor Cyan
    Write-Host "║   ███████╗██╗   ██╗███╗   ███╗███╗   ███╗███████╗████████╗██████╗ ██╗   ██╗ ║" -ForegroundColor Cyan
    Write-Host "║   ██╔════╝╚██╗ ██╔╝████╗ ████║████╗ ████║██╔════╝╚══██╔══╝██╔══██╗╚██╗ ██╔╝ ║" -ForegroundColor Cyan
    Write-Host "║   ███████╗ ╚████╔╝ ██╔████╔██║██╔████╔██║█████╗     ██║   ██████╔╝ ╚████╔╝  ║" -ForegroundColor Cyan
    Write-Host "║   ╚════██║  ╚██╔╝  ██║╚██╔╝██║██║╚██╔╝██║██╔══╝     ██║   ██╔══██╗  ╚██╔╝   ║" -ForegroundColor Cyan
    Write-Host "║   ███████║   ██║   ██║ ╚═╝ ██║██║ ╚═╝ ██║███████╗   ██║   ██║  ██║   ██║    ║" -ForegroundColor Cyan
    Write-Host "║   ╚══════╝   ╚═╝   ╚═╝     ╚═╝╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝    ║" -ForegroundColor Cyan
    Write-Host "║                                                                            ║" -ForegroundColor Cyan
    Write-Host "║             RESEARCH REPRODUCIBILITY PIPELINE (x94-v1 vs CNN)              ║" -ForegroundColor White
    Write-Host "╚════════════════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  • Working Directory : " -NoNewline -ForegroundColor DarkGray
    Write-Host "$scriptRoot" -ForegroundColor Gray
    Write-Host "  • Python Binary     : " -NoNewline -ForegroundColor DarkGray
    Write-Host "$pythonExe" -ForegroundColor Gray
    Write-Host "  • Target Stage      : " -NoNewline -ForegroundColor DarkGray
    Write-Host "$Stage" -ForegroundColor Yellow
    Write-Host "  • Execution Mode    : " -NoNewline -ForegroundColor DarkGray
    Write-Host "$(if ($DryRun) { 'DRY-RUN (Simulated)' } else { 'LIVE EXECUTION' })" -ForegroundColor $(if ($DryRun) { 'Yellow' } else { 'Green' })
    Write-Host "  • Skip CNN Training : " -NoNewline -ForegroundColor DarkGray
    Write-Host "$SkipCNN" -ForegroundColor Gray
    Write-Host "  • Skip All-Objects  : " -NoNewline -ForegroundColor DarkGray
    Write-Host "$SkipAllObjects" -ForegroundColor Gray
    Write-Host "────────────────────────────────────────────────────────────────────────────`n" -ForegroundColor DarkGray
}

function Show-StageHeader {
    param(
        [string]$Number,
        [string]$Title,
        [string]$Subtitle,
        [string]$Icon = "►",
        [ConsoleColor]$Color = [ConsoleColor]::Cyan
    )

    $boxWidth = 78
    $contentWidth = $boxWidth - 4 # 2 border chars + 2 padding spaces = 74

    Write-Host ""
    Write-Host ("╔" + ("═" * ($boxWidth - 2)) + "╗") -ForegroundColor $Color

    # Line 1: Badge + Icon + Title
    $badge = "[STAGE $Number]"
    $line1Content = "$badge $Icon $Title"
    if ($line1Content.Length -gt $contentWidth) {
        $line1Content = $line1Content.Substring(0, $contentWidth)
    }
    $line1Pad = " " * ($contentWidth - $line1Content.Length)
    
    Write-Host "║ " -NoNewline -ForegroundColor $Color
    Write-Host "$badge" -NoNewline -ForegroundColor Black -BackgroundColor $Color
    Write-Host " $Icon $Title$line1Pad" -NoNewline -ForegroundColor White
    Write-Host " ║" -ForegroundColor $Color

    # Line 2: Subtitle
    if ($Subtitle) {
        $line2Content = "   ↳ $Subtitle"
        if ($line2Content.Length -gt $contentWidth) {
            $line2Content = $line2Content.Substring(0, $contentWidth)
        }
        $line2Pad = " " * ($contentWidth - $line2Content.Length)
        Write-Host "║ " -NoNewline -ForegroundColor $Color
        Write-Host "$line2Content$line2Pad" -NoNewline -ForegroundColor Gray
        Write-Host " ║" -ForegroundColor $Color
    }

    Write-Host ("╚" + ("═" * ($boxWidth - 2)) + "╝") -ForegroundColor $Color
    Write-Host ""
}

function Invoke-Step {
    param(
        [string]$Title,
        [string[]]$ArgumentList
    )

    $cmdString = "$pythonExe " + ($ArgumentList -join " ")
    Write-Host "  ┌── " -NoNewline -ForegroundColor Cyan
    Write-Host "$Title" -ForegroundColor White
    Write-Host "  │   " -NoNewline -ForegroundColor Cyan
    Write-Host "CMD : " -NoNewline -ForegroundColor DarkGray
    Write-Host $cmdString -ForegroundColor DarkGray

    if ($DryRun) {
        Write-Host "  └── " -NoNewline -ForegroundColor Cyan
        Write-Host "[DRY-RUN] Command logged, skipped execution.`n" -ForegroundColor DarkYellow
        return
    }

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $pythonExe $ArgumentList
    $exitCode = $LASTEXITCODE
    $sw.Stop()
    $elapsed = [math]::Round($sw.Elapsed.TotalSeconds, 2)

    if ($exitCode -ne 0) {
        Write-Host "  └── " -NoNewline -ForegroundColor Cyan
        Write-Host "[FAILED] " -NoNewline -ForegroundColor Red
        Write-Host "Exit Code: $exitCode (Elapsed: ${elapsed}s)`n" -ForegroundColor Red
        if (-not $ContinueOnError) {
            Write-Error "Pipeline stopped due to error in step: '$Title'"
            exit $exitCode
        }
    } else {
        Write-Host "  └── " -NoNewline -ForegroundColor Cyan
        Write-Host "[SUCCESS] " -NoNewline -ForegroundColor Green
        Write-Host "Completed in ${elapsed}s`n" -ForegroundColor Gray
    }
}

# Display top-level banner
Show-PipelineHeader


# =====================================================================
#  STAGE 00: ENVIRONMENT CHECK & C99 NATIVE COMPILATION (Tasks 09 & 10)
# =====================================================================
# Verifies PyTorch CUDA runtime, GPU allocation, and compiles the
# single-pass C99 entropy kernel (entropy_pyramid.dll).
# When compiled, entropy_backend 'auto' resolves to 'c_single_pass'.
# =====================================================================
if ($Stage -in @('all', '0_build')) {
    Show-StageHeader -Number "00" `
        -Title "ENVIRONMENT CHECK & NATIVE C99 COMPILATION" `
        -Subtitle "Checking PyTorch CUDA & building fast single-pass C99 DLL" `
        -Icon "⚙" `
        -Color Cyan

    # 0.1 Check hardware, CUDA runtime, and torch functionality
    Invoke-Step -Title "Check CUDA & PyTorch Environment" `
        -ArgumentList @("tests/test_environment.py")

    # 0.2 Compile native C99 entropy pyramid shared library (MSVC x64)
    # Output: src/features/native/build/entropy_pyramid.dll
    Invoke-Step -Title "Compile Native C99 Single-Pass Entropy DLL" `
        -ArgumentList @("-m", "src.features.build_native", "--config", "configs/features/native_entropy.toml", "--clean")
}


# =====================================================================
#  STAGE 01: RAW DATASET PREPROCESSING (Tasks 01 & 03)
# =====================================================================
# Standardizes raw images into 128x128 grayscale ('L' mode) via centered
# letterboxing (pad_value = 0) with aspect ratio preserved.
# Order of operations: Grayscale -> Invert (if --inverted) -> Resize -> Pad.
# Inversion before padding guarantees outer padding remains 0 (black).
# Outputs:
#   - data/preprocessed/mvtec-ad/
#   - data/preprocessed/mvtec-ad-inverted/
#   - data/preprocessed/KolektorSDD2/
#   - data/preprocessed/KolektorSDD2-inverted/
# =====================================================================
if ($Stage -in @('all', '1_preprocess')) {
    Show-StageHeader -Number "01" `
        -Title "RAW IMAGE PREPROCESSING (128x128 GRAYSCALE LETTERBOX)" `
        -Subtitle "Standardizing MVTec-AD & KolektorSDD2 (Normal & Inverted 255-I)" `
        -Icon "✂" `
        -Color Yellow

    # 1.1 MVTec-AD Standard (15 categories, 128x128, grises, ground_truth NEAREST {0, 255})
    Invoke-Step -Title "Preprocess MVTec-AD (Standard)" `
        -ArgumentList @("-m", "experiments.preprocess_mvtec_ad")

    # 1.2 MVTec-AD Inverted (255 - I applied before letterbox padding)
    Invoke-Step -Title "Preprocess MVTec-AD (Inverted: 255 - I)" `
        -ArgumentList @("-m", "experiments.preprocess_mvtec_ad", "--inverted")

    # 1.3 KolektorSDD2 Standard (classified via mask.max() > 0; masks discarded from output)
    Invoke-Step -Title "Preprocess KolektorSDD2 (Standard: 3,335 pairs)" `
        -ArgumentList @("-m", "experiments.preprocess_kolektorsdd2")

    # 1.4 KolektorSDD2 Inverted (255 - I applied before letterbox padding)
    Invoke-Step -Title "Preprocess KolektorSDD2 (Inverted: 255 - I)" `
        -ArgumentList @("-m", "experiments.preprocess_kolektorsdd2", "--inverted")
}


# =====================================================================
#  STAGE 02: TABULAR FLAT PIXEL CSV EXPORT (Tasks 02 & 04)
# =====================================================================
# Serializes preprocessed images into flat row-major CSV files:
#   - MVTec-AD: 16,391 columns (5 metadata + 16,384 pixels px_00000..px_16383 + 2 dimensions)
#   - KolektorSDD2: 16,389 columns (3 metadata + 16,384 pixels + 2 dimensions)
# Outputs:
#   - data/csv/mvtec-ad.csv & data/csv/mvtec-ad-inverted.csv
#   - data/csv/kolektorsdd2.csv & data/csv/kolektorsdd2-inverted.csv
# =====================================================================
if ($Stage -in @('all', '2_csv')) {
    Show-StageHeader -Number "02" `
        -Title "TABULAR FLAT PIXEL CSV EXPORT" `
        -Subtitle "Flattening images to row-major pixel intensity tables (px_00000..px_16383)" `
        -Icon "≡" `
        -Color DarkYellow

    # 2.1 Export MVTec-AD standard
    Invoke-Step -Title "Export MVTec-AD Flat CSV (Standard)" `
        -ArgumentList @("-m", "experiments.export_mvtec_ad_csv")

    # 2.2 Export MVTec-AD inverted
    Invoke-Step -Title "Export MVTec-AD Flat CSV (Inverted)" `
        -ArgumentList @("-m", "experiments.export_mvtec_ad_csv", "--inverted")

    # 2.3 Export KolektorSDD2 standard
    Invoke-Step -Title "Export KolektorSDD2 Flat CSV (Standard)" `
        -ArgumentList @("-m", "experiments.export_kolektorsdd2_csv")

    # 2.4 Export KolektorSDD2 inverted
    Invoke-Step -Title "Export KolektorSDD2 Flat CSV (Inverted)" `
        -ArgumentList @("-m", "experiments.export_kolektorsdd2_csv", "--inverted")
}


# =====================================================================
#  STAGE 03: MULTISCALE REGIONAL ENTROPY EXTRACTION (x85) (Task 05)
# =====================================================================
# Computes Shannon entropy across spatial partitions g in {1, 2, 4, 8}
# (1 + 4 + 16 + 64 = 85 cells).
# Backend 'auto' automatically binds to the C99 single-pass kernel.
# Replaces pixel columns with 85 entropy features E_1x1_r0_c0..E_8x8_r7_c7.
# Outputs:
#   - data/csv/mvtec-ad-entropy.csv (+ .json sidecar)
#   - data/csv/mvtec-ad-inverted-entropy.csv (+ .json sidecar)
#   - data/csv/kolektorsdd2-entropy.csv (+ .json sidecar)
#   - data/csv/kolektorsdd2-inverted-entropy.csv (+ .json sidecar)
# =====================================================================
if ($Stage -in @('all', '3_entropy')) {
    Show-StageHeader -Number "03" `
        -Title "MULTISCALE REGION ENTROPY EXTRACTION (x85)" `
        -Subtitle "Shannon entropy across 1x1, 2x2, 4x4, 8x8 spatial grids via C99" `
        -Icon "◈" `
        -Color Magenta

    # 3.1 MVTec-AD standard entropy
    Invoke-Step -Title "Extract Entropy Maps: MVTec-AD (Standard)" `
        -ArgumentList @("-m", "experiments.export_entropy_maps", "--config", "configs/entropy/mvtec_ad.toml")

    # 3.2 MVTec-AD inverted entropy
    Invoke-Step -Title "Extract Entropy Maps: MVTec-AD (Inverted)" `
        -ArgumentList @("-m", "experiments.export_entropy_maps", "--config", "configs/entropy/mvtec_ad.toml", "--inverted")

    # 3.3 KolektorSDD2 standard entropy
    Invoke-Step -Title "Extract Entropy Maps: KolektorSDD2 (Standard)" `
        -ArgumentList @("-m", "experiments.export_entropy_maps", "--config", "configs/entropy/kolektorsdd2.toml")

    # 3.4 KolektorSDD2 inverted entropy
    Invoke-Step -Title "Extract Entropy Maps: KolektorSDD2 (Inverted)" `
        -ArgumentList @("-m", "experiments.export_entropy_maps", "--config", "configs/entropy/kolektorsdd2.toml", "--inverted")
}


# =====================================================================
#  STAGE 04: MULTISCALE SPATIAL ASYMMETRY APPENDING (x85 -> x94) (Task 06)
# =====================================================================
# Computes 9 spatial asymmetry residuals (Left-Right, Top-Bottom, Rot180)
# for levels g in {2, 4, 8} directly from the 85 entropy columns.
# Appends columns: asym_{lr,tb,rot180}_{2x2,4x4,8x8}.
# Produces the canonical 94-dimensional descriptor (schema x94-v1).
# Outputs:
#   - data/csv/mvtec-ad-entropy-full.csv (+ .json sidecar)
#   - data/csv/mvtec-ad-inverted-entropy-full.csv (+ .json sidecar)
#   - data/csv/kolektorsdd2-entropy-full.csv (+ .json sidecar)
#   - data/csv/kolektorsdd2-inverted-entropy-full.csv (+ .json sidecar)
# =====================================================================
if ($Stage -in @('all', '4_asymmetry')) {
    Show-StageHeader -Number "04" `
        -Title "SPATIAL ASYMMETRY FEATURE APPENDING (x85 -> x94)" `
        -Subtitle "Extracting 9 Left-Right, Top-Bottom & Rot180 residuals (schema x94-v1)" `
        -Icon "§" `
        -Color DarkMagenta

    # 4.1 Process all configured datasets and inverted variants in one sweep
    Invoke-Step -Title "Append Asymmetry Features (All Datasets & Inverted Variants)" `
        -ArgumentList @("-m", "experiments.append_asymmetry_features", "--all")
}


# =====================================================================
#  STAGE 05: SUPERVISED CLASSIFICATION & ABLATION (x94) (Tasks 06A & 10)
# =====================================================================
# Trains lightweight classifiers (dt, knn, lr, lr_dot, rf) on x94.
# Runs automated 7-subset ablation (x85_only, x9_only, x94_full, E_1x1..E_8x8)
# and spatial feature importance breakdown (Gini / Permutation).
# Splits:
#   - KolektorSDD2: official presplit train/test.
#   - MVTec-AD: filtered (no ground_truth), in-memory stratified 80/20 (seed 42).
# Outputs saved to:
#   - results/classification/<dataset>[_inverted]/<target>_<classifier>/
# =====================================================================
if ($Stage -in @('all', '5_ml')) {
    Show-StageHeader -Number "05" `
        -Title "SUPERVISED CLASSIFICATION & ABLATION ON x94" `
        -Subtitle "Training dt, knn, lr, lr_dot, rf with 7-subset ablation & heatmaps" `
        -Icon "★" `
        -Color Green

    # 5.1 KolektorSDD2: All classifiers (dt, knn, lr, lr_dot, rf) standard
    Invoke-Step -Title "Evaluate Classifiers: KolektorSDD2 (Standard, all models)" `
        -ArgumentList @("-m", "experiments.evaluate_descriptors", "--config", "configs/classification/kolektorsdd2.toml", "--classifier", "all", "--overwrite")

    # 5.2 KolektorSDD2: All classifiers inverted
    Invoke-Step -Title "Evaluate Classifiers: KolektorSDD2 (Inverted, all models)" `
        -ArgumentList @("-m", "experiments.evaluate_descriptors", "--config", "configs/classification/kolektorsdd2.toml", "--classifier", "all", "--inverted", "--overwrite")

    # 5.3 MVTec-AD Bottle (Canonical benchmark object): All classifiers standard
    Invoke-Step -Title "Evaluate Classifiers: MVTec-AD Bottle (Standard, all models)" `
        -ArgumentList @("-m", "experiments.evaluate_descriptors", "--config", "configs/classification/mvtec_ad.toml", "--object", "bottle", "--classifier", "all", "--overwrite")

    # 5.4 MVTec-AD Bottle (Canonical benchmark object): All classifiers inverted
    Invoke-Step -Title "Evaluate Classifiers: MVTec-AD Bottle (Inverted, all models)" `
        -ArgumentList @("-m", "experiments.evaluate_descriptors", "--config", "configs/classification/mvtec_ad.toml", "--object", "bottle", "--classifier", "all", "--inverted", "--overwrite")

    # 5.5 MVTec-AD All 15 Objects: Random Forest sweep
    if (-not $SkipAllObjects) {
        Invoke-Step -Title "Evaluate Classifiers: MVTec-AD All 15 Objects Sweep (Standard, Random Forest)" `
            -ArgumentList @("-m", "experiments.evaluate_descriptors", "--config", "configs/classification/mvtec_ad.toml", "--all-objects", "--classifier", "rf")

        Invoke-Step -Title "Evaluate Classifiers: MVTec-AD All 15 Objects Sweep (Inverted, Random Forest)" `
            -ArgumentList @("-m", "experiments.evaluate_descriptors", "--config", "configs/classification/mvtec_ad.toml", "--all-objects", "--classifier", "rf", "--inverted")
    }
}


# =====================================================================
#  STAGE 06: LIGHTWEIGHT CNN ZOO TRAINING & EVALUATION (Task 07)
# =====================================================================
# Trains 7 lightweight convolutional architectures:
#   ultra_tiny (1.1k), ds_cnn (12.8k), tiny_cnn (23.5k), mobilenet_like (70.7k),
#   tiny_resnet (77.2k), small_cnn (97.6k), torchvision_mobilenet (1.5M).
# Config: AdamW, lr=1e-3, cosine decay, 40 epochs, early stopping patience=8,
#         balanced class weights, threshold calibration on validation (val_f1).
# Hardware: CUDA GPU (NVIDIA GTX 1050 cu126).
# Outputs saved to:
#   - results/cnn/<dataset>[_inverted]/<target>/<model>/seed_42/
# =====================================================================
if ($Stage -in @('all', '6_cnn')) {
    if ($SkipCNN) {
        Write-Host "╔════════════════════════════════════════════════════════════════════════════╗" -ForegroundColor DarkGray
        Write-Host "║ [SKIPPED]  STAGE 06: LIGHTWEIGHT CNN ZOO (-SkipCNN switch active)          ║" -ForegroundColor Yellow
        Write-Host "╚════════════════════════════════════════════════════════════════════════════╝`n" -ForegroundColor DarkGray
    } else {
        Show-StageHeader -Number "06" `
            -Title "LIGHTWEIGHT CNN ZOO TRAINING & EVALUATION" `
            -Subtitle "Training 7 lightweight CNNs (ultra_tiny to mobilenet) with val_f1 calibration" `
            -Icon "▲" `
            -Color Red

        # 6.1 KolektorSDD2: 7 CNNs standard (seed 42)
        Invoke-Step -Title "Train CNN Zoo: KolektorSDD2 (Standard, all 7 models)" `
            -ArgumentList @("-m", "experiments.evaluate_cnn", "--config", "configs/cnn/kolektorsdd2.toml", "--model", "all", "--seeds", "42", "--overwrite")

        # 6.2 KolektorSDD2: 7 CNNs inverted (seed 42)
        Invoke-Step -Title "Train CNN Zoo: KolektorSDD2 (Inverted, all 7 models)" `
            -ArgumentList @("-m", "experiments.evaluate_cnn", "--config", "configs/cnn/kolektorsdd2.toml", "--model", "all", "--seeds", "42", "--inverted", "--overwrite")

        # 6.3 MVTec-AD Bottle (Canonical object): 7 CNNs standard (seed 42)
        Invoke-Step -Title "Train CNN Zoo: MVTec-AD Bottle (Standard, all 7 models)" `
            -ArgumentList @("-m", "experiments.evaluate_cnn", "--config", "configs/cnn/mvtec_ad.toml", "--object", "bottle", "--model", "all", "--seeds", "42", "--overwrite")

        # 6.4 MVTec-AD Bottle (Canonical object): 7 CNNs inverted (seed 42)
        Invoke-Step -Title "Train CNN Zoo: MVTec-AD Bottle (Inverted, all 7 models)" `
            -ArgumentList @("-m", "experiments.evaluate_cnn", "--config", "configs/cnn/mvtec_ad.toml", "--object", "bottle", "--model", "all", "--seeds", "42", "--inverted", "--overwrite")

        # 6.5 MVTec-AD All 15 Objects: All 7 models (Long execution)
        if (-not $SkipAllObjects) {
            Invoke-Step -Title "Train CNN Zoo: MVTec-AD All 15 Objects (Standard, all 7 models)" `
                -ArgumentList @("-m", "experiments.evaluate_cnn", "--config", "configs/cnn/mvtec_ad.toml", "--all-objects", "--model", "all", "--seeds", "42", "--overwrite")

            Invoke-Step -Title "Train CNN Zoo: MVTec-AD All 15 Objects (Inverted, all 7 models)" `
                -ArgumentList @("-m", "experiments.evaluate_cnn", "--config", "configs/cnn/mvtec_ad.toml", "--all-objects", "--model", "all", "--seeds", "42", "--inverted", "--overwrite")
        }
    }
}


# =====================================================================
#  STAGE 07: 100% CPU-FAIR BENCHMARKING (PHASE A) (Tasks 08 & 10)
# =====================================================================
# Measures end-to-end latency on 100% CPU (threads=1, batch=1 and batch=32):
#   ML  : T_total = T_descriptor (x94 via C99) + T_model
#   CNN : T_total = T_preprocess + T_model (CPU inference on loaded checkpoints)
# Uses 200 real test samples, gc disabled, 20 warmups, 200 timed iterations, 3 repeats.
# Checkpoints from Stage 6 are re-evaluated on CPU and verified to match within 1e-5.
# Outputs:
#   - results/comparison_cpu/<tag>_baseline_vs_cnn_cpu.csv
#   - results/comparison_cpu/summary_all_datasets.csv
#   - Pareto frontier plots (F1 vs latency, ROC-AUC vs latency)
# =====================================================================
if ($Stage -in @('all', '7_benchmark')) {
    Show-StageHeader -Number "07" `
        -Title "100% CPU-FAIR BENCHMARKING (PHASE A)" `
        -Subtitle "Measuring end-to-end CPU latency (threads=1, batch=1/32, Pareto frontiers)" `
        -Icon "⚡" `
        -Color Blue

    # 7.1 KolektorSDD2 Standard
    Invoke-Step -Title "100% CPU Benchmark: KolektorSDD2 (Standard)" `
        -ArgumentList @("-m", "experiments.benchmark_cpu_fair", "--config", "configs/profiling/kolektorsdd2.toml")

    # 7.2 KolektorSDD2 Inverted
    Invoke-Step -Title "100% CPU Benchmark: KolektorSDD2 (Inverted)" `
        -ArgumentList @("-m", "experiments.benchmark_cpu_fair", "--config", "configs/profiling/kolektorsdd2.toml", "--inverted")

    # 7.3 MVTec-AD Bottle Standard
    Invoke-Step -Title "100% CPU Benchmark: MVTec-AD Bottle (Standard)" `
        -ArgumentList @("-m", "experiments.benchmark_cpu_fair", "--config", "configs/profiling/mvtec_ad.toml", "--objects", "bottle")

    # 7.4 MVTec-AD Bottle Inverted
    Invoke-Step -Title "100% CPU Benchmark: MVTec-AD Bottle (Inverted)" `
        -ArgumentList @("-m", "experiments.benchmark_cpu_fair", "--config", "configs/profiling/mvtec_ad.toml", "--objects", "bottle", "--inverted")
}


# =====================================================================
#  STAGE 08: GRANULAR PIPELINE PROFILING (PHASE B) (Task 08)
# =====================================================================
# Deep scientific profiling decomposing where latency and memory reside:
#   - Descriptor stage breakdown: quantize, fine_hist, pyramid_sum, entropy_lut, asymmetry
#   - CNN layer breakdown & ATen operator tracing
#   - Non-parametric Mann-Whitney U statistical significance tests
#   - Quality vs. Cost trade-off: Delta_F1 / Delta_ms
# Outputs saved to:
#   - results/profiling/<tag>/ (environment.json, CSV tables, 6 publication PNGs, README.md)
# =====================================================================
if ($Stage -in @('all', '8_profiling')) {
    Show-StageHeader -Number "08" `
        -Title "GRANULAR PIPELINE PROFILING (PHASE B)" `
        -Subtitle "Stage breakdowns, CNN layer timings, ATen ops, and Mann-Whitney U tests" `
        -Icon "🔬" `
        -Color DarkCyan

    # 8.1 KolektorSDD2 Standard (full profiling)
    Invoke-Step -Title "Deep Profiling: KolektorSDD2 (Standard)" `
        -ArgumentList @("-m", "experiments.profile_pipelines", "--config", "configs/profiling/kolektorsdd2.toml")

    # 8.2 KolektorSDD2 Inverted (short descriptor check by default)
    Invoke-Step -Title "Deep Profiling: KolektorSDD2 (Inverted)" `
        -ArgumentList @("-m", "experiments.profile_pipelines", "--config", "configs/profiling/kolektorsdd2.toml", "--inverted")

    # 8.3 MVTec-AD Bottle Standard (full profiling)
    Invoke-Step -Title "Deep Profiling: MVTec-AD Bottle (Standard)" `
        -ArgumentList @("-m", "experiments.profile_pipelines", "--config", "configs/profiling/mvtec_ad.toml", "--objects", "bottle")

    # 8.4 MVTec-AD Bottle Inverted (short descriptor check by default)
    Invoke-Step -Title "Deep Profiling: MVTec-AD Bottle (Inverted)" `
        -ArgumentList @("-m", "experiments.profile_pipelines", "--config", "configs/profiling/mvtec_ad.toml", "--objects", "bottle", "--inverted")
}


# =====================================================================
#  STAGE 09: PURE-PYTHON PRE-MCU COMPARISON (OPTIONAL) (Task 08 & HYPOTHESIS.md)
# =====================================================================
# Measures CPU latency using 100% pure Python/NumPy backends
# (numpy_finest_pyramid / numpy_baseline) without relying on the C99 DLL.
# Proves that while quality metrics are bit-exact (<= 1.9e-15), pure Python
# loses in CPU latency to ultra_tiny due to Python interpreter overhead.
# Outputs saved to:
#   - results/comparison_cpu_python/ (Git-ignored)
# =====================================================================
if ($Stage -in @('9_python_cpu')) {
    Show-StageHeader -Number "09" `
        -Title "PURE-PYTHON PRE-MCU BENCHMARK (OPTIONAL)" `
        -Subtitle "Evaluating pure NumPy backends (numpy_finest_pyramid) without C99 DLL" `
        -Icon "🐍" `
        -Color DarkGray

    # 9.1 KolektorSDD2 pure Python
    Invoke-Step -Title "Pure-Python CPU Benchmark: KolektorSDD2" `
        -ArgumentList @("-m", "experiments.benchmark_cpu_fair", "--config", "configs/profiling/kolektorsdd2.toml", `
                        "--entropy-backend", "numpy_finest_pyramid", "--output-dir", "results/comparison_cpu_python", "--no-cuda-reference")

    # 9.2 MVTec-AD Bottle pure Python
    Invoke-Step -Title "Pure-Python CPU Benchmark: MVTec-AD Bottle" `
        -ArgumentList @("-m", "experiments.benchmark_cpu_fair", "--config", "configs/profiling/mvtec_ad.toml", `
                        "--objects", "bottle", "--entropy-backend", "numpy_finest_pyramid", "--output-dir", "results/comparison_cpu_python", "--no-cuda-reference")
}


# =====================================================================
#  STAGE 10: EMBEDDED C PIPELINE EXPORT & PARITY (Task 11)
# =====================================================================
# Exports self-contained C99 code, weights, Look-Up Tables, and test vectors
# for deployment on bare-metal microcontroller targets (ATmega328P, ESP32).
# Compiles desktop C test bench and verifies bit-exact q15 integer parity.
# Outputs:
#   - c_imp/generated/<target>/ (C headers, lookup tables, model weights, golden vectors)
#   - results/c_imp/<target>/ (summary_report.md, summary_models_metrics.csv, fig1..fig6)
# =====================================================================
if ($Stage -in @('10_c_embed')) {
    Show-StageHeader -Number "10" `
        -Title "EMBEDDED C PIPELINE EXPORT & PARITY (TASK 11)" `
        -Subtitle "Exporting C99 headers/weights & testing bit-exact q15 fixed-point parity" `
        -Icon "📟" `
        -Color DarkGreen

    # 10.1 KolektorSDD2: Export C pipeline headers, fixed-point weights, and test vectors
    Invoke-Step -Title "Export C Pipeline: KolektorSDD2" `
        -ArgumentList @("-m", "experiments.export_c_pipeline", "--config", "configs/c_imp/kolektorsdd2.toml")

    # 10.2 KolektorSDD2: Compile and run desktop C verification test bench with timing benchmark
    Invoke-Step -Title "Verify C Pipeline Desktop Parity: KolektorSDD2 (q15 bit-exact)" `
        -ArgumentList @("-m", "experiments.verify_c_pipeline", "--config", "configs/c_imp/kolektorsdd2.toml", "--benchmark")

    # 10.3 KolektorSDD2: Generate diagnostic summary tables and publication figures
    Invoke-Step -Title "Visualize C Pipeline Results: KolektorSDD2" `
        -ArgumentList @("-m", "experiments.visualize_c_pipeline", "--config", "configs/c_imp/kolektorsdd2.toml")

    # 10.4 MVTec-AD Bottle: Export C pipeline
    Invoke-Step -Title "Export C Pipeline: MVTec-AD Bottle" `
        -ArgumentList @("-m", "experiments.export_c_pipeline", "--config", "configs/c_imp/mvtec_ad.toml", "--object", "bottle")

    # 10.5 MVTec-AD Bottle: Compile and verify desktop C parity
    Invoke-Step -Title "Verify C Pipeline Desktop Parity: MVTec-AD Bottle (q15 bit-exact)" `
        -ArgumentList @("-m", "experiments.verify_c_pipeline", "--config", "configs/c_imp/mvtec_ad.toml", "--object", "bottle", "--benchmark")

    # 10.6 MVTec-AD Bottle: Generate diagnostic figures
    Invoke-Step -Title "Visualize C Pipeline Results: MVTec-AD Bottle" `
        -ArgumentList @("-m", "experiments.visualize_c_pipeline", "--config", "configs/c_imp/mvtec_ad.toml", "--object", "bottle")
}


# =====================================================================
#  STAGE 11: MICROCONTROLLER HARDWARE TEST BENCH (PC -> MCU -> PC) (Task 12)
# =====================================================================
# Discovers serial COM ports, validates staging images, evaluates
# hardware fit across boards (Nano, UNO, ESP8266, ESP32-CAM), and compiles
# the comprehensive hardware fit report.
# Note: Physical MCU flashing/sending ('flash', 'send') requires a board
# connected to a serial port (e.g. COM3).
# Outputs:
#   - results/hw_test/fit_report.csv & fit_report.md
#   - results/hw_test/<board>/<run_id>/
# =====================================================================
if ($Stage -in @('11_hw_test')) {
    Show-StageHeader -Number "11" `
        -Title "MICROCONTROLLER HARDWARE TEST BENCH (TASK 12)" `
        -Subtitle "Board catalog, test image staging, fit sweep & MCU benchmark reports" `
        -Icon "🔌" `
        -Color DarkYellow

    # 11.1 List configured boards, toolchain status, and detected COM ports
    Invoke-Step -Title "Hardware Test: Discover Boards & Serial Ports" `
        -ArgumentList @("-m", "experiments.hw_test", "boards", "--config", "configs/hw_test/boards.toml")

    # 11.2 Stage deterministic test images (4 good + 4 defect)
    Invoke-Step -Title "Hardware Test: Stage 8 Test Images (KolektorSDD2, seed 42)" `
        -ArgumentList @("-m", "experiments.hw_test", "images", "--config", "configs/hw_test/boards.toml", "--dataset", "kolektorsdd2", "--count", "8", "--seed", "42")

    # 11.3 Evaluate hardware fit matrix across all boards without physical hardware
    Invoke-Step -Title "Hardware Test: Unattended Fit-Only Benchmark (All Boards)" `
        -ArgumentList @("-m", "experiments.hw_test", "bench", "--config", "configs/hw_test/boards.toml", "--board", "all", "--fit-only")

    # 11.4 Generate consolidated fit report (fit_report.csv, fit_report.md)
    Invoke-Step -Title "Hardware Test: Generate Fit Report & Summary Tables" `
        -ArgumentList @("-m", "experiments.hw_test", "report", "--config", "configs/hw_test/boards.toml")

    <#
    # Interactive commands for physically connected ESP32-CAM (e.g. on COM3):
    Invoke-Step -Title "Flash ESP32-CAM (lr_dot, q15)" `
        -ArgumentList @("-m", "experiments.hw_test", "flash", "--config", "configs/hw_test/boards.toml", "--board", "esp32cam", "--port", "COM3", "--model", "lr_dot", "--variant", "q15")

    Invoke-Step -Title "Send Images to ESP32-CAM via Serial" `
        -ArgumentList @("-m", "experiments.hw_test", "send", "--config", "configs/hw_test/boards.toml", "--board", "esp32cam", "--transport", "serial", "--port", "COM3", "--model", "lr_dot", "--variant", "q15")
    #>
}

Write-Host "╔════════════════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                    RESEARCH PIPELINE EXECUTION COMPLETED                   ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan
