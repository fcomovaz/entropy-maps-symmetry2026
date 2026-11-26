<#
.SYNOPSIS
    Cleans up generated research results to allow restarting experiments from scratch.

.DESCRIPTION
    Safely removes generated result files (such as .csv and .json) from the results/ directory.
    - Strictly preserves directory structure (does NOT delete any folders).
    - Strictly preserves documentation (.md files such as README.md and environment_debug.md).
    - Strictly preserves git tracking (.gitkeep files).
    - Strictly preserves the data/ folder (raw, preprocessed images, and baseline CSVs are untouched).
    - Supports native -WhatIf / -DryRun to preview files before deleting.
    - Supports -AllArtifacts to also remove generated plots (.png), model checkpoints (.pt),
      cProfile traces (.txt), binary test vectors (.bin), and logs (.log) in results/.

.PARAMETER DryRun
    Preview files that would be removed without deleting anything. Equivalent to -WhatIf.

.PARAMETER AllArtifacts
    If set, removes all generated artifacts in results/ (.csv, .json, .png, .pt, .txt, .bin, .log)
    while still preserving folders, .md documentation, and .gitkeep files.
    Default is false (only removes .csv and .json files).

.PARAMETER TargetDirectory
    The results directory to clean. Defaults to 'results' relative to this script's directory.

.EXAMPLE
    # Preview which CSV and JSON files would be removed
    .\cleanup.ps1 -DryRun

.EXAMPLE
    # Remove all CSV and JSON results (default behavior)
    .\cleanup.ps1

.EXAMPLE
    # Remove all results including plots (.png) and checkpoints (.pt)
    .\cleanup.ps1 -AllArtifacts

.EXAMPLE
    # Native PowerShell WhatIf check
    .\cleanup.ps1 -WhatIf
#>

[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Medium')]
param(
    [Parameter(Mandatory = $false)]
    [switch]$DryRun,

    [Parameter(Mandatory = $false)]
    [switch]$AllArtifacts,

    [Parameter(Mandatory = $false)]
    [string]$TargetDirectory = ""
)

# Set strict mode and error preference
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Resolve script root reliably
$scriptDir = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($scriptDir)) {
    $scriptDir = (Get-Location).Path
} else {
    $scriptDir = (Resolve-Path -LiteralPath $scriptDir).Path
}

if ([string]::IsNullOrWhiteSpace($TargetDirectory)) {
    $TargetDirectory = Join-Path $scriptDir "results"
}

# Enable UTF-8 console output for crisp borders across PS versions
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
} catch {}

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                    RESEARCH RESULTS CLEANUP UTILITY                        ║" -ForegroundColor Cyan
Write-Host "║              Safely purging artifacts to restart fresh research            ║" -ForegroundColor Gray
Write-Host "╚════════════════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# 1. Resolve and validate target path
$resolvedTarget = $null
try {
    $resolvedTarget = (Resolve-Path -LiteralPath $TargetDirectory).Path
} catch {
    Write-Warning "Target results directory not found: '$TargetDirectory'. Nothing to clean."
    exit 0
}

# 2. Safety Guards: Verify this is inside a results folder and NEVER touches data/ or configs/
$dataDir = Join-Path $scriptDir "data"
$configsDir = Join-Path $scriptDir "configs"

if ($resolvedTarget.StartsWith($dataDir, [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Error "CRITICAL SAFETY ABORT: Target path '$resolvedTarget' is inside the data/ directory! Data folder must NEVER be touched."
    exit 1
}

if ($resolvedTarget.StartsWith($configsDir, [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Error "CRITICAL SAFETY ABORT: Target path '$resolvedTarget' is inside the configs/ directory! Configs folder must NEVER be touched."
    exit 1
}

$dirName = (Get-Item -LiteralPath $resolvedTarget).Name
if ($dirName -ne "results" -and -not $resolvedTarget.Contains([System.IO.Path]::DirectorySeparatorChar + "results")) {
    Write-Error "CRITICAL SAFETY ABORT: Target path '$resolvedTarget' does not appear to be a valid results directory."
    exit 1
}

# 3. Define target extensions
$validExtensions = if ($AllArtifacts) {
    @(".csv", ".json", ".png", ".pt", ".txt", ".bin", ".log")
} else {
    @(".csv", ".json")
}

$filterDisplay = if ($AllArtifacts) {
    "*.csv, *.json, *.png, *.pt, *.txt, *.bin, *.log"
} else {
    "*.csv, *.json (files only, folders and other formats preserved)"
}

Write-Host "  • Target Directory : " -NoNewline -ForegroundColor DarkGray
Write-Host "$resolvedTarget" -ForegroundColor Gray
Write-Host "  • File Filters     : " -NoNewline -ForegroundColor DarkGray
Write-Host "$filterDisplay" -ForegroundColor Yellow
Write-Host "  • Operation Mode   : " -NoNewline -ForegroundColor DarkGray
Write-Host "$(if ($DryRun -or $WhatIfPreference) { 'PREVIEW ONLY (Dry-Run)' } else { 'DELETE FILES' })" -ForegroundColor $(if ($DryRun -or $WhatIfPreference) { 'Yellow' } else { 'Green' })
Write-Host "  • Folder Policy    : " -NoNewline -ForegroundColor DarkGray
Write-Host "Directories are strictly preserved (folders will NOT be deleted)" -ForegroundColor Gray
Write-Host "  • Protection Policy: " -NoNewline -ForegroundColor DarkGray
Write-Host ".md documentation and .gitkeep files are strictly protected" -ForegroundColor Gray
Write-Host "  • Data Policy      : " -NoNewline -ForegroundColor DarkGray
Write-Host "data/ and configs/ directories are completely excluded and untouched" -ForegroundColor Gray
Write-Host "────────────────────────────────────────────────────────────────────────────`n" -ForegroundColor DarkGray

# 4. Enumerate files to delete
$protectedExtensions = @(".md", ".gitkeep")
$protectedFilenames = @("environment_debug.md", "README.md", ".gitkeep")

$allFiles = Get-ChildItem -LiteralPath $resolvedTarget -Recurse -File -ErrorAction SilentlyContinue

# Filter strictly by allowed extensions and protect documentation/gitkeep
$candidates = @()
foreach ($file in $allFiles) {
    $ext = $file.Extension.ToLowerInvariant()
    $name = $file.Name.ToLowerInvariant()
    
    if ($protectedExtensions -contains $ext -or $protectedFilenames -contains $name) {
        continue
    }

    if ($validExtensions -contains $ext) {
        $candidates += $file
    }
}

if ($candidates.Count -eq 0) {
    Write-Host "✔ No matching result files found to clean in '$resolvedTarget'. Directory is already clean.`n" -ForegroundColor Green
    exit 0
}

# 5. Display files & execute deletion
$totalBytes = 0
$deletedCount = 0

Write-Host "Found $($candidates.Count) candidate file(s):`n" -ForegroundColor White

foreach ($file in $candidates) {
    $totalBytes += $file.Length
    $relPath = $file.FullName.Substring($scriptDir.Length).TrimStart('\', '/')
    
    if ($DryRun -or $WhatIfPreference) {
        Write-Host "  [WOULD DELETE] " -NoNewline -ForegroundColor DarkYellow
        Write-Host "$relPath " -NoNewline -ForegroundColor White
        Write-Host "($([math]::Round($file.Length / 1KB, 2)) KB)" -ForegroundColor DarkGray
    } else {
        if ($PSCmdlet.ShouldProcess($relPath, "Delete file")) {
            try {
                Remove-Item -LiteralPath $file.FullName -Force
                Write-Host "  [DELETED]      " -NoNewline -ForegroundColor Green
                Write-Host "$relPath" -ForegroundColor Gray
                $deletedCount++
            } catch {
                Write-Warning "Failed to delete '$relPath': $_"
            }
        }
    }
}

$sizeMb = [math]::Round($totalBytes / 1MB, 2)

Write-Host "`n────────────────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
if ($DryRun -or $WhatIfPreference) {
    Write-Host "╔════════════════════════════════════════════════════════════════════════════╗" -ForegroundColor Yellow
    Write-Host "║ SUMMARY (DRY RUN):                                                         ║" -ForegroundColor Yellow
    Write-Host "║   Candidate Files : $($candidates.Count.ToString().PadRight(56))║" -ForegroundColor Yellow
    Write-Host "║   Total Volume    : $(($sizeMb.ToString() + ' MB').PadRight(56))║" -ForegroundColor Yellow
    Write-Host "║   Action          : No files were deleted. Re-run without -DryRun.         ║" -ForegroundColor Yellow
    Write-Host "╚════════════════════════════════════════════════════════════════════════════╝`n" -ForegroundColor Yellow
} else {
    Write-Host "╔════════════════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║ SUMMARY:                                                                   ║" -ForegroundColor Green
    Write-Host "║   Deleted Files   : $($deletedCount.ToString().PadRight(56))║" -ForegroundColor Green
    Write-Host "║   Freed Space     : $(($sizeMb.ToString() + ' MB').PadRight(56))║" -ForegroundColor Green
    Write-Host "║   Status          : Results folder is clean and ready for a fresh run!     ║" -ForegroundColor Green
    Write-Host "╚════════════════════════════════════════════════════════════════════════════╝`n" -ForegroundColor Green
}
