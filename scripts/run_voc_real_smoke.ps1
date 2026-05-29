$ErrorActionPreference = "Stop"

# Required environment variables:
#   VOC_ROOT       Path to VOCdevkit/VOC2012
#   SAM_CHECKPOINT Path to a SAM/MobileSAM checkpoint
#
# Optional environment variables:
#   OUTPUT_DIR     Output directory, default outputs/voc_real_smoke
#   LIMIT          Number of VOC val images to run, default 1
#   PYTHON         Python executable, default .\.venv\Scripts\python.exe

if ([string]::IsNullOrWhiteSpace($env:VOC_ROOT)) {
    Write-Error "VOC_ROOT is required. Example: `$env:VOC_ROOT='C:\path\to\VOCdevkit\VOC2012'"
    exit 1
}

if ([string]::IsNullOrWhiteSpace($env:SAM_CHECKPOINT)) {
    Write-Error "SAM_CHECKPOINT is required. Example: `$env:SAM_CHECKPOINT='C:\path\to\sam_vit_b.pth'"
    exit 1
}

$PythonExe = if ([string]::IsNullOrWhiteSpace($env:PYTHON)) { ".\.venv\Scripts\python.exe" } else { $env:PYTHON }
$Limit = if ([string]::IsNullOrWhiteSpace($env:LIMIT)) { "1" } else { $env:LIMIT }
$OutputDir = if ([string]::IsNullOrWhiteSpace($env:OUTPUT_DIR)) { "outputs/voc_real_smoke" } else { $env:OUTPUT_DIR }

if (-not (Test-Path -LiteralPath $PythonExe)) {
    Write-Error "Python executable was not found: $PythonExe. Set PYTHON to a valid Python executable."
    exit 1
}

if (-not (Test-Path -LiteralPath $env:VOC_ROOT)) {
    Write-Error "VOC_ROOT path does not exist: $env:VOC_ROOT"
    exit 1
}

if (-not (Test-Path -LiteralPath $env:SAM_CHECKPOINT)) {
    Write-Error "SAM_CHECKPOINT path does not exist: $env:SAM_CHECKPOINT"
    exit 1
}

& $PythonExe `
    eval_pascal_voc.py `
    --voc-root $env:VOC_ROOT `
    --split val `
    --limit $Limit `
    --output-dir $OutputDir `
    --semantic-backend clip `
    --mask-backend sam `
    --sam-checkpoint $env:SAM_CHECKPOINT `
    --sam-model-type vit_b `
    --feature-backend dinov2 `
    --dinov2-model facebook/dinov2-small `
    --max-masks 50 `
    --save-vis

exit $LASTEXITCODE
