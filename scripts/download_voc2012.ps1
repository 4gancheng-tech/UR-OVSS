param(
    [string]$DatasetDir = $(if ([string]::IsNullOrWhiteSpace($env:DATASET_DIR)) { "D:\datasets" } else { $env:DATASET_DIR }),
    [string]$Url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Get-FullPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    return [System.IO.Path]::GetFullPath($Path)
}

function Add-TrailingSeparator {
    param([Parameter(Mandatory = $true)][string]$Path)

    $trimmed = $Path.TrimEnd([char[]]@('\', '/'))
    return $trimmed + [System.IO.Path]::DirectorySeparatorChar
}

function Assert-OutsideRepository {
    param(
        [Parameter(Mandatory = $true)][string]$CandidatePath,
        [Parameter(Mandatory = $true)][string]$VariableName
    )

    $repoRoot = Add-TrailingSeparator (Get-FullPath (Join-Path $PSScriptRoot ".."))
    $candidate = Add-TrailingSeparator (Get-FullPath $CandidatePath)

    if ($candidate.StartsWith($repoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        Write-Error "$VariableName must point outside this repository so datasets are not committed. Current value: $CandidatePath"
        exit 1
    }
}

function Format-EnvironmentAssignment {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Value
    )

    return '$env:' + $Name + '="' + $Value + '"'
}

function Save-Download {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    $partial = "$Destination.partial"

    if (Test-Path -LiteralPath $partial) {
        Remove-Item -LiteralPath $partial -Force
    }

    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    }
    catch {
        # Older PowerShell/.NET combinations may not expose this setting.
    }

    try {
        Write-Host "Downloading Pascal VOC 2012 train/val archive..."
        Write-Host "Source: $Uri"
        Write-Host "Target: $Destination"
        Invoke-WebRequest -Uri $Uri -OutFile $partial -UseBasicParsing

        if (-not (Test-Path -LiteralPath $partial)) {
            throw "Download did not create $partial"
        }

        $downloaded = Get-Item -LiteralPath $partial
        if ($downloaded.Length -le 0) {
            throw "Downloaded file is empty: $partial"
        }

        Move-Item -LiteralPath $partial -Destination $Destination -Force
    }
    catch {
        if (Test-Path -LiteralPath $partial) {
            Remove-Item -LiteralPath $partial -Force
        }

        Write-Error "Failed to download Pascal VOC 2012 from '$Uri' to '$Destination'. $($_.Exception.Message)"
        exit 1
    }
}

function Get-MissingVocPaths {
    param([Parameter(Mandatory = $true)][string]$VocRoot)

    $required = @(
        "JPEGImages",
        "SegmentationClass",
        "ImageSets\Segmentation\val.txt"
    )

    foreach ($relativePath in $required) {
        $fullPath = Join-Path $VocRoot $relativePath
        if (-not (Test-Path -LiteralPath $fullPath)) {
            $relativePath
        }
    }
}

$datasetDirFull = Get-FullPath $DatasetDir
Assert-OutsideRepository -CandidatePath $datasetDirFull -VariableName "DATASET_DIR"

try {
    New-Item -ItemType Directory -Path $datasetDirFull -Force | Out-Null
}
catch {
    Write-Error "Failed to create DATASET_DIR '$datasetDirFull'. $($_.Exception.Message)"
    exit 1
}

$archivePath = Join-Path $datasetDirFull "VOCtrainval_11-May-2012.tar"
$vocRoot = Join-Path (Join-Path $datasetDirFull "VOCdevkit") "VOC2012"
$missingPaths = @(Get-MissingVocPaths -VocRoot $vocRoot)

if ($missingPaths.Count -eq 0) {
    Write-Host "Pascal VOC 2012 is already available at: $vocRoot"
}
else {
    if (Test-Path -LiteralPath $archivePath) {
        $archive = Get-Item -LiteralPath $archivePath
        if ($archive.Length -le 0) {
            Write-Error "Existing archive is empty: $archivePath. Delete it and rerun this script."
            exit 1
        }

        Write-Host "Found existing archive, skipping download: $archivePath"
    }
    else {
        Save-Download -Uri $Url -Destination $archivePath
    }

    $tar = Get-Command tar.exe -ErrorAction SilentlyContinue
    if ($null -eq $tar) {
        Write-Error "tar.exe was not found. Install a tar-capable tool or extract '$archivePath' into '$datasetDirFull' manually."
        exit 1
    }

    Write-Host "Extracting archive into: $datasetDirFull"
    & $tar.Source -xf $archivePath -C $datasetDirFull
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to extract '$archivePath' with tar.exe. Exit code: $LASTEXITCODE"
        exit $LASTEXITCODE
    }

    $missingPaths = @(Get-MissingVocPaths -VocRoot $vocRoot)
    if ($missingPaths.Count -gt 0) {
        Write-Error "Pascal VOC 2012 validation failed. Missing paths under '$vocRoot': $($missingPaths -join ', ')"
        exit 1
    }
}

Write-Host "Pascal VOC 2012 validation passed."
Write-Host "Use this environment variable for the current PowerShell session:"
Write-Host (Format-EnvironmentAssignment -Name "VOC_ROOT" -Value $vocRoot)
