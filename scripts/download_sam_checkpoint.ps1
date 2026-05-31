param(
    [string]$ModelDir = $(if ([string]::IsNullOrWhiteSpace($env:MODEL_DIR)) { "D:\models" } else { $env:MODEL_DIR }),
    [string]$CheckpointName = "sam_vit_b_01ec64.pth",
    [string]$Url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
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
        Write-Error "$VariableName must point outside this repository so model weights are not committed. Current value: $CandidatePath"
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
        Write-Host "Downloading SAM ViT-B checkpoint..."
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

        Write-Error "Failed to download SAM checkpoint from '$Uri' to '$Destination'. $($_.Exception.Message)"
        exit 1
    }
}

if ([System.IO.Path]::IsPathRooted($CheckpointName) -or $CheckpointName.Contains("\") -or $CheckpointName.Contains("/")) {
    Write-Error "CheckpointName must be a file name, not a path. Current value: $CheckpointName"
    exit 1
}

$modelDirFull = Get-FullPath $ModelDir
Assert-OutsideRepository -CandidatePath $modelDirFull -VariableName "MODEL_DIR"

try {
    New-Item -ItemType Directory -Path $modelDirFull -Force | Out-Null
}
catch {
    Write-Error "Failed to create MODEL_DIR '$modelDirFull'. $($_.Exception.Message)"
    exit 1
}

$checkpointPath = Join-Path $modelDirFull $CheckpointName

if (Test-Path -LiteralPath $checkpointPath) {
    $checkpoint = Get-Item -LiteralPath $checkpointPath
    if ($checkpoint.Length -le 0) {
        Write-Error "Existing checkpoint is empty: $checkpointPath. Delete it and rerun this script."
        exit 1
    }

    Write-Host "SAM checkpoint already exists, skipping download: $checkpointPath"
}
else {
    Save-Download -Uri $Url -Destination $checkpointPath
}

Write-Host "SAM checkpoint validation passed."
Write-Host "Use this environment variable for the current PowerShell session:"
Write-Host (Format-EnvironmentAssignment -Name "SAM_CHECKPOINT" -Value $checkpointPath)
