param(
    [Parameter(Mandatory = $true)]
    [string]$Dataset,

    [Parameter(Mandatory = $true)]
    [string]$Output,

    [switch]$InstallRequirements,
    [switch]$CleanOutput,

    [int]$FrameStep = 1,
    [int]$PixelStride = 2,
    [double]$VoxelM = 0.003,
    [double]$MLContamination = 0.035
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ScriptDir ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    $Python = "python"
}

if (-not (Test-Path $Dataset)) {
    throw "Dataset not found: $Dataset"
}

if ($InstallRequirements) {
    & $Python -m pip install -r (Join-Path $ScriptDir "requirements.txt")
}

if ($CleanOutput -and (Test-Path $Output)) {
    Remove-Item -LiteralPath $Output -Recurse -Force
}

& $Python (Join-Path $ScriptDir "sam2_rgbd_3pass_reconstruction.py") `
    --dataset $Dataset `
    --output $Output `
    --frame-step $FrameStep `
    --pixel-stride $PixelStride `
    --voxel-m $VoxelM `
    --ml-contamination $MLContamination
