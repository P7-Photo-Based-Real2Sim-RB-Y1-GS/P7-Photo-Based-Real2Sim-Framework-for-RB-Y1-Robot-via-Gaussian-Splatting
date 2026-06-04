param(
    [string]$Dataset = "",
    [string]$Output = "",
    [string]$OutputName = "reconstruction_realistic_bestview",
    [ValidateSet("visual-hull", "tsdf", "both", "all", "poisson", "bpa", "box")]
    [string]$MeshMethod = "visual-hull",
    [ValidateSet("turntable", "register", "reuse")]
    [string]$PoseMode = "turntable",
    [int]$Every = 1,
    [int]$MaxFrames = 0,
    [switch]$InstallRequirements,
    [switch]$CleanOutput
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ScriptDir ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    $Python = "python"
}

if ($Dataset -eq "") {
    $Dataset = Join-Path $ScriptDir "data\object_001"
}

if (-not (Test-Path $Dataset)) {
    throw @"
Dataset folder not found: $Dataset

Pass your dataset path explicitly:
  .\run_reconstruction.ps1 -Dataset "D:\datasets\object_001"

Expected dataset layout:
  <dataset>\rgb
  <dataset>\depth
  <dataset>\masks
  <dataset>\cam_K.txt
  <dataset>\frame_index.csv
"@
}

if ($Output -eq "") {
    $Output = Join-Path $Dataset $OutputName
}

if ($InstallRequirements) {
    & $Python -m pip install -r (Join-Path $ScriptDir "requirements.txt")
}

if ($CleanOutput -and (Test-Path $Output)) {
    Remove-Item -LiteralPath $Output -Recurse -Force
}

$ArgsList = @(
    (Join-Path $ScriptDir "reconstruct_rgbd_object.py"),
    "--dataset", $Dataset,
    "--output", $Output,
    "--pose-mode", $PoseMode,
    "--turntable-angle-sign", "auto",
    "--mesh-method", $MeshMethod,
    "--export-object-crops",
    "--depth-mask-source", "expanded-bbox",
    "--color-object-filter", "yellow",
    "--model-refine-iterations", "1",
    "--foreground-depth-filter",
    "--foreground-depth-span-m", "0.18",
    "--foreground-gap-m", "0.08",
    "--foreground-gap-margin-m", "0.02",
    "--visual-hull-voxel-size-m", "0.002",
    "--visual-hull-frame-step", "1",
    "--visual-hull-min-hit-ratio", "0.78",
    "--visual-hull-mask-dilate-px", "5",
    "--visual-hull-padding-m", "0.026",
    "--visual-hull-smooth-iterations", "1",
    "--visual-hull-color-mode", "best-view",
    "--visual-hull-color-smooth-iterations", "1",
    "--no-loop-correction",
    "--every", "$Every"
)

if ($MaxFrames -gt 0) {
    $ArgsList += @("--max-frames", "$MaxFrames")
}

Write-Host "Dataset: $Dataset"
Write-Host "Output : $Output"

& $Python @ArgsList

