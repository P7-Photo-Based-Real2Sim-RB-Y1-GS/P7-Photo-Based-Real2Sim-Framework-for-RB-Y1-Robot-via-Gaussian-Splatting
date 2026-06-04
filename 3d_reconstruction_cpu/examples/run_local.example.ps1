# Copy this file to run_local.ps1, then edit the paths below.

$Dataset = "D:\datasets\object_001"
$Output = "D:\datasets\object_001\reconstruction_realistic_bestview"

.\run_reconstruction.ps1 `
  -Dataset $Dataset `
  -Output $Output `
  -InstallRequirements `
  -CleanOutput

