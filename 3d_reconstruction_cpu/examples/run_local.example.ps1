$Dataset = "D:\datasets\object_001"
$Output = "D:\datasets\object_001\sam2_rgbd_reconstruction"

.\run_reconstruction.ps1 `
  -Dataset $Dataset `
  -Output $Output `
  -InstallRequirements `
  -CleanOutput
