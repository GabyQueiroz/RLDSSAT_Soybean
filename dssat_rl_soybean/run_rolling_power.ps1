# Rolling-origin evaluation on the NASA POWER series: candidate grid, PPO folds and analysis.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$log = "outputs/rolling_power_pipeline.log"

python -u -m src.rolling_origin grid --workers 11 2>&1 | Out-File -Append -Encoding utf8 $log
if ($LASTEXITCODE -ne 0) { throw "candidate grid failed" }
python -u -m src.rolling_origin train 2>&1 | Out-File -Append -Encoding utf8 $log
if ($LASTEXITCODE -ne 0) { throw "rolling training failed" }
python -u -m src.rolling_origin analyse 2>&1 | Out-File -Append -Encoding utf8 $log
if ($LASTEXITCODE -ne 0) { throw "analysis failed" }
"PIPELINE FINISHED $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
