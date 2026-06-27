$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (!(Test-Path -LiteralPath $Python)) {
    throw "Nao encontrei o Python do ambiente virtual em $Python"
}

Write-Host "1/3 Conferindo se o DSSAT real roda..." -ForegroundColor Cyan
& $Python -m src.check_dssat_real --config configs/experiment.yaml --run-name dssat_real_check --year 2024 --planting-date 2024-10-15

Write-Host "2/3 Calibrando produtividade DSSAT contra SIDRA..." -ForegroundColor Cyan
& $Python -m src.calibrate_yield --config configs/experiment.yaml --run-name calibration_v4_row_bias_corrected

Write-Host "3/3 Iniciando treino PPO + DSSAT real com 500000 etapas..." -ForegroundColor Cyan
& $Python -m src.train_ppo --config configs/experiment.yaml --run-name ppo_soja_castro_dssat --backend dssat --timesteps 500000

Write-Host "Treino finalizado. Veja outputs\calibration_v4_row_bias_corrected e outputs\ppo_soja_castro_dssat" -ForegroundColor Green
