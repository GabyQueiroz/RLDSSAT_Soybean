# RLDSSAT Soybean

Pipeline to simulate soybean crops in Castro (PR) using DSSAT-CSM and to train a PPO agent
for planting date and irrigation management decisions.

## Contents

- `dssat_rl_soybean/`: Python code, configuration, execution scripts, and final outputs.
- `base_consolidada_saida/`: Consolidated dataset used in the experiment.
- `*.xlsx`: SIDRA/IBGE spreadsheets used to build the agricultural dataset.
- `INMET_*.CSV` and `dados_A819_H_2006-07-08_2026-01-01.csv`: INMET weather data used for consolidation.

## Main Execution

On Windows/PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File "PATH"
```

The script verifies the actual DSSAT simulation, calibrates simulated yield against SIDRA data,
and initiates PPO training.

## Final Results

- Final calibration: `dssat_rl_soybean/outputs/calibration_v4_row_bias_corrected/`
- Correction model used by DSSAT: `dssat_rl_soybean/outputs/calibration/yield_correction.json`
- PPO evaluation: `dssat_rl_soybean/outputs/ppo_soja_castro_dssat/`

Virtual environment files, caches, and temporary outputs are not version-controlled.
