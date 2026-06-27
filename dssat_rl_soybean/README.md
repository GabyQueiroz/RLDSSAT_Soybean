# PPO + DSSAT para soja em Castro, PR

Este projeto calibra o DSSAT contra produtividade observada de soja em Castro e treina um agente PPO para escolher data de plantio e manejo de irrigacao.

## Dados

- Clima horario INMET Castro A819: `../base_consolidada_saida/clima_castro_horario.csv`
- Produtividade SIDRA/IBGE: `../base_consolidada_saida/produtividade_castro_ponta_grossa_longa.csv`

Divisao temporal:

- treino: 2006-2017
- validacao: 2018-2021
- teste: 2022-2024 para produtividade observada
- 2025 fica disponivel para simulacao climatica, mas sem produtividade SIDRA observada na base atual

## Comando principal

Execute tudo com:

```powershell
powershell -ExecutionPolicy Bypass -File "C:\Users\gabri\Documents\UTFPR\ArtigoMarcella\dssat_rl_soybean\run_dssat_training.ps1"
```

Esse script:

1. verifica se o DSSAT real executa;
2. calibra a produtividade DSSAT contra SIDRA;
3. treina PPO com DSSAT real por ate 500000 etapas.

## Calibracao final

Para rodar apenas a calibracao final:

```powershell
cd C:\Users\gabri\Documents\UTFPR\ArtigoMarcella\dssat_rl_soybean
.\.venv\Scripts\python.exe -m src.calibrate_yield --config configs/experiment.yaml --run-name calibration_v4_row_bias_corrected
```

Saidas principais:

- `outputs/calibration/yield_correction.json`: modelo de correcao usado pelo DSSAT no PPO;
- `outputs/calibration_v4_row_bias_corrected/tables/yield_correction_metrics_by_split.csv`;
- `outputs/calibration_v4_row_bias_corrected/tables/yield_calibration_summary.csv`;
- `outputs/calibration_v4_row_bias_corrected/figures/dssat_observed_vs_corrected_yield_en.png`;
- `outputs/calibration_v4_row_bias_corrected/figures/dssat_correction_error_by_year_en.png`.

Metricas finais da calibracao:

| split | MAE (kg/ha) | RMSE (kg/ha) | MAPE (%) | bias (kg/ha) |
|---|---:|---:|---:|---:|
| train | 229.56 | 319.94 | 7.49 | 136.40 |
| valid | 109.06 | 138.63 | 2.82 | 0.00 |
| test | 111.82 | 146.95 | 2.75 | -64.44 |

## Treino PPO

Treino DSSAT real:

```powershell
cd C:\Users\gabri\Documents\UTFPR\ArtigoMarcella\dssat_rl_soybean
.\.venv\Scripts\python.exe -m src.train_ppo --config configs/experiment.yaml --run-name ppo_soja_castro_dssat --backend dssat --timesteps 500000
```

O PPO usa:

- politica MLP com camadas `[128, 128, 64]` para ator e critico;
- `n_envs = 8`;
- `n_steps = 256`;
- `batch_size = 512`;
- `learning_rate = 0.0002`;
- `clip_range = 0.15`;
- `ent_coef = 0.01`;
- early stopping por validacao.

## Observacao

O arquivo `configs/experiment.yaml` esta pronto para execucao com perfil de solo padrao. Para artigo, o ideal e substituir esse perfil por `SOIL.SOL` local calibrado e cultivar DSSAT correspondente ao material usado na regiao.
