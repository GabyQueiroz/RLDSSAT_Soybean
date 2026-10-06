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

## Configuracao final (contexto em t0, DSSAT sem correcao)

A configuracao usada na avaliacao final esta em `configs/experiment_castro_t0.yaml`. Em relacao a rodada diagnostica anterior:

- a recompensa usa a produtividade simulada pelo DSSAT sem a correcao estatistica (`calibration.enabled: false`); a correcao ridge comprimia a resposta a data de semeadura e a irrigacao para poucos kg/ha e penalizava a irrigacao por conta propria;
- o contexto (`decision.context_mode: pre_season_enso`) contem apenas informacao disponivel em 1 de setembro: chuva dos 30 e 90 dias anteriores, temperatura media dos 30 dias anteriores, radiacao media dos 90 dias anteriores, ONI de MJJ e sua variacao desde FMA (`data/external/oni_psl_noaa.txt`, NOAA/PSL). O contexto e padronizado com media e desvio dos anos de treino (`models/context_scaler.json`);
- a simulacao comeca em 1 de agosto com o perfil na capacidade de campo (spin-up da agua do solo) e o arquivo de clima cobre ate 30 de junho do ano seguinte, para que semeaduras tardias tenham clima ate a maturacao;
- a regra de irrigacao e verificada a cada 4 dias, o gatilho vai de 10 a 150 mm e eventos menores que 5 mm nao sao aplicados (manejo de sequeiro fica acessivel no espaco continuo de acoes);
- a validacao durante o treino avalia cada safra de validacao uma vez (o DSSAT e deterministico);
- os ambientes de treino rodam em processos separados, cada um com sua propria pasta `DSSAT_HOME` (`dssat.isolate_home_per_process`), e as simulacoes repetidas sao lidas de um cache em memoria.

Fluxo completo (a partir da pasta `ArtigoMarcella`):

```powershell
python -m dssat_rl_soybean.src.final_comparators --config dssat_rl_soybean/configs/experiment_castro_t0.yaml --target valid
python -m dssat_rl_soybean.src.run_multiseed --config dssat_rl_soybean/configs/experiment_castro_t0.yaml --run-prefix ppo_castro_t0 --generated-dir _generated_ppo_castro_t0 --seeds 42,43,44,45,46
python -m dssat_rl_soybean.src.summarize_multiseed --run-prefix ppo_castro_t0
python -m dssat_rl_soybean.src.final_comparators --config dssat_rl_soybean/configs/experiment_castro_t0.yaml --target test --ppo-evaluation dssat_rl_soybean/outputs/ppo_castro_t0_summary/policy_evaluation_all_seeds.csv
```

O primeiro comando avalia a grade de regras nas safras de desenvolvimento (2006-2020) e compara, na validacao, regras escolhidas apenas com as safras de treino. O ultimo comando abre o bloco 2021-2024 uma unica vez: escolhe a regra fixa otimizada e a regra contextual kNN com todas as safras de desenvolvimento e calcula os efeitos pareados do PPO por safra e semente.

## Observacao

O arquivo `configs/experiment.yaml` esta pronto para execucao com perfil de solo padrao. Para artigo, o ideal e substituir esse perfil por `SOIL.SOL` local calibrado e cultivar DSSAT correspondente ao material usado na regiao.
