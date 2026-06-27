# RLDSSAT Soybean

Pipeline para simular soja em Castro (PR) com DSSAT-CSM e treinar um agente PPO
para decisão de data de plantio e manejo de irrigação.

## Conteúdo

- `dssat_rl_soybean/`: código Python, configuração, scripts de execução e saídas finais.
- `base_consolidada_saida/`: base consolidada usada no experimento.
- `*.xlsx`: planilhas SIDRA/IBGE usadas na construção da base agrícola.
- `INMET_*.CSV` e `dados_A819_H_2006-07-08_2026-01-01.csv`: dados meteorológicos INMET usados na consolidação.
- `metodologia_rl_dssat.tex`: seção de metodologia em LaTeX.
- `resultados_rl_dssat.tex`: seção de resultados em LaTeX.

## Execução principal

No Windows/PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File "C:\Users\gabri\Documents\UTFPR\ArtigoMarcella\dssat_rl_soybean\run_dssat_training.ps1"
```

O script verifica o DSSAT real, calibra a produtividade simulada contra o SIDRA e
inicia o treinamento PPO.

## Resultados finais

- Calibração final: `dssat_rl_soybean/outputs/calibration_v4_row_bias_corrected/`
- Modelo de correção usado pelo DSSAT: `dssat_rl_soybean/outputs/calibration/yield_correction.json`
- Avaliação PPO: `dssat_rl_soybean/outputs/ppo_soja_castro_dssat/`

Arquivos de ambiente virtual, caches e saídas temporárias não são versionados.
