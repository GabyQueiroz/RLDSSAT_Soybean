# TESTE_EXTERNO_COMPOSICAO_v6

- Version: 1
- Date: 2026-09-27
- Environment: Castro_A819
- Status: diagnostic_not_final
- Registry SHA-256: `324098422b36b33d3311920dfb73a1e9d25e178d82f34455b8db86ee3d3ab1d9`

## Blocking rule

The frozen evaluation years must not be used to choose context variables, action bounds, correction models, water penalty, PPO hyperparameters, checkpoint, baselines, or manuscript interpretation before the final run.

## Temporal blocks

| registry_name               |   registry_version | registry_date   | environment   | block                  | years                                                  |   n_years | role                                                 |
|:----------------------------|-------------------:|:----------------|:--------------|:-----------------------|:-------------------------------------------------------|----------:|:-----------------------------------------------------|
| TESTE_EXTERNO_COMPOSICAO_v6 |                  1 | 2026-09-27      | Castro_A819   | development_train      | 2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016 |        11 | model fitting and policy learning                    |
| TESTE_EXTERNO_COMPOSICAO_v6 |                  1 | 2026-09-27      | Castro_A819   | development_validation | 2017,2018,2019,2020                                    |         4 | checkpoint and development choices                   |
| TESTE_EXTERNO_COMPOSICAO_v6 |                  1 | 2026-09-27      | Castro_A819   | frozen_evaluation      | 2021,2022,2023,2024                                    |         4 | opened once after final choices are frozen           |
| TESTE_EXTERNO_COMPOSICAO_v6 |                  1 | 2026-09-27      | Castro_A819   | illustration           | 2025                                                   |         1 | unobserved scenario; excluded from empirical metrics |

## Weather summary

| block                  | years                                                  |   n_years |   rain_mean_mm |   rain_min_mm |   rain_max_mm |   temp_mean_c |   srad_mean_mj_m2_day |
|:-----------------------|:-------------------------------------------------------|----------:|---------------:|--------------:|--------------:|--------------:|----------------------:|
| development_train      | 2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016 |        11 |         892.75 |        380.95 |       1270.80 |         19.18 |                 17.21 |
| development_validation | 2017,2018,2019,2020                                    |         4 |        1024.75 |        746.00 |       1244.20 |         19.65 |                 15.74 |
| frozen_evaluation      | 2021,2022,2023,2024                                    |         4 |         980.10 |        873.38 |       1171.64 |         18.46 |                 17.53 |
| illustration           | 2025                                                   |         1 |         936.32 |        936.32 |        936.32 |         19.97 |                 17.44 |

## Observed-yield coverage

| block                  |   year |   observed_yield_kg_ha | has_observed_yield   |
|:-----------------------|-------:|-----------------------:|:---------------------|
| development_train      |   2006 |                3000.00 | True                 |
| development_train      |   2007 |                3300.00 | True                 |
| development_train      |   2008 |                3126.00 | True                 |
| development_train      |   2009 |                2578.00 | True                 |
| development_train      |   2010 |                3245.00 | True                 |
| development_train      |   2011 |                3496.00 | True                 |
| development_train      |   2012 |                3546.00 | True                 |
| development_train      |   2013 |                3690.00 | True                 |
| development_train      |   2014 |                3542.00 | True                 |
| development_train      |   2015 |                3727.00 | True                 |
| development_train      |   2016 |                3381.00 | True                 |
| development_validation |   2017 |                4000.00 | True                 |
| development_validation |   2018 |                3750.00 | True                 |
| development_validation |   2019 |                3700.00 | True                 |
| development_validation |   2020 |                4050.00 | True                 |
| frozen_evaluation      |   2021 |                3700.00 | True                 |
| frozen_evaluation      |   2022 |                3923.00 | True                 |
| frozen_evaluation      |   2023 |                4154.00 | True                 |
| frozen_evaluation      |   2024 |                3827.00 | True                 |
| illustration           |   2025 |                 nan    | False                |

## Validation

- ok
