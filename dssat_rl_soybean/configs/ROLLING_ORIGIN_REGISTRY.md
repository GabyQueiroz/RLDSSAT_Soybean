# Rolling-origin evaluation on the NASA POWER series (registered before any run)

Registered: 2026-10-07, before the candidate grid, the PPO runs and the comparators of this design were executed.

## Weather

- Source: NASA POWER daily point API, community AG, latitude -24.78, longitude -50.0464, 1981-01-01 to 2026-05-31,
  local solar time. Parameters PRECTOTCORR, T2M, T2M_MAX, T2M_MIN (MERRA-2) and ALLSKY_SFC_SW_DWN (CERES/GEWEX).
  File: `data/external/nasa_power/power_castro_daily_1981_2026.csv` (unmodified API output).
- Radiation is missing for 1981-1983, so the seasons are 1984-2024 (41 seasons, sowing year).
- INMET A819 is not used in this design except for the agreement table (`src/power_station_agreement.py`).

## Folds

Test seasons 2005-2024 in pairs. For a fold whose test seasons start in s:

| role | seasons |
|---|---|
| PPO training | 1984 .. s-6 |
| PPO checkpoint selection | s-5 .. s-1 |
| test | s, s+1 |
| fixed and kNN rule selection | 1984 .. s-1 |

Folds: f2005, f2007, ..., f2023 (10 folds, 20 test seasons). Seeds 42, 43, 44.

## Unchanged from the frozen t0 configuration

Context (rain 30/90 d, temperature 30 d, radiation 90 d, ONI MJJ and MJJ-FMA, standardized with the training seasons of the fold),
action bounds, irrigation rule, spin-up, objective (kappa = 0.0015 mm-1), PPO architecture and hyperparameters
(context noise 0.75, two layers of 32 units), candidate grid of 602 rules, kNN with k = 3.

## Changes fixed here

- Training budget per run: at least 8,192 and at most 24,576 steps; stop after 6 evaluations without improvement
  (evaluation every 2,048 steps on the 5 validation seasons).

## Reported outcomes

Mean objective, yield, irrigation, P10 and CVaR10 by policy over the 20 test seasons; paired differences PPO minus
comparator by season (mean over seeds) with a season-level bootstrap 95% interval (5,000 resamples); seed standard
deviation; selected checkpoints. Folds, seeds and outcomes are not changed after results are seen.
