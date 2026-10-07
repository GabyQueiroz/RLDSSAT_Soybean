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

## Amendment 1 (2026-10-07, before any test-season result of this design was produced)

The measured simulation rate on the workstation (about 1.5-4 training steps per second, because with 16-34 training
seasons few DSSAT treatments repeat and the cache rarely applies) implied about 50 h for 30 runs. The test block of each
fold was therefore extended from 2 to 4 seasons, keeping the same 20 test seasons and the same seeds:

| fold | PPO training | checkpoint selection | test |
|---|---|---|---|
| f2005 | 1984-1999 | 2000-2004 | 2005-2008 |
| f2009 | 1984-2003 | 2004-2008 | 2009-2012 |
| f2013 | 1984-2007 | 2008-2012 | 2013-2016 |
| f2017 | 1984-2011 | 2012-2016 | 2017-2020 |
| f2021 | 1984-2015 | 2016-2020 | 2021-2024 |

5 folds x 3 seeds = 15 runs. The f2005 / seed 42 run, whose training and validation seasons are identical in both
versions, is kept and its selected checkpoint is evaluated on 2005-2008. Training, validation and the PPO
configuration are unchanged.

## Reported outcomes

Mean objective, yield, irrigation, P10 and CVaR10 by policy over the 20 test seasons; paired differences PPO minus
comparator by season (mean over seeds) with a season-level bootstrap 95% interval (5,000 resamples); seed standard
deviation; selected checkpoints. Folds, seeds and outcomes are not changed after results are seen.
