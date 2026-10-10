"""Principal inferential model of the rolling-origin paired effects.

For each comparator and metric, the paired difference of season y and seed s is modelled as

    delta_{y,s} = mu + v_y + w_s + e_{y,s},

with crossed random effects of season (v) and training seed (w). The design is complete and balanced
(20 test seasons x 5 seeds), so the REML estimates equal the ANOVA estimates of the variance components
whenever these are non-negative, and

    Var(mu_hat) = (MS_season + MS_seed - MS_residual) / (n_seasons * n_seeds).

The 95% interval uses Satterthwaite degrees of freedom. The season bootstrap of rolling_origin.analyse is
kept as a sensitivity check.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from .config import load_config, make_paths

METRICS = {"d_yield": "yield_kg_ha", "d_irrigation": "irrigation_mm", "d_reward": "reward"}


def crossed_model(table: pd.DataFrame, value: str) -> dict:
    m = table.pivot_table(index="year", columns="seed", values=value, aggfunc="mean")
    if m.isna().any().any():
        raise ValueError("unbalanced season x seed table")
    n_y, n_s = m.shape
    if n_s == 1:
        # One seed: the model reduces to season + residual, and the interval is the t interval over seasons.
        d = m.iloc[:, 0].to_numpy()
        se = float(d.std(ddof=1) / np.sqrt(n_y))
        t = stats.t.ppf(0.975, n_y - 1)
        return {"n_seasons": n_y, "n_seeds": 1, "mu": float(d.mean()), "se": se, "df_satterthwaite": float(n_y - 1),
                "ci95_low": float(d.mean() - t * se), "ci95_high": float(d.mean() + t * se),
                "sd_season": float(d.std(ddof=1)), "sd_seed": np.nan, "sd_residual": np.nan}
    grand = m.values.mean()
    row = m.mean(axis=1).to_numpy()
    col = m.mean(axis=0).to_numpy()
    resid = m.values - row[:, None] - col[None, :] + grand
    ms_y = n_s * np.sum((row - grand) ** 2) / (n_y - 1)
    ms_s = n_y * np.sum((col - grand) ** 2) / (n_s - 1)
    ms_e = np.sum(resid ** 2) / ((n_y - 1) * (n_s - 1))
    var_v = max(0.0, (ms_y - ms_e) / n_s)
    var_w = max(0.0, (ms_s - ms_e) / n_y)
    # Variance of the grand mean from the fitted components (equal to the mean-square form when both are positive).
    var_mu = var_v / n_y + var_w / n_s + ms_e / (n_y * n_s)
    se = float(np.sqrt(var_mu))
    # Satterthwaite: var_mu is a linear combination of the three mean squares.
    k = 1 / (n_y * n_s)
    a, b = (k if var_v > 0 else 0.0), (k if var_w > 0 else 0.0)
    terms = [(a, ms_y, n_y - 1), (b, ms_s, n_s - 1), (k - a - b, ms_e, (n_y - 1) * (n_s - 1))]
    num = sum(c * ms for c, ms, _ in terms) ** 2
    den = sum((c * ms) ** 2 / df for c, ms, df in terms if c != 0)
    dof = float(num / den) if den > 0 else float(n_y - 1)
    t = stats.t.ppf(0.975, dof)
    return {"n_seasons": n_y, "n_seeds": n_s, "mu": float(grand), "se": se, "df_satterthwaite": dof,
            "ci95_low": float(grand - t * se), "ci95_high": float(grand + t * se),
            "sd_season": float(np.sqrt(var_v)), "sd_seed": float(np.sqrt(var_w)), "sd_residual": float(np.sqrt(ms_e))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_castro_power.yaml")
    parser.add_argument("--analysis", default="rolling_power_analysis")
    parser.add_argument("--output", default="article_rolling_power")
    args = parser.parse_args()
    cfg = load_config(args.config)
    outputs = make_paths(cfg, args.output).project_dir / "outputs"
    paired = pd.read_csv(outputs / args.analysis / "tables" / "paired_effects_seed_season.csv")
    rows = []
    for comp, g in paired.groupby("comparator"):
        for metric in METRICS:
            rows.append({"comparator": comp, "metric": metric, **crossed_model(g, metric)})
    res = pd.DataFrame(rows)
    suffix = "" if args.analysis == "rolling_power_analysis" else "_" + args.analysis.replace("_analysis", "")
    target = outputs / args.output / "tables" / f"paired_effects_mixed_model{suffix}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(target, index=False)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
