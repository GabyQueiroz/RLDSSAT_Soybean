from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import load_config, make_paths
from .data import load_weather_inmet, load_weather_power

VARIABLES = {
    "rain": ("rain_obs", 18),
    "tmax": ("temp_obs", 18),
    "tmin": ("temp_obs", 18),
    "temp_mean": ("temp_obs", 18),
    "srad": ("srad_obs", 6),
}


def _metrics(a: pd.Series, b: pd.Series) -> dict:
    d = a - b
    return {
        "n": int(len(d)),
        "mean_station": float(a.mean()),
        "mean_power": float(b.mean()),
        "bias_station_minus_power": float(d.mean()),
        "mae": float(d.abs().mean()),
        "rmse": float(np.sqrt((d**2).mean())),
        "pearson_r": float(np.corrcoef(a, b)[0, 1]) if len(d) > 2 else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Agreement between INMET A819 and NASA POWER on days the station is valid.")
    parser.add_argument("--config", default="configs/experiment_castro_power.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = make_paths(cfg, "power_station_agreement")
    station = load_weather_inmet(paths.project_dir, cfg)
    power = load_weather_power(paths.project_dir, cfg)
    merged = station.merge(power[["date", *VARIABLES]], on="date", suffixes=("_station", "_power"))

    daily_rows, season_rows, by_year = [], [], []
    for var, (obs_col, min_hours) in VARIABLES.items():
        ok = merged[merged[obs_col] >= min_hours]
        daily_rows.append({"variable": var, **_metrics(ok[f"{var}_station"], ok[f"{var}_power"])})
        for year in sorted(merged["year"].unique()):
            y = ok[ok["year"] == year]
            by_year.append({"variable": var, "year": int(year), "valid_station_days": int(len(y)),
                            "bias_station_minus_power": float((y[f"{var}_station"] - y[f"{var}_power"]).mean()) if len(y) else np.nan})

    # Season totals (Sep 1 - Apr 30) of rainfall for seasons in which at least 90% of the station days are valid.
    merged["season"] = np.where(merged["date"].dt.month >= 9, merged["date"].dt.year, merged["date"].dt.year - 1)
    window = merged[(merged["date"].dt.month >= 9) | (merged["date"].dt.month <= 4)]
    for season, g in window.groupby("season"):
        valid_share = float((g["rain_obs"] >= 18).mean())
        season_rows.append({"season": int(season), "days": len(g), "valid_rain_day_share": valid_share,
                            "rain_station_mm": float(g["rain_station"].sum()), "rain_power_mm": float(g["rain_power"].sum())})
    seasons = pd.DataFrame(season_rows)
    complete = seasons[(seasons["valid_rain_day_share"] >= 0.9) & (seasons["days"] >= 240)]
    season_summary = _metrics(complete["rain_station_mm"], complete["rain_power_mm"])

    pd.DataFrame(daily_rows).to_csv(paths.tables_dir / "power_vs_station_daily.csv", index=False)
    pd.DataFrame(by_year).to_csv(paths.tables_dir / "power_vs_station_bias_by_year.csv", index=False)
    seasons.to_csv(paths.tables_dir / "power_vs_station_season_rain.csv", index=False)
    pd.DataFrame([{"variable": "season_rain_sep_apr", **season_summary}]).to_csv(paths.tables_dir / "power_vs_station_season_summary.csv", index=False)
    print(pd.DataFrame(daily_rows).round(3).to_string(index=False))
    print(season_summary)


if __name__ == "__main__":
    main()
