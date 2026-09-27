from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import load_config, make_paths
from .data import WEATHER_RENAME


def _resolve(project_dir: Path, path_value: str) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else (project_dir / p).resolve()


def _read_numeric(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = series.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(series, errors="coerce")


def _season_year(dates: pd.Series) -> pd.Series:
    dates = pd.to_datetime(dates)
    return np.where(dates.dt.month >= 9, dates.dt.year, dates.dt.year - 1)


def _raw_daily(project_dir: Path, cfg: dict) -> pd.DataFrame:
    path = _resolve(project_dir, cfg["data"]["weather_csv"])
    raw = pd.read_csv(path, low_memory=False)
    raw["data_hora"] = pd.to_datetime(raw["data_hora"], errors="coerce")
    raw = raw.dropna(subset=["data_hora"]).rename(columns=WEATHER_RENAME).copy()
    raw["date"] = raw["data_hora"].dt.floor("D")

    for col in WEATHER_RENAME.values():
        if col in raw:
            raw[col] = _read_numeric(raw[col])
            raw.loc[raw[col] <= -90, col] = np.nan
    raw.loc[raw["rain"] < 0, "rain"] = np.nan
    for col in ["tmax_h", "tmin_h", "temp_h"]:
        raw.loc[(raw[col] < -20) | (raw[col] > 50), col] = np.nan
    raw.loc[(raw["srad_kj_m2"] < 0) | (raw["srad_kj_m2"] > 6000), "srad_kj_m2"] = np.nan

    daily = (
        raw.groupby("date", as_index=False)
        .agg(
            rain=("rain", lambda s: s.sum(min_count=1)),
            rain_obs=("rain", "count"),
            tmax=("tmax_h", "max"),
            tmin=("tmin_h", "min"),
            temp_mean=("temp_h", "mean"),
            temp_obs=("temp_h", "count"),
            srad_kj_m2=("srad_kj_m2", lambda s: s.sum(min_count=1)),
            srad_obs=("srad_kj_m2", "count"),
        )
        .sort_values("date")
    )
    full = pd.DataFrame({"date": pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")})
    daily = full.merge(daily, on="date", how="left")
    daily["rain_raw"] = daily["rain"]
    daily.loc[daily["rain_obs"].fillna(0) < 18, "rain"] = np.nan
    daily["rain_invalid_day"] = daily["rain"].isna()
    daily["season_year"] = _season_year(daily["date"])
    return daily


def _fill_daily_climatology(daily: pd.DataFrame) -> pd.Series:
    rain = daily["rain"].copy()
    doy = daily["date"].dt.dayofyear
    clim = daily.assign(doy=doy).groupby("doy")["rain"].mean()
    month = daily["date"].dt.month
    monthly = daily.assign(month=month).groupby("month")["rain"].mean()
    missing = rain.isna()
    rain.loc[missing] = doy.map(clim).loc[missing].to_numpy()
    still = rain.isna()
    rain.loc[still] = month.map(monthly).loc[still].to_numpy()
    return rain.fillna(0).clip(lower=0)


def _fill_monthly_climatology(daily: pd.DataFrame) -> pd.Series:
    rain = daily["rain"].copy()
    month = daily["date"].dt.month
    monthly = daily.assign(month=month).groupby("month")["rain"].mean()
    missing = rain.isna()
    rain.loc[missing] = month.map(monthly).loc[missing].to_numpy()
    return rain.fillna(0).clip(lower=0)


def _fill_short_interpolation(daily: pd.DataFrame, limit: int = 3) -> pd.Series:
    rain = daily["rain"].copy()
    interpolated = rain.interpolate(limit=limit, limit_direction="both")
    rain.loc[rain.isna()] = interpolated.loc[rain.isna()]
    tmp = daily.copy()
    tmp["rain"] = rain
    return _fill_daily_climatology(tmp)


def _filled_weather(daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    strategies = {
        "daily_climatology": _fill_daily_climatology(daily),
        "monthly_climatology": _fill_monthly_climatology(daily),
        "short_gap_interpolation": _fill_short_interpolation(daily),
        "zero_missing": daily["rain"].fillna(0).clip(lower=0),
    }
    out = {}
    for name, rain in strategies.items():
        df = daily[["date", "season_year", "rain_invalid_day"]].copy()
        df["strategy"] = name
        df["rain"] = rain.to_numpy()
        out[name] = df
    return out


def _seasonal_summary(filled: dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    split_map = {}
    for split in ["train", "valid", "test"]:
        for year in cfg["data"][f"{split}_years"]:
            split_map[int(year)] = split
    rows = []
    for strategy, df in filled.items():
        season = df[(df["date"].dt.month >= 9) | (df["date"].dt.month <= 4)].copy()
        for year, group in season.groupby("season_year"):
            if int(year) not in split_map:
                continue
            rows.append(
                {
                    "strategy": strategy,
                    "split": split_map[int(year)],
                    "season_year": int(year),
                    "rain_mm": float(group["rain"].sum()),
                    "invalid_rain_days": int(group["rain_invalid_day"].sum()),
                    "n_days": int(len(group)),
                }
            )
    out = pd.DataFrame(rows)
    base = out[out["strategy"] == "daily_climatology"][["season_year", "rain_mm"]].rename(
        columns={"rain_mm": "rain_daily_climatology_mm"}
    )
    out = out.merge(base, on="season_year", how="left")
    out["delta_vs_daily_climatology_mm"] = out["rain_mm"] - out["rain_daily_climatology_mm"]
    return out.sort_values(["season_year", "strategy"])


def _window_rain(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    mask = (df["date"] >= start) & (df["date"] <= end)
    return float(df.loc[mask, "rain"].sum())


def _action_exposure(filled: dict[str, pd.DataFrame], paths) -> pd.DataFrame:
    tables = paths.tables_dir
    rows = []
    action_path = tables / "policy_action_by_scenario_seed.csv"
    if action_path.exists():
        action = pd.read_csv(action_path)
        for _, row in action.iterrows():
            start = pd.to_datetime(row["planting_date"])
            end = pd.to_datetime(row["harvest_window_end"])
            for strategy, df in filled.items():
                rows.append(
                    {
                        "source": "ppo_policy",
                        "split": row["split"],
                        "year": int(row["year"]),
                        "policy": "PPO",
                        "strategy": strategy,
                        "planting_date": start.date().isoformat(),
                        "window_end": end.date().isoformat(),
                        "rain_window_mm": _window_rain(df, start, end),
                        "recorded_irrigation_mm": float(row.get("irrigation_mm", np.nan)),
                    }
                )
    for fname in ["baseline_evaluation_valid.csv", "baseline_evaluation_test.csv"]:
        path = tables / fname
        if not path.exists():
            continue
        base = pd.read_csv(path)
        for _, row in base.iterrows():
            start = pd.to_datetime(row["planting_date"])
            end = start + timedelta(days=149)
            for strategy, df in filled.items():
                rows.append(
                    {
                        "source": "constructed_baseline",
                        "split": row["split"],
                        "year": int(row["year"]),
                        "policy": row["policy"],
                        "strategy": strategy,
                        "planting_date": start.date().isoformat(),
                        "window_end": end.date().isoformat(),
                        "rain_window_mm": _window_rain(df, start, end),
                        "recorded_irrigation_mm": float(row.get("irrigation_mm", np.nan)),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    base = out[out["strategy"] == "daily_climatology"][
        ["source", "split", "year", "policy", "planting_date", "rain_window_mm"]
    ].rename(columns={"rain_window_mm": "rain_window_daily_climatology_mm"})
    out = out.merge(base, on=["source", "split", "year", "policy", "planting_date"], how="left")
    out["delta_vs_daily_climatology_mm"] = out["rain_window_mm"] - out["rain_window_daily_climatology_mm"]
    return out.sort_values(["source", "year", "policy", "strategy"])


def _summary_tables(seasonal: pd.DataFrame, exposure: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sens = (
        seasonal[seasonal["strategy"] != "daily_climatology"]
        .groupby(["split", "strategy"], as_index=False)
        .agg(
            seasons=("season_year", "nunique"),
            mean_delta_rain_mm=("delta_vs_daily_climatology_mm", "mean"),
            mean_abs_delta_rain_mm=("delta_vs_daily_climatology_mm", lambda s: float(np.mean(np.abs(s)))),
            max_abs_delta_rain_mm=("delta_vs_daily_climatology_mm", lambda s: float(np.max(np.abs(s)))),
            mean_invalid_rain_days=("invalid_rain_days", "mean"),
        )
    )
    if exposure.empty:
        action = pd.DataFrame()
    else:
        action = (
            exposure[exposure["strategy"] != "daily_climatology"]
            .groupby(["source", "split", "strategy"], as_index=False)
            .agg(
                scenarios=("year", "nunique"),
                policies=("policy", "nunique"),
                mean_abs_delta_window_rain_mm=("delta_vs_daily_climatology_mm", lambda s: float(np.mean(np.abs(s)))),
                max_abs_delta_window_rain_mm=("delta_vs_daily_climatology_mm", lambda s: float(np.max(np.abs(s)))),
                mean_recorded_irrigation_mm=("recorded_irrigation_mm", "mean"),
            )
        )
    return sens, action


def _plot_seasonal(seasonal: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot = seasonal[seasonal["split"].isin(["valid", "test"])].copy()
    plt.figure(figsize=(8.0, 4.6))
    sns.lineplot(data=plot, x="season_year", y="rain_mm", hue="strategy", marker="o")
    plt.xlabel("Season")
    plt.ylabel("September-April rainfall (mm)")
    plt.title("Rainfall sensitivity to precipitation imputation")
    plt.legend(title="Imputation strategy", fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "weather_imputation_seasonal_rain_en.png", dpi=300)
    plt.close()

    delta = plot[plot["strategy"] != "daily_climatology"].copy()
    plt.figure(figsize=(8.0, 4.6))
    sns.barplot(data=delta, x="season_year", y="delta_vs_daily_climatology_mm", hue="strategy")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Season")
    plt.ylabel("Difference from daily climatology (mm)")
    plt.title("Seasonal rainfall difference by imputation strategy")
    plt.legend(title="Imputation strategy", fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "weather_imputation_delta_en.png", dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--run-name", default="ppo_soja_castro_dssat")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = make_paths(cfg, args.run_name)
    daily = _raw_daily(paths.project_dir, cfg)
    filled = _filled_weather(daily)
    seasonal = _seasonal_summary(filled, cfg)
    exposure = _action_exposure(filled, paths)
    split_summary, action_summary = _summary_tables(seasonal, exposure)

    paths.tables_dir.mkdir(parents=True, exist_ok=True)
    seasonal.to_csv(paths.tables_dir / "weather_imputation_by_season.csv", index=False, encoding="utf-8-sig")
    exposure.to_csv(paths.tables_dir / "weather_imputation_action_exposure.csv", index=False, encoding="utf-8-sig")
    split_summary.to_csv(paths.tables_dir / "weather_imputation_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    action_summary.to_csv(
        paths.tables_dir / "weather_imputation_action_sensitivity_summary.csv", index=False, encoding="utf-8-sig"
    )
    _plot_seasonal(seasonal, paths.figures_dir)

    print("Wrote weather imputation sensitivity tables and figures to", paths.output_dir)
    print(split_summary.to_string(index=False))


if __name__ == "__main__":
    main()
