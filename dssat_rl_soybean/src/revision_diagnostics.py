from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import load_config, make_paths
from .data import (
    WEATHER_RENAME,
    build_year_weather,
    load_observed_soybean_yield,
    load_weather,
    summarize_weather_splits,
)


def _read_numeric(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = series.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(series, errors="coerce")


def _savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _season_year(ts: pd.Series) -> pd.Series:
    dates = pd.to_datetime(ts)
    return np.where(dates.dt.month >= 9, dates.dt.year, dates.dt.year - 1)


def weather_completeness(project_dir: Path, cfg: dict) -> pd.DataFrame:
    raw_path = (project_dir / cfg["data"]["weather_csv"]).resolve()
    raw = pd.read_csv(raw_path, low_memory=False)
    raw["data_hora"] = pd.to_datetime(raw["data_hora"], errors="coerce")
    raw = raw.dropna(subset=["data_hora"]).rename(columns=WEATHER_RENAME).copy()
    raw["date"] = raw["data_hora"].dt.date
    for col in WEATHER_RENAME.values():
        if col in raw:
            raw[col] = _read_numeric(raw[col])
            raw.loc[raw[col] <= -90, col] = np.nan
    raw.loc[raw["rain"] < 0, "rain"] = np.nan
    for col in ["tmax_h", "tmin_h", "temp_h"]:
        raw.loc[(raw[col] < -20) | (raw[col] > 50), col] = np.nan
    raw.loc[(raw["srad_kj_m2"] < 0) | (raw["srad_kj_m2"] > 6000), "srad_kj_m2"] = np.nan

    daily_qc = (
        raw.groupby("date", as_index=False)
        .agg(
            rain_obs=("rain", "count"),
            temp_obs=("temp_h", "count"),
            srad_obs=("srad_kj_m2", "count"),
        )
        .sort_values("date")
    )
    daily_qc["date"] = pd.to_datetime(daily_qc["date"])
    daily_qc = daily_qc[(daily_qc["date"].dt.month >= 9) | (daily_qc["date"].dt.month <= 4)].copy()
    daily_qc["season_year"] = _season_year(daily_qc["date"])
    daily_qc["rain_valid_day"] = daily_qc["rain_obs"] >= 18
    daily_qc["temp_valid_day"] = daily_qc["temp_obs"] >= 18
    daily_qc["srad_valid_day"] = daily_qc["srad_obs"] >= 6

    daily_final = load_weather(project_dir, cfg).copy()
    daily_final = daily_final[
        (pd.to_datetime(daily_final["date"]).dt.month >= 9) | (pd.to_datetime(daily_final["date"]).dt.month <= 4)
    ].copy()
    daily_final["season_year"] = _season_year(daily_final["date"])
    seasonal = (
        daily_final.groupby("season_year", as_index=False)
        .agg(
            rain_mm=("rain", "sum"),
            temp_mean_c=("temp_mean", "mean"),
            srad_mean_mj_m2_day=("srad", "mean"),
            final_days=("date", "count"),
        )
    )
    qc = (
        daily_qc.groupby("season_year", as_index=False)
        .agg(
            raw_days=("date", "count"),
            rain_invalid_days=("rain_valid_day", lambda s: int((~s).sum())),
            temp_invalid_days=("temp_valid_day", lambda s: int((~s).sum())),
            srad_invalid_days=("srad_valid_day", lambda s: int((~s).sum())),
            rain_obs_hours_mean=("rain_obs", "mean"),
            temp_obs_hours_mean=("temp_obs", "mean"),
            srad_obs_hours_mean=("srad_obs", "mean"),
        )
    )
    out = qc.merge(seasonal, on="season_year", how="outer").sort_values("season_year")
    split_map = {}
    for split in ["train", "valid", "test"]:
        for year in cfg["data"][f"{split}_years"]:
            split_map[int(year)] = split
    out["split"] = out["season_year"].map(split_map).fillna("outside_split")
    return out


def default_soil_profile_table(cfg: dict) -> pd.DataFrame:
    """Export the synthetic fallback profile defined in dssat_adapter.py."""
    rows = [
        {"layer_bottom_cm": 15, "slll": 0.18, "sdul": 0.32, "ssat": 0.48, "srgf": 1.00, "sbdm": 1.18, "sloc": 2.2, "slcl": 60, "slsi": 25},
        {"layer_bottom_cm": 30, "slll": 0.20, "sdul": 0.34, "ssat": 0.47, "srgf": 0.80, "sbdm": 1.22, "sloc": 1.8, "slcl": 62, "slsi": 24},
        {"layer_bottom_cm": 60, "slll": 0.22, "sdul": 0.35, "ssat": 0.46, "srgf": 0.55, "sbdm": 1.27, "sloc": 1.3, "slcl": 65, "slsi": 22},
        {"layer_bottom_cm": 100, "slll": 0.23, "sdul": 0.36, "ssat": 0.45, "srgf": 0.30, "sbdm": 1.32, "sloc": 0.9, "slcl": 66, "slsi": 21},
        {"layer_bottom_cm": 150, "slll": 0.24, "sdul": 0.37, "ssat": 0.44, "srgf": 0.15, "sbdm": 1.36, "sloc": 0.6, "slcl": 68, "slsi": 20},
    ]
    table = pd.DataFrame(rows)
    table.insert(0, "layer_top_cm", [0] + table["layer_bottom_cm"].iloc[:-1].tolist())
    table["thickness_cm"] = table["layer_bottom_cm"] - table["layer_top_cm"]
    table["available_water_mm"] = (table["sdul"] - table["slll"]) * table["thickness_cm"] * 10.0
    table["soil_profile_name"] = "CASTRO0001"
    table["soil_source"] = "synthetic fallback profile defined in dssat_adapter.py"
    table["latitude"] = cfg["dssat"]["latitude"]
    table["longitude"] = cfg["dssat"]["longitude"]
    return table[
        [
            "soil_profile_name",
            "soil_source",
            "layer_top_cm",
            "layer_bottom_cm",
            "thickness_cm",
            "slll",
            "sdul",
            "ssat",
            "sbdm",
            "sloc",
            "slcl",
            "slsi",
            "srgf",
            "available_water_mm",
            "latitude",
            "longitude",
        ]
    ]


def environment_table(cfg: dict, weather_split_summary: pd.DataFrame, soil_table: pd.DataFrame) -> pd.DataFrame:
    rainfall = float(weather_split_summary["rain_mean_mm_year"].mean())
    years = cfg["data"]["train_years"] + cfg["data"]["valid_years"] + cfg["data"]["test_years"]
    available_water = float(soil_table["available_water_mm"].sum())
    return pd.DataFrame(
        [
            {
                "environment": "Castro_A819",
                "period": f"{min(years)}-{max(years)}",
                "soil_texture": "synthetic clayey profile",
                "available_water_capacity_0_150cm_mm": available_water,
                "cultivar": cfg["agronomy"]["soybean_cultivar"],
                "maturity_group": "5, generic DSSAT cultivar",
                "mean_seasonal_rainfall_mm": rainfall,
                "irrigation_availability": "optional supplemental irrigation in simulation",
                "field_validation_observations": 0,
                "phenology_rmse_days": np.nan,
                "raw_yield_rmse_kg_ha": np.nan,
                "temporal_split": "train 2006-2017; validation 2018-2021; diagnostic 2022-2025",
                "validation_status": "not agronomically validated",
            }
        ]
    )


def season_sidra_pairing(project_dir: Path, cfg: dict) -> pd.DataFrame:
    obs = load_observed_soybean_yield(project_dir, cfg).rename(columns={"ano": "sidra_year"})
    obs_map = dict(zip(obs["sidra_year"].astype(int), obs["observed_yield_kg_ha"].astype(float)))
    years = sorted(set(cfg["data"]["train_years"] + cfg["data"]["valid_years"] + cfg["data"]["test_years"]))
    rows = []
    for y in years:
        rows.append(
            {
                "season_year": y,
                "sowing_year": y,
                "harvest_year": y + 1,
                "sidra_same_sowing_year": y,
                "yield_same_sowing_year_kg_ha": obs_map.get(y),
                "sidra_harvest_year": y + 1,
                "yield_harvest_year_kg_ha": obs_map.get(y + 1),
                "has_boundary_issue": pd.isna(obs_map.get(y)) or pd.isna(obs_map.get(y + 1)),
            }
        )
    return pd.DataFrame(rows)


def correction_alpha_summary(project_dir: Path) -> pd.DataFrame:
    candidates = sorted((project_dir / "outputs").glob("calibration*/tables/yield_correction_alpha_search.csv"))
    if not candidates:
        return pd.DataFrame()
    preferred = [p for p in candidates if "calibration_v4" in str(p)]
    source = preferred[-1] if preferred else candidates[-1]
    alpha = pd.read_csv(source)
    keep = alpha.sort_values("valid_bias_corrected_mae").head(5).copy()
    keep.insert(0, "source_file", str(source.relative_to(project_dir)))
    return keep[
        [
            "source_file",
            "alpha",
            "train_mae",
            "valid_mae",
            "valid_bias_corrected_mae",
            "valid_bias_corrected_rmse",
            "validation_intercept_adjustment",
        ]
    ]


def _load_policy_outputs(paths) -> pd.DataFrame:
    policy_path = paths.tables_dir / "policy_evaluation_all.csv"
    if not policy_path.exists():
        raise FileNotFoundError(policy_path)
    df = pd.read_csv(policy_path)
    meta_path = paths.output_dir / "run_metadata.json"
    seed = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            seed = meta.get("config", {}).get("seed")
        except Exception:
            seed = None
    df["seed"] = int(seed) if seed is not None else 0
    df["environment"] = "Castro_A819"
    return df


def seed_and_action_diagnostics(paths, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    policy = _load_policy_outputs(paths)
    seed_summary = (
        policy.groupby(["seed", "split"], as_index=False)
        .agg(
            n_scenarios=("year", "count"),
            reward_mean=("reward", "mean"),
            yield_mean_kg_ha=("yield_kg_ha", "mean"),
            irrigation_mean_mm=("irrigation_mm", "mean"),
            sowing_offset_mean_days=("planting_offset_days", "mean"),
            failures=("failed", "sum"),
        )
    )
    ag = cfg["agronomy"]
    action = policy.copy()
    action["trigger_upper_bound"] = np.isclose(action["trigger_dryness"], 90.0)
    action["event_depth_lower_bound"] = np.isclose(action["amount_mm"], 0.0) | np.isclose(action["amount_mm"], 2.0)
    action["event_depth_upper_bound"] = np.isclose(action["amount_mm"], ag["max_single_irrigation_mm"])
    action["rainfed_realized"] = (action["irrigation_mm"] == 0) & (action["n_irrigation_events"] == 0)
    saturation = (
        action.groupby(["split"], as_index=False)
        .agg(
            n=("year", "count"),
            trigger_upper_bound_freq=("trigger_upper_bound", "mean"),
            event_depth_lower_bound_freq=("event_depth_lower_bound", "mean"),
            event_depth_upper_bound_freq=("event_depth_upper_bound", "mean"),
            rainfed_realized_freq=("rainfed_realized", "mean"),
        )
    )
    return seed_summary, action, saturation


def paired_effects(paths) -> pd.DataFrame:
    policy = _load_policy_outputs(paths)
    base_path = paths.tables_dir / "policy_vs_baselines.csv"
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    comp = pd.read_csv(base_path)
    baseline = comp[comp["policy"] != "ppo_policy"].copy()
    ppo = policy[["split", "year", "seed", "yield_kg_ha", "irrigation_mm", "reward"]].rename(
        columns={
            "yield_kg_ha": "ppo_yield_kg_ha",
            "irrigation_mm": "ppo_irrigation_mm",
            "reward": "ppo_reward",
        }
    )
    merged = baseline.merge(ppo, on=["split", "year"], how="inner")
    merged["delta_yield_kg_ha"] = merged["ppo_yield_kg_ha"] - merged["yield_kg_ha"]
    merged["delta_irrigation_mm"] = merged["ppo_irrigation_mm"] - merged["irrigation_mm"]
    merged["delta_reward"] = merged["ppo_reward"] - merged["reward"]
    return merged


def paired_summary(effects: pd.DataFrame) -> pd.DataFrame:
    return (
        effects.groupby(["split", "policy"], as_index=False)
        .agg(
            n_independent_seasons=("year", "nunique"),
            n_seeds=("seed", "nunique"),
            delta_yield_mean_kg_ha=("delta_yield_kg_ha", "mean"),
            delta_yield_median_kg_ha=("delta_yield_kg_ha", "median"),
            delta_yield_sd_kg_ha=("delta_yield_kg_ha", "std"),
            delta_irrigation_mean_mm=("delta_irrigation_mm", "mean"),
            delta_reward_mean=("delta_reward", "mean"),
        )
        .sort_values(["split", "policy"])
    )


def risk_metrics(paths) -> pd.DataFrame:
    comp = pd.read_csv(paths.tables_dir / "policy_vs_baselines.csv")
    cols = ["split", "year", "policy", "yield_kg_ha", "irrigation_mm", "reward"]
    all_rows = comp[cols].copy()
    if "ppo_policy" not in set(all_rows["policy"]):
        policy = _load_policy_outputs(paths)[["split", "year", "yield_kg_ha", "irrigation_mm", "reward"]].assign(policy="ppo_policy")
        all_rows = pd.concat([all_rows, policy[cols]], ignore_index=True)
    rows = []
    for (split, policy_name), g in all_rows.groupby(["split", "policy"]):
        yields = g["yield_kg_ha"].dropna().sort_values()
        if yields.empty:
            continue
        q10 = float(yields.quantile(0.10))
        cvar10 = float(yields[yields <= q10].mean()) if (yields <= q10).any() else q10
        rows.append(
            {
                "split": split,
                "policy": policy_name,
                "n": int(len(yields)),
                "yield_p10_kg_ha": q10,
                "yield_cvar10_kg_ha": cvar10,
                "yield_mean_kg_ha": float(yields.mean()),
                "irrigation_mean_mm": float(g["irrigation_mm"].mean()),
            }
        )
    return pd.DataFrame(rows)


def context_action_map(project_dir: Path, cfg: dict, paths, action: pd.DataFrame) -> pd.DataFrame:
    daily = load_weather(project_dir, cfg)
    years = build_year_weather(daily, sorted(action["year"].unique().astype(int).tolist()))
    context = []
    for yw in years:
        context.append(
            {
                "year": yw.year,
                "realized_season_rain_feature": float(yw.features[0]),
                "realized_temp_mean_feature": float(yw.features[1]),
                "realized_srad_mean_feature": float(yw.features[4]),
                "realized_season_rain_mm": float(yw.features[0] * 1800.0),
                "realized_temp_mean_c": float(yw.features[1] * 35.0),
                "realized_srad_mean_mj_m2_day": float(yw.features[4] * 30.0),
            }
        )
    out = action.merge(pd.DataFrame(context), on="year", how="left")
    plt.figure(figsize=(6.8, 4.6))
    sns.scatterplot(
        data=out,
        x="realized_season_rain_mm",
        y="planting_offset_days",
        hue="irrigation_mm",
        style="split",
        s=90,
    )
    plt.xlabel("Realized seasonal rainfall used by diagnostic context (mm)")
    plt.ylabel("Sowing offset after September 15 (days)")
    plt.title("Diagnostic context-action map")
    _savefig(paths.figures_dir / "diagnostic_context_action_map_en.png")
    return out


def plot_diagnostics(paths, seed_summary: pd.DataFrame, effects: pd.DataFrame, risk: pd.DataFrame):
    plt.figure(figsize=(6.4, 4.2))
    sns.barplot(data=seed_summary, x="seed", y="reward_mean", hue="split")
    plt.xlabel("Training seed")
    plt.ylabel("Mean reward")
    plt.title("Per-seed performance distribution")
    _savefig(paths.figures_dir / "seed_performance_distribution_en.png")

    plot = effects[effects["split"].isin(["valid", "test"])].copy()
    if not plot.empty:
        plt.figure(figsize=(8.0, 4.8))
        sns.stripplot(data=plot, x="delta_yield_kg_ha", y="policy", hue="split", dodge=True, size=7)
        plt.axvline(0, color="black", linewidth=1)
        plt.xlabel("Paired yield difference: PPO - comparator (kg ha$^{-1}$)")
        plt.ylabel("")
        plt.title("Paired diagnostic effects against constructed schedules")
        _savefig(paths.figures_dir / "paired_effects_vs_constructed_schedules_en.png")

    if not risk.empty:
        plt.figure(figsize=(8.2, 4.5))
        sns.scatterplot(data=risk, x="irrigation_mean_mm", y="yield_mean_kg_ha", hue="policy", style="split", s=90)
        plt.xlabel("Mean seasonal irrigation (mm)")
        plt.ylabel("Mean yield (kg ha$^{-1}$)")
        plt.title("Diagnostic yield-water points")
        _savefig(paths.figures_dir / "diagnostic_yield_water_points_en.png")


def run_diagnostics(config: str, run_name: str):
    cfg = load_config(config)
    paths = make_paths(cfg, run_name)
    tables = paths.tables_dir
    figures = paths.figures_dir
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    weather = weather_completeness(paths.project_dir, cfg)
    weather.to_csv(tables / "weather_completeness_by_season.csv", index=False, encoding="utf-8-sig")
    daily = load_weather(paths.project_dir, cfg)
    weather_split_summary = summarize_weather_splits(daily, cfg)
    weather_split_summary.to_csv(tables / "weather_splits.csv", index=False, encoding="utf-8-sig")
    weather_split_summary.to_markdown(tables / "weather_splits.md", index=False)
    soil = default_soil_profile_table(cfg)
    soil.to_csv(tables / "synthetic_soil_profile_s2.csv", index=False, encoding="utf-8-sig")
    environment_table(cfg, weather_split_summary, soil).to_csv(
        tables / "environment_table_a_current.csv", index=False, encoding="utf-8-sig"
    )
    season_sidra_pairing(paths.project_dir, cfg).to_csv(tables / "season_sidra_pairing_audit.csv", index=False, encoding="utf-8-sig")
    alpha = correction_alpha_summary(paths.project_dir)
    if not alpha.empty:
        alpha.to_csv(tables / "yield_correction_alpha_ablation_s4.csv", index=False, encoding="utf-8-sig")

    seed_summary, action, saturation = seed_and_action_diagnostics(paths, cfg)
    seed_summary.to_csv(tables / "seed_performance_summary.csv", index=False, encoding="utf-8-sig")
    action.to_csv(tables / "policy_action_by_scenario_seed.csv", index=False, encoding="utf-8-sig")
    saturation.to_csv(tables / "action_bound_saturation.csv", index=False, encoding="utf-8-sig")

    effects = paired_effects(paths)
    effects.to_csv(tables / "paired_effects_vs_constructed_schedules.csv", index=False, encoding="utf-8-sig")
    paired_summary(effects).to_csv(tables / "paired_effects_vs_constructed_schedules_summary.csv", index=False, encoding="utf-8-sig")

    risk = risk_metrics(paths)
    risk.to_csv(tables / "diagnostic_risk_metrics.csv", index=False, encoding="utf-8-sig")
    context_action_map(paths.project_dir, cfg, paths, action).to_csv(
        tables / "diagnostic_context_action_map.csv", index=False, encoding="utf-8-sig"
    )
    plot_diagnostics(paths, seed_summary, effects, risk)


def main():
    parser = argparse.ArgumentParser(description="Build revision diagnostics requested by manuscript review comments.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--run-name", default="ppo_soja_castro_dssat")
    args = parser.parse_args()
    run_diagnostics(args.config, args.run_name)


if __name__ == "__main__":
    main()
