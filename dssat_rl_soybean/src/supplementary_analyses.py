"""Diagnostics required by the methodological review that do not involve PPO training.

thresholds  sensitivity of INMET seasonal rainfall and context to the daily completeness thresholds
proxy       proxy evaporative demand of the irrigation rule versus Hargreaves ET0 and versus the DSSAT soil-water deficit
spinup      effect of the August soil-water spin-up on rainfed yield and on the ranking of sowing dates
interval    sensitivity of the optimized fixed rule to the inspection interval of the irrigation rule
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, make_paths
from .data import build_year_weather_with_context, context_scaler, load_weather, load_weather_inmet, planting_date_for_year
from .dssat_adapter import PyDSSATRunner, build_irrigation_schedule
from .final_comparators import CONSTRUCTED, candidate_grid, select_fixed_rule
from .rolling_origin import folds

SEASONS = list(range(1984, 2025))


def _soilwat(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    head = next(i for i, line in enumerate(lines) if line.startswith("@"))
    cols = lines[head].lstrip("@").split()
    rows = [line.split() for line in lines[head + 1:] if line.strip() and not line.startswith(("*", "!", "@"))]
    df = pd.DataFrame([r[: len(cols)] for r in rows], columns=cols).apply(pd.to_numeric, errors="coerce")
    df["date"] = pd.to_datetime(df["YEAR"].astype(int).astype(str), format="%Y") + pd.to_timedelta(df["DOY"] - 1, unit="D")
    return df


def proxy_balance(daily: pd.DataFrame, start: date, days: int) -> pd.DataFrame:
    season = daily[(daily["date"].dt.date >= start) & (daily["date"].dt.date < start + timedelta(days=days))]
    b, rows = 0.0, []
    for row in season.itertuples(index=False):
        rows.append({"date": row.date, "proxy_deficit_mm": b})
        e = max(0.0, 0.16 * float(row.srad) + 0.08 * max(float(row.temp_mean) - 10.0, 0.0))
        b = max(0.0, b + e - float(row.rain))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- thresholds
def thresholds(cfg_path: str, out: Path) -> None:
    rows = []
    for hours, srad_hours in [(18, 6), (12, 4), (21, 8), (24, 6)]:
        cfg = load_config(cfg_path)
        cfg["data"]["qc_min_hours_rain_temp"], cfg["data"]["qc_min_hours_srad"] = hours, srad_hours
        daily = load_weather_inmet(Path(cfg["_project_dir"]), cfg)
        seasons = build_year_weather_with_context(daily, list(range(2006, 2025)), cfg)
        for s in seasons:
            w = s.daily[(s.daily["date"] >= pd.Timestamp(s.year, 9, 1)) & (s.daily["date"] <= pd.Timestamp(s.year + 1, 4, 30))]
            rows.append({"min_hours_rain_temp": hours, "min_hours_srad": srad_hours, "season": s.year,
                         "rain_sep_apr_mm": w["rain"].sum(), "tmean_sep_apr_c": w["temp_mean"].mean(), "srad_sep_apr": w["srad"].mean(),
                         **{f"ctx_{i}": v for i, v in enumerate(s.features)}})
    df = pd.DataFrame(rows)
    df.to_csv(out / "qc_threshold_by_season.csv", index=False)
    base = df[(df["min_hours_rain_temp"] == 18) & (df["min_hours_srad"] == 6)].set_index("season")
    summ = []
    for (h, sh), g in df.groupby(["min_hours_rain_temp", "min_hours_srad"]):
        g = g.set_index("season")
        d = g[["rain_sep_apr_mm", "tmean_sep_apr_c", "srad_sep_apr"]] - base[["rain_sep_apr_mm", "tmean_sep_apr_c", "srad_sep_apr"]]
        summ.append({"min_hours_rain_temp": h, "min_hours_srad": sh,
                     "rain_mean_abs_diff_mm": d["rain_sep_apr_mm"].abs().mean(), "rain_max_abs_diff_mm": d["rain_sep_apr_mm"].abs().max(),
                     "tmean_mean_abs_diff_c": d["tmean_sep_apr_c"].abs().mean(), "srad_mean_abs_diff": d["srad_sep_apr"].abs().mean(),
                     "temp30_mean_abs_diff_c": (g["ctx_2"] - base["ctx_2"]).abs().mean()})
    pd.DataFrame(summ).to_csv(out / "qc_threshold_sensitivity.csv", index=False)
    print(pd.DataFrame(summ).round(2).to_string(index=False))


# ---------------------------------------------------------------- DSSAT jobs
def _season_job(args):
    cfg, year, offsets, mode = args
    project_dir = Path(cfg["_project_dir"])
    run_cfg = copy.deepcopy(cfg)
    if mode == "no_spinup":
        run_cfg["simulation"]["spinup_start_month_day"] = None
    daily = load_weather(project_dir, run_cfg)
    yw = build_year_weather_with_context(daily, [year], run_cfg)[0]
    runner = PyDSSATRunner(run_cfg, project_dir)
    ag = run_cfg["agronomy"]
    rows = []
    for off in offsets:
        pdate = planting_date_for_year(year, ag["planting_window_start"], off)
        empty = build_irrigation_schedule(yw.daily, pdate, ag["season_length_days"], 9999, 0, 0, 4)
        sim = runner.run(yw.daily, pdate, empty, None)
        sw = _soilwat(runner.last_output_files["SoilWat"])
        at_sow = sw[sw["date"].dt.date == pdate]
        row = {"mode": mode, "year": year, "offset": off, "yield_kg_ha": sim.yield_kg_ha,
               "swxd_at_sowing_mm": float(at_sow["SWXD"].iloc[0]) if len(at_sow) else np.nan,
               "swxd_first_day_mm": float(sw["SWXD"].iloc[0])}
        if mode == "spinup" and off == 30:
            proxy = proxy_balance(yw.daily, pdate, ag["season_length_days"])
            m = proxy.merge(sw[["date", "SWXD"]], on="date")
            m["dssat_deficit_mm"] = 198.0 - m["SWXD"]
            m["year"] = year
            row["daily"] = m[["year", "date", "proxy_deficit_mm", "dssat_deficit_mm"]].to_dict("records")
        rows.append(row)
    return rows


def _interval_job(args):
    cfg, year, candidates, check_days = args
    project_dir = Path(cfg["_project_dir"])
    daily = load_weather(project_dir, cfg)
    yw = build_year_weather_with_context(daily, [year], cfg)[0]
    runner = PyDSSATRunner(cfg, project_dir)
    ag = cfg["agronomy"]
    rows = []
    for c in candidates:
        pdate = planting_date_for_year(year, ag["planting_window_start"], c["planting_offset_days"])
        sched = build_irrigation_schedule(yw.daily, pdate, ag["season_length_days"], c["trigger_dryness"], c["amount_mm"],
                                          c["max_irrigation_mm"], check_days, ag.get("min_irrigation_event_mm", 0.0))
        sim = runner.run(yw.daily, pdate, sched, None)
        rows.append({"check_days": check_days, "year": year, **c, "yield_kg_ha": sim.yield_kg_ha, "irrigation_mm": sim.irrigation_mm,
                     "reward": sim.yield_kg_ha / cfg["reward"]["target_yield_kg_ha"] - cfg["reward"]["water_penalty_per_mm"] * sim.irrigation_mm})
    return rows


def proxy_and_spinup(cfg_path: str, out: Path, workers: int) -> None:
    cfg = load_config(cfg_path)
    offsets = list(range(0, 96, 8)) + [96, 30]
    jobs = [(cfg, y, offsets, mode) for mode in ["spinup", "no_spinup"] for y in SEASONS]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [r for chunk in pool.map(_season_job, jobs) for r in chunk]
    daily_rows = [d for r in rows for d in r.pop("daily", [])]
    df = pd.DataFrame(rows)
    df.to_csv(out / "spinup_rainfed_by_season.csv", index=False)
    dd = pd.DataFrame(daily_rows)
    dd.to_csv(out / "proxy_vs_dssat_daily.csv", index=False)

    # Proxy versus DSSAT soil-water deficit (rainfed, sowing October 15).
    per = dd.groupby("year").apply(lambda g: pd.Series({"pearson_r": g["proxy_deficit_mm"].corr(g["dssat_deficit_mm"]),
                                                         "spearman_rho": g["proxy_deficit_mm"].corr(g["dssat_deficit_mm"], method="spearman")}))
    per.to_csv(out / "proxy_vs_dssat_by_season.csv")
    bins = pd.cut(dd["proxy_deficit_mm"], [-0.1, 30, 60, 90, 120, 150, 1e9], labels=["0-30", "30-60", "60-90", "90-120", "120-150", ">150"])
    by_bin = dd.groupby(bins, observed=True)["dssat_deficit_mm"].agg(["count", "mean", "median"]).reset_index()
    by_bin.to_csv(out / "proxy_bins_dssat_deficit.csv", index=False)
    pooled = {"days": len(dd), "pearson_r": dd["proxy_deficit_mm"].corr(dd["dssat_deficit_mm"]),
              "spearman_rho": dd["proxy_deficit_mm"].corr(dd["dssat_deficit_mm"], method="spearman"),
              "season_r_median": per["pearson_r"].median(), "season_r_min": per["pearson_r"].min(), "season_r_max": per["pearson_r"].max()}
    pd.DataFrame([pooled]).to_csv(out / "proxy_vs_dssat_summary.csv", index=False)
    print(pd.DataFrame([pooled]).round(3).to_string(index=False)); print(by_bin.round(1).to_string(index=False))

    # Proxy demand versus FAO-56 Hargreaves reference evapotranspiration (NASA POWER series, September-April).
    daily = load_weather(Path(cfg["_project_dir"]), cfg)
    d = daily[daily["date"].dt.month.isin([9, 10, 11, 12, 1, 2, 3, 4]) & (daily["date"].dt.year >= 1984)].copy()
    lat = np.deg2rad(cfg["dssat"]["latitude"])
    doy = d["date"].dt.dayofyear.to_numpy()
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    decl = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(lat) * np.tan(decl), -1, 1))
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (ws * np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.sin(ws))
    d["et0_hargreaves_mm"] = 0.0023 * 0.408 * ra * (d["temp_mean"] + 17.8) * np.sqrt((d["tmax"] - d["tmin"]).clip(lower=0))
    d["proxy_demand_mm"] = (0.16 * d["srad"] + 0.08 * (d["temp_mean"] - 10).clip(lower=0)).clip(lower=0)
    et = {"days": len(d), "proxy_mean_mm_d": d["proxy_demand_mm"].mean(), "et0_mean_mm_d": d["et0_hargreaves_mm"].mean(),
          "ratio_proxy_to_et0": d["proxy_demand_mm"].sum() / d["et0_hargreaves_mm"].sum(),
          "pearson_r": d["proxy_demand_mm"].corr(d["et0_hargreaves_mm"])}
    pd.DataFrame([et]).to_csv(out / "proxy_vs_et0_summary.csv", index=False)
    print(pd.DataFrame([et]).round(3).to_string(index=False))

    # Spin-up effect on rainfed yield, soil water at sowing and the ranking of sowing dates.
    g = df[df["offset"].isin(range(0, 97, 8)) | (df["offset"] == 96)].drop_duplicates(["mode", "year", "offset"])
    w = g.pivot_table(index=["year", "offset"], columns="mode", values=["yield_kg_ha", "swxd_at_sowing_mm"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.reset_index()
    w["yield_diff_kg_ha"] = w["yield_kg_ha_spinup"] - w["yield_kg_ha_no_spinup"]
    rank = w.groupby("year").apply(lambda x: pd.Series({
        "rank_spearman": x["yield_kg_ha_spinup"].corr(x["yield_kg_ha_no_spinup"], method="spearman"),
        "best_offset_spinup": x.loc[x["yield_kg_ha_spinup"].idxmax(), "offset"],
        "best_offset_no_spinup": x.loc[x["yield_kg_ha_no_spinup"].idxmax(), "offset"],
        "mean_abs_yield_diff": x["yield_diff_kg_ha"].abs().mean(),
        "swxd_at_sowing_spinup_mean": x["swxd_at_sowing_mm_spinup"].mean(),
        "swxd_at_sowing_spinup_min": x["swxd_at_sowing_mm_spinup"].min(),
        "swxd_at_sowing_no_spinup_mean": x["swxd_at_sowing_mm_no_spinup"].mean()}))
    rank.to_csv(out / "spinup_effect_by_season.csv")
    s = {"seasons": len(rank), "mean_abs_yield_diff_kg_ha": w["yield_diff_kg_ha"].abs().mean(),
         "max_abs_yield_diff_kg_ha": w["yield_diff_kg_ha"].abs().max(), "rank_spearman_median": rank["rank_spearman"].median(),
         "rank_spearman_min": rank["rank_spearman"].min(), "seasons_best_offset_changed": int((rank["best_offset_spinup"] != rank["best_offset_no_spinup"]).sum()),
         "swxd_at_sowing_spinup_mean": w["swxd_at_sowing_mm_spinup"].mean(), "swxd_at_sowing_spinup_min": w["swxd_at_sowing_mm_spinup"].min(),
         "swxd_at_sowing_no_spinup_mean": w["swxd_at_sowing_mm_no_spinup"].mean()}
    pd.DataFrame([s]).to_csv(out / "spinup_effect_summary.csv", index=False)
    print(pd.DataFrame([s]).round(2).to_string(index=False))


INTERVALS = (1, 3, 7, 12)


def interval(cfg_path: str, out: Path, workers: int) -> None:
    cfg = load_config(cfg_path)
    cands = [c for c in candidate_grid() if c["planting_offset_days"] in (24, 32, 40) and c["candidate_id"] not in CONSTRUCTED]
    path = out / "inspection_interval_grid.csv"
    grid = pd.read_csv(path) if path.exists() else pd.DataFrame()
    missing = [k for k in INTERVALS if grid.empty or k not in set(grid["check_days"])]
    if missing:
        jobs = [(cfg, y, cands, k) for k in missing for y in SEASONS]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            new = pd.DataFrame([r for chunk in pool.map(_interval_job, jobs) for r in chunk])
        grid = pd.concat([grid, new], ignore_index=True)
    if grid.empty or 4 not in set(grid["check_days"]):
        main = pd.read_csv(Path(cfg["_project_dir"]) / "outputs" / "rolling_power_analysis" / "tables" / "candidate_grid_all_seasons.csv")
        main = main[main["candidate_id"].isin([c["candidate_id"] for c in cands])].assign(check_days=4)
        grid = pd.concat([grid, main[grid.columns.intersection(main.columns)]], ignore_index=True)
    grid.to_csv(path, index=False)
    rows = []
    for k, g in grid.groupby("check_days"):
        test_rows = []
        for f in folds(cfg):
            pool = g[g["year"].isin(f["train"] + f["valid"])]
            fid = select_fixed_rule(pool)
            test_rows.append(g[(g["year"].isin(f["test"])) & (g["candidate_id"] == fid)].assign(fold=f["fold"]))
        t = pd.concat(test_rows)
        rows.append({"check_days": k, "fixed_rule_reward": t["reward"].mean(), "fixed_rule_yield_kg_ha": t["yield_kg_ha"].mean(),
                     "fixed_rule_irrigation_mm": t["irrigation_mm"].mean(),
                     "selected_rules": ";".join(sorted(t["candidate_id"].unique()))})
    res = pd.DataFrame(rows)
    res.to_csv(out / "inspection_interval_sensitivity.csv", index=False)
    print(res.round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["thresholds", "proxy", "interval"])
    parser.add_argument("--config", default="configs/experiment_castro_power.yaml")
    parser.add_argument("--station-config", default="configs/experiment_castro_t0.yaml")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = make_paths(cfg, "supplementary_analyses").tables_dir
    if args.command == "thresholds":
        thresholds(args.station_config, out)
    elif args.command == "proxy":
        proxy_and_spinup(args.config, out, args.workers)
    else:
        interval(args.config, out, args.workers)


if __name__ == "__main__":
    main()
