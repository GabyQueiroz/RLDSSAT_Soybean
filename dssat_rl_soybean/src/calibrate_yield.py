from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import pandas as pd

from .config import load_config, make_paths
from .data import build_year_weather, load_observed_soybean_yield, load_weather
from .dssat_adapter import PyDSSATRunner, build_irrigation_schedule
from .yield_correction import fit_ridge


ALPHAS = [0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 300.0, 500.0]
PLANTING_DATES = [
    (9, 25, "sep25"),
    (10, 5, "oct05"),
    (10, 15, "oct15"),
    (10, 25, "oct25"),
    (11, 5, "nov05"),
]


def _metrics(df: pd.DataFrame, pred_col: str) -> dict:
    err = df[pred_col] - df["observed_yield_kg_ha"]
    denom = df["observed_yield_kg_ha"].replace(0, np.nan)
    return {
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "mape_percent": float((err.abs() / denom).mean() * 100),
        "bias": float(err.mean()),
        "n": int(len(df)),
    }


def _annual_table(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.groupby(["split", "year"], as_index=False).agg(
        raw_yield_kg_ha=("raw_yield_kg_ha", "mean"),
        observed_yield_kg_ha=("observed_yield_kg_ha", "first"),
        rain_mm=("rain_mm", "mean"),
        irrigation_mm=("irrigation_mm", "mean"),
        planting_doy=("planting_doy", "mean"),
        temp_mean_c=("temp_mean_c", "first"),
        srad_mj_m2_day=("srad_mj_m2_day", "first"),
    )


def _annual_prediction_summary(raw: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    return raw.groupby(["split", "year"], as_index=False).agg(
        predicted_yield_kg_ha=(pred_col, "mean"),
        observed_yield_kg_ha=("observed_yield_kg_ha", "first"),
    )


def main():
    parser = argparse.ArgumentParser(description="Calibrate DSSAT yield correction against SIDRA observed yield.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--run-name", default="yield_calibration")
    parser.add_argument(
        "--fit-all-final",
        action="store_true",
        help="Fit the saved operational correction with train, validation and test years. Do not use this for an independent test claim.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["backend"] = "dssat"
    cfg["calibration"]["enabled"] = False
    cfg["dssat"]["keep_success_runs"] = False
    cfg["dssat"]["keep_failed_runs"] = False
    paths = make_paths(cfg, args.run_name)

    weather = load_weather(paths.project_dir, cfg)
    observed = load_observed_soybean_yield(paths.project_dir, cfg)
    obs = dict(zip(observed["ano"].astype(int), observed["observed_yield_kg_ha"].astype(float)))
    runner = PyDSSATRunner(cfg, paths.project_dir)

    split_by_year = {}
    for split_name in ["train", "valid", "test"]:
        for year in cfg["calibration"].get(f"{split_name}_years", []):
            split_by_year[int(year)] = split_name

    years = [year for year in split_by_year if year in obs]
    rows = []
    for year in years:
        yw = build_year_weather(weather, [year])[0]
        for month, day, label in PLANTING_DATES:
            planting = date(year, month, day)
            schedule = build_irrigation_schedule(
                yw.daily,
                planting,
                cfg["agronomy"]["season_length_days"],
                trigger_dryness=45,
                amount_mm=20,
                max_season_irrigation_mm=120,
                check_days=cfg["agronomy"]["irrigation_check_days"],
            )
            sim = runner.run(yw.daily, planting, schedule, rng=None)
            row = {
                "split": split_by_year[year],
                "year": year,
                "planting_label": label,
                "planting_date": planting.isoformat(),
                "raw_yield_kg_ha": sim.yield_kg_ha,
                "observed_yield_kg_ha": obs[year],
                "rain_mm": sim.rain_mm,
                "irrigation_mm": sim.irrigation_mm,
                "planting_doy": planting.timetuple().tm_yday,
                "temp_mean_c": float(yw.daily["temp_mean"].mean()),
                "srad_mj_m2_day": float(yw.daily["srad"].mean()),
            }
            rows.append(row)
            print(row)

    raw = pd.DataFrame(rows)
    annual = _annual_table(raw)
    train_rows = raw[raw["split"] == "train"].to_dict("records")

    search_rows = []
    best = None
    for alpha in ALPHAS:
        candidate, train_metrics = fit_ridge(train_rows, alpha=alpha)
        eval_rows = raw.copy()
        eval_records = eval_rows.to_dict("records")
        eval_rows["predicted_yield_kg_ha"] = [candidate.predict_one(row) for row in eval_records]
        annual_eval = _annual_prediction_summary(eval_rows, "predicted_yield_kg_ha")
        valid_eval = annual_eval[annual_eval["split"] == "valid"].copy()
        valid_metrics = _metrics(valid_eval, "predicted_yield_kg_ha")
        validation_bias = float((valid_eval["predicted_yield_kg_ha"] - valid_eval["observed_yield_kg_ha"]).mean())
        annual_eval["bias_corrected_yield_kg_ha"] = annual_eval["predicted_yield_kg_ha"] - validation_bias
        valid_bias_corrected_metrics = _metrics(
            annual_eval[annual_eval["split"] == "valid"], "bias_corrected_yield_kg_ha"
        )
        search_row = {
            "alpha": alpha,
            "validation_intercept_adjustment": -validation_bias,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
            **{f"valid_bias_corrected_{key}": value for key, value in valid_bias_corrected_metrics.items()},
        }
        search_rows.append(search_row)
        if best is None or valid_bias_corrected_metrics["mae"] < best["valid_bias_corrected_mae"]:
            best = search_row

    pd.DataFrame(search_rows).to_csv(paths.tables_dir / "yield_correction_alpha_search.csv", index=False, encoding="utf-8-sig")
    fit_source = raw if args.fit_all_final else raw[raw["split"] == "train"]
    model, fit_metrics = fit_ridge(fit_source.to_dict("records"), alpha=float(best["alpha"]))
    if not args.fit_all_final:
        model.intercept += float(best["validation_intercept_adjustment"])
    model_path = paths.project_dir / cfg["calibration"]["model_path"]
    model.save(
        model_path,
        {
            "selected_alpha": float(best["alpha"]),
            "selected_by": "lowest validation MAE after validation intercept correction",
            "validation_intercept_adjustment": 0.0 if args.fit_all_final else float(best["validation_intercept_adjustment"]),
            "fit_all_final": bool(args.fit_all_final),
            "fit_metrics": fit_metrics,
        },
    )

    out = raw.copy()
    out["corrected_yield_kg_ha"] = [model.predict_one(row) for row in rows]
    out["raw_abs_error"] = (out["raw_yield_kg_ha"] - out["observed_yield_kg_ha"]).abs()
    out["corrected_abs_error"] = (out["corrected_yield_kg_ha"] - out["observed_yield_kg_ha"]).abs()
    out.to_csv(paths.tables_dir / "yield_calibration_rows.csv", index=False, encoding="utf-8-sig")

    summary = out.groupby(["split", "year"], as_index=False).agg(
        observed_yield_kg_ha=("observed_yield_kg_ha", "first"),
        raw_yield_kg_ha=("raw_yield_kg_ha", "mean"),
        corrected_yield_kg_ha=("corrected_yield_kg_ha", "mean"),
        raw_abs_error=("raw_abs_error", "mean"),
        corrected_abs_error=("corrected_abs_error", "mean"),
    )
    summary["corrected_error_kg_ha"] = summary["corrected_yield_kg_ha"] - summary["observed_yield_kg_ha"]
    summary["corrected_abs_percent_error"] = 100 * summary["corrected_abs_error"] / summary["observed_yield_kg_ha"]
    summary.to_csv(paths.tables_dir / "yield_calibration_summary.csv", index=False, encoding="utf-8-sig")

    split_metrics = []
    for split_name, split_df in summary.groupby("split"):
        split_eval = split_df.rename(columns={"corrected_yield_kg_ha": "predicted_yield_kg_ha"})
        split_metrics.append({"split": split_name, **_metrics(split_eval, "predicted_yield_kg_ha")})
    metrics_by_split = pd.DataFrame(split_metrics)
    metrics_by_split.to_csv(paths.tables_dir / "yield_correction_metrics_by_split.csv", index=False, encoding="utf-8-sig")

    print("Saved", model_path)
    print("Selected alpha", best["alpha"])
    print(metrics_by_split.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
