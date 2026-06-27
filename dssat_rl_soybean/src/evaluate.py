from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from .config import load_config, make_paths
from .data import build_year_weather, load_observed_soybean_yield, load_weather
from .dssat_adapter import MockDSSATRunner, PyDSSATRunner
from .env import SoybeanDSSATEnv
from .report import build_report


def _attach_observed(paths, cfg, out: pd.DataFrame, split: str) -> pd.DataFrame:
    observed = load_observed_soybean_yield(paths.project_dir, cfg)
    observed = observed.rename(columns={"ano": "year"})
    merged = out.merge(observed, on="year", how="left")
    if "yield_kg_ha" in merged and "observed_yield_kg_ha" in merged:
        merged["yield_error_kg_ha"] = merged["yield_kg_ha"] - merged["observed_yield_kg_ha"]
        merged["yield_abs_error_kg_ha"] = merged["yield_error_kg_ha"].abs()
        merged["yield_abs_percent_error"] = 100 * merged["yield_abs_error_kg_ha"] / merged["observed_yield_kg_ha"]
        metrics = (
            merged.dropna(subset=["observed_yield_kg_ha"])
            .groupby("split")
            .agg(
                mae_kg_ha=("yield_abs_error_kg_ha", "mean"),
                rmse_kg_ha=("yield_error_kg_ha", lambda s: float(np.sqrt(np.mean(np.square(s))))),
                mape_percent=("yield_abs_percent_error", "mean"),
                bias_kg_ha=("yield_error_kg_ha", "mean"),
                n=("yield_error_kg_ha", "count"),
            )
            .reset_index()
        )
        metrics.to_csv(paths.tables_dir / f"policy_observed_metrics_{split}.csv", index=False, encoding="utf-8-sig")
    return merged


def evaluate_policy(cfg, paths, split: str, model_path=None, episodes_per_year: int = 1) -> pd.DataFrame:
    daily = load_weather(paths.project_dir, cfg)
    years = build_year_weather(daily, cfg["data"][f"{split}_years"])
    runner = PyDSSATRunner(cfg, paths.project_dir) if cfg["backend"] == "dssat" else MockDSSATRunner(cfg)
    env = SoybeanDSSATEnv(cfg, years, runner, cfg["seed"] + 20_000)
    default_model = paths.models_dir / "best_model"
    if not (default_model.with_suffix(".zip")).exists():
        default_model = paths.models_dir / "final_model"
    model = PPO.load(model_path or default_model)
    rows = []
    for yw in years:
        env.current = yw
        obs = env._obs(yw)
        for rep in range(episodes_per_year):
            action, _ = model.predict(obs, deterministic=True)
            _, reward, _, _, info = env.step(action)
            rows.append({"split": split, "rep": rep, "reward": reward, **info})
    out = _attach_observed(paths, cfg, pd.DataFrame(rows), split)
    out.to_csv(paths.tables_dir / f"policy_evaluation_{split}.csv", index=False, encoding="utf-8-sig")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--run-name", default="ppo_soybean")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--backend", choices=["mock", "dssat"], default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.backend:
        cfg["backend"] = args.backend
    paths = make_paths(cfg, args.run_name)
    evaluate_policy(cfg, paths, args.split)
    build_report(cfg, paths)


if __name__ == "__main__":
    main()
