from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, make_paths
from .data import build_year_weather_with_context, load_weather, planting_date_for_year
from .dssat_adapter import PyDSSATRunner, build_irrigation_schedule


def candidate_grid() -> list[dict]:
    offsets = [0, 15, 30, 45, 60, 75, 96]
    triggers = [35, 55, 75]
    amounts = [0, 12, 24]
    caps = [0, 60, 120]
    seen: set[tuple] = set()
    candidates: list[dict] = []
    for offset, trigger, amount, cap in product(offsets, triggers, amounts, caps):
        if amount == 0 or cap == 0:
            trigger, amount, cap = 9999, 0, 0
        key = (offset, trigger, amount, cap)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "candidate_id": f"off{offset:03d}_trig{trigger}_amt{amount}_cap{cap}",
                "planting_offset_days": offset,
                "trigger_dryness": trigger,
                "amount_mm": amount,
                "max_irrigation_mm": cap,
            }
        )
    return candidates


def evaluate_grid(cfg: dict, project_dir: Path, split: str, candidates: list[dict], out_path: Path) -> pd.DataFrame:
    if out_path.exists():
        return pd.read_csv(out_path)

    daily = load_weather(project_dir, cfg)
    years = build_year_weather_with_context(daily, cfg["data"][f"{split}_years"], cfg)
    runner = PyDSSATRunner(cfg, project_dir)
    rng = np.random.default_rng(cfg["seed"] + 91_000)
    rows = []
    for candidate in candidates:
        for yw in years:
            planting = planting_date_for_year(
                yw.year, cfg["agronomy"]["planting_window_start"], candidate["planting_offset_days"]
            )
            schedule = build_irrigation_schedule(
                yw.daily,
                planting,
                cfg["agronomy"]["season_length_days"],
                candidate["trigger_dryness"],
                candidate["amount_mm"],
                candidate["max_irrigation_mm"],
                cfg["agronomy"]["irrigation_check_days"],
            )
            try:
                sim = runner.run(yw.daily, planting, schedule, rng)
                reward = sim.yield_kg_ha / cfg["reward"]["target_yield_kg_ha"] - cfg["reward"]["water_penalty_per_mm"] * sim.irrigation_mm
                rows.append(
                    {
                        "split": split,
                        "year": yw.year,
                        "failed": False,
                        "planting_date": planting.isoformat(),
                        "yield_kg_ha": sim.yield_kg_ha,
                        "irrigation_mm": sim.irrigation_mm,
                        "rain_mm": sim.rain_mm,
                        "reward": reward,
                        **candidate,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "split": split,
                        "year": yw.year,
                        "failed": True,
                        "planting_date": planting.isoformat(),
                        "yield_kg_ha": np.nan,
                        "irrigation_mm": np.nan,
                        "rain_mm": np.nan,
                        "reward": cfg["reward"]["failed_run_penalty"],
                        "error": repr(exc),
                        **candidate,
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out


def _nearest_validation_years(cfg: dict, project_dir: Path) -> pd.DataFrame:
    daily = load_weather(project_dir, cfg)
    valid = build_year_weather_with_context(daily, cfg["data"]["valid_years"], cfg)
    test = build_year_weather_with_context(daily, cfg["data"]["test_years"], cfg)
    valid_features = np.vstack([v.features for v in valid])
    rows = []
    for t in test:
        distances = np.linalg.norm(valid_features - t.features, axis=1)
        idx = int(np.argmin(distances))
        rows.append({"test_year": t.year, "nearest_valid_year": valid[idx].year, "context_distance": float(distances[idx])})
    return pd.DataFrame(rows)


def build_comparators(valid_grid: pd.DataFrame, test_grid: pd.DataFrame, cfg: dict, project_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_ok = valid_grid.copy()
    test_ok = test_grid.copy()

    fixed_summary = (
        valid_ok.groupby("candidate_id", as_index=False)
        .agg(
            valid_reward_mean=("reward", "mean"),
            valid_yield_mean_kg_ha=("yield_kg_ha", "mean"),
            valid_irrigation_mean_mm=("irrigation_mm", "mean"),
            valid_failures=("failed", "sum"),
        )
        .sort_values(["valid_reward_mean", "valid_failures"], ascending=[False, True])
    )
    best_fixed_id = fixed_summary.iloc[0]["candidate_id"]
    fixed = test_ok[test_ok["candidate_id"] == best_fixed_id].copy()
    fixed["comparator"] = "optimized_fixed_rule"

    year_best_valid = valid_ok.sort_values("reward", ascending=False).drop_duplicates("year")
    nearest = _nearest_validation_years(cfg, project_dir)
    contextual_rows = []
    for row in nearest.itertuples(index=False):
        source = year_best_valid[year_best_valid["year"] == row.nearest_valid_year].iloc[0]
        selected = test_ok[(test_ok["year"] == row.test_year) & (test_ok["candidate_id"] == source["candidate_id"])].copy()
        selected["comparator"] = "simple_contextual_nearest_year"
        selected["source_valid_year"] = row.nearest_valid_year
        selected["context_distance"] = row.context_distance
        contextual_rows.append(selected)
    contextual = pd.concat(contextual_rows, ignore_index=True)

    oracle = test_ok.sort_values("reward", ascending=False).drop_duplicates("year").copy()
    oracle["comparator"] = "retrospective_grid_oracle"

    comparator_rows = pd.concat([fixed, contextual, oracle], ignore_index=True)
    comparator_rows.to_csv(Path(project_dir) / "outputs" / "final_comparators_castro" / "tables" / "final_comparator_rows_test.csv", index=False, encoding="utf-8-sig")
    fixed_summary.to_csv(Path(project_dir) / "outputs" / "final_comparators_castro" / "tables" / "fixed_grid_validation_summary.csv", index=False, encoding="utf-8-sig")
    nearest.to_csv(Path(project_dir) / "outputs" / "final_comparators_castro" / "tables" / "contextual_nearest_year_map.csv", index=False, encoding="utf-8-sig")
    return comparator_rows, fixed_summary


def paired_effects(comparators: pd.DataFrame, ppo_all_seeds: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    ppo = ppo_all_seeds[ppo_all_seeds["split"] == "test"].copy()
    rows = []
    for comp_name, comp in comparators.groupby("comparator"):
        merged = ppo.merge(
            comp[["year", "yield_kg_ha", "irrigation_mm", "reward"]],
            on="year",
            suffixes=("_ppo", "_comparator"),
        )
        merged["comparator"] = comp_name
        merged["d_reward"] = merged["reward_ppo"] - merged["reward_comparator"]
        merged["d_yield"] = merged["yield_kg_ha_ppo"] - merged["yield_kg_ha_comparator"]
        merged["d_irrigation"] = merged["irrigation_mm_ppo"] - merged["irrigation_mm_comparator"]
        rows.append(merged)
    effects = pd.concat(rows, ignore_index=True)
    effects.to_csv(out_dir / "final_primary_paired_effects_seed_year.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for comp_name, group in effects.groupby("comparator"):
        seed_means = group.groupby("seed")[["d_reward", "d_yield", "d_irrigation"]].mean()
        year_means = group.groupby("year")[["d_reward", "d_yield", "d_irrigation"]].mean()
        summary_rows.append(
            {
                "comparator": comp_name,
                "n_seasons": group["year"].nunique(),
                "n_seeds": group["seed"].nunique(),
                "mean_d_reward": group["d_reward"].mean(),
                "median_d_reward": group["d_reward"].median(),
                "seed_sd_d_reward": seed_means["d_reward"].std(ddof=1),
                "mean_d_yield_kg_ha": group["d_yield"].mean(),
                "median_d_yield_kg_ha": group["d_yield"].median(),
                "seed_sd_d_yield_kg_ha": seed_means["d_yield"].std(ddof=1),
                "year_min_d_yield_kg_ha": year_means["d_yield"].min(),
                "year_max_d_yield_kg_ha": year_means["d_yield"].max(),
                "mean_d_irrigation_mm": group["d_irrigation"].mean(),
                "median_d_irrigation_mm": group["d_irrigation"].median(),
                "seed_sd_d_irrigation_mm": seed_means["d_irrigation"].std(ddof=1),
                "year_min_d_irrigation_mm": year_means["d_irrigation"].min(),
                "year_max_d_irrigation_mm": year_means["d_irrigation"].max(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "final_primary_paired_effects_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate final Castro comparators with the same DSSAT candidate grid.")
    parser.add_argument("--config", default="configs/_generated_multiseed/experiment_seed_42.yaml")
    parser.add_argument("--run-name", default="final_comparators_castro")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["backend"] = "dssat"
    cfg["dssat"]["keep_success_runs"] = False
    cfg["dssat"]["keep_failed_runs"] = False
    paths = make_paths(cfg, args.run_name)
    candidates = candidate_grid()
    pd.DataFrame(candidates).to_csv(paths.tables_dir / "candidate_grid.csv", index=False, encoding="utf-8-sig")

    valid = evaluate_grid(cfg, paths.project_dir, "valid", candidates, paths.tables_dir / "candidate_grid_valid.csv")
    test = evaluate_grid(cfg, paths.project_dir, "test", candidates, paths.tables_dir / "candidate_grid_test.csv")
    comparators, _ = build_comparators(valid, test, cfg, paths.project_dir)

    ppo_path = paths.project_dir / "outputs" / "multiseed_final_summary" / "policy_evaluation_all_seeds.csv"
    ppo = pd.read_csv(ppo_path)
    summary = paired_effects(comparators, ppo, paths.tables_dir)

    call_rows = [
        {"method": "candidate_grid_validation", "dssat_calls": int(len(valid))},
        {"method": "candidate_grid_test", "dssat_calls": int(len(test))},
        {"method": "ppo_policy_evaluation_rows", "dssat_calls": int(len(ppo[ppo["split"] == "test"]))},
    ]
    pd.DataFrame(call_rows).to_csv(paths.tables_dir / "computational_budget_dssat_calls.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
