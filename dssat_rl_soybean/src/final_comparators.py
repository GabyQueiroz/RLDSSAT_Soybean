from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, make_paths
from .data import build_year_weather_with_context, context_scaler, load_weather, planting_date_for_year
from .dssat_adapter import PyDSSATRunner, build_irrigation_schedule

RAINFED = {"trigger_dryness": 9999.0, "amount_mm": 0.0, "max_irrigation_mm": 0.0}

CONSTRUCTED = {
    "rainfed_oct15": {"planting_offset_days": 30, **RAINFED},
    "irrigated_sep25_45_18_120": {"planting_offset_days": 10, "trigger_dryness": 45.0, "amount_mm": 18.0, "max_irrigation_mm": 120.0},
    "irrigated_oct15_45_18_120": {"planting_offset_days": 30, "trigger_dryness": 45.0, "amount_mm": 18.0, "max_irrigation_mm": 120.0},
    "irrigated_nov10_45_18_120": {"planting_offset_days": 56, "trigger_dryness": 45.0, "amount_mm": 18.0, "max_irrigation_mm": 120.0},
}


def candidate_grid() -> list[dict]:
    offsets = list(range(0, 96, 8)) + [96]
    irrigation = [RAINFED] + [
        {"trigger_dryness": float(t), "amount_mm": float(a), "max_irrigation_mm": float(c)}
        for t, a, c in product([30, 60, 90, 120, 150], [15, 25, 35], [80, 160, 240])
    ]
    candidates = []
    for offset, irr in product(offsets, irrigation):
        cid = f"off{offset:03d}_trig{int(irr['trigger_dryness'])}_amt{int(irr['amount_mm'])}_cap{int(irr['max_irrigation_mm'])}"
        candidates.append({"candidate_id": cid, "planting_offset_days": offset, **irr})
    for name, rule in CONSTRUCTED.items():
        candidates.append({"candidate_id": name, **rule})
    return candidates


def _evaluate_year(args) -> list[dict]:
    cfg, project_dir, year, candidates = args
    daily = load_weather(Path(project_dir), cfg)
    yw = build_year_weather_with_context(daily, [year], cfg)[0]
    runner = PyDSSATRunner(cfg, Path(project_dir))
    ag = cfg["agronomy"]
    rows = []
    for cand in candidates:
        planting = planting_date_for_year(year, ag["planting_window_start"], cand["planting_offset_days"])
        schedule = build_irrigation_schedule(
            yw.daily,
            planting,
            ag["season_length_days"],
            cand["trigger_dryness"],
            cand["amount_mm"],
            cand["max_irrigation_mm"],
            ag["irrigation_check_days"],
            ag.get("min_irrigation_event_mm", 0.0),
        )
        row = {"year": year, "planting_date": planting.isoformat(), **cand}
        try:
            sim = runner.run(yw.daily, planting, schedule, None)
            row.update(
                failed=False,
                yield_kg_ha=sim.yield_kg_ha,
                irrigation_mm=sim.irrigation_mm,
                rain_mm=sim.rain_mm,
                reward=sim.yield_kg_ha / cfg["reward"]["target_yield_kg_ha"] - cfg["reward"]["water_penalty_per_mm"] * sim.irrigation_mm,
            )
        except Exception as exc:
            row.update(failed=True, yield_kg_ha=np.nan, irrigation_mm=np.nan, rain_mm=np.nan,
                       reward=cfg["reward"]["failed_run_penalty"], error=repr(exc))
        rows.append(row)
    return rows


def evaluate_grid(cfg: dict, project_dir: Path, years: list[int], candidates: list[dict], out_path: Path, workers: int) -> pd.DataFrame:
    if out_path.exists():
        return pd.read_csv(out_path)
    jobs = [(cfg, str(project_dir), year, candidates) for year in years]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [row for chunk in pool.map(_evaluate_year, jobs) for row in chunk]
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out


def with_kappa(grid: pd.DataFrame, cfg: dict, kappa: float) -> pd.DataFrame:
    out = grid.copy()
    ok = ~out["failed"].astype(bool)
    out.loc[ok, "reward"] = out.loc[ok, "yield_kg_ha"] / cfg["reward"]["target_yield_kg_ha"] - kappa * out.loc[ok, "irrigation_mm"]
    return out


def select_fixed_rule(dev_grid: pd.DataFrame) -> str:
    """Best non-contextual rule over the development seasons (mean optimisation function)."""
    grid = dev_grid[~dev_grid["candidate_id"].isin(CONSTRUCTED)]
    summary = grid.groupby("candidate_id")["reward"].mean().sort_values(ascending=False)
    return str(summary.index[0])


def contextual_knn(dev_grid: pd.DataFrame, dev_context: pd.DataFrame, target_context: pd.DataFrame, k: int) -> pd.DataFrame:
    """For each target season, the rule with the best mean outcome over its k nearest development seasons."""
    grid = dev_grid[~dev_grid["candidate_id"].isin(CONSTRUCTED)]
    table = grid.pivot_table(index="year", columns="candidate_id", values="reward")
    feats = [c for c in dev_context.columns if c != "year"]
    rows = []
    for t in target_context.itertuples(index=False):
        pool = dev_context[dev_context["year"] != t.year]
        dist = np.linalg.norm(pool[feats].to_numpy() - np.asarray([getattr(t, f) for f in feats]), axis=1)
        neighbours = pool["year"].to_numpy()[np.argsort(dist)[:k]]
        best = table.loc[neighbours].mean().idxmax()
        rows.append({"year": t.year, "candidate_id": best, "neighbour_years": ",".join(map(str, sorted(neighbours)))})
    return pd.DataFrame(rows)


def context_table(cfg: dict, project_dir: Path, years: list[int], scaler) -> pd.DataFrame:
    daily = load_weather(project_dir, cfg)
    seasons = build_year_weather_with_context(daily, years, cfg)
    mean, std = scaler
    data = np.nan_to_num(np.vstack([(s.features - mean) / std for s in seasons]), nan=0.0)
    out = pd.DataFrame(data, columns=[f"c{i}" for i in range(data.shape[1])])
    out.insert(0, "year", [s.year for s in seasons])
    return out


def build_comparators(cfg: dict, project_dir: Path, dev_grid: pd.DataFrame, target_grid: pd.DataFrame, k: int) -> pd.DataFrame:
    daily = load_weather(project_dir, cfg)
    train = build_year_weather_with_context(daily, cfg["data"]["train_years"], cfg)
    scaler = context_scaler(train)
    dev_years = sorted(dev_grid["year"].unique().tolist())
    target_years = sorted(target_grid["year"].unique().tolist())
    dev_ctx = context_table(cfg, project_dir, dev_years, scaler)
    target_ctx = context_table(cfg, project_dir, target_years, scaler)

    parts = []
    fixed_id = select_fixed_rule(dev_grid)
    parts.append(target_grid[target_grid["candidate_id"] == fixed_id].assign(comparator="optimized_fixed_rule"))
    for name in CONSTRUCTED:
        parts.append(target_grid[target_grid["candidate_id"] == name].assign(comparator=name))
    knn = contextual_knn(dev_grid, dev_ctx, target_ctx, k)
    sel = target_grid.merge(knn, on=["year", "candidate_id"])
    parts.append(sel.assign(comparator="contextual_knn"))
    grid_only = target_grid[~target_grid["candidate_id"].isin(CONSTRUCTED)]
    oracle = grid_only.sort_values("reward", ascending=False).drop_duplicates("year")
    parts.append(oracle.assign(comparator="retrospective_oracle"))
    return pd.concat(parts, ignore_index=True)


def paired_effects(comparators: pd.DataFrame, ppo: pd.DataFrame, out_dir: Path, prefix: str) -> pd.DataFrame:
    rows = []
    for comp_name, comp in comparators.groupby("comparator"):
        merged = ppo.merge(comp[["year", "yield_kg_ha", "irrigation_mm", "reward"]], on="year", suffixes=("_ppo", "_comparator"))
        merged["comparator"] = comp_name
        merged["d_reward"] = merged["reward_ppo"] - merged["reward_comparator"]
        merged["d_yield"] = merged["yield_kg_ha_ppo"] - merged["yield_kg_ha_comparator"]
        merged["d_irrigation"] = merged["irrigation_mm_ppo"] - merged["irrigation_mm_comparator"]
        rows.append(merged)
    effects = pd.concat(rows, ignore_index=True)
    effects.to_csv(out_dir / f"{prefix}_paired_effects_seed_year.csv", index=False, encoding="utf-8-sig")

    summary = []
    for comp_name, group in effects.groupby("comparator"):
        seed_means = group.groupby("seed")[["d_reward", "d_yield", "d_irrigation"]].mean()
        year_means = group.groupby("year")[["d_reward", "d_yield", "d_irrigation"]].mean()
        summary.append(
            {
                "comparator": comp_name,
                "n_seasons": group["year"].nunique(),
                "n_seeds": group["seed"].nunique(),
                "mean_d_reward": group["d_reward"].mean(),
                "seed_sd_d_reward": seed_means["d_reward"].std(ddof=1),
                "mean_d_yield_kg_ha": group["d_yield"].mean(),
                "seed_sd_d_yield_kg_ha": seed_means["d_yield"].std(ddof=1),
                "year_min_d_yield_kg_ha": year_means["d_yield"].min(),
                "year_max_d_yield_kg_ha": year_means["d_yield"].max(),
                "seasons_ppo_better_reward": int((year_means["d_reward"] > 0).sum()),
                "mean_d_irrigation_mm": group["d_irrigation"].mean(),
                "seed_sd_d_irrigation_mm": seed_means["d_irrigation"].std(ddof=1),
            }
        )
    out = pd.DataFrame(summary)
    out.to_csv(out_dir / f"{prefix}_paired_effects_summary.csv", index=False, encoding="utf-8-sig")
    return out


def comparator_summary(comparators: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    def cvar10(s: pd.Series) -> float:
        q = s.quantile(0.10)
        return float(s[s <= q].mean())

    out = (
        comparators.groupby("comparator")
        .agg(
            n=("year", "count"),
            reward_mean=("reward", "mean"),
            yield_mean_kg_ha=("yield_kg_ha", "mean"),
            yield_p10_kg_ha=("yield_kg_ha", lambda s: s.quantile(0.10)),
            yield_cvar10_kg_ha=("yield_kg_ha", cvar10),
            irrigation_mean_mm=("irrigation_mm", "mean"),
        )
        .reset_index()
        .sort_values("reward_mean", ascending=False)
    )
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed, contextual and oracle comparators on the same DSSAT candidate grid.")
    parser.add_argument("--config", default="configs/experiment_castro_t0.yaml")
    parser.add_argument("--run-name", default="final_comparators_castro_t0")
    parser.add_argument("--target", choices=["valid", "test"], default="valid")
    parser.add_argument("--ppo-evaluation", default=None, help="CSV with PPO rows for the target split (columns seed, year, reward, yield_kg_ha, irrigation_mm).")
    parser.add_argument("--knn", type=int, default=3)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["backend"] = "dssat"
    paths = make_paths(cfg, args.run_name)
    candidates = candidate_grid()
    pd.DataFrame(candidates).to_csv(paths.tables_dir / "candidate_grid.csv", index=False, encoding="utf-8-sig")

    train_years = cfg["data"]["train_years"]
    valid_years = cfg["data"]["valid_years"]
    dev_grid = evaluate_grid(cfg, paths.project_dir, train_years + valid_years, candidates, paths.tables_dir / "candidate_grid_development.csv", args.workers)
    if args.target == "valid":
        # Development check: rules are selected on the training seasons and applied to validation.
        selection_grid = dev_grid[dev_grid["year"].isin(train_years)]
        target_grid = dev_grid[dev_grid["year"].isin(valid_years)]
    else:
        selection_grid = dev_grid
        target_grid = evaluate_grid(cfg, paths.project_dir, cfg["data"]["test_years"], candidates, paths.tables_dir / "candidate_grid_test.csv", args.workers)

    comparators = build_comparators(cfg, paths.project_dir, selection_grid, target_grid, args.knn)
    comparators.to_csv(paths.tables_dir / f"comparator_rows_{args.target}.csv", index=False, encoding="utf-8-sig")
    summary = comparator_summary(comparators, paths.tables_dir / f"comparator_summary_{args.target}.csv")
    print(summary.to_string(index=False))

    if args.ppo_evaluation:
        ppo = pd.read_csv(args.ppo_evaluation)
        ppo = ppo[ppo["year"].isin(target_grid["year"].unique())]
        print(paired_effects(comparators, ppo, paths.tables_dir, args.target).to_string(index=False))

    budget = pd.DataFrame(
        [
            {"method": "candidate_grid_development", "candidates": len(candidates), "dssat_calls": int(len(dev_grid))},
            {"method": f"candidate_grid_{args.target}", "candidates": len(candidates), "dssat_calls": int(len(target_grid))},
        ]
    )
    budget.to_csv(paths.tables_dir / "computational_budget_dssat_calls.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
