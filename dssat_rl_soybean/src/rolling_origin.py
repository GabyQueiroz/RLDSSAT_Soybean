from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import load_config, make_paths
from .data import build_year_weather_with_context, context_scaler, load_weather
from .final_comparators import CONSTRUCTED, candidate_grid, contextual_knn, evaluate_grid, select_fixed_rule


def folds(cfg: dict) -> list[dict]:
    design = cfg["evaluation_design"]
    first, last = design["seasons"]
    test_first, test_last = design["test_seasons"]
    n_test, n_valid = int(design["fold_test_length"]), int(design["validation_length"])
    out = []
    for start in range(test_first, test_last + 1, n_test):
        test = list(range(start, min(start + n_test, test_last + 1)))
        valid = list(range(start - n_valid, start))
        train = list(range(first, start - n_valid))
        out.append({"fold": f"f{start}", "train": train, "valid": valid, "test": test})
    return out


def fold_config(cfg: dict, fold: dict, seed: int) -> dict:
    run = copy.deepcopy(cfg)
    for key in ["_config_path", "_project_dir"]:
        run.pop(key, None)
    run["seed"] = seed
    run["data"]["train_years"] = fold["train"]
    run["data"]["valid_years"] = fold["valid"]
    run["data"]["test_years"] = fold["test"]
    return run


def run_training(cfg: dict, prefix: str, seeds: list[int], log_path: Path) -> None:
    project_dir = Path(cfg["_project_dir"])
    gen_dir = project_dir / "configs" / f"_generated_{prefix}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(f, s) for f in folds(cfg) for s in seeds]
    durations: list[float] = []

    def log(msg: str) -> None:
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    for i, (fold, seed) in enumerate(jobs, 1):
        run_name = f"{prefix}/{fold['fold']}_seed_{seed}"
        done = project_dir / "outputs" / run_name / "tables" / "policy_evaluation_test.csv"
        if done.exists():
            log(f"Skipping {run_name} (already evaluated).")
            continue
        cfg_path = gen_dir / f"{fold['fold']}_seed_{seed}.yaml"
        cfg_path.write_text(yaml.safe_dump(fold_config(cfg, fold, seed), sort_keys=False, allow_unicode=True), encoding="utf-8")
        eta = ""
        if durations:
            remaining = np.mean(durations) * (len(jobs) - i + 1)
            eta = f" ETA {datetime.fromtimestamp(time.time() + remaining):%Y-%m-%d %H:%M}."
        log(f"Starting {run_name} ({i}/{len(jobs)}): train {fold['train'][0]}-{fold['train'][-1]}, "
            f"valid {fold['valid'][0]}-{fold['valid'][-1]}, test {fold['test']}.{eta}")
        t0 = time.monotonic()
        cmd = [sys.executable, "-u", "-m", "src.train_ppo", "--config", str(cfg_path), "--run-name", run_name]
        with subprocess.Popen(cmd, cwd=project_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            run_log = project_dir / "outputs" / prefix / f"{fold['fold']}_seed_{seed}.log"
            run_log.parent.mkdir(parents=True, exist_ok=True)
            with run_log.open("w", encoding="utf-8") as fh:
                for line in proc.stdout:
                    if "created." in line or "has been removed" in line:
                        continue
                    fh.write(line)
            code = proc.wait()
        if code != 0:
            log(f"{run_name} failed with return code {code}; see {run_log}.")
            raise SystemExit(code)
        durations.append(time.monotonic() - t0)
        log(f"Finished {run_name} in {int(durations[-1] // 60)} min.")
    log("ROLLING TRAINING FINISHED")


def analyse(cfg: dict, prefix: str, seeds: list[int], workers: int) -> None:
    project_dir = Path(cfg["_project_dir"])
    paths = make_paths(cfg, f"{prefix}_analysis")
    first, last = cfg["evaluation_design"]["seasons"]
    all_years = list(range(first, last + 1))
    grid = evaluate_grid(cfg, project_dir, all_years, candidate_grid(), paths.tables_dir / "candidate_grid_all_seasons.csv", workers)
    daily = load_weather(project_dir, cfg)

    comp_rows, ppo_rows, ckpt_rows = [], [], []
    for fold in folds(cfg):
        pool = grid[grid["year"].isin(fold["train"] + fold["valid"])]
        target = grid[grid["year"].isin(fold["test"])]
        fixed_id = select_fixed_rule(pool)
        comp_rows.append(target[target["candidate_id"] == fixed_id].assign(comparator="optimized_fixed_rule"))
        for name in CONSTRUCTED:
            comp_rows.append(target[target["candidate_id"] == name].assign(comparator=name))
        train_seasons = build_year_weather_with_context(daily, fold["train"], cfg)
        mean, std = context_scaler(train_seasons)

        def ctx(years):
            seasons = build_year_weather_with_context(daily, years, cfg)
            data = np.nan_to_num(np.vstack([(s.features - mean) / std for s in seasons]), nan=0.0)
            out = pd.DataFrame(data, columns=[f"c{i}" for i in range(data.shape[1])])
            out.insert(0, "year", [s.year for s in seasons])
            return out

        knn = contextual_knn(pool, ctx(fold["train"] + fold["valid"]), ctx(fold["test"]), k=3)
        comp_rows.append(target.merge(knn, on=["year", "candidate_id"]).assign(comparator="contextual_knn"))
        oracle = target[~target["candidate_id"].isin(CONSTRUCTED)].sort_values("reward", ascending=False).drop_duplicates("year")
        comp_rows.append(oracle.assign(comparator="retrospective_oracle"))
        for seed in seeds:
            tables = project_dir / "outputs" / prefix / f"{fold['fold']}_seed_{seed}" / "tables"
            ev = tables / "policy_evaluation_test.csv"
            if ev.exists():
                ppo_rows.append(pd.read_csv(ev).assign(seed=seed, fold=fold["fold"]))
                curve = pd.read_csv(tables / "validation_curve.csv").drop_duplicates("timesteps", keep="last")
                best = curve.loc[curve["mean_reward"].idxmax()]
                ckpt_rows.append({"fold": fold["fold"], "seed": seed, "n_train": len(fold["train"]),
                                  "selected_timesteps": int(best["timesteps"]), "valid_reward": best["mean_reward"],
                                  "valid_irrigation_mm": best["mean_irrigation_mm"], "stop_timesteps": int(curve["timesteps"].max())})

    comps = pd.concat(comp_rows, ignore_index=True)
    comps.to_csv(paths.tables_dir / "comparator_rows_test.csv", index=False)
    ppo = pd.concat(ppo_rows, ignore_index=True) if ppo_rows else pd.DataFrame()
    ppo.to_csv(paths.tables_dir / "ppo_rows_test.csv", index=False)
    pd.DataFrame(ckpt_rows).to_csv(paths.tables_dir / "selected_checkpoints.csv", index=False)
    if ppo.empty:
        print("No PPO evaluations found yet.")
        return

    years = sorted(ppo["year"].unique())
    comps = comps[comps["year"].isin(years)]
    rows = pd.concat([comps[["comparator", "year", "candidate_id", "planting_date", "yield_kg_ha", "irrigation_mm", "reward"]],
                      ppo.assign(comparator="PPO")[["comparator", "year", "seed", "planting_date", "yield_kg_ha", "irrigation_mm", "reward"]]],
                     ignore_index=True)

    def cvar10(s):
        q = s.quantile(0.10)
        return float(s[s <= q].mean())

    summary = rows.groupby("comparator").agg(
        reward_mean=("reward", "mean"), yield_mean_kg_ha=("yield_kg_ha", "mean"),
        yield_p10_kg_ha=("yield_kg_ha", lambda s: s.quantile(0.10)), yield_cvar10_kg_ha=("yield_kg_ha", cvar10),
        irrigation_mean_mm=("irrigation_mm", "mean"), n=("year", "count"),
    ).reset_index().sort_values("reward_mean", ascending=False)
    summary.to_csv(paths.tables_dir / "test_summary.csv", index=False)

    paired = []
    for comp, c in comps.groupby("comparator"):
        m = ppo.merge(c[["year", "yield_kg_ha", "irrigation_mm", "reward"]], on="year", suffixes=("_ppo", "_comp"))
        m["comparator"] = comp
        m["d_reward"] = m["reward_ppo"] - m["reward_comp"]
        m["d_yield"] = m["yield_kg_ha_ppo"] - m["yield_kg_ha_comp"]
        m["d_irrigation"] = m["irrigation_mm_ppo"] - m["irrigation_mm_comp"]
        paired.append(m)
    paired = pd.concat(paired, ignore_index=True)
    paired.to_csv(paths.tables_dir / "paired_effects_seed_season.csv", index=False)
    rng = np.random.default_rng(2026)
    eff = []
    for comp, g in paired.groupby("comparator"):
        season = g.groupby("year")[["d_reward", "d_yield", "d_irrigation"]].mean()
        boot = [season.sample(len(season), replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(5000)]
        boot = pd.DataFrame(boot)
        eff.append({
            "comparator": comp, "n_seasons": len(season), "n_seeds": g["seed"].nunique(),
            "mean_d_reward": season["d_reward"].mean(), "ci95_d_reward_low": boot["d_reward"].quantile(0.025), "ci95_d_reward_high": boot["d_reward"].quantile(0.975),
            "mean_d_yield": season["d_yield"].mean(), "ci95_d_yield_low": boot["d_yield"].quantile(0.025), "ci95_d_yield_high": boot["d_yield"].quantile(0.975),
            "mean_d_irrigation": season["d_irrigation"].mean(),
            "seasons_ppo_better_reward": int((season["d_reward"] > 0).sum()),
            "sd_seed_d_yield": g.groupby("seed")["d_yield"].mean().std(ddof=1),
        })
    eff = pd.DataFrame(eff)
    eff.to_csv(paths.tables_dir / "paired_effects_summary.csv", index=False)
    rows.groupby(["comparator", "year"])[["yield_kg_ha", "irrigation_mm", "reward"]].mean().reset_index().to_csv(
        paths.tables_dir / "season_level.csv", index=False)
    print(summary.round(3).to_string(index=False))
    print(eff.round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling-origin evaluation of PPO and comparators on the NASA POWER series.")
    parser.add_argument("command", choices=["folds", "grid", "train", "analyse"])
    parser.add_argument("--config", default="configs/experiment_castro_power.yaml")
    parser.add_argument("--prefix", default="rolling_power")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["backend"] = "dssat"
    seeds = [int(s) for s in cfg["evaluation_design"]["seeds"]]
    if args.command == "folds":
        table = pd.DataFrame([{**f, "train": f"{f['train'][0]}-{f['train'][-1]} ({len(f['train'])})",
                               "valid": f"{f['valid'][0]}-{f['valid'][-1]}", "test": ",".join(map(str, f["test"]))} for f in folds(cfg)])
        print(table.to_string(index=False))
    elif args.command == "grid":
        paths = make_paths(cfg, f"{args.prefix}_analysis")
        first, last = cfg["evaluation_design"]["seasons"]
        evaluate_grid(cfg, paths.project_dir, list(range(first, last + 1)), candidate_grid(),
                      paths.tables_dir / "candidate_grid_all_seasons.csv", args.workers)
    elif args.command == "train":
        log_path = Path(cfg["_project_dir"]) / "outputs" / f"{args.prefix}_run.log"
        run_training(cfg, args.prefix, seeds, log_path)
    else:
        analyse(cfg, args.prefix, seeds, args.workers)


if __name__ == "__main__":
    main()
