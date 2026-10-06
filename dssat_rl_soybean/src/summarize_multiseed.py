from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect per-seed PPO evaluations into one table.")
    parser.add_argument("--project-dir", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--splits", default="valid,test")
    args = parser.parse_args()

    outputs = Path(args.project_dir) / "outputs"
    out_dir = outputs / f"{args.run_prefix}_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    splits = [s for s in args.splits.split(",") if s.strip()]

    evaluations, checkpoints = [], []
    for seed in seeds:
        tables = outputs / f"{args.run_prefix}_seed_{seed}" / "tables"
        for split in splits:
            path = tables / f"policy_evaluation_{split}.csv"
            if path.exists():
                evaluations.append(pd.read_csv(path).assign(seed=seed))
        curve_path = tables / "validation_curve.csv"
        if curve_path.exists():
            curve = pd.read_csv(curve_path)
            best = curve.loc[curve["mean_reward"].idxmax()].to_dict()
            checkpoints.append({"seed": seed, **best, "n_evaluations": len(curve), "last_timesteps": int(curve["timesteps"].max())})

    ev = pd.concat(evaluations, ignore_index=True)
    ev.to_csv(out_dir / "policy_evaluation_all_seeds.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(checkpoints).to_csv(out_dir / "selected_checkpoints_by_seed.csv", index=False, encoding="utf-8-sig")

    by_seed = (
        ev.groupby(["split", "seed"])
        .agg(
            reward_mean=("reward", "mean"),
            yield_mean_kg_ha=("yield_kg_ha", "mean"),
            irrigation_mean_mm=("irrigation_mm", "mean"),
            failed_runs=("failed", "sum"),
            n=("year", "count"),
        )
        .reset_index()
    )
    by_seed.to_csv(out_dir / "policy_summary_by_seed.csv", index=False, encoding="utf-8-sig")
    actions = ev[["split", "seed", "year", "planting_date", "trigger_dryness", "amount_mm", "max_irrigation_mm", "n_irrigation_events", "irrigation_mm", "yield_kg_ha", "reward"]]
    actions.to_csv(out_dir / "actions_by_seed_year.csv", index=False, encoding="utf-8-sig")
    print(by_seed.to_string(index=False))


if __name__ == "__main__":
    main()
