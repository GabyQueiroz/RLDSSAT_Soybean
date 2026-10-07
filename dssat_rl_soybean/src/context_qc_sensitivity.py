from __future__ import annotations

import argparse
import shutil

import pandas as pd

from .config import load_config, make_paths
from .evaluate import evaluate_policy
from .final_comparators import build_comparators, comparator_summary, paired_effects


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post hoc check: trained policies and the kNN rule evaluated on test contexts in which station windows "
        "with less than a minimum share of valid days are replaced by the training mean."
    )
    parser.add_argument("--config", default="configs/experiment_castro_t0.yaml")
    parser.add_argument("--run-prefix", default="ppo_castro_t0")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--min-valid-day-share", type=float, default=0.75)
    parser.add_argument("--comparators-run", default="final_comparators_castro_t0")
    args = parser.parse_args()

    tag = "_context_qc"
    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        base = load_config(f"configs/_generated_{args.run_prefix}/experiment_seed_{seed}.yaml")
        base["decision"]["context_min_valid_day_share"] = args.min_valid_day_share
        paths = make_paths(base, f"{args.run_prefix}_seed_{seed}")
        rows.append(evaluate_policy(base, paths, "test", tag=tag).assign(seed=seed))
    ppo = pd.concat(rows, ignore_index=True)

    cfg = load_config(args.config)
    cfg["backend"] = "dssat"
    cfg["decision"]["context_min_valid_day_share"] = args.min_valid_day_share
    paths = make_paths(cfg, f"{args.comparators_run}{tag}")
    source = paths.project_dir / "outputs" / args.comparators_run / "tables"
    for name in ["candidate_grid_development.csv", "candidate_grid_test.csv"]:
        shutil.copy(source / name, paths.tables_dir / name)
    dev_grid = pd.read_csv(paths.tables_dir / "candidate_grid_development.csv")
    test_grid = pd.read_csv(paths.tables_dir / "candidate_grid_test.csv")
    comparators = build_comparators(cfg, paths.project_dir, dev_grid, test_grid, k=3)
    comparators.to_csv(paths.tables_dir / "comparator_rows_test.csv", index=False)
    ppo.to_csv(paths.tables_dir / "ppo_policy_evaluation_test.csv", index=False)
    summary = comparator_summary(pd.concat([comparators, ppo.assign(comparator="PPO")], ignore_index=True), paths.tables_dir / "comparator_summary_test.csv")
    print(summary.to_string(index=False))
    print(paired_effects(comparators, ppo, paths.tables_dir, "test").to_string(index=False))
    print(ppo[["seed", "year", "planting_date", "trigger_dryness", "amount_mm", "irrigation_mm", "yield_kg_ha", "reward"]].to_string(index=False))


if __name__ == "__main__":
    main()
