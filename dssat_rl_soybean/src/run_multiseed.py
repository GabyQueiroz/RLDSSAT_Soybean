from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import yaml

from .config import load_config


def main():
    parser = argparse.ArgumentParser(description="Run PPO training for multiple seeds with isolated output folders.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--run-prefix", default="ppo_multiseed")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--backend", choices=["mock", "dssat"], default=None)
    parser.add_argument("--use-frozen-split", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(str(cfg_path))
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    generated_dir = cfg_path.parent / "_generated_multiseed"
    generated_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for seed in seeds:
        run_cfg = dict(cfg)
        run_cfg.pop("_config_path", None)
        run_cfg.pop("_project_dir", None)
        run_cfg["data"] = dict(cfg["data"])
        run_cfg["ppo"] = dict(cfg["ppo"])
        run_cfg["seed"] = seed
        if args.use_frozen_split:
            design = cfg["evaluation_design"]
            run_cfg["data"]["train_years"] = design["development_train_years"]
            run_cfg["data"]["valid_years"] = design["development_validation_years"]
            run_cfg["data"]["test_years"] = design["frozen_evaluation_years"]
            run_cfg["evaluation_design"] = dict(design)
            run_cfg["evaluation_design"]["applied_to_generated_config"] = True
        if args.timesteps is not None:
            run_cfg["ppo"]["total_timesteps"] = args.timesteps
        if args.backend is not None:
            run_cfg["backend"] = args.backend
        seed_cfg_path = generated_dir / f"experiment_seed_{seed}.yaml"
        seed_cfg_path.write_text(yaml.safe_dump(run_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        run_name = f"{args.run_prefix}_seed_{seed}"
        cmd = [
            sys.executable,
            "-m",
            "dssat_rl_soybean.src.train_ppo",
            "--config",
            str(seed_cfg_path),
            "--run-name",
            run_name,
        ]
        if args.timesteps is not None:
            cmd.extend(["--timesteps", str(args.timesteps)])
        if args.backend is not None:
            cmd.extend(["--backend", args.backend])
        portable_cmd = [
            "python",
            "-m",
            "dssat_rl_soybean.src.train_ppo",
            "--config",
            str(seed_cfg_path),
            "--run-name",
            run_name,
        ]
        if args.timesteps is not None:
            portable_cmd.extend(["--timesteps", str(args.timesteps)])
        if args.backend is not None:
            portable_cmd.extend(["--backend", args.backend])
        manifest_rows.append(
            {
                "seed": seed,
                "run_name": run_name,
                "config_path": str(seed_cfg_path),
                "backend": run_cfg["backend"],
                "timesteps": run_cfg["ppo"]["total_timesteps"],
                "train_years": ",".join(map(str, run_cfg["data"]["train_years"])),
                "valid_years": ",".join(map(str, run_cfg["data"]["valid_years"])),
                "test_years": ",".join(map(str, run_cfg["data"]["test_years"])),
                "uses_frozen_split": bool(args.use_frozen_split),
                "command": " ".join(portable_cmd),
            }
        )
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
    manifest_path = generated_dir / f"{args.run_prefix}_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
