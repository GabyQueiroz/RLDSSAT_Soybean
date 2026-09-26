from __future__ import annotations

import argparse
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(str(cfg_path))
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    generated_dir = cfg_path.parent / "_generated_multiseed"
    generated_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        run_cfg = dict(cfg)
        run_cfg["seed"] = seed
        seed_cfg_path = generated_dir / f"experiment_seed_{seed}.yaml"
        seed_cfg_path.write_text(yaml.safe_dump(run_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        cmd = [
            sys.executable,
            "-m",
            "dssat_rl_soybean.src.train_ppo",
            "--config",
            str(seed_cfg_path),
            "--run-name",
            f"{args.run_prefix}_seed_{seed}",
        ]
        if args.timesteps is not None:
            cmd.extend(["--timesteps", str(args.timesteps)])
        if args.backend is not None:
            cmd.extend(["--backend", args.backend])
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
