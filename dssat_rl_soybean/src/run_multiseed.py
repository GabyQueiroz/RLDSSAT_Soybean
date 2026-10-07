from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import subprocess
import sys
import time
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
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--generated-dir", default=None, help="Folder (next to the config) for the per-seed configs.")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(str(cfg_path))
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    generated_dir = cfg_path.parent / (args.generated_dir or "_generated_multiseed")
    generated_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    log_path = Path(args.log_file) if args.log_file else generated_dir / f"{args.run_prefix}_run.log"

    def log(message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

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
        if args.skip_test:
            cmd.append("--skip-test")
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
        if args.skip_test:
            portable_cmd.append("--skip-test")
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
        log("Prepared command: " + " ".join(portable_cmd))
        manifest_path = generated_dir / f"{args.run_prefix}_manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0]))
            writer.writeheader()
            writer.writerows(manifest_rows)
    if args.dry_run:
        log(f"Manifest written to {manifest_path}")
        return

    log(f"Manifest written to {manifest_path}")
    log(f"Starting multi-seed run with {len(manifest_rows)} seeds. Log file: {log_path}")
    completed_durations: list[float] = []
    overall_start = time.monotonic()
    for index, row in enumerate(manifest_rows, start=1):
        seed = row["seed"]
        cmd = row["command"].replace("python -m", f"{sys.executable} -u -m", 1).split()
        if completed_durations:
            avg_seconds = sum(completed_durations) / len(completed_durations)
            remaining_seconds = avg_seconds * (len(manifest_rows) - index + 1)
            eta = datetime.now() + timedelta(seconds=remaining_seconds)
            log(
                f"Starting seed {seed} ({index}/{len(manifest_rows)}). "
                f"Mean completed seed time: {timedelta(seconds=int(avg_seconds))}. "
                f"Estimated remaining: {timedelta(seconds=int(remaining_seconds))}; ETA {eta:%Y-%m-%d %H:%M:%S}."
            )
        else:
            log(f"Starting seed {seed} ({index}/{len(manifest_rows)}). ETA will be estimated after the first seed finishes.")
        seed_start = time.monotonic()
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(line)
            return_code = proc.wait()
        seed_seconds = time.monotonic() - seed_start
        if return_code != 0:
            log(f"Seed {seed} failed after {timedelta(seconds=int(seed_seconds))} with return code {return_code}.")
            raise subprocess.CalledProcessError(return_code, cmd)
        completed_durations.append(seed_seconds)
        elapsed = time.monotonic() - overall_start
        log(
            f"Finished seed {seed} in {timedelta(seconds=int(seed_seconds))}. "
            f"Completed {index}/{len(manifest_rows)}; elapsed {timedelta(seconds=int(elapsed))}."
        )
    total_elapsed = time.monotonic() - overall_start
    log(f"Finished all seeds in {timedelta(seconds=int(total_elapsed))}.")


if __name__ == "__main__":
    main()
