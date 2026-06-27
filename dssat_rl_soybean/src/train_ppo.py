from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .callbacks import ValidationEarlyStopCallback
from .baselines import evaluate_baselines
from .config import load_config, make_paths
from .data import build_year_weather, load_observed_soybean_yield, load_weather, summarize_weather_splits
from .dssat_adapter import MockDSSATRunner, PyDSSATRunner
from .env import SoybeanDSSATEnv
from .report import build_report
from .evaluate import evaluate_policy


def make_runner(cfg, project_dir):
    if cfg["backend"] == "dssat":
        return PyDSSATRunner(cfg, project_dir)
    return MockDSSATRunner(cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--run-name", default="ppo_soybean")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--backend", choices=["mock", "dssat"], default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.backend:
        cfg["backend"] = args.backend
    if args.timesteps:
        cfg["ppo"]["total_timesteps"] = args.timesteps
    paths = make_paths(cfg, args.run_name)
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    daily = load_weather(paths.project_dir, cfg)
    observed = load_observed_soybean_yield(paths.project_dir, cfg)
    summarize_weather_splits(daily, cfg).to_csv(paths.tables_dir / "weather_splits.csv", index=False, encoding="utf-8-sig")
    observed.to_csv(paths.tables_dir / "observed_soybean_yield_castro.csv", index=False, encoding="utf-8-sig")

    train_years = build_year_weather(daily, cfg["data"]["train_years"])
    valid_years = build_year_weather(daily, cfg["data"]["valid_years"])
    runner = make_runner(cfg, paths.project_dir)

    def make_env(i: int):
        return Monitor(SoybeanDSSATEnv(cfg, train_years, runner, seed + i))

    env = DummyVecEnv([lambda i=i: make_env(i) for i in range(cfg["ppo"]["n_envs"])])
    eval_env = SoybeanDSSATEnv(cfg, valid_years, runner, seed + 10_000)

    policy_kwargs = dict(net_arch=dict(pi=[128, 128, 64], vf=[128, 128, 64]), activation_fn=torch.nn.Tanh)
    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=1,
        n_steps=cfg["ppo"]["n_steps"],
        batch_size=cfg["ppo"]["batch_size"],
        n_epochs=cfg["ppo"]["n_epochs"],
        gamma=cfg["ppo"]["gamma"],
        gae_lambda=cfg["ppo"]["gae_lambda"],
        learning_rate=cfg["ppo"]["learning_rate"],
        clip_range=cfg["ppo"]["clip_range"],
        ent_coef=cfg["ppo"]["ent_coef"],
        vf_coef=cfg["ppo"]["vf_coef"],
        max_grad_norm=cfg["ppo"]["max_grad_norm"],
        tensorboard_log=str(paths.output_dir / "tensorboard"),
        policy_kwargs=policy_kwargs,
    )
    cb = ValidationEarlyStopCallback(
        eval_env,
        paths,
        eval_freq=cfg["ppo"]["eval_freq"],
        patience_evals=cfg["ppo"]["patience_evals"],
    )
    model.learn(total_timesteps=cfg["ppo"]["total_timesteps"], callback=cb, progress_bar=True)
    model.save(paths.models_dir / "final_model")
    evaluate_policy(cfg, paths, "valid")
    evaluate_policy(cfg, paths, "test")
    evaluate_baselines(cfg, paths, "valid")
    evaluate_baselines(cfg, paths, "test")

    metadata = {"config": cfg, "best_validation_reward": cb.best_reward, "output_dir": str(paths.output_dir)}
    (paths.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    build_report(cfg, paths)


if __name__ == "__main__":
    main()
