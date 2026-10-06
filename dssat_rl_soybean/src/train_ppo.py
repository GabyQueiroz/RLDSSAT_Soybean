from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .callbacks import ValidationEarlyStopCallback
from .baselines import evaluate_baselines
from .config import load_config, make_paths
from .data import (
    build_year_weather_with_context,
    context_scaler,
    load_observed_soybean_yield,
    load_weather,
    save_scaler,
    summarize_weather_splits,
)
from .dssat_adapter import MockDSSATRunner, PyDSSATRunner
from .env import SoybeanDSSATEnv
from .report import build_report
from .evaluate import evaluate_policy


def make_runner(cfg, project_dir):
    if cfg["backend"] == "dssat":
        return PyDSSATRunner(cfg, project_dir)
    return MockDSSATRunner(cfg)


def make_train_env(cfg, project_dir, years, seed, scaler, obs_noise_std):
    """Each worker builds its own DSSAT runner so run folders and caches are not shared."""
    def _init():
        return Monitor(SoybeanDSSATEnv(cfg, years, make_runner(cfg, project_dir), seed, scaler, obs_noise_std))

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--run-name", default="ppo_soybean")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--backend", choices=["mock", "dssat"], default=None)
    parser.add_argument("--resume-model", default=None)
    parser.add_argument("--skip-test", action="store_true", help="Development runs: do not evaluate the frozen test block.")
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

    train_years = build_year_weather_with_context(daily, cfg["data"]["train_years"], cfg)
    valid_years = build_year_weather_with_context(daily, cfg["data"]["valid_years"], cfg)
    ppo_cfg = cfg["ppo"]
    scaler = None
    if ppo_cfg.get("standardize_context", False):
        scaler = context_scaler(train_years)
        save_scaler(paths.models_dir / "context_scaler.json", scaler)
    obs_noise_std = float(ppo_cfg.get("context_noise_std", 0.0))

    env_fns = [
        make_train_env(cfg, paths.project_dir, train_years, seed + i, scaler, obs_noise_std)
        for i in range(ppo_cfg["n_envs"])
    ]
    env = SubprocVecEnv(env_fns) if ppo_cfg.get("subproc_envs", False) else DummyVecEnv(env_fns)
    eval_env = SoybeanDSSATEnv(cfg, valid_years, make_runner(cfg, paths.project_dir), seed + 10_000, scaler)

    net_arch = ppo_cfg.get("net_arch", [128, 128, 64])
    policy_kwargs = dict(
        net_arch=dict(pi=list(net_arch), vf=list(net_arch)),
        activation_fn=torch.nn.Tanh,
        log_std_init=float(ppo_cfg.get("log_std_init", 0.0)),
    )
    if args.resume_model:
        model = PPO.load(args.resume_model, env=env, seed=seed, tensorboard_log=str(paths.output_dir / "tensorboard"))
    else:
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
    cb.min_timesteps = int(ppo_cfg.get("min_timesteps", 0))
    model.learn(
        total_timesteps=cfg["ppo"]["total_timesteps"],
        callback=cb,
        progress_bar=False,
        reset_num_timesteps=args.resume_model is None,
    )
    model.save(paths.models_dir / "final_model")
    env.close()
    splits = ["valid"] if args.skip_test else ["valid", "test"]
    for split in splits:
        evaluate_policy(cfg, paths, split)
        evaluate_baselines(cfg, paths, split)

    metadata = {"config": cfg, "best_validation_reward": cb.best_reward, "output_dir": str(paths.output_dir)}
    (paths.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    build_report(cfg, paths)


if __name__ == "__main__":
    main()
