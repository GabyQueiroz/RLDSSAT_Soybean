from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ValidationEarlyStopCallback(BaseCallback):
    def __init__(self, eval_env, paths, eval_freq: int, patience_evals: int, n_eval_episodes: int = 64):
        super().__init__()
        self.eval_env = eval_env
        self.paths = paths
        self.eval_freq = eval_freq
        self.patience_evals = patience_evals
        self.n_eval_episodes = n_eval_episodes
        self.best_reward = -np.inf
        self.bad_evals = 0
        self.history_path = paths.tables_dir / "validation_curve.csv"

    def _on_training_start(self) -> None:
        with self.history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timesteps", "mean_reward", "std_reward", "mean_yield_kg_ha", "mean_irrigation_mm"])
            writer.writeheader()

    def _evaluate(self) -> dict:
        rewards, yields, irrig = [], [], []
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            action, _ = self.model.predict(obs, deterministic=True)
            _, reward, _, _, info = self.eval_env.step(action)
            rewards.append(reward)
            yields.append(info.get("yield_kg_ha", np.nan))
            irrig.append(info.get("irrigation_mm", np.nan))
        return {
            "timesteps": self.num_timesteps,
            "mean_reward": float(np.nanmean(rewards)),
            "std_reward": float(np.nanstd(rewards)),
            "mean_yield_kg_ha": float(np.nanmean(yields)),
            "mean_irrigation_mm": float(np.nanmean(irrig)),
        }

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True
        row = self._evaluate()
        with self.history_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)
        if row["mean_reward"] > self.best_reward + 1e-4:
            self.best_reward = row["mean_reward"]
            self.bad_evals = 0
            self.model.save(self.paths.models_dir / "best_model")
        else:
            self.bad_evals += 1
        return self.bad_evals < self.patience_evals
