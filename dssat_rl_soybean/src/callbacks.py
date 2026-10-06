from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ValidationEarlyStopCallback(BaseCallback):
    def __init__(self, eval_env, paths, eval_freq: int, patience_evals: int, n_eval_episodes: int | None = None):
        super().__init__()
        self.eval_env = eval_env
        self.paths = paths
        self.eval_freq = eval_freq
        self.patience_evals = patience_evals
        self.n_eval_episodes = n_eval_episodes
        self.best_reward = -np.inf
        self.bad_evals = 0
        self._last_eval = 0
        self.min_timesteps = 0
        self.history_path = paths.tables_dir / "validation_curve.csv"

    def _on_training_start(self) -> None:
        if self.history_path.exists() and self.history_path.stat().st_size > 0:
            try:
                import pandas as pd

                history = pd.read_csv(self.history_path)
                if "mean_reward" in history and not history.empty:
                    self.best_reward = float(history["mean_reward"].max())
            except Exception:
                pass
            return
        with self.history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timesteps", "mean_reward", "std_reward", "mean_yield_kg_ha", "mean_irrigation_mm"])
            writer.writeheader()

    def _evaluate(self) -> dict:
        # DSSAT is deterministic, so each validation season is evaluated exactly once.
        rewards, yields, irrig = [], [], []
        if self.n_eval_episodes is None:
            episodes = list(self.eval_env.years)
        else:
            episodes = [None] * self.n_eval_episodes
        for yw in episodes:
            if yw is None:
                obs, _ = self.eval_env.reset()
            else:
                self.eval_env.current = yw
                obs = self.eval_env._obs(yw)
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
        if self.num_timesteps - self._last_eval < self.eval_freq:
            return True
        self._last_eval = self.num_timesteps
        row = self._evaluate()
        with self.history_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)
        self.model.save(self.paths.models_dir / "latest_model")
        if row["mean_reward"] > self.best_reward + 1e-4:
            self.best_reward = row["mean_reward"]
            self.bad_evals = 0
            self.model.save(self.paths.models_dir / "best_model")
        else:
            self.bad_evals += 1
        return self.num_timesteps < self.min_timesteps or self.bad_evals < self.patience_evals
