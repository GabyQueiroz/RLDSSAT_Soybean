from __future__ import annotations

import argparse
import json

import numpy as np
from stable_baselines3 import PPO

from .config import load_config, make_paths
from .data import build_year_weather, load_weather
from .dssat_adapter import MockDSSATRunner, PyDSSATRunner, build_irrigation_schedule
from .env import SoybeanDSSATEnv


def main():
    parser = argparse.ArgumentParser(description="Gera recomendacao operacional de plantio/irrigacao para um ano.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--run-name", default="ppo_soybean")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--backend", choices=["mock", "dssat"], default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.backend:
        cfg["backend"] = args.backend
    paths = make_paths(cfg, args.run_name)
    daily = load_weather(paths.project_dir, cfg)
    years = build_year_weather(daily, [args.year])
    runner = PyDSSATRunner(cfg, paths.project_dir) if cfg["backend"] == "dssat" else MockDSSATRunner(cfg)
    env = SoybeanDSSATEnv(cfg, years, runner, cfg["seed"] + 40_000)
    env.current = years[0]
    model_path = paths.models_dir / "best_model"
    if not (model_path.with_suffix(".zip")).exists():
        model_path = paths.models_dir / "final_model"
    model = PPO.load(model_path)
    action, _ = model.predict(env._obs(years[0]), deterministic=True)
    decoded = env._decode_action(np.asarray(action, dtype=np.float32))
    _, reward, _, _, info = env.step(action)
    recommendation = {
        "year": args.year,
        "backend": cfg["backend"],
        "recommendation": {
            "planting_date": info.get("planting_date"),
            "irrigation_trigger_dryness_proxy_mm": decoded["trigger_dryness"],
            "irrigation_event_mm": decoded["amount_mm"],
            "season_irrigation_cap_mm": decoded["max_irrigation_mm"],
        },
        "expected_simulated_outcome": {
            "yield_kg_ha": info.get("yield_kg_ha"),
            "irrigation_mm": info.get("irrigation_mm"),
            "rain_mm": info.get("rain_mm"),
            "reward": reward,
            "n_irrigation_events": info.get("n_irrigation_events"),
        },
        "note": "Use em campo exige previsao meteorologica atualizada, solo calibrado e cultivar DSSAT validada localmente.",
    }
    out = paths.tables_dir / f"field_recommendation_{args.year}.json"
    out.write_text(json.dumps(recommendation, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(recommendation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
