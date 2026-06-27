from __future__ import annotations

import argparse
from datetime import date
import json

from .config import load_config, make_paths
from .data import build_year_weather, load_weather
from .dssat_adapter import PyDSSATRunner, build_irrigation_schedule


def main():
    parser = argparse.ArgumentParser(description="Roda uma safra DSSAT real para validar configuracao.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--run-name", default="dssat_real_check")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--planting-date", default=None, help="YYYY-MM-DD. Padrao: 15 de outubro do ano.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["backend"] = "dssat"
    paths = make_paths(cfg, args.run_name)
    daily = load_weather(paths.project_dir, cfg)
    yw = build_year_weather(daily, [args.year])[0]
    runner = PyDSSATRunner(cfg, paths.project_dir)
    pdate = date.fromisoformat(args.planting_date) if args.planting_date else date(args.year, 10, 15)
    irrigation = build_irrigation_schedule(
        yw.daily,
        pdate,
        cfg["agronomy"]["season_length_days"],
        trigger_dryness=45,
        amount_mm=20,
        max_season_irrigation_mm=120,
        check_days=cfg["agronomy"]["irrigation_check_days"],
    )
    sim = runner.run(yw.daily, pdate, irrigation, rng=None)
    out = {
        "status": "ok",
        "backend": "dssat",
        "year": args.year,
        "planting_date": sim.planting_date.isoformat(),
        "yield_kg_ha": sim.yield_kg_ha,
        "irrigation_mm": sim.irrigation_mm,
        "rain_mm": sim.rain_mm,
        "raw_summary_keys": list(sim.raw.keys())[:30] if isinstance(sim.raw, dict) else [],
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
