from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .config import load_config, make_paths
from .data import build_year_weather, load_observed_soybean_yield, load_weather


def _years(cfg: dict, key: str) -> list[int]:
    return [int(y) for y in cfg["evaluation_design"][key]]


def _block_rows(cfg: dict) -> pd.DataFrame:
    design = cfg["evaluation_design"]
    blocks = [
        ("development_train", _years(cfg, "development_train_years"), "model fitting and policy learning"),
        ("development_validation", _years(cfg, "development_validation_years"), "checkpoint and development choices"),
        ("frozen_evaluation", _years(cfg, "frozen_evaluation_years"), "opened once after final choices are frozen"),
        ("illustration", _years(cfg, "illustration_years"), "unobserved scenario; excluded from empirical metrics"),
    ]
    rows = []
    for block, years, role in blocks:
        rows.append(
            {
                "registry_name": design["registry_name"],
                "registry_version": design["registry_version"],
                "registry_date": design["registry_date"],
                "environment": design["environment"],
                "block": block,
                "years": ",".join(map(str, years)),
                "n_years": len(years),
                "role": role,
            }
        )
    return pd.DataFrame(rows)


def _validate_design(cfg: dict) -> list[str]:
    design = cfg["evaluation_design"]
    train = set(_years(cfg, "development_train_years"))
    valid = set(_years(cfg, "development_validation_years"))
    frozen = set(_years(cfg, "frozen_evaluation_years"))
    illustration = set(_years(cfg, "illustration_years"))
    messages: list[str] = []
    overlaps = {
        "train_validation": sorted(train & valid),
        "train_frozen": sorted(train & frozen),
        "validation_frozen": sorted(valid & frozen),
        "development_illustration": sorted((train | valid | frozen) & illustration),
    }
    for name, years in overlaps.items():
        if years:
            messages.append(f"overlap_{name}: {years}")
    if design.get("exclude_frozen_years_from_development", True) and (train & frozen or valid & frozen):
        messages.append("frozen years overlap with development years")
    if not messages:
        messages.append("ok")
    return messages


def _weather_summary(project_dir: Path, cfg: dict) -> pd.DataFrame:
    daily = load_weather(project_dir, cfg)
    block_map = {
        "development_train": _years(cfg, "development_train_years"),
        "development_validation": _years(cfg, "development_validation_years"),
        "frozen_evaluation": _years(cfg, "frozen_evaluation_years"),
        "illustration": _years(cfg, "illustration_years"),
    }
    rows = []
    for block, years in block_map.items():
        seasons = build_year_weather(daily, years)
        d = pd.concat([s.daily.assign(season_year=s.year) for s in seasons], ignore_index=True)
        annual = d.groupby("season_year", as_index=False).agg(
            seasonal_rain_mm=("rain", "sum"),
            seasonal_temp_mean_c=("temp_mean", "mean"),
            seasonal_srad_mean_mj_m2_day=("srad", "mean"),
            n_days=("date", "count"),
        )
        rows.append(
            {
                "block": block,
                "years": ",".join(map(str, years)),
                "n_years": len(years),
                "rain_mean_mm": float(annual["seasonal_rain_mm"].mean()),
                "rain_min_mm": float(annual["seasonal_rain_mm"].min()),
                "rain_max_mm": float(annual["seasonal_rain_mm"].max()),
                "temp_mean_c": float(annual["seasonal_temp_mean_c"].mean()),
                "srad_mean_mj_m2_day": float(annual["seasonal_srad_mean_mj_m2_day"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _observed_yield_coverage(project_dir: Path, cfg: dict) -> pd.DataFrame:
    obs = load_observed_soybean_yield(project_dir, cfg).rename(columns={"ano": "year"})
    obs_map = dict(zip(obs["year"].astype(int), obs["observed_yield_kg_ha"].astype(float)))
    rows = []
    for block_key, block in [
        ("development_train_years", "development_train"),
        ("development_validation_years", "development_validation"),
        ("frozen_evaluation_years", "frozen_evaluation"),
        ("illustration_years", "illustration"),
    ]:
        for year in _years(cfg, block_key):
            rows.append(
                {
                    "block": block,
                    "year": year,
                    "observed_yield_kg_ha": obs_map.get(year),
                    "has_observed_yield": year in obs_map,
                }
            )
    return pd.DataFrame(rows)


def _write_markdown(
    path: Path,
    cfg: dict,
    blocks: pd.DataFrame,
    weather: pd.DataFrame,
    coverage: pd.DataFrame,
    validation_messages: list[str],
) -> None:
    design = cfg["evaluation_design"]
    payload = {
        "registry_name": design["registry_name"],
        "registry_version": design["registry_version"],
        "registry_date": design["registry_date"],
        "rule": design["rule"],
        "blocks": blocks.to_dict(orient="records"),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    text = [
        f"# {design['registry_name']}",
        "",
        f"- Version: {design['registry_version']}",
        f"- Date: {design['registry_date']}",
        f"- Environment: {design['environment']}",
        f"- Status: {design['current_run_role']}",
        f"- Registry SHA-256: `{digest}`",
        "",
        "## Blocking rule",
        "",
        design["rule"],
        "",
        "## Temporal blocks",
        "",
        blocks.to_markdown(index=False),
        "",
        "## Weather summary",
        "",
        weather.to_markdown(index=False, floatfmt=".2f"),
        "",
        "## Observed-yield coverage",
        "",
        coverage.to_markdown(index=False, floatfmt=".2f"),
        "",
        "## Validation",
        "",
        "\n".join(f"- {msg}" for msg in validation_messages),
        "",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def build_temporal_split_registry(cfg: dict, run_name: str) -> dict[str, Path]:
    paths = make_paths(cfg, run_name)
    project_dir = paths.project_dir
    blocks = _block_rows(cfg)
    weather = _weather_summary(project_dir, cfg)
    coverage = _observed_yield_coverage(project_dir, cfg)
    messages = _validate_design(cfg)
    paths.tables_dir.mkdir(parents=True, exist_ok=True)

    out_paths = {
        "blocks": paths.tables_dir / "frozen_temporal_split_blocks.csv",
        "weather": paths.tables_dir / "frozen_temporal_split_weather_summary.csv",
        "coverage": paths.tables_dir / "frozen_temporal_split_yield_coverage.csv",
        "json": paths.tables_dir / "TESTE_EXTERNO_COMPOSICAO_v6.json",
        "markdown": paths.tables_dir / "TESTE_EXTERNO_COMPOSICAO_v6.md",
    }
    blocks.to_csv(out_paths["blocks"], index=False, encoding="utf-8-sig")
    weather.to_csv(out_paths["weather"], index=False, encoding="utf-8-sig")
    coverage.to_csv(out_paths["coverage"], index=False, encoding="utf-8-sig")
    payload = {
        "evaluation_design": cfg["evaluation_design"],
        "validation": messages,
        "blocks": blocks.to_dict(orient="records"),
        "weather": weather.to_dict(orient="records"),
        "coverage": coverage.to_dict(orient="records"),
    }
    out_paths["json"].write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(out_paths["markdown"], cfg, blocks, weather, coverage, messages)
    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and validate the frozen Castro temporal split registry.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--run-name", default="ppo_soja_castro_dssat")
    args = parser.parse_args()
    cfg = load_config(args.config)
    outputs = build_temporal_split_registry(cfg, args.run_name)
    print("Temporal split registry written:")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
