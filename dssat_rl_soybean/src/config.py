from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Paths:
    project_dir: Path
    output_dir: Path
    figures_dir: Path
    tables_dir: Path
    models_dir: Path


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path.resolve())
    cfg["_project_dir"] = str(path.resolve().parents[1])
    return cfg


def make_paths(cfg: dict[str, Any], run_name: str = "ppo_soybean") -> Paths:
    project_dir = Path(cfg["_project_dir"]).resolve()
    output_dir = project_dir / "outputs" / run_name
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    models_dir = output_dir / "models"
    for p in [output_dir, figures_dir, tables_dir, models_dir]:
        p.mkdir(parents=True, exist_ok=True)
    return Paths(project_dir, output_dir, figures_dir, tables_dir, models_dir)
