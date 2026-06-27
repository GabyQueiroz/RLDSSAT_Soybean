from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


BASE_FEATURES = [
    "year",
    "raw_yield_kg_ha",
    "rain_mm",
    "irrigation_mm",
    "planting_doy",
    "temp_mean_c",
    "srad_mj_m2_day",
]


def expand_features(row: dict[str, float]) -> dict[str, float]:
    year = float(row.get("year", 2000.0))
    raw = float(row.get("raw_yield_kg_ha", 0.0))
    rain = float(row.get("rain_mm", 0.0))
    irrig = float(row.get("irrigation_mm", 0.0))
    doy = float(row.get("planting_doy", 0.0))
    temp = float(row.get("temp_mean_c", 0.0))
    srad = float(row.get("srad_mj_m2_day", 0.0))
    trend = year - 2000.0
    return {
        "intercept_helper": 1.0,
        "year_since_2000": trend,
        "year_since_2000_sq": trend * trend,
        "raw_yield_kg_ha": raw,
        "raw_yield_kg_ha_sq": (raw / 1000.0) ** 2,
        "rain_mm": rain,
        "rain_mm_sq": (rain / 100.0) ** 2,
        "irrigation_mm": irrig,
        "planting_doy": doy,
        "planting_doy_sq": ((doy - 285.0) / 10.0) ** 2,
        "temp_mean_c": temp,
        "srad_mj_m2_day": srad,
        "raw_rain_interaction": (raw / 1000.0) * (rain / 100.0),
        "rain_irrigation_interaction": (rain / 100.0) * (irrig / 10.0),
    }


FEATURES = list(expand_features({}).keys())


@dataclass
class YieldCorrection:
    intercept: float
    coefficients: dict[str, float]
    clip_min: float = 0.0
    clip_max: float = 7000.0

    @classmethod
    def load(cls, path: str | Path) -> "YieldCorrection":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            intercept=float(data["intercept"]),
            coefficients={k: float(v) for k, v in data["coefficients"].items()},
            clip_min=float(data.get("clip_min", 0.0)),
            clip_max=float(data.get("clip_max", 7000.0)),
        )

    def save(self, path: str | Path, metrics: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "intercept": self.intercept,
            "coefficients": self.coefficients,
            "base_features": BASE_FEATURES,
            "features": list(self.coefficients.keys()),
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "metrics": metrics or {},
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def predict_one(self, features: dict[str, float]) -> float:
        expanded = expand_features(features)
        y = self.intercept
        for name, value in expanded.items():
            y += self.coefficients.get(name, 0.0) * float(value)
        return float(np.clip(y, self.clip_min, self.clip_max))


def fit_ridge(rows: list[dict], alpha: float = 10.0) -> tuple[YieldCorrection, dict]:
    feature_names = [name for name in FEATURES if name != "intercept_helper"]
    x = np.array([[float(expand_features(r)[f]) for f in feature_names] for r in rows], dtype=float)
    y = np.array([float(r["observed_yield_kg_ha"]) for r in rows], dtype=float)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std == 0] = 1.0
    y_mean = y.mean()
    xs = (x - x_mean) / x_std
    ys = y - y_mean
    eye = np.eye(xs.shape[1])
    beta_s = np.linalg.solve(xs.T @ xs + alpha * eye, xs.T @ ys)
    beta = beta_s / x_std
    intercept = y_mean - x_mean @ beta
    pred = intercept + x @ beta
    mae = float(np.mean(np.abs(pred - y)))
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    r2 = float(1 - np.sum((pred - y) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-9))
    model = YieldCorrection(
        intercept=float(intercept),
        coefficients={name: float(value) for name, value in zip(feature_names, beta)},
    )
    return model, {"mae": mae, "rmse": rmse, "r2": r2, "n": int(len(rows)), "alpha": alpha}
