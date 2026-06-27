from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


WEATHER_RENAME = {
    "precipitacao_total_horario_mm": "rain",
    "radiacao_global_kj_m2": "srad_kj_m2",
    "temperatura_maxima_na_hora_ant_aut_c": "tmax_h",
    "temperatura_minima_na_hora_ant_aut_c": "tmin_h",
    "temperatura_do_ar_bulbo_seco_horaria_c": "temp_h",
}


@dataclass(frozen=True)
class YearWeather:
    year: int
    daily: pd.DataFrame
    features: np.ndarray


def _resolve(project_dir: Path, path_value: str) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else (project_dir / p).resolve()


def load_weather(project_dir: Path, cfg: dict) -> pd.DataFrame:
    path = _resolve(project_dir, cfg["data"]["weather_csv"])
    df = pd.read_csv(path, low_memory=False)
    df["data_hora"] = pd.to_datetime(df["data_hora"], errors="coerce")
    df = df.dropna(subset=["data_hora"]).copy()
    df["date"] = df["data_hora"].dt.date
    df = df.rename(columns=WEATHER_RENAME)
    for col in WEATHER_RENAME.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] <= -90, col] = np.nan
    if "rain" in df.columns:
        df.loc[df["rain"] < 0, "rain"] = np.nan
    for col in ["tmax_h", "tmin_h", "temp_h"]:
        if col in df.columns:
            df.loc[(df[col] < -20) | (df[col] > 50), col] = np.nan
    if "srad_kj_m2" in df.columns:
        df.loc[(df["srad_kj_m2"] < 0) | (df["srad_kj_m2"] > 6000), "srad_kj_m2"] = np.nan
    if "rain" not in df:
        raise ValueError("Coluna de precipitação não encontrada na base climática.")
    daily = (
        df.groupby("date", as_index=False)
        .agg(
            year=("ano", "first"),
            rain=("rain", lambda s: s.sum(min_count=1)),
            rain_obs=("rain", "count"),
            tmax=("tmax_h", "max"),
            tmin=("tmin_h", "min"),
            temp_mean=("temp_h", "mean"),
            temp_obs=("temp_h", "count"),
            srad_kj_m2=("srad_kj_m2", lambda s: s.sum(min_count=1)),
            srad_obs=("srad_kj_m2", "count"),
        )
        .sort_values("date")
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily["year"] = daily["date"].dt.year
    daily.loc[daily["rain_obs"] < 18, "rain"] = np.nan
    daily.loc[daily["temp_obs"] < 18, ["tmax", "tmin", "temp_mean"]] = np.nan
    daily.loc[daily["srad_obs"] < 6, "srad_kj_m2"] = np.nan
    daily["rain"] = _fill_missing_rain(daily)
    daily["srad"] = daily["srad_kj_m2"].clip(lower=0) / 1000.0
    daily["tmax"] = daily["tmax"].fillna(daily["temp_mean"]).interpolate(limit_direction="both")
    daily["tmin"] = daily["tmin"].fillna(daily["temp_mean"]).interpolate(limit_direction="both")
    daily["temp_mean"] = daily["temp_mean"].fillna((daily["tmax"] + daily["tmin"]) / 2)
    daily["rain"] = daily["rain"].clip(lower=0)
    daily["srad"] = _fill_missing_srad(daily, cfg["dssat"]["latitude"])
    return daily[["date", "year", "rain", "tmax", "tmin", "temp_mean", "srad"]]


def _fill_missing_rain(daily: pd.DataFrame) -> pd.Series:
    rain = daily["rain"].copy()
    if not rain.isna().any():
        return rain.clip(lower=0)
    doy = pd.to_datetime(daily["date"]).dt.dayofyear
    clim = daily.assign(doy=doy).groupby("doy")["rain"].mean()
    monthly = daily.assign(month=pd.to_datetime(daily["date"]).dt.month).groupby("month")["rain"].mean()
    fill = doy.map(clim)
    missing = rain.isna()
    rain.loc[missing] = fill.loc[missing].to_numpy()
    still_missing = rain.isna()
    if still_missing.any():
        month = pd.to_datetime(daily["date"]).dt.month
        rain.loc[still_missing] = month.map(monthly).loc[still_missing].to_numpy()
    return rain.fillna(0).clip(lower=0)


def _fill_missing_srad(daily: pd.DataFrame, latitude: float) -> pd.Series:
    """Estimate solar radiation with Hargreaves when INMET radiation is missing/zero."""
    srad = daily["srad"].copy()
    missing = srad.isna() | (srad <= 0.5)
    if not missing.any():
        return srad
    lat_rad = np.deg2rad(latitude)
    doy = pd.to_datetime(daily["date"]).dt.dayofyear.to_numpy()
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    decl = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(decl), -1, 1))
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.sin(ws)
    )
    td = (daily["tmax"] - daily["tmin"]).clip(lower=0).to_numpy()
    estimated = 0.16 * np.sqrt(td) * ra
    srad.loc[missing] = estimated[missing.to_numpy()]
    return srad.clip(lower=1.0, upper=35.0)


def load_observed_soybean_yield(project_dir: Path, cfg: dict) -> pd.DataFrame:
    path = _resolve(project_dir, cfg["data"]["productivity_csv"])
    df = pd.read_csv(path, low_memory=False)
    mask = (
        (df["municipio"] == cfg["data"]["city"])
        & (df["produto"] == cfg["data"]["crop_product"])
        & df["variavel"].astype(str).str.contains("Rendimento", case=False, na=False)
    )
    out = df.loc[mask, ["ano", "valor_numerico"]].copy()
    out["ano"] = pd.to_numeric(out["ano"], errors="coerce").astype("Int64")
    out["observed_yield_kg_ha"] = pd.to_numeric(out["valor_numerico"], errors="coerce")
    out = out.dropna(subset=["ano", "observed_yield_kg_ha"])[["ano", "observed_yield_kg_ha"]]
    out = out.groupby("ano", as_index=False)["observed_yield_kg_ha"].mean()
    return out.sort_values("ano")


def build_year_weather(daily: pd.DataFrame, years: list[int]) -> list[YearWeather]:
    result: list[YearWeather] = []
    for year in years:
        start = pd.Timestamp(year=year, month=9, day=1)
        end = pd.Timestamp(year=year + 1, month=4, day=30)
        y = daily[(daily["date"] >= start) & (daily["date"] <= end)].copy()
        if y.empty:
            continue
        features = np.array(
            [
                y["rain"].sum() / 1800.0,
                y["temp_mean"].mean() / 35.0,
                y["tmax"].max() / 45.0,
                y["tmin"].min() / 20.0,
                y["srad"].mean() / 30.0,
                len(y) / 366.0,
            ],
            dtype=np.float32,
        )
        result.append(YearWeather(year=year, daily=y, features=np.nan_to_num(features)))
    if not result:
        raise ValueError(f"Nenhum ano disponível na base climática para {years}.")
    return result


def planting_date_for_year(year: int, month_day: str, offset_days: int) -> date:
    month, day = map(int, month_day.split("-"))
    return date(year, month, day) + timedelta(days=int(offset_days))


def summarize_weather_splits(daily: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    for split in ["train_years", "valid_years", "test_years"]:
        years = cfg["data"][split]
        seasons = build_year_weather(daily, years)
        d = pd.concat([s.daily.assign(season_year=s.year) for s in seasons], ignore_index=True)
        rows.append(
            {
                "split": split.replace("_years", ""),
                "years": ",".join(map(str, years)),
                "n_days": len(d),
                "rain_mean_mm_year": d.groupby("season_year")["rain"].sum().mean(),
                "rain_min_mm_year": d.groupby("season_year")["rain"].sum().min(),
                "rain_max_mm_year": d.groupby("season_year")["rain"].sum().max(),
                "temp_mean_c": d["temp_mean"].mean(),
                "srad_mean_mj_m2_day": d["srad"].mean(),
            }
        )
    return pd.DataFrame(rows)
