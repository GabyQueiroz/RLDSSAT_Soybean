from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .yield_correction import YieldCorrection


@dataclass
class SimulationResult:
    yield_kg_ha: float
    irrigation_mm: float
    rain_mm: float
    planting_date: date
    raw: dict[str, Any]


def build_irrigation_schedule(
    daily: pd.DataFrame,
    planting_date: date,
    season_length_days: int,
    trigger_dryness: float,
    amount_mm: float,
    max_season_irrigation_mm: float,
    check_days: int,
) -> pd.DataFrame:
    season = daily[
        (daily["date"].dt.date >= planting_date)
        & (daily["date"].dt.date < planting_date + timedelta(days=season_length_days))
    ].copy()
    if season.empty:
        return pd.DataFrame(columns=["idate", "irval", "irop"])

    water_balance = 0.0
    total_irrig = 0.0
    events = []
    for i, row in enumerate(season.itertuples(index=False)):
        evap_proxy = max(0.0, 0.16 * float(row.srad) + 0.08 * max(float(row.temp_mean) - 10.0, 0.0))
        water_balance += evap_proxy - float(row.rain)
        water_balance = max(0.0, water_balance)
        if i % max(check_days, 1) == 0 and water_balance >= trigger_dryness and total_irrig < max_season_irrigation_mm:
            applied = min(float(amount_mm), max_season_irrigation_mm - total_irrig)
            if applied > 0:
                events.append({"idate": row.date.date(), "irval": round(applied, 1), "irop": "IR001"})
                total_irrig += applied
                water_balance = max(0.0, water_balance - applied)
    return pd.DataFrame(events)


class MockDSSATRunner:
    """Fast surrogate used only for pipeline tests when DSSAT-CSM is unavailable."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def run(
        self,
        daily: pd.DataFrame,
        planting_date: date,
        irrigation_schedule: pd.DataFrame,
        rng: np.random.Generator,
    ) -> SimulationResult:
        ag = self.cfg["agronomy"]
        season = daily[
            (daily["date"].dt.date >= planting_date)
            & (daily["date"].dt.date < planting_date + timedelta(days=ag["season_length_days"]))
        ].copy()
        if season.empty:
            return SimulationResult(0.0, 0.0, 0.0, planting_date, {"status": "empty_season"})

        rain = float(season["rain"].sum())
        irrig = float(irrigation_schedule["irval"].sum()) if not irrigation_schedule.empty else 0.0
        mean_temp = float(season["temp_mean"].mean())
        heat_penalty = max(0.0, mean_temp - 24.5) * 170.0 + max(0.0, 17.0 - mean_temp) * 130.0
        water_supply = rain + 0.86 * irrig
        water_score = 1.0 - abs(water_supply - 560.0) / 560.0
        water_score = float(np.clip(water_score, 0.15, 1.08))
        planting_doy = planting_date.timetuple().tm_yday
        planting_score = 1.0 - abs(planting_doy - 295) / 90.0
        planting_score = float(np.clip(planting_score, 0.55, 1.05))
        radiation_score = float(np.clip(season["srad"].mean() / 17.0, 0.7, 1.2))
        noise = rng.normal(0, 90)
        yld = 4300.0 * water_score * planting_score * radiation_score - heat_penalty + noise
        yld = float(np.clip(yld, 250.0, 6200.0))
        return SimulationResult(yld, irrig, rain, planting_date, {"status": "mock"})


class PyDSSATRunner:
    """DSSATTools-backed soybean runner.

    Requires DSSAT-CSM and a valid DSSAT soil profile. Configure `dssat.soil_file`
    and `dssat.soil_id` in `configs/experiment.yaml`.
    """

    def __init__(self, cfg: dict, project_dir: Path):
        self.cfg = cfg
        self.project_dir = project_dir
        try:
            from DSSATTools.crop import Soybean
            from DSSATTools.filex import (
                Field,
                Irrigation,
                IrrigationEvent,
                Planting,
                SCGeneral,
                SCManagement,
                SCMethods,
                SCOptions,
                SCOutputs,
                SimulationControls,
            )
            from DSSATTools.run import DSSAT
            from DSSATTools.soil import SoilLayer, SoilProfile
            from DSSATTools.weather import WeatherRecord, WeatherStation
        except Exception as exc:
            raise RuntimeError(
                "DSSATTools não está instalado ou não importou corretamente. "
                "Instale requirements.txt e confira o DSSAT-CSM."
            ) from exc

        self._api = {
            "Soybean": Soybean,
            "Field": Field,
            "Irrigation": Irrigation,
            "IrrigationEvent": IrrigationEvent,
            "Planting": Planting,
            "SCGeneral": SCGeneral,
            "SCManagement": SCManagement,
            "SCMethods": SCMethods,
            "SCOptions": SCOptions,
            "SCOutputs": SCOutputs,
            "SimulationControls": SimulationControls,
            "DSSAT": DSSAT,
            "SoilLayer": SoilLayer,
            "SoilProfile": SoilProfile,
            "WeatherStation": WeatherStation,
            "WeatherRecord": WeatherRecord,
        }
        soil_file = cfg["dssat"].get("soil_file")
        soil_id = cfg["dssat"].get("soil_id")
        if soil_file and soil_id:
            soil_path = Path(soil_file)
            if not soil_path.is_absolute():
                soil_path = (project_dir / soil_path).resolve()
            self.soil = SoilProfile.from_file(soil_id, str(soil_path))
        elif cfg["dssat"].get("use_default_soil", False):
            self.soil = self._default_soil_profile()
        else:
            raise ValueError(
                "Configure dssat.soil_file/dssat.soil_id ou deixe dssat.use_default_soil: true "
                "para um teste inicial não calibrado."
            )
        self.yield_correction = None
        if cfg.get("calibration", {}).get("enabled", False):
            model_path = Path(cfg["calibration"]["model_path"])
            if not model_path.is_absolute():
                model_path = project_dir / model_path
            if model_path.exists():
                self.yield_correction = YieldCorrection.load(model_path)

    def _default_soil_profile(self):
        SoilLayer = self._api["SoilLayer"]
        SoilProfile = self._api["SoilProfile"]
        layers = [
            SoilLayer(slb=15, slll=0.18, sdul=0.32, ssat=0.48, srgf=1.00, sbdm=1.18, sloc=2.2, slcl=60, slsi=25),
            SoilLayer(slb=30, slll=0.20, sdul=0.34, ssat=0.47, srgf=0.80, sbdm=1.22, sloc=1.8, slcl=62, slsi=24),
            SoilLayer(slb=60, slll=0.22, sdul=0.35, ssat=0.46, srgf=0.55, sbdm=1.27, sloc=1.3, slcl=65, slsi=22),
            SoilLayer(slb=100, slll=0.23, sdul=0.36, ssat=0.45, srgf=0.30, sbdm=1.32, sloc=0.9, slcl=66, slsi=21),
            SoilLayer(slb=150, slll=0.24, sdul=0.37, ssat=0.44, srgf=0.15, sbdm=1.36, sloc=0.6, slcl=68, slsi=20),
        ]
        return SoilProfile(
            table=layers,
            name="CASTRO0001",
            salb=0.13,
            slu1=6.0,
            sldr=0.5,
            slro=65.0,
            slnf=1.0,
            slpf=1.0,
            soil_data_source="DEFAULT",
            soil_clasification="CLAYEY TEST PROFILE",
            soil_series_name="CASTRO DEFAULT",
            site="CASTRO",
            country="BRAZIL",
            lat=self.cfg["dssat"]["latitude"],
            long=self.cfg["dssat"]["longitude"],
        )

    def _weather_station(self, daily: pd.DataFrame):
        WeatherStation = self._api["WeatherStation"]
        WeatherRecord = self._api["WeatherRecord"]
        table = daily[["date", "srad", "tmax", "tmin", "rain"]].copy()
        table["date"] = pd.to_datetime(table["date"]).dt.date
        records = [
            WeatherRecord(
                date=row.date,
                srad=max(float(row.srad), 0.0),
                tmax=float(row.tmax),
                tmin=float(row.tmin),
                rain=max(float(row.rain), 0.0),
            )
            for row in table.itertuples(index=False)
        ]
        return WeatherStation(
            insi="CAST",
            lat=self.cfg["dssat"]["latitude"],
            long=self.cfg["dssat"]["longitude"],
            elev=self.cfg["dssat"]["elevation_m"],
            table=records,
        )

    def run(
        self,
        daily: pd.DataFrame,
        planting_date: date,
        irrigation_schedule: pd.DataFrame,
        rng: np.random.Generator,
    ) -> SimulationResult:
        api = self._api
        ag = self.cfg["agronomy"]
        weather = self._weather_station(daily)
        crop = api["Soybean"](ag["soybean_cultivar"])
        field = api["Field"](
            id_field="CAST0001",
            wsta=weather,
            id_soil=self.soil,
            flob=0,
            fldd=0,
            flds=0,
            fldt="DR000",
            elev=self.cfg["dssat"]["elevation_m"],
        )
        planting = api["Planting"](
            pdate=planting_date,
            ppop=ag["plant_population_m2"],
            ppoe=ag["plant_population_m2"],
            plme="S",
            plds="R",
            plrs=ag["row_spacing_cm"],
            pldp=ag["planting_depth_cm"],
        )
        if not irrigation_schedule.empty:
            irrigation_events = [
                api["IrrigationEvent"](idate=row.idate, irval=float(row.irval), irop=row.irop)
                for row in irrigation_schedule.itertuples(index=False)
            ]
            irrigation = api["Irrigation"](table=irrigation_events)
        else:
            irrigation = None
        controls = api["SimulationControls"](
            general=api["SCGeneral"](sdate=planting_date),
            options=api["SCOptions"](water="Y", nitro="N"),
            methods=api["SCMethods"](),
            management=api["SCManagement"](plant="R", irrig="R" if irrigation else "N", ferti="N"),
            outputs=api["SCOutputs"](sumry="Y", grout="Y", waout="Y"),
        )
        run_root = Path(self.cfg["dssat"]["run_root"])
        if not run_root.is_absolute():
            run_root = self.project_dir / run_root
        run_root.mkdir(parents=True, exist_ok=True)
        dssat = api["DSSAT"](str(run_root / f"run_{pd.Timestamp.utcnow().value}"))
        ok = False
        try:
            result = dssat.run_treatment(
                field=field,
                cultivar=crop,
                planting=planting,
                irrigation=irrigation,
                simulation_controls=controls,
                verbose=False,
            )
            ok = True
            yld = float(result.get("harwt", result.get("HARWT", np.nan)))
            rain = float(result.get("rain", result.get("RAIN", daily["rain"].sum())))
            if not np.isfinite(yld):
                raise RuntimeError(f"DSSAT executou, mas nao retornou produtividade valida. Summary={result}")
            irrig = float(irrigation_schedule["irval"].sum()) if not irrigation_schedule.empty else 0.0
            raw_yld = yld
            if self.yield_correction is not None:
                yld = self.yield_correction.predict_one(
                    {
                        "year": planting_date.year,
                        "raw_yield_kg_ha": raw_yld,
                        "rain_mm": rain,
                        "irrigation_mm": irrig,
                        "planting_doy": planting_date.timetuple().tm_yday,
                        "temp_mean_c": float(daily["temp_mean"].mean()),
                        "srad_mj_m2_day": float(daily["srad"].mean()),
                    }
                )
            if isinstance(result, dict):
                result = {**result, "raw_harwt": raw_yld, "corrected_harwt": yld}
            return SimulationResult(yld, irrig, rain, planting_date, result)
        finally:
            keep_success = self.cfg["dssat"].get("keep_success_runs", False)
            keep_failed = self.cfg["dssat"].get("keep_failed_runs", False)
            if (ok and not keep_success) or ((not ok) and not keep_failed):
                dssat.close()
