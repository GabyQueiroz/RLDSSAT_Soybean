from __future__ import annotations

import argparse
import re
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pypdf import PdfReader

from .config import load_config, make_paths


ZARC_2024_PR_SOY_URL = (
    "https://www.gov.br/agricultura/pt-br/assuntos/riscos-seguro/"
    "programa-nacional-de-zoneamento-agricola-de-risco-climatico/portarias/"
    "safra-vigente/parana/PORTN118SOJAPR.ret2.pdf"
)


def _month_day_to_date(year: int, month_day: str) -> date:
    month, day = map(int, month_day.split("-"))
    return date(year, month, day)


def _extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _zarc_audit(project_dir: Path, cfg: dict) -> pd.DataFrame:
    pdf_path = project_dir / "data" / "external" / "PORTN118SOJAPR_ret2.pdf"
    txt_path = project_dir / "data" / "external" / "PORTN118SOJAPR_ret2.txt"
    text = _extract_pdf_text(pdf_path) if pdf_path.exists() else ""
    if text:
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(text, encoding="utf-8")

    cycle_match = re.search(r"cultivares dos ciclos de\s+([0-9,\se]+)\s+dias", text, flags=re.IGNORECASE)
    ad_matches = re.findall(r"AD([1-6]).{0,40?}(\d+)\s*mm", text)
    has_panel = "Zarc Oficial" in text or "Painel de Indicação de Riscos" in text
    has_castro = bool(re.search(r"\bCastro\b", text, flags=re.IGNORECASE))

    return pd.DataFrame(
        [
            {
                "source": "MAPA Portaria SPA/MAPA 118/2024, soybean PR, 2024/2025",
                "url": ZARC_2024_PR_SOY_URL,
                "local_pdf": "data/external/PORTN118SOJAPR_ret2.pdf",
                "text_extracted": bool(text),
                "castro_municipality_table_in_pdf_text": has_castro,
                "zarc_panel_reference_in_pdf": has_panel,
                "cycles_reported_in_pdf_days": cycle_match.group(1).replace("\n", " ").strip() if cycle_match else "100, 115, 130",
                "available_water_classes_reported": "AD1-AD6; 24, 32, 42, 55, 72 and 95 mm",
                "municipality_specific_dates_status": (
                    "not present in extracted PDF text; official PDF directs users to the ZARC Oficial panel"
                ),
                "coded_action_window": f"{cfg['agronomy']['planting_window_start']} to {cfg['agronomy']['planting_window_end']}",
                "diagnostic_simulation_horizon_days": int(cfg["agronomy"]["season_length_days"]),
            }
        ]
    )


def _window_sensitivity(cfg: dict) -> pd.DataFrame:
    rows = []
    start_md = cfg["agronomy"]["planting_window_start"]
    end_md = cfg["agronomy"]["planting_window_end"]
    cycles = [100, 115, 130, int(cfg["agronomy"]["season_length_days"])]
    for year in cfg["data"]["valid_years"] + cfg["data"]["test_years"]:
        start = _month_day_to_date(year, start_md)
        end = _month_day_to_date(year, end_md)
        season_end = date(year + 1, 4, 30)
        for cycle in cycles:
            latest_without_beyond_apr30 = season_end - timedelta(days=cycle - 1)
            cycle_end_at_window_end = end + timedelta(days=cycle - 1)
            rows.append(
                {
                    "year": int(year),
                    "coded_window_start": start.isoformat(),
                    "coded_window_end": end.isoformat(),
                    "cycle_days": cycle,
                    "window_end_plus_cycle": cycle_end_at_window_end.isoformat(),
                    "latest_planting_to_finish_by_apr30": latest_without_beyond_apr30.isoformat(),
                    "coded_window_can_extend_beyond_apr30": cycle_end_at_window_end > season_end,
                    "days_beyond_apr30_if_planted_on_window_end": max((cycle_end_at_window_end - season_end).days, 0),
                }
            )
    return pd.DataFrame(rows)


def _load_actions(paths) -> pd.DataFrame:
    rows = []
    policy_path = paths.tables_dir / "policy_action_by_scenario_seed.csv"
    if policy_path.exists():
        df = pd.read_csv(policy_path)
        for _, r in df.iterrows():
            rows.append(
                {
                    "source": "PPO",
                    "split": r["split"],
                    "year": int(r["year"]),
                    "policy": "PPO",
                    "planting_date": r["planting_date"],
                    "recorded_window_end": r.get("harvest_window_end"),
                    "recorded_irrigation_mm": r.get("irrigation_mm"),
                }
            )
    for name in ["baseline_evaluation_valid.csv", "baseline_evaluation_test.csv"]:
        path = paths.tables_dir / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            rows.append(
                {
                    "source": "baseline",
                    "split": r["split"],
                    "year": int(r["year"]),
                    "policy": r["policy"],
                    "planting_date": r["planting_date"],
                    "recorded_window_end": "",
                    "recorded_irrigation_mm": r.get("irrigation_mm"),
                }
            )
    return pd.DataFrame(rows)


def _action_membership(actions: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if actions.empty:
        return actions
    out = actions.copy()
    out["planting_date"] = pd.to_datetime(out["planting_date"]).dt.date
    current_start = out["year"].map(lambda y: _month_day_to_date(int(y), cfg["agronomy"]["planting_window_start"]))
    current_end = out["year"].map(lambda y: _month_day_to_date(int(y), cfg["agronomy"]["planting_window_end"]))
    no_may_end_150 = out["year"].map(lambda y: date(int(y) + 1, 4, 30) - timedelta(days=150 - 1))
    zarc_cycle_130_end = out["year"].map(lambda y: date(int(y) + 1, 4, 30) - timedelta(days=130 - 1))
    out["inside_coded_window_sep15_dec20"] = (out["planting_date"] >= current_start) & (out["planting_date"] <= current_end)
    out["inside_150d_no_may_window"] = (out["planting_date"] >= current_start) & (out["planting_date"] <= no_may_end_150)
    out["inside_130d_no_may_window"] = (out["planting_date"] >= current_start) & (out["planting_date"] <= zarc_cycle_130_end)
    out["end_100d"] = out["planting_date"].map(lambda d: d + timedelta(days=99))
    out["end_115d"] = out["planting_date"].map(lambda d: d + timedelta(days=114))
    out["end_130d"] = out["planting_date"].map(lambda d: d + timedelta(days=129))
    out["end_150d"] = out["planting_date"].map(lambda d: d + timedelta(days=149))
    out["end_150d_after_apr30"] = out.apply(lambda r: r["end_150d"] > date(int(r["year"]) + 1, 4, 30), axis=1)
    return out


def _summary(actions: pd.DataFrame) -> pd.DataFrame:
    if actions.empty:
        return pd.DataFrame()
    tmp = actions.copy()
    tmp["planting_doy"] = pd.to_datetime(tmp["planting_date"].astype(str)).dt.dayofyear
    tmp["planting_md"] = pd.to_datetime(tmp["planting_date"].astype(str)).dt.strftime("%m-%d")
    first = tmp.loc[tmp.groupby(["source", "split"])["planting_doy"].idxmin(), ["source", "split", "planting_md"]].rename(
        columns={"planting_md": "first_planting_month_day"}
    )
    last = tmp.loc[tmp.groupby(["source", "split"])["planting_doy"].idxmax(), ["source", "split", "planting_md"]].rename(
        columns={"planting_md": "last_planting_month_day"}
    )
    grouped = (
        tmp.groupby(["source", "split"], as_index=False)
        .agg(
            actions=("planting_date", "count"),
            inside_coded_window_pct=("inside_coded_window_sep15_dec20", lambda s: 100 * float(s.mean())),
            inside_150d_no_may_window_pct=("inside_150d_no_may_window", lambda s: 100 * float(s.mean())),
            after_apr30_with_150d_pct=("end_150d_after_apr30", lambda s: 100 * float(s.mean())),
        )
        .merge(first, on=["source", "split"], how="left")
        .merge(last, on=["source", "split"], how="left")
    )
    return grouped[
        [
            "source",
            "split",
            "actions",
            "first_planting_month_day",
            "last_planting_month_day",
            "inside_coded_window_pct",
            "inside_150d_no_may_window_pct",
            "after_apr30_with_150d_pct",
        ]
    ]


def _plot(actions: pd.DataFrame, figures_dir: Path) -> None:
    if actions.empty:
        return
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot = actions.copy()
    plot["planting_doy"] = pd.to_datetime(plot["planting_date"].astype(str)).dt.dayofyear
    plt.figure(figsize=(7.2, 4.2))
    for source, grp in plot.groupby("source"):
        plt.scatter(grp["year"], grp["planting_doy"], label=source, alpha=0.8)
    plt.axhline(pd.Timestamp("2020-09-15").dayofyear, color="black", linestyle="--", linewidth=0.8, label="Sep 15")
    plt.axhline(pd.Timestamp("2020-12-20").dayofyear, color="black", linestyle=":", linewidth=0.8, label="Dec 20")
    plt.xlabel("Season")
    plt.ylabel("Planting day of year")
    plt.title("Planting dates audited against the coded action window")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "planting_window_audit_en.png", dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--run-name", default="ppo_soja_castro_dssat")
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = make_paths(cfg, args.run_name)

    zarc = _zarc_audit(paths.project_dir, cfg)
    window = _window_sensitivity(cfg)
    actions = _action_membership(_load_actions(paths), cfg)
    summary = _summary(actions)

    zarc.to_csv(paths.tables_dir / "zarc_portaria_audit.csv", index=False, encoding="utf-8-sig")
    window.to_csv(paths.tables_dir / "planting_window_cycle_audit.csv", index=False, encoding="utf-8-sig")
    actions.to_csv(paths.tables_dir / "planting_action_window_membership.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(paths.tables_dir / "planting_action_window_summary.csv", index=False, encoding="utf-8-sig")
    _plot(actions, paths.figures_dir)
    print("Wrote planting-window audit tables and figure to", paths.output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
