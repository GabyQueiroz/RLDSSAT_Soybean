from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import load_config, make_paths


POLICY_LABELS = {
    "ppo_policy": "PPO",
    "rainfed_calendar_oct15": "Rainfed Oct 15",
    "fixed_calendar_oct15_moderate_irrig": "Irrigated Oct 15",
    "early_sep25_moderate_irrig": "Irrigated Sep 25",
    "late_nov10_moderate_irrig": "Irrigated Nov 10",
}


def _load_evaluated_points(tables_dir: Path) -> pd.DataFrame:
    path = tables_dir / "policy_vs_baselines.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = df[df["split"].isin(["valid", "test"])].copy()
    df["policy_label"] = df["policy"].map(POLICY_LABELS).fillna(df["policy"])
    return df


def _policy_summary(df: pd.DataFrame, kappa_values: list[float], target_yield: float) -> pd.DataFrame:
    rows = []
    for kappa in kappa_values:
        tmp = df.copy()
        tmp["water_cost"] = kappa
        tmp["objective"] = tmp["yield_kg_ha"] / target_yield - kappa * tmp["irrigation_mm"]
        summary = (
            tmp.groupby(["split", "policy", "policy_label", "water_cost"], as_index=False)
            .agg(
                seasons=("year", "nunique"),
                mean_yield_kg_ha=("yield_kg_ha", "mean"),
                mean_irrigation_mm=("irrigation_mm", "mean"),
                mean_objective=("objective", "mean"),
                p10_yield_kg_ha=("yield_kg_ha", lambda s: float(np.percentile(s, 10))),
            )
        )
        summary["rank"] = summary.groupby("split")["mean_objective"].rank(method="min", ascending=False).astype(int)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True).sort_values(["split", "water_cost", "rank", "policy"])


def _best_by_kappa(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.loc[summary.groupby(["split", "water_cost"])["mean_objective"].idxmax()].sort_values(
        ["split", "water_cost"]
    )


def _pareto(summary: pd.DataFrame) -> pd.DataFrame:
    base = summary[summary["water_cost"] == 0].copy()
    rows = []
    for split, grp in base.groupby("split"):
        for _, row in grp.iterrows():
            others = grp[grp["policy"] != row["policy"]]
            dominated = (
                (others["mean_yield_kg_ha"] >= row["mean_yield_kg_ha"])
                & (others["mean_irrigation_mm"] <= row["mean_irrigation_mm"])
                & (
                    (others["mean_yield_kg_ha"] > row["mean_yield_kg_ha"])
                    | (others["mean_irrigation_mm"] < row["mean_irrigation_mm"])
                )
            ).any()
            rows.append(
                {
                    "split": split,
                    "policy": row["policy"],
                    "policy_label": row["policy_label"],
                    "mean_yield_kg_ha": row["mean_yield_kg_ha"],
                    "mean_irrigation_mm": row["mean_irrigation_mm"],
                    "pareto_efficient": not bool(dominated),
                }
            )
    return pd.DataFrame(rows).sort_values(["split", "pareto_efficient", "mean_irrigation_mm"], ascending=[True, False, True])


def _paired_vs_rainfed(df: pd.DataFrame) -> pd.DataFrame:
    rainfed = df[df["policy"] == "rainfed_calendar_oct15"][
        ["split", "year", "yield_kg_ha", "irrigation_mm"]
    ].rename(columns={"yield_kg_ha": "rainfed_yield_kg_ha", "irrigation_mm": "rainfed_irrigation_mm"})
    paired = df.merge(rainfed, on=["split", "year"], how="left")
    paired = paired[paired["policy"] != "rainfed_calendar_oct15"].copy()
    paired["delta_yield_vs_rainfed_kg_ha"] = paired["yield_kg_ha"] - paired["rainfed_yield_kg_ha"]
    paired["delta_irrigation_vs_rainfed_mm"] = paired["irrigation_mm"] - paired["rainfed_irrigation_mm"]
    return (
        paired.groupby(["split", "policy", "policy_label"], as_index=False)
        .agg(
            seasons=("year", "nunique"),
            mean_delta_yield_vs_rainfed_kg_ha=("delta_yield_vs_rainfed_kg_ha", "mean"),
            mean_delta_irrigation_vs_rainfed_mm=("delta_irrigation_vs_rainfed_mm", "mean"),
        )
        .sort_values(["split", "mean_delta_yield_vs_rainfed_kg_ha"], ascending=[True, False])
    )


def _plot_frontier(pareto: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.2, 4.6))
    sns.scatterplot(
        data=pareto,
        x="mean_irrigation_mm",
        y="mean_yield_kg_ha",
        hue="policy_label",
        style="split",
        s=90,
    )
    for _, row in pareto.iterrows():
        plt.text(row["mean_irrigation_mm"] + 0.6, row["mean_yield_kg_ha"], row["policy_label"], fontsize=7)
    plt.xlabel("Mean seasonal irrigation (mm)")
    plt.ylabel("Mean corrected yield (kg ha$^{-1}$)")
    plt.title("Evaluated productivity-water points")
    plt.legend(fontsize=7, title="")
    plt.tight_layout()
    plt.savefig(figures_dir / "diagnostic_productivity_water_frontier_en.png", dpi=300)
    plt.close()


def _plot_kappa(summary: pd.DataFrame, figures_dir: Path) -> None:
    plot = summary.copy()
    plot["water_cost_label"] = plot["water_cost"].map(lambda x: f"{x:g}")
    plt.figure(figsize=(7.6, 4.8))
    sns.lineplot(
        data=plot,
        x="water_cost",
        y="mean_objective",
        hue="policy_label",
        style="split",
        marker="o",
    )
    plt.xlabel(r"Water penalty $\kappa$ (mm$^{-1}$)")
    plt.ylabel("Mean objective")
    plt.title("Diagnostic objective sensitivity to water penalty")
    plt.legend(fontsize=7, title="")
    plt.tight_layout()
    plt.savefig(figures_dir / "diagnostic_kappa_sensitivity_en.png", dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--run-name", default="ppo_soja_castro_dssat")
    parser.add_argument("--kappa-values", default="0,0.0005,0.0015,0.003,0.006")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = make_paths(cfg, args.run_name)
    target_yield = float(cfg["reward"]["target_yield_kg_ha"])
    kappa_values = [float(x.strip()) for x in args.kappa_values.split(",") if x.strip()]

    points = _load_evaluated_points(paths.tables_dir)
    summary = _policy_summary(points, kappa_values, target_yield)
    best = _best_by_kappa(summary)
    pareto = _pareto(summary)
    paired = _paired_vs_rainfed(points)

    points.to_csv(paths.tables_dir / "water_frontier_evaluated_points.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(paths.tables_dir / "water_frontier_kappa_summary.csv", index=False, encoding="utf-8-sig")
    best.to_csv(paths.tables_dir / "water_frontier_best_by_kappa.csv", index=False, encoding="utf-8-sig")
    pareto.to_csv(paths.tables_dir / "water_frontier_pareto_diagnostic.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(paths.tables_dir / "water_frontier_paired_vs_rainfed.csv", index=False, encoding="utf-8-sig")
    _plot_frontier(pareto, paths.figures_dir)
    _plot_kappa(summary, paths.figures_dir)

    print("Wrote diagnostic water frontier tables and figures to", paths.output_dir)
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
