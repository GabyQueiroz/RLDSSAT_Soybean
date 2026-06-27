from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=240)
    plt.close()


def _plot_observed_vs_predicted(df: pd.DataFrame, path: Path, pred_col: str, title: str):
    plot = df.dropna(subset=["observed_yield_kg_ha", pred_col]).copy()
    if plot.empty:
        return
    lo = min(plot["observed_yield_kg_ha"].min(), plot[pred_col].min()) * 0.95
    hi = max(plot["observed_yield_kg_ha"].max(), plot[pred_col].max()) * 1.05
    plt.figure(figsize=(6.2, 5.4))
    sns.scatterplot(data=plot, x="observed_yield_kg_ha", y=pred_col, hue="split", s=82)
    plt.plot([lo, hi], [lo, hi], color="black", linewidth=1.2, linestyle="--")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("Observed yield (kg ha$^{-1}$)")
    plt.ylabel("Predicted yield (kg ha$^{-1}$)")
    plt.title(title)
    _savefig(path)


def build_report(cfg: dict, paths):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)

    val_path = paths.tables_dir / "validation_curve.csv"
    if val_path.exists():
        val = pd.read_csv(val_path)
        if not val.empty:
            fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
            sns.lineplot(data=val, x="timesteps", y="mean_reward", marker="o", ax=ax1, color="#1f77b4")
            ax1.set_xlabel("PPO training steps")
            ax1.set_ylabel("Validation reward")
            ax2 = ax1.twinx()
            sns.lineplot(data=val, x="timesteps", y="mean_yield_kg_ha", marker="s", ax=ax2, color="#2ca02c")
            ax2.set_ylabel("Validation yield (kg ha$^{-1}$)")
            _savefig(paths.figures_dir / "validation_learning_curve_en.png")

    calibration_path = paths.tables_dir / "yield_calibration_summary.csv"
    if calibration_path.exists():
        cal = pd.read_csv(calibration_path)
        _plot_observed_vs_predicted(
            cal,
            paths.figures_dir / "dssat_observed_vs_corrected_yield_en.png",
            "corrected_yield_kg_ha",
            "DSSAT yield correction",
        )
        plt.figure(figsize=(8.2, 4.4))
        sns.barplot(data=cal, x="year", y="corrected_error_kg_ha", hue="split", dodge=False)
        plt.axhline(0, color="black", linewidth=1)
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Year")
        plt.ylabel("Prediction error (kg ha$^{-1}$)")
        _savefig(paths.figures_dir / "dssat_correction_error_by_year_en.png")

    eval_files = [p for p in paths.tables_dir.glob("policy_evaluation_*.csv") if p.name != "policy_evaluation_all.csv"]
    if eval_files:
        ev = pd.concat([pd.read_csv(p) for p in eval_files], ignore_index=True)
        ev.to_csv(paths.tables_dir / "policy_evaluation_all.csv", index=False, encoding="utf-8-sig")
        summary = (
            ev.groupby("split")
            .agg(
                reward_mean=("reward", "mean"),
                yield_mean_kg_ha=("yield_kg_ha", "mean"),
                yield_std_kg_ha=("yield_kg_ha", "std"),
                irrigation_mean_mm=("irrigation_mm", "mean"),
                mae_kg_ha=("yield_abs_error_kg_ha", "mean"),
                mape_percent=("yield_abs_percent_error", "mean"),
                planting_offset_mean_days=("planting_offset_days", "mean"),
                n=("reward", "count"),
            )
            .reset_index()
        )
        summary.to_csv(paths.tables_dir / "policy_summary_by_split.csv", index=False, encoding="utf-8-sig")

        _plot_observed_vs_predicted(
            ev,
            paths.figures_dir / "ppo_observed_vs_predicted_yield_en.png",
            "yield_kg_ha",
            "PPO policy evaluated with DSSAT",
        )

        plt.figure(figsize=(7.5, 4.4))
        sns.boxplot(data=ev, x="split", y="yield_kg_ha", hue="split", legend=False)
        sns.stripplot(data=ev, x="split", y="observed_yield_kg_ha", color="black", size=5, jitter=0.08)
        plt.ylabel("Yield (kg ha$^{-1}$)")
        plt.xlabel("Data split")
        _savefig(paths.figures_dir / "ppo_yield_distribution_en.png")

        plt.figure(figsize=(7.2, 4.4))
        sns.scatterplot(data=ev, x="irrigation_mm", y="yield_kg_ha", hue="split", style="split", s=75)
        plt.xlabel("Seasonal irrigation (mm)")
        plt.ylabel("Predicted yield (kg ha$^{-1}$)")
        _savefig(paths.figures_dir / "ppo_yield_water_tradeoff_en.png")

    baseline_files = list(paths.tables_dir.glob("baseline_evaluation_*.csv"))
    policy_all = paths.tables_dir / "policy_evaluation_all.csv"
    if baseline_files and policy_all.exists():
        base = pd.concat([pd.read_csv(p) for p in baseline_files], ignore_index=True)
        pol = pd.read_csv(policy_all)
        pol = pol.assign(policy="ppo_policy")
        cols = ["split", "year", "policy", "yield_kg_ha", "irrigation_mm", "reward"]
        comp = pd.concat([base[cols], pol[cols]], ignore_index=True)
        comp.to_csv(paths.tables_dir / "policy_vs_baselines.csv", index=False, encoding="utf-8-sig")
        plt.figure(figsize=(9.2, 4.6))
        sns.barplot(data=comp, x="policy", y="reward", hue="split", errorbar="sd")
        plt.xticks(rotation=25, ha="right")
        plt.xlabel("")
        plt.ylabel("Reward")
        _savefig(paths.figures_dir / "ppo_vs_baselines_reward_en.png")

    weather_path = paths.tables_dir / "weather_splits.csv"
    if weather_path.exists():
        weather = pd.read_csv(weather_path)
        weather.to_markdown(paths.tables_dir / "weather_splits.md", index=False)
