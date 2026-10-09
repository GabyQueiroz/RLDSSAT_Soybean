from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .article_results import COLORS, LABELS, SEED_SHADES, _save, _style, _tensorboard_scalars
from .config import load_config, make_paths
from .final_comparators import CONSTRUCTED
from .rolling_origin import folds

FOLD_SHADES = ["#0d3b73", "#1d5aa6", "#2a78d6", "#6aa3e6", "#a9c9f0"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Figures and tables for the rolling-origin NASA POWER evaluation.")
    parser.add_argument("--config", default="configs/experiment_castro_power.yaml")
    parser.add_argument("--prefix", default="rolling_power")
    args = parser.parse_args()

    _style()
    cfg = load_config(args.config)
    paths = make_paths(cfg, "article_rolling_power")
    out = paths.output_dir
    outputs = paths.project_dir / "outputs"
    tables = outputs / f"{args.prefix}_analysis" / "tables"
    seeds = [int(s) for s in cfg["evaluation_design"]["seeds"]]
    fold_list = folds(cfg)

    grid = pd.read_csv(tables / "candidate_grid_all_seasons.csv")
    comps = pd.read_csv(tables / "comparator_rows_test.csv")
    ppo = pd.read_csv(tables / "ppo_rows_test.csv")
    paired = pd.read_csv(tables / "paired_effects_seed_season.csv")
    season_level = pd.read_csv(tables / "season_level.csv")

    # Response surface over all 41 seasons.
    rainfed = grid[grid["candidate_id"].str.endswith("trig9999_amt0_cap0")]
    best_any = grid[~grid["candidate_id"].isin(CONSTRUCTED)].groupby("year")["yield_kg_ha"].max()
    best_rf = rainfed.groupby("year")["yield_kg_ha"].max()
    landscape = pd.DataFrame({
        "rainfed_sowing_range_kg_ha": rainfed.groupby("year")["yield_kg_ha"].agg(lambda s: s.max() - s.min()),
        "best_rainfed_kg_ha": best_rf, "best_any_kg_ha": best_any, "irrigation_gain_kg_ha": best_any - best_rf,
        "rainfed_oct15_kg_ha": grid[grid["candidate_id"] == "rainfed_oct15"].set_index("year")["yield_kg_ha"],
    })
    landscape.to_csv(out / "tables" / "response_landscape_all_seasons.csv")

    # Learning curves and selected checkpoints per fold and seed.
    curves, ckpts, scalars = [], [], []
    for fold in fold_list:
        for seed in seeds:
            run = outputs / args.prefix / f"{fold['fold']}_seed_{seed}"
            path = run / "tables" / "validation_curve.csv"
            if not path.exists():
                continue
            c = pd.read_csv(path).drop_duplicates("timesteps", keep="last").assign(fold=fold["fold"], seed=seed)
            curves.append(c)
            best = c.loc[c["mean_reward"].idxmax()]
            ckpts.append({"fold": fold["fold"], "seed": seed, "n_train": len(fold["train"]),
                          "train": f"{fold['train'][0]}-{fold['train'][-1]}", "valid": f"{fold['valid'][0]}-{fold['valid'][-1]}",
                          "test": f"{fold['test'][0]}-{fold['test'][-1]}", "selected_timesteps": int(best["timesteps"]),
                          "valid_reward": best["mean_reward"], "valid_yield_kg_ha": best["mean_yield_kg_ha"],
                          "valid_irrigation_mm": best["mean_irrigation_mm"], "stop_timesteps": int(c["timesteps"].max()),
                          "n_evaluations": len(c)})
            s = _tensorboard_scalars(run)
            if not s.empty:
                scalars.append(s.assign(fold=fold["fold"], seed=seed))
    curves = pd.concat(curves, ignore_index=True)
    curves.to_csv(out / "tables" / "learning_validation_curves.csv", index=False)
    ckpts = pd.DataFrame(ckpts)
    ckpts.to_csv(out / "tables" / "selected_checkpoints.csv", index=False)
    scalars = pd.concat(scalars, ignore_index=True) if scalars else pd.DataFrame()
    if not scalars.empty:
        scalars.to_csv(out / "tables" / "ppo_training_scalars.csv", index=False)
        last = scalars.sort_values("step").groupby(["fold", "seed", "tag"])["value"].last().unstack("tag")
        first = scalars.sort_values("step").groupby(["fold", "seed", "tag"])["value"].first().unstack("tag")
        pd.concat({"first": first, "last": last}, axis=1).to_csv(out / "tables" / "ppo_diagnostics_first_last.csv")

    # Validation reference per fold: rules chosen on the training seasons of the fold, evaluated on its validation seasons.
    refs = []
    for fold in fold_list:
        tr = grid[grid["year"].isin(fold["train"]) & ~grid["candidate_id"].isin(CONSTRUCTED)]
        fid = tr.groupby("candidate_id")["reward"].mean().idxmax()
        va = grid[grid["year"].isin(fold["valid"])]
        oracle = va[~va["candidate_id"].isin(CONSTRUCTED)].groupby("year")["reward"].max().mean()
        refs.append({"fold": fold["fold"], "fixed_rule_valid": va[va["candidate_id"] == fid]["reward"].mean(),
                     "rainfed_oct15_valid": va[va["candidate_id"] == "rainfed_oct15"]["reward"].mean(), "oracle_valid": oracle})
    refs = pd.DataFrame(refs)
    refs.to_csv(out / "tables" / "validation_references_by_fold.csv", index=False)

    fig, axes = plt.subplots(1, len(fold_list), figsize=(11, 2.8), sharey=True)
    for ax, fold, color in zip(axes, fold_list, FOLD_SHADES):
        r = refs[refs["fold"] == fold["fold"]].iloc[0]
        for i, seed in enumerate(seeds):
            c = curves[(curves["fold"] == fold["fold"]) & (curves["seed"] == seed)]
            ax.plot(c["timesteps"] / 1000, c["mean_reward"], color=SEED_SHADES[i % len(SEED_SHADES)], marker="o", markersize=2, linewidth=1.2,
                    label=f"seed {seed}")
        ax.axhline(r["oracle_valid"], color=COLORS["retrospective_oracle"], linestyle=":", linewidth=1.0, label="Oracle")
        ax.axhline(r["rainfed_oct15_valid"], color=COLORS["rainfed_oct15"], linestyle="--", linewidth=1.0, label="Rainfed Oct 15")
        ax.axhline(r["fixed_rule_valid"], color=COLORS["optimized_fixed_rule"], linestyle="-.", linewidth=1.0, label="Fixed rule (train)")
        ax.set_title(f"Test {fold['test'][0]}-{fold['test'][-1]}\nvalid {fold['valid'][0]}-{fold['valid'][-1]}")
        ax.set_xlabel("Steps (thousands)")
    axes[0].set_ylabel("Validation objective")
    axes[-1].legend(frameon=False, fontsize=6.5, loc="lower right")
    _save(fig, out / "figures" / "fig_rolling_learning_curves.png")

    if not scalars.empty:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
        for ax, tag, title in [(axes[0], "rollout/ep_rew_mean", "Mean objective of training episodes"),
                               (axes[1], "train/std", "Policy standard deviation")]:
            for fold, color in zip(fold_list, FOLD_SHADES):
                for seed in seeds:
                    s = scalars[(scalars["tag"] == tag) & (scalars["fold"] == fold["fold"]) & (scalars["seed"] == seed)]
                    ax.plot(s["step"] / 1000, s["value"], color=color, linewidth=0.9,
                            label=f"test {fold['test'][0]}-{fold['test'][-1]}" if seed == seeds[0] else None)
            ax.set_title(title)
            ax.set_xlabel("Training steps (thousands)")
        axes[0].legend(frameon=False, fontsize=6.5)
        _save(fig, out / "figures" / "fig_rolling_training_diagnostics.png")

    # Season-level paired differences in the 20 test seasons.
    comp_order = ["rainfed_oct15", "optimized_fixed_rule", "contextual_knn", "retrospective_oracle"]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    years = sorted(ppo["year"].unique())
    width = 0.2
    for j, comp in enumerate(comp_order):
        g = paired[paired["comparator"] == comp].groupby("year")["d_yield"].mean()
        ax.bar(np.arange(len(years)) + (j - 1.5) * width, [g.get(y, np.nan) for y in years], width=width * 0.9,
               color=COLORS[comp], label=LABELS[comp])
    ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.set_xticks(range(len(years)), [str(y) for y in years], rotation=45)
    ax.set_xlabel("Test season (sowing year)")
    ax.set_ylabel("PPO minus comparator yield (kg ha$^{-1}$)")
    ax.legend(frameon=False, ncol=4, fontsize=7)
    _save(fig, out / "figures" / "fig_rolling_paired_yield.png")

    # Yield-water plane over the 20 test seasons.
    summ = pd.read_csv(tables / "test_summary.csv").set_index("comparator")
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    test_grid = grid[grid["year"].isin(years) & ~grid["candidate_id"].isin(CONSTRUCTED)].groupby("candidate_id")[["yield_kg_ha", "irrigation_mm"]].mean()
    ax.scatter(test_grid["irrigation_mm"], test_grid["yield_kg_ha"], s=6, color="#c3c2b7", label="Grid fixed rules (test means)")
    for comp, marker in [("rainfed_oct15", "s"), ("optimized_fixed_rule", "D"), ("contextual_knn", "^"), ("retrospective_oracle", "P")]:
        if comp in summ.index:
            ax.scatter(summ.loc[comp, "irrigation_mean_mm"], summ.loc[comp, "yield_mean_kg_ha"], marker=marker, s=46,
                       color=COLORS[comp], label=LABELS[comp], edgecolor="white", linewidth=0.6, zorder=4)
    seed_means = ppo.groupby("seed")[["yield_kg_ha", "irrigation_mm"]].mean()
    for i, (seed, r) in enumerate(seed_means.iterrows()):
        ax.scatter(r["irrigation_mm"], r["yield_kg_ha"], s=34, color=SEED_SHADES[i % len(SEED_SHADES)], edgecolor="white", linewidth=0.6,
                   zorder=5, label="PPO seeds" if i == 0 else None)
    ax.set_xlabel("Mean seasonal irrigation (mm)")
    ax.set_ylabel("Mean yield (kg ha$^{-1}$)")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    _save(fig, out / "figures" / "fig_rolling_yield_water.png")

    # Irrigation decisions: PPO versus oracle by season.
    irr = season_level.pivot(index="year", columns="comparator", values="irrigation_mm")
    irr.to_csv(out / "tables" / "irrigation_by_season.csv")
    gain = landscape.loc[years, "irrigation_gain_kg_ha"]
    fig, ax = plt.subplots(figsize=(10, 3.0))
    x = np.arange(len(years))
    for j, comp in enumerate(["PPO", "optimized_fixed_rule", "contextual_knn", "retrospective_oracle"]):
        ax.bar(x + (j - 1.5) * width, irr[comp].reindex(years), width=width * 0.9, color=COLORS[comp], label=LABELS[comp])
    ax.set_xticks(x, [f"{y}\n(+{g:.0f})" for y, g in zip(years, gain)], fontsize=7)
    ax.set_ylabel("Seasonal irrigation (mm)")
    ax.set_xlabel("Test season (irrigation gain of the best grid rule over the best rainfed rule, kg ha$^{-1}$)")
    ax.legend(frameon=False, ncol=4, fontsize=7)
    _save(fig, out / "figures" / "fig_rolling_irrigation_by_season.png")

    # Agreement between NASA POWER and INMET.
    agree_dir = outputs / "power_station_agreement" / "tables"
    if (agree_dir / "power_vs_station_season_rain.csv").exists():
        s = pd.read_csv(agree_dir / "power_vs_station_season_rain.csv")
        s = s[(s["valid_rain_day_share"] >= 0.9) & (s["days"] >= 240)]
        fig, ax = plt.subplots(figsize=(3.6, 3.4))
        ax.scatter(s["rain_station_mm"], s["rain_power_mm"], color=COLORS["PPO"], s=26)
        for r in s.itertuples():
            ax.annotate(str(r.season), (r.rain_station_mm, r.rain_power_mm), fontsize=6, xytext=(3, 2), textcoords="offset points")
        lo, hi = min(s["rain_station_mm"].min(), s["rain_power_mm"].min()) * 0.9, max(s["rain_station_mm"].max(), s["rain_power_mm"].max()) * 1.05
        ax.plot([lo, hi], [lo, hi], color="#52514e", linestyle="--", linewidth=0.8)
        ax.set_xlabel("INMET A819 rainfall, Sep-Apr (mm)")
        ax.set_ylabel("NASA POWER rainfall, Sep-Apr (mm)")
        _save(fig, out / "figures" / "fig_power_vs_inmet_season_rain.png")

    by_fold = (
        pd.concat([comps.assign(seed=np.nan), ppo.assign(comparator="PPO")], ignore_index=True)
        .merge(pd.DataFrame([{"year": y, "fold": f["fold"]} for f in fold_list for y in f["test"]]), on="year", suffixes=("_x", ""))
    )
    if "fold_x" in by_fold:
        by_fold = by_fold.drop(columns="fold_x")
    by_fold.groupby(["fold", "comparator"])[["reward", "yield_kg_ha", "irrigation_mm"]].mean().reset_index().to_csv(
        out / "tables" / "summary_by_fold.csv", index=False)
    print(ckpts.round(3).to_string(index=False))
    print(landscape.loc[years].round(0).to_string())


if __name__ == "__main__":
    main()
