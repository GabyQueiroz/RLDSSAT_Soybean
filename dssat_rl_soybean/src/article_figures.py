"""Final article figures: one PDF per panel, one font size, no titles above the axes.

All figures are redrawn from the result tables already stored in outputs/ (no simulation is repeated).
Panels that were combined in a single image are written as separate files with the suffixes _a, _b, ...,
and a shared legend, when needed, is written as <name>_legend.pdf.

    python -m src.article_figures --out ../figuras_finais
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .article_results import COLORS, LABELS, SEED_SHADES
from .config import load_config, make_paths
from .data import load_weather
from .final_comparators import CONSTRUCTED
from .rolling_extra_results import _contexts
from .rolling_origin import folds

FONT = 18
FOLD_SHADES = ["#0d3b73", "#1d5aa6", "#2a78d6", "#6aa3e6", "#a9c9f0"]
# Sizes are chosen so that every figure is included in the article with the same scale (\figscale = 0.43,
# printed font of about 7.7 pt): two panels fit side by side and a wide figure fits the text width.
PANEL = (5.4, 4.2)       # one panel of a multi-panel figure
SINGLE = (7.2, 5.0)      # stand-alone figure
WIDE = (12.4, 5.6)       # season-by-season bar charts


def _style() -> None:
    plt.rcParams.update({
        "font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT, "xtick.labelsize": FONT,
        "ytick.labelsize": FONT, "legend.fontsize": FONT, "legend.title_fontsize": FONT, "figure.titlesize": FONT,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
        "grid.linewidth": 0.8, "lines.linewidth": 2.2, "lines.markersize": 6, "pdf.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    })


def _save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def _legend_file(handles, labels, out: Path, name: str, ncol: int) -> None:
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.legend(handles, labels, loc="center", ncol=ncol, frameon=False)
    _save(fig, out, name)


def _legend_above(ax, ncol: int, **kw) -> None:
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=ncol, frameon=False, borderaxespad=0.0, **kw)


def _valid_refs_exp1(outputs: Path) -> dict:
    rows = pd.read_csv(outputs / "final_comparators_castro_t0" / "tables" / "comparator_rows_valid.csv", encoding="utf-8-sig")
    return rows.groupby("comparator")["reward"].mean().to_dict()


# ------------------------------------------------------------------ Experiment 1
def exp1_figures(outputs: Path, out: Path) -> None:
    tab = outputs / "article_castro_t0" / "tables"
    curves = pd.read_csv(tab / "learning_validation_curves.csv")
    selected = pd.read_csv(tab / "selected_checkpoints.csv")
    scalars = pd.read_csv(tab / "ppo_training_scalars.csv")
    seeds = sorted(curves["seed"].unique())
    refs = _valid_refs_exp1(outputs)
    ref_styles = [("retrospective_oracle", ":"), ("rainfed_oct15", "--"), ("optimized_fixed_rule", "-.")]

    # fig_learning_curves: (a) training episodes, (b) validation objective; shared legend
    fig, ax = plt.subplots(figsize=PANEL)
    for i, seed in enumerate(seeds):
        s = scalars[(scalars["tag"] == "rollout/ep_rew_mean") & (scalars["seed"] == seed)]
        ax.plot(s["step"] / 1000, s["value"], color=SEED_SHADES[i])
    ax.set_xlabel("Training steps (thousands)")
    ax.set_ylabel("Mean episode objective, training")
    _save(fig, out, "fig_learning_curves_a")

    fig, ax = plt.subplots(figsize=PANEL)
    for i, seed in enumerate(seeds):
        c = curves[curves["seed"] == seed]
        ax.plot(c["timesteps"] / 1000, c["mean_reward"], color=SEED_SHADES[i], marker="o", markersize=4)
        b = selected[selected["seed"] == seed].iloc[0]
        ax.plot(b["selected_timesteps"] / 1000, b["valid_reward"], marker="*", markersize=18, color=SEED_SHADES[i],
                markeredgecolor="white", markeredgewidth=0.8, zorder=5)
    for key, ls in ref_styles:
        ax.axhline(refs[key], color=COLORS[key], linestyle=ls, linewidth=1.6)
    ax.set_xlabel("Training steps (thousands)")
    ax.set_ylabel("Mean objective, validation")
    _save(fig, out, "fig_learning_curves_b")

    handles = [Line2D([], [], color=SEED_SHADES[i]) for i in range(len(seeds))]
    labels = [f"seed {s}" for s in seeds]
    handles += [Line2D([], [], color=SEED_SHADES[2], marker="*", markersize=18, linestyle="none", markeredgecolor="white")]
    labels += ["Selected checkpoint"]
    handles += [Line2D([], [], color=COLORS[k], linestyle=ls, linewidth=1.6) for k, ls in ref_styles]
    labels += [LABELS[k] for k, _ in ref_styles]
    _legend_file(handles, labels, out, "fig_learning_curves_legend", ncol=3)

    # fig_ppo_diagnostics: four panels and a legend
    tags = [("train/std", "Policy standard deviation"), ("train/entropy_loss", "Entropy loss"),
            ("train/approx_kl", "Approximate KL divergence"), ("train/explained_variance", "Explained variance")]
    for letter, (tag, ylabel) in zip("abcd", tags):
        fig, ax = plt.subplots(figsize=PANEL)
        for i, seed in enumerate(seeds):
            s = scalars[(scalars["tag"] == tag) & (scalars["seed"] == seed)]
            ax.plot(s["step"] / 1000, s["value"], color=SEED_SHADES[i], linewidth=1.8)
        ax.set_xlabel("Training steps (thousands)")
        ax.set_ylabel(ylabel)
        _save(fig, out, f"fig_ppo_diagnostics_{letter}")
    _legend_file([Line2D([], [], color=SEED_SHADES[i]) for i in range(len(seeds))], [f"seed {s}" for s in seeds],
                 out, "fig_ppo_diagnostics_legend", ncol=5)

    # fig_regularization_seed42: legend above the axes
    runs = {"Without regularization (noise 0.25, [64, 64])": ("dev_t0_seed42", "#eda100"),
            "Final configuration (noise 0.75, [32, 32])": ("ppo_castro_t0_seed_42", "#2a78d6")}
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for label, (run, color) in runs.items():
        c = pd.read_csv(outputs / run / "tables" / "validation_curve.csv").drop_duplicates("timesteps", keep="last")
        ax.plot(c["timesteps"] / 1000, c["mean_reward"], marker="o", markersize=5, color=color, label=label)
    for key, ls in [("rainfed_oct15", "--"), ("optimized_fixed_rule", "-.")]:
        ax.axhline(refs[key], color=COLORS[key], linestyle=ls, linewidth=1.6, label=LABELS[key])
    ax.set_xlabel("Training steps (thousands)")
    ax.set_ylabel("Mean objective, validation")
    _legend_above(ax, ncol=1)
    _save(fig, out, "fig_regularization_seed42")

    # fig_rainfed_sowing_response: legend in the lower-left corner
    dev = pd.read_csv(outputs / "final_comparators_castro_t0" / "tables" / "candidate_grid_development.csv", encoding="utf-8-sig")
    rainfed = dev[dev["candidate_id"].str.endswith("trig9999_amt0_cap0")]
    fig, ax = plt.subplots(figsize=SINGLE)
    for y in sorted(rainfed["year"].unique()):
        s = rainfed[rainfed["year"] == y].sort_values("planting_offset_days")
        ax.plot(s["planting_offset_days"], s["yield_kg_ha"], color="#2a78d6" if y <= 2016 else "#eb6834", alpha=0.5, linewidth=1.2)
    for years, color, label in [(rainfed["year"] <= 2016, "#2a78d6", "Training seasons (mean)"),
                                (rainfed["year"] > 2016, "#eb6834", "Validation seasons (mean)")]:
        m = rainfed[years].groupby("planting_offset_days")["yield_kg_ha"].mean()
        ax.plot(m.index, m.values, color=color, linewidth=3.5, label=label)
    ax.set_xlabel("Sowing offset after September 15 (days)")
    ax.set_ylabel("Rainfed DSSAT yield (kg ha$^{-1}$)")
    ax.legend(loc="lower left", frameon=False)
    _save(fig, out, "fig_rainfed_sowing_response")


# ------------------------------------------------------------------ Experiment 2
def exp2_figures(cfg: dict, project: Path, outputs: Path, out: Path) -> None:
    art = outputs / "article_rolling_power" / "tables"
    ana = outputs / "rolling_power_analysis" / "tables"
    curves = pd.read_csv(art / "learning_validation_curves.csv")
    scalars = pd.read_csv(art / "ppo_training_scalars.csv")
    refs = pd.read_csv(art / "validation_references_by_fold.csv")
    fold_list = folds(cfg)
    seeds = sorted(curves["seed"].unique())

    # fig_rolling_learning_curves: one file per fold, same y range, shared legend
    ylo, yhi = curves["mean_reward"].min(), max(curves["mean_reward"].max(), refs["oracle_valid"].max())
    pad = 0.05 * (yhi - ylo)
    for letter, fold in zip("abcde", fold_list):
        r = refs[refs["fold"] == fold["fold"]].iloc[0]
        fig, ax = plt.subplots(figsize=PANEL)
        for i, seed in enumerate(seeds):
            c = curves[(curves["fold"] == fold["fold"]) & (curves["seed"] == seed)]
            ax.plot(c["timesteps"] / 1000, c["mean_reward"], color=SEED_SHADES[i % len(SEED_SHADES)], marker="o", markersize=4, linewidth=1.8)
        ax.axhline(r["oracle_valid"], color=COLORS["retrospective_oracle"], linestyle=":", linewidth=1.6)
        ax.axhline(r["rainfed_oct15_valid"], color=COLORS["rainfed_oct15"], linestyle="--", linewidth=1.6)
        ax.axhline(r["fixed_rule_valid"], color=COLORS["optimized_fixed_rule"], linestyle="-.", linewidth=1.6)
        ax.set_ylim(ylo - pad, yhi + pad)
        ax.set_xlabel("Training steps (thousands)")
        ax.set_ylabel("Validation objective")
        _save(fig, out, f"fig_rolling_learning_curves_{letter}")
    handles = [Line2D([], [], color=SEED_SHADES[i % len(SEED_SHADES)], marker="o") for i in range(len(seeds))]
    labels = [f"seed {s}" for s in seeds]
    handles += [Line2D([], [], color=COLORS["retrospective_oracle"], linestyle=":"),
                Line2D([], [], color=COLORS["rainfed_oct15"], linestyle="--"),
                Line2D([], [], color=COLORS["optimized_fixed_rule"], linestyle="-.")]
    labels += ["Oracle", "Rainfed Oct 15", "Fixed rule (training seasons)"]
    _legend_file(handles, labels, out, "fig_rolling_learning_curves_legend", ncol=3)

    # fig_rolling_training_diagnostics: two files and a separate legend
    for letter, (tag, ylabel) in zip("ab", [("rollout/ep_rew_mean", "Mean objective, training episodes"),
                                            ("train/std", "Policy standard deviation")]):
        fig, ax = plt.subplots(figsize=PANEL)
        for fold, color in zip(fold_list, FOLD_SHADES):
            for seed in seeds:
                s = scalars[(scalars["tag"] == tag) & (scalars["fold"] == fold["fold"]) & (scalars["seed"] == seed)]
                ax.plot(s["step"] / 1000, s["value"], color=color, linewidth=1.3)
        ax.set_xlabel("Training steps (thousands)")
        ax.set_ylabel(ylabel)
        _save(fig, out, f"fig_rolling_training_diagnostics_{letter}")
    _legend_file([Line2D([], [], color=c, linewidth=3) for c in FOLD_SHADES],
                 [f"Test {f['test'][0]}-{f['test'][-1]}" for f in fold_list], out, "fig_rolling_training_diagnostics_legend", ncol=3)

    # fig_rolling_paired_yield: legend above the axes, outside the plot
    paired = pd.read_csv(ana / "paired_effects_seed_season.csv")
    ppo = pd.read_csv(ana / "ppo_rows_test.csv")
    years = sorted(ppo["year"].unique())
    width = 0.2
    fig, ax = plt.subplots(figsize=WIDE)
    for j, comp in enumerate(["rainfed_oct15", "optimized_fixed_rule", "contextual_knn", "retrospective_oracle"]):
        g = paired[paired["comparator"] == comp].groupby("year")["d_yield"].mean()
        ax.bar(np.arange(len(years)) + (j - 1.5) * width, [g.get(y, np.nan) for y in years], width=width * 0.9,
               color=COLORS[comp], label=LABELS[comp])
    ax.axhline(0, color="#52514e", linewidth=1.0)
    ax.set_xticks(range(len(years)), [str(y) for y in years], rotation=45)
    ax.set_xlabel("Test season (sowing year)")
    ax.set_ylabel("PPO minus comparator (kg ha$^{-1}$)")
    _legend_above(ax, ncol=2)
    _save(fig, out, "fig_rolling_paired_yield")

    # fig_rolling_irrigation_by_season: legend above, centred, outside the plot
    season_level = pd.read_csv(ana / "season_level.csv")
    irr = season_level.pivot(index="year", columns="comparator", values="irrigation_mm")
    land = pd.read_csv(art / "response_landscape_all_seasons.csv", index_col=0)
    gain = land.loc[years, "irrigation_gain_kg_ha"]
    fig, ax = plt.subplots(figsize=WIDE)
    x = np.arange(len(years))
    for j, comp in enumerate(["PPO", "optimized_fixed_rule", "contextual_knn", "retrospective_oracle"]):
        ax.bar(x + (j - 1.5) * width, irr[comp].reindex(years), width=width * 0.9, color=COLORS[comp], label=LABELS[comp])
    ax.set_xticks(x, [f"{y} (+{g:.0f})" for y, g in zip(years, gain)], rotation=60, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Seasonal irrigation (mm)")
    ax.set_xlabel("Test season (irrigation gain, kg ha$^{-1}$)")
    _legend_above(ax, ncol=2)
    _save(fig, out, "fig_rolling_irrigation_by_season")

    # fig_rolling_context_action: three files, colour bar above each panel
    grid_ppo = ppo
    daily = load_weather(project, cfg)
    ctx = pd.concat([_contexts(cfg, daily, f, f["test"]).assign(fold=f["fold"]) for f in fold_list], ignore_index=True)
    act = grid_ppo.merge(ctx, on="year")
    vmax = max(1, act["irrigation_mm"].max())
    ymin, ymax = act["planting_offset_days"].min() - 3, act["planting_offset_days"].max() + 3
    for letter, (feat, label) in zip("abc", [("rain_90d", "90-day rainfall (standardized)"), ("oni_mjj", "ONI May-July (standardized)"),
                                             ("temp_30d", "30-day temperature (standardized)")]):
        fig, ax = plt.subplots(figsize=(5.4, 5.2))
        sc = ax.scatter(act[feat], act["planting_offset_days"], c=act["irrigation_mm"], cmap="Blues", s=70, edgecolor="#52514e",
                        linewidth=0.5, vmin=0, vmax=vmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(label)
        ax.set_ylabel("Sowing offset after Sep 15 (days)")
        cb = fig.colorbar(sc, ax=ax, orientation="horizontal", location="top", fraction=0.06, pad=0.04)
        cb.set_label("Seasonal irrigation (mm)")
        cb.ax.xaxis.set_label_position("top")
        _save(fig, out, f"fig_rolling_context_action_{letter}")

    # fig_rolling_water_frontier
    front = pd.read_csv(art / "water_frontier_by_kappa.csv")
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    styles = {"optimized fixed rule": ("D", COLORS["optimized_fixed_rule"], "Optimized fixed rule"),
              "retrospective oracle": ("P", COLORS["retrospective_oracle"], "Retrospective oracle"),
              "PPO retrained (seed 42)": ("o", COLORS["PPO"], "PPO retrained (seed 42)")}
    offsets = {("optimized fixed rule", 0.006): (30, 14), ("PPO retrained (seed 42)", 0.006): (30, -16),
               ("optimized fixed rule", 0.0): (-30, -28), ("retrospective oracle", 0.0): (-30, -28)}
    for pol, (marker, color, label) in styles.items():
        g = front[front["policy"] == pol].sort_values("kappa")
        ax.plot(g["irrigation_mm"], g["yield_kg_ha"], marker=marker, color=color, label=label, markersize=10)
        for r in g.itertuples():
            if r.kappa in (0.0, 0.0015, 0.006):
                ax.annotate(f"$\\kappa$={r.kappa:g}", (r.irrigation_mm, r.yield_kg_ha), xytext=offsets.get((pol, r.kappa), (10, -22)),
                            textcoords="offset points")
    r0 = front[front["policy"] == "rainfed Oct 15"].iloc[0]
    ax.scatter([0], [r0["yield_kg_ha"]], marker="s", s=110, color=COLORS["rainfed_oct15"], label="Rainfed Oct 15", zorder=4)
    ax.set_xlabel("Mean seasonal irrigation (mm)")
    ax.set_ylabel("Mean yield (kg ha$^{-1}$)")
    ax.set_xlim(-8, 135)
    ax.legend(loc="lower right", frameon=False)
    _save(fig, out, "fig_rolling_water_frontier")

    # fig_power_vs_inmet_season_rain
    s = pd.read_csv(outputs / "power_station_agreement" / "tables" / "power_vs_station_season_rain.csv")
    s = s[(s["valid_rain_day_share"] >= 0.9) & (s["days"] >= 240)]
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    ax.scatter(s["rain_station_mm"], s["rain_power_mm"], color=COLORS["PPO"], s=80, zorder=3)
    shift = {2018: (8, -20), 2023: (-62, 6), 2011: (-62, 6)}
    for r in s.itertuples():
        ax.annotate(str(r.season), (r.rain_station_mm, r.rain_power_mm), xytext=shift.get(r.season, (8, 4)), textcoords="offset points")
    lo = min(s["rain_station_mm"].min(), s["rain_power_mm"].min()) * 0.9
    hi = max(s["rain_station_mm"].max(), s["rain_power_mm"].max()) * 1.05
    ax.plot([lo, hi], [lo, hi], color="#52514e", linestyle="--", linewidth=1.2)
    ax.set_xlabel("INMET A819 rainfall, Sep-Apr (mm)")
    ax.set_ylabel("NASA POWER rainfall, Sep-Apr (mm)")
    _save(fig, out, "fig_power_vs_inmet_season_rain")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/experiment_castro_power.yaml")
    parser.add_argument("--out", default="../figuras_finais")
    args = parser.parse_args()
    _style()
    cfg = load_config(args.config)
    paths = make_paths(cfg, "article_rolling_power")
    project = paths.project_dir
    outputs = project / "outputs"
    out = Path(args.out)
    if not out.is_absolute():
        out = (project / out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    exp1_figures(outputs, out)
    exp2_figures(cfg, project, outputs, out)
    print("figures written to", out)


if __name__ == "__main__":
    main()
