from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_config, make_paths
from .data import build_year_weather_with_context, context_scaler, load_weather
from .evaluate import evaluate_policy
from .final_comparators import CONSTRUCTED, select_fixed_rule, with_kappa

KAPPAS = [0.0, 0.0005, 0.0015, 0.003, 0.006]
COLORS = {
    "PPO": "#2a78d6",
    "optimized_fixed_rule": "#eb6834",
    "contextual_knn": "#1baf7a",
    "rainfed_oct15": "#52514e",
    "retrospective_oracle": "#0b0b0b",
}
LABELS = {
    "PPO": "PPO",
    "optimized_fixed_rule": "Optimized fixed rule",
    "contextual_knn": "Contextual kNN rule",
    "rainfed_oct15": "Rainfed Oct 15",
    "irrigated_sep25_45_18_120": "Irrigated Sep 25 (45/18/120)",
    "irrigated_oct15_45_18_120": "Irrigated Oct 15 (45/18/120)",
    "irrigated_nov10_45_18_120": "Irrigated Nov 10 (45/18/120)",
    "retrospective_oracle": "Retrospective oracle",
}
SEED_SHADES = ["#0d3b73", "#1d5aa6", "#2a78d6", "#6aa3e6", "#a9c9f0"]


def _style():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e4e3df",
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.6,
            "savefig.dpi": 300,
        }
    )


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _cvar10(s: pd.Series) -> float:
    q = s.quantile(0.10)
    return float(s[s <= q].mean())


def _summary(df: pd.DataFrame, by: str) -> pd.DataFrame:
    return (
        df.groupby(by)
        .agg(
            reward_mean=("reward", "mean"),
            yield_mean_kg_ha=("yield_kg_ha", "mean"),
            yield_p10_kg_ha=("yield_kg_ha", lambda s: s.quantile(0.10)),
            yield_cvar10_kg_ha=("yield_kg_ha", _cvar10),
            irrigation_mean_mm=("irrigation_mm", "mean"),
            n=("year", "count"),
        )
        .reset_index()
    )


def _tensorboard_scalars(run_dir: Path) -> pd.DataFrame:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    rows = []
    for event_file in sorted((run_dir / "tensorboard").glob("PPO_*/events.out.tfevents.*")):
        acc = EventAccumulator(str(event_file))
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            for ev in acc.Scalars(tag):
                rows.append({"tag": tag, "step": ev.step, "value": ev.value})
    return pd.DataFrame(rows)


def learning_curves(outputs: Path, prefix: str, seeds: list[int], refs: dict, out: Path) -> pd.DataFrame:
    curves, scalars, selected = [], [], []
    for seed in seeds:
        run = outputs / f"{prefix}_seed_{seed}"
        c = pd.read_csv(run / "tables" / "validation_curve.csv").drop_duplicates("timesteps", keep="last").assign(seed=seed)
        curves.append(c)
        best = c.loc[c["mean_reward"].idxmax()]
        selected.append(
            {
                "seed": seed,
                "selected_timesteps": int(best["timesteps"]),
                "valid_reward": best["mean_reward"],
                "valid_yield_kg_ha": best["mean_yield_kg_ha"],
                "valid_irrigation_mm": best["mean_irrigation_mm"],
                "stop_timesteps": int(c["timesteps"].max()),
                "n_validation_evaluations": len(c),
                "simulation_requests": int(c["timesteps"].max() + 4 * len(c)),
            }
        )
        s = _tensorboard_scalars(run)
        if not s.empty:
            scalars.append(s.assign(seed=seed))
    curves = pd.concat(curves, ignore_index=True)
    curves.to_csv(out / "tables" / "learning_validation_curves.csv", index=False)
    selected = pd.DataFrame(selected)
    selected.to_csv(out / "tables" / "selected_checkpoints.csv", index=False)
    scalars = pd.concat(scalars, ignore_index=True) if scalars else pd.DataFrame()
    scalars.to_csv(out / "tables" / "ppo_training_scalars.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
    ax = axes[0]
    if not scalars.empty:
        for i, seed in enumerate(seeds):
            s = scalars[(scalars["tag"] == "rollout/ep_rew_mean") & (scalars["seed"] == seed)]
            ax.plot(s["step"] / 1000, s["value"], color=SEED_SHADES[i], label=f"seed {seed}")
    ax.set_xlabel("Training steps (thousands)")
    ax.set_ylabel("Mean episode objective, training seasons")
    ax.set_title("(a) Training seasons 2006-2016")
    ax.legend(frameon=False, ncol=1)
    ax = axes[1]
    for i, seed in enumerate(seeds):
        c = curves[curves["seed"] == seed]
        ax.plot(c["timesteps"] / 1000, c["mean_reward"], color=SEED_SHADES[i], marker="o", markersize=2.5)
        b = selected[selected["seed"] == seed].iloc[0]
        ax.plot(b["selected_timesteps"] / 1000, b["valid_reward"], marker="*", markersize=9, color=SEED_SHADES[i],
                markeredgecolor="white", markeredgewidth=0.6, zorder=5)
    for key, style in [("retrospective_oracle", ":"), ("rainfed_oct15", "--"), ("optimized_fixed_rule", "-.")]:
        if key in refs:
            ax.axhline(refs[key], color=COLORS[key], linestyle=style, linewidth=1.0, label=LABELS[key])
    ax.set_xlabel("Training steps (thousands)")
    ax.set_ylabel("Mean objective, validation seasons")
    ax.set_title("(b) Validation seasons 2017-2020")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, out / "figures" / "fig_learning_curves.png")

    if not scalars.empty:
        tags = [("train/std", "Policy standard deviation"), ("train/entropy_loss", "Entropy loss"),
                ("train/approx_kl", "Approximate KL divergence"), ("train/explained_variance", "Explained variance (critic)")]
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
        for ax, (tag, title) in zip(axes.ravel(), tags):
            for i, seed in enumerate(seeds):
                s = scalars[(scalars["tag"] == tag) & (scalars["seed"] == seed)]
                ax.plot(s["step"] / 1000, s["value"], color=SEED_SHADES[i], label=f"seed {seed}", linewidth=1.2)
            ax.set_title(title)
            ax.set_xlabel("Training steps (thousands)")
        axes[0, 0].legend(frameon=False)
        _save(fig, out / "figures" / "fig_ppo_diagnostics.png")
    return selected


def regularization_figure(outputs: Path, out: Path, refs: dict):
    runs = {"Without regularization (noise 0.25, [64, 64])": "dev_t0_seed42",
            "Final configuration (noise 0.75, [32, 32])": "ppo_castro_t0_seed_42"}
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for (label, run), color in zip(runs.items(), ["#eda100", "#2a78d6"]):
        path = outputs / run / "tables" / "validation_curve.csv"
        if path.exists():
            c = pd.read_csv(path).drop_duplicates("timesteps", keep="last")
            ax.plot(c["timesteps"] / 1000, c["mean_reward"], marker="o", markersize=2.5, color=color, label=label)
    for key, style in [("rainfed_oct15", "--"), ("optimized_fixed_rule", "-.")]:
        if key in refs:
            ax.axhline(refs[key], color=COLORS[key], linestyle=style, linewidth=1.0, label=LABELS[key])
    ax.set_xlabel("Training steps (thousands)")
    ax.set_ylabel("Mean objective, validation seasons")
    ax.legend(frameon=False, fontsize=7)
    _save(fig, out / "figures" / "fig_regularization_seed42.png")


def response_landscape(dev_grid: pd.DataFrame, out: Path):
    rainfed = dev_grid[dev_grid["candidate_id"].str.endswith("trig9999_amt0_cap0")]
    rng = rainfed.groupby("year")["yield_kg_ha"].agg(lambda s: s.max() - s.min())
    best_irr = dev_grid[~dev_grid["candidate_id"].isin(CONSTRUCTED)].groupby("year")["yield_kg_ha"].max()
    best_rf = rainfed.groupby("year")["yield_kg_ha"].max()
    table = pd.DataFrame({"rainfed_sowing_range_kg_ha": rng, "best_rainfed_kg_ha": best_rf,
                          "best_any_kg_ha": best_irr, "irrigation_gain_kg_ha": best_irr - best_rf})
    table.to_csv(out / "tables" / "development_response_landscape.csv")

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    years = sorted(rainfed["year"].unique())
    for y in years:
        s = rainfed[rainfed["year"] == y].sort_values("planting_offset_days")
        color = "#2a78d6" if y <= 2016 else "#eb6834"
        ax.plot(s["planting_offset_days"], s["yield_kg_ha"], color=color, alpha=0.55, linewidth=1.0)
    mean_tr = rainfed[rainfed["year"] <= 2016].groupby("planting_offset_days")["yield_kg_ha"].mean()
    mean_va = rainfed[rainfed["year"] > 2016].groupby("planting_offset_days")["yield_kg_ha"].mean()
    ax.plot(mean_tr.index, mean_tr.values, color="#2a78d6", linewidth=2.6, label="Training seasons (mean)")
    ax.plot(mean_va.index, mean_va.values, color="#eb6834", linewidth=2.6, label="Validation seasons (mean)")
    ax.set_xlabel("Sowing offset after September 15 (days)")
    ax.set_ylabel("Rainfed DSSAT yield (kg ha$^{-1}$)")
    ax.legend(frameon=False)
    _save(fig, out / "figures" / "fig_rainfed_sowing_response.png")
    return table


def context_skill(cfg: dict, project_dir: Path, dev_grid: pd.DataFrame, out: Path) -> pd.DataFrame:
    daily = load_weather(project_dir, cfg)
    train = build_year_weather_with_context(daily, cfg["data"]["train_years"], cfg)
    years = sorted(dev_grid["year"].unique())
    seasons = build_year_weather_with_context(daily, years, cfg)
    names = ["rain_30d", "rain_90d", "temp_30d", "srad_90d", "oni_mjj", "oni_mjj_minus_fma"]
    ctx = pd.DataFrame([s.features for s in seasons], columns=names)
    ctx["year"] = [s.year for s in seasons]
    rainfed = dev_grid[dev_grid["candidate_id"].str.endswith("trig9999_amt0_cap0")]
    best_offset = rainfed.loc[rainfed.groupby("year")["yield_kg_ha"].idxmax()].set_index("year")["planting_offset_days"]
    grid = dev_grid[~dev_grid["candidate_id"].isin(CONSTRUCTED)]
    irr_gain = grid.groupby("year")["yield_kg_ha"].max() - rainfed.groupby("year")["yield_kg_ha"].max()
    oct15 = dev_grid[dev_grid["candidate_id"] == "rainfed_oct15"].set_index("year")["yield_kg_ha"]
    targets = pd.DataFrame({"best_rainfed_offset_days": best_offset, "irrigation_gain_kg_ha": irr_gain, "rainfed_oct15_yield_kg_ha": oct15})
    merged = ctx.set_index("year").join(targets)
    rows = []
    for f in names:
        for t in targets.columns:
            rows.append({"context_variable": f, "target": t, "spearman_rho": merged[f].corr(merged[t], method="spearman"), "n_seasons": int(merged[[f, t]].dropna().shape[0])})
    table = pd.DataFrame(rows)
    table.to_csv(out / "tables" / "context_skill_development.csv", index=False)
    merged.to_csv(out / "tables" / "context_and_targets_development.csv")

    # Position of every season's context relative to the training seasons (standardised units).
    mean, std = context_scaler(train)
    all_years = sorted(set(years) | set(cfg["data"]["test_years"]))
    every = build_year_weather_with_context(daily, all_years, cfg)
    z = pd.DataFrame([(e.features - mean) / std for e in every], columns=names)
    z.insert(0, "year", [e.year for e in every])
    z_train = z[z["year"].isin(cfg["data"]["train_years"])]
    lo, hi = z_train[names].min(), z_train[names].max()
    feats = z[names].to_numpy()
    train_feats = z_train[names].to_numpy()
    z["n_features_outside_training_range"] = ((z[names] < lo) | (z[names] > hi)).sum(axis=1)
    z["distance_nearest_training_season"] = [
        float(np.min(np.linalg.norm(train_feats[z_train["year"].to_numpy() != y] - f, axis=1))) for y, f in zip(z["year"], feats)
    ]
    z["block"] = z["year"].map(lambda y: "train" if y in cfg["data"]["train_years"] else "valid" if y in cfg["data"]["valid_years"] else "test")
    z.to_csv(out / "tables" / "context_position_vs_training.csv", index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Tables and figures for the final Castro article results.")
    parser.add_argument("--config", default="configs/experiment_castro_t0.yaml")
    parser.add_argument("--run-prefix", default="ppo_castro_t0")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--comparators-run", default="final_comparators_castro_t0")
    parser.add_argument("--skip-train-eval", action="store_true")
    args = parser.parse_args()

    _style()
    cfg = load_config(args.config)
    paths = make_paths(cfg, "article_castro_t0")
    out = paths.output_dir
    outputs = paths.project_dir / "outputs"
    seeds = [int(s) for s in args.seeds.split(",")]
    comp_dir = outputs / args.comparators_run / "tables"
    dev_grid = pd.read_csv(comp_dir / "candidate_grid_development.csv")
    test_grid = pd.read_csv(comp_dir / "candidate_grid_test.csv")
    comp_valid = pd.read_csv(comp_dir / "comparator_rows_valid.csv")
    comp_test = pd.read_csv(comp_dir / "comparator_rows_test.csv")
    ppo = pd.read_csv(outputs / f"{args.run_prefix}_summary" / "policy_evaluation_all_seeds.csv")
    train_years, valid_years = cfg["data"]["train_years"], cfg["data"]["valid_years"]

    # In-sample evaluation of the selected checkpoints on the training seasons.
    train_rows = []
    for seed in seeds:
        seed_cfg = load_config(paths.project_dir / "configs" / f"_generated_{args.run_prefix}" / f"experiment_seed_{seed}.yaml")
        seed_paths = make_paths(seed_cfg, f"{args.run_prefix}_seed_{seed}")
        path = seed_paths.tables_dir / "policy_evaluation_train.csv"
        if not path.exists() and not args.skip_train_eval:
            evaluate_policy(seed_cfg, seed_paths, "train")
        if path.exists():
            train_rows.append(pd.read_csv(path).assign(seed=seed))
    ppo_all = pd.concat([ppo, *train_rows], ignore_index=True)
    ppo_all.to_csv(out / "tables" / "ppo_evaluation_train_valid_test.csv", index=False)

    # Validation references: rules chosen on the training seasons only.
    valid_summary = _summary(comp_valid, "comparator")
    refs_valid = valid_summary.set_index("comparator")["reward_mean"].to_dict()
    selected = learning_curves(outputs, args.run_prefix, seeds, refs_valid, out)
    regularization_figure(outputs, out, refs_valid)

    # Performance by split for PPO and the comparators.
    split_rows = []
    for split, years in [("train", train_years), ("valid", valid_years), ("test", cfg["data"]["test_years"])]:
        p = ppo_all[ppo_all["split"] == split]
        if not p.empty:
            per_seed = p.groupby("seed")[["reward", "yield_kg_ha", "irrigation_mm"]].mean()
            split_rows.append({"split": split, "policy": "PPO", "reward_mean": p["reward"].mean(), "reward_seed_min": per_seed["reward"].min(),
                               "reward_seed_max": per_seed["reward"].max(), "yield_mean_kg_ha": p["yield_kg_ha"].mean(),
                               "irrigation_mean_mm": p["irrigation_mm"].mean()})
        grid = dev_grid if split != "test" else test_grid
        grid = grid[grid["year"].isin(years)]
        sel_pool = dev_grid[dev_grid["year"].isin(train_years)] if split != "test" else dev_grid
        fixed_id = select_fixed_rule(sel_pool)
        for name, rows in [("optimized_fixed_rule", grid[grid["candidate_id"] == fixed_id]),
                           ("rainfed_oct15", grid[grid["candidate_id"] == "rainfed_oct15"]),
                           ("retrospective_oracle", grid[~grid["candidate_id"].isin(CONSTRUCTED)].sort_values("reward", ascending=False).drop_duplicates("year"))]:
            split_rows.append({"split": split, "policy": name, "rule": fixed_id if name == "optimized_fixed_rule" else "",
                               "reward_mean": rows["reward"].mean(), "yield_mean_kg_ha": rows["yield_kg_ha"].mean(),
                               "irrigation_mean_mm": rows["irrigation_mm"].mean()})
    by_split = pd.DataFrame(split_rows)
    by_split.to_csv(out / "tables" / "performance_by_split.csv", index=False)

    # Test block: comparator summary, paired effects, decomposition, season-level effects.
    ppo_test = ppo[ppo["split"] == "test"].copy()
    test_rows = pd.concat([comp_test[["comparator", "year", "candidate_id", "planting_date", "trigger_dryness", "amount_mm",
                                      "max_irrigation_mm", "yield_kg_ha", "irrigation_mm", "reward"]],
                           ppo_test.assign(comparator="PPO")], ignore_index=True)
    test_summary = _summary(test_rows, "comparator")
    seed_means = ppo_test.groupby("seed")[["reward", "yield_kg_ha", "irrigation_mm"]].mean()
    test_summary["seed_sd_reward"] = np.where(test_summary["comparator"] == "PPO", seed_means["reward"].std(ddof=1), np.nan)
    test_summary["seed_sd_yield_kg_ha"] = np.where(test_summary["comparator"] == "PPO", seed_means["yield_kg_ha"].std(ddof=1), np.nan)
    test_summary.to_csv(out / "tables" / "test_comparator_summary.csv", index=False)

    season = test_rows.groupby(["comparator", "year"])[["yield_kg_ha", "irrigation_mm", "reward"]].mean().reset_index()
    season.to_csv(out / "tables" / "test_season_level.csv", index=False)
    wide = season.pivot(index="year", columns="comparator", values="reward")
    wide_y = season.pivot(index="year", columns="comparator", values="yield_kg_ha")

    s = test_summary.set_index("comparator")
    base, fixed, knn, ppo_r, oracle = (s.loc[k, "reward_mean"] for k in ["rainfed_oct15", "optimized_fixed_rule", "contextual_knn", "PPO", "retrospective_oracle"])
    decomposition = pd.DataFrame(
        [
            {"component": "fixed rule vs calendar (optimized fixed rule - rainfed Oct 15)", "d_reward": fixed - base,
             "d_yield_kg_ha": s.loc["optimized_fixed_rule", "yield_mean_kg_ha"] - s.loc["rainfed_oct15", "yield_mean_kg_ha"]},
            {"component": "context (contextual kNN - optimized fixed rule)", "d_reward": knn - fixed,
             "d_yield_kg_ha": s.loc["contextual_knn", "yield_mean_kg_ha"] - s.loc["optimized_fixed_rule", "yield_mean_kg_ha"]},
            {"component": "algorithm (PPO - contextual kNN)", "d_reward": ppo_r - knn,
             "d_yield_kg_ha": s.loc["PPO", "yield_mean_kg_ha"] - s.loc["contextual_knn", "yield_mean_kg_ha"]},
            {"component": "adaptation (PPO - optimized fixed rule)", "d_reward": ppo_r - fixed,
             "d_yield_kg_ha": s.loc["PPO", "yield_mean_kg_ha"] - s.loc["optimized_fixed_rule", "yield_mean_kg_ha"]},
            {"component": "remaining regret (oracle - PPO)", "d_reward": oracle - ppo_r,
             "d_yield_kg_ha": s.loc["retrospective_oracle", "yield_mean_kg_ha"] - s.loc["PPO", "yield_mean_kg_ha"]},
            {"component": "share of oracle gap over rainfed Oct 15 captured by PPO", "d_reward": (ppo_r - base) / (oracle - base) if oracle > base else np.nan, "d_yield_kg_ha": np.nan},
            {"component": "share of oracle gap over rainfed Oct 15 captured by optimized fixed rule", "d_reward": (fixed - base) / (oracle - base) if oracle > base else np.nan, "d_yield_kg_ha": np.nan},
        ]
    )
    decomposition.to_csv(out / "tables" / "test_value_decomposition.csv", index=False)

    # Regret per season under the same objective (oracle - policy).
    regret = (wide["retrospective_oracle"].values[:, None] - wide.drop(columns="retrospective_oracle")).reset_index()
    regret.to_csv(out / "tables" / "test_regret_by_season.csv", index=False)

    # Variance components of the paired PPO effect: season and seed means.
    paired = pd.read_csv(comp_dir / "test_paired_effects_seed_year.csv")
    comp_list = ["optimized_fixed_rule", "contextual_knn", "rainfed_oct15", "retrospective_oracle"]
    rows = []
    for comp in comp_list:
        g = paired[paired["comparator"] == comp]
        if g.empty:
            continue
        mu = g["d_yield"].mean()
        season_eff = g.groupby("year")["d_yield"].mean() - mu
        seed_eff = g.groupby("seed")["d_yield"].mean() - mu
        resid = g["d_yield"] - mu - g["year"].map(season_eff) - g["seed"].map(seed_eff)
        rows.append({"comparator": comp, "mu_d_yield": mu, "sd_season_effect": season_eff.std(ddof=1), "sd_seed_effect": seed_eff.std(ddof=1),
                     "sd_residual": resid.std(ddof=1), "mu_d_reward": g["d_reward"].mean(),
                     "mu_d_irrigation": g["d_irrigation"].mean(), "seasons_positive_d_reward": int((g.groupby("year")["d_reward"].mean() > 0).sum()),
                     "seed_season_pairs_positive_d_reward": int((g["d_reward"] > 0).sum()), "n_pairs": len(g)})
    pd.DataFrame(rows).to_csv(out / "tables" / "test_paired_effect_components.csv", index=False)

    # Actions in the test block.
    acts = ppo_test.groupby("year").agg(
        sowing_first=("planting_date", "min"), sowing_last=("planting_date", "max"),
        trigger_mean=("trigger_dryness", "mean"), depth_mean=("amount_mm", "mean"), cap_mean=("max_irrigation_mm", "mean"),
        irrigation_mean=("irrigation_mm", "mean"), irrigation_max=("irrigation_mm", "max"), yield_mean=("yield_kg_ha", "mean"),
    ).reset_index()
    acts.to_csv(out / "tables" / "test_ppo_actions_by_season.csv", index=False)
    ppo_all.groupby(["split", "seed"]).agg(
        sowing_doy_sd=("planting_date", lambda s: pd.to_datetime(s).dt.dayofyear.std()),
        irrigated_share=("irrigation_mm", lambda s: (s > 0).mean()),
        trigger_at_bound=("trigger_dryness", lambda s: ((s <= 10.5) | (s >= 149.5)).mean()),
        depth_below_event_min=("amount_mm", lambda s: (s < cfg["agronomy"]["min_irrigation_event_mm"]).mean()),
    ).reset_index().to_csv(out / "tables" / "ppo_action_diagnostics.csv", index=False)
    comp_test[comp_test["comparator"].isin(["optimized_fixed_rule", "contextual_knn", "retrospective_oracle"])][
        ["comparator", "year", "candidate_id", "planting_date", "irrigation_mm", "yield_kg_ha", "reward"]
    ].to_csv(out / "tables" / "test_comparator_actions.csv", index=False)

    # Water penalty sensitivity: fixed rule re-optimised on development seasons for each kappa; PPO points re-scored.
    krows = []
    for k in KAPPAS:
        dev_k, test_k = with_kappa(dev_grid, cfg, k), with_kappa(test_grid, cfg, k)
        fid = select_fixed_rule(dev_k)
        f = test_k[test_k["candidate_id"] == fid]
        o = test_k[~test_k["candidate_id"].isin(CONSTRUCTED)].sort_values("reward", ascending=False).drop_duplicates("year")
        p = ppo_test.assign(reward=ppo_test["yield_kg_ha"] / cfg["reward"]["target_yield_kg_ha"] - k * ppo_test["irrigation_mm"])
        krows += [
            {"kappa": k, "policy": "optimized_fixed_rule", "rule": fid, "reward": f["reward"].mean(), "yield_kg_ha": f["yield_kg_ha"].mean(), "irrigation_mm": f["irrigation_mm"].mean()},
            {"kappa": k, "policy": "retrospective_oracle", "rule": "", "reward": o["reward"].mean(), "yield_kg_ha": o["yield_kg_ha"].mean(), "irrigation_mm": o["irrigation_mm"].mean()},
            {"kappa": k, "policy": "PPO (trained at 0.0015, re-scored)", "rule": "", "reward": p["reward"].mean(), "yield_kg_ha": p["yield_kg_ha"].mean(), "irrigation_mm": p["irrigation_mm"].mean()},
        ]
    pd.DataFrame(krows).to_csv(out / "tables" / "test_kappa_sensitivity.csv", index=False)

    landscape = response_landscape(dev_grid, out)
    context_skill(cfg, paths.project_dir, dev_grid, out)

    # Figure: season-level paired yield differences on the test block.
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    comps = ["rainfed_oct15", "optimized_fixed_rule", "contextual_knn", "retrospective_oracle"]
    width = 0.18
    years = sorted(ppo_test["year"].unique())
    for j, comp in enumerate(comps):
        g = paired[paired["comparator"] == comp]
        for i, y in enumerate(years):
            v = g[g["year"] == y]["d_yield"]
            x = i + (j - 1.5) * width
            ax.scatter(np.full(len(v), x), v, s=14, color=COLORS[comp], alpha=0.75, edgecolor="white", linewidth=0.4,
                       label=LABELS[comp] if i == 0 else None, zorder=3)
            ax.plot([x - width * 0.35, x + width * 0.35], [v.mean()] * 2, color=COLORS[comp], linewidth=2.0)
    ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.set_xticks(range(len(years)), [str(y) for y in years])
    ax.set_xlabel("Test season (sowing year)")
    ax.set_ylabel("PPO minus comparator yield (kg ha$^{-1}$)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    _save(fig, out / "figures" / "fig_test_paired_yield.png")

    # Figure: yield-water points on the test block.
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    fixed_rules = test_grid[~test_grid["candidate_id"].isin(CONSTRUCTED)].groupby("candidate_id")[["yield_kg_ha", "irrigation_mm"]].mean()
    ax.scatter(fixed_rules["irrigation_mm"], fixed_rules["yield_kg_ha"], s=6, color="#c3c2b7", label="Grid fixed rules (test means)", zorder=1)
    for comp, marker in [("rainfed_oct15", "s"), ("optimized_fixed_rule", "D"), ("contextual_knn", "^"), ("retrospective_oracle", "P")]:
        r = s.loc[comp]
        ax.scatter(r["irrigation_mean_mm"], r["yield_mean_kg_ha"], marker=marker, s=46, color=COLORS[comp], label=LABELS[comp], edgecolor="white", linewidth=0.6, zorder=4)
    for i, (seed, r) in enumerate(seed_means.iterrows()):
        ax.scatter(r["irrigation_mm"], r["yield_kg_ha"], s=34, color=SEED_SHADES[i], edgecolor="white", linewidth=0.6, zorder=5,
                   label="PPO seeds" if i == 0 else None)
    for c in CONSTRUCTED:
        if c.startswith("irrigated") and c in s.index:
            ax.scatter(s.loc[c, "irrigation_mean_mm"], s.loc[c, "yield_mean_kg_ha"], marker="x", s=30, color="#52514e", zorder=3,
                       label="Constructed irrigated schedules" if c.startswith("irrigated_sep") else None)
    ax.set_xlabel("Mean seasonal irrigation (mm)")
    ax.set_ylabel("Mean yield (kg ha$^{-1}$)")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    _save(fig, out / "figures" / "fig_test_yield_water.png")

    # Figure: sowing dates selected in the test block.
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for comp, marker, dx in [("optimized_fixed_rule", "D", -0.2), ("contextual_knn", "^", 0.0), ("retrospective_oracle", "P", 0.2)]:
        g = comp_test[comp_test["comparator"] == comp].sort_values("year")
        ax.scatter([years.index(y) + dx for y in g["year"]], pd.to_datetime(g["planting_date"]).dt.dayofyear, marker=marker, s=40,
                   color=COLORS[comp], label=LABELS[comp], edgecolor="white", linewidth=0.5, zorder=3)
    doy = pd.to_datetime(ppo_test["planting_date"]).dt.dayofyear
    ax.scatter([years.index(y) + 0.4 for y in ppo_test["year"]], doy, s=18, color=COLORS["PPO"], alpha=0.8, label="PPO (one point per seed)", zorder=4)
    ticks = [258, 274, 288, 305, 319, 335, 349]
    ax.set_yticks(ticks, ["Sep 15", "Oct 1", "Oct 15", "Nov 1", "Nov 15", "Dec 1", "Dec 15"])
    ax.set_xticks(range(len(years)), [str(y) for y in years])
    ax.set_xlabel("Test season (sowing year)")
    ax.set_ylabel("Selected sowing date")
    ax.legend(frameon=False, fontsize=7)
    _save(fig, out / "figures" / "fig_test_sowing_dates.png")

    print(by_split.to_string(index=False))
    print(test_summary.to_string(index=False))
    print(decomposition.to_string(index=False))
    print(selected.to_string(index=False))
    print(landscape.describe().round(1).to_string())


if __name__ == "__main__":
    main()
