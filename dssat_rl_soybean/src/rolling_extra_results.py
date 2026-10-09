"""Additional results of the rolling-origin evaluation requested by the methodological review.

- action-bound saturation and context-to-action map of the PPO policies
- value-of-information ladder (fixed rule, kNN on subsets of the context, full kNN, PPO, oracle)
- productivity-water frontier over the water penalty, with PPO retrained at each penalty when available
- yield regret under a common seasonal water budget
- three-seed versus five-seed summaries
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .article_results import COLORS, LABELS, _save, _style
from .config import load_config, make_paths
from .data import build_year_weather_with_context, context_scaler, load_weather
from .final_comparators import CONSTRUCTED, select_fixed_rule, with_kappa
from .rolling_origin import folds

CONTEXT_NAMES = ["rain_30d", "rain_90d", "temp_30d", "srad_90d", "oni_mjj", "oni_change"]
KAPPAS = {"rolling_power_k0": 0.0, "rolling_power": 0.0015, "rolling_power_k6": 0.006}
BUDGETS = [0, 25, 50, 100, 260]


def _contexts(cfg, daily, fold, years):
    mean, std = context_scaler(build_year_weather_with_context(daily, fold["train"], cfg))
    seasons = build_year_weather_with_context(daily, years, cfg)
    z = np.nan_to_num(np.vstack([(s.features - mean) / std for s in seasons]), nan=0.0)
    out = pd.DataFrame(z, columns=CONTEXT_NAMES)
    out.insert(0, "year", [s.year for s in seasons])
    return out


def _knn(pool, ctx_pool, ctx_target, feats, k=3):
    table = pool[~pool["candidate_id"].isin(CONSTRUCTED)].pivot_table(index="year", columns="candidate_id", values="reward")
    rows = []
    for t in ctx_target.itertuples(index=False):
        dist = np.linalg.norm(ctx_pool[feats].to_numpy() - np.asarray([getattr(t, f) for f in feats]), axis=1)
        neighbours = ctx_pool["year"].to_numpy()[np.argsort(dist)[:k]]
        rows.append({"year": t.year, "candidate_id": table.loc[neighbours].mean().idxmax()})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/experiment_castro_power.yaml")
    args = parser.parse_args()
    _style()
    cfg = load_config(args.config)
    paths = make_paths(cfg, "article_rolling_power")
    out = paths.output_dir
    outputs = paths.project_dir / "outputs"
    tables = outputs / "rolling_power_analysis" / "tables"
    grid = pd.read_csv(tables / "candidate_grid_all_seasons.csv")
    ppo = pd.read_csv(tables / "ppo_rows_test.csv")
    comps = pd.read_csv(tables / "comparator_rows_test.csv")
    daily = load_weather(paths.project_dir, cfg)
    fold_list = folds(cfg)
    ag = cfg["agronomy"]

    # Action-bound saturation.
    lo, hi = ag["trigger_range_mm"]
    sat = pd.DataFrame([{
        "n_season_seed": len(ppo),
        "sowing_at_window_start": (ppo["planting_offset_days"] == 0).mean(),
        "sowing_at_window_end": (ppo["planting_offset_days"] >= 96).mean(),
        "trigger_at_lower_bound": (ppo["trigger_dryness"] <= lo + 0.5).mean(),
        "trigger_at_upper_bound": (ppo["trigger_dryness"] >= hi - 0.5).mean(),
        "depth_below_event_minimum": (ppo["amount_mm"] < ag["min_irrigation_event_mm"]).mean(),
        "depth_at_upper_bound": (ppo["amount_mm"] >= ag["max_single_irrigation_mm"] - 0.25).mean(),
        "cap_at_lower_bound": (ppo["max_irrigation_mm"] <= ag["min_season_cap_mm"] + 2.5).mean(),
        "cap_at_upper_bound": (ppo["max_irrigation_mm"] >= ag["min_season_cap_mm"] + ag["max_season_irrigation_mm"] - 2.5).mean(),
        "seasons_without_irrigation": (ppo["irrigation_mm"] == 0).mean(),
    }])
    sat.to_csv(out / "tables" / "action_bound_saturation.csv", index=False)

    # Context-to-action map: standardized test context of each fold against the PPO action.
    ctx_rows = []
    for f in fold_list:
        c = _contexts(cfg, daily, f, f["test"]).assign(fold=f["fold"])
        ctx_rows.append(c)
    ctx = pd.concat(ctx_rows, ignore_index=True)
    act = ppo.merge(ctx, on="year")
    act["sowing_doy"] = pd.to_datetime(act["planting_date"]).dt.dayofyear
    corr = []
    for f in CONTEXT_NAMES:
        for target in ["planting_offset_days", "trigger_dryness", "irrigation_mm"]:
            corr.append({"context_variable": f, "action": target,
                         "spearman_rho": act[f].corr(act[target], method="spearman")})
    pd.DataFrame(corr).to_csv(out / "tables" / "context_action_correlation.csv", index=False)
    by_season = act.groupby("year").agg(sowing_offset_mean=("planting_offset_days", "mean"), sowing_offset_sd=("planting_offset_days", "std"),
                                        trigger_mean=("trigger_dryness", "mean"), irrigation_mean=("irrigation_mm", "mean"))
    by_season.to_csv(out / "tables" / "ppo_actions_by_season.csv")
    fixed = comps[comps["comparator"] == "optimized_fixed_rule"][["year", "planting_offset_days"]].rename(columns={"planting_offset_days": "fixed_offset"})
    m = act.merge(fixed, on="year")
    change = pd.DataFrame([{
        "between_season_sd_sowing_offset_days": act.groupby("seed")["planting_offset_days"].std().mean(),
        "within_season_between_seed_sd_days": by_season["sowing_offset_sd"].mean(),
        "share_sowing_differs_from_fixed_rule_by_more_than_7_days": ((m["planting_offset_days"] - m["fixed_offset"]).abs() > 7).mean(),
    }])
    change.to_csv(out / "tables" / "decision_change_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), sharey=False)
    for ax, (feat, label) in zip(axes, [("rain_90d", "90-day rainfall (standardized)"), ("oni_mjj", "ONI May-July (standardized)"),
                                        ("temp_30d", "30-day temperature (standardized)")]):
        sc = ax.scatter(act[feat], act["planting_offset_days"], c=act["irrigation_mm"], cmap="Blues", s=22, edgecolor="#52514e", linewidth=0.3,
                        vmin=0, vmax=max(1, act["irrigation_mm"].max()))
        ax.set_xlabel(label)
        ax.set_ylabel("Sowing offset after Sep 15 (days)")
    fig.colorbar(sc, ax=axes, label="Seasonal irrigation (mm)", shrink=0.85)
    fig.savefig(out / "figures" / "fig_rolling_context_action.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Value-of-information ladder with the same candidate grid.
    ladder = {"fixed rule (no context)": [], "kNN, ONI only": [], "kNN, station weather only": [], "kNN, full context": []}
    for f in fold_list:
        pool = grid[grid["year"].isin(f["train"] + f["valid"])]
        target = grid[grid["year"].isin(f["test"])]
        fid = select_fixed_rule(pool)
        ladder["fixed rule (no context)"].append(target[target["candidate_id"] == fid])
        cp, ct = _contexts(cfg, daily, f, f["train"] + f["valid"]), _contexts(cfg, daily, f, f["test"])
        for name, feats in [("kNN, ONI only", ["oni_mjj", "oni_change"]), ("kNN, station weather only", CONTEXT_NAMES[:4]),
                            ("kNN, full context", CONTEXT_NAMES)]:
            sel = _knn(pool, cp, ct, feats)
            ladder[name].append(target.merge(sel, on=["year", "candidate_id"]))
    oracle = grid[grid["year"].isin(ppo["year"].unique()) & ~grid["candidate_id"].isin(CONSTRUCTED)].sort_values("reward", ascending=False).drop_duplicates("year")
    rows = [{"information": k, "reward": pd.concat(v)["reward"].mean(), "yield_kg_ha": pd.concat(v)["yield_kg_ha"].mean(),
             "irrigation_mm": pd.concat(v)["irrigation_mm"].mean()} for k, v in ladder.items()]
    rows.append({"information": "PPO, full context", "reward": ppo["reward"].mean(), "yield_kg_ha": ppo["yield_kg_ha"].mean(), "irrigation_mm": ppo["irrigation_mm"].mean()})
    rows.append({"information": "retrospective oracle", "reward": oracle["reward"].mean(), "yield_kg_ha": oracle["yield_kg_ha"].mean(), "irrigation_mm": oracle["irrigation_mm"].mean()})
    pd.DataFrame(rows).to_csv(out / "tables" / "value_of_information_ladder.csv", index=False)

    # Productivity-water frontier over kappa.
    front = []
    for prefix, kappa in KAPPAS.items():
        g = with_kappa(grid, cfg, kappa)
        sel = []
        for f in fold_list:
            pool = g[g["year"].isin(f["train"] + f["valid"])]
            fid = select_fixed_rule(pool)
            sel.append(g[(g["year"].isin(f["test"])) & (g["candidate_id"] == fid)])
        fr = pd.concat(sel)
        orc = g[g["year"].isin(fr["year"].unique()) & ~g["candidate_id"].isin(CONSTRUCTED)].sort_values("reward", ascending=False).drop_duplicates("year")
        front.append({"kappa": kappa, "policy": "optimized fixed rule", "yield_kg_ha": fr["yield_kg_ha"].mean(), "irrigation_mm": fr["irrigation_mm"].mean(), "reward": fr["reward"].mean(), "n": len(fr)})
        front.append({"kappa": kappa, "policy": "retrospective oracle", "yield_kg_ha": orc["yield_kg_ha"].mean(), "irrigation_mm": orc["irrigation_mm"].mean(), "reward": orc["reward"].mean(), "n": len(orc)})
        p_path = outputs / f"{prefix}_analysis" / "tables" / "ppo_rows_test.csv"
        if p_path.exists():
            p = pd.read_csv(p_path)
            if prefix != "rolling_power":
                p = p[p["seed"] == 42]
            label = "PPO retrained (seed 42)" if prefix != "rolling_power" else "PPO (seeds 42-46)"
            front.append({"kappa": kappa, "policy": label, "yield_kg_ha": p["yield_kg_ha"].mean(), "irrigation_mm": p["irrigation_mm"].mean(), "reward": p["reward"].mean(), "n": len(p)})
            if prefix == "rolling_power":
                p42 = p[p["seed"] == 42]
                front.append({"kappa": kappa, "policy": "PPO retrained (seed 42)", "yield_kg_ha": p42["yield_kg_ha"].mean(), "irrigation_mm": p42["irrigation_mm"].mean(), "reward": p42["reward"].mean(), "n": len(p42)})
    rf = grid[(grid["candidate_id"] == "rainfed_oct15") & grid["year"].isin(ppo["year"].unique())]
    front.append({"kappa": np.nan, "policy": "rainfed Oct 15", "yield_kg_ha": rf["yield_kg_ha"].mean(), "irrigation_mm": 0.0, "reward": np.nan, "n": len(rf)})
    front = pd.DataFrame(front)
    front.to_csv(out / "tables" / "water_frontier_by_kappa.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    styles = {"optimized fixed rule": ("D", COLORS["optimized_fixed_rule"]), "retrospective oracle": ("P", COLORS["retrospective_oracle"]),
              "PPO retrained (seed 42)": ("o", COLORS["PPO"])}
    for pol, (marker, color) in styles.items():
        g = front[front["policy"] == pol].sort_values("kappa")
        if g.empty:
            continue
        ax.plot(g["irrigation_mm"], g["yield_kg_ha"], marker=marker, color=color, label=LABELS.get(pol, pol.capitalize()), linewidth=1.2)
        for r in g.itertuples():
            ax.annotate(f"$\\kappa$={r.kappa:g}", (r.irrigation_mm, r.yield_kg_ha), fontsize=6, xytext=(4, -8), textcoords="offset points")
    r0 = front[front["policy"] == "rainfed Oct 15"].iloc[0]
    ax.scatter([0], [r0["yield_kg_ha"]], marker="s", color=COLORS["rainfed_oct15"], label="Rainfed Oct 15", zorder=4)
    ax.set_xlabel("Mean seasonal irrigation (mm)")
    ax.set_ylabel("Mean yield (kg ha$^{-1}$)")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    _save(fig, out / "figures" / "fig_rolling_water_frontier.png")

    # Yield regret under a common seasonal water budget (Eq. 2): oracle restricted to W <= budget.
    base = grid[grid["year"].isin(ppo["year"].unique()) & ~grid["candidate_id"].isin(CONSTRUCTED)]
    fixed_rows = comps[comps["comparator"] == "optimized_fixed_rule"]
    reg = []
    for b in BUDGETS:
        best = base[base["irrigation_mm"] <= b].groupby("year")["yield_kg_ha"].max()
        for name, rows_ in [("PPO", ppo), ("optimized fixed rule", fixed_rows)]:
            ok = rows_[rows_["irrigation_mm"] <= b]
            d = best.loc[ok["year"]].to_numpy() - ok["yield_kg_ha"].to_numpy()
            reg.append({"budget_mm": b, "policy": name, "pairs_within_budget": len(ok), "pairs_total": len(rows_),
                        "mean_yield_regret_kg_ha": float(np.mean(d)) if len(d) else np.nan})
    pd.DataFrame(reg).to_csv(out / "tables" / "yield_regret_common_budget.csv", index=False)

    # Skill of each context variable over the 41 seasons (standardized with all seasons; ranks are scale-free).
    seasons = build_year_weather_with_context(daily, sorted(grid["year"].unique()), cfg)
    ctx_all = pd.DataFrame([s.features for s in seasons], columns=CONTEXT_NAMES, index=[s.year for s in seasons])
    rf = grid[grid["candidate_id"].str.endswith("trig9999_amt0_cap0")]
    best_off = rf.loc[rf.groupby("year")["yield_kg_ha"].idxmax()].set_index("year")["planting_offset_days"]
    gain = grid[~grid["candidate_id"].isin(CONSTRUCTED)].groupby("year")["yield_kg_ha"].max() - rf.groupby("year")["yield_kg_ha"].max()
    oct15 = grid[grid["candidate_id"] == "rainfed_oct15"].set_index("year")["yield_kg_ha"]
    targets = pd.DataFrame({"best_rainfed_offset_days": best_off, "irrigation_gain_kg_ha": gain, "rainfed_oct15_yield_kg_ha": oct15})
    joined = ctx_all.join(targets)
    skill = pd.DataFrame([{"context_variable": f, "target": t, "spearman_rho": joined[f].corr(joined[t], method="spearman"), "n_seasons": len(joined)}
                          for f in CONTEXT_NAMES for t in targets.columns])
    skill.to_csv(out / "tables" / "context_skill_power.csv", index=False)

    # Three-seed versus five-seed summary.
    s3, s5 = tables.parent / "tables_3seeds", tables
    if (s3 / "paired_effects_summary.csv").exists():
        a = pd.read_csv(s3 / "paired_effects_summary.csv").assign(seeds="42-44")
        b = pd.read_csv(s5 / "paired_effects_summary.csv").assign(seeds="42-46")
        pd.concat([a, b]).to_csv(out / "tables" / "paired_effects_3_vs_5_seeds.csv", index=False)

    for name in ["action_bound_saturation", "decision_change_summary", "value_of_information_ladder", "water_frontier_by_kappa", "yield_regret_common_budget"]:
        print(f"== {name}")
        print(pd.read_csv(out / "tables" / f"{name}.csv").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
