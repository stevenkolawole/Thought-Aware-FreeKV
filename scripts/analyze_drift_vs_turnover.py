"""How does the *number of drifted kv-groups* relate to total cache turnover?

For each correction event (step, layer with need_corr=1):
  - From sims_*.npz: derive per-kv-group sim by averaging over 4 q-heads/group
  - Count `n_drifted` = how many of 8 groups have group-sim < 0.9 (the trigger)
  - From recall_*.csv: look up `n_pages` (sum of new pages over all 8 groups)

If `n_pages` rises steeply with `n_drifted`:
    drifted groups are the dominant contributors → per-group suppression matters
    (BIMODAL world: skip non-drifted groups' correction work)

If `n_pages` is roughly constant regardless of `n_drifted`:
    all groups churn similarly → per-step suppression is the right level
    (UNIFORM world: per-group skip buys nothing, just skip whole events)

Outputs:
  analysis/full_aime/drift_vs_turnover.md
  analysis/full_aime/plots/n_pages_vs_n_drifted_*.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent

N_LAYERS = 32
N_Q_HEADS = 32
N_KV_GROUPS = 8
N_Q_PER_GROUP = N_Q_HEADS // N_KV_GROUPS  # = 4
TOPK_PER_GROUP = 32
TOTAL_TOPK = TOPK_PER_GROUP * N_KV_GROUPS  # = 256
TAU = 0.9


def collect_for_problem(input_dir: Path, pid: str) -> pd.DataFrame:
    sim_path = input_dir / f"sims_{pid}.npz"
    corr_path = input_dir / f"corr_{pid}.csv"
    recall_path = input_dir / f"recall_{pid}.csv"
    if not (sim_path.exists() and corr_path.exists() and recall_path.exists()):
        return pd.DataFrame()

    sims = np.load(sim_path)["sim"]  # [n_steps, 32 layers, 32 q heads]
    # group-sims: [n_steps, 32 layers, 8 kv groups]
    group_sims = sims.reshape(sims.shape[0], N_LAYERS, N_KV_GROUPS, N_Q_PER_GROUP).mean(axis=-1)

    corr = pd.read_csv(corr_path)
    recall = pd.read_csv(recall_path)
    corr_events = corr[corr["need_corr"] == 1][["step_id", "layer_id"]].copy()
    if corr_events.empty:
        return pd.DataFrame()

    # Dedupe recall to one row per (step, layer): each correction has up to
    # 2 logged rows with the same n_pages — keep the first.
    recall_one = (
        recall.sort_values(["step_id", "layer_id"])
        .drop_duplicates(subset=["step_id", "layer_id"], keep="first")
        [["step_id", "layer_id", "n_pages"]]
    )
    joined = corr_events.merge(recall_one, on=["step_id", "layer_id"], how="inner")
    if joined.empty:
        return pd.DataFrame()

    # Add n_drifted by indexing into group_sims
    s = joined["step_id"].to_numpy()
    L = joined["layer_id"].to_numpy()
    valid_mask = (
        (s < group_sims.shape[0]) & (L < group_sims.shape[1])
    )
    joined = joined[valid_mask].reset_index(drop=True)
    s = joined["step_id"].to_numpy()
    L = joined["layer_id"].to_numpy()
    g = group_sims[s, L]  # [n_events, 8]

    # If sim is NaN (pre-budget step), drop the row.
    nan_rows = np.isnan(g).any(axis=-1)
    joined = joined[~nan_rows].reset_index(drop=True)
    g = g[~nan_rows]

    joined["n_drifted"] = (g < TAU).sum(axis=-1)
    joined["min_group_sim"] = g.min(axis=-1)
    joined["mean_group_sim"] = g.mean(axis=-1)
    joined["problem_id"] = pid
    return joined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=Path,
                    default=REPO_ROOT / "modal_logs" / "full_aime" / "full_aime")
    ap.add_argument("--output_dir", type=Path,
                    default=REPO_ROOT / "analysis" / "full_aime")
    args = ap.parse_args()

    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    pids = sorted(
        re.match(r"sims_(.+)\.npz", p.name).group(1)
        for p in args.input_dir.glob("sims_*.npz")
    )
    print(f"problems with per-head sim: {len(pids)}")

    parts = []
    for pid in pids:
        df = collect_for_problem(args.input_dir, pid)
        if not df.empty:
            print(f"  {pid:<12} n_corrections={len(df):,}")
            parts.append(df)

    big = pd.concat(parts, ignore_index=True)
    print(f"\ntotal correction events with both sim + recall: {len(big):,}")

    # Bucket by n_drifted (1..8 — by definition need_corr fires when at least 1 drifts)
    buckets = (
        big.groupby("n_drifted")
        .agg(
            n_events=("n_pages", "size"),
            n_pages_mean=("n_pages", "mean"),
            n_pages_median=("n_pages", "median"),
            n_pages_std=("n_pages", "std"),
            n_pages_p10=("n_pages", lambda s: s.quantile(0.10)),
            n_pages_p90=("n_pages", lambda s: s.quantile(0.90)),
            min_sim_mean=("min_group_sim", "mean"),
        )
        .reset_index()
    )
    buckets["fraction_of_events"] = buckets["n_events"] / buckets["n_events"].sum()
    buckets["mean_pages_per_drifted_group"] = buckets["n_pages_mean"] / buckets["n_drifted"]
    buckets["mean_pages_per_quiet_group"] = (
        (buckets["n_pages_mean"] - buckets["mean_pages_per_drifted_group"] * buckets["n_drifted"])
        / (N_KV_GROUPS - buckets["n_drifted"]).replace(0, np.nan)
    )
    print("\n=== n_pages bucketed by n_drifted ===")
    print(buckets.round(3).to_string(index=False))

    # Pearson correlation between n_drifted and n_pages
    corr_coef = float(np.corrcoef(big["n_drifted"], big["n_pages"])[0, 1])
    print(f"\nPearson(n_drifted, n_pages) = {corr_coef:.4f}")

    # If only drifted groups contribute new pages, n_pages should scale ~linearly
    # with n_drifted. Fit a slope.
    slope = (big["n_pages"].cov(big["n_drifted"]) / big["n_drifted"].var())
    intercept = big["n_pages"].mean() - slope * big["n_drifted"].mean()
    print(f"OLS fit: n_pages ≈ {slope:.2f} * n_drifted + {intercept:.2f}")
    print(f"   slope of {slope:.2f} pages/drifted-group means each additional drifted "
          f"group adds ~{slope:.1f} new pages on average.")
    print(f"   intercept of {intercept:.2f} is the BASELINE turnover that exists "
          "even when 0 groups are technically drifted (≈ background churn from "
          "non-drifted groups).")

    # === plots ===
    # 1. mean n_pages by bucket
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(buckets["n_drifted"], buckets["n_pages_mean"],
           yerr=buckets["n_pages_std"], capsize=4, color="tab:red", alpha=0.85,
           label="mean ± 1 std")
    # Reference line: if drifted groups contributed ALL the turnover
    # n_pages = 32 * n_drifted (all 32 slots fresh per drifted group)
    ref_x = np.arange(1, 9)
    ax.plot(ref_x, 32 * ref_x, "k--", lw=1, alpha=0.6,
            label="upper bound (32 × n_drifted)")
    # Reference line: linear fit
    ax.plot(ref_x, slope * ref_x + intercept, "g-", lw=1.5, alpha=0.85,
            label=f"OLS fit: {slope:.1f}x + {intercept:.1f}")
    ax.set_xlabel("number of drifted kv-groups (sim < 0.9) at this (step, layer)")
    ax.set_ylabel("n_pages summed over all 8 groups")
    ax.set_title("Does drifted-group count predict cache turnover?")
    ax.set_xticks(ref_x)
    ax.set_ylim(0, max(buckets["n_pages_mean"].max() * 1.1, 256))
    ax.axhline(TOTAL_TOPK, color="gray", lw=0.5, ls=":", alpha=0.7,
               label=f"total top-k = {TOTAL_TOPK}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = plot_dir / "n_pages_vs_n_drifted_bar.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"  wrote {p}")

    # 2. fraction of events per drifted-group count
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(buckets["n_drifted"], buckets["fraction_of_events"], color="tab:blue")
    ax.set_xlabel("number of drifted kv-groups")
    ax.set_ylabel("fraction of correction events")
    ax.set_title("How many groups typically drift when correction fires?")
    ax.set_xticks(np.arange(1, 9))
    ax.grid(True, axis="y", alpha=0.3)
    for x, frac in zip(buckets["n_drifted"], buckets["fraction_of_events"]):
        ax.text(x, frac + 0.005, f"{frac:.1%}", ha="center", fontsize=8)
    fig.tight_layout()
    p = plot_dir / "n_drifted_distribution.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"  wrote {p}")

    # 3. 2-D heatmap (n_drifted × n_pages bucket)
    fig, ax = plt.subplots(figsize=(10, 5))
    H, xedges, yedges = np.histogram2d(
        big["n_drifted"], big["n_pages"],
        bins=[np.arange(0.5, 9.5, 1), np.arange(0, 260, 8)],
    )
    im = ax.imshow(
        H.T, origin="lower", aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="viridis",
    )
    ax.set_xlabel("number of drifted kv-groups")
    ax.set_ylabel("n_pages")
    ax.set_title("Joint distribution: drifted-group count vs cache turnover")
    plt.colorbar(im, ax=ax, label="event count")
    ax.set_xticks(np.arange(1, 9))
    fig.tight_layout()
    p = plot_dir / "n_pages_vs_n_drifted_heatmap.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"  wrote {p}")

    # === markdown report ===
    lines = []
    lines.append("# Drifted-group count vs cache turnover (full_aime)\n")
    lines.append(
        "Question: when correction fires, is the cache turnover concentrated "
        "in the drifted kv-groups, or spread across all 8?\n"
    )
    lines.append(f"Source: {len(pids)} problems with per-head sim cached.\n")
    lines.append(f"Total correction events analyzed: **{len(big):,}**\n")
    lines.append(f"Pearson correlation between `n_drifted` and `n_pages`: "
                 f"**{corr_coef:.4f}**\n")
    lines.append(f"OLS fit: `n_pages ≈ {slope:.2f} × n_drifted + {intercept:.2f}`\n")

    if corr_coef > 0.6:
        verdict = (
            "**Verdict: bimodal — drifted groups dominate the turnover.** "
            "Per-group correction suppression is high-leverage: skipping "
            "correction work for non-drifted groups would save substantial "
            "compute / synchronization, even though bandwidth-wise the "
            "kernel already does this via need_recall_corr."
        )
    elif corr_coef > 0.3:
        verdict = (
            "**Verdict: mixed.** Drifted-group count partially predicts "
            "turnover but there's significant baseline churn from non-drifted "
            "groups. Per-group suppression has some value, but the gain is "
            "smaller than the headline 80%-reuse number suggests."
        )
    else:
        verdict = (
            "**Verdict: uniform — turnover is roughly the same regardless of "
            "how many groups drifted.** Per-group suppression buys little; "
            "the simpler design is per-step suppression (skip whole events "
            "when predicted reuse is high)."
        )
    lines.append(f"\n{verdict}\n")

    lines.append("\n## Bucket statistics\n")
    lines.append("| n_drifted | n_events | events% | mean n_pages | median | std | p10 | p90 | mean per drifted-group |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in buckets.iterrows():
        per_drifted = r["mean_pages_per_drifted_group"]
        lines.append(
            f"| {int(r['n_drifted'])} | {int(r['n_events']):,} | "
            f"{r['fraction_of_events']:.2%} | {r['n_pages_mean']:.1f} | "
            f"{r['n_pages_median']:.0f} | {r['n_pages_std']:.1f} | "
            f"{r['n_pages_p10']:.0f} | {r['n_pages_p90']:.0f} | "
            f"{per_drifted:.1f} |"
        )

    lines.append("\n## Interpretation\n")
    lines.append(
        f"- The *upper bound* if drifted groups contributed ALL the turnover: "
        f"`n_pages = 32 × n_drifted`. Compare the empirical mean to `32x` in "
        f"the bar chart.\n"
        f"- The *lower bound* if drifted groups contributed nothing: `n_pages` "
        f"would be flat with respect to `n_drifted`. We can see how flat or "
        f"steep the relationship is.\n"
        f"- Slope/intercept tell us: each additional drifted group raises "
        f"turnover by ~{slope:.1f} pages, on top of a baseline of "
        f"~{intercept:.1f} pages that doesn't depend on drift count.\n"
    )

    lines.append("\n## Plots\n")
    lines.append("- [n_pages_vs_n_drifted_bar.png](plots/n_pages_vs_n_drifted_bar.png) — mean n_pages per bucket with OLS fit and 32×n_drifted upper bound")
    lines.append("- [n_drifted_distribution.png](plots/n_drifted_distribution.png) — how many groups typically drift per event")
    lines.append("- [n_pages_vs_n_drifted_heatmap.png](plots/n_pages_vs_n_drifted_heatmap.png) — joint distribution heatmap")

    out_path = args.output_dir / "drift_vs_turnover.md"
    out_path.write_text("\n".join(lines))
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    main()
