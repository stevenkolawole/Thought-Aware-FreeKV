"""Per-component CUDA-event timing analysis answering 4 questions:

  Q1. How much longer does each decode step take if a correction happens?
  Q2. What is the slowest component when correcting vs not correcting?
  Q3. What is the correction overhead relative to total model time?
  Q4. How does latency and bandwidth correlate to the number of pages
      that are new upon a correction?

Inputs (from a Modal run with the timing instrumentation):
  modal_logs/<run>/<run>/{corr,recall,timing,tbt}_<pid>.csv

Output:
  analysis/<run>/timing.md   + plots/timing_*.png
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

# components in modeling.py time_block_start calls
ALL_COMPONENTS = [
    "qkv_rope", "cos", "estimate_select", "recall_sync", "attn", "oproj",
]
# components ALWAYS present per layer per step (cos appears whenever past budget+spec_ret)
ALWAYS_COMPONENTS = ["qkv_rope", "attn", "oproj"]
# components only when correction triggers
CORR_COMPONENTS = ["estimate_select", "recall_sync"]


def load_run(input_dir: Path) -> dict:
    """Returns dict[pid] = {timing, corr, recall, tbt}."""
    out: dict = {}
    for tcsv in sorted(input_dir.glob("timing_*.csv")):
        pid = tcsv.stem.removeprefix("timing_")
        try:
            t = pd.read_csv(tcsv)
            c = pd.read_csv(input_dir / f"corr_{pid}.csv")
            r = pd.read_csv(input_dir / f"recall_{pid}.csv")
            tb = pd.read_csv(input_dir / f"tbt_{pid}.csv")
        except FileNotFoundError:
            continue
        if t.empty or c.empty or tb.empty:
            continue
        out[pid] = {"timing": t, "corr": c, "recall": r, "tbt": tb}
    return out


def step_level_corr_flags(corr: pd.DataFrame) -> pd.DataFrame:
    """For each step, summary of correction activity across layers."""
    g = corr.groupby("step_id").agg(
        any_corr=("need_corr", "max"),
        n_layers_corr=("need_corr", "sum"),
    ).reset_index()
    return g


def q1_step_delta(per_problem: dict) -> tuple[pd.DataFrame, dict]:
    """Q1: TBT delta when correction happens. Returns (per-problem table,
    cross-problem aggregate dict)."""
    rows = []
    pooled = []
    for pid, d in per_problem.items():
        flags = step_level_corr_flags(d["corr"])
        joined = d["tbt"].merge(flags, on="step_id", how="left")
        joined["any_corr"] = joined["any_corr"].fillna(0).astype(int)
        joined["n_layers_corr"] = joined["n_layers_corr"].fillna(0).astype(int)
        joined["pid"] = pid
        pooled.append(joined)
        sub_yes = joined[joined["any_corr"] == 1]
        sub_no = joined[joined["any_corr"] == 0]
        rows.append({
            "pid": pid,
            "n_steps": int(len(joined)),
            "n_steps_with_corr": int(sub_yes.shape[0]),
            "tbt_mean_corr_ms": float(sub_yes["total_ms"].mean()) if not sub_yes.empty else float("nan"),
            "tbt_mean_no_corr_ms": float(sub_no["total_ms"].mean()) if not sub_no.empty else float("nan"),
            "tbt_p50_corr_ms": float(sub_yes["total_ms"].median()) if not sub_yes.empty else float("nan"),
            "tbt_p99_corr_ms": float(sub_yes["total_ms"].quantile(0.99)) if not sub_yes.empty else float("nan"),
        })
    big = pd.concat(pooled, ignore_index=True) if pooled else pd.DataFrame()
    df = pd.DataFrame(rows)
    df["delta_ms"] = df["tbt_mean_corr_ms"] - df["tbt_mean_no_corr_ms"]
    df["delta_pct"] = df["delta_ms"] / df["tbt_mean_no_corr_ms"]
    pooled_yes = big[big["any_corr"] == 1]["total_ms"]
    pooled_no = big[big["any_corr"] == 0]["total_ms"]
    agg = {
        "tbt_corr_mean": float(pooled_yes.mean()) if len(pooled_yes) else float("nan"),
        "tbt_no_corr_mean": float(pooled_no.mean()) if len(pooled_no) else float("nan"),
        "delta_mean_ms": (
            float(pooled_yes.mean() - pooled_no.mean())
            if (len(pooled_yes) and len(pooled_no))
            else float("nan")
        ),
        "n_steps_with_corr": int(len(pooled_yes)),
        "n_steps_total": int(len(big)),
    }
    return df, agg, big


def q2_component_breakdown(per_problem: dict) -> pd.DataFrame:
    """Q2: Sum component time per step then stratify by phase.

    Phases:
      - "pre_budget": cos didn't run this step (KV under budget; no correction
        code path). Components: qkv_rope, attn, oproj only.
      - "post_no_corr": cos ran, no layer fired need_corr. Rare in practice
        (~0% on AIME with 89% trigger rate × 32 layers).
      - "post_corr": cos ran, at least one layer corrected.

    Returns long-form: pid, phase, component, mean, median, count.
    """
    rows = []
    for pid, d in per_problem.items():
        timing = d["timing"]
        flags = step_level_corr_flags(d["corr"])
        post_budget_steps = set(
            timing[timing["component"] == "cos"]["step_id"].unique().tolist()
        )
        per_step = timing.groupby(["step_id", "component"])["us"].sum().reset_index()
        per_step = per_step.merge(flags, on="step_id", how="left")
        per_step["any_corr"] = per_step["any_corr"].fillna(0).astype(int)
        per_step["phase"] = per_step["step_id"].apply(
            lambda s, _pb=post_budget_steps: "post" if s in _pb else "pre"
        )
        per_step.loc[per_step["phase"] == "pre", "phase"] = "pre_budget"
        per_step.loc[
            (per_step["phase"] == "post") & (per_step["any_corr"] == 0),
            "phase"
        ] = "post_no_corr"
        per_step.loc[
            (per_step["phase"] == "post") & (per_step["any_corr"] == 1),
            "phase"
        ] = "post_corr"
        agg = (
            per_step.groupby(["phase", "component"])["us"]
            .agg(["mean", "median", "count"])
            .reset_index()
        )
        agg["pid"] = pid
        rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def q3_overhead(per_problem: dict) -> pd.DataFrame:
    """Q3: correction overhead (cos + estimate_select + recall_sync) as
    fraction of TBT, per problem."""
    rows = []
    for pid, d in per_problem.items():
        timing = d["timing"]
        tbt = d["tbt"]
        # always-on overhead = cos
        cos_us = (timing[timing["component"] == "cos"]
                  .groupby("step_id")["us"].sum().reset_index()
                  .rename(columns={"us": "cos_us"}))
        # triggered overhead = estimate_select + recall_sync
        trig = (timing[timing["component"].isin(["estimate_select", "recall_sync"])]
                .groupby("step_id")["us"].sum().reset_index()
                .rename(columns={"us": "trig_us"}))
        per_step = (tbt.merge(cos_us, on="step_id", how="left")
                       .merge(trig, on="step_id", how="left"))
        per_step["cos_us"] = per_step["cos_us"].fillna(0)
        per_step["trig_us"] = per_step["trig_us"].fillna(0)
        per_step["overhead_us"] = per_step["cos_us"] + per_step["trig_us"]
        per_step["overhead_frac"] = (
            per_step["overhead_us"] / 1000.0 / per_step["total_ms"]
        )
        rows.append({
            "pid": pid,
            "n_steps": len(per_step),
            "tbt_mean_ms": float(per_step["total_ms"].mean()),
            "tbt_sum_s": float(per_step["total_ms"].sum() / 1000.0),
            "cos_mean_us": float(per_step["cos_us"].mean()),
            "trig_mean_us": float(per_step["trig_us"].mean()),
            "overhead_mean_us": float(per_step["overhead_us"].mean()),
            "overhead_mean_frac": float(per_step["overhead_frac"].mean()),
            "overhead_total_frac": float(
                per_step["overhead_us"].sum() / 1000.0 / per_step["total_ms"].sum()
            ),
        })
    return pd.DataFrame(rows)


def q4_recall_latency_vs_pages(per_problem: dict) -> pd.DataFrame:
    """Q4: join recall_sync timing rows with recall log so each row has
    (n_pages, n_pages_actual, bytes_actual, recall_sync_us, kind)."""
    rows = []
    for pid, d in per_problem.items():
        timing = d["timing"]
        recall = d["recall"]
        sync_t = (
            timing[timing["component"] == "recall_sync"]
            [["step_id", "layer_id", "us"]]
            .rename(columns={"us": "recall_sync_us"})
        )
        # The recall log has 2 rows per (step, layer) for correction events
        # (synchronous + post-correction async). Both share the same n_pages.
        # We pick one row per (step, layer) — first.
        recall_one = (
            recall.sort_values(["step_id", "layer_id"])
            .drop_duplicates(["step_id", "layer_id"], keep="first")
        )
        joined = sync_t.merge(
            recall_one, on=["step_id", "layer_id"], how="inner"
        )
        joined["pid"] = pid
        rows.append(joined)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _safe_corr(x, y):
    if len(x) < 3:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dir", type=Path,
        default=REPO_ROOT / "modal_logs" / "profile_aime" / "profile_aime",
    )
    ap.add_argument(
        "--output_dir", type=Path,
        default=REPO_ROOT / "analysis" / "profile_aime",
    )
    args = ap.parse_args()
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    per = load_run(args.input_dir)
    if not per:
        raise SystemExit(f"no timing data found in {args.input_dir}")
    print(f"Loaded {len(per)} problems: {sorted(per.keys())}")

    # =================================================================
    # Q1
    # =================================================================
    q1_df, q1_agg, big_tbt = q1_step_delta(per)
    print("\n=== Q1: TBT delta when correction happens ===")
    print(q1_df.round(3).to_string(index=False))
    print(f"\nPooled across all problems:")
    for k, v in q1_agg.items():
        print(f"  {k}: {v}")

    # plot: distribution of TBT, corr vs no-corr
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bins = np.linspace(big_tbt["total_ms"].min(),
                       big_tbt["total_ms"].quantile(0.995), 80)
    ax.hist(big_tbt[big_tbt["any_corr"] == 0]["total_ms"], bins=bins,
            alpha=0.6, color="tab:blue", density=True,
            label=f"no correction (n={(big_tbt['any_corr']==0).sum():,})")
    ax.hist(big_tbt[big_tbt["any_corr"] == 1]["total_ms"], bins=bins,
            alpha=0.6, color="tab:red", density=True,
            label=f"any correction (n={(big_tbt['any_corr']==1).sum():,})")
    ax.set_xlabel("step total time (ms)")
    ax.set_ylabel("density")
    ax.set_title(f"Q1: TBT distribution by correction status "
                 f"(Δmean = {q1_agg['delta_mean_ms']:.2f} ms)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "q1_tbt_distribution.png", dpi=120)
    plt.close(fig)

    # plot: TBT vs n_layers_corr (dose-response)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    by_n = (
        big_tbt.groupby("n_layers_corr")["total_ms"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )
    ax.bar(by_n["n_layers_corr"], by_n["mean"], yerr=by_n["std"],
           color="tab:red", alpha=0.85, capsize=3)
    ax.set_xlabel("number of layers with need_corr=1 in this step")
    ax.set_ylabel("mean step TBT (ms)")
    ax.set_title(f"Q1: TBT vs how many layers corrected in a step")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "q1_tbt_vs_n_layers_corr.png", dpi=120)
    plt.close(fig)

    # =================================================================
    # Q2
    # =================================================================
    q2_long = q2_component_breakdown(per)
    print("\n=== Q2: per-step mean component time (us) by phase ===")
    pivot = (
        q2_long.groupby(["phase", "component"])["mean"].mean()
        .unstack("component").reindex(columns=ALL_COMPONENTS, fill_value=np.nan)
    )
    pivot = pivot.reindex(["pre_budget", "post_no_corr", "post_corr"])
    print(pivot.round(2).to_string())
    # Show how many steps fall into each bucket pooled across problems
    n_per_phase = q2_long.groupby("phase")["count"].sum().reindex(
        ["pre_budget", "post_no_corr", "post_corr"], fill_value=0
    ) // len(ALL_COMPONENTS)  # rough — count per phase via any component
    print(f"\napprox steps per phase (pooled): {dict(n_per_phase)}")

    # stacked bar: components per phase
    phases = ["pre_budget", "post_no_corr", "post_corr"]
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.55
    x = np.arange(len(phases))
    bottom = np.zeros(len(phases))
    colors = {
        "qkv_rope": "tab:blue",
        "cos": "tab:purple",
        "estimate_select": "tab:olive",
        "recall_sync": "tab:red",
        "attn": "tab:green",
        "oproj": "tab:cyan",
    }
    for c in ALL_COMPONENTS:
        means = []
        for ph in phases:
            v = q2_long[(q2_long["phase"] == ph) & (q2_long["component"] == c)]["mean"].mean()
            means.append(0.0 if pd.isna(v) else float(v))
        ax.bar(x, means, width, bottom=bottom, color=colors.get(c, "gray"),
               label=c)
        bottom += np.array(means)
    ax.set_xticks(x)
    ax.set_xticklabels([
        "pre-budget\n(no correction code path)",
        "post-budget,\nno correction triggered",
        "post-budget,\ncorrection triggered",
    ])
    ax.set_ylabel("mean component time per step (µs)")
    ax.set_title("Q2: per-step component time by phase")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "q2_component_stacked.png", dpi=120)
    plt.close(fig)

    # =================================================================
    # Q3
    # =================================================================
    q3 = q3_overhead(per)
    print("\n=== Q3: correction overhead vs total time ===")
    print(q3.round(4).to_string(index=False))

    # plot: per-problem overhead fraction
    fig, ax = plt.subplots(figsize=(10, 4))
    pids = q3["pid"].tolist()
    ax.bar(pids, q3["overhead_total_frac"], color="tab:red", alpha=0.85)
    ax.set_ylabel("correction overhead / total wall time")
    ax.set_title("Q3: correction overhead as fraction of total model time")
    for i, v in enumerate(q3["overhead_total_frac"]):
        ax.text(i, v + 0.005, f"{v:.2%}", ha="center", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "q3_overhead_fraction.png", dpi=120)
    plt.close(fig)

    # =================================================================
    # Q4
    # =================================================================
    q4 = q4_recall_latency_vs_pages(per)
    print(f"\n=== Q4: recall_sync latency vs n_pages ===")
    print(f"  joined rows: {len(q4):,}")
    if not q4.empty:
        q4 = q4[q4["recall_sync_us"] > 0].copy()
        # bytes / µs == MB / s; divide by 1000 to get GB / s.
        q4["bandwidth_GBps"] = (
            q4["bytes_actual"] / q4["recall_sync_us"] / 1000.0
        )

        print("  recall_sync_us:    "
              f"mean={q4['recall_sync_us'].mean():.1f}  "
              f"p50={q4['recall_sync_us'].median():.1f}  "
              f"p99={q4['recall_sync_us'].quantile(0.99):.1f}")
        print("  n_pages:           "
              f"mean={q4['n_pages'].mean():.1f}  median={q4['n_pages'].median():.0f}")
        print("  n_pages_actual:    "
              f"mean={q4['n_pages_actual'].mean():.1f}  median={q4['n_pages_actual'].median():.0f}")
        print("  bytes_actual (KB): "
              f"mean={q4['bytes_actual'].mean()/1024:.1f}")
        print("  bandwidth_GBps:    "
              f"mean={q4['bandwidth_GBps'].mean():.2f}  "
              f"p50={q4['bandwidth_GBps'].median():.2f}  "
              f"p99={q4['bandwidth_GBps'].quantile(0.99):.2f}")
        print(f"  Pearson(n_pages_actual, recall_sync_us) = "
              f"{_safe_corr(q4['n_pages_actual'].to_numpy(), q4['recall_sync_us'].to_numpy()):.4f}")
        print(f"  Pearson(bytes_actual, recall_sync_us)   = "
              f"{_safe_corr(q4['bytes_actual'].to_numpy(), q4['recall_sync_us'].to_numpy()):.4f}")

        # scatter latency vs pages
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
        sample = q4.sample(min(20000, len(q4)), random_state=0)
        axs[0].scatter(sample["n_pages_actual"], sample["recall_sync_us"],
                       alpha=0.3, s=4, color="tab:blue")
        axs[0].set_xlabel("n_pages_actual (after need_recall_corr mask)")
        axs[0].set_ylabel("recall_sync latency (µs)")
        axs[0].set_title("Q4: recall_sync latency vs actual pages transferred")
        axs[0].grid(True, alpha=0.3)
        axs[1].scatter(sample["bytes_actual"] / 1024,
                       sample["bandwidth_GBps"],
                       alpha=0.3, s=4, color="tab:green")
        axs[1].set_xlabel("bytes_actual (KB)")
        axs[1].set_ylabel("effective PCIe bandwidth (GB/s)")
        axs[1].set_title("Q4: effective bandwidth vs transfer size")
        axs[1].grid(True, alpha=0.3)
        # A100 host↔device practical max ~25 GB/s
        axs[1].axhline(25, color="gray", lw=0.6, ls="--",
                       alpha=0.6, label="A100 PCIe ~25 GB/s")
        axs[1].legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "q4_latency_vs_pages.png", dpi=120)
        plt.close(fig)

    # =================================================================
    # Markdown report
    # =================================================================
    report = []
    report.append("# `profile_aime` — systems profiling analysis\n")
    try:
        rel = args.input_dir.resolve().relative_to(REPO_ROOT)
    except ValueError:
        rel = args.input_dir
    report.append(f"Source: `{rel}`\n")
    report.append(f"Problems: {sorted(per.keys())}\n")

    report.append("## Q1. How much longer does a step take when correction happens?\n")
    report.append("Per-step TBT bucketed by whether ANY layer fired `need_corr` in that step.\n")
    report.append("| pid | n_steps | n_steps_corr | TBT no_corr (ms) | TBT corr (ms) | Δ ms | Δ % |")
    report.append("|---|---|---|---|---|---|---|")
    for _, r in q1_df.iterrows():
        report.append(
            f"| `{r['pid']}` | {int(r['n_steps']):,} | {int(r['n_steps_with_corr']):,} | "
            f"{r['tbt_mean_no_corr_ms']:.2f} | {r['tbt_mean_corr_ms']:.2f} | "
            f"{r['delta_ms']:.2f} | {r['delta_pct']:.2%} |"
        )
    report.append("")
    report.append(f"**Pooled across all 5 problems:**")
    report.append(f"- TBT mean (no correction): {q1_agg['tbt_no_corr_mean']:.2f} ms")
    report.append(f"- TBT mean (any correction): {q1_agg['tbt_corr_mean']:.2f} ms")
    report.append(f"- **Δ mean: {q1_agg['delta_mean_ms']:.2f} ms** "
                  f"({q1_agg['delta_mean_ms']/q1_agg['tbt_no_corr_mean']:.2%} relative)\n")
    report.append("![TBT distribution](plots/q1_tbt_distribution.png)\n")
    report.append("![TBT vs n_layers_corr](plots/q1_tbt_vs_n_layers_corr.png)\n")

    report.append("## Q2. Slowest component, with vs without correction\n")
    report.append(
        "Per-step component time (sum across 32 layers), µs. Three phases:\n"
        "- **pre_budget** — KV under budget, no correction code path runs at all.\n"
        "- **post_no_corr** — past budget, cos check ran, no layer fired need_corr.\n"
        "  This bucket is essentially empty on AIME because at 89% per-(step, layer) "
        "trigger rate, P(no layer fires across 32 layers) ≈ 0.\n"
        "- **post_corr** — past budget, ≥1 layer corrected. Almost all post-budget "
        "steps fall here.\n"
    )
    report.append("| component | pre_budget (µs) | post_no_corr (µs) | post_corr (µs) |")
    report.append("|---|---|---|---|")
    for c in ALL_COMPONENTS:
        vals = []
        for ph in ["pre_budget", "post_no_corr", "post_corr"]:
            v = q2_long[(q2_long["phase"] == ph) & (q2_long["component"] == c)]["mean"].mean()
            vals.append(v)
        report.append(
            f"| {c} | {vals[0]:.1f} | {vals[1] if not pd.isna(vals[1]) else 0:.1f} "
            f"| {vals[2]:.1f} |"
        )
    report.append("")
    report.append("![Stacked components](plots/q2_component_stacked.png)\n")
    report.append(
        "**Caveat on the `cos` component:** our per-head sim caching adds an "
        "extra `.float().cpu().numpy()` per layer per step (32 fp32 values), "
        "which forces a GPU→CPU sync. So the `cos` time we measure is "
        "FreeKV's correction trigger cost PLUS our logging cost. Without "
        "per-head logging, cos would be much smaller.\n"
    )

    report.append("## Q3. Correction overhead as fraction of total wall time\n")
    report.append("Overhead = `cos + estimate_select + recall_sync` per step.\n")
    report.append("| pid | TBT mean (ms) | cos (µs) | trigger (µs) | overhead total fraction |")
    report.append("|---|---|---|---|---|")
    for _, r in q3.iterrows():
        report.append(
            f"| `{r['pid']}` | {r['tbt_mean_ms']:.2f} | {r['cos_mean_us']:.1f} | "
            f"{r['trig_mean_us']:.1f} | {r['overhead_total_frac']:.4f} ({r['overhead_total_frac']:.2%}) |"
        )
    report.append(f"\n![Overhead fraction](plots/q3_overhead_fraction.png)\n")

    report.append("## Q4. Recall latency and bandwidth vs pages-actually-transferred\n")
    if not q4.empty:
        rows_yes = q4[q4["recall_sync_us"] > 0]
        report.append(f"Joined `recall_sync` timing with the recall log "
                      f"({len(rows_yes):,} correction events).\n")
        report.append(f"- Mean recall_sync latency: **{rows_yes['recall_sync_us'].mean():.1f} µs**")
        report.append(f"- Median: {rows_yes['recall_sync_us'].median():.1f} µs, "
                      f"p99: {rows_yes['recall_sync_us'].quantile(0.99):.1f} µs")
        report.append(f"- Mean pages actually transferred (after mask): "
                      f"**{rows_yes['n_pages_actual'].mean():.1f}** of 256 top-k slots")
        report.append(f"- Mean bytes actually transferred: "
                      f"**{rows_yes['bytes_actual'].mean()/1024:.1f} KB**")
        report.append(f"- Effective PCIe bandwidth: mean **{rows_yes['bandwidth_GBps'].mean():.2f} GB/s** "
                      f"(p50 {rows_yes['bandwidth_GBps'].median():.2f}, "
                      f"p99 {rows_yes['bandwidth_GBps'].quantile(0.99):.2f}; A100 practical peak ~25 GB/s)")
        report.append(f"- Pearson correlation `n_pages_actual` ↔ `recall_sync_us`: "
                      f"**{_safe_corr(rows_yes['n_pages_actual'].to_numpy(), rows_yes['recall_sync_us'].to_numpy()):.4f}**")
        report.append("")
        report.append("![Recall latency vs pages](plots/q4_latency_vs_pages.png)\n")

    out = args.output_dir / "timing.md"
    out.write_text("\n".join(report))
    print(f"\nReport: {out}")


if __name__ == "__main__":
    main()
