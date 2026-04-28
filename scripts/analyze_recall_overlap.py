"""How much of each correction's top-k is *actually new* vs already in cache?

For each logged recall row we know `n_pages` = sum over all 8 kv groups of
`rids[i, j, 0]` — i.e. the number of NEW pages whose corrected top-k didn't
overlap with the speculative top-k. The TOTAL top-k capacity is:

    n_groups × topk_per_group = 8 × 32 = 256 pages per (step, layer)

So the fraction *reused* = 1 − n_pages / 256.

Hypothesis: most corrections preserve most of the cache. If true, a large
chunk of the bandwidth implied by the 89% trigger rate is "wasted" — sim
dropped, but the top-k set didn't actually change much, so we paid PCIe to
re-fetch pages that were already there.

We split the analysis three ways:
  1. ALL logged recalls
  2. recalls attributable to a `need_corr=1` event (joined via step_id, layer_id)
  3. the very first recall per (problem, layer) — the synchronous initial fetch

Caveat: our recall log sums over ALL 8 kv-groups, not just the drifted ones.
That over-reports actual bandwidth, but for THIS analysis (set-diff fraction
of the 256-slot top-k) it's the right thing — we want "how many of the
top-k slots would have to change if we re-selected against q_t", not "how
many bytes actually crossed PCIe."
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent

# Per our config: budget=2048 tokens / page_size=32 = 64 pages per layer total
# but only the top-k slots count for correction (sink+window are fixed).
# pred.py passes page_topks = page_budgets - 1 = 63, then InferState adjusts:
#   k = 63 - n_sink_pages - (n_win_pages - 1)
#     = 63 - 16 - 15 = 32
# Times 8 kv-groups = 256.
TOPK_PER_GROUP = 32
N_KV_GROUPS = 8
TOTAL_TOPK = TOPK_PER_GROUP * N_KV_GROUPS


def load_run(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr_dfs, recall_dfs = [], []
    for p in sorted(input_dir.glob("corr_*.csv")):
        pid = p.stem.removeprefix("corr_")
        df = pd.read_csv(p)
        if df.empty:
            continue
        df["problem_id"] = pid
        corr_dfs.append(df)
    for p in sorted(input_dir.glob("recall_*.csv")):
        pid = p.stem.removeprefix("recall_")
        df = pd.read_csv(p)
        if df.empty:
            continue
        df["problem_id"] = pid
        recall_dfs.append(df)
    return (
        pd.concat(corr_dfs, ignore_index=True) if corr_dfs else pd.DataFrame(),
        pd.concat(recall_dfs, ignore_index=True) if recall_dfs else pd.DataFrame(),
    )


def attribute_recalls(corr: pd.DataFrame, recall: pd.DataFrame) -> pd.DataFrame:
    """Tag each recall row with what triggered it.

    For (problem, layer) the very first recall step_id is the first-time-past-
    budget synchronous fetch. Subsequent recalls on need_corr=1 step/layer
    pairs are correction events (one synchronous + one async post-correction).
    Recalls on need_corr=0 step/layer pairs use the fused pool kernel and
    aren't logged at all.
    """
    # find first recall step_id per (problem, layer)
    first = (
        recall.groupby(["problem_id", "layer_id"])["step_id"]
        .min().reset_index().rename(columns={"step_id": "first_step"})
    )
    recall = recall.merge(first, on=["problem_id", "layer_id"], how="left")
    # left-join with corr to know need_corr at each (problem, step, layer)
    recall = recall.merge(
        corr[["problem_id", "step_id", "layer_id", "need_corr"]],
        on=["problem_id", "step_id", "layer_id"],
        how="left",
    )
    def _kind(row):
        if row["step_id"] == row["first_step"]:
            return "first_step"
        if row["need_corr"] == 1:
            return "correction"
        return "other"  # shouldn't really happen given the code paths
    recall["kind"] = recall.apply(_kind, axis=1)
    recall["new_pages"] = recall["n_pages"]
    recall["new_frac"] = recall["new_pages"] / TOTAL_TOPK
    recall["reuse_frac"] = 1.0 - recall["new_frac"]
    # avg per group
    recall["new_per_group"] = recall["new_pages"] / N_KV_GROUPS
    recall["reuse_frac_per_group"] = 1.0 - recall["new_per_group"] / TOPK_PER_GROUP
    return recall


def summarize(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {"label": label, "n": 0}
    s = df["reuse_frac"]
    return {
        "label": label,
        "n": len(df),
        "n_pages_mean": float(df["n_pages"].mean()),
        "new_frac_mean": float((df["n_pages"] / TOTAL_TOPK).mean()),
        "reuse_frac_mean": float(s.mean()),
        "reuse_frac_median": float(s.median()),
        "reuse_frac_p10": float(s.quantile(0.10)),
        "reuse_frac_p90": float(s.quantile(0.90)),
        "pct_high_reuse": float((s >= 0.75).mean()),  # >=75% reused
        "pct_full_reuse": float((s >= 0.99).mean()),  # >=99% reused
        "pct_zero_reuse": float((s <= 0.01).mean()),  # <=1% reused (full turnover)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=Path,
                    default=REPO_ROOT / "modal_logs" / "full_aime" / "full_aime")
    ap.add_argument("--output_dir", type=Path,
                    default=REPO_ROOT / "analysis" / "full_aime")
    args = ap.parse_args()

    corr, recall = load_run(args.input_dir)
    if corr.empty or recall.empty:
        raise SystemExit(f"empty data under {args.input_dir}")

    print(f"corr rows:   {len(corr):>10,}")
    print(f"recall rows: {len(recall):>10,}")

    # need_corr columns from corr CSV
    n_trigger = int(corr["need_corr"].sum())
    print(f"need_corr=1 events:   {n_trigger:,}")
    print(f"max possible reuse  : 1 - 0/256 = 100% (no pages changed)")
    print(f"min possible reuse  : 1 - 256/256 = 0%   (entire top-k churned)\n")

    rec = attribute_recalls(corr, recall)
    by_kind = (
        rec.groupby("kind").apply(lambda g: pd.Series(summarize(g, g.name)))
        .drop(columns=["label"])
    )
    print("=== summary by recall kind ===")
    print(by_kind.round(4).to_string())
    print()
    overall = summarize(rec, "all")
    print("=== overall (all logged recalls pooled) ===")
    for k, v in overall.items():
        if k == "label":
            continue
        if isinstance(v, float):
            print(f"  {k:<22s} {v:.4f}")
        else:
            print(f"  {k:<22s} {v:,}")

    # plots
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Histogram of n_pages per recall
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bins = np.arange(0, TOTAL_TOPK + 5, 4)
    for kind, color in [("correction", "tab:red"),
                        ("first_step", "tab:blue"),
                        ("other", "tab:gray")]:
        sub = rec[rec["kind"] == kind]
        if sub.empty:
            continue
        ax.hist(sub["n_pages"], bins=bins, alpha=0.55,
                label=f"{kind} (n={len(sub):,}, mean={sub['n_pages'].mean():.1f})",
                color=color, density=True)
    ax.axvline(TOTAL_TOPK, color="k", lw=1.0, ls="--",
               label=f"total top-k capacity = {TOTAL_TOPK}")
    ax.set_xlabel("new pages per recall (= rids[:,:,0].sum() summed over 8 groups)")
    ax.set_ylabel("density")
    ax.set_title("How many pages a recall actually changes vs the 256-slot top-k cap")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plot_dir / "recall_overlap_hist.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"\n  wrote {p}")

    # Reuse-fraction CDF for corrections vs first-step
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for kind, color in [("correction", "tab:red"),
                        ("first_step", "tab:blue")]:
        sub = rec[rec["kind"] == kind]
        if sub.empty:
            continue
        s = np.sort(sub["reuse_frac"].to_numpy())
        y = np.arange(1, len(s) + 1) / len(s)
        ax.plot(s, y, lw=2.0, label=f"{kind} (n={len(sub):,})", color=color)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("fraction of top-k slots reused (1 − n_pages/256)")
    ax.set_ylabel("CDF")
    ax.set_title("How much of the cache is reused per recall — empirical CDF")
    ax.grid(True, alpha=0.3)
    ax.axvline(0.75, color="gray", lw=0.6, ls=":")
    ax.axvline(0.99, color="gray", lw=0.6, ls=":")
    ax.legend()
    fig.tight_layout()
    p = plot_dir / "recall_reuse_cdf.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"  wrote {p}")

    # Per-problem average reuse fraction (corrections only)
    corr_only = rec[rec["kind"] == "correction"]
    if not corr_only.empty:
        per_pid = corr_only.groupby("problem_id")["reuse_frac"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(per_pid.index, per_pid.values, color="tab:red")
        ax.set_xlim(0, 1.0)
        ax.axvline(per_pid.mean(), color="k", ls="--",
                   label=f"avg = {per_pid.mean():.3f}")
        ax.set_xlabel("mean reuse fraction during corrections")
        ax.set_title("Per-problem: of the 256 top-k slots, "
                     "what fraction was already in cache when correction fired?")
        ax.legend()
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        p = plot_dir / "recall_reuse_per_problem.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
