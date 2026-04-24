"""Analyze the Thought-Aware-FreeKV baseline instrumentation CSVs.

Inputs (expected):
  modal_logs/baseline/baseline/corr_<problem_id>.csv
  modal_logs/baseline/baseline/recall_<problem_id>.csv

Outputs:
  analysis/report.md
  analysis/plots/*.png

Answers:
  1. Overall correction rate on AIME24 (issue #1 reproduction number)
  2. Thought-type vs correction co-occurrence (issue #6 hypothesis)
  3. Per-problem time series (issue #6 figure)
  4. Cosine-similarity distribution by thought type
  5. Per-layer correction rate
  6. Recall bandwidth per problem
  7. If-we-predicted-corrections-from-thought baseline quality

This is pure analysis — no Modal, no GPU, runs on a laptop in ~1 min.
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

# Thought-type labels matching utils.ThoughtType
TT = {0: "R", 1: "E", 2: "T"}


def load_csvs(log_dir: Path, prefix: str) -> pd.DataFrame:
    frames = []
    for p in sorted(log_dir.glob(f"{prefix}_*.csv")):
        m = re.match(rf"{prefix}_(.+)\.csv", p.name)
        if not m:
            continue
        pid = m.group(1)
        df = pd.read_csv(p)
        if df.empty:
            continue
        df["problem_id"] = pid
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def drop_truncated_problem(corr: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Heuristic: the timeout-cancelled problem's max step_id will be much
    lower than the median, and its row count similarly anomalous. Flag and
    drop it so it doesn't skew aggregate stats."""
    by_problem = corr.groupby("problem_id")["step_id"].max()
    med = by_problem.median()
    # Anything with < 10% of the median max-step is probably the truncated one
    candidates = by_problem[by_problem < 0.1 * med]
    if len(candidates) == 0:
        return corr, None
    drop_id = candidates.idxmin()
    return corr[corr["problem_id"] != drop_id].copy(), drop_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dir", type=Path,
        default=REPO_ROOT / "modal_logs" / "baseline" / "baseline",
        help="Directory containing corr_*.csv and recall_*.csv",
    )
    ap.add_argument(
        "--output_dir", type=Path,
        default=REPO_ROOT / "analysis" / "baseline",
        help="Directory to write report.md and plots/",
    )
    ap.add_argument(
        "--label", type=str, default="baseline",
        help="Short label shown in the report title",
    )
    args = ap.parse_args()

    log_dir = args.input_dir
    out_dir = args.output_dir
    plot_dir = out_dir / "plots"
    out_dir.mkdir(exist_ok=True, parents=True)
    plot_dir.mkdir(exist_ok=True, parents=True)

    print(f"Loading CSVs from {log_dir}")
    corr = load_csvs(log_dir, "corr")
    recall = load_csvs(log_dir, "recall")
    print(f"  corr rows:   {len(corr):>10,}")
    print(f"  recall rows: {len(recall):>10,}")
    print(f"  problems:    {corr['problem_id'].nunique()}")

    corr, dropped = drop_truncated_problem(corr)
    if dropped:
        recall = recall[recall["problem_id"] != dropped].copy()
        print(f"  (dropping {dropped} — truncated by Modal timeout)")

    problems = sorted(corr["problem_id"].unique())
    n_layers = int(corr["layer_id"].max()) + 1
    print(f"  kept:        {len(problems)} problems, {n_layers} layers")

    corr["thought_label"] = corr["thought_type"].map(TT)

    # ================================================================
    # 1. Headline correction rate
    # ================================================================
    overall_rate = corr["need_corr"].mean()
    per_problem = corr.groupby("problem_id")["need_corr"].mean().rename("rate")
    print(f"\n[1] Overall correction rate: {overall_rate:.4f} "
          f"({int(corr['need_corr'].sum()):,}/{len(corr):,})")

    # ================================================================
    # 2. Thought-type vs correction co-occurrence
    # ================================================================
    # Use per-(step, problem) rows at layer 0 for the thought_type label
    # (thought_type is propagated to all layers within a step, so any layer works,
    #  but layer 0 is the canonical place where it's computed)
    step_labels = (
        corr[corr["layer_id"] == 0]
        [["problem_id", "step_id", "thought_type", "thought_label",
          "cos_sim", "sim_ema", "need_corr"]]
        .copy()
    )
    thought_dist = step_labels["thought_label"].value_counts(normalize=True)
    corr_by_thought = (
        corr.groupby("thought_label")["need_corr"].mean().rename("corr_rate")
    )
    cross = pd.crosstab(corr["thought_label"], corr["need_corr"], margins=True)
    cross.columns = ["no_corr", "corr", "total"]
    print("\n[2] Thought-type distribution (per-step, layer 0):")
    print(thought_dist.to_string())
    print("\n    Correction rate | thought_type (all layers):")
    print(corr_by_thought.to_string())

    # ================================================================
    # 3. If we predicted corrections from thought_type=='T' alone,
    #    what's the F1?
    # ================================================================
    pred_is_T = (corr["thought_label"] == "T")
    actual_corr = corr["need_corr"].astype(bool)
    tp = int((pred_is_T & actual_corr).sum())
    fp = int((pred_is_T & ~actual_corr).sum())
    fn = int((~pred_is_T & actual_corr).sum())
    tn = int((~pred_is_T & ~actual_corr).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"\n[3] 'thought=T' as corr predictor: "
          f"precision={prec:.3f} recall={rec:.3f} F1={f1:.3f} "
          f"(tp={tp:,} fp={fp:,} fn={fn:,} tn={tn:,})")

    # ================================================================
    # 4. Per-layer correction rate
    # ================================================================
    per_layer = corr.groupby("layer_id")["need_corr"].mean()

    # ================================================================
    # 5. Recall bandwidth
    # ================================================================
    bytes_per_problem_gb = (
        recall.groupby("problem_id")["bytes"].sum() / (1 << 30)
    ).rename("gb")

    # ================================================================
    # Plots
    # ================================================================

    # Fig: per-problem time series — one row per problem, showing cos_sim,
    # sim_ema, thought_type, need_corr aligned to step_id. Use layer 0 samples
    # (single cos_sim per step to avoid 32x overplotting).
    n_prob = len(problems)
    fig, axes = plt.subplots(n_prob, 1, figsize=(13, 2.0 * n_prob),
                             sharex=False, squeeze=False)
    for i, pid in enumerate(problems):
        ax = axes[i, 0]
        s = step_labels[step_labels["problem_id"] == pid].sort_values("step_id")
        ax.plot(s["step_id"], s["cos_sim"], lw=0.6, alpha=0.6,
                label="cos_sim (layer 0)")
        ax.plot(s["step_id"], s["sim_ema"], lw=1.0, label="sim_ema")
        # Mark corrections with vertical red bars
        corr_steps = s.loc[s["need_corr"] == 1, "step_id"]
        if len(corr_steps) > 0:
            ax.vlines(corr_steps, ymin=0.0, ymax=0.05, color="red", alpha=0.3,
                      lw=0.4)
        # Shade T-segments
        t_steps = s.loc[s["thought_label"] == "T", "step_id"]
        if len(t_steps) > 0:
            ax.vlines(t_steps, ymin=0.95, ymax=1.0, color="purple", alpha=0.25,
                      lw=0.4)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel(pid, fontsize=8)
        ax.set_xlabel("step_id")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=7)
    fig.suptitle("Per-problem: cos_sim + sim_ema (line), corrections (red bars bottom), "
                 "thought=T (purple bars top)")
    fig.tight_layout()
    p = plot_dir / "per_problem_timeseries.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"  wrote {p}")

    # Fig: cos_sim histogram by thought_type (sampled to keep plotting fast)
    sample = corr.sample(min(200_000, len(corr)), random_state=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    for lab, color in [("R", "tab:green"), ("E", "tab:orange"),
                       ("T", "tab:purple")]:
        s = sample.loc[sample["thought_label"] == lab, "cos_sim"]
        if len(s) == 0:
            continue
        ax.hist(s, bins=60, alpha=0.5, label=f"{lab} (n={len(s):,})",
                color=color)
    ax.set_xlabel("cos_sim between adjacent decode-step queries")
    ax.set_ylabel("count (200k sampled rows)")
    ax.set_title("Cosine-similarity distribution, conditioned on thought_type")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plot_dir / "cos_sim_by_thought.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"  wrote {p}")

    # Fig: per-layer correction rate
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(per_layer.index, per_layer.values, color="tab:blue")
    ax.set_xlabel("layer_id")
    ax.set_ylabel("P(need_corr)")
    ax.set_title("Correction rate per attention layer")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = plot_dir / "corr_rate_per_layer.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"  wrote {p}")

    # Fig: per-problem correction rate + bytes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
    order = per_problem.sort_values().index
    ax1.barh(order, per_problem.reindex(order).values, color="tab:red")
    ax1.set_xlim(0, 1)
    ax1.axvline(overall_rate, color="k", ls="--",
                label=f"avg = {overall_rate:.3f}")
    ax1.set_xlabel("P(need_corr)")
    ax1.set_title("Correction rate per problem")
    ax1.legend()
    ax1.grid(True, axis="x", alpha=0.3)

    b_order = bytes_per_problem_gb.reindex(order)
    ax2.barh(order, b_order.values, color="tab:blue")
    ax2.set_xlabel("total PCIe bytes recalled (GB)")
    ax2.set_title("Per-problem recall bandwidth")
    ax2.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    p = plot_dir / "per_problem_rate_and_bw.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"  wrote {p}")

    # ================================================================
    # Markdown report
    # ================================================================
    lines = []
    lines.append(f"# Thought-Aware-FreeKV analysis — run `{args.label}`\n")
    rel_input = log_dir.relative_to(REPO_ROOT) if log_dir.is_absolute() else log_dir
    lines.append(f"Source: `{rel_input}/{{corr,recall}}_*.csv` "
                 f"({len(problems)} problems, {n_layers} layers)\n")
    if dropped:
        lines.append(f"> Dropped `{dropped}` — truncated by the 4h Modal "
                     "function timeout.\n")

    lines.append("## 1. Headline correction rate\n")
    lines.append(f"- **Overall**: {overall_rate:.4f} "
                 f"({int(corr['need_corr'].sum()):,} of {len(corr):,} "
                 f"(step, layer) checks triggered a correction)")
    lines.append(f"- **Per-problem mean**: {per_problem.mean():.4f}, "
                 f"median {per_problem.median():.4f}, "
                 f"range [{per_problem.min():.4f}, {per_problem.max():.4f}]")
    lines.append("- Paper's AIME24 claim: 43–52%. "
                 + ("**Reproduced.**" if 0.30 <= overall_rate <= 0.65
                    else "**Does not match — investigate.**"))
    lines.append("\n### Per-problem\n")
    lines.append("| Problem | Corr rate | Steps | Recall GB |")
    lines.append("|---|---|---|---|")
    for pid in order:
        n_steps = int(step_labels[step_labels["problem_id"] == pid]["step_id"].max() + 1)
        gb = float(bytes_per_problem_gb.get(pid, np.nan))
        lines.append(f"| `{pid}` | {per_problem[pid]:.4f} | {n_steps:,} | "
                     f"{gb:.2f} |")

    lines.append("\n## 2. Thought-type co-occurrence with corrections\n")
    lines.append("Thought-type distribution (per decode step, layer 0):")
    lines.append("")
    for lab, pct in thought_dist.items():
        lines.append(f"- **{lab}**: {pct:.4f}")
    lines.append("")
    lines.append("Correction rate conditioned on thought_type (all layers):")
    lines.append("")
    lines.append("| thought_type | P(need_corr) | rows |")
    lines.append("|---|---|---|")
    for lab in ["R", "E", "T"]:
        if lab in corr_by_thought.index:
            sub = corr[corr["thought_label"] == lab]
            lines.append(f"| {lab} | {corr_by_thought[lab]:.4f} | {len(sub):,} |")
    lines.append("")
    lines.append("Cross-tab (thought × need_corr, all layers):\n")
    lines.append("```")
    lines.append(cross.to_string())
    lines.append("```\n")

    lines.append("## 3. `thought_type == T` as a correction predictor\n")
    lines.append(f"- precision = {prec:.4f}")
    lines.append(f"- recall    = {rec:.4f}")
    lines.append(f"- F1        = {f1:.4f}")
    lines.append(f"- tp={tp:,} fp={fp:,} fn={fn:,} tn={tn:,}")
    lines.append(
        "\nInterpretation: a naive \"predict correction iff thought==T\" "
        "classifier. Low recall means most corrections happen during R/E; "
        "if so, the EMA thresholds need retuning or the cos-sim signal "
        "itself is the better direct predictor.\n"
    )

    lines.append("## 4. Per-layer correction rate\n")
    lines.append("| layer | P(need_corr) |")
    lines.append("|---|---|")
    for L in range(n_layers):
        lines.append(f"| {L} | {per_layer.get(L, float('nan')):.4f} |")

    lines.append("\n## 5. Plots\n")
    lines.append("- [per_problem_timeseries.png](plots/per_problem_timeseries.png)")
    lines.append("- [cos_sim_by_thought.png](plots/cos_sim_by_thought.png)")
    lines.append("- [corr_rate_per_layer.png](plots/corr_rate_per_layer.png)")
    lines.append("- [per_problem_rate_and_bw.png](plots/per_problem_rate_and_bw.png)")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
