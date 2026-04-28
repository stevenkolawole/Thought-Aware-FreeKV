"""Analyze whether cos_sim dips correlate with ThinKV-style transition keywords
in the generated text.

Inputs (expected):
  modal_logs/verify_dips/verify_dips/corr_<pid>.csv
  modal_logs/verify_dips/verify_dips/recall_<pid>.csv
  modal_logs/verify_dips/verify_dips/tokens_<pid>.csv
  (optional) modal_logs/verify_dips/verify_dips/preds.jsonl

Outputs:
  analysis/verify_dips/report.md
  analysis/verify_dips/plots/*.png

Approach:
  1. For each problem, join the corr (layer 0 row per step) CSV with the tokens
     CSV on step_id, giving a per-step frame of (cos_sim, token_text).
  2. Mark each step as a "transition step" if its token matches a T-keyword
     regex (wait, hmm, actually, alternatively, etc.).
  3. Ask: do cos_sim dips predict transition steps?
        - Compare mean cos_sim at transition vs non-transition steps
        - Bucket steps into cos_sim quintiles; report P(transition) per bucket
        - Simple AUC of (1 - cos_sim) as predictor of transition step
  4. Overlay transition-step marks on the per-problem cos_sim time-series plot
     so visual correlation is obvious.

Pure pandas + matplotlib — no compute cost, runs on laptop in <1 min.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent

# ThinKV (per project README) marks T-segments with these keywords. Match
# whole words, case-insensitive. "Wait" and "But wait" are the flagship
# markers; we broaden slightly to cover common DeepSeek-R1 phrasings.
T_KEYWORDS = [
    r"wait",
    r"hmm",
    r"actually",
    r"alternatively",
    r"instead",
    r"however",
    r"hold on",
    r"on second thought",
    r"let me (?:re|check|verify|think|reconsider)",
    r"but (?:wait|actually)",
    r"oh",
    r"oops",
]
T_REGEX = re.compile(
    r"(?i)(?<![a-z])(?:" + "|".join(T_KEYWORDS) + r")(?![a-z])"
)


def load_per_problem(log_dir: Path) -> dict[str, pd.DataFrame]:
    """For each problem, join layer-0 corr rows with tokens on step_id."""
    out: dict[str, pd.DataFrame] = {}
    for corr_path in sorted(log_dir.glob("corr_*.csv")):
        pid = corr_path.stem.removeprefix("corr_")
        tok_path = log_dir / f"tokens_{pid}.csv"
        if not tok_path.exists():
            print(f"  [skip] tokens_{pid}.csv missing")
            continue
        corr = pd.read_csv(corr_path)
        if corr.empty:
            continue
        corr0 = corr[corr["layer_id"] == 0].copy()
        tok = pd.read_csv(tok_path)
        df = corr0.merge(tok, on="step_id", how="inner")
        df["is_transition"] = df["token_text"].fillna("").str.contains(
            T_REGEX, regex=True
        )
        out[pid] = df
    return out


def auc_roc(score: np.ndarray, label: np.ndarray) -> float:
    """AUC of a continuous score as a predictor of a binary label.
    Higher score → predict label=1. Implementation via Mann-Whitney U."""
    pos = score[label == 1]
    neg = score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    all_ = np.concatenate([pos, neg])
    ranks = pd.Series(all_).rank(method="average").to_numpy()
    pos_ranks = ranks[: len(pos)]
    U = pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2.0
    return U / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dir", type=Path,
        default=REPO_ROOT / "modal_logs" / "verify_dips" / "verify_dips",
    )
    ap.add_argument(
        "--output_dir", type=Path,
        default=REPO_ROOT / "analysis" / "verify_dips",
    )
    args = ap.parse_args()

    log_dir = args.input_dir
    out_dir = args.output_dir
    plot_dir = out_dir / "plots"
    out_dir.mkdir(exist_ok=True, parents=True)
    plot_dir.mkdir(exist_ok=True, parents=True)

    print(f"Loading verify_dips CSVs from {log_dir}")
    per_problem = load_per_problem(log_dir)
    if not per_problem:
        raise SystemExit(
            f"No matched corr/tokens pairs under {log_dir}. "
            f"Run scripts/fetch_logs.sh verify_dips first."
        )

    # Combined frame across problems
    combined = pd.concat(
        [df.assign(problem_id=pid) for pid, df in per_problem.items()],
        ignore_index=True,
    )
    print(f"  matched steps across {len(per_problem)} problems: "
          f"{len(combined):,}")
    print(f"  transition-marked steps: "
          f"{int(combined['is_transition'].sum()):,} "
          f"({combined['is_transition'].mean():.4f})")

    # -------------------------------------------------------------------
    # 1. Global: mean cos_sim at transition vs non-transition
    # -------------------------------------------------------------------
    grouped = combined.groupby("is_transition")["cos_sim"].agg(
        ["count", "mean", "std", "median", "min"]
    )
    print("\n[1] cos_sim stats by transition flag:")
    print(grouped.to_string())

    # -------------------------------------------------------------------
    # 2. Quintile bucket: P(transition | cos_sim bucket)
    # -------------------------------------------------------------------
    try:
        combined["q"] = pd.qcut(
            combined["cos_sim"], q=5, labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"],
            duplicates="drop",
        )
        by_q = combined.groupby("q", observed=True).agg(
            n=("is_transition", "size"),
            n_T=("is_transition", "sum"),
            p_T=("is_transition", "mean"),
            cos_low=("cos_sim", "min"),
            cos_high=("cos_sim", "max"),
        )
    except ValueError as e:
        by_q = None
        print(f"  quintile compute failed: {e}")
    print("\n[2] P(transition) per cos_sim quintile:")
    if by_q is not None:
        print(by_q.to_string())

    # -------------------------------------------------------------------
    # 3. AUC: can 1 - cos_sim predict transition step?
    # -------------------------------------------------------------------
    auc = auc_roc(
        score=(1.0 - combined["cos_sim"].to_numpy()),
        label=combined["is_transition"].astype(int).to_numpy(),
    )
    print(f"\n[3] AUC of (1 - cos_sim) as transition predictor: {auc:.4f}")

    # Same for sim_ema (slower signal)
    auc_ema = auc_roc(
        score=(1.0 - combined["sim_ema"].to_numpy()),
        label=combined["is_transition"].astype(int).to_numpy(),
    )
    print(f"    AUC of (1 - sim_ema) as transition predictor: {auc_ema:.4f}")

    # -------------------------------------------------------------------
    # 4. Nearest-dip-distance analysis: when a transition fires, how far
    #    is it from the nearest local sim minimum? Informative even if the
    #    AUC is modest.
    # -------------------------------------------------------------------
    near_stats = []
    for pid, df in per_problem.items():
        s = df.sort_values("step_id").reset_index(drop=True)
        cos = s["cos_sim"].to_numpy()
        # A local dip = step where cos_sim is below the rolling-median by some delta
        window = 31
        roll_med = pd.Series(cos).rolling(window, center=True, min_periods=1).median().to_numpy()
        dip_mask = (roll_med - cos) > 0.03
        dip_steps = np.where(dip_mask)[0]
        trans_steps = np.where(s["is_transition"].to_numpy())[0]
        if len(trans_steps) == 0 or len(dip_steps) == 0:
            continue
        # For each transition, distance to nearest dip
        d = np.min(
            np.abs(trans_steps[:, None] - dip_steps[None, :]), axis=1
        )
        near_stats.append({
            "problem_id": pid,
            "n_transitions": len(trans_steps),
            "n_dips": len(dip_steps),
            "median_dist_to_dip": int(np.median(d)),
            "pct_within_2_steps": float(np.mean(d <= 2)),
            "pct_within_5_steps": float(np.mean(d <= 5)),
        })
    near_df = pd.DataFrame(near_stats)
    print("\n[4] Transition-to-nearest-dip distance (per problem):")
    if not near_df.empty:
        print(near_df.to_string(index=False))

    # -------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------

    # (A) Per-problem time series with transition markers overlaid
    n_prob = len(per_problem)
    fig, axes = plt.subplots(n_prob, 1, figsize=(14, 2.3 * n_prob),
                             sharex=False, squeeze=False)
    for i, (pid, df) in enumerate(sorted(per_problem.items())):
        ax = axes[i, 0]
        s = df.sort_values("step_id")
        ax.plot(s["step_id"], s["cos_sim"], lw=0.6, alpha=0.6,
                label="cos_sim (layer 0)", color="tab:blue")
        ax.plot(s["step_id"], s["sim_ema"], lw=1.2, label="sim_ema",
                color="tab:orange")
        t_steps = s.loc[s["is_transition"], "step_id"]
        if len(t_steps) > 0:
            ax.scatter(
                t_steps, [0.05] * len(t_steps), marker="v",
                color="red", s=25, alpha=0.85,
                label=f"T-keyword token (n={len(t_steps)})",
            )
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel(pid, fontsize=9)
        ax.set_xlabel("step_id")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=7)
    fig.suptitle(
        "cos_sim (blue), sim_ema (orange), transition-keyword tokens (red ▼) "
        "per decode step"
    )
    fig.tight_layout()
    p = plot_dir / "per_problem_transitions.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"  wrote {p}")

    # (B) Histogram: cos_sim at transition vs non-transition steps
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0.4, 1.0, 61)
    ax.hist(
        combined.loc[~combined["is_transition"], "cos_sim"],
        bins=bins, alpha=0.55, label="non-transition steps",
        color="tab:blue", density=True,
    )
    ax.hist(
        combined.loc[combined["is_transition"], "cos_sim"],
        bins=bins, alpha=0.75, label="transition-keyword steps",
        color="tab:red", density=True,
    )
    ax.set_xlabel("cos_sim (mean across q heads)")
    ax.set_ylabel("density")
    ax.set_title(
        f"cos_sim distribution — transition vs non-transition steps "
        f"(AUC = {auc:.3f})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plot_dir / "cos_sim_hist.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"  wrote {p}")

    # (C) Bucket bar chart
    if by_q is not None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(by_q.index.astype(str), by_q["p_T"].values, color="tab:red")
        ax.set_ylabel("P(transition keyword in token)")
        ax.set_xlabel("cos_sim quintile")
        ax.set_title("P(transition) by cos_sim quintile (all problems pooled)")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        p = plot_dir / "p_transition_by_quintile.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        print(f"  wrote {p}")

    # -------------------------------------------------------------------
    # Markdown report
    # -------------------------------------------------------------------
    lines = []
    lines.append("# Thought-Aware-FreeKV — `verify_dips` analysis\n")
    lines.append(
        "Question: do cos_sim dips during decode actually correspond to "
        "ThinKV-style transition markers (\"wait\", \"hmm\", \"actually\", "
        "etc.) in the generated text?\n"
    )
    rel_input = log_dir.relative_to(REPO_ROOT) if log_dir.is_absolute() else log_dir
    lines.append(
        f"Source: `{rel_input}/{{corr,recall,tokens}}_*.csv` — "
        f"{len(per_problem)} problems, {len(combined):,} aligned decode steps.\n"
    )
    lines.append(
        f"**Transition-marked steps: {int(combined['is_transition'].sum()):,} "
        f"({combined['is_transition'].mean():.4%} of all steps).** "
        "Markers determined by regex against the decoded token text "
        f"(see `T_KEYWORDS` in `scripts/analyze_dips.py`).\n"
    )

    lines.append("\n## 1. cos_sim at transition vs non-transition\n")
    lines.append("| flag | count | mean | std | median | min |")
    lines.append("|---|---|---|---|---|---|")
    for is_t, row in grouped.iterrows():
        lab = "transition" if is_t else "other"
        lines.append(
            f"| {lab} | {int(row['count']):,} | {row['mean']:.4f} | "
            f"{row['std']:.4f} | {row['median']:.4f} | {row['min']:.4f} |"
        )

    lines.append("\n## 2. P(transition) per cos_sim quintile\n")
    if by_q is not None:
        lines.append("Quintile boundaries are adaptive to the observed "
                     "distribution. If cos_sim dips predict transitions, "
                     "the lowest quintile should have a much higher P(T).\n")
        lines.append("| bucket | n | n_T | P(T) | cos range |")
        lines.append("|---|---|---|---|---|")
        for q, row in by_q.iterrows():
            lines.append(
                f"| {q} | {int(row['n']):,} | {int(row['n_T']):,} | "
                f"{row['p_T']:.4%} | [{row['cos_low']:.3f}, "
                f"{row['cos_high']:.3f}] |"
            )

    lines.append("\n## 3. Predictive quality\n")
    lines.append(f"- **AUC of (1 − cos_sim)** as transition predictor: "
                 f"**{auc:.4f}** — 0.5 is chance, 1.0 is perfect.")
    lines.append(f"- AUC of (1 − sim_ema): {auc_ema:.4f}")
    lines.append(
        "\nA value meaningfully above 0.5 is direct evidence for the "
        "project's premise that cos-sim dips carry thought-transition "
        "signal, independent of attention-weight features.\n"
    )

    lines.append("## 4. Per-problem transition-to-dip distance\n")
    if not near_df.empty:
        lines.append(
            "For each transition-marked step, the distance to the nearest "
            "local cos_sim dip (rolling-median dip of >0.03 within a "
            "31-step window).\n"
        )
        lines.append("| problem | #trans | #dips | median dist | ≤2 steps | ≤5 steps |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in near_df.iterrows():
            lines.append(
                f"| `{r['problem_id']}` | {int(r['n_transitions'])} | "
                f"{int(r['n_dips'])} | {int(r['median_dist_to_dip'])} | "
                f"{r['pct_within_2_steps']:.1%} | {r['pct_within_5_steps']:.1%} |"
            )

    lines.append("\n## 5. Plots\n")
    lines.append("- [per_problem_transitions.png](plots/per_problem_transitions.png)")
    lines.append("- [cos_sim_hist.png](plots/cos_sim_hist.png)")
    if by_q is not None:
        lines.append("- [p_transition_by_quintile.png](plots/p_transition_by_quintile.png)")

    report_path = out_dir / "dips_report.md"
    report_path.write_text("\n".join(lines))
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
