"""Compare fetch_interval={1,4} runs on AIME

Inputs (after `modal volume get tafkv-logs <subdir>/ results/<subdir>/`):
  results/math50_fetch1/tbt_<pid>.csv
  results/math50_fetch1/preds_ds-r1-llama-8b_MATH50.jsonl
  results/math50_fetch2/...
  results/math50_fetch8/...

Outputs (printed + saved to results/fetch_interval_analysis.md):
  - Per-run latency summary (mean/median total latency, mean TBT per step)
  - Per-run accuracy (MATH-style string match on \\boxed{})
  - Cross-run comparison table

Usage:
  python scripts/analyze_fetch_interval.py
  python scripts/analyze_fetch_interval.py --results_dir /path/to/results
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_TOKENS = 16384
RUNS = {
    "F=1 (baseline)": "aime_debug_fetch1",
    "F=4":            "aime_debug_fetch4",
}

BOXED = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}")


# ---------------------------------------------------------------------------
# Grading helpers
# ---------------------------------------------------------------------------

def extract_last_boxed(text: str) -> str | None:
    blocks = BOXED.findall(text)
    return blocks[-1] if blocks else None


def normalize(s: str | None) -> str | None:
    if s is None:
        return None
    return s.replace(" ", "").replace("\\!", "").replace("\\,", "")


def grade_preds(preds_path: Path) -> pd.DataFrame:
    rows = []
    with preds_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec.get("id")
            ref = normalize(str(rec.get("answer", "")))
            pred_text = rec.get("pred") or ""
            boxed = extract_last_boxed(pred_text)
            extracted = normalize(boxed)
            correct = extracted is not None and ref is not None and extracted == ref
            in_len = rec.get("input_len")
            out_len = rec.get("output_len")
            gen_tokens = (out_len - in_len) if (out_len and in_len) else None
            rows.append({
                "id": pid,
                "ref": ref,
                "extracted": extracted,
                "correct": correct,
                "gen_tokens": gen_tokens,
                "hit_cap": gen_tokens is not None and gen_tokens >= (MAX_TOKENS - 4),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Latency helpers
# ---------------------------------------------------------------------------

def load_tbt(run_dir: Path) -> pd.DataFrame:
    """Load all tbt_<pid>.csv files and return a long-form DataFrame."""
    rows = []
    for f in sorted(run_dir.glob("tbt_*.csv")):
        try:
            # Handle potentially empty or corrupted CSVs safely
            df = pd.read_csv(f)
            if df.empty:
                continue
            pid = f.stem.removeprefix("tbt_")
            df["pid"] = pid
            rows.append(df)
        except pd.errors.EmptyDataError:
            continue
            
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def latency_summary(tbt: pd.DataFrame) -> dict:
    """Per-problem total latency, then summarise across problems."""
    if tbt.empty:
        return {}
    per_problem = tbt.groupby("pid")["total_ms"].agg(
        total_ms="sum",
        n_steps="count",
        mean_tbt_ms="mean",
    ).reset_index()
    return {
        "n_problems": len(per_problem),
        "mean_total_s": per_problem["total_ms"].mean() / 1000,
        "median_total_s": per_problem["total_ms"].median() / 1000,
        "p90_total_s": per_problem["total_ms"].quantile(0.9) / 1000,
        "mean_tbt_ms": tbt["total_ms"].mean(),
        "median_tbt_ms": tbt["total_ms"].median(),
        "mean_steps": per_problem["n_steps"].mean(),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_tbt_distribution(
    run_tbts: dict[str, pd.DataFrame],
    out_path: Path,
    shared_acc: dict[str, tuple[int, int]] | None = None,
) -> None:
    """Plot per-token latency distribution for each run, with mean annotated.

    shared_acc: maps run label -> (n_correct, n_shared) on the intersected problem set.
    """
    n_runs = len(run_tbts)
    if n_runs == 0:
        return

    fig, axes = plt.subplots(1, n_runs, figsize=(7 * n_runs, 5), squeeze=False)
    colors = ["steelblue", "darkorange", "seagreen", "crimson"]

    baseline_mean: float | None = None
    for ax, (label, tbt), color in zip(axes[0], run_tbts.items(), colors):
        # Generation tokens only (step_id >= 0); step_id == -1 is the prefill decode step
        gen = tbt[tbt["step_id"] >= 0]["total_ms"].dropna()
        mean_ms = gen.mean()
        p95_ms = gen.quantile(0.95)

        if baseline_mean is None:
            baseline_mean = mean_ms

        ax.hist(gen, bins=80, range=(0, p95_ms * 1.1), color=color,
                alpha=0.75, edgecolor="white", linewidth=0.3, density=True,
                label="per-token latency")
        ax.axvline(mean_ms, color="black", linewidth=1.8, linestyle="--",
                   label=f"mean = {mean_ms:.1f} ms")

        annotation_lines = []
        if mean_ms != baseline_mean:
            speedup = baseline_mean / mean_ms
            annotation_lines.append(f"{speedup:.2f}× faster than baseline")
        if shared_acc and label in shared_acc:
            n_correct, n_shared = shared_acc[label]
            annotation_lines.append(f"acc = {n_correct}/{n_shared} ({n_correct/n_shared:.1%}) on shared")
        if annotation_lines:
            ax.text(0.97, 0.95, "\n".join(annotation_lines),
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=10, color="black",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                              edgecolor="gray", alpha=0.85))

        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Time per token (ms)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Token Generation Latency Distribution w/ Less Fetching", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Distribution plot saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results_dir",
        default=str(REPO_ROOT / "results"),
        help="Directory containing math50_fetch{1,2,8}/ subdirs",
    )
    args = ap.parse_args()
    results_dir = Path(args.results_dir)

    latency_rows = []
    accuracy_rows = []
    run_tbts: dict[str, pd.DataFrame] = {}
    run_grades: dict[str, pd.DataFrame] = {}

    for label, subdir in RUNS.items():
        run_dir = results_dir / subdir
        if not run_dir.exists():
            print(f"[warn] missing: {run_dir}")
            continue

        # --- latency ---
        tbt = load_tbt(run_dir)
        if not tbt.empty:
            run_tbts[label] = tbt
        lat = latency_summary(tbt)
        if lat:
            latency_rows.append({"run": label, **lat})

        # --- accuracy ---
        preds_files = list(run_dir.glob("preds_*.jsonl")) or list(run_dir.glob("preds.jsonl"))
        if not preds_files:
            print(f"[warn] no preds file found in {run_dir}")
            continue
        preds_path = preds_files[0]
        grades = grade_preds(preds_path)
        if grades.empty:
            print(f"[warn] empty preds in {preds_path}")
            continue
        run_grades[label] = grades
        n = len(grades)
        n_correct = int(grades["correct"].sum())
        n_extracted = int(grades["extracted"].notna().sum())
        n_capped = int(grades["hit_cap"].sum())
        accuracy_rows.append({
            "run": label,
            "problems": n,
            "correct": n_correct,
            "accuracy": f"{n_correct/n:.1%}",
            "no_extract": n - n_extracted,
            "hit_cap": n_capped,
        })

        print(f"\n{'='*60}")
        print(f"  {label}  ({subdir})")
        print(f"{'='*60}")
        if lat:
            print(f"  Latency  — mean total: {lat['mean_total_s']:.1f}s  "
                  f"median: {lat['median_total_s']:.1f}s  "
                  f"p90: {lat['p90_total_s']:.1f}s")
            print(f"             mean TBT/step: {lat['mean_tbt_ms']:.1f}ms  "
                  f"median: {lat['median_tbt_ms']:.1f}ms  "
                  f"avg steps: {lat['mean_steps']:.0f}")
        print(f"  Accuracy — {n_correct}/{n} ({n_correct/n:.1%})  "
              f"no_extract={n - n_extracted}  hit_cap={n_capped}")

        # Per-problem detail
        view = grades.copy()
        view["match"] = view.apply(
            lambda r: "✓" if r["correct"] else (
                "✗" if r["extracted"] is not None else "—"
            ),
            axis=1,
        )
        print(view[["id", "ref", "extracted", "match",
                    "gen_tokens", "hit_cap"]].to_string(index=False))

    # --- cross-run summary ---
    if not latency_rows and not accuracy_rows:
        print("\nNo results found. Run:\n"
              "  modal volume get tafkv-logs math50_fetch1/ results/math50_fetch1/\n"
              "  modal volume get tafkv-logs math50_fetch2/ results/math50_fetch2/\n"
              "  modal volume get tafkv-logs math50_fetch8/ results/math50_fetch8/")
        return

    print(f"\n{'='*60}")
    print("  CROSS-RUN COMPARISON")
    print(f"{'='*60}")

    if latency_rows:
        lat_df = pd.DataFrame(latency_rows).set_index("run")
        lat_df = lat_df.rename(columns={
            "mean_total_s":   "mean_tot(s)",
            "median_total_s": "med_tot(s)",
            "p90_total_s":    "p90_tot(s)",
            "mean_tbt_ms":    "mean_tbt(ms)",
            "median_tbt_ms":  "med_tbt(ms)",
            "mean_steps":     "avg_steps",
        })
        print("\nLatency:")
        print(lat_df[["n_problems", "mean_tot(s)", "med_tot(s)",
                       "p90_tot(s)", "mean_tbt(ms)", "avg_steps"]].to_string())

    if accuracy_rows:
        acc_df = pd.DataFrame(accuracy_rows).set_index("run")
        print("\nAccuracy (all finished problems per run):")
        print(acc_df[["problems", "correct", "accuracy",
                       "no_extract", "hit_cap"]].to_string())

    # Accuracy on shared problems only
    if len(run_grades) > 1:
        shared_ids = set.intersection(*(set(g["id"]) for g in run_grades.values()))
        print(f"\nAccuracy on shared problems only ({len(shared_ids)} problems finished in all runs):")
        shared_rows = []
        for run_label, grades in run_grades.items():
            g = grades[grades["id"].isin(shared_ids)]
            n = len(g)
            n_correct = int(g["correct"].sum())
            shared_rows.append({
                "run": run_label,
                "shared_problems": n,
                "correct": n_correct,
                "accuracy": f"{n_correct/n:.1%}" if n else "N/A",
            })
        shared_df = pd.DataFrame(shared_rows).set_index("run")
        print(shared_df.to_string())

    # Latency speedup relative to F=1
    if latency_rows and len(latency_rows) > 1:
        # Explicitly find the baseline run rather than relying on index 0
        baseline_row = next((r for r in latency_rows if "baseline" in r["run"]), None)
        
        if baseline_row:
            baseline_mean = baseline_row["mean_total_s"]
            baseline_tbt  = baseline_row["mean_tbt_ms"]
            print("\nSpeedup vs F=1 baseline:")
            for row in latency_rows:
                sp_tot = baseline_mean / row["mean_total_s"] if row["mean_total_s"] else float("nan")
                sp_tbt = baseline_tbt  / row["mean_tbt_ms"]  if row["mean_tbt_ms"]  else float("nan")
                print(f"  {row['run']:20s}  total_latency: {sp_tot:.3f}x  "
                      f"mean_tbt: {sp_tbt:.3f}x")
        else:
            print("\n[warn] Baseline (F=1) missing, skipping speedup calculation.")

    # Distribution plot
    if run_tbts:
        shared_acc: dict[str, tuple[int, int]] = {}
        if len(run_grades) > 1:
            shared_ids = set.intersection(*(set(g["id"]) for g in run_grades.values()))
            for run_label, grades in run_grades.items():
                g = grades[grades["id"].isin(shared_ids)]
                shared_acc[run_label] = (int(g["correct"].sum()), len(g))
        plot_tbt_distribution(run_tbts, results_dir / "tbt_distribution.png",
                              shared_acc=shared_acc or None)

    # Save markdown summary
    out_path = results_dir / "fetch_interval_analysis.md"
    lines = ["# Fetch Interval Experiment — MATH50\n"]
    if latency_rows:
        lines.append("## Latency\n")
        lines.append(lat_df[["n_problems", "mean_tot(s)", "med_tot(s)",
                              "p90_tot(s)", "mean_tbt(ms)", "avg_steps"]]
                     .to_markdown())
        lines.append("\n")
    if accuracy_rows:
        lines.append("## Accuracy\n")
        lines.append(acc_df[["problems", "correct", "accuracy",
                              "no_extract", "hit_cap"]].to_markdown())
        lines.append("\n")
    
    # Ensure directory exists before writing
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()