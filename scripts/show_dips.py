"""Dump the deepest cos_sim dips per problem with surrounding token context.

Lets a human read what the model is generating at the moments when the
query direction flips hard. Intended to be read alongside
`analysis/verify_dips/report.md`.

Output: analysis/verify_dips/dips_context.md

Usage:
  python scripts/show_dips.py
    # default: top 25 deepest dips per problem, ±8 tokens context

  python scripts/show_dips.py --top 10 --context 5
  python scripts/show_dips.py --threshold 0.7      # all steps below cos_sim
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

T_KEYWORDS = [
    r"wait", r"hmm", r"actually", r"alternatively", r"instead", r"however",
    r"hold on", r"on second thought",
    r"let me (?:re|check|verify|think|reconsider)",
    r"but (?:wait|actually)", r"oh", r"oops",
]
T_REGEX = re.compile(
    r"(?i)(?<![a-z])(?:" + "|".join(T_KEYWORDS) + r")(?![a-z])"
)


def clean(s: str) -> str:
    """Make a token printable in markdown: normalize whitespace, escape pipes."""
    if not isinstance(s, str):
        return ""
    s = s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    s = s.replace("|", "\\|")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dir", type=Path,
        default=REPO_ROOT / "modal_logs" / "verify_dips" / "verify_dips",
    )
    ap.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "analysis" / "verify_dips" / "dips_context.md",
    )
    ap.add_argument("--top", type=int, default=25,
                    help="Top N deepest dips per problem to show")
    ap.add_argument("--context", type=int, default=8,
                    help="Tokens of context on each side of the dip")
    ap.add_argument("--threshold", type=float, default=None,
                    help="If set, show ALL steps below this cos_sim "
                         "(ignores --top)")
    args = ap.parse_args()

    corr_paths = sorted(args.input_dir.glob("corr_*.csv"))
    if not corr_paths:
        raise SystemExit(f"No corr_*.csv under {args.input_dir}")

    lines = []
    lines.append("# Dip context dump — `verify_dips`\n")
    lines.append(
        "For each problem, the deepest cos_sim dips at layer 0 with a context "
        f"window of ±{args.context} generated tokens around the dip.\n"
    )
    lines.append(
        "**Legend:** `[→token←]` = the dip token itself. Column `T?` = whether "
        "the dip token matched our transition-keyword regex. A markdown "
        "escape `\\n` is a real newline in the generation.\n"
    )

    for corr_path in corr_paths:
        pid = corr_path.stem.removeprefix("corr_")
        tok_path = args.input_dir / f"tokens_{pid}.csv"
        if not tok_path.exists():
            continue

        # Layer-0 row gives one cos_sim per decode step
        corr = pd.read_csv(corr_path)
        corr0 = corr[corr["layer_id"] == 0][["step_id", "cos_sim", "need_corr",
                                             "sim_ema"]].copy()
        tok = pd.read_csv(tok_path)
        joined = corr0.merge(tok, on="step_id", how="inner").sort_values("step_id")
        joined["is_transition"] = joined["token_text"].fillna("").str.contains(
            T_REGEX, regex=True
        )

        if args.threshold is not None:
            picked = joined[joined["cos_sim"] < args.threshold].copy()
            picked = picked.sort_values("cos_sim").head(args.top * 4)  # safety cap
            header = (f"all steps with cos_sim < {args.threshold} "
                      f"({len(picked)} shown; full count "
                      f"{(joined['cos_sim'] < args.threshold).sum()})")
        else:
            picked = joined.nsmallest(args.top, "cos_sim")
            header = f"top {len(picked)} deepest cos_sim dips"

        total_steps = len(joined)
        lines.append(f"\n## Problem `{pid}`  ({total_steps:,} decode steps)\n")
        lines.append(f"{header}:\n")
        lines.append("| step | cos_sim | sim_ema | corr | T? | context |")
        lines.append("|---|---|---|---|---|---|")

        by_step = dict(zip(joined["step_id"], joined["token_text"]))
        for _, r in picked.iterrows():
            s = int(r["step_id"])
            parts = []
            for off in range(-args.context, args.context + 1):
                t = by_step.get(s + off, "")
                tclean = clean(t)
                if off == 0:
                    parts.append(f"**[→{tclean}←]**")
                else:
                    parts.append(tclean)
            ctx = "".join(parts)
            lines.append(
                f"| {s} | {r['cos_sim']:.4f} | {r['sim_ema']:.4f} | "
                f"{int(r['need_corr'])} | {'✓' if r['is_transition'] else ''} | "
                f"{ctx} |"
            )

    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text("\n".join(lines))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
