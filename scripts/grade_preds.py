"""Grade AIME predictions: extract the final \\boxed{...} from each model
output and compare to the dataset's integer reference answer.

Inputs:
  modal_logs/<run>/<run>/preds.jsonl   (one JSON per line)

Outputs (printed):
  - per-problem: id | reference | extracted | correct?
  - per-run accuracy summary
  - cross-run table

AIME answers are integers in [0, 999]. R1-Distill traces typically emit a
final \\boxed{N} after </think>. We take the LAST \\boxed{...} in the text
to handle intermediate boxes that show up during reasoning.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS = ["verify_dips", "verify_dips_v2", "dips_v2", "full_aime", "math50"]

# Datasets where the reference answer is an integer (AIME-style).
# Everything else is treated as a string-equality-with-whitespace-stripped check
# (matches accuracy/eval/reasoning/eval.py).
INTEGER_ANSWER_RUNS = {"verify_dips", "verify_dips_v2", "dips_v2", "full_aime"}

# Match \boxed{...} with one level of nested braces (handles \boxed{\frac{1}{2}}).
BOXED = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}")
INT_IN_BLOCK = re.compile(r"-?\d[\d,]*")


def extract_last_boxed(text: str) -> str | None:
    """Return the raw contents of the last \\boxed{...} block in text."""
    if not text:
        return None
    blocks = BOXED.findall(text)
    if not blocks:
        return None
    return blocks[-1]


def extract_int(boxed: str | None) -> int | None:
    """Parse an integer out of a \\boxed{} block. Returns None if absent."""
    if boxed is None:
        return None
    m = INT_IN_BLOCK.search(boxed)
    if not m:
        return None
    s = m.group(0).replace(",", "")
    try:
        return int(s)
    except ValueError:
        return None


def grade_one(rec: dict, mode: str = "int") -> dict:
    """mode='int'  -> compare integer extracted from \\boxed{} to int(ref)
       mode='str'  -> compare whitespace-stripped \\boxed{} content to ref"""
    pid = rec.get("id")
    ref = rec.get("answer")
    pred_text = rec.get("pred") or ""
    boxed = extract_last_boxed(pred_text)
    if mode == "int":
        extracted = extract_int(boxed)
        correct = (
            extracted is not None
            and ref is not None
            and int(extracted) == int(ref)
        )
        extracted_disp = extracted
    else:  # str
        extracted = (
            None if boxed is None
            else boxed.replace(" ", "").replace("\\!", "").replace("\\,", "")
        )
        ref_norm = (
            None if ref is None
            else str(ref).replace(" ", "").replace("\\!", "").replace("\\,", "")
        )
        correct = (
            extracted is not None
            and ref_norm is not None
            and extracted == ref_norm
        )
        extracted_disp = extracted
    n_boxed = len(BOXED.findall(pred_text))
    out_len = rec.get("output_len")
    in_len = rec.get("input_len")
    gen_tokens = (out_len - in_len) if (out_len and in_len) else None
    hit_cap = gen_tokens is not None and gen_tokens >= 16380  # 16384-ish
    return {
        "id": pid,
        "ref": ref,
        "extracted": extracted_disp,
        "correct": correct,
        "n_boxed": n_boxed,
        "gen_tokens": gen_tokens,
        "hit_cap": hit_cap,
    }


def grade_run(preds_path: Path, dedupe: str = "best",
              mode: str = "int") -> pd.DataFrame:
    """dedupe = "best" (keep best-scoring attempt per id), "last" (keep last),
    or "all" (keep every entry — useful for diagnosing restart artifacts).
    mode = "int" (AIME-style) or "str" (MATH-style)."""
    rows = []
    with preds_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(grade_one(json.loads(line), mode=mode))
    df = pd.DataFrame(rows)
    if dedupe == "all" or df.empty:
        return df
    # Sort so that "best" (correct first, else has extraction, else by gen_tokens)
    # comes first, then keep the first row per id.
    df = df.assign(
        _has_extract=df["extracted"].notna(),
        _correct_int=df["correct"].astype(int),
    )
    if dedupe == "best":
        df = df.sort_values(
            ["_correct_int", "_has_extract", "gen_tokens"],
            ascending=[False, False, True],
        )
    elif dedupe == "last":
        # preserve original order — this is what the file order represents
        pass
    else:
        raise ValueError(f"unknown dedupe mode: {dedupe}")
    df = df.drop_duplicates(subset=["id"], keep="last" if dedupe == "last" else "first")
    return df.drop(columns=["_has_extract", "_correct_int"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=RUNS,
                    help="Run subdirs under modal_logs/ to grade")
    ap.add_argument("--dedupe", choices=["best", "last", "all"], default="best",
                    help="How to handle duplicate ids from restarted runs")
    args = ap.parse_args()

    run_summaries = []
    print("=== per-run details ===")
    for run in args.runs:
        preds_path = REPO_ROOT / "modal_logs" / run / run / "preds.jsonl"
        if not preds_path.exists():
            print(f"\n[{run}]  preds.jsonl missing → skipping")
            continue
        mode = "int" if run in INTEGER_ANSWER_RUNS else "str"
        df = grade_run(preds_path, dedupe=args.dedupe, mode=mode)
        if df.empty:
            print(f"\n[{run}]  empty preds.jsonl")
            continue
        n = len(df)
        n_correct = int(df["correct"].sum())
        n_extracted = int(df["extracted"].notna().sum())
        n_capped = int(df["hit_cap"].sum())
        print(f"\n[{run}]  problems={n}  extracted={n_extracted}  "
              f"correct={n_correct}/{n}  hit_cap={n_capped}")
        # Show problem-level table
        view = df.copy()
        view["match"] = view.apply(
            lambda r: "✓" if r["correct"] else (
                "✗" if r["extracted"] is not None else "—"
            ),
            axis=1,
        )
        print(view[["id", "ref", "extracted", "match", "gen_tokens",
                    "hit_cap", "n_boxed"]].to_string(index=False))
        run_summaries.append({
            "run": run,
            "n": n,
            "n_correct": n_correct,
            "accuracy": n_correct / n if n else 0.0,
            "n_no_extract": n - n_extracted,
            "n_cap_hit": n_capped,
            "cap_hit_correct": int(df.loc[df["hit_cap"], "correct"].sum()),
            "cap_hit_total": n_capped,
            "non_cap_correct": int(df.loc[~df["hit_cap"], "correct"].sum()),
            "non_cap_total": n - n_capped,
        })

    if run_summaries:
        print("\n=== cross-run summary ===")
        s = pd.DataFrame(run_summaries).set_index("run")
        s["accuracy"] = s["accuracy"].map(lambda x: f"{x:.1%}")
        s["non_cap_acc"] = s.apply(
            lambda r: f"{r['non_cap_correct'] / r['non_cap_total']:.1%}"
            if r["non_cap_total"] else "—",
            axis=1,
        )
        s["cap_acc"] = s.apply(
            lambda r: f"{r['cap_hit_correct'] / r['cap_hit_total']:.1%}"
            if r["cap_hit_total"] else "—",
            axis=1,
        )
        print(s[["n", "n_correct", "accuracy", "n_no_extract",
                 "non_cap_total", "non_cap_acc",
                 "cap_hit_total", "cap_acc"]].to_string())


if __name__ == "__main__":
    main()
