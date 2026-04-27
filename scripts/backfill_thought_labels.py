"""Backfill EMA-classified thought labels onto an existing decode_log.csv.gz.

Usage:
    python scripts/backfill_thought_labels.py \
        /data/user_data/skolawol/freekv_logs/phase0/freekv_aime24_<jobid>/ \
        --classifier_layer 16 \
        --ema_alpha 0.1 \
        --r_threshold 0.92 \
        --t_threshold 0.75 \
        --segment_window 16

Produces: <log_dir>/decode_log_with_labels.csv.gz

Reads cos_sim from rows where layer_id == classifier_layer (one per step),
runs the classifier, then merges thought_type / segment_id / ema_sim back
onto every row of the original decode log via (prompt_id, step_id) join.

Idempotent — different threshold/alpha sweeps re-run the script and overwrite.
"""
import argparse
import os
import sys
from pathlib import Path

# allow `import freekv.thought_classifier` from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "source"))

import pandas as pd
from freekv.thought_classifier import ThoughtClassifier
from freekv.utils import ThoughtType


def relabel(df_one_layer: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Run classifier across (prompt_id, step_id) order. df has one row per step."""
    out = []
    for prompt_id, group in df_one_layer.groupby("prompt_id", sort=True):
        clf = ThoughtClassifier(**kwargs)
        for _, row in group.sort_values("step_id").iterrows():
            clf.update(row.cos_sim, int(row.step_id))
            out.append({
                "prompt_id": prompt_id,
                "step_id": int(row.step_id),
                "ema_sim": clf.ema_sim if clf.ema_sim is not None else float("nan"),
                "thought_type": int(clf.current_type),
                "segment_id": clf.segment_id,
            })
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir", type=Path)
    ap.add_argument("--classifier_layer", type=int, default=None,
                    help="layer to read cos_sim from (default: middle layer)")
    ap.add_argument("--ema_alpha", type=float, default=0.1)
    ap.add_argument("--r_threshold", type=float, default=0.92)
    ap.add_argument("--t_threshold", type=float, default=0.75)
    ap.add_argument("--segment_window", type=int, default=16)
    args = ap.parse_args()

    decode_path = args.log_dir / "decode_log.csv.gz"
    if not decode_path.exists():
        sys.exit(f"missing: {decode_path}")
    df = pd.read_csv(decode_path)

    classifier_layer = args.classifier_layer
    if classifier_layer is None:
        classifier_layer = int(df.layer_id.median())
    print(f"using classifier_layer={classifier_layer} "
          f"(layers in log: {sorted(df.layer_id.unique())[:5]}...)")

    one_layer = df[df.layer_id == classifier_layer].dropna(subset=["cos_sim"])
    print(f"classifying {len(one_layer)} (prompt, step) rows")

    labels = relabel(
        one_layer,
        ema_alpha=args.ema_alpha,
        r_threshold=args.r_threshold,
        t_threshold=args.t_threshold,
        segment_window=args.segment_window,
    )

    # drop any stale label columns and merge
    df = df.drop(columns=[c for c in ("ema_sim", "thought_type", "segment_id") if c in df.columns])
    merged = df.merge(labels, on=["prompt_id", "step_id"], how="left")

    # human-readable label column
    name_map = {int(t): t.name for t in ThoughtType}
    merged["thought_label"] = merged.thought_type.map(name_map)

    out_path = args.log_dir / "decode_log_with_labels.csv.gz"
    merged.to_csv(out_path, index=False, compression="gzip")
    print(f"wrote {out_path} ({len(merged)} rows)")

    # quick summary
    print("\nSegment counts:")
    print(merged.groupby("thought_label").size())
    print(f"\nUnique segments per prompt:")
    print(merged.groupby("prompt_id").segment_id.nunique())


if __name__ == "__main__":
    main()
