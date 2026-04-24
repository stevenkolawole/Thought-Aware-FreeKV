"""Render each problem's full generated text to its own markdown file.

Outputs:
  analysis/verify_dips/texts/<pid>.md
  analysis/verify_dips/texts/README.md   (index)

Each per-problem file contains:
  - the prompt
  - the model's full chain-of-thought + final answer
  - metadata (input/output length, reference answer if present)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", type=Path,
        default=REPO_ROOT / "modal_logs" / "verify_dips" / "verify_dips" / "preds.jsonl",
    )
    ap.add_argument(
        "--output_dir", type=Path,
        default=REPO_ROOT / "analysis" / "verify_dips" / "texts",
    )
    args = ap.parse_args()

    args.output_dir.mkdir(exist_ok=True, parents=True)

    index = ["# Generated text per problem — `verify_dips`\n"]
    index.append(
        "Each link opens the full prompt + model output for one AIME problem, "
        "straight from `preds.jsonl`.\n"
    )
    index.append("| Problem | Ref answer | Gen tokens | File |")
    index.append("|---|---|---|---|")

    for line in args.input.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        pid = rec.get("id") or "unknown"
        prompt = rec.get("input:") or rec.get("input") or ""
        pred = rec.get("pred") or ""
        ans = rec.get("answer", "")
        in_len = rec.get("input_len")
        out_len = rec.get("output_len")
        gen_tokens = (out_len - in_len) if (out_len and in_len) else None

        # The model's prediction contains the prompt as prefix (since
        # tokenizer.decode(output) includes the whole sequence). Strip it so
        # the file shows just the generation.
        generation = pred
        if prompt and pred.startswith(prompt):
            generation = pred[len(prompt):]

        body = []
        body.append(f"# Problem `{pid}`\n")
        if ans != "":
            body.append(f"**Reference answer:** `{ans}`\n")
        if gen_tokens is not None:
            body.append(f"**Generated tokens:** {gen_tokens:,} "
                        f"(input {in_len}, output {out_len})\n")
        body.append("\n## Prompt\n")
        body.append("```")
        body.append(prompt)
        body.append("```\n")
        body.append("## Model output (full chain-of-thought)\n")
        body.append("```")
        body.append(generation)
        body.append("```\n")

        fp = args.output_dir / f"{pid}.md"
        fp.write_text("\n".join(body))
        rel = fp.relative_to(args.output_dir.parent)
        index.append(
            f"| `{pid}` | `{ans}` | "
            f"{gen_tokens:,}" if gen_tokens else "—"
        )
        # the previous join loses a cell — rewrite properly
        index[-1] = (
            f"| `{pid}` | `{ans}` | "
            f"{gen_tokens:,} | [{pid}.md]({fp.name}) |"
            if gen_tokens is not None else
            f"| `{pid}` | `{ans}` | — | [{pid}.md]({fp.name}) |"
        )
        print(f"wrote {fp}")

    (args.output_dir / "README.md").write_text("\n".join(index))
    print(f"wrote {args.output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
