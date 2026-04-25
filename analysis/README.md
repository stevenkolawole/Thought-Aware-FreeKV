# Analysis directory layout

Each subdirectory holds a **self-contained** analysis of one experimental run.
Runs do not share data: each one comes from its own Modal volume subdir
(`tafkv-logs/{subdir}`) and is downloaded to `modal_logs/{subdir}/`.

```
analysis/
├── README.md          ← you are here
├── baseline/          ← Phase 3 run (2026-04-23): 14 AIME24 problems, max_gen 32000
│   ├── report.md
│   └── plots/
└── verify_dips/       ← Phase 4 run: 5 specific AIME24 problems, token-level logging
    ├── report.md
    └── plots/
```

## `baseline/` — what this run was and what it found

**Setup:**
- Config: `--budget 2048 --sink 512 --recent 512 --recall_impl cuda_cpy --spec_ret --corr 0.9 --max_gen 32000`
- Model: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- Intended scope: 29 AIME24 problems.
- Actual scope: **13 problems completed + 1 partial**, because Modal's 4h
  function timeout cancelled the run before the remaining 16 problems ran.
- Token-level logging (tokens_*.csv) was **not** enabled for this run —
  only step-level corr + recall CSVs exist.
- Per-problem model answers (`preds.jsonl`) were lost because `pred.py` only
  flushed them at end-of-loop, which never ran.

**Headline findings (see `baseline/report.md`):**
- **Observed correction rate: 89.4%**, well above the paper's claimed
  43–52%. Almost certainly because `need_corr` fires on an OR-over-heads
  check while our scalar cos_sim logs an average-over-heads.
- **EMA thought-type classifier never fires T**: 99.0% R, 0.97% E, 0.005% T
  with the initial thresholds (tau_r=0.84, tau_t=0.6, alpha=0.1).
- **Per-layer correction rate is heterogeneous**: layer 1 at 31.5%, layer
  12 at 56.3%, most others >85%. A few layers carry the signal.

**Do not mix these numbers with `verify_dips/`** — different problem set,
different purpose.

## `full_aime/` — paper-config sweep on the full benchmark

**Setup:**
- All AIME24 problems with the paper's reasoning config:
  `--max_gen 16384 --budget 2048 --sink 512 --recent 512 --corr 0.9 --spec_ret`
  (Section 5.2 of arXiv:2505.13109 specifies max_gen = 16K).
- Logging includes per-(step, layer, q_head) cosine sim cached as
  `sims_<pid>.npz` (28 problems present; 2 problems were not reached:
  `2024-I-5`, `2024-II-13`).

**Headline at τ=0.9, mean across 28 problems:**
- `need_corr` trigger rate (any of 8 kv groups < τ): **90.14%** (std 1.74%)
- Per-q-head rate (single q-head < τ): **55.74%** (std 2.91%)
- Per-kv-head rate (single kv-head < τ): **58.90%** (std 3.23%)

The paper's claim of 43–52% lands as a per-head rate; our measurement is
slightly higher, consistent with what we saw on smaller subsets.
Cross-problem variance is remarkably small.

## `verify_dips/` — what this run was and what it asks

**Setup:**
- Same FreeKV config as `baseline/`.
- Problem subset: `2024-I-1, 2024-I-2, 2024-I-3, 2024-I-4, 2024-II-4` (5 problems).
- Token-level logging enabled — each generated token is saved with its
  decode step_id in `tokens_<pid>.csv`.
- Incremental `preds.jsonl` flushing (timeout-safe).

**Question:**
Do cos_sim dips during decode correspond to ThinKV-style transition tokens
("wait", "hmm", "actually", …) in the generated text?

**Methodology (see `scripts/analyze_dips.py`):**
1. Join corr_*.csv (layer-0 rows) with tokens_*.csv on step_id.
2. Regex-mark each step as transition or not based on the decoded token.
3. Compare cos_sim distributions; compute AUC; bucket into quintiles;
   measure per-transition distance to nearest local cos_sim dip.

**Findings:** see `verify_dips/report.md` after the run lands.

## Reproducing

```bash
# download logs from the Modal volume
bash scripts/fetch_logs.sh baseline
bash scripts/fetch_logs.sh verify_dips

# re-run analyses (pure Python, no GPU)
python scripts/analyze_baseline.py --input_dir modal_logs/baseline/baseline \
                                    --output_dir analysis/baseline \
                                    --label baseline
python scripts/analyze_dips.py     --input_dir modal_logs/verify_dips/verify_dips \
                                    --output_dir analysis/verify_dips
```
