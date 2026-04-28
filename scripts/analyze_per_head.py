"""Per-head cosine-similarity analysis for the `dips_v2` run.

Inputs (expected):
  modal_logs/dips_v2/dips_v2/sims_<pid>.npz    # shape [N, 32, 32] fp32
  modal_logs/dips_v2/dips_v2/corr_<pid>.csv    # scalar-mean + need_corr per (step,layer)
  modal_logs/dips_v2/dips_v2/tokens_<pid>.csv  # step -> generated token text

Outputs:
  analysis/dips_v2/report.md
  analysis/dips_v2/plots/per_layer_meanmin_<pid>.png   (user's original ask,
                                                        one plot per problem)
  analysis/dips_v2/plots/per_kv_head_rate_<pid>.png
  analysis/dips_v2/plots/head_correlation_<pid>.png
  analysis/dips_v2/plots/n_drifted_hist_<pid>.png

What the tensor contains (per problem, shape [n_steps, 32 layers, 32 q heads]):
  sim[s, L, h] = cos_sim(q_t[layer=L, head=h], q_{t-1}[layer=L, head=h])
  Rows where budget was not yet exceeded are NaN (decode with spec_ret
  only logs when n_pages > budget_pages).

Key quantities we compute:
  - mean_over_q_heads[s, L]           = nanmean of sim[s, L, :]  (matches our old cos_sim scalar)
  - min_over_q_heads[s, L]            = nanmin  of sim[s, L, :]
  - kv_head_mean[s, L, kv]            = sim[s, L, 4*kv:4*kv+4].mean(axis=-1)
                                        (what need_corr's internal `sim` tensor holds)
  - kv_head_min[s, L]                 = min over 8 kv-head means — THIS is what need_corr thresholds
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

N_LAYERS = 32
N_Q_HEADS = 32
N_KV_HEADS = 8              # 4 q heads per kv group for Llama-8B

LAYERS_TO_PLOT = [0, 5, 10, 15, 20, 25, 30]
THRESHOLDS = [0.95, 0.90, 0.85, 0.80]

T_KEYWORDS = [
    r"wait", r"hmm", r"actually", r"alternatively", r"instead", r"however",
    r"hold on", r"on second thought",
    r"let me (?:re|check|verify|think|reconsider)",
    r"but (?:wait|actually)", r"oh", r"oops",
]
T_REGEX = re.compile(
    r"(?i)(?<![a-z])(?:" + "|".join(T_KEYWORDS) + r")(?![a-z])"
)


def load_problem(log_dir: Path, pid: str):
    sim_path = log_dir / f"sims_{pid}.npz"
    corr_path = log_dir / f"corr_{pid}.csv"
    tok_path = log_dir / f"tokens_{pid}.csv"
    if not (sim_path.exists() and corr_path.exists() and tok_path.exists()):
        return None
    sim = np.load(sim_path)["sim"]            # [n_steps, 32, 32]
    corr = pd.read_csv(corr_path)
    tok = pd.read_csv(tok_path)
    tok["is_transition"] = tok["token_text"].fillna("").str.contains(
        T_REGEX, regex=True
    )
    return {"pid": pid, "sim": sim, "corr": corr, "tok": tok}


def aggregate_stats(sim: np.ndarray) -> dict | None:
    """Return flat per-(step, layer) numpy arrays of the derived statistics.
    Returns None if there's no valid (non-NaN) data."""
    # sim: [N, 32, 32]
    if sim.size == 0:
        return None
    valid = ~np.isnan(sim[:, 0, 0])          # [N] rows where sim was logged
    if not valid.any():
        return None
    sub = sim[valid]                          # [Nv, 32, 32]

    q_mean = sub.mean(axis=-1)               # [Nv, 32]   mean over 32 q heads
    q_min  = sub.min (axis=-1)               # [Nv, 32]
    kv_mean = sub.reshape(sub.shape[0], N_LAYERS, N_KV_HEADS, -1).mean(axis=-1)
    # kv_mean: [Nv, 32, 8]   — this mirrors the tensor `sim` inside
    #                           get_corr_torch_compile
    kv_min = kv_mean.min(axis=-1)            # [Nv, 32]   what need_corr actually thresholds

    return {
        "valid_mask": valid,
        "q_mean": q_mean, "q_min": q_min,
        "kv_mean": kv_mean, "kv_min": kv_min,
        "sim_sub": sub,
    }


def threshold_rates(agg: dict, taus=THRESHOLDS) -> pd.DataFrame:
    rows = []
    for tau in taus:
        q_min = agg["q_min"]
        kv_min = agg["kv_min"]
        q_mean = agg["q_mean"]
        kv_mean = agg["kv_mean"]
        sim_sub = agg["sim_sub"]
        n = q_min.size
        rows.append({
            "tau": tau,
            "any_kv_group < tau (= need_corr trigger)":
                float((kv_min < tau).mean()),
            "any_q_head < tau":
                float((q_min < tau).mean()),
            "mean_across_q_heads < tau":
                float((q_mean < tau).mean()),
            "single_kv_head < tau (per-kv-head rate)":
                float((kv_mean < tau).mean()),
            "single_q_head < tau (per-q-head rate)":
                float((sim_sub < tau).mean()),
        })
    return pd.DataFrame(rows).set_index("tau")


def ema_nan(x: np.ndarray, alpha: float) -> np.ndarray:
    """EMA that skips NaN entries (carries state; leaves NaN slots as NaN)."""
    out = np.full_like(x, np.nan, dtype=np.float64)
    state = None
    for i, v in enumerate(x):
        if np.isnan(v):
            continue
        state = v if state is None else (1.0 - alpha) * state + alpha * v
        out[i] = state
    return out


def plot_per_layer_meanmin(problem, out_path, ema_alpha: float = 0.1):
    pid = problem["pid"]
    sim = problem["sim"]
    tok = problem["tok"]

    n_steps = sim.shape[0]
    fig, axes = plt.subplots(len(LAYERS_TO_PLOT), 1,
                             figsize=(14, 2.1 * len(LAYERS_TO_PLOT)),
                             sharex=True, squeeze=False)
    valid = ~np.isnan(sim[:, 0, 0])
    step_ids = np.arange(n_steps)

    # transition-marker step ids (for overlay)
    t_steps = tok.loc[tok["is_transition"], "step_id"].to_numpy()
    t_steps = t_steps[t_steps < n_steps]

    for i, L in enumerate(LAYERS_TO_PLOT):
        if L >= N_LAYERS:
            continue
        ax = axes[i, 0]
        slayer = sim[:, L, :]                  # [N, 32]
        mean_q = np.full(n_steps, np.nan)
        min_q  = np.full(n_steps, np.nan)
        kv_min = np.full(n_steps, np.nan)
        mean_q[valid] = slayer[valid].mean(axis=-1)
        min_q[valid] = slayer[valid].min(axis=-1)
        kv_layer = slayer[valid].reshape(-1, N_KV_HEADS, 4).mean(axis=-1)
        kv_min[valid] = kv_layer.min(axis=-1)

        # EMA smoothing of each signal
        mean_q_ema = ema_nan(mean_q, ema_alpha)
        min_q_ema  = ema_nan(min_q,  ema_alpha)
        kv_min_ema = ema_nan(kv_min, ema_alpha)

        # Faded raw signals
        ax.plot(step_ids, mean_q, lw=0.35, alpha=0.18, color="tab:blue")
        ax.plot(step_ids, min_q,  lw=0.35, alpha=0.18, color="tab:red")
        ax.plot(step_ids, kv_min, lw=0.35, alpha=0.18, color="tab:orange")
        # Solid EMA lines on top
        ax.plot(step_ids, mean_q_ema, lw=1.4, alpha=1.0, color="tab:blue",
                label=f"EMA(mean over 32 q heads), α={ema_alpha}")
        ax.plot(step_ids, min_q_ema, lw=1.4, alpha=1.0, color="tab:red",
                label=f"EMA(min over 32 q heads)")
        ax.plot(step_ids, kv_min_ema, lw=1.6, alpha=1.0, color="tab:orange",
                label="EMA(min over 8 kv-group means) — need_corr signal")
        ax.axhline(0.9, color="gray", lw=0.6, ls="--", alpha=0.6,
                   label="τ=0.9")
        # transition overlay
        if len(t_steps) > 0:
            ax.scatter(t_steps, np.full_like(t_steps, 0.02, dtype=float),
                       marker="v", color="purple", s=16, alpha=0.6,
                       label=f"T-keyword (n={len(t_steps)})" if i == 0 else None)
        ax.set_ylim(-0.7, 1.05)
        ax.set_ylabel(f"layer {L}", fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="lower right", fontsize=7, ncol=2)
    axes[-1, 0].set_xlabel("decode step")
    fig.suptitle(
        f"{pid}: per-layer per-head sim — faded raw signals with "
        f"EMA (α={ema_alpha}) on top"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_per_kv_head_rate(problem, agg, out_path):
    pid = problem["pid"]
    kv_mean = agg["kv_mean"]                   # [Nv, 32, 8]
    # per-kv-head firing rate at τ=0.9
    per_kv = (kv_mean < 0.9).mean(axis=(0, 1))  # [8]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(np.arange(N_KV_HEADS), per_kv, color="tab:blue")
    ax.set_xlabel("kv_head_index (0..7)")
    ax.set_ylabel("P(sim < 0.9)")
    ax.set_title(f"{pid}: firing rate per kv-head (averaged over all "
                 f"steps×layers)")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_head_correlation(problem, agg, out_path):
    pid = problem["pid"]
    sim_sub = agg["sim_sub"]                    # [Nv, 32, 32]
    # Flatten across (step, layer) and correlate the 32 q heads
    flat = sim_sub.reshape(-1, N_Q_HEADS)       # [Nv*32, 32]
    # cap sample size for speed
    if flat.shape[0] > 200_000:
        idx = np.random.default_rng(0).choice(flat.shape[0], 200_000,
                                               replace=False)
        flat = flat[idx]
    # centered correlation
    flat = flat - flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True) + 1e-9
    flat = flat / std
    corr_mat = (flat.T @ flat) / flat.shape[0]  # [32, 32]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr_mat, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xlabel("q_head_index")
    ax.set_ylabel("q_head_index")
    ax.set_title(f"{pid}: correlation of sim across q heads\n"
                 "(diagonal=1 by construction; off-diagonal = how "
                 "synchronously heads drift)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_n_drifted_hist(problem, agg, out_path):
    pid = problem["pid"]
    kv_mean = agg["kv_mean"]                    # [Nv, 32, 8]
    n_drifted = (kv_mean < 0.9).sum(axis=-1)    # [Nv, 32] — how many of 8 fired per (step, layer)
    flat = n_drifted.ravel()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    vals, counts = np.unique(flat, return_counts=True)
    ax.bar(vals, counts / counts.sum(), color="tab:purple")
    ax.set_xlabel("# of 8 kv-groups with sim < 0.9 per (step, layer)")
    ax.set_ylabel("fraction")
    ax.set_xticks(np.arange(9))
    ax.set_title(f"{pid}: how many kv-groups drift per event\n"
                 "(0 = no correction; 8 = all heads drift)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=Path,
                    default=REPO_ROOT / "modal_logs" / "dips_v2" / "dips_v2")
    ap.add_argument("--output_dir", type=Path,
                    default=REPO_ROOT / "analysis" / "dips_v2")
    args = ap.parse_args()

    plot_dir = args.output_dir / "plots"
    args.output_dir.mkdir(exist_ok=True, parents=True)
    plot_dir.mkdir(exist_ok=True, parents=True)

    # Pick up all problem ids that have a .npz file on disk.
    pids = sorted(p.stem.removeprefix("sims_")
                  for p in args.input_dir.glob("sims_*.npz"))
    if not pids:
        raise SystemExit(f"No sims_*.npz in {args.input_dir}")
    print(f"Problems with per-head sim: {pids}")

    # Global aggregate across all problems
    global_rows = []
    report = []
    label = args.input_dir.parent.name if args.input_dir.parent.name != "modal_logs" else args.input_dir.name
    report.append(f"# `{label}` — per-head cosine-similarity analysis\n")
    report.append(
        "Uses the full `[n_steps, n_layers=32, n_q_heads=32]` cosine-sim "
        "tensor cached per problem. This lets us compute any aggregation "
        "over heads without re-running.\n"
    )
    report.append(f"Problems included: {', '.join('`'+p+'`' for p in pids)}\n")

    for pid in pids:
        prob = load_problem(args.input_dir, pid)
        if prob is None:
            continue
        sim = prob["sim"]
        print(f"\n== {pid} ==  sim.shape={sim.shape}  "
              f"nan_rows={int(np.isnan(sim[:,0,0]).sum()):,}")

        agg = aggregate_stats(sim)
        if agg is None:
            print(f"  [skip {pid}] no valid sim rows (problem was skipped or empty)")
            continue
        n_valid = int(agg["valid_mask"].sum())
        n_trans = int(prob["tok"]["is_transition"].sum())

        report.append(f"\n## Problem `{pid}`\n")
        report.append(f"- Decode steps logged: **{n_valid:,}** "
                      f"(of {sim.shape[0]:,} total; earlier steps are "
                      "before budget was exceeded)")
        report.append(f"- Transition-keyword tokens: **{n_trans:,}** "
                      f"({n_trans/len(prob['tok']):.4%} of generated tokens)")
        rates = threshold_rates(agg)
        report.append("\n**Rates of `sim < τ` under five aggregations:**\n")
        lines = ["| τ | any kv-group (= trigger) | any q-head | mean q-heads "
                 "| single kv-head | single q-head |",
                 "|---|---|---|---|---|---|"]
        for tau, row in rates.iterrows():
            lines.append(
                f"| {tau:.2f} | {row.iloc[0]:.4f} | {row.iloc[1]:.4f} | "
                f"{row.iloc[2]:.4f} | {row.iloc[3]:.4f} | {row.iloc[4]:.4f} |"
            )
        report += lines

        print(rates.round(4).to_string())

        # how many kv-groups fire per event, at τ=0.9
        kv_mean = agg["kv_mean"]
        n_drifted = (kv_mean < 0.9).sum(axis=-1)  # [Nv, 32]
        dist = pd.Series(n_drifted.ravel()).value_counts(normalize=True).sort_index()
        report.append(
            "\n**# of 8 kv-groups drifting per (step, layer) at τ=0.9:**\n")
        report.append("| drifted | fraction |")
        report.append("|---|---|")
        for k, v in dist.items():
            report.append(f"| {int(k)} | {v:.4f} |")

        # per-kv-head firing rate
        per_kv = (kv_mean < 0.9).mean(axis=(0, 1))
        report.append("\n**Per-kv-head firing rate (τ=0.9):**\n")
        report.append("| kv_head | P(sim < 0.9) |")
        report.append("|---|---|")
        for h in range(N_KV_HEADS):
            report.append(f"| {h} | {per_kv[h]:.4f} |")

        # per-layer rate (all layers, need_corr signal)
        per_layer_need_corr = (agg["kv_min"] < 0.9).mean(axis=0)  # [32]
        report.append("\n**Per-layer `need_corr` rate (τ=0.9):**\n")
        report.append("| layer | rate | | layer | rate | | layer | rate | | layer | rate |")
        report.append("|---|---|---|---|---|---|---|---|---|---|---|")
        chunks = [per_layer_need_corr[i*8:(i+1)*8] for i in range(4)]
        for row_i in range(8):
            row = []
            for col_i in range(4):
                L = col_i * 8 + row_i
                row += [f"{L}", f"{per_layer_need_corr[L]:.3f}", ""]
            # drop trailing sep
            row[-1] = row[-1]
            report.append("| " + " | ".join(row[:-1]) + " |")

        # plots
        plot_per_layer_meanmin(
            prob, plot_dir / f"per_layer_meanmin_{pid}.png"
        )
        plot_per_kv_head_rate(
            prob, agg, plot_dir / f"per_kv_head_rate_{pid}.png"
        )
        plot_head_correlation(
            prob, agg, plot_dir / f"head_correlation_{pid}.png"
        )
        plot_n_drifted_hist(
            prob, agg, plot_dir / f"n_drifted_hist_{pid}.png"
        )
        report.append(
            f"\nPlots: "
            f"[per_layer_meanmin]({plot_dir.name}/per_layer_meanmin_{pid}.png) · "
            f"[per_kv_head_rate]({plot_dir.name}/per_kv_head_rate_{pid}.png) · "
            f"[head_correlation]({plot_dir.name}/head_correlation_{pid}.png) · "
            f"[n_drifted_hist]({plot_dir.name}/n_drifted_hist_{pid}.png)\n"
        )

        global_rows.append({
            "pid": pid, "n_valid": n_valid,
            "need_corr@0.9": float((agg["kv_min"] < 0.9).mean()),
            "per_q_head@0.9": float((agg["sim_sub"] < 0.9).mean()),
            "per_kv_head@0.9": float((agg["kv_mean"] < 0.9).mean()),
            "mean@0.9": float((agg["q_mean"] < 0.9).mean()),
        })

    gdf = pd.DataFrame(global_rows).set_index("pid")
    report.insert(3, "\n## Global rates at τ=0.9 (for quick comparison)\n")
    lines = ["| pid | n_valid steps | need_corr | per-q-head | per-kv-head | mean<0.9 |",
             "|---|---|---|---|---|---|"]
    for pid, r in gdf.iterrows():
        lines.append(
            f"| `{pid}` | {int(r['n_valid']):,} | "
            f"{r['need_corr@0.9']:.4f} | {r['per_q_head@0.9']:.4f} | "
            f"{r['per_kv_head@0.9']:.4f} | {r['mean@0.9']:.4f} |"
        )
    report[3] = "\n## Global rates at τ=0.9 (for quick comparison)\n\n" + "\n".join(lines) + "\n"

    report_path = args.output_dir / "report.md"
    report_path.write_text("\n".join(report))
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
