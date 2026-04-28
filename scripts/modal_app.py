"""Modal app for Thought-Aware-FreeKV.

Image layers (cached):
  1. nvidia/cuda:12.1.1-devel + python 3.10 + apt deps
  2. pip install torch 2.5.1 (cu121)
  3. pip install the rest of requirements.txt
  4. pip install flash-attn 2.6.3 (no build isolation)
  5. git clone the public repo (pinned commit) + submodule init + apply
     flashinfer/raft patches + build freekv_cpp via `pip install -e .`

Anything in source/freekv/ or source/pred.py is overlayed at runtime via
add_local_dir so local Python edits take effect without rebuilding the
image.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = "/opt/tafkv"

# Public upstream for the cached clone. Pin to a commit so the image layers
# are deterministic. The overlay at the bottom of this file replaces the
# Python source with whatever is on disk locally, so new edits flow through
# without rebuilding.
REPO_URL = "https://github.com/stevenkolawole/Thought-Aware-FreeKV.git"
REPO_COMMIT = "22d4b16"

app = modal.App("thought-aware-freekv")

hf_cache = modal.Volume.from_name("tafkv-hf-cache", create_if_missing=True)
logs_vol = modal.Volume.from_name("tafkv-logs", create_if_missing=True)

_patch_script_remote = "/tmp/patch_deps.py"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git",
        "cmake",
        "ninja-build",
        "build-essential",
        "wget",
        "ca-certificates",
    )
    .env(
        {
            # A100 = 8.0. Keep the list tight so nvcc doesn't fan out builds.
            "TORCH_CUDA_ARCH_LIST": "8.0",
            "CUDA_HOME": "/usr/local/cuda",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
    .pip_install(
        "torch==2.5.1",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "accelerate",
        "compressed_tensors==0.11.0",
        "datasets",
        "einops",
        "flashinfer-python==0.2.4",
        "huggingface_hub==0.25.2",
        "hf_transfer",
        "numpy",
        "ninja",
        "packaging",
        "pandas",
        "protobuf",
        "pybind11",
        "sentencepiece",
        "scipy",
        "tqdm",
        "transformers==4.45.2",
        "setuptools",
        "wheel",
    )
    # Prebuilt flash-attn wheel matching torch 2.5 + cu12 + py3.10 to skip
    # the ~20 min nvcc compile (which also OOMs on Modal's default builder).
    # 2.7.x is API-compatible with 2.6.x for our use (FreeKV only depends on
    # it transitively via flashinfer).
    .run_commands(
        "pip install --no-build-isolation "
        "'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/"
        "flash_attn-2.7.3%2Bcu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'"
    )
    # Clone the upstream repo at a pinned commit and build freekv_cpp once.
    # Some nested submodules (e.g. flashinfer → spdlog) list SSH URLs; rewrite
    # to HTTPS so the unauthenticated Modal builder can fetch them.
    .run_commands(
        "git config --global url.https://github.com/.insteadOf git@github.com:",
        f"git clone {REPO_URL} {REMOTE_ROOT}",
        f"cd {REMOTE_ROOT} && git checkout {REPO_COMMIT}",
        f"cd {REMOTE_ROOT} && git submodule update --init --recursive",
    )
    # Bake the patch script into the image so it's available during build.
    .add_local_file(
        str(REPO_ROOT / "scripts" / "patch_deps.py"),
        remote_path=_patch_script_remote,
        copy=True,
    )
    .run_commands(
        f"python3 {_patch_script_remote} {REMOTE_ROOT}",
        # apt's cmake (3.22) is too old for the freekv_cpp CMakeLists
        # (needs 3.26.4+); pip wheel ships a recent version.
        "pip install 'cmake>=3.26.4'",
        f"cd {REMOTE_ROOT}/source && pip install -e . --no-build-isolation",
    )
    # Runtime overlays: Python edits do not invalidate the baked layers.
    .add_local_dir(
        str(REPO_ROOT / "source" / "freekv"),
        remote_path=f"{REMOTE_ROOT}/source/freekv",
    )
    .add_local_file(
        str(REPO_ROOT / "source" / "pred.py"),
        remote_path=f"{REMOTE_ROOT}/source/pred.py",
    )
    # Datasets + configs (in case the local copy has additions).
    .add_local_dir(
        str(REPO_ROOT / "accuracy"),
        remote_path=f"{REMOTE_ROOT}/accuracy",
    )
    .add_local_dir(
        str(REPO_ROOT / "config"),
        remote_path=f"{REMOTE_ROOT}/config",
    )
)


_RUN_PRED_FN_KW = dict(
    image=image,
    gpu="A100-40GB",
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/logs": logs_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
)


def _run_pred_impl(pred_args: list[str], log_subdir: str | None) -> int:
    import subprocess
    os.chdir(REMOTE_ROOT)
    cmd = ["python", "source/pred.py", *pred_args]
    if log_subdir:
        log_path = f"/logs/{log_subdir}"
        os.makedirs(log_path, exist_ok=True)
        if not any(a == "--log_dir" for a in pred_args):
            cmd += ["--log_dir", log_path]
        print(f"[modal] instrumentation logs -> {log_path}")
    print(f"[modal] exec: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd)
    logs_vol.commit()
    return proc.returncode


@app.function(timeout=60 * 60 * 12, **_RUN_PRED_FN_KW)
def run_pred(pred_args: list[str], log_subdir: str | None = None) -> int:
    """Default 12h timeout — for full benchmark sweeps."""
    return _run_pred_impl(pred_args, log_subdir)


@app.function(timeout=60 * 60, **_RUN_PRED_FN_KW)
def run_pred_1h(pred_args: list[str], log_subdir: str | None = None) -> int:
    """1h timeout — for short profiling/validation runs."""
    return _run_pred_impl(pred_args, log_subdir)


@app.function(timeout=60 * 60 * 12, **_RUN_PRED_FN_KW)
def run_pred_per_problem(
    pred_args: list[str],
    log_subdir: str,
    dataset_path: str,
    id_field: str = "unique_id",
) -> int:
    """Run pred.py SEPARATELY per problem to avoid cross-problem state leaks.

    Each problem runs in its own Python subprocess: fresh InferState, fresh
    CUDA context, fresh JIT-compiled kernels. The container persists so the
    image and HF cache stay warm. Per-problem overhead ~30s (model load +
    flashinfer rope JIT). Worth it for stability on long sweeps.

    pred_args should NOT include --data_ids or --log_dir; we set those.
    """
    import subprocess, json
    os.chdir(REMOTE_ROOT)

    log_path = f"/logs/{log_subdir}"
    os.makedirs(log_path, exist_ok=True)

    # Resolve dataset path relative to REMOTE_ROOT if not absolute
    ds_full = (
        dataset_path if os.path.isabs(dataset_path)
        else os.path.join(REMOTE_ROOT, dataset_path)
    )
    ids: list[str] = []
    with open(ds_full) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pid = row.get(id_field) or row.get("id")
            if pid:
                ids.append(pid)
    print(f"[modal] running {len(ids)} problems individually under {log_subdir}/", flush=True)

    def _safe_tag(pid: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in pid)

    def _already_done(pid: str) -> bool:
        # A problem is considered done if its sims_*.npz exists (sims are
        # written at close_logs which only happens after generation completes
        # OR after a clean skip; either way the problem won't restart cleanly).
        tag = _safe_tag(pid)
        sims_path = os.path.join(log_path, f"sims_{tag}.npz")
        return os.path.exists(sims_path)

    overall_rc = 0
    n_skip_resume = 0
    for i, pid in enumerate(ids):
        if _already_done(pid):
            n_skip_resume += 1
            print(f"[modal {i+1}/{len(ids)}] {pid} — already done, resuming", flush=True)
            continue
        cmd = [
            "python", "source/pred.py",
            *pred_args,
            "--data_ids", pid,
            "--log_dir", log_path,
        ]
        print(f"[modal {i+1}/{len(ids)}] {pid}", flush=True)
        proc = subprocess.run(cmd)
        rc = proc.returncode
        if rc != 0:
            print(f"[modal] problem {pid} returned {rc}; continuing", flush=True)
            overall_rc = max(overall_rc, rc)
        # Commit logs after each problem so partial progress survives any
        # later subprocess failure.
        logs_vol.commit()
    if n_skip_resume:
        print(f"[modal] resumed by skipping {n_skip_resume} already-done problems", flush=True)
    return overall_rc


@app.function(
    image=image,
    volumes={"/logs": logs_vol},
    timeout=60 * 10,
)
def ls_logs(subdir: str = "") -> list[str]:
    base = Path("/logs") / subdir
    if not base.exists():
        return []
    return sorted(str(p.relative_to("/logs")) for p in base.rglob("*") if p.is_file())


@app.function(
    image=modal.Image.debian_slim(python_version="3.10").pip_install(
        "huggingface_hub==0.25.2"
    ),
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
    timeout=120,
)
def verify_hf_token(model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B") -> dict:
    """Prove that HF_TOKEN is visible and can reach the target model repo."""
    import os

    from huggingface_hub import HfApi, whoami

    token = os.environ.get("HF_TOKEN")
    if not token:
        return {"ok": False, "error": "HF_TOKEN missing from env"}

    out: dict = {"ok": True, "model_id": model_id}
    try:
        out["whoami"] = whoami(token=token).get("name")
    except Exception as e:
        out["ok"] = False
        out["whoami_error"] = f"{type(e).__name__}: {e}"
        return out

    try:
        api = HfApi()
        info = api.model_info(model_id, token=token)
        out["model_reachable"] = True
        out["siblings"] = len(info.siblings or [])
    except Exception as e:
        out["ok"] = False
        out["model_error"] = f"{type(e).__name__}: {e}"
    return out


@app.local_entrypoint()
def check_token():
    res = verify_hf_token.remote()
    print("[check_token] result:")
    for k, v in res.items():
        print(f"  {k}: {v}")
    if not res.get("ok"):
        raise SystemExit(1)


@app.local_entrypoint()
def smoke():
    """Minimal smoke run: 1 AIME problem, max_gen=512, with instrumentation."""
    args = [
        "--model", "ds-r1-llama-8b",
        "--dataset", "AIME24",
        "--max_gen", "512",
        "--data_idx", "0",
        "--budget", "2048",
        "--sink", "512",
        "--recent", "512",
        "--recall_impl", "cuda_cpy",
        "--spec_ret",
        "--corr", "0.9",
        "--warmup", "0",
    ]
    rc = run_pred.remote(args, log_subdir="smoke")
    print(f"[smoke] returned {rc}")
    print("[smoke] logs on volume:")
    for p in ls_logs.remote("smoke"):
        print(f"  {p}")


@app.local_entrypoint()
def profile_aime():
    """Systems-profiling run on 5 short AIME problems with full per-component
    CUDA-event timing, per-step TBT log, and recall log with the
    need_recall_corr mask applied (bytes_actual column).

    Picked these 5 because they all terminated naturally <5K tokens in prior
    runs, so every problem will get steady-state decode without hitting the
    max_gen cap. Total expected wall ~30 min on A100-40GB.

    Writes to log_subdir='profile_aime'.
    """
    problem_ids = [
        "2024-II-4",   # ~3,292 gen tokens previously
        "2024-I-1",    # ~3,008
        "2024-I-3",    # ~3,749
        "2024-I-6",    # ~2,510
        "2024-II-12",  # ~4,202
    ]
    args = [
        "--model", "ds-r1-llama-8b",
        "--dataset", "AIME24",
        "--max_gen", "16384",
        "--data_ids", ",".join(problem_ids),
        "--budget", "2048",
        "--sink", "512",
        "--recent", "512",
        "--recall_impl", "cuda_cpy",
        "--spec_ret",
        "--corr", "0.9",
        "--warmup", "0",
    ]
    # Use the 1h-timeout sibling so blast radius is bounded if anything hangs.
    rc = run_pred_1h.remote(args, log_subdir="profile_aime")
    print(f"[profile_aime] returned {rc}")
    print("[profile_aime] logs on volume:")
    for p in ls_logs.remote("profile_aime"):
        print(f"  {p}")


@app.local_entrypoint()
def math50_test_crash():
    """Run JUST the problem that crashed earlier (test/prealgebra/1139.json)
    with the CUDA event-pool fix. If this completes without a CUDA illegal
    memory access, the fix is good and we can run full math50.
    Uses run_pred_1h (1h timeout) for blast-radius bounding."""
    args = [
        "--model", "ds-r1-llama-8b",
        "--dataset", "MATH50",
        "--max_gen", "16384",
        "--data_ids", "test/prealgebra/1139.json",
        "--budget", "2048",
        "--sink", "512",
        "--recent", "512",
        "--recall_impl", "cuda_cpy",
        "--spec_ret",
        "--corr", "0.9",
        "--warmup", "0",
    ]
    rc = run_pred_1h.remote(args, log_subdir="math50_test_crash")
    print(f"[math50_test_crash] returned {rc}")
    print("[math50_test_crash] logs on volume:")
    for p in ls_logs.remote("math50_test_crash"):
        print(f"  {p}")


@app.local_entrypoint()
def math50_per_problem():
    """Full MATH50 run, one problem per Python subprocess. Slower per problem
    by ~30s but immune to the cross-problem state leak we hit. Same logging
    as the standard math50 entrypoint."""
    args = [
        "--model", "ds-r1-llama-8b",
        "--dataset", "MATH50",
        "--max_gen", "16384",
        "--budget", "2048",
        "--sink", "512",
        "--recent", "512",
        "--recall_impl", "cuda_cpy",
        "--spec_ret",
        "--corr", "0.9",
        "--warmup", "0",
    ]
    rc = run_pred_per_problem.remote(
        args,
        log_subdir="math50",
        dataset_path="accuracy/eval/reasoning/datasets/math50.jsonl",
        id_field="unique_id",
    )
    print(f"[math50_per_problem] returned {rc}")
    print("[math50_per_problem] logs on volume:")
    for p in ls_logs.remote("math50"):
        print(f"  {p}")


@app.local_entrypoint()
def math50():
    """Full MATH50 (49 problems) with the same FreeKV paper-config as full_aime:
    max_gen=16384, sink=512, recent=512, corr=0.9. Per-head sim cached as npz.
    Writes to log_subdir='math50'."""
    args = [
        "--model", "ds-r1-llama-8b",
        "--dataset", "MATH50",
        "--max_gen", "16384",
        "--budget", "2048",
        "--sink", "512",
        "--recent", "512",
        "--recall_impl", "cuda_cpy",
        "--spec_ret",
        "--corr", "0.9",
        "--warmup", "0",
    ]
    rc = run_pred.remote(args, log_subdir="math50")
    print(f"[math50] returned {rc}")
    print("[math50] logs on volume:")
    for p in ls_logs.remote("math50"):
        print(f"  {p}")


@app.local_entrypoint()
def full_aime():
    """Full AIME24 (all 30 problems) with FreeKV's paper-matched config:
    max_gen=16384 (paper Section 5.2 says 16K), sink=512, recent=512, corr=0.9.
    Writes to log_subdir='full_aime'. Includes per-head sim npz, tokens, and
    incremental preds.jsonl per problem.
    """
    args = [
        "--model", "ds-r1-llama-8b",
        "--dataset", "AIME24",
        "--max_gen", "16384",
        "--budget", "2048",
        "--sink", "512",
        "--recent", "512",
        "--recall_impl", "cuda_cpy",
        "--spec_ret",
        "--corr", "0.9",
        "--warmup", "0",
    ]
    rc = run_pred.remote(args, log_subdir="full_aime")
    print(f"[full_aime] returned {rc}")
    print("[full_aime] logs on volume:")
    for p in ls_logs.remote("full_aime"):
        print(f"  {p}")


@app.local_entrypoint()
def verify_dips_v2():
    """Re-run the original verify_dips 5 problems WITH per-head sim caching.
    Writes to log_subdir='verify_dips_v2' — separate from the earlier
    'verify_dips' run which did not cache the per-head tensor."""
    problem_ids = [
        "2024-I-1",
        "2024-I-2",
        "2024-I-3",
        "2024-I-4",
        "2024-II-4",
    ]
    args = [
        "--model", "ds-r1-llama-8b",
        "--dataset", "AIME24",
        "--max_gen", "32000",
        "--data_ids", ",".join(problem_ids),
        "--budget", "2048",
        "--sink", "512",
        "--recent", "512",
        "--recall_impl", "cuda_cpy",
        "--spec_ret",
        "--corr", "0.9",
        "--warmup", "0",
    ]
    rc = run_pred.remote(args, log_subdir="verify_dips_v2")
    print(f"[verify_dips_v2] returned {rc}")
    print("[verify_dips_v2] logs on volume:")
    for p in ls_logs.remote("verify_dips_v2"):
        print(f"  {p}")


@app.local_entrypoint()
def dips_v2():
    """Second dip-analysis run, on 5 AIME24 problems not covered by
    baseline/ or verify_dips/. Uses the enhanced logging that caches the
    full [n_steps, n_layers, n_q_heads] cosine-sim tensor as a compressed
    npz sidecar, plus per-problem tokens + incremental preds.jsonl.

    Writes to log_subdir='dips_v2' — segregated from earlier runs.
    """
    problem_ids = [
        "2024-I-5",
        "2024-I-9",
        "2024-I-14",
        "2024-II-5",
        "2024-II-9",
    ]
    args = [
        "--model", "ds-r1-llama-8b",
        "--dataset", "AIME24",
        "--max_gen", "32000",
        "--data_ids", ",".join(problem_ids),
        "--budget", "2048",
        "--sink", "512",
        "--recent", "512",
        "--recall_impl", "cuda_cpy",
        "--spec_ret",
        "--corr", "0.9",
        "--warmup", "0",
    ]
    rc = run_pred.remote(args, log_subdir="dips_v2")
    print(f"[dips_v2] returned {rc}")
    print("[dips_v2] logs on volume:")
    for p in ls_logs.remote("dips_v2"):
        print(f"  {p}")


@app.local_entrypoint()
def verify_dips():
    """Re-run 5 specific AIME24 problems with token-level logging so we can
    cross-reference cos_sim dips against transition keywords in the generated
    text. Writes to a SEPARATE subdir on the volume ('verify_dips') — does
    NOT disturb the 13-problem baseline run under 'baseline'."""
    problem_ids = [
        "2024-I-1",
        "2024-I-2",
        "2024-I-3",
        "2024-I-4",
        "2024-II-4",
    ]
    args = [
        "--model", "ds-r1-llama-8b",
        "--dataset", "AIME24",
        "--max_gen", "32000",
        "--data_ids", ",".join(problem_ids),
        "--budget", "2048",
        "--sink", "512",
        "--recent", "512",
        "--recall_impl", "cuda_cpy",
        "--spec_ret",
        "--corr", "0.9",
        "--warmup", "0",
    ]
    rc = run_pred.remote(args, log_subdir="verify_dips")
    print(f"[verify_dips] returned {rc}")
    print("[verify_dips] logs on volume:")
    for p in ls_logs.remote("verify_dips"):
        print(f"  {p}")


@app.local_entrypoint()
def smoke_instr():
    """Tight-budget smoke to force the spec_ret + corr path so instrumentation
    actually fires (cos_sim, recall bytes, thought types)."""
    args = [
        "--model", "ds-r1-llama-8b",
        "--dataset", "AIME24",
        "--max_gen", "256",
        "--data_idx", "0",
        "--budget", "256",
        "--sink", "64",
        "--recent", "64",
        "--recall_impl", "cuda_cpy",
        "--spec_ret",
        "--corr", "0.9",
        "--warmup", "0",
    ]
    rc = run_pred.remote(args, log_subdir="smoke_instr")
    print(f"[smoke_instr] returned {rc}")
    print("[smoke_instr] logs on volume:")
    for p in ls_logs.remote("smoke_instr"):
        print(f"  {p}")


@app.local_entrypoint()
def baseline():
    """Full FreeKV baseline on AIME24: 29 problems, --spec_ret --corr 0.9."""
    args = [
        "--model", "ds-r1-llama-8b",
        "--dataset", "AIME24",
        "--max_gen", "32000",
        "--data_idx_to", "29",
        "--budget", "2048",
        "--sink", "512",
        "--recent", "512",
        "--recall_impl", "cuda_cpy",
        "--spec_ret",
        "--corr", "0.9",
        "--warmup", "0",
    ]
    rc = run_pred.remote(args, log_subdir="baseline")
    print(f"[baseline] returned {rc}")
    print("[baseline] logs on volume:")
    for p in ls_logs.remote("baseline"):
        print(f"  {p}")
