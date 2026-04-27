#!/bin/bash
#SBATCH --job-name=fkv_p0_full_kv_aime24
#SBATCH --partition=general
#SBATCH --output=/home/%u/workspace/FreeKV/slurm_logs/phase0/full_kv_aime24_%j.out
#SBATCH --error=/home/%u/workspace/FreeKV/slurm_logs/phase0/full_kv_aime24_%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=96G
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=skolawol@andrew.cmu.edu

# Phase 0 — Full-KV AIME24 reproduction baseline.
# No offload, no eviction, no recall. The accuracy upper bound + slow-but-clean
# TPOT reference. 30 problems, max_gen=8192, greedy.

set -euo pipefail

WORKDIR=/home/skolawol/workspace/FreeKV
VENV=$WORKDIR/.venv

source "$VENV/bin/activate"

export HF_HOME=/data/hf_cache/skolawol
export HF_HUB_CACHE=/data/hf_cache/skolawol/hub
export HF_DATASETS_CACHE=/data/hf_cache/skolawol/datasets
export TRANSFORMERS_CACHE=/data/hf_cache/skolawol
export TORCH_EXTENSIONS_DIR=/data/user_data/skolawol/torch_extensions/freekv_shared

cd "$WORKDIR"
mkdir -p slurm_logs/phase0

LOG_DIR=/data/user_data/skolawol/freekv_logs/phase0/full_kv_aime24_${SLURM_JOB_ID:-local}
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Phase 0 baseline: Full-KV AIME24"
echo "Node: $(hostname)   Job: ${SLURM_JOB_ID:-n/a}"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Logs: $LOG_DIR"
echo "Started: $(date)"
echo "=========================================="

cd "$WORKDIR"
python source/pred.py \
    --model ds-r1-llama-8b \
    --dataset AIME24 \
    --temperature 0.0 \
    --max_gen 8192 \
    --data_idx_to 30 \
    --warmup 0 \
    --budget 0 \
    --recall_impl cuda_cpy \
    --log_dir "$LOG_DIR" \
    --run_tag full_kv

EXIT=$?
echo ""
echo "=========================================="
echo "Complete at $(date) (exit $EXIT)"
echo "Logs: $LOG_DIR"
echo "=========================================="
exit $EXIT
