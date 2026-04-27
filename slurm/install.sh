#!/bin/bash
#SBATCH --job-name=install_freekv_micro
#SBATCH --partition=general
#SBATCH --output=/home/%u/workspace/FreeKV/slurm_logs/install_freekv_%j.out
#SBATCH --error=/home/%u/workspace/FreeKV/slurm_logs/install_freekv_%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=skolawol@andrew.cmu.edu

set -euo pipefail

WORKDIR=/home/skolawol/workspace/FreeKV
VENV=$WORKDIR/.venv
LOGDIR=$WORKDIR/slurm_logs

mkdir -p "$LOGDIR"
cd "$WORKDIR"

if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "[ERROR] Missing venv at $VENV"
  exit 1
fi

source "$VENV/bin/activate"

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
# Conservative defaults: large node allocation, but limited concurrent NVCC jobs.
export MAX_JOBS="${MAX_JOBS:-6}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$MAX_JOBS}"
export NVCC_THREADS=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

echo "=========================================="
echo "FreeKV editable install job"
echo "Host:   $(hostname)"
echo "Job:    ${SLURM_JOB_ID:-n/a}"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo n/a)"
echo "RAM:    $(free -h | awk '/Mem:/ {print $2 " total, " $7 " available"}')"
echo "MAX_JOBS=${MAX_JOBS} CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo "Start:  $(date)"
echo "=========================================="

cd "$WORKDIR/source"
uv pip install -U setuptools wheel ninja cmake
uv pip install -e . --no-build-isolation

python - <<'PY'
import importlib
for m in ('freekv', 'freekv_cpp'):
    mod = importlib.import_module(m)
    print(m, 'OK', getattr(mod, '__file__', None))
PY

echo "=========================================="
echo "SUCCESS at $(date)"
echo "freekv editable install completed in $VENV"
echo "=========================================="
