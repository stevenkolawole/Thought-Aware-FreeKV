#!/bin/bash
#SBATCH --job-name=math50_f1
#SBATCH --partition=general
#SBATCH --gres=gpu:A100_40GB:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=8:00:00
#SBATCH --output=/data/user_data/akouloge/math50_fetch1.out
#SBATCH --error=/data/user_data/akouloge/math50_fetch1.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akouloge@andrew.cmu.edu

cd /home/akouloge/Thought-Aware-FreeKV
source .venv/bin/activate

LOG_DIR=/data/user_data/akouloge/logs/math50_fetch1
mkdir -p "$LOG_DIR"

python source/pred.py \
  --model ds-r1-llama-8b \
  --dataset MATH50 \
  --spec_ret \
  --recall_impl cuda_cpy \
  --budget 2048 \
  --sink 512 \
  --recent 512 \
  --cpu_layout HND \
  --max_gen 8192 \
  --fetch_interval 1 \
  --log_dir "$LOG_DIR"

# pred.py writes to a fixed path; copy to a fetch-interval-specific location
PRED_SRC=tmp_res/ds-r1-llama-8b/MATH50.jsonl
PRED_DST=/data/user_data/akouloge/math50_fetch1_preds.jsonl
if [ -f "$PRED_SRC" ]; then
    cp "$PRED_SRC" "$PRED_DST"
    echo "Predictions saved to $PRED_DST"
fi
