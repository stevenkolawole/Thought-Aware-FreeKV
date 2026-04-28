#!/bin/bash
#SBATCH --job-name=ds_llama8b
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=8:00:00
#SBATCH --output=/data/user_data/akouloge/freekv/ds_llama8b.out
#SBATCH --error=/data/user_data/akouloge/freekv/ds_llama8b.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akouloge@andrew.cmu.edu


# turn on env
cd /home/akouloge/Thought-Aware-FreeKV
source .venv/bin/activate

# run command
python source/pred.py \
  --model ds-r1-llama-8b \
  --dataset AIME24 \
  --spec_ret \
  --recall_impl cuda_cpy \
  --budget 2048 \
  --sink 512 \
  --recent 512 \
  --cpu_layout HND \
  --max_gen 8192