#!/usr/bin/env bash
# Queue a training run as a 4-GPU Slurm batch job (half a node; two runs fit side by side).
#   bash scripts/sbatch_train.sh configs/runs/<run>.env [extra hydra overrides...]
# Log: logs/<run>.out. Slurm's CUDA_VISIBLE_DEVICES is the GPU fence; never override it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_FILE=${1:?usage: sbatch_train.sh configs/runs/<run>.env [overrides...]}
NAME=$(basename "$RUN_FILE" .env)
mkdir -p "$ROOT/logs"
sbatch --gres=gpu:4 --cpus-per-task=96 --mem=800G --nodes=1 --ntasks=1 -J "$USER-$NAME" \
  -o "$ROOT/logs/$NAME.out" -D "$ROOT" \
  --wrap "bash $ROOT/scripts/train_grpo.sh $(printf '%q ' "$@")"
