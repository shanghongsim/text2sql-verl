#!/usr/bin/env bash
# Multi-turn text2sql GRPO on stock verl 0.9.1, hyperparameters following SkyRL-SQL-7B.
# Everything text2sql-specific comes in through 3 paths: tool config, agent loop config, reward fn.
#
#   bash scripts/train_grpo.sh configs/runs/<run>.env [extra hydra overrides...]
#
# The run file sets the variables below (anything unset falls back to the 7B-repro default).
# Per run, everything lands in runs/<RUN_NAME>/: run.env (copy), train.log, hydra/ (resolved config).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_FILE=${1:?usage: train_grpo.sh configs/runs/<run>.env [overrides...]}
shift
source "$ROOT/.env"
source "$ROOT/.venv/bin/activate"
bash "$ROOT/scripts/apply_verl_patches.sh"   # local verl fixes, see patches/ and README.md
source "$RUN_FILE"

MODEL=${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-$ROOT/data/verl/train.parquet}
VAL_FILE=${VAL_FILE:-$ROOT/data/verl/validation.parquet}
NGPUS=${SLURM_GPUS_ON_NODE:-${NGPUS:-8}}   # under Slurm, train on every GPU the job was given
TP=${TP:-1}
TRAIN_BS=${TRAIN_BS:-256}          # SkyRL: train_batch_size=256, policy_mini_batch_size=256
MINI_BS=${MINI_BS:-$TRAIN_BS}
N=${N:-5}                          # SkyRL: n_samples_per_prompt=5
MAX_TURNS=${MAX_TURNS:-6}          # SkyRL: generator.max_turns=6 (must match tool_config.yaml)
MAX_PROMPT=${MAX_PROMPT:-6000}     # SkyRL: max_prompt_length=6000
MAX_RESPONSE=${MAX_RESPONSE:-16384} # all turns incl. tool outputs (SkyRL: 29000 input cap, 3000/turn)
LR=${LR:-1e-6}
EPOCHS=${EPOCHS:-30}
TOTAL_STEPS=${TOTAL_STEPS:-null}
TEST_FREQ=${TEST_FREQ:-5}
SAVE_FREQ=${SAVE_FREQ:-30}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
PROJECT=${PROJECT:-text2sql-verl}
RUN_NAME=${RUN_NAME:-$(basename "$RUN_FILE" .env)}-$(date +%m%d-%H%M)
RUN_DIR="$ROOT/runs/$RUN_NAME"
CKPT_DIR=${CKPT_DIR:-$RUN_DIR/checkpoints}

# Optional knobs that only exist for some models
OPT=()
if [ -n "${ENABLE_THINKING:-}" ]; then   # Qwen3: chat-template enable_thinking (eval used False)
  OPT+=("+data.apply_chat_template_kwargs.enable_thinking=$ENABLE_THINKING")
fi
if [ -n "${TOOL_FORMAT:-}" ]; then       # hermes (default) | qwen3_coder | ...
  OPT+=("actor_rollout_ref.rollout.multi_turn.format=$TOOL_FORMAT")
fi
if [ -n "${TEST_SUITE_MANIFEST:-}" ]; then   # test-suite reward (scripts/build_test_suites.py); scope: all | empty_gold
  OPT+=("+reward.custom_reward_function.reward_kwargs.test_suite_manifest=$TEST_SUITE_MANIFEST"
        "+reward.custom_reward_function.reward_kwargs.test_suite_scope=${TEST_SUITE_SCOPE:-all}")
fi
if [ -n "${SEED:-}" ]; then                  # repeat a run with different data order + sampling
  OPT+=("data.seed=$SEED" "actor_rollout_ref.rollout.seed=$SEED")
fi

mkdir -p "$RUN_DIR"
cp "$RUN_FILE" "$RUN_DIR/run.env"
printf '%q ' "$@" > "$RUN_DIR/extra_overrides.txt"
echo "run dir: $RUN_DIR"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"   # verl workers import text2sql_verl
# Slurm's GPU gres also sets the AMD-only vars; verl refuses to start when ROCR_ and CUDA_VISIBLE_DEVICES
# are both set. CUDA_VISIBLE_DEVICES (the actual NVIDIA fence from srun) is left untouched.
unset ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1

python -m verl.trainer.main_ppo \
  hydra.run.dir="$RUN_DIR/hydra" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=$TRAIN_BS \
  data.max_prompt_length=$MAX_PROMPT \
  data.max_response_length=$MAX_RESPONSE \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=$LR \
  actor_rollout_ref.actor.grad_clip=0.5 \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BS \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(( (MAX_PROMPT + MAX_RESPONSE) * 2 )) \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.max_num_batched_tokens=$(( MAX_PROMPT + MAX_RESPONSE )) \
  actor_rollout_ref.rollout.n=$N \
  actor_rollout_ref.rollout.temperature=${TEMPERATURE:-0.6} \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$MAX_TURNS \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=10000 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$ROOT/configs/tool_config.yaml" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG:-$ROOT/configs/agent_loop.yaml}" \
  actor_rollout_ref.rollout.agent.default_agent_loop=${AGENT_LOOP:-text2sql_agent} \
  reward.custom_reward_function.path="${REWARD_PATH:-$ROOT/text2sql_verl/reward.py}" \
  reward.custom_reward_function.name=compute_score \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="$PROJECT" \
  trainer.experiment_name="$RUN_NAME" \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.n_gpus_per_node=$NGPUS \
  trainer.nnodes=1 \
  trainer.total_epochs=$EPOCHS \
  trainer.total_training_steps=$TOTAL_STEPS \
  trainer.test_freq=$TEST_FREQ \
  trainer.save_freq=$SAVE_FREQ \
  trainer.val_before_train=$VAL_BEFORE_TRAIN \
  trainer.rollout_data_dir="$RUN_DIR/rollouts" \
  trainer.validation_data_dir="$RUN_DIR/val_generations" \
  trainer.log_val_generations=${LOG_VAL_GENERATIONS:-10} \
  "${OPT[@]}" "$@" 2>&1 | tee "$RUN_DIR/train.log"
