# text2sql-verl

Multi-turn text-to-SQL GRPO on [verl](https://github.com/volcengine/verl) 0.9.1, following the SkyRL-SQL recipe. The model explores a SQLite database with an `execute_sql` tool, then answers with `submit_sql`. The reward is execution match against the gold SQL.

Everything text2sql-specific plugs into the installed verl through its config, including a subclass of verl's `ToolAgentLoop`, so verl isn't forked. The only changes to verl itself are two small patches to its v1 trainer.

## Layout

```
text2sql_verl/
  agent_loop.py     Text2SQLAgentLoop: ToolAgentLoop + two episode-end rules
  tools.py          execute_sql and submit_sql (verl BaseTool subclasses)
  sql_exec.py       read-only SQLite execution with a timeout, shared by tools and reward
  reward.py         -1 no submit / 0 wrong / 1 result sets match (optional test-suite matching)
  test_suite.py     random database copies for test-suite matching (optional)
configs/
  agent_loop.yaml   registers text2sql_agent for verl
  tool_config.yaml  tool schemas and settings
  runs/*.env        one file per run (model, batch size, n, steps, ...)
patches/            the two verl 0.9.1 patches
scripts/
  apply_verl_patches.sh  applies patches/ to the installed verl (idempotent)
  train_grpo.sh          launches verl.trainer.main_ppo for a run file
  sbatch_train.sh        same, as a 4-GPU Slurm batch job
  test_plugins.py        CPU-only checks for tools, reward and agent-loop config
  export_jsonl.py        SkyRL-SQL parquet -> canonical JSONL
  build_verl_dataset.py  JSONL -> verl agent-loop parquet (prompt, tools_kwargs, ground truth)
  build_test_suites.py   pre-builds test-suite database copies (optional)
```

## How it plugs into verl

| Piece | How verl loads it |
|---|---|
| `Text2SQLAgentLoop` (`text2sql_verl/agent_loop.py`) | `actor_rollout_ref.rollout.agent.agent_loop_config_path` → `configs/agent_loop.yaml` |
| `ExecuteSQLTool`, `SubmitSQLTool` (`text2sql_verl/tools.py`) | `actor_rollout_ref.rollout.multi_turn.tool_config_path` → `configs/tool_config.yaml` |
| Reward (`text2sql_verl/reward.py`) | `reward.custom_reward_function.path` |

Each dataset row carries the database path to the tools (`extra_info.tools_kwargs.*.create_kwargs.db_path`). `submit_sql` stores its answer in the agent's `extra_fields`, which verl passes to the reward as `extra_info`, so the reward never parses model text.

## The ToolAgentLoop subclass

`Text2SQLAgentLoop` is about 15 lines and adds two rules that stock `ToolAgentLoop` lacks:

1. **End the episode once `submit_sql` is called.** Stock `ToolAgentLoop` only ends when the model makes no tool call, or hits the turn or length limit.
2. **Count tool calls made on the last turn.** Stock verl checks `assistant_turns >= max_assistant_turns` right after generation, before it parses tool calls (`tool_agent_loop.py:292` in 0.9.1). A `submit_sql` on the final turn is dropped and scored −1, even though the tool's turn reminder tells the model to submit on that turn. The subclass takes the limit over as `turn_budget` and enforces it after the tools run, so the last turn's answer counts, as in SkyRL.

It overrides a private method (`_handle_processing_tools_state`) and resets the `max_assistant_turns` attribute, so a verl upgrade could break it without warning. Keep verl pinned at 0.9.1. `scripts/test_plugins.py` checks that the config resolves to the subclass.

**To change the loop more deeply**, copy `verl/experimental/agent_loop/tool_agent_loop.py` into `text2sql_verl/`, rename the class, remove its `@register("tool_agent")` decorator (otherwise importing it re-points verl's stock `tool_agent`), and point `_target_` in `configs/agent_loop.yaml` at it. That's still your code, not a verl fork.

## The verl patches

Both patch `verl/trainer/ppo/v1/trainer_base.py`. `scripts/apply_verl_patches.sh` applies them to the installed verl, skips ones already applied, and refuses any verl version other than 0.9.1. `train_grpo.sh` runs it before every launch. Rerun it after reinstalling verl.

1. **`verl-0.9.1-v1-trainer-optional-ref-critic-fields.patch` fixes a crash.** With GRPO and no KL term, the 0.9.1 v1 trainer still requests `ref_log_prob`, `values` and `token_level_rewards` from its TransferQueue. Those fields are never written, so the request fails with `Fields [...] are not ready`. The patch requests each field only when it exists. Config changes can't avoid this, because `values` is requested even without a critic, and GRPO never has one. Upstream fixed `values` in #8461, after 0.9.1. As of 2026-11-04, `ref_log_prob` and `token_level_rewards` were still unguarded on main. The legacy trainer (`trainer.use_v1=False`) doesn't need the patch.
2. **`verl-0.9.1-v1-trainer-zero-variance-group-metrics.patch` adds logging only.** It logs `training/groups/{num_groups,zero_variance_frac,all_positive_frac,all_zero_frac,all_negative_frac}`: the share of GRPO groups (samples sharing a prompt) whose rewards are all equal and so contribute no gradient, split by the reward's sign. Training works the same without it.

## Setup

Versions used: Python 3.12, verl 0.9.1, vLLM 0.24.0 (torch 2.11, CUDA 13), transformers 5.9.0, flash-attn 2.8.3.

```bash
uv venv .venv && source .venv/bin/activate
uv pip install vllm==0.24.0 --torch-backend=cu130
uv pip install verl==0.9.1
uv pip install -e .
cp .env.example .env    # then fill in HF_TOKEN and WANDB_API_KEY
```

verl's FSDP workers default to flash-attention 2, and there's no prebuilt flash-attn wheel for torch 2.11. Build it from source against a CUDA 13.0 toolkit whose nvcc and headers match.

## Data

1. Download the train and validation splits of `NovaSky-AI/SkyRL-SQL-653-data-newfmt` from Hugging Face. Train has 653 questions (540 SynSQL, 113 Spider). Validation is Spider dev (1034 questions). Save them as `raw/skyrl/newfmt_train.parquet` and `raw/skyrl/newfmt_validation.parquet`.
2. Download `data.zip` from `seeklhy/OmniSQL-datasets` (22 GB). Extract `spider/database/` and the SynSQL databases the training set uses into `data/db/`, keeping the zip's layout (`data/db/spider/database/...`, `data/db/SynSQL-2.5M/databases/...`).
3. Convert:

```bash
python scripts/export_jsonl.py --parquet raw/skyrl/newfmt_train.parquet --db-root data/db --out data/jsonl/train.jsonl
python scripts/export_jsonl.py --parquet raw/skyrl/newfmt_validation.parquet --db-root data/db --out data/jsonl/validation.jsonl
python scripts/build_verl_dataset.py --jsonl data/jsonl/train.jsonl --out data/verl/train.parquet
python scripts/build_verl_dataset.py --jsonl data/jsonl/validation.jsonl --out data/verl/validation.parquet
python scripts/build_verl_dataset.py --jsonl data/jsonl/validation.jsonl --out data/verl/validation_64.parquet --limit 64
```

`export_jsonl.py` reports how many database files are missing; it should be 0.

## Run

```bash
python scripts/test_plugins.py                                  # CPU-only plugin checks
bash scripts/train_grpo.sh configs/runs/smoke_qwen3_1.7b.env    # 3 steps, validation on 64 Spider-dev questions
bash scripts/train_grpo.sh configs/runs/grpo_qwen3_1.7b.env     # full run: Qwen3-1.7B, about 60 steps
bash scripts/sbatch_train.sh configs/runs/grpo_qwen3_1.7b.env   # the same as a 4-GPU Slurm job; log in logs/
```

Each run writes to `runs/<run>-<MMDD-HHMM>/`: a copy of the run file, `train.log`, the resolved Hydra config, rollout and validation dumps, and checkpoints. Extra Hydra overrides can follow the run file on the command line. Under Slurm, `train_grpo.sh` trains on every GPU the job was given, whatever `NGPUS` the run file sets.

For test-suite rewards, run `python scripts/build_test_suites.py data/verl/train.parquet` and set `TEST_SUITE_MANIFEST=$ROOT/data/test_suite/manifest.json` in the run file. `TEST_SUITE_SCOPE=empty_gold` applies it only where the gold result is empty.

## Gotchas

- **The parquet `agent_name` column overrides `default_agent_loop`.** `build_verl_dataset.py` sets it to `text2sql_agent`. If you register a different loop, change it there too, or verl will run the old loop without any error.
- **Slurm's GPU allocation also sets `ROCR_VISIBLE_DEVICES`, and verl refuses to start when that's set alongside `CUDA_VISIBLE_DEVICES`.** `train_grpo.sh` unsets the AMD-only variables. Don't override `CUDA_VISIBLE_DEVICES`.
- **Don't edit `train_grpo.sh` in place while runs are using it.** Bash reads a script as it executes, so a running job can execute garbage after training ends. Write a new file and `mv` it over.
- **`multi_turn.max_tool_response_length` defaults to 256 characters**, which truncates query results. `train_grpo.sh` sets it to 10000.
