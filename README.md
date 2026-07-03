# MedEnvScale

MedEnvScale builds executable biomedical coding-agent environments from MedAgentGym/BioCoder-style tasks, scales them into multiple difficulty levels, runs tool-agent evaluations, and trains/evaluates SFT/RL LoRA adapters.

This README only covers setup, the main pipeline, important runtime parameters, and LLM configuration.

## 1. Environment Setup

Use Python 3.10+.

```bash
git clone <your-fork-or-repo-url> medenvscale
cd medenvscale

conda create -n medenvscale python=3.10 -y
conda activate medenvscale

python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

If you use GPU training or local model inference, install a PyTorch build matching your CUDA driver first, then install the remaining requirements. See the official PyTorch install selector for the correct command.

Candidate code is executed in a separate Python environment configured by `local_python_bin`. For example:

```yaml
dataset:
  code_execution:
    local_python_bin: /path/to/execution/env/bin/python
```

That execution environment should contain task-specific libraries required by the dataset, such as `numpy`, `pandas`, `biopython`, `scipy`, and any packages used by the seed solutions.

## 2. Data And Output Layout

本项目使用的数据集为 **MedAgentGym/MedAgentGym-Data**，数据集地址：

```
https://huggingface.co/datasets/MedAgentGym/MedAgentGym-Data
```

由于该数据集托管在 Hugging Face 上，下载前需要先登录 Hugging Face 账号，并在数据集页面完成访问授权。

### 1. 安装 Hugging Face 下载工具

```
pip install -U huggingface_hub hf_xet
```

### 2. 登录 Hugging Face 账号

在终端中执行：

```
hf auth login
```

随后根据提示粘贴自己的 Hugging Face Access Token。

Access Token 可以在 Hugging Face 账户设置页面中创建，权限选择 `Read` 即可。

登录完成后，可以使用以下命令检查是否登录成功：

```
hf auth whoami
```

### 3. 下载数据集

建议将数据集下载到项目根目录下的 `data/` 文件夹中：

```
mkdir -p ./data/medagentgym

hf download MedAgentGym/MedAgentGym-Data \
  --repo-type dataset \
  --local-dir ./data/medagentgym
```

下载完成后，数据集会保存在：

```
./data/medagentgym
```



Default config:

```bash
configs/biocoder/medagentgym_pilot.yaml
```

Default dataset name:

```bash
biocoder
```

For a new machine, make a local copy of the config and edit paths there:

```bash
cp configs/biocoder/medagentgym_pilot.yaml configs/biocoder/medagentgym_local.yaml
```

Then edit medagentgym_local.yaml:

主要配置下面三个

```yaml
dataset:
  dataset_root: /path/to/medagentgym(下载的数据集的绝对路径)
  default_dataset: biocoder
  task_files:
    train: train_tasks.jsonl
    test: test_tasks.jsonl
  code_execution:
    local_python_bin: /path/to/execution/env/bin/python（创建环境，该环境的绝对路径）

stage06:
  llm_config: configs/agent_llm.yaml

stage09_rlvr_grpo:
  base_model: /path/to/base/model（要训练的模型路径，可以下载qwen3.5-2B-Base）
  sft_adapter: experiments/biocoder/tool_sft_lora/adapter
```

With `--dataset biocoder`, outputs are written under:

```text
data/biocoder/
result/biocoder/
experiments/biocoder/
```

Raw MedAgentGym task files are read from the dataset root in the config:

```yaml
dataset:
  dataset_root: /path/to/medagentgym
  default_dataset: biocoder
  task_files:
    train: train_tasks.jsonl
    test: test_tasks.jsonl
```

For `--dataset biocoder`, the expected files are:

```text
/path/to/medagentgym/biocoder/train_tasks.jsonl
/path/to/medagentgym/biocoder/test_tasks.jsonl
```

Each JSONL row should describe one executable coding task. The loader accepts several common field names; this is a recommended minimal shape:

```json
{
  "task_id": "example_001",
  "instruction": "Write a Python function that ...",
  "code": "def reference_solution(...):\n    ...",
  "ground_truth": "Optional natural-language answer or notes",
  "resources": ["optional_relative_resource_file.csv"],
  "category": "bioinformatics"
}
```

Common aliases are also accepted:

```text
instruction/problem/question/prompt/description
code/full_code/reference_code
ground_truth/solution/answer/expected_answer
resources/resource_files/resource_paths/files/artifacts
```

Resource paths are interpreted relative to the dataset item/resource layout used by the raw dataset. The executable `code` field is used by Stage 00 to build a seed execution case and expected output signature.

Use your local config copy in commands:

```bash
--config configs/biocoder/medagentgym_local.yaml
```

## 3.LLM Configuration

There are two LLM configs.

### Generation LLM

Used by Stage 00-05 for repair, routing, scaling, operator synthesis, and oracle/gold generation.

```bash
configs/llm.yaml
```

Example API config:

这几个参数值都是可以换的，模型当然是越强越好

```yaml
api:
  base_url: https://api.deepseek.com 
  api_key_env: DEEPSEEK_API_KEY
  model: deepseek-v4-flash
  temperature: 0.2
  response_format: json
```



```bash
export DEEPSEEK_API_KEY=...
```

### Agent LLM

Used by Stage 06, Stage 07 teacher trajectories, and API-mode tool-agent evaluation.

```bash
configs/agent_llm.yaml
```

Example API config:

和上面一样

```yaml
api:
  base_url: https://maas-api.cn-huabei-1.xf-yun.com/v2
  api_key_env: AGENT_LLM_API_KEY
  model: xopqwen35v35b
  temperature: 0.2
  response_format: text
```

Run with:

```bash
export AGENT_LLM_API_KEY=...
```

For local model inference:

```yaml
local:
  enabled: true
  model_path: /path/to/base/model
  device_map: auto
  torch_dtype: auto
  max_new_tokens: 2048
  do_sample: false
```

Or pass the path directly:

```bash
python scripts/06_run_tool_agent.py \
  --config configs/biocoder/medagentgym_local.yaml \
  --llm_mode local \
  --model_path /path/to/base/model
```

## 

## 4. Main Pipeline（主要跑这里）

### Generate Scaled Data: Stage 00 To 05_5

Run the data-generation block with one command:

```bash
python scripts/00_05_5_generate_data.py \
  --config configs/biocoder/medagentgym_local.yaml \
  --dataset biocoder \
  --workers 8 \
  --resume
```

Useful options:

```bash
--limit N          # debug on N seed tasks
--sample_seed S    # sample before applying --limit
--workers N        # used by supported stages: 00, 02, 05
--resume           # resume between stages and inside supported stages
--stop_stage 05    # stop at a specific stage: 00, 01, 02, 03, 04, 05, 05_5
--llm_mode mock    # smoke test without real LLM calls
--llm_mode api     # use configs/llm.yaml for generation/scaling calls
```

Stage 05_5 assigns `train/dev/test` labels used by later stages. Split ratios are configured in `stage05_5.split_ratios`.

### Evaluate Base Agent: Stage 06(这个可以先不跑)

```bash
python scripts/06_run_tool_agent.py \
  --config configs/biocoder/medagentgym_local.yaml \
  --dataset biocoder \
  --split test \
  --llm_mode local \
  --model_path /path/to/base/model \
  --resume
```

Useful options:

```bash
--split train|dev|test|all
--workers N          # API mode can parallelize; local mode is forced to 1
--retry_failed       # rerun retryable failed rows
--user_feedback      # add user repair feedback after public preflight failure
```

### Build Tool-SFT Data: Stage 07

```bash
python scripts/07_generate_tool_sft_data.py \
  --config configs/biocoder/medagentgym_local.yaml \
  --dataset biocoder \
  --llm_mode api \
  --workers 8 \
  --resume
```

The teacher agent is controlled by `configs/agent_llm.yaml`. Stage 07 writes OpenAI-style tool-call trajectories for Stage 08.



### Train SFT LoRA: Stage 08

Dry run first:

**这几个命令里面的teacher_slug就是前面配置configs/agent_llm.yaml这个文件里的model**

```bash
python scripts/08_train_sft_lora.py \
  --config configs/biocoder/medagentgym_local.yaml \
  --dataset biocoder \
  --teacher_slug xopqwen35v35b \
  --dry_run
```

Single-process training:

```bash
python scripts/08_train_sft_lora.py \
  --config configs/biocoder/medagentgym_local.yaml \
  --dataset biocoder \
  --teacher_slug xopqwen35v35b \
  --model_name_or_path /path/to/base/model \
  --max_steps 200 \
  --resume
```

Multi-GPU training:（**推荐用这个**）

```bash
torchrun --nproc_per_node=NUM_GPUS scripts/08_train_sft_lora.py \
  --config configs/biocoder/medagentgym_local.yaml \
  --dataset biocoder \
  --teacher_slug xopqwen35v35b \
  --model_name_or_path /path/to/base/model \ 要训练的模型的路径
  --resume
```

Training defaults live in:

```bash
configs/biocoder/train_sft.yaml
```

### Evaluate SFT Adapter: Stage 08_5 

```bash
python scripts/08_5_eval_sft_adapter.py \
  --config configs/biocoder/medagentgym_local.yaml \
  --dataset biocoder \
  --split test \
  --resume
```

Override paths when needed:

```bash
--model_path /path/to/base_model
--sft_adapter /path/to/sft_adapter
--retry_failed
```

### RLVR / GRPO: Stage 09

Train with TRL GRPO:

```bash
torchrun --nproc_per_node=NUM_GPUS scripts/09_train_rlvr_grpo.py \
  --config configs/biocoder/medagentgym_local.yaml \
  --dataset biocoder \
  --split train \
  --eval_split dev \
  --eval_steps 25 \
  --train \
  --resume
```

Important Stage 09 config:

```yaml
stage09_rlvr_grpo:
  base_model: /path/to/base/model
  sft_adapter: experiments/biocoder/tool_sft_lora/adapter
  rollouts_per_env: 4
  num_generations: 4
  max_steps: 100
  reward:
    sample_pass_weight: 1.0
    case_pass_rate_weight: 0.5
    rubric_score_weight: 0.4
    tool_budget_penalty: 0.1
```

### Evaluate RL Adapter: Stage 09_5

```bash
python scripts/09_5_eval_rl_adapter.py \
  --config configs/biocoder/medagentgym_local.yaml \
  --dataset biocoder \
  --split test \
  --resume
```

Optional:

```bash
--model_path /path/to/base_model
--rl_adapter /path/to/rl_adapter
--retry_failed
```

### Compare Base / SFT / SFT+RL On Original Tasks: Stage 10

Stage 10 evaluates original Stage 00 tasks, not scaled difficulty tasks. Ground truth comes from each raw task's executable seed result.

```bash
python scripts/10_eval_original_models.py \
  --config configs/biocoder/medagentgym_local.yaml \
  --dataset biocoder \
  --resume
```

Useful options:

```bash
--model_path /path/to/base_model
--sft_adapter /path/to/sft_adapter
--rl_adapter /path/to/rl_adapter
--no_sft
--no_rl
--retry_failed
```

Stage 10 prints which model is currently running: `base`, `sft`, or `sft+RL`.

## 5. Important Parameters

Common script parameters:

```bash
--config PATH       # required project config
--dataset NAME      # output/input namespace, e.g. biocoder
--limit N           # debug subset
--sample_seed S     # deterministic sampling before --limit
--llm_mode mock|api|local
--workers N         # supported stages only
--resume            # reuse complete outputs/checkpoints
```

Agent/eval parameters:

```bash
--split train|dev|test|all
--retry_failed
--user_feedback
--model_path PATH
--sft_adapter PATH
--rl_adapter PATH
```

Training parameters:

```bash
--dry_run
--max_steps N
--teacher_slug NAME
--train
--rollout_only
--collect_rollouts
--use_existing_rollouts
--eval_split dev
--eval_steps N
```

## 6. Key Outputs

```text
data/biocoder/interim/                 # normalized/routed/seed/scaled intermediates
data/biocoder/processed/               # clean/rejected envs, quality reports, RLVR envs
data/biocoder/splits/                  # Stage07 tool-SFT train/dev/test files
result/biocoder/06/                    # base agent evaluation
result/biocoder/08_5/                  # SFT adapter evaluation
result/biocoder/09/                    # RL rollouts/reward reports
result/biocoder/09_5/                  # RL adapter evaluation
result/biocoder/10/original_model_eval # original-task model comparison
experiments/biocoder/tool_sft_lora/    # Stage08 SFT adapter
experiments/biocoder/tool_rl_grpo_lora/# Stage09 RL adapters
```

## 7. Smoke Test

```bash
PYTHONPATH=src python -m unittest tests.test_pipeline_smoke
```

For a fast mock pipeline:

```bash
python scripts/00_05_5_generate_data.py \
  --config configs/biocoder/medagentgym_local.yaml \
  --dataset biocoder \
  --limit 5 \
  --llm_mode mock \
  --resume
```