# MedEnvScale MedAgentGym

`medenvscale` 现在默认以 `MedAgentGym` 风格的可执行生物医学 agent task 作为 seed 数据，而不是原来的 MedQA 多选题。

现在所有阶段脚本都支持 `--dataset`。当传入例如 `--dataset biocoder` 时，数据输入会从
`/archive/zengjiaqi/dataset/medagentgym/biocoder/` 读取，对应的原始数据、阶段中间产物、结果和实验报告会写到：

- `data/biocoder/`
- `result/biocoder/`
- `experiments/biocoder/`

公共配置仍然保留在根目录，例如 `configs/llm.yaml`、`configs/agent_llm.yaml`、`configs/biocoder/task_axis_priority.yaml`。

当前实现重点做了这几件事：

- 输入改成 `train_tasks.jsonl` / `test_tasks.jsonl` 一类的本地 JSONL 任务文件。
- Stage 00 会优先使用每条任务自带的 `code` 样例做可执行性检查：在隔离运行环境里执行脚本，保存标准化 `stdout`、`return value` 和文件产物摘要；若脚本失败，会最多尝试 3 次 LLM repair，repair 成功后会用新 `code` 覆盖，并把旧版本存到 `wrong_code`。
- 前半段 pipeline 改成 `MedAgentGym task -> routing -> seed task -> executable environment`。
- 保留了后半段的 difficulty scaling、tool-agent rollout、question points、rubrics、SFT / DPO / PRM / RLVR 导出。
- `05_apply_scaling_operators.py` 现在会按实验方案走 LLM-first 的 7-axis weight planning 和 dynamic operator synthesis；`mock` 模式用于 smoke test，`api` 模式会实际调用模型。
- 全项目统一采用 `trust raw gold` policy：原始数据集里的 `gold solution` 被视为可信参考答案，不做 gold compatibility gate，也不再产出 `gold_compatibility_results.jsonl`。
- Stage 05 生成 scaled task、validated oracle cases、scaled gold 和 clean/rejected 结果。
- Stage 06 使用独立 agent API 配置运行 tool-calling coding agent：agent 只能访问公开上下文和公开资源，通过工具自测后提交完整代码；系统再用 Stage 05 的 `validated_oracle_cases` 做隐藏评估，但不会把 oracle case 暴露给 agent。
- 内部仍保留了一部分旧字段名做兼容，所以现有 rubric/export 逻辑还能继续复用。

## Quick start

```bash
cd medenvscale
python -m pip install -r requirements.txt

# Stage 00-05 的生成/修题模型使用 configs/llm.yaml
export DEEPSEEK_API_KEY=...

# Stage 06 的被评估 coding agent 使用 configs/agent_llm.yaml
export AGENT_LLM_API_KEY=...

# 把真实 MedAgentGym 任务文件放到：
# data/raw/medagentgym/source/train_tasks.jsonl
# data/raw/medagentgym/source/test_tasks.jsonl

python scripts/00_prepare_medagentgym.py --config configs/biocoder/medagentgym_pilot.yaml --limit 20 --llm_mode api
python scripts/01_normalize_medagentgym.py --config configs/biocoder/medagentgym_pilot.yaml --limit 20
python scripts/02_route_medagentgym_task.py --config configs/biocoder/medagentgym_pilot.yaml --limit 20 --llm_mode mock
python scripts/03_generate_seed_cases.py --config configs/biocoder/medagentgym_pilot.yaml --limit 20
python scripts/04_generate_environment_skeletons.py --config configs/biocoder/medagentgym_pilot.yaml --limit 20 --llm_mode mock
python scripts/05_apply_scaling_operators.py --config configs/biocoder/medagentgym_pilot.yaml --limit 20
python scripts/06_run_tool_agent.py --config configs/biocoder/medagentgym_pilot.yaml --limit 20 --llm_mode mock
python scripts/07_generate_qpoints_rubrics_answers.py --config configs/biocoder/medagentgym_pilot.yaml --limit 20 --llm_mode mock
python scripts/08_build_safety_gates.py --config configs/biocoder/medagentgym_pilot.yaml
python scripts/09_export_training_views.py --config configs/biocoder/medagentgym_pilot.yaml
python scripts/10_quality_filter.py --config configs/biocoder/medagentgym_pilot.yaml
python scripts/11_make_splits.py --config configs/biocoder/medagentgym_pilot.yaml
python scripts/15_eval_all.py --config configs/biocoder/medagentgym_pilot.yaml
```

`--llm_mode mock` 可用于 smoke test；真实 Stage 06 agent rollout 使用 `--llm_mode api` 或配置里的 `stage06.llm_mode: api`。

## Expected raw format

项目会优先读取本地 JSONL，每条任务尽量包含这些字段中的一部分：

```json
{
  "task_id": "stroke_pathway_db",
  "source_split": "train",
  "task_family": "diagnostic_workup",
  "category": "neurology",
  "instruction": "Query the stroke pathway database ...",
  "ground_truth": "Prioritize urgent vessel imaging ...",
  "resources": ["stroke/pathway.db"],
  "system_prompt": "You are a biomedical coding agent ...",
  "verifier_reference": {"expected_columns": ["patient_id", "needs_cta"]}
}
```

字段名不需要完全一致；normalizer 也会兼容 `prompt`、`question`、`problem_description`、`expected_output`、`resource_files` 等常见名字。

## Output summary

主要输出仍然落在：

- `data/<dataset>/interim/`
- `data/<dataset>/processed/`
- `data/<dataset>/splits/`
- `result/<dataset>/`
- `experiments/<dataset>/reports/`

新增的 Stage 06 agent rollout 会写到：

- `result/<dataset>/06/agent_runs.jsonl`
- `result/<dataset>/06/agent_traces.jsonl`
- `result/<dataset>/06/agent_eval_report.jsonl`

其中 smoke test 会直接用内置的 demo `MedAgentGym` rows 验证整条 mock pipeline 可以跑通。
