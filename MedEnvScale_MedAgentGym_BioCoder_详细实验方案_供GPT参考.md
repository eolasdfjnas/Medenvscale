# MedEnvScale-MedAgentGym-BioCoder：详细实验方案（供外部 GPT / 研究讨论使用）

## 0. 文档目的

这份文档的目标不是介绍一个理想化的未来系统，而是**尽可能准确地描述当前 `/home/zengjiaqi/medenvscale` 项目已经实现和正在演进的实验方案**，方便拿给外部 GPT、协作者或论文写作辅助工具做上下文理解。

文档重点回答以下问题：

1. 这个项目当前到底在做什么。
2. 数据来自哪里，经过哪些阶段。
3. 当前最重要的中间对象和结果文件是什么。
4. 7 个 difficulty axis 在当前实现里分别代表什么。
5. Stage 05 的当前设计是什么，为什么它最复杂。
6. 当前系统的已实现部分、验证逻辑、以及已知问题分别是什么。

---

## 1. 项目一句话定义

**当前项目的核心目标是：**

> 以 MedAgentGym / BioCoder 风格的可执行生物医学代码任务为种子数据，先将原始任务清洗为可执行、可观察、可复现的 seed environment，再通过 7 轴动态 operator 对任务进行 M1-M4 难度扩展，并导出可用于后训练与评测的结构化数据。

它和传统 `prompt -> answer` 数据构造方式不同，当前项目强调的是：

```text
可执行代码任务
+ 可观察输出
+ 结构化环境状态
+ 动态难度扩展
+ 可验证的扩展结果
→ 训练 / 评测数据
```

---

## 2. 当前项目定位

当前项目更接近以下类型的研究系统：

- **executable dataset construction**
- **environment-based difficulty scaling**
- **LLM-assisted benchmark generation**
- **post-training data pipeline for code / agent tasks**

它已经不再是早期的 MedQA 风格“多选题 + 文本 gold answer”的流水线，而是改成了：

```text
BioCoder code task
→ executable seed task
→ executable environment
→ dynamic scaling
→ verifier / gate
→ export views
```

因此，当前项目最重要的不是“生成一段看起来像答案的文本”，而是：

1. 原始代码任务能否执行。
2. 扩展后的任务是否真的变难。
3. 扩展后的正确实现是否可执行。
4. 扩展后的验证样例是否真能验证新增语义。

---

## 3. 当前项目目录与组织方式

项目根目录在：

```text
/home/zengjiaqi/medenvscale
```

当前主要目录包括：

```text
configs/
data/
experiments/
prompts/
result/
scripts/
src/
tests/
```

当前采用**按数据集隔离**的组织方式。以 `biocoder` 为例：

```text
configs/biocoder/
data/biocoder/
result/biocoder/
experiments/biocoder/
```

数据集原始输入来自：

```text
/archive/zengjiaqi/dataset/medagentgym/biocoder/
```

当前 `medagentgym_pilot.yaml` 中配置的输入文件是：

```text
train_tasks.jsonl
test_tasks.jsonl
```

---

## 4. 当前项目的总体流程

当前 pipeline 的阶段入口集中在：

```text
src/medenvscale/pipeline_ops.py
```

当前已经定义的主要阶段如下：

1. `stage00_download`
2. `stage01_normalize`
3. `stage02_route`
4. `stage03_seed`
5. `stage04_skeleton`
6. `stage05_scale`
7. `stage06_qpoints_rubrics`
8. `stage07_safety`
9. `stage08_export`
10. `stage09_quality_filter`
11. `stage10_make_splits`
12. `stage14_eval`

这些阶段的主线可以概括成：

```text
raw BioCoder task
→ Stage 00: code validation / repair / ground truth capture
→ Stage 01: normalize
→ Stage 02: route domain + task type + solution form
→ Stage 03: seed executable env
→ Stage 04: environment skeleton
→ Stage 05: dynamic scaling
→ Stage 06: question points / rubrics
→ Stage 07: safety / stage05 reports
→ Stage 08: export training views
→ Stage 09: quality filter
→ Stage 10: split
→ Stage 14: eval summary
```

当前真正最关键、迭代最多的是两个阶段：

- **Stage 00**
- **Stage 05**

---

## 5. 当前的核心数据对象

为了理解这个项目，必须先理解几个贯穿全流程的对象。

### 5.1 Raw task

来自 BioCoder / MedAgentGym 数据集的原始任务行，通常包含：

- `task_id`
- `source_split`
- `instruction` / `problem`
- `code`
- `ground_truth`
- 其他资源字段

当前项目不要求字段名完全统一，normalizer 会做兼容映射。

---

### 5.2 Normalized task

Stage 01 之后的任务会被标准化为统一表示，核心字段包括：

- `problem`
- `context`
- `signature`
- `code`
- `wrong_code`
- `ground_truth`
- `ground_truth_output_signature`
- `resource_files`

它是后续 routing 和 seed environment 构造的输入。

---

### 5.3 ExecutableEnvSpec

当前项目中最核心的结构化对象是：

```text
src/medenvscale/schemas/environment.py
```

其中 `ExecutableEnvSpec` 是整个 Stage 03+ 的主对象。

它承载的字段非常多，关键的有：

- 基础任务字段
  - `env_id`
  - `original_task_id`
  - `split`
  - `problem`
  - `context`
  - `signature`
  - `solution_form`
  - `primary_domain`
  - `primary_task_type`

- 代码 / gold 相关字段
  - `code`
  - `gold_solution`
  - `seed_gold_solution`
  - `scaled_gold_solution`
  - `scaled_executable_gold_code`

- 输出 / oracle / verifier 相关字段
  - `scaled_oracle_cases`
  - `scaled_oracle_case_failures`
  - `scaled_oracle_coverage_summary`
  - `seed_ground_truth_output_signature`
  - `scaled_ground_truth_output_signature`
  - `output_requirements`
  - `output_constraint_spec`
  - `hidden_tests`
  - `semantic_test_specs`
  - `verifier_delta`
  - `verifier_spec`

- environment 状态字段
  - `visible_state`
  - `task_state`
  - `data_state`
  - `tool_state`
  - `gold_state`
  - `verifier_state`
  - `test_state`
  - `turn_state`

- scaling / quality 字段
  - `difficulty`
  - `tool_config`
  - `scaling_plan`
  - `operator_instances`
  - `operator_realization_report`
  - `gate_results`
  - `stage05_quality_report`
  - `quality_flags`

可以把 `ExecutableEnvSpec` 理解成：

> “一个能被扩难、能被执行、能被验证、能被导出的任务环境实例”

---

### 5.4 Output signature

当前项目的重要设计之一，是把“正确答案”从纯文本转成**可观察输出签名**。

当前 output signature 主要由三部分组成：

```json
{
  "return_value": "...",
  "stdout": "...",
  "file_artifacts": [...]
}
```

它在 Stage 00 和 Stage 05 都很关键：

- Stage 00 用它记录 seed code 的 ground truth。
- Stage 05 用它记录 scaled code 的执行结果。

---

### 5.5 Scaled oracle case

当前 Stage 05 的一个核心对象是：

```text
scaled_oracle_case
```

它是“扩展后任务的结构化验证样例”，而不是普通自然语言例子。

一个 case 的典型字段包括：

- `case_id`
- `description`
- `targets_operator_id`
- `axis`
- `semantic_intent`
- `target_constraint`
- `expected_failure_mode`
- `setup_code`
- `call_code`
- `assertion_code`
- `covers_requirements`
- `expected_output_signature`

当前项目逐渐把 `scaled_oracle_cases` 当成 Stage 05 的主要验证对象。

---

## 6. Stage 00：原始代码可执行性治理

Stage 00 的入口是：

```text
stage00_download(cfg, limit, llm_mode)
```

虽然名字叫 `download`，但当前对 BioCoder 而言，它真正做的是：

```text
读取本地 JSONL
→ 合并 train/test
→ 检查 code 是否可执行
→ 若失败则用 LLM repair
→ 保存 ground_truth_output_signature
```

### 6.1 当前读取方式

配置位于：

```text
configs/biocoder/medagentgym_pilot.yaml
```

当前设置是：

- `dataset_root = /archive/zengjiaqi/dataset/medagentgym`
- `default_dataset = biocoder`
- `task_files.train = train_tasks.jsonl`
- `task_files.test = test_tasks.jsonl`

因此 Stage 00 会从：

```text
/archive/zengjiaqi/dataset/medagentgym/biocoder/train_tasks.jsonl
/archive/zengjiaqi/dataset/medagentgym/biocoder/test_tasks.jsonl
```

读取任务。

### 6.2 当前执行逻辑

核心实现位于：

```text
src/medenvscale/ingest/code_execution.py
```

其中最重要的函数是：

- `validate_and_repair_code_rows`
- `prepare_executable_row`
- `execute_code_for_ground_truth`

执行流程如下：

1. 取出任务中的 `code`
2. 在临时目录中落盘
3. 通过 wrapper 运行 candidate code
4. 收集：
   - `stdout`
   - `stderr`
   - `return_value`
   - `file_artifacts`
5. 组装为 `ground_truth_output_signature`

### 6.3 当前 repair loop

如果 `code` 运行失败，会进入 Stage 00 的 repair loop：

- 最多 3 次 repair
- repair 成功后：
  - 新代码写回 `code`
  - 旧代码写到 `wrong_code`
- repair 失败则该行进入 reject

### 6.4 当前运行后端

当前配置里有两种后端：

- `local`
- `docker`

但当前 `medagentgym_pilot.yaml` 里设置的是：

```yaml
backend: local
local_python_bin: /home/zengjiaqi/miniconda3/envs/latentmas/bin/python
```

因此**当前默认是用本地 Python 环境执行 seed code**，而不是 Docker。

Docker image 仍然作为一个预留执行选项：

```text
medenvscale-biocoder:latest
```

### 6.5 Stage 00 输出

主要输出包括：

- `result/biocoder/00/medagentgym_tasks_raw.jsonl`
- `result/biocoder/00/prepare_rejected.jsonl`
- `result/biocoder/00/prepare_meta.json`

这一步的关键成果是：

> 后续 pipeline 不再直接依赖原始脏代码，而依赖“已执行验证 / 已修复”的 seed code。

---

## 7. Stage 01-04：标准化、路由与环境骨架

### 7.1 Stage 01 Normalize

Stage 01 负责把 Stage 00 的结果标准化成统一任务对象。

这一阶段最重要的作用是：

- 保留 Stage 00 新增字段
- 统一任务 schema
- 为 routing 做输入准备

保留的关键信息包括：

- `code`
- `wrong_code`
- `ground_truth_output_signature`

### 7.2 Stage 02 Route

Stage 02 调用 taxonomy + LLM router，为每个任务确定：

- `primary_domain`
- `secondary_domains`
- `primary_task_type`
- `secondary_task_types`
- `solution_form`
- `verifier_type_hint`

当前 routing 既支持规则，也支持 LLM。

配置中：

- `use_rule_router: true`
- `use_llm_router: true`

### 7.3 Stage 03 Seed

Stage 03 把 normalized task 和 routing result 结合，生成第一版 `ExecutableEnvSpec`。

这一阶段构造的重要字段包括：

- `env_id = seed_<task_id>`
- `problem`
- `context`
- `signature`
- `code`
- `gold_solution`
- `seed_ground_truth_output_signature`
- `scaled_ground_truth_output_signature`
- `task_state`
- `data_state`
- `visible_state`
- `verifier_state`

也就是说，到 Stage 03 之后，任务已经从“原始数据行”变成了“结构化 executable env”。

### 7.4 Stage 04 Skeleton

当前 Stage 04 逻辑很轻，基本是沿用 Stage 03 的 seed env 作为 skeleton，并将其发布为 Stage 04 的结果。

当前它更像是一个 pipeline 分段点，而不是复杂的内容生成阶段。

---

## 8. Stage 05：当前最核心的动态难度扩展阶段

Stage 05 是当前项目最复杂、最关键、问题也最多的阶段。

入口：

```text
stage05_scale(cfg, limit=None, llm_mode=None, sample_seed=None)
```

它的输入是 Stage 03 的 `ExecutableEnvSpec`，输出是一批扩展后的 M1-M4 environment。

---

### 8.1 Stage 05 的总体目标

对于每个 seed env，当前系统希望生成：

- `M1`: baseline / 原题
- `M2`: 轻度语义扩展
- `M3`: 中等强度多轴扩展
- `M4`: 高强度全轴扩展

每个 level 都会对应一个新的：

```text
env_<original_task_id>_<level>
```

例如：

```text
env_medagentgym_train_838_M3
```

---

### 8.2 Stage 05 的主子流程

对每个 seed env、每个 level，当前执行顺序大致是：

1. **Axis weight planning**
2. **Scaling plan construction**
3. **Tool config planning**
4. **Dynamic operator synthesis**
5. **Operator repair / validation**
6. **Apply operators to env**
7. **Collect semantic test specs**
8. **Normalize output requirements / output constraints**
9. **Rewrite final prompt**
10. **Generate scaled oracle cases**
11. **Generate scaled executable gold code**
12. **Repair scaled executable gold code**
13. **Build verifier / hidden tests / quality signals**
14. **Run Stage 05 gates**
15. **Split clean vs rejected**

---

### 8.3 7 个轴的定义

当前 `configs/biocoder/axis_definitions_7axis.yaml` 中定义如下：

| 轴 | 名称 | 当前含义 |
|---|---|---|
| H | Horizon / Process Complexity | 增加过程复杂度、步骤深度、中间处理链 |
| R | Biomedical / Clinical Consequence Risk | 增加风险敏感性、单位/数值合理性、结果后果 |
| I | Information Ambiguity | 增加信息歧义、解释负担、误解风险 |
| E | Evidence / Data Structure Complexity | 增加输入载体、路径、文件、数据结构复杂度 |
| C | Computation / Constraint Complexity | 增加边界条件、组合约束、异常路径 |
| A | Adversarial / Robustness Challenge | 增加防 shortcut、防 hardcode、鲁棒性挑战 |
| V | Verifier / Test Complexity | 增加验证复杂度、测试数量、检查覆盖 |

这些轴的最大强度目前都设为 `3`。

---

### 8.4 M1-M4 的预算控制

当前 `configs/biocoder/m_level_budgets_7axis.yaml` 对不同难度层级规定了预算边界。

概念上：

- `M1` = 不扩展
- `M2` = 2-3 个轴，小总强度
- `M3` = 3-5 个轴，中等强度，可允许对抗 / 多步
- `M4` = 7 个轴全开，高总强度，要求 safety gate

除了 axis budget 外，还定义了 tool budget bounds，比如：

- allowed tool count range
- max total tool calls
- max calls per tool
- allow debug tool
- allow external resource tools

注意：当前你和项目的最新理解里，`tool_budget` 更多是 planner / config 约束的一部分，不是最终验证的核心对象。

---

### 8.5 Tool pool

当前工具池配置位于：

```text
configs/biocoder/tool_pool.yaml
```

它区分：

- `agent_tools`
- `evaluator_tools`

其中：

- `agent_tools` 会暴露给 agent / planner
- `evaluator_tools` 是评测内部工具，不给 agent 用

当前 agent_tools 包括：

- `get_task_brief`
- `get_context`
- `search_context`
- `get_signature_info`
- `list_context_imports`
- `assemble_candidate_code`
- `check_syntax`
- `check_target_signature`
- `dependency_probe`
- `run_assembled_code`
- `run_custom_test_snippet`
- `debug_traceback`

当前 evaluator_tools 包括：

- `generate_hidden_tests`
- `run_hidden_tests`
- `score_process_trace`
- `check_tool_leakage`

---

### 8.6 Dynamic operator synthesis

Stage 05 会先构造：

- `axis_weights`
- `scaling_plan`
- `tool_config`

然后调用 LLM 生成 `operator_instances`。

当前相关 prompt 包括：

- `axis_weight_planner_7axis.jinja`
- `tool_config_planner.jinja`
- `dynamic_verifiable_operator_planner.jinja`
- `prompt_rewriter.jinja`

生成出的 operator 会经过：

- `repair_operator_instances`
- `validate_dynamic_operator_instances`
- `validate_verifier_delta`

之后再应用到 environment 上。

---

### 8.7 Operator 作用到哪些状态字段

当前 operator 的语义改动不是只改 metadata，而是理论上会改这些 patch：

- `visible_state_patch`
- `task_state_patch`
- `gold_state_patch`
- `data_state_patch`
- `test_state_patch`
- `verifier_state_patch`

这些 patch 会在 `apply_operator_instances(...)` 后写回到 environment。

这也是为什么当前项目一直强调：

> “operator 必须真正改变任务语义，而不是只改 intensity / tool budget / metadata”

---

### 8.8 Semantic test specs

当前 Stage 05 还会从 operator 中收集：

```text
semantic_test_specs
```

这是一层“语义测试意图”的中间对象，主要描述：

- 要验证哪类新增语义
- 目标 operator 是谁
- 预期弱解会怎么错

它最初更多是为了 hidden test materialization 服务。

在当前项目最新演进中，它仍然存在，但地位正在弱化，逐渐让位给 `scaled_oracle_cases`。

---

### 8.9 Output requirements 与 output constraint spec

当前系统还会从 operator / patch 中提取：

- `output_requirements`
- `output_constraint_spec`

`output_constraint_spec` 本质上是一份“结构化输出检查规则表”，例如：

- 返回值类型必须是 tuple
- stdout 必须包含某个 token
- stdout 必须匹配某个 regex
- 必须产生某个文件产物

它更像是：

> verifier-style global constraint table

而不是具体案例本身。

在当前项目里，它的定位逐渐从“主验证对象”弱化为“全局兜底 / 补充约束层”。

---

## 9. 当前 Stage 05 中最关键的新对象：scaled_oracle_cases

当前项目最新的 Stage 05 设计里，`scaled_oracle_cases` 是一个非常核心的对象。

当前 prompt 位于：

```text
prompts/scaled_oracle_examples_generate.jinja
```

虽然文件名里还有 `examples`，但它的真实语义已经变成：

```text
generate scaled_oracle_cases
```

### 9.1 一个 case 的主要字段

当前 case schema 包括：

- `case_id`
- `description`
- `targets_operator_id`
- `axis`
- `semantic_intent`
- `target_constraint`
- `expected_failure_mode`
- `setup_code`
- `call_code`
- `assertion_code`
- `covers_requirements`
- `expected_output_signature`

### 9.2 setup_code / call_code 的含义

- `setup_code`
  - 用来准备环境、输入文件、变量等
- `call_code`
  - 真正调用目标代码

当前项目已经逐渐把一个 case 看成：

```text
可执行小场景
```

而不是普通自然语言样例。

### 9.3 expected_output_signature

这是每个 case 的“结构化预期输出”。

当前可表达的信息包括：

- `return_type`
- `return_keys`
- `return_value`
- `stdout_contains`
- `stdout_regex`
- `file_artifacts`

也就是说，case 不只是“给输入”，而是给：

> 输入场景 + 期望可观察结果

---

## 10. 当前 scaled executable gold code 生成与 repair

Stage 05 当前不是只让 LLM 生成一个 solution snippet，而是要生成：

```text
scaled_executable_gold_code
```

也就是：

> 一整段可执行 Python 代码

### 10.1 当前生成 prompt

当前相关 prompt 是：

- `prompts/scaled_gold_generate.jinja`
- `prompts/scaled_gold_repair.jinja`

LLM 会拿到：

- seed problem
- seed executable code
- scaled final user prompt
- signature
- operator instances
- semantic test specs
- scaled_oracle_cases
- compiled hidden tests

然后输出：

- `scaled_executable_gold_code`
- `scaled_oracle_cases`（可被同步修正）
- `covered_operator_ids`
- `covered_requirements`
- 其他 gold metadata

### 10.2 当前 repair loop

如果候选 scaled gold code 不通过，会进入 repair loop。

当前 repair loop 最多 3 次。

它会把以下信息回传给 LLM：

- compile result
- execution result
- hidden test result
- failure summary
- existing scaled_oracle_cases
- compiled hidden tests
- observed output signature

也就是说，当前 repair 已经不是盲修，而是基于执行反馈修。

---

## 11. 当前 output signature 与执行结果采集

当前执行候选代码的核心逻辑在：

```text
src/medenvscale/scaling/output_signature.py
```

最关键函数包括：

- `materialize_executable_gold_code`
- `execute_materialized_code`
- `execute_candidate_solution`

当前执行后会收集：

```json
{
  "return_value": ...,
  "stdout": ...,
  "file_artifacts": ...
}
```

这就是当前 `scaled_ground_truth_output_signature` 的来源。

需要注意的一点是：

当前 `_normalize_runtime_value(...)` 会把 tuple 规范化为 list，这在类型严格校验时可能带来误差。这是当前设计里一个值得注意的细节。

---

## 12. 当前 hidden_tests 的地位

当前项目的一个关键演化点，是 `hidden_tests` 的地位在变化。

### 12.1 当前实现状态

当前代码里仍然保留：

- `hidden_tests`
- `run_hidden_test_execution_check`
- `hidden_tests_quality_gate`

并且 `scaled_oracle_cases` 目前还会被编译成 hidden test 代码：

```text
scaled_oracle_case
→ compile to hidden test code
→ run hidden test execution check
```

### 12.2 当前问题

你当前项目里已经暴露出一个很重要的问题：

> `scaled_oracle_case -> hidden_test` 这层编译容易自带契约 bug，导致失败并不一定来自代码本身，而来自测试脚本自身。

例如：

- `call_code` 没有定义 `result`
- 但 hidden test 自动断言 `result`
- 于是报 `NameError`

这种错误会污染 Stage 05 的验证链。

### 12.3 当前演进方向

因此，你当前最新的思路已经转向：

```text
LLM 生成 scaled_oracle_cases
→ 系统直接执行 case
→ 系统比较 observed output vs expected_output_signature
→ 失败信息反馈给 repair loop
```

也就是说，`hidden_tests` 将逐渐从“主验证对象”退化为“兼容导出 / 可选中间格式”。

---

## 13. 当前 Stage 05 的 gate 体系

当前 Stage 05 的 gate runner 位于：

```text
src/medenvscale/validation/stage05_gate_runner.py
```

对于非 M1 样本，当前会跑 3 个 gate：

1. `hidden_tests_quality_gate`
2. `scaled_task_consistency_gate`
3. `pipeline_artifact_admission_gate`

### 13.1 hidden_tests_quality_gate

检查重点包括：

- hidden tests 是否可编译 / 可执行
- 是否是弱测试
- 是否有 semantic intent
- 数量是否满足要求
- 是否 targeting 到 operator / requirement

### 13.2 scaled_task_consistency_gate

这是当前很核心的 gate。

它主要检查：

- 是否提取出了新增 requirement
- scaled gold 是否与扩展后的 prompt / verifier 对齐
- requirement 是否被测试覆盖
- verifier state 是否知道这些新增 requirement
- prompt / gold / test / verifier 是否闭环

它的典型 hard fail 包括：

- `SCALED_GOLD_DOES_NOT_MATCH_VERIFIER_SPECS`
- `REQUIREMENT_CHAIN_BROKEN`
- `PROMPT_GOLD_TEST_VERIFIER_MISMATCH`
- `NO_TEST_COVERS_NEW_REQUIREMENT`

### 13.3 pipeline_artifact_admission_gate

这个 gate 更偏“结果文件是否能被后续阶段消费”，例如：

- tool config 是否可接受
- 结果字段是否完整
- operator realization 是否通过
- clean / rejected 分流是否正确
- 测试运行器是否能加载

---

## 14. 当前 Stage 05 的 clean / rejected 逻辑

Stage 05 最终会把样本拆成：

- `scaled_envs_clean.jsonl`
- `scaled_envs_rejected.jsonl`

同时还会产出很多伴随文件，例如：

- `scaling_plans.jsonl`
- `tool_configs.jsonl`
- `operator_instances.jsonl`
- `verifier_specs.jsonl`
- `hidden_tests.jsonl`
- `hidden_tests_clean.jsonl`
- `quality_report.jsonl`
- `operator_realization_report.jsonl`
- `hidden_tests_quality_report.jsonl`
- `scaled_task_consistency_report.jsonl`
- `artifact_admission_report.jsonl`
- `stage05_quality_report.jsonl`

这说明当前 Stage 05 不只是“生成环境”，还是“生成一整套质量审计产物”。

---

## 15. Stage 06-10：下游后训练导出

虽然当前你的主要精力在 Stage 00 和 Stage 05，但后面的导出阶段也已经在代码里连通了。

### Stage 06

根据 environment 生成：

- question points
- rubrics
- sampled trajectories

### Stage 07

收拢 safety / quality report。

### Stage 08

导出训练视图：

- `sft.jsonl`
- `dpo.jsonl`
- `preference.jsonl`
- `prm.jsonl`
- `prm_steps.jsonl`
- `rlvr_envs.jsonl`

### Stage 09

质量过滤。

### Stage 10

生成 train/dev/test split。

### Stage 14

产出简单 eval summary，比如：

- environment 数量
- DPO pair 数量
- 各 level 分布
- domain 分布
- 每个 env 平均 hidden test 数量

---

## 16. 当前项目的 prompt 体系

当前 `prompts/` 目录下已经存在较完整的 prompt 体系。

重要 prompt 包括：

- `route_medagentgym_task.jinja`
- `axis_weight_planner_7axis.jinja`
- `tool_config_planner.jinja`
- `dynamic_verifiable_operator_planner.jinja`
- `prompt_rewriter.jinja`
- `scaled_oracle_examples_generate.jinja`
- `scaled_gold_generate.jinja`
- `scaled_gold_repair.jinja`
- `qpoint_extract.jinja`
- `rubric_generate.jinja`
- `stage00_code_repair.jinja`

从当前项目设计看，LLM 的主要职责包括：

1. Stage 00 repair code
2. Route / classify task
3. Plan axis weights
4. Plan tool config
5. Synthesize operators
6. Rewrite prompt
7. Generate scaled oracle cases
8. Generate scaled executable gold code
9. Repair scaled executable gold code
10. Generate qpoints / rubrics 等下游对象

因此这个项目本质上是一个：

> 多 prompt、多中间对象、LLM-assisted benchmark construction pipeline

---

## 17. 当前系统的主要设计原则

根据当前实现和你最近的实验决策，系统已经形成了几条比较明确的原则。

### 17.1 Trust raw gold policy

当前全项目采用：

```text
trust raw gold
```

也就是：

> 原始数据集中的 gold solution / reference code 被视为可信，不再做 gold compatibility gate。

这意味着：

- `gold_compatibility_results.jsonl` 不再属于主流程
- 关注点转移到：
  - seed code 是否可执行
  - scaled code 是否满足 scaled task
  - 测试 / oracle case 是否能验证新增语义

### 17.2 Dataset-specific config policy

当前项目已经明显从“共享全局配置”转向“按数据集隔离配置”。

BioCoder 相关配置尽量放在：

```text
configs/biocoder/
```

### 17.3 Executable-first policy

当前项目强调：

> 如果一个任务不能执行、不能产生可观察输出、不能进入验证闭环，就不应进入后续高质量样本集合。

### 17.4 Structured artifact policy

当前项目不是只输出一个最终数据文件，而是输出：

- 中间环境对象
- 工具配置
- operator 配置
- verifier 结果
- quality gate 报告

这使得后续可以定位某一层出错，而不是只看到一个最终失败。

---

## 18. 当前项目的主要已知问题

截至目前，项目已经有比较清晰的几个痛点。

### 18.1 Stage 05 的主要瓶颈不再是“能不能生成”

现在更大的问题是：

- 生成出来的 operator 是否真的改变了语义
- 生成出来的 scaled_oracle_cases 是否真的正确
- 生成出来的 scaled_executable_gold_code 是否真的满足扩展后任务
- 测试闭环是否真的成立

### 18.2 hidden_tests 容易成为错误来源

目前 `scaled_oracle_case -> hidden_test` 这层编译仍然可能引入：

- 变量名契约不一致
- stdout capture 重复包裹
- 自动断言对象错误

这会导致“测试脚本错”掩盖“代码本身错”。

### 18.3 case correctness 仍需更强闭环

当前你已经在收敛到一个更干净的方向：

```text
case 必须通过：
schema 正确
+ case 可执行
+ scaled code 能通过
+ case 确实覆盖新增 requirement
```

这是后续很重要的一步。

### 18.4 output_constraint_spec 与 scaled_oracle_cases 还存在部分重叠

当前两者的职责正在重构：

- `scaled_oracle_cases` 更适合当主验证对象
- `output_constraint_spec` 更适合保留为全局 contract / 兜底层

但这套边界还在继续清晰化。

---

## 19. 当前项目的下一步自然演进方向

根据当前实现和最近的实验讨论，一个合理的演进方向是：

### 19.1 用 scaled_oracle_cases 取代 hidden_tests 的主链路地位

建议主验证链路改成：

```text
LLM 生成 scaled_oracle_cases
→ 系统执行 setup_code + call_code
→ 系统收集 observed_output_signature
→ 直接与 expected_output_signature 对比
→ 失败信息反馈给 repair loop
```

这样可以减少中间 test-code 编译噪音。

### 19.2 把 case correctness 变成显式 validator

一个 case 只有在同时满足以下条件时才算有效：

- schema 合法
- 可执行
- gold code 通过
- 覆盖新增 requirement

### 19.3 把 output_constraint_spec 收缩为辅助层

让它只负责：

- 跨 case 的全局输出 contract
- 无法自然放进 case 的补充规则
- verifier-style fallback

### 19.4 让 repair loop 直接消费 case-level failure

不要只传 `AssertionError` 文本，而是传：

- 哪个 case 失败
- 哪个输出字段不匹配
- 是 return_value / stdout / artifact 哪一层错

这样 LLM 更容易修。

---

## 20. 如何向外部 GPT 简述这个项目

如果要给网页 GPT 一个简洁但正确的介绍，可以用下面这段：

> 这是一个把 BioCoder / MedAgentGym 可执行代码任务加工成动态难度 benchmark 和后训练数据的流水线。系统先在 Stage 00 检查原始任务代码是否可运行，并在必要时用 LLM 自动 repair，然后把任务标准化并构造成 executable environment。接着在 Stage 05 通过 7 个 difficulty axes 对环境进行 M1-M4 难度扩展，生成 operator、tool config、scaled oracle cases 和 scaled executable gold code，并通过 verifier / quality gate 审查样本质量。最后系统导出 question points、rubrics、SFT / DPO / PRM / RLVR 等训练视图。当前最大的研究重点是：如何让扩展后的任务语义真实变化、验证样例真实正确、以及 scaled code 与扩展后任务形成闭环。 

---

## 21. 当前项目的最简总结

把这套系统再压缩成一句话，就是：

> 先把原始 BioCoder 代码题清洗成可执行 seed environment，再基于 7 轴动态 operator 对其进行可验证的难度扩展，并把这些扩展后的环境实例导出为可供后训练和评测使用的数据。

---

## 22. 备注

本说明文档是基于当前 `/home/zengjiaqi/medenvscale` 目录下的真实代码、配置、prompt、阶段脚本和结果组织方式整理而成。

额外说明：

- 当前目录不是 git worktree，因此这里没有纳入完整 git 历史演化分析。
- 文档重点描述“当前实现与当前实验意图”，而不是历史上所有旧方案。
- Stage 05 仍处于快速迭代阶段，因此其中关于 `hidden_tests`、`scaled_oracle_cases` 和 `output_constraint_spec` 的关系，属于“当前实现 + 正在收敛的设计方向”。
