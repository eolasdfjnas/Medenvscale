# MedEnvScale-Train-MedAgentGym：融合实验方案总览

## 1. 一句话概括

这个融合方案的目标是：

```text
把 MedAgentGym 的可执行医学 agent 任务，扩展成带 M-level 难度控制、动态 operator、rubric、verifier、trajectory 和多种后训练视图的数据生成框架。
```

最终产物不只是 benchmark，也可以作为 SFT / DPO / PRM / RLVR 的数据引擎。

---

## 2. 两个方案如何融合

原来的 MedAgentGym 方案负责整体训练/评测框架：

```text
MedAgentGym task
→ task normalization
→ executable env
→ trajectory sampling
→ qpoints
→ rubrics
→ verifier/reward
→ SFT / DPO / PRM / RLVR views
```

动态 operator 方案负责难度生成核心：

```text
LLM axis weight planning
→ probability-weighted axis sampling
→ axis_intensity allocation
→ DynamicOperatorPlanner
→ OperatorValidator
→ OperatorApplier
→ PromptRewriter
```

融合后形成完整 pipeline：

```text
MedAgentGym seed task
→ Access filtering
→ Task normalization
→ Capability routing
→ Base executable environment
→ Base difficulty inference
→ LLM AxisWeightPlanner
→ M-level budget control
→ Selected axes sampling
→ Axis intensity allocation
→ Dynamic operator generation
→ Operator validation / repair / fallback
→ Structured state patch application
→ Prompt/task rewriting
→ Verifier synchronization
→ Trajectory sampling
→ QPoint extraction
→ Rubric generation
→ Reward construction
→ Training view export
```

---

## 3. 核心设计思想

### 3.1 M-level 只控制额外难度

M1/M2/M3/M4 控制的是 scaling difficulty，不直接等于最终难度。

```text
base_difficulty = seed task 自带难度
axis_intensity = 额外施加的难度
final_difficulty = min(3, base_difficulty + axis_intensity)
```

这样可以避免 M1 被错误加 operator，也避免把原始任务的天然复杂度和扩展难度混在一起。

---

### 3.2 LLM 只决定“哪些轴重要”

LLM AxisWeightPlanner 只输出：

```text
axis_weight_hint: H/R/I/E/C/A 每个轴 1-6 权重
axis_weight_reason: 简短原因
```

LLM 不直接决定：

```text
selected_axes
axis_intensity
operator_instances
```

这些由程序根据 M-level budget、primary_task_type hard constraints、概率抽样和 validator 决定。

---

### 3.3 primary_task_type 主导，secondary_task_types 轻量修正

每条任务先通过 capability routing 得到：

```text
primary_task_type
secondary_task_types
primary_domain
secondary_domains
clinical_topic
medical_concepts
```

primary_task_type 决定主轴优先级，例如：

```text
evidence_interpretation → E, I 优先
medication_safety → C, R 优先
multi_step_agent_planning → H, E 优先
database_querying → E, C 优先
```

secondary_task_types 最多 3 个，只能通过 average boost 轻量影响 final_axis_weights，不能覆盖 primary hard constraints。

---

### 3.4 Dynamic operator 修改 state，不直接拼 prompt

operator 只能修改结构化环境状态：

```text
task_state
data_state
tool_state
visible_state
gold_state
verifier_state
turn_state
```

operator 不允许直接写最终 user_prompt。

最后由 PromptRewriter 统一生成：

```text
system_prompt
user_prompt
prompt_format
```

这样可以保证难度变化可追踪、可验证、可回滚，也便于生成 trajectory 和 verifier。

---

### 3.5 MedAgentGym 版本必须同步 verifier

由于 MedAgentGym 是 executable task，operator 不能只改题面。

任何改变任务条件的 operator 都必须同步更新：

```text
gold_state
verifier_state
reward rules
safety gate
```

例如：

```text
E 轴增加一个数据表 → verifier 必须检查是否使用该数据表或结果是否正确。
C 轴增加 JSON 输出格式 → verifier 必须检查 JSON schema。
A 轴增加误导性 shortcut → verifier 必须拒绝 shortcut-only answer。
R 轴增加安全风险 → safety gate 必须拒绝 unsafe answer。
```

---

## 4. 六个难度轴

融合后仍固定使用 H/R/I/E/C/A 六轴。

```text
H = Clinical / Execution Horizon
多步执行、action-observation loop、时间推进、动态状态变化。

R = Risk Acuity / Safety Criticality
临床风险、安全关键判断、红旗症状、错误答案的危险性。

I = Information Completeness
信息缺失、字段模糊、资源说明不完整、需要主动检查或声明不确定性。

E = Evidence / Data Complexity
多数据源、多表、多证据、多文件、实验室/报告/文献/数据库整合。

C = Constraint / Execution Complexity
患者约束、输出格式约束、工具限制、运行时限制、资源访问限制。

A = Adversarial / Robustness Surface
误导线索、错误捷径、用户压力、plausible-but-wrong cue、reward hacking 表面答案。
```

每个轴 intensity 范围是 0-3：

```text
0 = 不激活
1 = 轻度加难
2 = 中度加难
3 = 强加难
```

---

## 5. M-level 设计

```text
M1：base environment，不加 operator。
M2：轻度多轴扩展，激活 2-3 个轴，总强度 2-4。
M3：中高难度复合扩展，激活 3-5 个轴，总强度 5-8。
M4：最高难度，六轴全开，总强度 9-14。
```

关键 hard constraints：

```text
M2 必须包含 primary_task_type 的 top-1 axis。
M3 必须包含 primary_task_type 的 top-2 axes。
M4 必须包含 H/R/I/E/C/A 全部六个轴。
```

---

## 6. 主要模块

### 6.1 数据与任务模块

```text
MedAgentGymLoader
AccessFilter
TaskNormalizer
TaskCapabilityRouter
SeedAgentTaskBuilder
ExecutableEnvSpecBuilder
```

作用：

```text
读取 MedAgentGym 原始任务，过滤不可访问资源，统一成可执行 agent environment。
```

---

### 6.2 难度规划模块

```text
AxisWeightPlanner
ScalingPlanBuilder
ScalingValidator
```

作用：

```text
先由 LLM 判断轴重要性，再由程序根据 M-level 预算抽轴和分配强度。
```

核心输出：

```text
selected_axes
axis_intensity
total_intensity
final_axis_weights
```

---

### 6.3 Dynamic operator 模块

```text
DynamicOperatorPlanner
OperatorValidator
OperatorApplier
```

作用：

```text
把抽到的轴和强度落实到具体环境修改。
```

operator 修改的是 structured state，而不是 prompt 文本。

---

### 6.4 Prompt / Task Rewriter

```text
PromptRewriter
```

作用：

```text
根据 visible_state、task_state、data_state、tool_state、turn_state 生成最终 executable user_prompt。
```

它是唯一能生成最终 prompt 的模块。

---

### 6.5 Trajectory / Rubric / Verifier 模块

```text
TrajectorySampler
QuestionPointExtractor
RubricGenerator
VerifierBuilder
RewardBuilder
SafetyExecutionGate
```

作用：

```text
采样 agent 轨迹，提炼考点，生成 rubric，构造 executable verifier 和 reward。
```

---

### 6.6 Export 模块

```text
SFTExporter
DPOExporter
PRMExporter
RLVRExporter
```

作用：

```text
把同一批 scaled environments 转换成不同后训练格式。
```

---

## 7. 最终数据流

```text
Raw MedAgentGym task
  ↓
Access filtering
  ↓
NormalizedAgentTask
  ↓
RoutingResult
  ↓
SeedAgentTask + ExecutableEnvSpec
  ↓
Base difficulty
  ↓
ScalingPlan
  ↓
OperatorInstances
  ↓
ScaledEnvironmentState
  ↓
PromptRewriteResult
  ↓
ExecutableVerifier + Reward
  ↓
AgentTrajectory
  ↓
QuestionPoints + Rubrics
  ↓
SFT / DPO / PRM / RLVR exports
```

---

## 8. 最终每条 scaled environment 应包含

```json
{
  "env_id": "...",
  "original_task_id": "...",
  "system_prompt": "...",
  "user_prompt": "...",
  "resource_manifest": [],
  "base_difficulty": {
    "H": 0,
    "R": 0,
    "I": 1,
    "E": 1,
    "C": 0,
    "A": 0
  },
  "scaling": {
    "global_level": "M3",
    "axis_weight_source": "llm",
    "axis_weight_hint": {
      "H": 2,
      "R": 1,
      "I": 5,
      "E": 6,
      "C": 3,
      "A": 1
    },
    "axis_priority": ["E", "I", "C", "H", "A", "R"],
    "final_axis_weights": {},
    "selected_axes": ["E", "I", "C", "H"],
    "axis_intensity": {
      "H": 1,
      "R": 0,
      "I": 2,
      "E": 3,
      "C": 1,
      "A": 0
    },
    "total_intensity": 7,
    "sampling_seed": 12345,
    "allow_multiturn": true,
    "allow_adversarial": false,
    "require_safety_gate": false,
    "operator_instances": []
  },
  "difficulty": {
    "H": 1,
    "R": 0,
    "I": 3,
    "E": 3,
    "C": 1,
    "A": 0
  },
  "verifier_state": {},
  "safety_gate_required": false
}
```

---

## 9. MVP 实验建议

第一阶段只做小规模验证：

```text
seed tasks: 50-100
M-level: M1-M4
任务类型：text / csv / db / json / python computation
模型：先用一个强模型采 teacher trajectory，再用一个弱模型采 failure trajectory
```

先验证三件事：

```text
1. M-level 是否真的让任务更难。
2. operator 是否能稳定生成且不破坏 ground truth / verifier。
3. 生成的 rubric / verifier 是否能区分好坏轨迹。
```

成功后再扩展到：

```text
500 seed tasks
2,000 scaled envs
SFT / DPO / PRM / RLVR 多视图导出
```

---

## 10. 推荐实验指标

难度有效性：

```text
pass@1 by M-level
verifier score by M-level
rubric score by M-level
平均 step 数
tool error rate
unsafe answer rate
```

数据质量：

```text
operator validation pass rate
prompt leakage rate
verifier execution pass rate
rubric observability pass rate
trajectory usable rate
```

训练效果：

```text
SFT held-out pass@1
DPO win rate
PRM step label accuracy
best-of-N reranking gain
RLVR reward improvement
```

---

## 11. 最重要的实现原则

```text
1. MedAgentGym 是 executable task，所以所有 difficulty scaling 都必须保持任务可执行。
2. M1 不加 operator。
3. M4 六轴全开。
4. LLM 只给 axis weights，不直接决定抽哪些轴和加多强。
5. primary_task_type 主导轴优先级。
6. secondary_task_types 只做轻量 boost。
7. axis_intensity 不等于 operator 数量。
8. 每个轴 operator_intensity 之和必须等于 axis_intensity。
9. operator 只改 structured state，不直接写 prompt。
10. PromptRewriter 是唯一最终生成 user_prompt 的模块。
11. operator 如果改变任务条件，必须同步 verifier。
12. 不允许泄露 ground_truth。
13. 不允许改变 original_task_id、answer、core medical concept。
14. validator 失败要 repair；repair 失败用 fallback；fallback 失败标记 needs_review 或 filtered。
```

---

## 12. 最终产出

项目最终会产出四类文件：

```text
1. scaled_envs.jsonl
   每条是一个 M-level scaled executable environment。

2. trajectories.jsonl
   每条是一个 agent action-observation trajectory。

3. rubrics.jsonl / qpoints.jsonl
   每个 environment 的考点与 rubric。

4. train_views/
   sft.jsonl
   dpo.jsonl
   prm.jsonl
   rlvr_envs.jsonl
```

这套设计的核心价值在于：

```text
同一批 MedAgentGym seed tasks 可以被系统扩展成不同难度、不同训练用途、可验证、可追踪的 medical agent 数据。
```
