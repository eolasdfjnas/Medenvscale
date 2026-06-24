# MedEnvScale-Train-MedAgentGym：7 轴动态可验证 Operator + LLM 工具配置版 Codex 实验方案

> 版本：v5-7Axis-DynamicVerifiableOperator-LLMSecondaryWeights-ToolBudget  
> 数据基础：用户提供的 MedAgentGym `train_tasks.jsonl` / `test_tasks.jsonl` 子集  
> 适用任务：biomedical / scientific code completion tasks  
> 核心变化：取消固定 `Operator Candidate Pool / Operator Template Registry`，改为**针对每条样本动态合成可验证 Operator**。  
> 核心原则：**轴固定，协议固定，operator 内容动态；domain 不决定轴，primary/secondary task type 的轴权重由 LLM 逐样本判断；工具集合、工具 schema、工具调用 budget 也由 LLM 在 M-level 约束内逐样本规划，程序负责校验、修复、fallback、概率抽轴和强度分配。**

---

## 0. 方案定位

本方案面向当前 MedAgentGym 子集。该子集主要是 biomedical / scientific Python code completion 任务，每条样本通常包含：

```text
idx
problem
solution
context
signature
code
```

字段含义：

```text
problem   = 自然语言任务描述
context   = 可执行代码上下文，通常包含 <<insert solution here>> 占位符
signature = 目标函数签名
solution  = gold code / reference solution
code      = 完整代码或辅助代码字段
```

本方案目标是构建如下 pipeline：

```text
raw MedAgentGym task
→ task normalization
→ domain / task_type / solution_form routing
→ task_type-driven axis priority
→ LLM AxisWeightPlanner for primary + secondary task types
→ primary/secondary axis weight fusion
→ probability-weighted selected_axes sampling
→ axis_intensity allocation
→ LLM ToolConfigPlanner
→ tool budget / tool schema / tool policy validation
→ Dynamic Verifiable Operator Synthesis
→ GenericOperatorValidator
→ VerifierDeltaValidator
→ GoldCompatibilityRunner / GoldRegenerationRunner
→ OperatorApplier
→ PromptRewriter
→ Executable verifier / hidden tests
→ trajectory sampling
→ qpoints / rubrics
→ reward construction
→ SFT / DPO / PRM / RLVR training views
```

本版和上一版的关键区别：

```text
上一版：
Operator Candidate Pool / Template Registry
→ LLM 从候选模板中选择并实例化 operator

本版：
不固定 operator 候选池
→ LLM 针对每条样本动态合成 sample-specific OperatorInstance
→ 每个 operator 必须自带 state patch + verifier_delta + hidden tests / checks
→ 程序只验证它是否满足可验证协议
```

因此，本方案中固定的东西只有：

```text
1. domain taxonomy
2. task_type taxonomy
3. solution_form taxonomy
4. 七个难度轴 H/R/I/E/C/A/V
5. M-level budget
6. state schema
7. verifier/test schema
8. operator verification contract
9. generic validator
10. tool schema / budget validation
11. gold compatibility / gold regeneration execution check
```

不固定的东西是：

```text
1. operator_type 名称
2. operator 的具体语义
3. 新增什么边界条件
4. 新增什么 hidden tests
5. 新增什么 verifier checks
6. 新增什么 rubric criteria
7. 如何针对当前样本使任务变难
```

一句话：

```text
Operator 不再是预定义转换函数，而是带有可验证协议的 sample-specific task transformation instance。
```

---

## 1. MVP 目标

### 1.1 输入

```text
train_tasks.jsonl
test_tasks.jsonl
optional resources: csv / json / txt / py / db / pkl / image / fasta / pdb 等
```

MVP 阶段先支持当前文件中的 code completion 任务：

```text
1. function_definition
2. function_body
3. expression_completion
4. statement_block_completion
5. decorated_function_definition
6. patch_or_bugfix
```

暂缓支持：

```text
1. 需要 DUA 的 EHR 原始数据
2. 大型医学影像原始数据
3. 私有 API 或联网依赖任务
4. 无法构造 verifier 的开放式解释任务
```

### 1.2 输出

```text
normalized_tasks.jsonl
routed_tasks.jsonl
seed_envs.jsonl
scaling_plans.jsonl
tool_configs.jsonl
dynamic_operator_instances.jsonl
scaled_envs_M1_M4.jsonl
verifier_specs.jsonl
hidden_tests.jsonl
rubrics.jsonl
sampled_trajectories.jsonl
sft.jsonl
dpo.jsonl
prm.jsonl
rlvr_envs.jsonl
quality_report.jsonl
```

### 1.3 MVP 数量建议

```text
seed tasks: 500
M-level variants: M1 / M2 / M3 / M4
scaled envs: 2,000
successful trajectories: 2,000+
failed / contrastive trajectories: 2,000+
DPO pairs: 4,000+
PRM step labels: 10,000+
rubric criteria: 12,000+
RLVR env specs: 2,000
```

---

## 2. 简化版 Domain Taxonomy

Domain 表示任务所属的生物医学 / 科研场景。**Domain 不参与轴选择，不决定 axis_priority、selected_axes 或 axis_intensity。**

### 2.1 Domain 列表

```yaml
domains:
  scientific_software_engineering:
    description: "科研软件工程、文件处理、报告生成、CLI、配置、通用 Python utility。"
    examples:
      - "LaTeX / Jinja2 report generation"
      - "file path parsing"
      - "configuration serialization"
      - "CLI / subprocess helper"

  bioinformatics_sequence_structure:
    description: "生物序列、基因组数据、FASTA/FASTQ/BAM/VCF、PDB/蛋白结构。"
    examples:
      - "FASTA parsing"
      - "genomic interval processing"
      - "motif counting"
      - "PDB chain / residue extraction"

  biomedical_data_analysis:
    description: "NumPy/Pandas/统计/矩阵/表格/生物医学图像与 mask 处理。"
    examples:
      - "DataFrame transformation"
      - "array / matrix computation"
      - "segmentation mask processing"
      - "feature extraction"

  systems_molecular_modeling:
    description: "系统生物学、代谢网络、reaction/compound/flux、分子模拟、free energy。"
    examples:
      - "metabolic reaction analysis"
      - "stoichiometry / flux processing"
      - "BAR / MBAR free energy computation"
      - "thermodynamic state processing"

  omics_measurement_analysis:
    description: "蛋白质组学、代谢组学、脂质组学、FDR、PSM、m/z、LC-MS、retention time。"
    examples:
      - "FDR calculation"
      - "peptide-spectrum match processing"
      - "retention time correction"
      - "m/z range query"
```

### 2.2 Domain 的作用

Domain 只用于：

```text
1. 理解领域术语和对象类型
2. 帮助 LLM 在动态 operator 中生成领域合理的 edge cases
3. 帮助 VerifierBuilder 解释数据对象类型
4. 帮助 RubricGenerator 生成领域相关 criterion
5. 数据采样分层
6. 实验结果分 domain 报告
7. out-of-domain generalization split
```

Domain 不用于：

```text
1. axis_priority
2. selected_axes
3. axis_intensity
4. M-level budget
5. 强制激活 R / V / A 等任何轴
6. 固定 operator 模板选择
```

核心规则：

```text
Domain = 语义上下文 / 报告维度
Task type = 轴规划主依据
Solution form = prompt / verifier / execution 插入方式依据
```

---

## 3. 简化版 Task Type Taxonomy

Task type 表示模型需要完成的代码操作类型。它是 axis planning 的主依据。

```yaml
task_types:
  file_io_and_formatting:
    description: "文件读写、格式解析、格式转换、报告生成、CLI/API/database 调用。"
    typical_verifier:
      - runtime_output_match
      - file_output_check
      - regex_check
      - unit_test

  sequence_and_structure_processing:
    description: "生物序列、基因组区间、FASTA/FASTQ/BAM/VCF、PDB/结构对象处理。"
    typical_verifier:
      - unit_test
      - exact_match
      - set_match

  numerical_and_statistical_computation:
    description: "数值计算、矩阵运算、统计分析、科学公式、free energy、FDR 等。"
    typical_verifier:
      - numeric_tolerance
      - array_close
      - unit_test

  tabular_data_transformation:
    description: "Pandas/DataFrame/table 的列处理、筛选、groupby、merge、聚合、缺失值处理。"
    typical_verifier:
      - dataframe_equal
      - column_check
      - runtime_output_match

  domain_model_or_image_analysis:
    description: "处理特定生物医学对象或模型，例如 metabolic model、reaction network、image mask、segmentation label。"
    typical_verifier:
      - object_state_check
      - array_mask_check
      - unit_test

  validation_and_code_utility:
    description: "参数验证、异常处理、class/object utility、decorator、property、通用函数逻辑。"
    typical_verifier:
      - exception_unit_test
      - unit_test
      - static_check
```

Task type 决定：

```text
1. task_axis_priority
2. base_difficulty prior
3. verifier default type
4. qpoint 模板
5. rubric criterion 模板
6. hidden test 生成策略
7. DynamicOperatorPlanner 的主要语义目标
```

---

## 4. Solution Form Taxonomy

Solution form 表示 `<<insert solution here>>` 位置需要补什么。

```yaml
solution_forms:
  function_definition:
    description: "需要输出完整 def 函数。"

  function_body:
    description: "函数签名已存在，只补函数体。"

  expression_completion:
    description: "占位符在赋值右侧或表达式位置，只补表达式或函数调用。"

  statement_block_completion:
    description: "占位符在 main/body 中，需要补一段或多段 Python statements。"

  decorated_function_definition:
    description: "需要补带 decorator 的函数，例如 click command。"

  patch_or_bugfix:
    description: "需要修复已有错误代码。"
```

Solution form 决定：

```text
1. PromptRewriter 如何描述补全任务
2. Verifier 如何插入 solution
3. Hidden tests 如何运行
4. Exporter 如何构造 SFT message
5. 是否允许输出完整函数定义
6. GoldCompatibilityRunner 如何把 solution 插回 context
```

---

## 5. 七个难度轴：H/R/I/E/C/A/V

本版使用七轴：

```text
H = Horizon / Process Complexity
R = Biomedical / Clinical Consequence Risk
I = Information Ambiguity
E = Evidence / Data Structure Complexity
C = Computation / Constraint Complexity
A = Adversarial / Robustness Challenge
V = Verifier / Test Complexity
```

每个轴 intensity 范围：

```yaml
axis_max_intensity:
  H: 3
  R: 3
  I: 3
  E: 3
  C: 3
  A: 3
  V: 3
```

含义：

```text
0 = 该轴不激活
1 = 轻度加难
2 = 中度加难
3 = 强加难
```

核心约束：

```text
axis_intensity 不等于 operator 数量。
axis_intensity 表示轴级目标强度。
operator_intensity 表示单个动态 operator 对该轴贡献的强度。
同一轴上的所有 operator_intensity 之和必须等于该轴 axis_intensity。
```

---

## 6. 七轴定义与强度 Rubric

### 6.1 H：Horizon / Process Complexity

H 表示完成任务需要多少步骤、是否需要多阶段处理、是否需要运行和 debug。

```yaml
H:
  0:
    meaning: "No added process complexity."
  1:
    meaning: "Single function or 1-2 logical steps."
    requirements:
      - "可以通过直接实现或简单调用完成。"
  2:
    meaning: "Multi-branch or multi-stage processing."
    requirements:
      - "需要中间变量、helper、循环、分支或多阶段 pipeline。"
  3:
    meaning: "Execution-feedback or debug-driven process."
    requirements:
      - "需要运行、根据错误信息修复、或显式验证中间结果。"
```

Validation：

```text
If H >= 2, task_state.execution_plan_requirements must be present.
If H = 3, trajectory/verifier must check debug or intermediate validation behavior.
```

### 6.2 R：Biomedical / Clinical Consequence Risk

R 表示错误代码结果对生物医学结论、科研指标、患者级数据或临床解释的潜在影响。

当前子集不是临床决策任务，因此 R 多数为 0/1/2。R 不由 domain 直接决定，必须由具体 problem、output consequence、是否涉及生物医学指标或患者级信息决定。

```yaml
R:
  0:
    meaning: "No biomedical consequence beyond generic code correctness."
  1:
    meaning: "Low biomedical/scientific consequence."
    examples:
      - "科研流程辅助代码，错误影响 workflow 但不直接影响指标解释。"
  2:
    meaning: "Moderate biomedical consequence."
    examples:
      - "FDR, retention time, sequence feature, statistical estimate, free energy 等指标计算错误会影响科研结论。"
  3:
    meaning: "High biomedical/clinical consequence."
    examples:
      - "患者级 EHR、临床预测、治疗/预后相关输出。当前子集中较少。"
```

Validation：

```text
If R >= 2, verifier/rubric must include consequence-sensitive correctness criteria.
If R = 3, safety_gate_required = true.
If R = 3, answer must avoid overclaiming and unsafe clinical inference.
```

### 6.3 I：Information Ambiguity

I 表示题目、context、signature、placeholder 位置是否清楚。

```yaml
I:
  0:
    meaning: "Fully specified task."
  1:
    meaning: "Minor implicit requirement."
  2:
    meaning: "Requires context/signature/placeholder inference."
  3:
    meaning: "Ambiguous input/output format, object schema, or conflicting cues."
```

Validation：

```text
If I > 0, visible_state or task_state must record hidden/implicit/ambiguous requirement.
If I >= 2, verifier must check that the solution uses the correct solution_form and input/output contract.
If I = 3, PromptRewriter must preserve ambiguity without leaking answer.
```

### 6.4 E：Evidence / Data Structure Complexity

E 表示输入数据、文件、对象结构、表格、序列、图像或模型资源的复杂度。

```yaml
E:
  0:
    meaning: "No data structure beyond scalar/simple text."
  1:
    meaning: "Simple list/dict/string or small object."
  2:
    meaning: "Array/table/sequence/file-level structure."
  3:
    meaning: "Complex domain object, multi-resource, image/mask/model/table combination."
```

Validation：

```text
If E > 0, data_state must include concrete data/object/resource complexity.
If E >= 2, hidden tests must include non-trivial data structures.
If E = 3, verifier must check integrated behavior over complex structures.
```

### 6.5 C：Computation / Constraint Complexity

C 表示算法、边界条件、异常处理、格式约束、数值精度、库限制和运行约束。

```yaml
C:
  0:
    meaning: "No added computation or constraint."
  1:
    meaning: "Simple loop/condition/format requirement."
  2:
    meaning: "Multiple conditions, basic exception handling, or output format constraints."
  3:
    meaning: "Numerical precision, robust edge cases, interacting constraints, no-hardcoding, runtime/library restrictions."
```

Validation：

```text
If C > 0, structured constraint must be updated.
If C >= 2, constraint must materially affect solution or verifier.
If C = 3, hidden tests must include edge cases or interacting constraints.
```

### 6.6 A：Adversarial / Robustness Challenge

A 表示干扰信息、相似变量、错误捷径、off-by-one、空值/重复值、误导性示例等。

```yaml
A:
  0:
    meaning: "No adversarial cue."
  1:
    meaning: "Mild edge case or irrelevant context."
  2:
    meaning: "Similar variable/column, duplicate/missing values, misleading example."
  3:
    meaning: "Strong wrong shortcut or plausible but incorrect implementation path."
```

Validation：

```text
If A > 0, visible_state.include or data_state must contain a robustness challenge.
If A >= 2, gold_state.must_not_follow_shortcut must be updated.
If A = 3, suitable_for_dpo_or_stress_test = true.
A operator must not change ground truth.
```

### 6.7 V：Verifier / Test Complexity

V 表示自动验证答案需要多复杂的 hidden tests / verifier。

```yaml
V:
  0:
    meaning: "Simple exact output or visible example only."
  1:
    meaning: "Basic unit test."
  2:
    meaning: "Multiple hidden unit tests or runtime output checks."
  3:
    meaning: "Composite verifier: exception + edge cases + numeric tolerance / dataframe equality / array close / file check / object state check."
```

Validation：

```text
If V > 0, verifier_state or test_state must be updated.
If V >= 2, hidden_tests must contain multiple cases.
If V = 3, verifier_state must include at least two verifier mechanisms or one complex verifier with edge cases.
```

V 轴在本方案中还有一个额外职责：

```text
V controls the complexity of executable verification and acts as the validity anchor for dynamically synthesized operators.
```

也就是：

```text
V 轴不仅控制 verifier / hidden test 难度，也是动态 operator 是否可信的可验证锚点。
```

---

## 7. Task Type → Axis Priority

Axis priority 只由 task_type 主导，domain 不参与。

```yaml
task_axis_priority:
  file_io_and_formatting:
    axis_priority: [E, I, V, C, A, H, R]
    default_axis:
      H: 1
      R: 0
      I: 2
      E: 2
      C: 2
      A: 1
      V: 2

  sequence_and_structure_processing:
    axis_priority: [E, C, V, I, A, H, R]
    default_axis:
      H: 2
      R: 1
      I: 2
      E: 2
      C: 2
      A: 1
      V: 2

  numerical_and_statistical_computation:
    axis_priority: [C, V, H, A, E, I, R]
    default_axis:
      H: 2
      R: 1
      I: 1
      E: 1
      C: 3
      A: 1
      V: 3

  tabular_data_transformation:
    axis_priority: [E, C, V, I, A, H, R]
    default_axis:
      H: 2
      R: 1
      I: 2
      E: 2
      C: 2
      A: 1
      V: 3

  domain_model_or_image_analysis:
    axis_priority: [E, C, H, V, I, A, R]
    default_axis:
      H: 3
      R: 1
      I: 2
      E: 3
      C: 3
      A: 1
      V: 3

  validation_and_code_utility:
    axis_priority: [C, A, V, I, H, E, R]
    default_axis:
      H: 1
      R: 0
      I: 2
      E: 1
      C: 3
      A: 2
      V: 3
```

---

## 8. M-level Budget：7 轴版本

```yaml
m_level_budgets:
  M1:
    num_axes_range: [0, 0]
    total_intensity_range: [0, 0]
    per_axis_intensity_range: [0, 0]
    require_all_axes: false
    allow_multiturn: false
    allow_adversarial: false
    require_safety_gate: false
    include_primary_top_k: 0

  M2:
    num_axes_range: [2, 3]
    total_intensity_range: [2, 4]
    per_axis_intensity_range: [1, 3]
    require_all_axes: false
    allow_multiturn: false
    allow_adversarial: false
    require_safety_gate: false
    include_primary_top_k: 1

  M3:
    num_axes_range: [3, 5]
    total_intensity_range: [5, 8]
    per_axis_intensity_range: [1, 3]
    require_all_axes: false
    allow_multiturn: true
    allow_adversarial: true
    require_safety_gate: false
    include_primary_top_k: 2

  M4:
    num_axes_range: [7, 7]
    total_intensity_range: [10, 16]
    per_axis_intensity_range: [1, 3]
    require_all_axes: true
    allow_multiturn: true
    allow_adversarial: true
    require_safety_gate: true
    include_primary_top_k: 7
```

解释：

```text
M1：base task，不加 operator。
M2：轻度扩展，必须包含 task_type top-1 axis。
M3：中高难扩展，必须包含 task_type top-2 axes，可以启用 A 和 H。
M4：七轴全开，适合作为 stress test / DPO / PRM / RLVR hard setting。
```

M-level 只控制额外 scaling difficulty：

```text
final_difficulty = min(3, base_difficulty + axis_intensity)
```

---

## 9. Routing Pipeline

### 9.1 输出 Schema

```python
class RoutingResult(BaseModel):
    domain: Literal[
        "scientific_software_engineering",
        "bioinformatics_sequence_structure",
        "biomedical_data_analysis",
        "systems_molecular_modeling",
        "omics_measurement_analysis",
    ]

    task_type: Literal[
        "file_io_and_formatting",
        "sequence_and_structure_processing",
        "numerical_and_statistical_computation",
        "tabular_data_transformation",
        "domain_model_or_image_analysis",
        "validation_and_code_utility",
    ]

    solution_form: Literal[
        "function_definition",
        "function_body",
        "expression_completion",
        "statement_block_completion",
        "decorated_function_definition",
        "patch_or_bugfix",
    ]

    secondary_task_types: list[str] = []
    domain_concepts: list[str] = []
    required_capabilities: list[str] = []
    verifier_type_hint: str | None = None
    routing_reason: str
    confidence: float
```

### 9.2 Router 顺序

```text
1. RuleRouter：基于 problem/context/signature/code 的关键词规则。
2. PlaceholderAnalyzer：判断 solution_form。
3. VerifierRouter：根据 task_type + solution_form 推断 verifier_type。
4. LLMRouter：只在 confidence < 0.75 时使用。
5. RoutingValidator：检查 domain/task_type/solution_form/verifier_type 是否一致。
```

---

## 10. AxisWeightPlanner：Primary + Secondary 均由 LLM 决定轴权重

AxisWeightPlanner 的职责是：针对**当前样本**，分别为 primary task type 和每个 secondary task type 输出 H/R/I/E/C/A/V 七轴权重。

本版取消旧的 deterministic secondary rank boost：

```text
旧版：secondary_task_types 根据 task_axis_priority 排名固定转分
新版：secondary_task_types 的轴权重也由 LLM 逐样本判断
```

程序仍然负责：

```text
1. 校验 LLM 输出是否合法；
2. 融合 primary / secondary 权重；
3. 保留 primary hard constraints；
4. 根据 M-level budget 概率抽轴；
5. 分配 axis_intensity；
6. fallback / repair。
```

LLM 不直接决定：

```text
selected_axes
axis_intensity
operator_instances
hidden_tests
verifier_delta
```

---

### 10.1 输入

```json
{
  "task_type": "sequence_and_structure_processing",
  "task_axis_priority": ["E", "C", "V", "I", "A", "H", "R"],
  "secondary_task_types": [
    "file_io_and_formatting",
    "validation_and_code_utility"
  ],
  "secondary_task_axis_priorities": {
    "file_io_and_formatting": ["E", "I", "V", "C", "A", "H", "R"],
    "validation_and_code_utility": ["C", "A", "V", "I", "H", "E", "R"]
  },
  "domain": "bioinformatics_sequence_structure",
  "solution_form": "function_definition",
  "problem": "...",
  "context_summary": "...",
  "signature": "...",
  "verifier_type_hint": "unit_test"
}
```

注意：

```text
1. domain 可以作为语义上下文，但不得参与 axis priority hard constraints。
2. task_type 的 axis_priority 仍是 primary hard constraints 的依据。
3. secondary_task_types 由 LLM 单独给出轴权重和 relevance。
```

---

### 10.2 输出 Schema

```json
{
  "primary_axis_weight_hint": {
    "H": 2,
    "R": 1,
    "I": 4,
    "E": 6,
    "C": 5,
    "A": 2,
    "V": 5
  },
  "secondary_axis_weight_hints": [
    {
      "task_type": "file_io_and_formatting",
      "relevance": 0.8,
      "axis_weight_hint": {
        "H": 1,
        "R": 1,
        "I": 5,
        "E": 6,
        "C": 4,
        "A": 2,
        "V": 5
      },
      "reason": "The task requires parsing file-like biological input before sequence processing."
    },
    {
      "task_type": "validation_and_code_utility",
      "relevance": 0.6,
      "axis_weight_hint": {
        "H": 1,
        "R": 1,
        "I": 4,
        "E": 2,
        "C": 6,
        "A": 5,
        "V": 6
      },
      "reason": "The task includes validation and exception behavior."
    }
  ],
  "axis_weight_reason": "The primary operation is sequence processing, while file parsing and validation are secondary but relevant."
}
```

字段含义：

```text
primary_axis_weight_hint:
  当前样本在 primary task_type 视角下的七轴重要性。

secondary_axis_weight_hints:
  LLM 对每个 secondary_task_type 单独判断 relevance 和七轴权重。

relevance:
  该 secondary_task_type 对当前样本的实际相关性，范围 [0, 1]。
  0 = 几乎只是背景；1 = 非核心但非常重要。
```

---

### 10.3 AxisWeightPlanner Prompt

创建：

```text
prompts/axis_weight_planner_7axis.jinja
```

Prompt：

```text
You are planning difficulty-axis weights for a MedAgentGym biomedical/scientific code-completion task.

You must output axis weights for both the primary task type and each secondary task type.

Input:

Primary task type:
{{ task_type }}

Primary task axis priority:
{{ task_axis_priority }}

Secondary task types:
{{ secondary_task_types }}

Secondary task axis priorities:
{{ secondary_task_axis_priorities }}

Domain:
{{ domain }}

Solution form:
{{ solution_form }}

Problem:
{{ problem }}

Context summary:
{{ context_summary }}

Signature:
{{ signature }}

Verifier type hint:
{{ verifier_type_hint }}

Difficulty axes:
H = Horizon / Process Complexity: number of execution steps, multi-stage processing, validation/debug loop.
R = Biomedical / Clinical Consequence Risk: consequence of wrong biomedical/scientific output or clinical/patient-level interpretation.
I = Information Ambiguity: ambiguity in task description, context, signature, placeholder, or input/output contract.
E = Evidence / Data Structure Complexity: complexity of files, arrays, tables, sequences, masks, model objects, or structured resources.
C = Computation / Constraint Complexity: algorithms, edge cases, exceptions, numeric precision, output format, library/runtime constraints.
A = Adversarial / Robustness Challenge: misleading examples, similar variables, shortcut traps, off-by-one cases, duplicates/missing values.
V = Verifier / Test Complexity: complexity of hidden tests, executable checks, numeric tolerance, dataframe/array equality, file/object state checks.

Task:
1. For the primary task type, assign primary_axis_weight_hint over H/R/I/E/C/A/V.
2. For each secondary task type, assign:
   - relevance from 0.0 to 1.0;
   - secondary-specific axis_weight_hint over H/R/I/E/C/A/V;
   - a short reason.

Weight scale:
1 = low relevance
2 = mild relevance
3 = moderate relevance
4 = high relevance
5 = very high relevance
6 = central relevance

Important rules:
1. The primary task type remains dominant.
2. Use primary task_axis_priority as the main reference for primary_axis_weight_hint.
3. The top-1 primary axis should usually receive at least 5.
4. The top-2 primary axes should usually receive at least 4.
5. Secondary task types may influence final weights but must not override primary hard constraints.
6. Domain provides semantic context only; it must not override task_type axis priority.
7. Do not choose selected_axes.
8. Do not assign axis_intensity.
9. Do not generate operators.
10. Output JSON only.

Output JSON only:
{
  "primary_axis_weight_hint": {
    "H": 1,
    "R": 1,
    "I": 4,
    "E": 6,
    "C": 5,
    "A": 2,
    "V": 5
  },
  "secondary_axis_weight_hints": [
    {
      "task_type": "file_io_and_formatting",
      "relevance": 0.7,
      "axis_weight_hint": {
        "H": 1,
        "R": 1,
        "I": 5,
        "E": 6,
        "C": 4,
        "A": 2,
        "V": 5
      },
      "reason": "..."
    }
  ],
  "axis_weight_reason": "..."
}
```

---

### 10.4 Validator

实现：

```python
class SecondaryAxisWeightHint(BaseModel):
    task_type: str
    relevance: float
    axis_weight_hint: dict[str, int]
    reason: str | None = None

class AxisWeightPlannerResult(BaseModel):
    primary_axis_weight_hint: dict[str, int]
    secondary_axis_weight_hints: list[SecondaryAxisWeightHint] = []
    axis_weight_reason: str | None = None
```

校验规则：

```text
1. primary_axis_weight_hint 必须包含 H/R/I/E/C/A/V 七个轴。
2. 每个 secondary axis_weight_hint 也必须包含 H/R/I/E/C/A/V 七个轴。
3. 所有 weight 必须是 int。
4. 所有 weight 必须在 [1, 6]。
5. secondary_task_types 最多 3 个。
6. secondary relevance 必须在 [0, 1]。
7. secondary_axis_weight_hints 中的 task_type 必须来自 routing 结果的 secondary_task_types。
8. primary task_type top-1 axis 权重不能低于 5。
9. primary task_type top-2 axes 权重不能低于 4。
10. domain 不得覆盖 primary hard constraints。
11. LLM 输出非法时先 repair 一次；repair 仍失败则 fallback。
```

Fallback 规则：

```text
primary fallback:
  使用 primary task_type 的 axis_priority 转成 rank weight。

secondary fallback:
  使用对应 secondary task_type 的 axis_priority 转成 rank weight。

relevance fallback:
  若 LLM 没给或非法，默认 0.5。
```

七轴 rank fallback：

```text
rank 1 -> 7
rank 2 -> 6
rank 3 -> 5
rank 4 -> 4
rank 5 -> 3
rank 6 -> 2
rank 7 -> 1
```

---

## 11. Primary / Secondary Axis Weight Fusion

本版不再使用 deterministic secondary rank boost，而是使用 LLM 输出的 secondary_axis_weight_hints。

配置：

```yaml
axis_weight_fusion:
  enabled: true
  secondary_fusion_strength: 0.30
  max_secondary_task_types: 3
  mode: relevance_weighted_average
  primary_hard_constraints: true
```

### 11.1 融合公式

对每个轴 `a`：

```text
final_axis_weight(a)
= primary_axis_weight_hint(a)
+ λ * weighted_average_secondary_axis_weight(a)
```

其中：

```text
λ = secondary_fusion_strength = 0.30
```

secondary 部分：

```text
weighted_average_secondary_axis_weight(a)
= sum_j relevance_j * secondary_axis_weight_hint_j(a)
  / sum_j relevance_j
```

如果没有 secondary_task_types：

```text
final_axis_weight(a) = primary_axis_weight_hint(a)
```

如果 secondary_task_types 存在但 relevance 全部为 0：

```text
final_axis_weight(a) = primary_axis_weight_hint(a)
```

### 11.2 程序约束

```text
1. secondary 只能影响 final_axis_weights。
2. secondary 不直接决定 selected_axes。
3. secondary 不直接决定 axis_intensity。
4. secondary 不能删除 M2/M3 primary hard constraints。
5. M2 必须包含 primary task_type top-1 axis。
6. M3 必须包含 primary task_type top-2 axes。
7. M4 必须包含 H/R/I/E/C/A/V 全部七轴。
```

示例：

```json
{
  "primary_axis_weight_hint": {
    "H": 2,
    "R": 1,
    "I": 4,
    "E": 6,
    "C": 5,
    "A": 2,
    "V": 5
  },
  "secondary_axis_weight_hints": [
    {
      "task_type": "file_io_and_formatting",
      "relevance": 0.8,
      "axis_weight_hint": {"H": 1, "R": 1, "I": 5, "E": 6, "C": 4, "A": 2, "V": 5}
    },
    {
      "task_type": "validation_and_code_utility",
      "relevance": 0.6,
      "axis_weight_hint": {"H": 1, "R": 1, "I": 4, "E": 2, "C": 6, "A": 5, "V": 6}
    }
  ],
  "final_axis_weights": {
    "H": 2.30,
    "R": 1.30,
    "I": 5.36,
    "E": 7.54,
    "C": 6.41,
    "A": 3.33,
    "V": 6.65
  }
}
```

---

## 12. ScalingPlan：7 轴版本

```python
class SecondaryAxisWeightHint(BaseModel):
    task_type: str
    relevance: float
    axis_weight_hint: dict[str, int]
    reason: str | None = None

class ScalingPlan(BaseModel):
    global_level: Literal["M1", "M2", "M3", "M4"]

    task_type: str
    secondary_task_types: list[str] = []
    domain: str
    solution_form: str

    axis_weight_source: Literal["llm", "fallback", "repaired"]
    primary_axis_weight_hint: dict[str, int]
    secondary_axis_weight_hints: list[SecondaryAxisWeightHint] = []
    axis_weight_reason: str | None = None

    axis_priority: list[str]
    final_axis_weights: dict[str, float]
    axis_weight_fusion_mode: str = "primary_plus_relevance_weighted_secondary"
    secondary_fusion_strength: float = 0.30

    selected_axes: list[Literal["H", "R", "I", "E", "C", "A", "V"]]
    axis_intensity: dict[str, int]
    total_intensity: int
    sampling_seed: int

    allow_multiturn: bool
    allow_adversarial: bool
    require_safety_gate: bool
```

### 12.1 selected_axes 规则

```text
M1: selected_axes = []
M2: 必须包含 primary task_type top-1 axis
M3: 必须包含 primary task_type top-2 axes
M4: selected_axes = [H, R, I, E, C, A, V]
```

抽样逻辑：

```text
1. 读取 M-level budget。
2. 加入 primary hard constraints。
3. 在剩余轴中按 final_axis_weights 不放回抽样。
4. 如果 allow_adversarial=false，A 不参与抽样。
5. 如果 allow_multiturn=false，H 可选但 intensity 最高为 1，且不得生成真正多轮 operator。
```

### 12.2 axis_intensity 分配

```text
1. 从 total_intensity_range 采样 total_intensity。
2. 每个 selected_axis 先分配 1。
3. remaining 按 final_axis_weights 加权分配。
4. 每个轴 intensity <= 3。
5. 未选中轴 intensity = 0。
6. sum(axis_intensity.values()) == total_intensity。
```

### 12.3 ScalingPlan Validator

```text
1. M1 selected_axes 必须为空。
2. M1 total_intensity 必须为 0。
3. M1 不允许 operator。
4. selected_axes 数量必须在 num_axes_range 内。
5. total_intensity 必须在 total_intensity_range 内。
6. axis_intensity 只能是 0-3。
7. 未选中的轴 intensity 必须为 0。
8. 被选中的轴 intensity 必须 >= 1。
9. axis_intensity 总和必须等于 total_intensity。
10. M2 必须包含 primary task_type top-1 axis。
11. M3 必须包含 primary task_type top-2 axes。
12. M4 必须包含 H/R/I/E/C/A/V 全部七轴。
13. secondary_axis_weight_hints 不能覆盖 primary hard constraints。
14. 如果 allow_adversarial=false，则 A 轴不能被选中，或 A intensity 必须为 0。
15. 如果 allow_multiturn=false，则 H 轴不能生成真正多轮 operator；如果 H 被选中，H<=1。
16. require_safety_gate=true 时，后续必须生成 safety_gate_required=true。
17. final_axis_weights 必须由 primary_axis_weight_hint + relevance-weighted secondary hints 计算得到。
```
---

# 13. Dynamic Verifiable Operator Synthesis


## 13. ToolConfigPlanner：LLM 动态工具配置与预算规划

MedAgentGym-style agent 任务不能只给最终 prompt，还必须给出工具环境。不同 M-level 的任务应有不同的工具集合、工具 schema、工具调用 budget 和 forbidden tools。工具配置不由固定规则完全写死，而是由 LLM 根据 task_type、solution_form、样本内容、selected_axes、axis_intensity 和目标 M-level 动态规划，程序负责校验和 fallback。

参考目标格式：

```text
Task ID: chest_pain_triage_L2_003
Mode: evaluation
Allowed tools:
- ehr_reader
- lab_query
- risk_score_calculator

Forbidden tools:
- web_search
- python_exec
- guideline_search

Tool budget:
- max total tool calls: 8
- max calls per tool:
  - ehr_reader: 4
  - lab_query: 3
  - risk_score_calculator: 2

Output requirement:
- return JSON with fields: triage_level, next_step, rationale, safety_flags
```

以及每个工具需要附 schema 和使用说明：

```text
Available tools:

1. ehr_reader(patient_id, section)
   Use to read structured patient records.
   Valid sections: chief_complaint, history, meds, allergies, vitals.

2. lab_query(patient_id, test_name)
   Use to retrieve existing lab results only.
   This tool cannot order new tests.

3. risk_score_calculator(score_name, inputs)
   Use to calculate predefined clinical scores.
   Valid score_name: HEART
```

### 13.1 ToolConfig 的作用

ToolConfig 不决定 domain，也不决定 selected_axes。它决定 agent 在当前 scaled environment 中能用什么工具、不能用什么工具、每个工具能调用几次，以及最终输出必须满足什么格式。

ToolConfig 影响：

```text
1. action_space
2. tool_state
3. PromptRewriter 的 tool block
4. trajectory sampler 的工具调用限制
5. verifier / reward 中的 tool budget penalty
6. PRM step labels 中的 tool-use correctness
7. RLVR env 的 action space 和 max_steps
```

ToolConfig 不允许：

```text
1. 泄露 solution / hidden tests
2. 绕过 verifier
3. 给出不可执行工具
4. 允许 dangerous shell / network / unrestricted python exec
5. 让 forbidden_tools 同时出现在 allowed_tools
```

### 13.2 ToolConfig schema

```python
class ToolSpec(BaseModel):
    tool_name: str
    description: str
    input_schema: dict
    output_schema: dict | None = None
    when_to_use: str
    limitations: list[str] = []
    examples: list[dict] = []

class ToolBudget(BaseModel):
    max_total_tool_calls: int
    max_calls_per_tool: dict[str, int]
    max_consecutive_calls_per_tool: dict[str, int] = {}
    max_debug_calls: int = 0
    max_validation_calls: int = 0

class OutputRequirement(BaseModel):
    output_format: Literal["text", "json", "code", "file", "table"]
    required_fields: list[str] = []
    forbidden_fields: list[str] = []
    json_schema: dict | None = None
    strict: bool = True

class ToolConfig(BaseModel):
    env_id: str
    global_level: Literal["M1", "M2", "M3", "M4"]
    planning_source: Literal["llm", "fallback"]

    allowed_tools: list[ToolSpec]
    forbidden_tools: list[str]
    tool_budget: ToolBudget
    output_requirement: OutputRequirement

    tool_choice_reason: str
    budget_reason: str
    related_axes: list[Literal["H", "R", "I", "E", "C", "A", "V"]] = []

    validation_trace: list[str] = []
```

### 13.3 ToolConfigPlanner 输入

```json
{
  "seed_task": "...",
  "domain": "biomedical_data_analysis",
  "task_type": "tabular_data_transformation",
  "solution_form": "function_definition",
  "resource_manifest": [],
  "scaling_plan": {
    "global_level": "M3",
    "selected_axes": ["E", "C", "V", "I"],
    "axis_intensity": {"H": 1, "R": 0, "I": 2, "E": 2, "C": 2, "A": 0, "V": 2}
  },
  "base_environment": "..."
}
```

### 13.4 ToolConfigPlanner 输出

```json
{
  "allowed_tools": [
    {
      "tool_name": "read_file",
      "description": "Read a local resource file mounted in the task workspace.",
      "input_schema": {"path": "string"},
      "output_schema": {"content": "string"},
      "when_to_use": "Use when the task requires inspecting an input file before writing the solution.",
      "limitations": ["Cannot read outside resource_root."],
      "examples": [{"path": "data/input.csv"}]
    },
    {
      "tool_name": "validate_code",
      "description": "Run the proposed code against visible tests and return errors.",
      "input_schema": {"code": "string"},
      "output_schema": {"passed": "boolean", "errors": "array"},
      "when_to_use": "Use after drafting a solution to catch runtime or assertion errors.",
      "limitations": ["Does not reveal hidden tests."],
      "examples": []
    }
  ],
  "forbidden_tools": ["web_search", "network_request", "unrestricted_shell"],
  "tool_budget": {
    "max_total_tool_calls": 6,
    "max_calls_per_tool": {
      "read_file": 2,
      "validate_code": 3,
      "debug": 1
    },
    "max_consecutive_calls_per_tool": {
      "validate_code": 2
    },
    "max_debug_calls": 1,
    "max_validation_calls": 3
  },
  "output_requirement": {
    "output_format": "code",
    "required_fields": [],
    "forbidden_fields": [],
    "json_schema": null,
    "strict": true
  },
  "tool_choice_reason": "This task benefits from reading resources and validating generated code.",
  "budget_reason": "M3 allows moderate multi-step execution but still limits repeated validation."
}
```

### 13.5 M-level tool budget bounds

LLM 决定具体工具和 budget，但必须落在 M-level 的程序化上下界内。

```yaml
tool_budget_bounds:
  M1:
    allowed_tool_count_range: [0, 2]
    max_total_tool_calls_range: [0, 2]
    max_calls_per_tool_upper: 2
    allow_debug_tool: false
    allow_external_resource_tools: false

  M2:
    allowed_tool_count_range: [1, 3]
    max_total_tool_calls_range: [2, 5]
    max_calls_per_tool_upper: 3
    allow_debug_tool: false
    allow_external_resource_tools: true

  M3:
    allowed_tool_count_range: [2, 5]
    max_total_tool_calls_range: [4, 9]
    max_calls_per_tool_upper: 4
    allow_debug_tool: true
    max_debug_calls_upper: 2
    allow_external_resource_tools: true

  M4:
    allowed_tool_count_range: [3, 7]
    max_total_tool_calls_range: [7, 14]
    max_calls_per_tool_upper: 6
    allow_debug_tool: true
    max_debug_calls_upper: 4
    allow_external_resource_tools: true
```

解释：

```text
M1：基本单轮任务，可不给工具或只给 validate_code。
M2：少量工具，适合简单 read/validate。
M3：中等工具预算，允许 debug 和多步验证。
M4：工具集更丰富、总调用预算更高，适合 hard agent setting。
```

### 13.6 ToolConfigPlanner Prompt

创建：

```text
prompts/tool_config_planner.jinja
```

模板：

```text
You are planning tools for a MedAgentGym-style executable agent task.

Your task is to decide the allowed tools, forbidden tools, tool schemas, tool-call budget, and output requirement for this sample.

Input:
Seed task:
{{ seed_task }}

Domain:
{{ domain }}

Task type:
{{ task_type }}

Solution form:
{{ solution_form }}

Resource manifest:
{{ resource_manifest }}

Scaling plan:
{{ scaling_plan }}

M-level tool budget bounds:
{{ tool_budget_bounds }}

Rules:
1. Choose tools that are useful for the current task and difficulty level.
2. Harder M-levels may have more tools and higher tool-call budgets.
3. Use selected_axes and axis_intensity to decide whether multi-step execution, debug, resource inspection, or stricter validation is needed.
4. If H is high, allow enough calls for multi-step execution or debug.
5. If E is high, include tools for resource/schema/file/table inspection when resources exist.
6. If C is high, include validation tools for constraints and edge cases.
7. If V is high, include validate_code / hidden-test-compatible validation, but never reveal hidden tests.
8. If A is high, avoid tools that make shortcut exploitation easier, and require validation.
9. If R is high, include safety-sensitive output requirements and forbid overclaiming or unsafe tools.
10. Do not include forbidden tools in allowed_tools.
11. Do not allow web search or network access unless explicitly permitted by the environment.
12. Do not allow unrestricted shell or arbitrary filesystem access.
13. Output must fit within the M-level tool budget bounds.
14. Return JSON only.

Output JSON:
{
  "allowed_tools": [
    {
      "tool_name": "...",
      "description": "...",
      "input_schema": {},
      "output_schema": {},
      "when_to_use": "...",
      "limitations": [],
      "examples": []
    }
  ],
  "forbidden_tools": [],
  "tool_budget": {
    "max_total_tool_calls": 0,
    "max_calls_per_tool": {},
    "max_consecutive_calls_per_tool": {},
    "max_debug_calls": 0,
    "max_validation_calls": 0
  },
  "output_requirement": {
    "output_format": "json | text | code | file | table",
    "required_fields": [],
    "forbidden_fields": [],
    "json_schema": null,
    "strict": true
  },
  "tool_choice_reason": "...",
  "budget_reason": "..."
}
```

### 13.7 ToolConfigValidator

新增：

```text
src/medenvscale/scaling/tool_config_validator.py
```

检查规则：

```text
1. allowed_tools 数量必须在 M-level tool_count_range 内。
2. max_total_tool_calls 必须在 M-level max_total_tool_calls_range 内。
3. max_calls_per_tool 不能超过 M-level upper bound。
4. forbidden_tools 不能出现在 allowed_tools。
5. 每个 allowed_tool 必须有 description、input_schema、when_to_use、limitations。
6. 所有 max_calls_per_tool 的 key 必须属于 allowed_tools。
7. 如果 H >= 2 或 M3/M4，允许 debug；否则 debug 工具默认不允许。
8. 如果 E >= 2 且有 resource_manifest，至少应有 read_file / inspect_file / inspect_table 类工具之一。
9. 如果 V >= 2，至少应有 validate_code 或等价 verifier-facing tool。
10. 如果 C >= 2，output_requirement 或 verifier-facing tool 必须能检查约束。
11. 禁止 web_search、network_request、unrestricted_shell，除非 config 显式允许。
12. ToolConfig 不得泄露 hidden tests / solution / verifier expected outputs。
13. output_requirement 必须和 task_type / solution_form 兼容。
```

失败处理：

```text
1. LLM tool config 不合法，repair 一次。
2. repair 后仍失败，使用 rule-based fallback tool config。
3. fallback 也失败，任务标记 needs_review 或 filtered。
```

### 13.8 Rule-based fallback tool config

```python
def fallback_tool_config(task_type, solution_form, global_level, axis_intensity, resource_manifest):
    tools = []

    if resource_manifest:
        tools.append(read_file_tool())

    if solution_form in {"function_definition", "function_body", "statement_block_completion", "patch_or_bugfix"}:
        tools.append(validate_code_tool())

    if global_level in {"M3", "M4"} or axis_intensity.get("H", 0) >= 2:
        tools.append(debug_tool())

    if task_type == "tabular_data_transformation":
        tools.append(inspect_table_tool())

    if task_type == "numerical_and_statistical_computation":
        tools.append(validate_numeric_output_tool())

    return clamp_to_m_level_bounds(tools, global_level)
```

### 13.9 ToolConfig 如何进入 PromptRewriter

PromptRewriter 必须在最终 prompt 中加入工具块：

```text
Task ID: {{ env_id }}
Mode: evaluation

Allowed tools:
{{ allowed_tools_with_schema }}

Forbidden tools:
{{ forbidden_tools }}

Tool budget:
- max total tool calls: {{ max_total_tool_calls }}
- max calls per tool:
{{ max_calls_per_tool }}

Output requirement:
{{ output_requirement }}

Task:
{{ task_prompt }}
```

对于 code completion 子集，工具块可以是：

```text
Allowed tools:
1. validate_code(code)
   Run the candidate solution against visible tests. Does not reveal hidden tests.
2. debug(code, error)
   Use to repair code after a validation error. Limited by debug budget.

Forbidden tools:
- web_search
- network_request
- unrestricted_shell

Tool budget:
- max total tool calls: 4
- validate_code: 3
- debug: 1

Output requirement:
- return only the code that should replace <<insert solution here>>
```

### 13.10 ToolConfig 与七轴的关系

```text
H：提高多步调用预算、允许 debug、多轮 validate。
R：增加安全输出字段或禁止 unsafe tools；临床任务中要求 safety_flags。
I：允许 schema/resource inspection 工具，但不泄露 hidden info。
E：增加 read_file / inspect_table / schema inspection 工具。
C：增加 validate_code / constraint checker / format checker。
A：限制 shortcut-prone tools，要求 validation；可降低某些直接查询工具预算。
V：增加 verifier-facing 工具和 hidden-test-compatible validation，但不暴露 hidden tests。
```

### 13.11 Trajectory sampler 的工具预算执行

trajectory sampler 必须维护工具调用计数：

```python
class ToolCallState(BaseModel):
    total_calls: int = 0
    calls_per_tool: dict[str, int] = {}
    consecutive_calls_per_tool: dict[str, int] = {}
```

每次 action 前检查：

```text
1. tool_name 是否在 allowed_tools 中。
2. tool_name 是否在 forbidden_tools 中。
3. total_calls 是否超过 max_total_tool_calls。
4. calls_per_tool[tool_name] 是否超过 max_calls_per_tool。
5. consecutive_calls 是否超过 max_consecutive_calls_per_tool。
6. tool args 是否满足 input_schema。
```

超预算行为：

```text
1. 返回 tool_budget_exceeded observation。
2. 记录 trajectory failure_type = tool_budget_exceeded。
3. reward 中加入 tool_budget_penalty。
4. PRM step 标注为 incorrect 或 invalid_tool_use。
```

### 13.12 ToolConfig 与 Reward

Reward 增加 tool-use 项：

```python
final_score = (
    0.50 * verifier_pass_score
    + 0.25 * rubric_score
    + 0.10 * process_score
    + 0.10 * tool_use_score
    + 0.05 * output_format_score
    - penalties
)
```

`tool_use_score` 计算：

```text
1. 使用 allowed tools。
2. 未使用 forbidden tools。
3. 未超过总预算。
4. 未超过 per-tool budget。
5. 在需要时正确使用 validate/debug/read/inspect 工具。
6. 没有无意义重复调用。
```

---

## 14.1 为什么取消 Operator Candidate Pool

本方案取消固定 `Operator Candidate Pool / Operator Template Registry`。

原因：

```text
1. 当前任务类型虽然只有 6 类，但每条样本的可验证加难方式差异很大。
2. 固定 operator 列表容易限制扩展空间。
3. 论文主张的是 sample-specific dynamic scaling，不应让 operator 类型过早固化。
4. 只要 operator 能生成可执行 verifier_delta 和 hidden tests，就不需要预先限定 operator 名称。
```

因此，operator 不从预定义列表中选择。DynamicOperatorPlanner 根据每条样本动态合成 operator。

固定的是协议：

```text
1. operator 必须绑定 selected axis。
2. operator 必须满足 axis_intensity。
3. operator 必须修改 structured state patch。
4. operator 必须自带 verifier_delta / hidden tests。
5. operator 必须通过 generic validator。
6. operator 必须通过 gold compatibility 或 gold regeneration 检查。
```

开放的是内容：

```text
1. operator_type 可以动态命名。
2. transformation_goal 可以按样本动态生成。
3. hidden tests 可以按样本动态生成。
4. verifier checks 可以按样本动态生成。
5. rubric_delta 可以按样本动态生成。
```

## 14.2 两种生成模式

### 模式 A：Gold-compatible operator

MVP 默认使用该模式。

要求：

```text
operator 可以加难，但原始 solution 必须仍然通过新 verifier / hidden tests。
```

优点：

```text
1. 稳定。
2. 成本低。
3. 不需要重新生成 gold solution。
4. 适合大规模自动生成。
```

缺点：

```text
加难空间受限，因为不能加入原 solution 不支持的新行为。
```

### 模式 B：Gold-regenerating operator

进阶版本支持。

流程：

```text
operator 修改任务要求
→ 生成 updated_gold_solution
→ 插回 context
→ 在 Docker / sandbox 中执行 hidden tests
→ updated_gold_solution 必须通过
→ 原 solution 可以失败
```

优点：

```text
1. 扩展能力强。
2. 可以真正改变任务要求。
3. 适合 M4 hard generation。
```

风险：

```text
1. 成本高。
2. 需要更强 verifier。
3. 容易生成错误 gold。
4. 必须有多轮 repair。
```

建议：

```text
MVP：gold-compatible only
Full version：M3/M4 可启用 gold-regenerating mode
```

---

## 15None. DynamicOperatorInstance Schema

```python
class StateUpdates(BaseModel):
    task_state_patch: dict = Field(default_factory=dict)
    data_state_patch: dict = Field(default_factory=dict)
    tool_state_patch: dict = Field(default_factory=dict)
    visible_state_patch: dict = Field(default_factory=dict)
    gold_state_patch: dict = Field(default_factory=dict)
    verifier_state_patch: dict = Field(default_factory=dict)
    test_state_patch: dict = Field(default_factory=dict)
    turn_state_patch: dict = Field(default_factory=dict)

class VerifierDelta(BaseModel):
    new_checks: list[dict] = []
    new_hidden_tests: list[dict] = []
    exception_tests: list[dict] = []
    numeric_tolerance_tests: list[dict] = []
    array_close_tests: list[dict] = []
    dataframe_equal_tests: list[dict] = []
    file_output_tests: list[dict] = []
    object_state_tests: list[dict] = []
    static_checks: list[dict] = []
    expected_failure_modes: list[str] = []

class OperatorConstraints(BaseModel):
    must_preserve_core_task: bool = True
    must_not_change_ground_truth: bool = True
    must_not_leak_answer: bool = True
    must_keep_verifier_executable: bool = True
    must_not_change_solution_form: bool = True

class VerificationContract(BaseModel):
    generation_mode: Literal["gold_compatible", "gold_regenerating"] = "gold_compatible"
    gold_solution_must_pass: bool = True
    must_add_executable_test: bool = True
    must_update_verifier_if_behavior_changes: bool = True
    must_not_change_solution_form: bool = True
    must_not_leak_answer: bool = True

class DynamicOperatorInstance(BaseModel):
    axis: Literal["H", "R", "I", "E", "C", "A", "V"]
    operator_type: str
    operator_intensity: int

    transformation_goal: str
    rationale: str

    state_updates: StateUpdates
    verifier_delta: VerifierDelta
    rubric_delta: list[dict] = []

    expected_effect: dict = {}
    verification_contract: VerificationContract
    constraints: OperatorConstraints
```

关键说明：

```text
operator_type 是 LLM 动态生成的描述性名称，不需要预先出现在固定列表中。
真正生效的是 state_updates + verifier_delta + rubric_delta。
```

---

## 16None. DynamicOperatorPlanner Prompt

创建：

```text
prompts/dynamic_verifiable_operator_planner.jinja
```

模板：

```text
You are synthesizing sample-specific, verifiable difficulty operators for a MedAgentGym-style biomedical/scientific code completion task.

Do not choose from a fixed operator list.
Instead, synthesize dynamic OperatorInstances that are specific to the current task.

Input:

Seed task:
{{ seed_task }}

Base environment:
{{ base_environment }}

Domain:
{{ domain }}

Task type:
{{ task_type }}

Solution form:
{{ solution_form }}

Domain concepts:
{{ domain_concepts }}

Scaling plan:
{{ scaling_plan }}

Tool config:
{{ tool_config }}

Axis intensity rubric:
{{ intensity_rubric }}

Rules:
1. Generate operators only for selected_axes.
2. Do not generate operators for axes with axis_intensity = 0.
3. For each selected axis, generate one or more operators.
4. For each axis, sum(operator_intensity) must equal axis_intensity[axis].
5. operator_intensity must be 1, 2, or 3.
6. Do not choose from a fixed operator list; synthesize sample-specific operator_type names.
7. Each operator must modify structured state, not the final user prompt.
8. Each behavior-changing operator must include verifier_delta.
9. If axis = V, verifier_delta must add hidden tests or verifier checks.
10. If axis = C, add checkable constraints or edge-case behavior.
11. If axis = E, add data/object/resource complexity that can be tested.
12. If axis = I, add implicit/ambiguous requirements that remain verifiable.
13. If axis = A, add a misleading/shortcut challenge and verifier_delta that rejects shortcut failure.
14. If axis = R, add biomedical consequence-aware rubric/checks.
15. Respect the allowed_tools / forbidden_tools / tool_budget in tool_config.
16. Preserve solution_form.
16. Do not leak the gold solution.
17. Keep the task executable.
18. In gold_compatible mode, original gold solution must still pass all new tests.
19. In gold_regenerating mode, include a requirement to regenerate updated_gold_solution.
20. Return JSON only.

Output JSON:
{
  "operator_instances": [
    {
      "axis": "C",
      "operator_type": "sample_specific_operator_name",
      "operator_intensity": 2,
      "transformation_goal": "...",
      "rationale": "...",
      "state_updates": {
        "task_state_patch": {},
        "data_state_patch": {},
        "tool_state_patch": {},
        "visible_state_patch": {},
        "gold_state_patch": {},
        "verifier_state_patch": {},
        "test_state_patch": {},
        "turn_state_patch": {}
      },
      "verifier_delta": {
        "new_checks": [],
        "new_hidden_tests": [],
        "exception_tests": [],
        "numeric_tolerance_tests": [],
        "array_close_tests": [],
        "dataframe_equal_tests": [],
        "file_output_tests": [],
        "object_state_tests": [],
        "static_checks": [],
        "expected_failure_modes": []
      },
      "rubric_delta": [],
      "expected_effect": {},
      "verification_contract": {
        "generation_mode": "gold_compatible",
        "gold_solution_must_pass": true,
        "must_add_executable_test": true,
        "must_update_verifier_if_behavior_changes": true,
        "must_not_change_solution_form": true,
        "must_not_leak_answer": true
      },
      "constraints": {
        "must_preserve_core_task": true,
        "must_not_change_ground_truth": true,
        "must_not_leak_answer": true,
        "must_keep_verifier_executable": true,
        "must_not_change_solution_form": true
      }
    }
  ]
}
```

---

## 17None. GenericOperatorValidator

GenericOperatorValidator 不依赖固定 operator_type 名称，只检查 operator 是否满足可验证协议。

主函数：

```python
def validate_dynamic_operator_instances(
    operator_instances: list[DynamicOperatorInstance],
    scaling_plan: ScalingPlan,
    seed_task: NormalizedCodeTask,
    base_environment: ExecutableEnvSpec,
    intensity_rubric: dict,
) -> ValidationResult:
    ...
```

必须检查：

```text
1. operator.axis 必须属于 H/R/I/E/C/A/V。
2. operator.axis 必须在 scaling_plan.selected_axes 中。
3. axis_intensity=0 的轴不能有 operator。
4. 每个 selected_axis 至少有一个 operator。
5. 每个轴 sum(operator_intensity) == axis_intensity[axis]。
6. operator_intensity 必须是 1、2、3。
7. operator 必须修改至少一个 structured state patch。
8. operator 不能直接写 user_prompt。
9. operator 不能泄露 solution / answer。
10. operator 不能破坏 context scaffold。
11. operator 不能修改 idx / original_task_id。
12. operator 不能改变 solution_form，除非明确进入 gold_regenerating mode 并通过验证。
13. 行为变化必须有 verifier_delta。
14. V > 0 必须有 verifier_delta.new_hidden_tests 或 new_checks。
15. C > 0 必须新增可检查约束或边界行为。
16. E > 0 必须新增可测试的数据结构复杂度。
17. A > 0 必须新增干扰项，并让 verifier 能拒绝 shortcut failure。
18. R > 0 必须新增 biomedical consequence-aware rubric/check。
19. 所有 verifier_delta 必须可构造 verifier。
20. gold-compatible mode 下原始 solution 必须通过新 verifier。
```

轴级检查：

```text
H > 0:
  task_state 或 turn_state 必须体现步骤、顺序、验证或 debug 要求。

R >= 2:
  rubric_delta 或 verifier_delta 必须体现 biomedical consequence-sensitive correctness。

I > 0:
  visible_state / task_state 必须体现隐含、缺失或需推断的信息。

E > 0:
  data_state 必须增加数据结构或对象复杂度。

C > 0:
  task_state / gold_state / verifier_delta 必须新增约束或边界行为。

A > 0:
  visible_state 或 data_state 必须包含干扰项、错误捷径或鲁棒性挑战。

V > 0:
  verifier_delta 必须新增测试复杂度。
```

---

## 18None. VerifierDeltaValidator

新增模块：

```text
src/medenvscale/scaling/verifier_delta_validator.py
```

作用：验证动态 operator 生成的 verifier_delta 是否真的可执行。

主函数：

```python
def validate_verifier_delta(
    op: DynamicOperatorInstance,
    env: ExecutableEnvSpec,
) -> ValidationResult:
    ...
```

检查规则：

```text
1. 如果 op.axis == V，必须新增 hidden tests 或 verifier checks。
2. 如果 operator 修改行为，必须有 verifier_delta。
3. hidden test 必须包含输入、预期输出或可执行断言。
4. numeric_tolerance_tests 必须包含 expected 和 tolerance。
5. exception_tests 必须包含 expected_exception。
6. dataframe_equal_tests 必须包含 expected dataframe spec 或 check function。
7. array_close_tests 必须包含 expected array 和 tolerance。
8. file_output_tests 必须包含文件路径和内容检查规则。
9. object_state_tests 必须包含对象状态 invariant。
10. static_checks 必须可程序化执行。
11. verifier_delta 必须能被 VerifierBuilder 转成 VerifierSpec。
```

伪代码：

```python
def validate_verifier_delta(op, env):
    errors = []

    if op.axis == "V" and not has_any_verifier_delta(op):
        errors.append("V operator must add hidden tests or verifier checks")

    if modifies_behavior(op) and not has_any_verifier_delta(op):
        errors.append("Behavior-changing operator must include verifier_delta")

    for test in op.verifier_delta.new_hidden_tests:
        if not is_executable_test(test, env.solution_form):
            errors.append("Hidden test is not executable")

    if not verifier_can_be_built(op.verifier_delta, env.solution_form):
        errors.append("Verifier delta cannot be built")

    return ValidationResult(errors=errors)
```

---

## 19None. GoldCompatibilityRunner

新增模块：

```text
src/medenvscale/scaling/gold_compatibility_runner.py
```

作用：在 gold-compatible 模式下，将原始 `solution` 插回 `context`，运行新 verifier / hidden tests，确认动态 operator 没有把原任务改坏。

主函数：

```python
def run_gold_compatibility_check(
    env: ExecutableEnvSpec,
    operator_instances: list[DynamicOperatorInstance],
) -> GoldCheckResult:
    ...
```

流程：

```text
1. apply_operator_instances 得到 updated_env。
2. VerifierBuilder 根据 updated_env.verifier_state + verifier_delta 构造 verifier。
3. 将原始 solution 按 solution_form 插回 context。
4. 在 sandbox / local isolated runner 中执行。
5. 运行全部 visible tests + hidden tests。
6. 如果全部通过，则 operator 合法。
7. 如果失败：
   - gold_compatible mode：reject / repair operator。
   - gold_regenerating mode：进入 GoldRegenerationRunner。
```

伪代码：

```python
def run_gold_compatibility_check(env, operator_instances):
    updated_env = apply_operator_instances(env, operator_instances)
    verifier = build_verifier(updated_env)

    result = verifier.run(
        context=updated_env.context,
        solution=updated_env.solution,
        solution_form=updated_env.solution_form,
    )

    if result.pass_all:
        return GoldCheckResult(status="pass")

    if updated_env.generation_mode == "gold_compatible":
        return GoldCheckResult(status="reject_or_repair", errors=result.errors)

    if updated_env.generation_mode == "gold_regenerating":
        return GoldCheckResult(status="needs_gold_regeneration", errors=result.errors)
```

---

## 20None. GoldRegenerationRunner

新增模块：

```text
src/medenvscale/scaling/gold_regeneration_runner.py
```

作用：在 gold-regenerating mode 下，允许动态 operator 修改任务要求，并生成新的 gold solution。

流程：

```text
1. 接收 updated_env + failed gold compatibility result。
2. 调用 LLM 生成 updated_gold_solution。
3. 将 updated_gold_solution 插回 context。
4. 运行 VerifierRunner。
5. 如果失败，repair updated_gold_solution，最多 N 次。
6. 成功后写入 updated_env.gold_state.updated_solution。
7. 标记 original_solution_may_fail = true。
```

限制：

```text
1. 默认 MVP 不启用。
2. 只允许 M3/M4 启用。
3. 必须有强 verifier_delta。
4. 必须通过 sandbox execution。
5. 失败则整条 scaled env filtered 或回退到 gold-compatible operator。
```

---

## 21None. OperatorApplier

主函数：

```python
def apply_operator_instances(
    base_environment: ExecutableEnvSpec,
    operator_instances: list[DynamicOperatorInstance],
) -> ExecutableEnvSpec:
    ...
```

功能：

```text
按顺序应用 state_updates 到 base_environment。
```

支持更新：

```text
task_state
data_state
tool_state
visible_state
gold_state
verifier_state
test_state
turn_state
safety_gate_required
robust_verifier_required
suitable_for_dpo_or_stress_test
```

Patch 操作：

```text
set: 设置字段
add: 列表追加或 dict 合并
remove: 删除字段或列表元素
hide: 添加到 visible_state.hide
include: 添加到 visible_state.include
append_constraint: 添加约束
append_verifier_rule: 添加 verifier rule
append_hidden_test: 添加 hidden test
```

硬性禁止修改：

```text
idx
original_task_id
split
raw solution
raw context scaffold
immutable_resource_hash
```

---

## 22None. PromptRewriter

PromptRewriter 是唯一允许生成最终 user prompt 的模块。

输入：

```text
scaled_environment_state
scaling_plan
operator_instances
```

输出：

```python
class PromptRewriteResult(BaseModel):
    system_prompt: str | None = None
    user_prompt: str
    prompt_format: Literal["single_turn", "multi_turn", "agent_loop"]
    visible_fields_used: list[str] = []
    hidden_fields_respected: list[str] = []
    resource_mentions: list[str] = []
```

规则：

```text
1. 不泄露 solution / ground_truth。
2. 不暴露 visible_state.hide。
3. 必须保留 context 中必要代码 scaffold。
4. 必须保留 <<insert solution here>> 或明确 solution insertion point。
5. 如果 solution_form 是 expression_completion，不得要求模型输出完整函数。
6. 如果 V > 0，prompt 不直接暴露 hidden test。
7. 如果 A > 0，干扰项自然出现在 context/problem 中。
8. 如果 C > 0，输出格式/异常/边界条件约束必须明确。
9. 如果 gold_compatible mode，不能加入原 solution 不支持的新要求。
10. 如果 gold_regenerating mode，必须使用 updated_gold_solution 对应的新任务要求。
```

---

## 23None. VerifierBuilder

Verifier 根据以下内容构建：

```text
task_type
solution_form
domain
base verifier_state
operator verifier_delta
test_state
hidden_tests
rubric hooks
```

Verifier 类型：

```text
exact_match
runtime_output_match
unit_test
exception_unit_test
numeric_tolerance
array_close
dataframe_equal
regex_check
file_output_check
object_state_check
hybrid_static_runtime_check
llm_judge_fallback
```

VerifierSpec：

```python
class VerifierSpec(BaseModel):
    verifier_id: str
    env_id: str
    verifier_type: str
    solution_form: str
    checks: list[dict]
    hidden_tests: list[dict] = []
    exception_tests: list[dict] = []
    numeric_tolerance: float | None = None
    rubric_links: list[str] = []
    static_checks: list[dict] = []
    generated_from_operator_ids: list[str] = []
```

---

## 24None. QPoints 与 Rubrics

### 24.1 QuestionPoint

```python
class QuestionPoint(BaseModel):
    qpoint_id: str
    env_id: str
    capability: str
    description: str
    expected_behavior: str
    failure_modes: list[str] = []
    related_axes: list[str] = []
    verifier_hook: str | None = None
```

### 24.2 RubricCriterion

```python
class RubricCriterion(BaseModel):
    criterion_id: str
    qpoint_id: str
    description: str
    score_type: Literal["binary", "ordinal", "continuous"] = "binary"
    weight: float = 1.0
    judge_method: Literal[
        "unit_test",
        "exception_test",
        "numeric_tolerance",
        "array_close",
        "dataframe_equal",
        "regex_check",
        "file_check",
        "object_state_check",
        "llm_judge"
    ]
    positive_examples: list[str] = []
    negative_examples: list[str] = []
    generated_from_operator_id: str | None = None
```

Rubric 原则：

```text
1. 每个 qpoint 生成 1-3 条 rubric。
2. 优先绑定 verifier_hook。
3. V 轴越高，rubric 越应程序化可判定。
4. R 轴越高，rubric 越应强调 biomedical consequence 和 no-overclaiming。
5. A 轴越高，rubric 必须覆盖 shortcut / misleading cue failure。
6. operator.rubric_delta 可以直接转成 rubric candidate，但必须通过 RubricValidator。
```

---

## 25None. Training Views

### 25.1 SFT View

```json
{
  "id": "...",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "env_id": "...",
  "domain": "bioinformatics_sequence_structure",
  "task_type": "sequence_and_structure_processing",
  "solution_form": "function_definition",
  "tool_config": {"allowed_tools": [], "tool_budget": {}},
  "difficulty": {"H": 2, "R": 1, "I": 2, "E": 3, "C": 2, "A": 1, "V": 3},
  "operator_mode": "gold_compatible",
  "rubrics": ["..."],
  "verifier_id": "..."
}
```

### 25.2 DPO View

```json
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "...",
  "preference_reason": "chosen passes verifier and satisfies higher-weight rubrics",
  "env_id": "...",
  "domain": "...",
  "task_type": "...",
  "tool_config": {"allowed_tools": [], "tool_budget": {}},
  "difficulty": {"H": 2, "R": 0, "I": 2, "E": 2, "C": 3, "A": 2, "V": 3},
  "rubric_deltas": {},
  "operator_failure_modes": ["shortcut_failure", "hidden_test_failure"]
}
```

### 25.3 PRM View

```json
{
  "env_id": "...",
  "trajectory_id": "...",
  "step_id": 3,
  "state": "...",
  "action": {},
  "label": "partially_correct",
  "rubric_hits": ["..."],
  "score": 0.5,
  "related_axes": ["H", "C", "V"],
  "related_operator_ids": ["..."]
}
```

### 25.4 RLVR View

```json
{
  "env_id": "...",
  "initial_observation": "...",
  "action_space": ["validate_code", "debug", "read_file", "submit_answer"],
  "tool_config": {"allowed_tools": [], "forbidden_tools": [], "tool_budget": {}},
  "verifier": {},
  "reward_fn": {},
  "max_steps": 12,
  "difficulty": {"H": 3, "R": 1, "I": 2, "E": 3, "C": 3, "A": 2, "V": 3}
}
```

---

## 26None. Pipeline Stages

```text
Stage 01: load raw train/test tasks
Stage 02: normalize task fields
Stage 03: infer domain / task_type / solution_form
Stage 04: build base executable env
Stage 05: infer base difficulty
Stage 06: build scaling plan with 7 axes
Stage 07: Dynamic Verifiable Operator Synthesis
Stage 08: GenericOperatorValidator
Stage 09: VerifierDeltaValidator
Stage 10: OperatorApplier
Stage 11: GoldCompatibilityRunner / GoldRegenerationRunner
Stage 12: PromptRewriter
Stage 13: Verifier / hidden tests construction
Stage 14: trajectory sampling
Stage 15: qpoint / rubric generation
Stage 16: reward computation
Stage 17: export SFT / DPO / PRM / RLVR views
Stage 18: quality filtering and reporting
```

---

## 27None. 目录结构

```text
MedEnvScale-MedAgentGym/
  configs/
    medagentgym_pilot.yaml
    domain_taxonomy.yaml
    task_type_taxonomy.yaml
    solution_form_taxonomy.yaml
    task_axis_priority.yaml
    axis_definitions_7axis.yaml
    m_level_budgets_7axis.yaml
    dynamic_operator_protocol.yaml
    verifier_delta_schema.yaml
    verifier_rules.yaml
    export_views.yaml

  prompts/
    task_router.jinja
    axis_weight_planner_7axis.jinja
    tool_config_planner.jinja
    dynamic_verifiable_operator_planner.jinja
    prompt_rewriter.jinja
    qpoint_extract.jinja
    rubric_generate.jinja
    verifier_generate.jinja
    gold_solution_regenerate.jinja
    trajectory_judge.jinja

  src/medenvscale/
    data/
      medagentgym_loader.py
      task_normalizer.py
      placeholder_analyzer.py

    schema/
      raw_task.py
      normalized_task.py
      routing.py
      env_spec.py
      scaling.py
      operator.py
      verifier.py
      rubric.py
      trajectory.py
      export_views.py

    routing/
      rule_router.py
      llm_router.py
      routing_validator.py

    scaling/
      base_difficulty.py
      axis_weight_planner.py
      scaling_plan.py
      scaling_validator.py
      dynamic_verifiable_operator_planner.py
      generic_operator_validator.py
      verifier_delta_validator.py
      operator_applier.py
      gold_compatibility_runner.py
      gold_regeneration_runner.py
      prompt_rewriter.py
      difficulty_builder.py

    verifier/
      verifier_builder.py
      verifier_runner.py
      hidden_test_builder.py
      reward.py

    rubric/
      qpoint_extractor.py
      rubric_generator.py
      rubric_validator.py

    trajectory/
      sampler.py
      trajectory_judge.py
      trajectory_filter.py

    export/
      sft_exporter.py
      dpo_exporter.py
      prm_exporter.py
      rlvr_exporter.py

    pipeline/
      stage01_load_tasks.py
      stage02_route_tasks.py
      stage03_build_seed_envs.py
      stage04_scale_envs.py
      stage05_verifier_reward.py
      stage06_sample_trajectories.py
      stage07_qpoints_rubrics.py
      stage08_export_views.py

  scripts/
    run_pipeline.py
    run_stage.py
    inspect_scaled_env.py
    validate_dataset.py

  tests/
    test_loader.py
    test_router.py
    test_placeholder_analyzer.py
    test_axis_weight_planner_7axis.py
    test_scaling_plan_7axis.py
    test_dynamic_verifiable_operator_planner.py
    test_generic_operator_validator.py
    test_verifier_delta_validator.py
    test_gold_compatibility_runner.py
    test_gold_regeneration_runner.py
    test_prompt_rewriter.py
    test_verifier_builder.py
    test_export_views.py
```

---

## 28None. Quality Filtering

过滤条件：

```text
1. 缺少 problem 或 context。
2. 找不到 <<insert solution here>> 且无法判断 solution_form。
3. solution 为空，且无法生成 verifier。
4. route confidence < 0.6。
5. ScalingPlan 不合法。
6. operator intensity sum 不匹配。
7. operator 没有 state_updates。
8. 行为变化但没有 verifier_delta。
9. V > 0 但没有 hidden test / verifier update。
10. operator 泄露 solution。
11. operator 改变 solution_form。
12. gold-compatible mode 下原 solution 不能通过新 verifier。
13. gold-regenerating mode 下 updated_gold_solution 不能通过新 verifier。
14. PromptRewriter 泄露 hidden tests / solution。
15. Verifier 无法执行。
16. A > 0 但没有真实干扰项。
17. C > 0 但没有真实约束。
18. R >= 2 但没有 biomedical consequence-aware rubric/check。
```

---

## 29None. 实验设计

### 实验 1：M-level 难度单调性验证

目标：验证 M1-M4 是否真的越来越难。

设置：

```text
同一批 seed tasks 生成 M1/M2/M3/M4。
评测模型在每个 M-level 上的 pass@1、verifier score、rubric score、debug steps。
```

预期：

```text
M1 > M2 > M3 > M4
```

### 实验 2：Axis ablation

目标：验证七个轴是否各自带来不同错误模式。

设置：

```text
只激活单轴 H/R/I/E/C/A/V。
比较模型错误类型。
```

重点：

```text
C: 边界/约束错误
E: 数据结构错误
V: hidden test/generalization 错误
I: 误解 placeholder/输入输出 contract
A: shortcut 错误
H: 多步骤/debug 失败
R: 生物医学指标/过度声称相关错误
```

### 实验 3：Dynamic operator vs template operator

目标：证明动态可验证 operator 合成比固定模板池更灵活。

对比：

```text
A. Fixed template operator baseline
B. Dynamic Verifiable Operator Synthesis (ours)
```

指标：

```text
operator validity rate
gold compatibility pass rate
hidden test diversity
rubric coverage
model error diversity
filtered sample rate
```

### 实验 4：Gold-compatible vs Gold-regenerating

目标：比较两种 operator 生成模式。

对比：

```text
A. gold-compatible operator only
B. gold-regenerating operator for M3/M4
```

指标：

```text
valid env rate
average difficulty increase
verifier pass rate of updated gold
manual audit correctness
M4 model failure rate
```

### 实验 5：Rubric-grounded SFT

对比：

```text
baseline: original successful trajectories
ours: scaled env + verifier/rubric-filtered successful trajectories
```

指标：

```text
held-out pass@1
rubric score
hidden test pass rate
M3/M4 generalization
```

### 实验 6：DPO preference learning

构造：

```text
chosen: pass verifier + high rubric score
rejected: fail hidden test / shortcut / format error / exception failure
```

指标：

```text
preference win rate
pass@1
hidden test pass rate
format error rate
shortcut error rate
```

### 实验 7：PRM / verifier reranking

目标：训练 step-level process reward model 或 verifier reranker。

指标：

```text
step label accuracy
error localization accuracy
Best-of-N reranking gain
V-axis hard subset improvement
```

---

## 30None. 测试要求

### 30.1 Routing tests

```text
test_domain_has_5_allowed_values
test_task_type_has_6_allowed_values
test_solution_form_detection_expression
test_solution_form_detection_function_definition
test_domain_does_not_decide_axis_priority
test_task_type_decides_axis_priority
```

### 30.2 Scaling tests

```text
test_m1_has_no_axes
test_m2_contains_task_type_top1
test_m3_contains_task_type_top2
test_m4_has_all_7_axes
test_axis_intensity_sum_equals_total
test_unselected_axes_have_zero_intensity
test_a_axis_blocked_when_allow_adversarial_false
test_h_axis_capped_when_allow_multiturn_false
test_v_axis_requires_verifier_or_test_update
```

### 30.3 Dynamic operator tests

```text
test_operator_axis_must_be_selected
test_no_operator_for_zero_intensity_axis
test_each_selected_axis_has_operator
test_operator_intensity_sum_matches_axis_intensity
test_dynamic_operator_type_need_not_be_registered
test_operator_does_not_write_user_prompt
test_operator_preserves_ground_truth
test_behavior_changing_operator_requires_verifier_delta
test_v_axis_updates_hidden_tests
test_a_axis_adds_real_robustness_challenge
test_r_axis_updates_consequence_rubric
```

### 30.4 VerifierDelta tests

```text
test_verifier_delta_hidden_test_executable
test_verifier_delta_numeric_tolerance_schema
test_verifier_delta_exception_test_schema
test_verifier_delta_array_close_schema
test_verifier_delta_dataframe_equal_schema
test_verifier_delta_builds_verifier_spec
```

### 30.5 Gold check tests

```text
test_gold_compatible_original_solution_passes_new_tests
test_gold_compatible_rejects_broken_operator
test_gold_regenerating_generates_updated_solution
test_gold_regenerating_rejects_unverified_solution
test_solution_form_preserved_by_default
```

### 30.6 Verifier / Export tests

```text
test_verifier_runs_on_function_definition
test_verifier_runs_on_expression_completion
test_numeric_tolerance_verifier
test_dataframe_equal_verifier
test_exception_unit_test_verifier
test_sft_export_schema
test_dpo_export_schema
test_prm_export_schema
test_rlvr_export_schema
test_rlvr_export_contains_tool_budget
```

---

## 31None. 一键 Pipeline

```bash
python scripts/run_pipeline.py \
  --config configs/medagentgym_pilot.yaml \
  --input_train data/train_tasks.jsonl \
  --input_test data/test_tasks.jsonl \
  --output_root outputs/medenvscale_medagentgym_7axis_dynamic_operator \
  --seed_limit 500 \
  --m_levels M1 M2 M3 M4 \
  --operator_mode gold_compatible \
  --llm_mode api \
  --sample_trajectories true \
  --export_views sft dpo prm rlvr
```

启用 gold-regenerating：

```bash
python scripts/run_pipeline.py \
  --config configs/medagentgym_pilot.yaml \
  --input_train data/train_tasks.jsonl \
  --input_test data/test_tasks.jsonl \
  --output_root outputs/medenvscale_medagentgym_7axis_gold_regen \
  --seed_limit 500 \
  --m_levels M1 M2 M3 M4 \
  --operator_mode gold_regenerating \
  --allow_gold_regeneration_for M3 M4 \
  --llm_mode api \
  --sample_trajectories true \
  --export_views sft dpo prm rlvr
```

分阶段：

```bash
python scripts/run_stage.py --stage load_tasks --config configs/medagentgym_pilot.yaml
python scripts/run_stage.py --stage route_tasks --config configs/medagentgym_pilot.yaml
python scripts/run_stage.py --stage build_seed_envs --config configs/medagentgym_pilot.yaml
python scripts/run_stage.py --stage scale_envs --config configs/medagentgym_pilot.yaml
python scripts/run_stage.py --stage verify_gold_compatibility --config configs/medagentgym_pilot.yaml
python scripts/run_stage.py --stage verifier_reward --config configs/medagentgym_pilot.yaml
python scripts/run_stage.py --stage sample_trajectories --config configs/medagentgym_pilot.yaml
python scripts/run_stage.py --stage qpoints_rubrics --config configs/medagentgym_pilot.yaml
python scripts/run_stage.py --stage export_views --config configs/medagentgym_pilot.yaml
```

---

## 32None. Codex 实现任务拆分

### Task 1：创建配置与 taxonomy

```text
configs/domain_taxonomy.yaml
configs/task_type_taxonomy.yaml
configs/solution_form_taxonomy.yaml
configs/task_axis_priority.yaml
configs/axis_definitions_7axis.yaml
configs/m_level_budgets_7axis.yaml
configs/axis_weight_fusion.yaml
configs/dynamic_operator_protocol.yaml
configs/verifier_delta_schema.yaml
```

### Task 2：实现 schema

```text
RawMedAgentGymCodeTask
NormalizedCodeTask
RoutingResult
ExecutableEnvSpec
ScalingPlan
DynamicOperatorInstance
VerifierDelta
VerifierSpec
QuestionPoint
RubricCriterion
AgentTrajectory
SFT/DPO/PRM/RLVR views
```

### Task 3：实现 loader / normalizer / placeholder analyzer

```text
读取 train/test jsonl。
保留 idx/problem/solution/context/signature/code。
识别 <<insert solution here>>。
判断 solution_form。
```

### Task 4：实现 routing

```text
RuleRouter + LLM fallback。
输出 domain/task_type/solution_form/verifier_type_hint。
保证 domain 不进入 axis_priority。
```

### Task 5：实现 7-axis scaling

```text
AxisWeightPlanner
ScalingPlan builder
ScalingPlan validator
M-level budget
Task-type-based hard constraints
Probability-weighted sampling without replacement
axis_intensity allocation
```

### Task 6：实现 Dynamic Verifiable Operator Synthesis

```text
不使用固定 Operator Candidate Pool。
LLM 针对每条样本动态生成 OperatorInstance。
每个 OperatorInstance 必须包含 state_updates、verifier_delta、rubric_delta、verification_contract。
```

### Task 7：实现 GenericOperatorValidator / VerifierDeltaValidator

```text
不根据 operator_type 名称校验。
根据 axis、intensity、state_updates、verifier_delta、gold compatibility 校验。
```

### Task 8：实现 GoldCompatibilityRunner / GoldRegenerationRunner

```text
gold-compatible mode：原始 solution 必须通过新 verifier。
gold-regenerating mode：生成 updated_gold_solution 并验证。
```

### Task 9：实现 verifier / hidden tests

```text
unit_test
exception_unit_test
numeric_tolerance
array_close
dataframe_equal
regex_check
file_output_check
object_state_check
static_check
```

### Task 10：实现 qpoint / rubric / reward

```text
从 task_type + axes + verifier_delta 生成 qpoints/rubrics。
reward = verifier_score + rubric_score + process_score - penalties。
```

### Task 11：实现 trajectory + export

```text
采样 successful / failed trajectories。
导出 SFT / DPO / PRM / RLVR。
```

---

## 33None. 最终验收标准

每条 scaled environment 必须包含：

```text
env_id
original_task_id
split
problem
context
signature
solution_form
domain
task_type
base_difficulty
difficulty(H/R/I/E/C/A/V)
tool_config.allowed_tools
tool_config.forbidden_tools
tool_config.tool_budget
tool_config.output_requirement
scaling.global_level
scaling.primary_axis_weight_hint
scaling.secondary_axis_weight_hints
scaling.axis_priority
scaling.final_axis_weights
scaling.selected_axes
scaling.axis_intensity
scaling.total_intensity
scaling.operator_instances
operator_mode
verifier_state
test_state
verifier_delta
hidden_tests
gold_compatibility_result
rubrics
```

核心原则：

```text
1. Domain 使用 5 类简化 taxonomy。
2. Task type 使用 6 类简化 taxonomy。
3. 难度轴使用 H/R/I/E/C/A/V 七轴。
4. Domain 不决定轴。
5. Task type 是轴优先级主依据。
6. Solution form 决定插入代码和 verifier 运行方式。
7. M1 不加 operator。
8. M4 七轴全开。
9. 不使用固定 Operator Candidate Pool。
10. DynamicOperatorPlanner 针对每条样本动态合成 OperatorInstance。
11. 每个 operator 必须自带 verifier_delta / hidden tests。
12. GenericOperatorValidator 不依赖固定 operator 名称。
13. VerifierDeltaValidator 必须验证 verifier_delta 可执行。
14. GoldCompatibilityRunner 必须验证原 solution 仍通过，除非启用 gold-regenerating mode。
15. Operator 不得泄露 solution / answer。
16. PromptRewriter 是唯一生成最终 user_prompt 的模块。
17. Verifier 必须随 operator 同步更新。
```

---

## 34None. 可直接给 Codex 的总 Prompt

```text
请按照本文件实现 MedEnvScale-Train-MedAgentGym 7-axis Dynamic Verifiable Operator 版本。

当前数据是 MedAgentGym 风格 biomedical/scientific code completion 子集，每条样本通常包含 idx/problem/solution/context/signature/code。

请实现一个 pipeline：
raw task → normalization → domain/task_type/solution_form routing → base executable env → 7-axis scaling → Dynamic Verifiable Operator Synthesis → verifier_delta validation → gold compatibility check → verifier/hidden tests → rubrics → trajectories → SFT/DPO/PRM/RLVR exports。

关键要求：
1. domain 使用 5 类：scientific_software_engineering, bioinformatics_sequence_structure, biomedical_data_analysis, systems_molecular_modeling, omics_measurement_analysis。
2. task_type 使用 6 类：file_io_and_formatting, sequence_and_structure_processing, numerical_and_statistical_computation, tabular_data_transformation, domain_model_or_image_analysis, validation_and_code_utility。
3. difficulty axes 使用 7 个：H/R/I/E/C/A/V。
4. domain 不决定 axis priority、selected_axes 或 axis_intensity。
5. axis priority 的 hard constraints 只由 primary task_type 主导，solution_form 和 sample-level content 可以辅助。
6. primary task_type 和 secondary_task_types 的轴权重都由 LLM 逐样本输出；secondary 不再使用固定 rank boost。
7. 程序使用 primary_axis_weight_hint + relevance-weighted secondary_axis_weight_hints 计算 final_axis_weights。
8. secondary_axis_weight_hints 只能影响 final_axis_weights，不能覆盖 M2/M3 primary hard constraints。
9. V = Verifier/Test Complexity，必须控制 hidden tests 和 verifier 复杂度。
7. R = Biomedical/Clinical Consequence Risk，当前子集中主要用于科研指标/生物医学后果，不强行作为 clinical risk。
8. M1 不加 operator；M2 包含 task_type top-1 axis；M3 包含 task_type top-2 axes；M4 七轴全开。
9. 不使用固定 Operator Candidate Pool / Template Registry。
10. DynamicOperatorPlanner 必须针对每条样本动态生成 sample-specific OperatorInstance。
11. operator_type 名称可以动态生成，不需要出现在预定义列表。
12. 每个 OperatorInstance 必须包含 state_updates、verifier_delta、rubric_delta、verification_contract。
13. 行为变化必须同步 verifier_delta。
14. V 轴必须新增 hidden tests 或 verifier checks。
15. GenericOperatorValidator 不依赖 operator_type 名称，只检查可验证协议。
16. VerifierDeltaValidator 必须确认 verifier_delta 可执行。
17. gold-compatible mode 下，原始 solution 必须通过新 verifier。
18. gold-regenerating mode 只允许 M3/M4，必须生成 updated_gold_solution 并通过 verifier。
19. Operator 不得修改 raw solution / original_task_id / core task，也不得泄露答案。
20. PromptRewriter 是唯一生成最终 prompt 的模块。
21. 需要实现 tests，保证 routing、scaling、dynamic operator、verifier_delta、gold compatibility、verifier 和 export 全部可 smoke test。
```
