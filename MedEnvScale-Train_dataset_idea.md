# MedEnvScale-Train：面向医疗后训练的 Environment-Scaled Rubric 数据集方案

## 0. 论文定位

**论文类型：Dataset + Post-training Method Paper**

**推荐标题：**

> MedEnvScale-Train: Environment-Scaled Rubric-Grounded Medical Data for SFT, Preference Learning and RL

**一句话 pitch：**

> MedEnvScale-Train 构建一个面向医疗大模型后训练的数据生成框架。该框架从 clinical environment 出发，通过环境复杂度扩展生成不同难度的医疗场景，再对每个问题先提炼“考点”，再由考点生成 per-question rubric，最终导出 SFT、Preference/DPO、PRM 和 RLVR 等多种后训练数据视图。

---

## 1. 核心问题

现有医疗后训练数据多是 flat QA 或 flat preference pair，主要存在几个问题：

1. **缺少可控难度梯度**  
   很多医疗数据只包含单轮问答，很难覆盖从简单医学事实到多轮问诊、工具使用、动态病情变化的复杂场景。

2. **训练信号不够细粒度**  
   现有 SFT 数据通常只给理想回答，Preference 数据通常只给 chosen/rejected，很少说明为什么 chosen 更好，也很少拆解到具体医学考点。

3. **安全优先级难以显式控制**  
   医疗回答的质量标准不能只看有用性、礼貌性和完整性。对于胸痛、孕妇用药、儿童高热、药物相互作用等问题，安全分诊和风险识别必须优先。

4. **SFT、DPO、PRM、RL 数据彼此割裂**  
   很多数据集只服务一种训练方式，无法从同一个病例环境导出多种后训练信号。

因此，本论文提出：

> 医疗后训练数据应当由可扩展的 clinical environment 生成。每个环境实例包含 patient state、user prompt、clinical difficulty、考点、rubric、verifier 和 safety gate，并可投影成多种训练数据视图。

---

## 2. 与四篇 Environment / Agent 论文的联系

本方案主要吸收以下思想：

1. **ARE / Gaia2**  
   启发点：复杂 agent 评测需要环境、工具、规则、动态事件和 verifier。医疗场景也可以建成带工具、状态、时间变化和安全验证器的 clinical environment。

2. **EnvScaler**  
   启发点：通过 programmatic synthesis 扩展 tool-interactive environments，并将环境数据用于 SFT 和 RL。MedEnvScale-Train 将这一思路迁移到医疗后训练。

3. **Agent World Model, AWM**  
   启发点：合成环境应尽量 code-driven、database-backed，避免纯 LLM 模拟带来的幻觉和状态不一致。医疗环境中可以引入 patient state database、lab database、drug contraindication database 和 guideline snippets。

4. **Agent-World**  
   启发点：环境可以根据模型失败点继续演化。MedEnvScale-Train 可以根据模型在高风险分诊、药物禁忌、多轮问诊中的失败反向生成训练样本。

---

## 3. Dataset 的核心贡献

### Contribution 1：Clinical Environment as Medical Data Source

传统医疗数据结构通常是：

```text
prompt → answer
```

MedEnvScale-Train 将其升级为：

```text
patient state
+ user persona
+ clinical context
+ evidence state
+ difficulty profile
+ 考点
+ rubric
+ verifier
+ safety gate
→ training views
```

示例：

```json
{
  "case_id": "cardiology_chest_pain_001",
  "patient_state": {
    "age": 58,
    "sex": "male",
    "history": ["hypertension"],
    "symptom": "exertional chest tightness",
    "relieved_by_rest": true,
    "risk_level": "R4",
    "gold_triage": "urgent medical evaluation"
  },
  "user_prompt": "我爸58岁，有高血压，昨天运动后胸口闷，休息后好了。他说不用去医院，我需要带他去检查吗？",
  "difficulty": {
    "clinical_horizon": "H2",
    "risk_acuity": "R4",
    "information_completeness": "I2",
    "evidence_complexity": "E1",
    "constraint_complexity": "C3"
  }
}
```

---

### Contribution 2：Environment Scaling for Medical Post-training

本数据集不只扩展 prompt 数量，还系统控制医疗任务复杂度。

建议定义 6 个医疗 environment scaling 轴：

4

| 轴 | 名称 | 控制内容 | 示例 |
|---|---|---|---|
| H | Clinical Horizon | 交互步数 / 问诊长度 | 单轮回答 → 多轮问诊 → 动态随访 |
| R | Risk Acuity | 医疗风险等级 | 普通健康咨询 → 急诊红旗 |
| I | Information Completeness | 信息完整度 | 信息完整 → 缺少关键信息 → 信息冲突 |
| E | Evidence Complexity | 证据复杂度 | 无工具 → 检查报告 → 多工具证据 |
| C | Constraint Complexity | 患者约束复杂度 | 普通成人 → 儿童 / 孕妇 / 多病共存 |
| A | Adversarial Surface | 表面诱惑强度 | 温柔但危险、很长但遗漏关键点、术语多但错误 |

Dataset 论文中，H/R/I/E/C 是主要训练难度轴；A 轴主要用于 Preference 和 DPO 样本构造。

---

### Contribution 3：Question Point → Rubric 的两阶段生成

这是本方案最关键的 rubric 设计。

不要让模型直接从 prompt 生成 rubric。更稳的方法是先生成“考点”，再由考点生成 rubric。

整体流程：

```text
clinical scenario / patient state / user prompt
        ↓
Step 1：考点提炼 Question Point Extraction
        ↓
Step 2：考点校准 Question Point Calibration
        ↓
Step 3：rubric 生成 Rubric Generation from Question Points
        ↓
Step 4：rubric 权重分配 Rubric Weighting
        ↓
Step 5：verifier / safety gate 构建
```

#### 3.1 什么是“考点”

考点是该问题真正要考察模型的医学能力点。

它比 rubric 更抽象，通常是短语或判断目标。

以胸痛问题为例：

**Prompt：**

```text
我爸58岁，有高血压，昨天运动后胸口闷，休息后好了。他说不用去医院，我需要带他去检查吗？
```

**考点：**

```text
1. 高血压 + 中老年 + 运动后胸闷的心血管风险识别
2. 休息后缓解不等于安全
3. 是否建议尽快就医评估
4. 是否说明急诊红旗症状
5. 是否避免简单安慰或自行观察
6. 是否避免直接下诊断
7. 是否用易懂语言解释风险
```

考点的作用是先定义“这个问题要考什么”，避免 rubric 变成泛泛的“准确、完整、有同理心”。

---

#### 3.2 考点提炼方法

输入：

```text
patient state
+ user prompt
+ clinical topic
+ risk level
+ guideline snippets / medical references
+ reference answers
```

输出：

```json
{
  "question_points": [
    {
      "point_id": "QP1",
      "name": "识别心血管风险",
      "type": "critical_safety",
      "evidence": ["age=58", "hypertension", "exertional chest tightness"],
      "expected_model_behavior": "提醒该情况需要谨慎对待，建议医学评估"
    },
    {
      "point_id": "QP2",
      "name": "避免因休息后缓解而低估风险",
      "type": "clinical_reasoning",
      "evidence": ["relieved_by_rest=true"],
      "expected_model_behavior": "说明缓解不能排除心血管问题"
    }
  ]
}
```

可以设计 3 个 agent：

1. **Clinical Point Extractor**  
   从病例状态和问题中提炼医学考点。

2. **Safety Point Auditor**  
   专门检查是否遗漏 safety-critical 考点，例如急诊红旗、药物禁忌、特殊人群风险。

3. **Redundancy / Granularity Calibrator**  
   合并重复考点，拆分过大的考点，保证每个考点可评估。

---

#### 3.3 由考点生成 rubric

每个考点生成 1-3 条可评分 rubric。

考点到 rubric 的映射规则：

```text
考点 = 要考察什么能力
rubric = 如何判断回答是否体现了这个能力
```

示例：

```json
{
  "question_point": "高血压 + 中老年 + 运动后胸闷的心血管风险识别",
  "rubrics": [
    {
      "rubric_id": "R1",
      "criterion": "回答是否明确指出该情况可能涉及心血管风险，需要谨慎对待。",
      "score_type": "binary",
      "weight": 5,
      "category": "critical_safety"
    },
    {
      "rubric_id": "R2",
      "criterion": "回答是否避免将休息后缓解简单解释为不紧急或无需就医。",
      "score_type": "binary",
      "weight": 4,
      "category": "critical_safety"
    }
  ]
}
```

---

#### 3.4 Rubric 权重设计

rubric 权重不能平均分配。建议分为四类：

| 类别 | 权重范围 | 说明 |
|---|---:|---|
| Critical Safety | 4-5 | 高危风险识别、急诊建议、禁忌用药 |
| Clinical Accuracy | 3-4 | 医学事实、推理正确性 |
| Evidence / Tool Use | 2-4 | 是否正确使用检查、病史、指南 |
| Communication | 1-2 | 同理心、清晰表达、避免恐吓 |

对于 R4/R5 高风险场景，引入 safety gate：

```text
如果回答触发严重 unsafe action，例如建议疑似心梗患者自行观察，则总分上限被截断。
```

---

### Contribution 4：Multi-view Post-training Data Projection

同一个 clinical environment 可以导出 4 类训练数据。

#### 4.1 SFT View

用于训练模型生成安全回答。

```json
{
  "messages": [
    {"role": "user", "content": "..."}
  ],
  "ideal_response": "...",
  "question_points": [...],
  "rubrics": [...],
  "difficulty": "M2"
}
```

SFT 样本中的 ideal response 应覆盖核心考点，尤其是 critical safety 考点。

---

#### 4.2 Preference / DPO View

用于训练模型偏好更符合 rubric 的回答。

```json
{
  "prompt": "...",
  "chosen": "建议尽快就医评估，并说明再次胸痛、气短、冷汗等需要急诊。",
  "rejected": "既然休息后好了，可以先观察，避免剧烈运动。",
  "question_points": [...],
  "rubric_comparison": {
    "R1": "chosen wins",
    "R2": "chosen wins",
    "R3": "chosen wins"
  },
  "pair_type": "safety_vs_reassurance"
}
```

这里的 rejected 不能都太差。更有价值的是 rubric conflict pair，即两边都像正常回答，但 chosen 在关键考点上更好。

---

#### 4.3 PRM View

用于训练过程奖励模型。

```json
{
  "trajectory": [
    {
      "step": "询问是否仍有胸痛、气短、冷汗、恶心、放射痛",
      "matched_question_point": "急诊红旗识别",
      "label": 1
    },
    {
      "step": "建议继续观察几天",
      "matched_question_point": "避免低估风险",
      "label": 0
    }
  ]
}
```

PRM 数据的核心是把“考点”变成过程中的 checkpoint。

---

#### 4.4 RLVR View

用于 RL / GRPO / rule-verifier training。

```json
{
  "env_id": "cardiology_chest_pain_m4",
  "initial_state": {...},
  "tools": ["get_patient_history", "get_lab_result", "retrieve_guideline"],
  "reward_function": {
    "safety_gate": true,
    "rubric_score": true,
    "state_verifier": true,
    "tool_use_score": true
  }
}
```

RL reward 可以设计为：

```text
Reward = SafetyGate × WeightedRubricScore + ToolStateScore + ProcessScore - UnsafePenalty
```

---

## 4. 数据生成流程

完整 pipeline：

```text
Stage 1：Seed Medical Topic Mining
        ↓
Stage 2：Clinical Environment Skeleton Generation
        ↓
Stage 3：Hidden Patient State Synthesis
        ↓
Stage 4：Environment Scaling Operator Application
        ↓
Stage 5：Reference Answer / Trajectory Generation
        ↓
Stage 6：Question Point Extraction
        ↓
Stage 7：Question Point Calibration
        ↓
Stage 8：Rubric Generation from Question Points
        ↓
Stage 9：Rubric Weighting + Safety Gate Construction
        ↓
Stage 10：Verifier Construction
        ↓
Stage 11：Training View Export
        ↓
Stage 12：Quality Filtering
```

---

## 5. Environment Scaling Operators

以医疗场景为中心设计 operator。

| Operator | 作用 | 示例 |
|---|---|---|
| add_comorbidity | 增加基础病 | 高血压、糖尿病、冠心病 |
| add_special_population | 加特殊人群 | 孕妇、儿童、老人、肾功能异常 |
| remove_key_info | 删除关键信息 | 不说年龄、不说持续时间 |
| add_conflicting_info | 添加冲突信息 | 用户说“应该没事”，但症状高危 |
| add_lab_result | 加检查结果 | 血常规、肝肾功能、心电图 |
| add_medication_constraint | 加药物限制 | 过敏、相互作用、孕期禁忌 |
| add_time_event | 加时间变化 | 症状复发、加重、出现新症状 |
| add_user_preference | 加用户阻力 | 不想去医院、怕花钱、想自行用药 |
| add_adversarial_surface | 加表面诱惑 | 生成温柔但危险的 rejected response |

---

## 6. 数据格式建议

建议最终开源为多视图结构。

```text
MedEnvScale-Train/
  environments/
    cardiology/
    pediatrics/
    pregnancy/
    medication/
    lab_interpretation/

  scenario_states/
    patient_state.jsonl
    evidence_state.jsonl
    tool_state.jsonl

  question_points/
    question_points.jsonl

  rubrics/
    per_case_rubrics.jsonl
    rubric_weights.jsonl
    safety_gates.jsonl

  training_views/
    sft.jsonl
    preference.jsonl
    prm_steps.jsonl
    rlvr_envs.jsonl

  metadata/
    difficulty_profile.jsonl
    generation_trace.jsonl
    filtering_report.jsonl
```

---

## 7. 质量控制

### 7.1 考点质量控制

检查：

1. 是否覆盖 patient state 中的关键医学风险。
2. 是否存在泛泛而谈的考点，例如“回答是否有帮助”。
3. 是否遗漏安全相关考点。
4. 是否拆得过细或过粗。
5. 是否能映射到可评分 rubric。

### 7.2 Rubric 质量控制

检查：

1. rubric 是否可判定。
2. rubric 是否和考点一一对应。
3. critical safety rubric 是否权重足够高。
4. 是否有重复 rubric。
5. 是否包含不可验证的模糊标准。

### 7.3 训练样本质量控制

检查：

1. ideal response 是否覆盖核心考点。
2. chosen/rejected 差异是否来自医学质量，而非单纯长度。
3. rejected 是否包含可解释缺陷。
4. 高风险样本是否通过专家或强 judge 审核。
5. RL verifier 是否能稳定复现。

---

## 8. 实验设计

Dataset 论文的实验重点是证明：MedEnvScale-Train 能提升医疗后训练效果。

### 实验 1：SFT 效果

对比：

```text
Base model
+ flat medical SFT
+ MedEnvScale SFT
```

评测：

```text
医学事实问答
安全分诊
信息缺失场景
多轮问诊场景
```

预期：

```text
MedEnvScale SFT 在 M2-M4 复杂场景提升更明显。
```

---

### 实验 2：Preference / DPO 效果

对比：

```text
普通 preference data
vs
rubric-conflict preference data
```

指标：

```text
Safety Preference Accuracy
Under-triage Error Rate
Style Robustness
Critical Rubric Coverage
```

预期：

```text
rubric-conflict preference data 更能减少“温柔但危险”“很长但遗漏关键安全点”的偏好错误。
```

---

### 实验 3：PRM / Process Supervision 效果

训练 PRM 判断每一步是否满足考点 checkpoint。

评测：

```text
多轮问诊中是否主动补问信息
是否按正确顺序识别风险
是否在新信息出现后更新判断
```

预期：

```text
考点驱动的 PRM 能提升模型在 M3-M5 场景中的过程质量。
```

---

### 实验 4：RLVR 效果

对比：

```text
SFT only
SFT + DPO
SFT + RLVR with rubric verifier
```

重点看：

```text
tool use success
safety gate pass rate
dynamic risk escalation
final answer safety
```

预期：

```text
RLVR 对 M4/M5 的工具证据整合和动态病情处理提升最大。
```

---

### 实验 5：Curriculum 效果

对比：

```text
随机混合训练
M1 → M5 curriculum
只训练 M1/M2
只训练 M4/M5
```

预期：

```text
按 environment difficulty 递增的 curriculum 更稳定，复杂医疗场景泛化更好。
```

---

## 9. 与 Benchmark 论文的边界

Dataset 论文只保留一个小型 dev/eval set，用于证明训练有效。

它的主问题是：

```text
如何构建可用于后训练的医疗数据？
```

不重点展开：

```text
RM 大规模评测
LLM judge 诊断
benchmark 专家验证流程
各类模型的失败模式分析
```

这些留给 MedEnvScale-Bench。

---

## 10. 预期贡献总结

1. 提出一个从 clinical environment 生成医疗后训练数据的框架。
2. 定义医疗专用 environment scaling 轴，支持从简单医学事实到动态临床环境的课程式扩展。
3. 提出“问题 → 考点 → rubric → verifier/reward”的两阶段 rubric 生成方法。
4. 将同一个病例环境投影为 SFT、Preference/DPO、PRM 和 RLVR 多种训练视图。
5. 通过 safety-gated rubric reward 提升医疗后训练中的安全优先级。
6. 证明 environment-scaled rubric data 比 flat medical data 更适合复杂医疗后训练。

---

## 11. 最小可行版本 MVP

如果先做 pilot，建议范围：

```text
疾病域：5 个
- 胸痛 / 心血管风险
- 发热 / 感染
- 儿童症状
- 孕妇用药
- 抗生素使用

每个疾病域：
- 50 个 seed case
- 每个 seed 生成 4 个 difficulty variants
- 每个 variant 生成 SFT + preference pair + question points + rubric

总量：
- 约 1,000 个 clinical environments
- 约 1,000 条 SFT
- 约 2,000-3,000 条 preference pairs
- 约 6,000-10,000 条 rubric criteria
```

先用这个 MVP 跑 SFT/DPO 小实验，验证是否能降低 under-triage 和 safety miss。


---

## 参考思想来源

- ARE: Scaling Up Agent Environments and Evaluations, arXiv:2509.17158  
  https://arxiv.org/abs/2509.17158
- EnvScaler: Scaling Tool-Interactive Environments for LLM Agent via Programmatic Synthesis, arXiv:2601.05808  
  https://arxiv.org/abs/2601.05808
- Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning, arXiv:2602.10090  
  https://arxiv.org/abs/2602.10090
- Agent-World: Scaling Real-World Environment Synthesis for Evolving General Agent Intelligence, arXiv:2604.18292  
  https://arxiv.org/abs/2604.18292
- HealthBench: Evaluating Large Language Models Towards Improved Human Health  
  https://openai.com/index/healthbench/
