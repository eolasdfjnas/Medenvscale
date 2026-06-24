# MedEnvScale-MedAgentGym-BioCoder：当前实验方案

## 0. 实验定位

**当前实验类型：**

> Executable Dataset Construction + Dynamic Difficulty Scaling Pipeline

**一句话概括：**

> 当前方案以 MedAgentGym / BioCoder 的可执行代码任务为 seed 数据源，先做代码可执行性治理，再构建可执行 environment，最后通过 7 轴动态 operator 对任务进行 M1-M4 难度扩展，并导出可用于后续训练与评测的结构化数据。

---

## 1. 核心目标

当前这套实验方案主要解决三个问题：

1. **把原始 BioCoder 任务变成可执行 seed 数据**  
   不是直接拿原始 `problem -> solution` 进入后续阶段，而是先确认任务代码本身能运行、能产生可观察输出，并且能被后续 pipeline 稳定消费。

2. **把 seed task 变成可扩展难度的 executable environment**  
   不是只保留自然语言题面，而是把任务组织成带 `problem`、`context`、`signature`、`code`、`ground_truth_output_signature` 等字段的环境实例。

3. **围绕 7 个轴做可验证的难度扩展**  
   Stage 05 不是简单改 metadata，而是对环境做动态 operator scaling，使任务从原始难度 `M1` 扩展到 `M2 / M3 / M4`，并尝试为扩展后的任务生成对应的验证样例与可执行代码。

---

## 2. 当前方案的总体思路

当前方案已经不再是最早的“围绕自然语言 gold answer 做评测”的流程，而是逐步转向：

```text
可执行代码
+ 可观察输出
+ 动态难度扩展
+ 结构化验证样例
→ 训练 / 评测数据
```

更具体地说，当前主线是：

```text
BioCoder raw task
→ 执行与 repair
→ 标准化任务
→ 路由与环境构建
→ 7 轴动态 scaling
→ 导出后续训练视图
```

---

## 3. 当前数据与目录组织

当前项目按数据集子目录组织产物。以 `biocoder` 为例：

```text
configs/biocoder/
data/biocoder/
result/biocoder/
experiments/biocoder/
```

数据输入来自：

```text
/archive/zengjiaqi/dataset/medagentgym/biocoder/
```

其中公共配置仍保留在根目录，例如：

```text
configs/llm.yaml
```

当前实验的关键配置入口包括：

```text
configs/biocoder/medagentgym_pilot.yaml
configs/biocoder/tool_pool.yaml
```

---

## 4. 当前 Pipeline 分阶段说明

### Stage 00：Prepare

这一阶段的目标不是简单搬运原始数据，而是先治理 BioCoder 任务代码的可执行性。

当前逻辑是：

```text
读取 raw task
→ 提取 code
→ 在隔离环境里执行
→ 保存标准化 output signature
→ 若失败则进入 LLM repair loop
→ 最多 repair 3 次
→ 成功后用 repaired code 覆盖原 code
→ 原始失败代码保存到 wrong_code
```

这一阶段最关键的输出是：

```text
ground_truth_output_signature = {
  return_value,
  stdout,
  file_artifacts
}
```

也就是说，Stage 00 建立的是“可执行 seed code + 可观察 ground truth”。

---

### Stage 01：Normalize

这一阶段负责把前面处理过的任务统一整理成标准字段表示。

典型保留下来的字段包括：

```text
problem
context
signature
code
wrong_code
ground_truth
ground_truth_output_signature
split
```

它的作用是把 raw task 变成后续阶段都能稳定读取的标准任务对象。

---

### Stage 02：Route

这一阶段主要做任务路由与标签补全，为后面的动态扩展做准备。

当前主要会识别和组织：

```text
primary_domain
secondary_domains
primary_task_type
secondary_task_types
solution_form
```

这一阶段的作用是让后续 operator 生成时知道“这条任务属于哪类任务、适合哪种扩难策略”。

---

### Stage 03：Generate Seed Cases

这一阶段的目标是把标准化任务转成 seed task / seed env 的前置形态。

它会继续把前面得到的可执行信息一路带下去，例如：

```text
code
ground_truth_output_signature
```

这样到后面的 environment 阶段，就不只是有自然语言题面，而是已经有基础执行语义。

---

### Stage 04：Generate Environment Skeletons

这一阶段会把 seed task 进一步组织成 executable environment skeleton。

可以理解为，这里开始形成 Stage 05 真正要使用的环境对象，里面会包含：

```text
problem
context
signature
code
gold_solution / seed_gold_solution
seed_ground_truth_output_signature
visible_state
task_state
verifier_state
test_state
```

这一阶段的目标是：

> 让后续难度扩展不是对“纯文本题目”做操作，而是对一个结构化、可执行的环境实例做操作。

---

### Stage 05：Apply Scaling Operators

这是当前实验最核心的阶段。

这一阶段的总体流程是：

```text
对 seed environment 做 7 轴权重规划
→ 生成 tool config
→ 生成 dynamic operator instances
→ 把任务从 M1 扩展到 M2 / M3 / M4
→ 为扩展后的任务生成验证样例和 scaled code
→ 执行验证与 repair
→ 输出 clean / rejected 结果
```

当前你最关注的，也主要是这一阶段。

这里的核心对象包括：

```text
operator_instances
scaled_oracle_cases
scaled_executable_gold_code
scaled_ground_truth_output_signature
quality_report / gate results
```

目前这条链路正在从“hidden test 主导验证”逐步收敛到：

```text
scaled_oracle_cases
→ 直接执行 case
→ 对比 expected output
→ 把失败信息反馈给 repair loop
```

---

### Stage 06-10：下游导出与过滤

Stage 05 之后，pipeline 还会继续做：

```text
Stage 06: qpoints / rubrics / answers
Stage 07: safety gates
Stage 08: export training views
Stage 09: quality filter
Stage 10: make splits
```

这些阶段的职责是把 Stage 05 产出的环境实例继续整理成可供训练和评测使用的视图，例如：

```text
SFT
DPO
PRM
RLVR
```

---

## 5. 7 个轴的含义

当前动态难度扩展围绕 7 个轴展开。它们不是简单标签，而是 7 个“任务变难的方向”。

| 轴 | 含义 | 当前理解 |
|---|---|---|
| H | Horizon | 增加任务步骤深度、处理链条和中间操作负担 |
| R | Reliability / Risk | 增加结果可靠性、风险敏感性、单位与数值合理性等要求 |
| I | Interpretation | 增加题意理解难度、减少歧义容忍度、强化正确解释要求 |
| E | Execution | 增加执行环境与输入载体复杂度，例如路径、文件、数据形态变化 |
| C | Constraint | 增加边界条件、异常路径、组合限制和额外约束 |
| A | Adversarial / Robustness | 增加防 shortcut、防 hardcode、对抗性或鲁棒性挑战 |
| V | Verification | 增加可验证性要求，要求更多、更强的测试或验证覆盖 |

这 7 个轴共同决定某个任务在 Stage 05 会被扩成什么样的 `M2 / M3 / M4` 样本。

---

## 6. 当前实验的核心产物

从当前实验视角看，最重要的中间产物包括：

1. **Stage 00 的可执行 seed code 与 ground truth output signature**  
   它保证原始数据不是“只看起来像代码”，而是真能跑。

2. **Stage 03-04 的 executable seed environment**  
   它保证后续 scaling 是对结构化环境做，而不是对裸题面做。

3. **Stage 05 的动态 operator 与 scaled 样本**  
   它决定最终数据是否真的形成了难度梯度。

4. **Stage 05 的验证信息与质量报告**  
   它决定哪些样本能进入 clean，哪些必须 rejected。

5. **Stage 08-10 的训练视图与 split**  
   它决定最终这套数据如何进入后训练与评测。

---

## 7. 当前方案的重点与难点

从目前项目状态看，实验重点主要集中在两个位置：

### 7.1 Stage 00

重点是：

```text
把原始 BioCoder code 清成“可执行、可观察、可复现”的 seed 数据
```

这一步不稳，后面所有阶段都会被污染。

### 7.2 Stage 05

重点是：

```text
让 operator 真正把任务语义变难
+ 让扩展后的任务能被验证
+ 让生成的 scaled code 能被执行和 repair
```

这里也是当前实验复杂度最高、迭代最多的地方。

---

## 8. 当前方案的一句话总结

可以把当前实验方案概括为：

> 先把 BioCoder 原始代码任务清洗成可执行的 seed environment，再基于 7 轴动态 operator 对环境做 M1-M4 难度扩展，并尝试为扩展后的任务生成可执行代码与结构化验证样例，最终导出可用于后训练与评测的数据视图。
