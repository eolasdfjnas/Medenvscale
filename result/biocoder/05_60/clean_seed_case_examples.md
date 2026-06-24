# Stage 05 BioCoder Clean Seed Case 示例分析

来源目录：`/home/zengjiaqi/medenvscale/result/biocoder/05`

本文件从 `scaled_envs_clean.jsonl` 中挑了两个质量比较好的 seed 样本组：

- `env_medagentgym_train_835`：proteomics peptide transition 映射。
- `env_medagentgym_train_838`：Deblur CLI command 构造。

选择原因：

- 两个样本组在当前 Stage 05 clean 结果中 M1、M2、M3、M4 都存在。
- 每个等级都有非空 `validated_oracle_cases`。
- `scaled_oracle_case_failures` 为空。
- 扩展方向比较清楚，能看出从原始 seed case 到更强 oracle coverage 的递进。

## 4 个难度轴含义

当前 4-axis 配置中，每个 env 的 `difficulty` 字段会记录 `D/C/A/V` 四个轴的强度。强度越高，表示该轴被加得越多；`total_intensity` 是四个轴强度之和。

| 轴 | 名称 | 含义 |
|---|---|---|
| D | Data / Input Complexity | 增加输入文件、数据结构、格式、缺失值、异常记录、多文件输入、内容解析等复杂度。 |
| C | Computation / Constraint Complexity | 增加算法行为、参数规则、输出契约、边界处理、排序规则、数值约束、多步逻辑等复杂度。 |
| A | Adversarial / Robustness Complexity | 增加对 hardcoding、冲突输入、随机名称、重复参数、误导性元数据、脆弱假设的鲁棒性要求。 |
| V | Verification / Oracle Complexity | 增强 executable oracle cases、expected output signature、文件产物检查、stdout/stderr 检查、异常检查和 coverage。 |

## 样本总览

| Seed 组 | M-level 覆盖 | validated oracle cases | 轴强度概览 | 主要任务类型 | 质量信号 |
|---|---:|---:|---|---|---|
| `env_medagentgym_train_835` | M1-M4 | 12 | M1 `D0/C0/A0/V0`; M2 `D0/C0/A0/V2`; M3 `D0/C2/A0/V2`; M4 `D1/C1/A2/V0` | CSV row 到 proteomics mapping | 多 case 覆盖空 transition、charge 缺省、重复 protein、多行累积 |
| `env_medagentgym_train_838` | M1-M4 | 9 | M1 `D0/C0/A0/V0`; M2 `D1/C0/A0/V0`; M3 `D1/C0/A1/V0`; M4 `D1/C2/A2/V1` | Deblur CLI command builder | 覆盖特殊文件名、固定 flag 去重、缺失文件、冲突 flag、system call error |

## Seed 样本 1：`env_medagentgym_train_835`

### 原始 M1 题目

题目要求实现：

```python
def mapRow(this_row, header_dict, precursors_mapping, sequences_mapping, protein_mapping)
```

函数处理 proteomics peptide CSV 的一行数据。输入行可能包含：

- `FullPeptideName`：peptide sequence。
- `aggr_Fragment_Annotation`：用分号分隔的 fragment transitions。
- `aggr_prec_Fragment_Annotation`：用分号分隔的 precursor transitions。
- `Charge`：charge state。
- `ProteinName`：protein identifier。

函数需要原地更新三个 mapping：

- `precursors_mapping`
- `sequences_mapping`
- `protein_mapping`

M1 validated case：

```text
seed_case_main
```

M1 轴与强度：

```text
selected_axes: []
D=0, C=0, A=0, V=0, total_intensity=0
applied_operators: []
```

M1 gold 核心逻辑：

```python
def mapRow(this_row, header_dict, precursors_mapping, sequences_mapping,
           protein_mapping):
    if 'FullPeptideName' in header_dict:
        peptide_name = this_row[header_dict['FullPeptideName']]
        transitions = []
        pr_transitions = []
        if 'aggr_Fragment_Annotation' in header_dict:
            transitions = this_row[header_dict['aggr_Fragment_Annotation']].split(';')
        if 'aggr_prec_Fragment_Annotation' in header_dict:
            pr_transitions = this_row[header_dict['aggr_prec_Fragment_Annotation']].split(';')
        if len(transitions) == 0:
            return
        if len(transitions[-1]) == 0:
            transitions = transitions[:-1]
        if len(pr_transitions) > 0 and len(pr_transitions[-1]) == 0:
            pr_transitions = pr_transitions[:-1]
        charge_state = '0'
        if 'Charge' in header_dict:
            charge_state = this_row[header_dict['Charge']]
        key = peptide_name + '/' + charge_state
        prkey = peptide_name + '/' + charge_state + '_pr'
        precursors_mapping[key] = transitions
        precursors_mapping[prkey] = pr_transitions
        sequences_mapping[peptide_name] = [key, prkey]
        if 'ProteinName' in header_dict:
            protein_name = this_row[header_dict['ProteinName']]
            protein_mapping[protein_name] = [peptide_name]
```

### M2 扩展

轴与强度：

```text
selected_axes: [V]
D=0, C=0, A=0, V=2, total_intensity=2
applied_operators:
- env_medagentgym_train_835_M2_v_01
```

新增重点：

- 覆盖 empty transitions。
- 覆盖 `Charge == "NA"` 和空 charge。
- 覆盖 duplicate protein。
- 覆盖多行调用。
- 覆盖 trailing semicolon。

M2 validated cases：

```text
scaled_seed_case_main
test_empty_transitions
test_charge_na
test_charge_empty
test_duplicate_protein
test_multiple_rows
test_trailing_semicolons
```

M2 gold 的关键增强：

```python
if len(transitions) > 0 and len(transitions[-1]) == 0:
    transitions = transitions[:-1]
if len(pr_transitions) > 0 and len(pr_transitions[-1]) == 0:
    pr_transitions = pr_transitions[:-1]
if len(transitions) == 0:
    return

charge_state = '0'
if 'Charge' in header_dict:
    charge_state = this_row[header_dict['Charge']]
if charge_state == 'NA' or charge_state == '':
    charge_state = '0'

mapped_precursors = sequences_mapping.get(peptide_name, [])
if key not in mapped_precursors:
    mapped_precursors.append(key)
if prkey not in mapped_precursors:
    mapped_precursors.append(prkey)
sequences_mapping[peptide_name] = mapped_precursors
```

评价：

- M2 的扩展非常自然，主要加强 V 轴 coverage。
- case 数从 1 个增加到 7 个，能明显防止只覆盖 happy path 的实现。
- `Charge` 缺省和重复项处理都比较贴合真实 CSV 数据。

### M3 扩展

轴与强度：

```text
selected_axes: [C, V]
D=0, C=2, A=0, V=2, total_intensity=4
applied_operators:
- env_medagentgym_train_835_M3_c_01
- env_medagentgym_train_835_M3_c_02
- env_medagentgym_train_835_M3_v_03
- env_medagentgym_train_835_M3_v_04
```

新增重点：

- fragment annotation 需要去重。
- peptide name 和 charge 需要 `strip()` 清洗。
- 空 transition 要 early return。
- default charge handling 要被 oracle 明确检查。

M3 validated case：

```text
scaled_seed_case_main
```

M3 gold 的关键增强：

```python
peptide_name = this_row[header_dict['FullPeptideName']].strip()

raw = this_row[header_dict['aggr_Fragment_Annotation']].split(';')
if raw and raw[-1] == '':
    raw = raw[:-1]
transitions = list(dict.fromkeys(raw))

raw = this_row[header_dict['aggr_prec_Fragment_Annotation']].split(';')
if raw and raw[-1] == '':
    raw = raw[:-1]
pr_transitions = list(dict.fromkeys(raw))

if not any(transitions):
    return

charge_state = this_row[header_dict['Charge']].strip()
if charge_state == 'NA' or charge_state == '':
    charge_state = '0'
```

评价：

- M3 不只是增加 case 数，而是加入了更强的语义约束：清洗、去重、空 transition 判定。
- `dict.fromkeys` 保留原顺序去重，这对 transition list 比较合适。
- 当前 M3 只有一个 validated case，但这个 case 覆盖多个 operator，适合作为“复合约束 case”的示例。

### M4 扩展

轴与强度：

```text
selected_axes: [C, A, D]
D=1, C=1, A=2, V=0, total_intensity=4
applied_operators:
- env_medagentgym_train_835_M4_d_01
- env_medagentgym_train_835_M4_c_02
- env_medagentgym_train_835_M4_a_03
- env_medagentgym_train_835_M4_a_04
```

新增重点：

- row 中可以有额外列，函数不能被额外列干扰。
- 多次调用时要检查 mapping 的累积和 dedup。
- `Charge` 列缺失时默认 `0`。
- `ProteinName` 列缺失时不能报错。

M4 validated cases：

```text
scaled_case_extra_and_duplicate
scaled_case_missing_charge
scaled_case_missing_protein
```

M4 gold 的关键增强：

```python
if 'FullPeptideName' not in header_dict:
    return

charge_state = '0'
if 'Charge' in header_dict:
    charge_state = this_row[header_dict['Charge']]
    if charge_state == 'NA' or charge_state == '':
        charge_state = '0'

if peptide_name in sequences_mapping:
    sequences_mapping[peptide_name].extend([key, prkey])
else:
    sequences_mapping[peptide_name] = [key, prkey]

if 'ProteinName' in header_dict:
    protein_name = this_row[header_dict['ProteinName']]
    if protein_name in protein_mapping:
        if peptide_name not in protein_mapping[protein_name]:
            protein_mapping[protein_name].append(peptide_name)
    else:
        protein_mapping[protein_name] = [peptide_name]
```

### 难度递进总结

| 等级 | 主要变化 | 好的信号 | 注意点 |
|---|---|---|---|
| M1 | 基础 row mapping | 结构简单，seed case 可执行 | 对缺失列、重复项、空值覆盖弱 |
| M2 | 多 case 覆盖常见脏数据 | case 数足够多，边界清晰 | 主要是 coverage 增强 |
| M3 | 清洗、去重、默认值 | 语义约束更强 | case 数少但复合约束较多 |
| M4 | 额外列、缺失列、多次调用 | 更接近真实 CSV ingestion | `sequences_mapping` 仍可能追加重复 key，后续可继续增强 |

这个样本组适合作为“从 seed case 扩展到 robust data-row mapper”的正例。

## Seed 样本 2：`env_medagentgym_train_838`

### 原始 M1 题目

题目要求实现：

```python
def deblur_system_call(params, input_fp)
```

函数需要构造 Deblur CLI command：

- command 以 `['deblur', 'workflow']` 开头。
- 固定包含 `--seqs-fp input_fp`。
- 固定包含 `--is-worker-thread`。
- 固定包含 `--keep-tmp-files`。
- 追加 `params` 中的额外参数。
- 通过 `_system_call(command)` 执行。

M1 validated case：

```text
seed_case_main
```

M1 轴与强度：

```text
selected_axes: []
D=0, C=0, A=0, V=0, total_intensity=0
applied_operators: []
```

M1 gold 核心逻辑：

```python
def deblur_system_call(params, input_fp):
    logger = logging.getLogger(__name__)
    logger.debug('[%s] deblur system call params %s, input_fp %s' % (
        mp.current_process().name, params, input_fp
    ))
    script_name = 'deblur'
    script_subprogram = 'workflow'
    command = [
        script_name,
        script_subprogram,
        '--seqs-fp',
        input_fp,
        '--is-worker-thread',
        '--keep-tmp-files',
    ]
    command.extend(params)
    logger.debug('[%s] running command %s' % (mp.current_process().name, command))
    return _system_call(command)
```

### M2 扩展

轴与强度：

```text
selected_axes: [D]
D=1, C=0, A=0, V=0, total_intensity=1
applied_operators:
- env_medagentgym_train_838_M2_d_01
```

新增重点：

- 输入 FASTA 文件名可以包含空格，例如 `my sequences.fasta`。
- oracle 检查 command list 中路径作为一个完整参数出现，而不是被拆分。

M2 validated case：

```text
scaled_case_special_chars
```

评价：

- 这是 D 轴输入复杂度增强。
- 该扩展很干净：没有改变函数签名，只要求 command list 构造正确。
- 对 CLI 类任务来说，这是非常实用的 case。

### M3 扩展

轴与强度：

```text
selected_axes: [D, A]
D=1, C=0, A=1, V=0, total_intensity=2
applied_operators:
- env_medagentgym_train_838_M3_d_01
- env_medagentgym_train_838_M3_a_02
```

新增重点：

- 读取 FASTA 文件并统计 sequence 数量。
- logger 中记录 sequence count。
- 如果 `params` 已经包含 mandatory flags，不应重复添加。

M3 validated cases：

```text
scaled_seed_case_main
scaled_coverage_dup_flag
```

M3 gold 的关键增强：

```python
sequence_count = 0
with open(input_fp, 'r') as f:
    for line in f:
        if line.startswith('>'):
            sequence_count += 1
logger.debug('Number of sequences in %s: %d', input_fp, sequence_count)

command = [script_name, script_subprogram, '--seqs-fp', input_fp]
if '--is-worker-thread' not in params:
    command.append('--is-worker-thread')
if '--keep-tmp-files' not in params:
    command.append('--keep-tmp-files')
command.extend(params)
```

评价：

- M3 从单纯 command construction 变成了带输入文件检查和 logging semantics 的工具函数。
- `scaled_coverage_dup_flag` 可以防止重复 flag，是很好的 adversarial-style check。
- 这里的扩展比 M2 明显更强，难度递进比较合理。

### M4 扩展

轴与强度：

```text
selected_axes: [D, C, V, A]
D=1, C=2, A=2, V=1, total_intensity=6
applied_operators:
- env_medagentgym_train_838_M4_d_01
- env_medagentgym_train_838_M4_c_02
- env_medagentgym_train_838_M4_c_03
- env_medagentgym_train_838_M4_a_04
- env_medagentgym_train_838_M4_a_05
- env_medagentgym_train_838_M4_v_06
```

新增重点：

- 缺失输入文件要抛 `FileNotFoundError`。
- `params` 中不能覆盖固定 flag，例如 `--seqs-fp`。
- logger 只能记录 basename，避免泄露完整路径。
- system call error 要被保留并返回。
- oracle 需要通过 mock 检查 command list structure。

M4 validated cases：

```text
scaled_seed_case_main
ext_missing_file
ext_conflicting_flag
ext_error_handling
ext_command_structure_empty_params
```

M4 gold 的关键增强：

```python
if not os.path.isfile(input_fp):
    raise FileNotFoundError(f"Input file {input_fp} does not exist.")

safe_fp = os.path.basename(input_fp)
logger.debug(
    '[%s] deblur system call params %s, input_fp %s',
    mp.current_process().name,
    params,
    safe_fp,
)

command = [
    script_name,
    script_subprogram,
    '--seqs-fp',
    input_fp,
    '--is-worker-thread',
    '--keep-tmp-files',
]

for item in params:
    if item.startswith('--'):
        flag = item.split('=')[0]
        if flag in {'--seqs-fp', '--is-worker-thread', '--keep-tmp-files'}:
            raise ValueError(f"Flag {item} is a fixed flag and cannot be overridden.")

sanitized_params = [param.strip() for param in params]
command.extend(sanitized_params)
return _system_call(command)
```

### 难度递进总结

| 等级 | 主要变化 | 好的信号 | 注意点 |
|---|---|---|---|
| M1 | 基础 Deblur command construction | command list 结构明确 | 只覆盖 happy path |
| M2 | 特殊文件名路径 | 防止错误拆分路径 | 仍是单 case |
| M3 | sequence count logging、mandatory flag 去重 | 更贴近真实 wrapper 行为 | 需要读文件，任务复杂度上升 |
| M4 | missing file、conflicting flag、mock command、error handling | case 覆盖很强 | 对 agent 来说要求较多，容易在精确错误信息上失败 |

这个样本组适合作为“CLI wrapper 从基础构造到鲁棒参数校验”的正例。

## 为什么这两个样本比较适合做 clean 示例

1. M1-M4 链条完整，适合观察难度递进。
2. oracle case 都是 executable case，不是纯文本 rubric。
3. 任务本身有明确工程语义：
   - `mapRow` 是数据清洗和 mapping 更新。
   - `deblur_system_call` 是 CLI wrapper 和参数安全。
4. 扩展后的需求能落到具体 case：
   - 空值、重复、缺失列。
   - 特殊路径、重复 flag、缺失文件、错误处理。
5. gold code 和 oracle case 基本能互相解释，不是“为了过 case 而硬造”的不自然扩展。

## 后续可改进点

- `env_medagentgym_train_835_M4` 可以进一步要求 `sequences_mapping` 中 `[key, prkey]` 不重复，当前 gold 更偏向追加行为。
- `env_medagentgym_train_838_M4` 对错误消息和 fixed flag 的精确要求较强，后续如果用于 agent RL，可以考虑把错误类型作为主要信号、错误文本作为次要信号。
- 两个样本都适合加入 pass-rate calibration：如果某个模型在 M1 失败、M2/M3 反而通过，应检查是否是 seed case 过于细碎或 stdout/return 格式过严。
