# Stage 05 BioCoder Clean Seed Case 示例分析

来源目录：`/home/zengjiaqi/medenvscale/result/biocoder/05`

本文件从 clean 结果中挑了两个 seed 样本组：

- `env_medagentgym_train_879`：single-cell read ID 解析。
- `env_medagentgym_train_441`：metabolic exchange file 解析。

这两个样本组在本轮结果中 M1、M2、M3、M4 都是 clean。下面分别列出原题、M2/M3/M4 扩展后的题面要求、对应 gold 答案，以及难度扩展前后的对比。

## Seed 样本 1：`env_medagentgym_train_879`

### 原始 M1 题目

题目要求实现 `get_cell_umi_read_string(read_id, sep='_')`。在 single-cell RNA-seq 的 read identifier 中，最后两个由分隔符分开的字段分别表示 cell barcode 和 UMI。函数需要返回 `(UMI_bytes, cell_barcode_bytes)`。如果 `read_id` 里没有足够的分隔符，应该抛出 `ValueError`。

原始 scaffold：

```python
def main():
    read_id_good = 'NB551068:51:H3JTYBGXX:1:11101:10000:1001_CB10123_UMIACTGGA'
    read_id_bad = 'NB551068:51:H3JTYBGXX:1:11101:10000:1001'
    sep = '_'

    # <<insert solution here>>

    umi, cell_barcode = get_cell_umi_read_string(read_id_good, sep)
    print("Good read -> UMI bytes:", umi, "Cell barcode bytes:", cell_barcode)

    try:
        get_cell_umi_read_string(read_id_bad, sep)
    except ValueError as e:
        print("Expected error for bad read ID:", e)

if __name__ == "__main__":
    main()
```

M1 gold 答案：

```python
def get_cell_umi_read_string(read_id, sep='_'):
    """ extract the umi and cell barcode from the read id (input as a
    string) using the specified separator """
    try:
        return read_id.split(sep)[-1].encode('utf-8'), read_id.split(sep)[-2].encode('utf-8')
    except IndexError:
        raise ValueError(
            'Could not extract UMI or CB from the read ID, pleasecheck UMI and CB are encoded in the read name:%s'
            % read_id)
```

### M2 扩展题

新增要求：

- 函数必须能使用多字符分隔符切分 read ID。
- 返回值仍然是最后两个字段对应的 bytes。
- oracle case 至少要包含一个多字符分隔符，例如 `::`。
- 难度主要体现在输入格式变复杂，不能只假设分隔符是 `_` 这类单字符。

M2 gold 答案：

```python
def get_cell_umi_read_string(read_id, sep='_'):
    """ extract the umi and cell barcode from the read id (input as a
    string) using the specified separator """
    parts = read_id.split(sep)
    if len(parts) < 2:
        raise ValueError(
            'Could not extract UMI or CB from the read ID, pleasecheck UMI and CB are encoded in the read name:%s' % read_id)
    return parts[-1].encode('utf-8'), parts[-2].encode('utf-8')
```

和 M1 的对比：

- M1 虽然也用了 `split(sep)`，但错误处理依赖 `IndexError`。
- M2 显式检查字段数量，分隔符处理更清楚。
- 这个扩展是比较自然的 D 轴输入复杂度增强。

### M3 扩展题

新增要求：

- 支持多字符分隔符。
- read ID 前半部分可以包含额外分隔符，但函数必须始终取最后两个字段。
- 函数签名增加可选参数 `encoding='utf-8'`。
- 如果提取出来的 barcode 或 UMI 为空，必须抛出 `ValueError`。

M3 gold 答案：

```python
def get_cell_umi_read_string(read_id, sep='_', encoding='utf-8'):
    """Extract UMI and cell barcode from read_id using given separator.
    Returns tuple (UMI_bytes, cell_barcode_bytes) encoded with specified encoding.
    Raises ValueError if read_id has fewer than two separators or if either field is empty."""
    try:
        fields = read_id.split(sep)
        umi = fields[-1]
        cb = fields[-2]
        if not umi or not cb:
            raise ValueError(
                'One of the required fields (UMI or CB) is empty in read ID: %s' % read_id)
        return umi.encode(encoding), cb.encode(encoding)
    except IndexError:
        raise ValueError(
            'Could not extract UMI or CB from the read ID, please check UMI and CB are encoded in the read name: %s'
            % read_id)
```

oracle 覆盖情况：

- 一个 validated oracle case 覆盖了 M3 的 4 个 operator。
- case 同时检查默认分隔符、多字符分隔符、额外分隔符、自定义 encoding、空字段和分隔符不足的错误路径。

和 M2 的对比：

- M3 增加了真实的接口变化：`encoding='utf-8'`。
- 增加了非空字段检查，语义约束更强。
- 对“取最后两个字段”的规则验证更充分。
- 这是这个样本组里效果最好的扩展，题面、oracle case、gold 答案和 checker 都比较一致。

### M4 扩展题

新增要求：

- 不能 hardcode `_`，必须始终使用传入的 `sep`。
- 无论 read ID 前面有多少字段，都必须取最后两个字段。
- 返回值必须是两个 bytes 对象组成的 tuple。
- `ValueError` 信息必须包含原始 `read_id`，方便 debug。
- oracle 要覆盖正常输入、边界输入和 adversarial separator 场景。

M4 gold 答案：

```python
def get_cell_umi_read_string(read_id, sep='_'):
    """ extract the umi and cell barcode from the read id (input as a
    string) using the specified separator """
    parts = read_id.split(sep)
    if len(parts) < 3:
        raise ValueError(
            'Could not extract UMI or CB from the read ID, pleasecheck UMI and CB are encoded in the read name:%s'
            % read_id
        )
    umi_str = parts[-1]
    cb_str = parts[-2]
    return (umi_str.encode('utf-8'), cb_str.encode('utf-8'))
```

和 M3 的对比：

- M4 更强调 adversarial coverage、输出类型约束和错误信息约束。
- 生成的 gold 答案比 M3 简单，没有保留 M3 的 `encoding` 参数。
- 它能通过 M4 validated oracle case，但从语义丰富度看，M3 的答案更完整。

### 难度递进总结

| 等级 | 主要变化 | 好的信号 | 注意点 |
|---|---|---|---|
| M1 | 基础的最后两字段提取 | 简单 baseline 可执行 | 错误处理依赖索引异常 |
| M2 | 支持多字符分隔符 | 输入复杂度提升自然 | 还没有空字段检查 |
| M3 | 多分隔符、额外字段、encoding、空字段检查 | 语义扩展最完整 | 签名更复杂 |
| M4 | anti-hardcoding 和更强 oracle 覆盖 | adversarial separator 覆盖较好 | gold 比 M3 简化 |

## Seed 样本 2：`env_medagentgym_train_441`

### 原始 M1 题目

题目要求实现 `parse_exchange_file(path, default_compartment)`。函数需要读取 metabolic modeling 中的 exchange reaction 文件，文件可能是 TSV 或 YAML。每个 entry 应返回包含 `compound`、`lower`、`upper` 的字典；YAML 格式还要包含 `compartment`。格式需要自动识别：TSV 调用 `parse_exchange_table_file(f)`，YAML 调用 `parse_exchange_yaml_file(context, f, default_compartment)`。如果无法识别格式，需要抛出 `ParseError`。

原始 scaffold 摘要：

```python
import logging
import os

class FilePathContext:
    ...

def resolve_format(fmt, filepath):
    ...

def parse_exchange_table_file(f):
    ...

def parse_exchange_yaml_file(context, f, default_compartment):
    ...

class ParseError(Exception):
    pass

logger = logging.getLogger(__name__)

# <<insert solution here>>
```

M1 gold 答案：

```python
def parse_exchange_file(path, default_compartment):
    """Parse a file as a list of exchange compounds with flux limits.

    The file format is detected and the file is parsed accordingly. Path can
    be given as a string or a context.
    """
    context = FilePathContext(path)
    format = resolve_format(None, context.filepath)
    if format == 'tsv':
        logger.debug('Parsing exchange file {} as TSV'.format(context.filepath))
        with context.open('r') as f:
            for entry in parse_exchange_table_file(f):
                yield entry
    elif format == 'yaml':
        logger.debug('Parsing exchange file {} as YAML'.format(context.filepath))
        with context.open('r') as f:
            for entry in parse_exchange_yaml_file(context, f, default_compartment):
                yield entry
    else:
        raise ParseError('Unable to detect format of exchange file {}'.format(context.filepath))
```

### M2 扩展题

新增要求：

- TSV 和 YAML 都要跳过以 `#` 开头的注释行。
- 每个 entry 都要检查 `lower <= upper`。
- 如果 `lower > upper`，抛出带描述信息的 `ParseError`。
- oracle case 要包含注释行和非法 bound 条目。

M2 gold 答案：

```python
def parse_exchange_table_file(f):
    header = f.readline()
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split('\t')
        if len(parts) != 3:
            continue
        lower = float(parts[1])
        upper = float(parts[2])
        if lower > upper:
            raise ParseError(f"Invalid flux bounds for {parts[0]}: lower {lower} > upper {upper}")
        yield {'compound': parts[0], 'lower': lower, 'upper': upper}

def parse_exchange_yaml_file(context, f, default_compartment):
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split(':')
        if len(parts) < 3:
            continue
        lower = float(parts[1])
        upper = float(parts[2])
        if lower > upper:
            raise ParseError(f"Invalid flux bounds for {parts[0]}: lower {lower} > upper {upper}")
        entry = {
            'compound': parts[0],
            'lower': lower,
            'upper': upper,
            'compartment': parts[3] if len(parts) > 3 else default_compartment
        }
        yield entry

def parse_exchange_file(path, default_compartment):
    context = FilePathContext(path)
    format = resolve_format(None, context.filepath)
    if format == 'tsv':
        logger.debug('Parsing exchange file {} as TSV'.format(context.filepath))
        with context.open('r') as f:
            for entry in parse_exchange_table_file(f):
                yield entry
    elif format == 'yaml':
        logger.debug('Parsing exchange file {} as YAML'.format(context.filepath))
        with context.open('r') as f:
            for entry in parse_exchange_yaml_file(context, f, default_compartment):
                yield entry
    else:
        raise ParseError('Unable to detect format of exchange file {}'.format(context.filepath))
```

和 M1 的对比：

- 从“只负责分发解析器”升级为“解析时带语义校验”。
- 注释行处理让输入更接近真实文件。
- bound validation 是很清楚的 C 轴约束。

### M3 扩展题

新增要求：

- 保留 `lower <= upper` 检查。
- 当扩展名缺失或不明确时，根据文件内容 fallback 判断格式。
- YAML 中少于 3 个冒号字段的 malformed line 要跳过，并输出 warning，不能让整个解析失败。
- 不能 hardcode seed-only 文件名或只处理 happy path。

M3 gold 答案：

```python
def parse_exchange_yaml_file(context, f, default_compartment):
    for line in f:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        parts = stripped.split(':')
        if len(parts) < 3:
            logger.warning('Skipping malformed YAML line: %s', stripped)
            continue
        entry = {
            'compound': parts[0],
            'lower': float(parts[1]),
            'upper': float(parts[2]),
            'compartment': parts[3] if len(parts) > 3 else default_compartment
        }
        yield entry

def parse_exchange_file(path, default_compartment):
    context = FilePathContext(path)
    format = resolve_format(None, context.filepath)
    if format is None:
        with context.open('r') as f:
            first_line = f.readline()
            if '\t' in first_line:
                format = 'tsv'
            elif ':' in first_line:
                format = 'yaml'
            else:
                raise ParseError('Unable to detect format of exchange file {}'.format(context.filepath))
    if format == 'tsv':
        logger.debug('Parsing exchange file {} as TSV'.format(context.filepath))
        with context.open('r') as f:
            for entry in parse_exchange_table_file(f):
                if entry['lower'] > entry['upper']:
                    raise ParseError('lower bound {} greater than upper bound {} for compound {}'.format(
                        int(entry['lower']), int(entry['upper']), entry['compound']))
                yield entry
    elif format == 'yaml':
        logger.debug('Parsing exchange file {} as YAML'.format(context.filepath))
        with context.open('r') as f:
            for entry in parse_exchange_yaml_file(context, f, default_compartment):
                if entry['lower'] > entry['upper']:
                    raise ParseError('lower bound {} greater than upper bound {} for compound {}'.format(
                        int(entry['lower']), int(entry['upper']), entry['compound']))
                yield entry
    else:
        raise ParseError('Unable to detect format of exchange file {}'.format(context.filepath))
```

和 M2 的对比：

- 不只是校验数值，还增强了格式鲁棒性。
- 无扩展名或扩展名不明确时，可以通过内容判断 TSV/YAML。
- malformed YAML line 不再直接导致失败，而是被跳过并记录 warning。

### M4 扩展题

新增要求：

- 即使扩展名误导，也要通过内容判断格式。
- 检查 `lower <= upper`；非法顺序抛出 `ValueError`。
- 空行要静默跳过。
- 空文件必须抛出 `ParseError`。
- oracle 包含 misleading extension 和 empty file 这类 adversarial case。

M4 gold 答案：

```python
def parse_exchange_file(path, default_compartment):
    context = FilePathContext(path)
    with context.open('r') as f:
        first_line = f.readline()
        if not first_line:
            raise ParseError('Empty exchange file')
        f.seek(0)
        if first_line.strip() == 'compound\tlower\tupper':
            format = 'tsv'
        else:
            format = 'yaml'
    if format == 'tsv':
        with context.open('r') as f:
            for entry in parse_exchange_table_file(f):
                if entry['lower'] > entry['upper']:
                    raise ValueError('Lower bound {} > upper bound {} for compound {}'.format(
                        entry['lower'], entry['upper'], entry['compound']))
                yield entry
    else:
        with context.open('r') as f:
            for entry in parse_exchange_yaml_file(context, f, default_compartment):
                if entry['lower'] > entry['upper']:
                    raise ValueError('Lower bound {} > upper bound {} for compound {}'.format(
                        entry['lower'], entry['upper'], entry['compound']))
                yield entry
```

和 M3 的对比：

- M4 更强调 adversarial 文件格式行为。
- 不再信任文件扩展名，而是使用内容判断。
- 增加了空文件错误处理。
- 这份答案能通过 M4 validated oracle case，不过格式判断仍然是比较紧的启发式规则：只把首行等于 TSV header 的文件判断为 TSV，其余都当 YAML。

### 难度递进总结

| 等级 | 主要变化 | 好的信号 | 注意点 |
|---|---|---|---|
| M1 | 基于扩展名分发 TSV/YAML parser | baseline 清晰 | 没有语义校验和鲁棒性 |
| M2 | 注释行处理 + bound validation | C 轴约束明确 | 校验逻辑部分放进 helper |
| M3 | 内容 fallback + malformed YAML 容错 | 鲁棒性明显增强 | 仍是首行判断启发式 |
| M4 | misleading extension + empty file adversarial case | A/D 轴覆盖更强 | 内容检测不算完全泛化 |

## 两个 clean 样本组的横向对比

| Seed 样本组 | 表现最好的扩展 | 为什么效果比较好 |
|---|---|---|
| `879` | M3 | 一个 executable oracle case 同时覆盖 separator 变化、额外字段、encoding 和空字段校验，prompt/case/gold 对齐较好。 |
| `441` | M3/M4 | 从简单格式分发扩展到 bound validation、content-based detection 和 malformed input 处理，难度递进清楚。 |

总体判断：

- 这两个 clean 样本组说明：当 Additional requirements 字段足够具体，并且 linked oracle case 能覆盖这些要求时，当前流程能生成比较有意义的难度扩展。
- `879_M3` 是最稳定的正例：题面、oracle case、gold 答案、operator checker 基本一致。
- `441_M3` 和 `441_M4` 展示了数据格式鲁棒性的提升，但 gold 仍偏启发式。
- 相比 `838_M3` 这类 rejected case，这两个 clean 样本组都没有依赖脆弱的文件副作用，比如必须生成某个 log 文件，而是直接验证返回值、异常或解析行为。
