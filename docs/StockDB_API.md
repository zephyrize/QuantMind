# StockDB 局域网 HTTP 查询接口

本文档用于回测／研究程序调用已部署的服务：

```text
http://192.168.1.111:7899/
```

**调用约定：**所有请求均为 `GET /`，参数通过 query string 传递，响应为 JSON，编码为 UTF-8。参数中的中文、冒号和逗号必须由 HTTP 客户端做 URL 编码；Python `requests.get(..., params=...)` 会自动完成编码。服务不需要认证。本文档描述网页实际使用且已对该局域网服务验证的 `keys`、`vals`、`get` 协议。

> 不要把本协议与仓库 C++ 源码中的简化服务混用：源码只实现了另一套 `cmd=get&t=<完整key>` 等接口，且不实现 `keys` / `vals`。若自行编译 `cpp/` 下的服务，请看 [WEB_API.md](WEB_API.md)。回测项目应连接本文档开头的已部署服务。

## 1. 最常用：读取日 K 或分钟 K

### 请求

```http
GET /?cmd=vals&t=<表名>&k1=key:<股票代码>&k2=<时间选择器>[&num=-N]
```

| 参数 | 必填 | 类型 | 可用值／格式 | 说明 |
|---|---:|---|---|---|
| `cmd` | 是 | string | `vals` | 返回 value（行情记录），不返回底层 key。 |
| `t` | 是 | string | `日k`、`分钟k` | 日线使用 `日k`；所有分钟数据使用 `分钟k`。`5m/15m/30m/60m` 由调用端用 1 分钟数据自行聚合。 |
| `k1` | 是 | string | `key:<6位代码>` | 股票或 ETF 代码，例如 `key:600633`。 |
| `k2` | 是 | string | 见下表 | 日期／时间选择器。 |
| `num` | 否 | integer string | `-N` | 仅在 `k2=all:` 时建议使用。`-100` 表示取该代码最新 100 条；返回数组仍按时间升序排列。 |

`k2` 的稳定用法：

| `k2` | 含义 | `日k` 时间格式 | `分钟k` 时间格式 |
|---|---|---|---|
| `key:<时间>` | 精确一条记录 | `YYYYMMDD` | `YYYYMMDDHHMMSS` |
| `fwd:<起始>,<结束>` | 闭区间，按时间升序返回 | `YYYYMMDD,YYYYMMDD` | `YYYYMMDDHHMMSS,YYYYMMDDHHMMSS` |
| `all:` | 该代码的全部记录 | - | - |

### 日 K 示例

精确查询 600633 在 2000-09-15 的日 K：

```text
http://192.168.1.111:7899/?cmd=vals&t=%E6%97%A5k&k1=key%3A600633&k2=key%3A20000915
```

查询一个日期区间：

```python
import requests

BASE = "http://192.168.1.111:7899/"
params = {
    "cmd": "vals",
    "t": "日k",
    "k1": "key:600633",
    "k2": "fwd:20260801,20260806",
}
response = requests.get(BASE, params=params, timeout=15)
response.raise_for_status()
rows = response.json()              # list[dict]
```

取最新 100 条：

```python
rows = requests.get(BASE, params={
    "cmd": "vals", "t": "日k", "k1": "key:600633",
    "k2": "all:", "num": "-100",
}, timeout=15).json()
```

### 分钟 K 示例

分钟查询必须传 14 位时间。若输入的是自然日，调用端应补齐为当天的 `000000` 和 `235959`：

```python
rows = requests.get(BASE, params={
    "cmd": "vals",
    "t": "分钟k",
    "k1": "key:600422",
    "k2": "fwd:20260625000000,20260625235959",
}, timeout=15).json()
```

### 成功响应

始终是数组；没有记录时应按空数组 `[]` 处理。下面是服务实际返回的一条日 K 的结构（JSON number 的小数精度不应与显示精度混淆）：

```json
[
  {
    "amount": 1227460,
    "amplitude": 0,
    "close": 6.5,
    "code": "600633",
    "date": 20000915,
    "float_mv": 83812941.635,
    "float_share": 13200000,
    "high": 6.5,
    "is_st": false,
    "low": 6.5,
    "name": "浙数文化",
    "open": 6.5,
    "pb": 779.798619,
    "pct_chg": 0,
    "pe_ttm": -55.410044,
    "pre_close": 2.294,
    "total_mv": 965441274.696,
    "total_share": 152050800,
    "turnover": 1.430606,
    "vol_ratio": 1.35,
    "volume": 188800
  }
]
```

## 2. K 线字段字典

同一表中可能有缺字段或字段值为 `null`；程序不应假定所有记录都完整。JSON 标准只有 `number`，没有单独的 `int`／`float` 类型：价格、比率、市值应使用 Python `float` / `float64`（如需财务精度可转 `Decimal`）；成交量、股本通常是整数值，但应按 API 返回的 JSON `number` 解析。

| 字段 | JSON 类型 | 含义／单位 |
|---|---|---|
| `date` | integer | 日 K：`YYYYMMDD`；分钟 K：`YYYYMMDDHHMMSS`，例如 `20260625145200`。 |
| `code` | string | 6 位证券代码，必须按字符串保存，保留前导零。 |
| `name` | string | 证券简称。 |
| `open` / `high` / `low` / `close` | number | 不复权的开／高／低／收盘价，货币单位为元。 |
| `pre_close` | number | 前收盘价，元。若要严格计算区间第一条的涨跌幅，应额外查询其上一交易日并自行校验。 |
| `volume` | number | 成交量，单位为股。 |
| `amount` | number | 成交额，单位为元。 |
| `turnover` | number | 换手率，单位为百分比；`2.51` 即 2.51%，不是 0.0251。 |
| `pct_chg` | number | 涨跌幅，单位为百分比；`-0.61` 即 -0.61%。 |
| `amplitude` | number | 振幅，单位为百分比。 |
| `vol_ratio` | number | 量比，倍数，无百分号。 |
| `is_st` | boolean | 是否 ST。 |
| `total_share` / `float_share` | number | 总股本／流通股本，单位为股。 |
| `total_mv` / `float_mv` | number | 总市值／流通市值，单位为元。 |
| `pe_ttm` | number | 滚动市盈率；可为负数或缺失。 |
| `pb` | number | 市净率；可为缺失。 |

**顺序与频率：**`fwd:` 和 `num=-N` 的返回结果是升序，即最早记录在 `rows[0]`、最新记录在 `rows[-1]`。服务直接提供日 K 和 1 分钟 K；周 K、月 K 以及 5/15/30/60 分钟 K 不是服务端表，需按回测策略聚合。

## 3. `cmd=keys`：查询逻辑 key（代码列表或日期列表）

### 请求

```http
GET /?cmd=keys&t=<表名>&k1=<key前缀>&k2=<时间选择器>
```

`keys` 与 `vals` 的 `t`、`k1`、`k2` 语义相同，但只返回 key 字符串数组，不返回行情 value。

| 任务 | 参数 | 响应示例 |
|---|---|---|
| 加载某个交易日的全市场代码 | `cmd=keys&t=日k&k1=all:&k2=key:20260625` | `["日k:000001:20260625", "日k:000002:20260625", ...]` |
| 加载某只股票的所有日 K 日期 | `cmd=keys&t=日k&k1=key:600633&k2=all:` | `["日k:600633:20000915", ...]` |
| 查询日期范围 | `cmd=keys&t=日k&k1=key:600633&k2=fwd:20000915,20000916` | `["日k:600633:20000915"]` |

网页还使用了一个等价的旧式前缀写法：

```text
GET /?cmd=keys&t=日k:600633*
```

回测项目建议使用三段式 `t=日k&k1=key:600633&k2=all:`，更明确且与 `vals` 一致。

## 4. `cmd=get`：通用读取；读取复权因子

### 请求

```http
GET /?cmd=get&t=<表名>&k1=<key前缀>&k2=<时间选择器>
```

`get` 还支持完整逻辑 key 的精确读取，适合只取一条记录：

```http
GET /?cmd=get&t=日k:600633:20000915
```

该请求返回一条 value 对象（不是数组）。完整 key 的构成是 `<表名>:<代码>:<时间>`，例如 `日k:600633:20000915`、`分钟k:600422:20260625145200`。涉及多行时，请使用本节的三段式参数，或优先使用 `cmd=vals`。

对回测最有用的用法是读取某只证券的全部复权因子：

```python
factors = requests.get(BASE, params={
    "cmd": "get",
    "t": "复权",
    "k1": "key:600633",
    "k2": "all:",
}, timeout=15).json()
```

多条记录时，服务返回 `[完整逻辑 key, value]` 对的数组：

```json
[
  [
    "复权:600633:19930705",
    {"div": 0, "give": 0.1, "trans": 0, "mult": 1.1, "cum": 1.1}
  ],
  [
    "复权:600633:19940613",
    {"div": 0.1, "give": 0.2, "trans": 0, "mult": 1.217, "cum": 1.339}
  ]
]
```

| 复权 value 字段 | JSON 类型 | 含义 |
|---|---|---|
| `div` | number | 数据快照提供的本次现金分红参数。 |
| `give` | number | 数据快照提供的本次送股参数。 |
| `trans` | number | 数据快照提供的本次转增参数。 |
| `mult` | number | 数据快照提供的本次公司行为乘数。 |
| `cum` | number | **累计复权因子**；回测计算前／后复权时应使用此字段。 |

复权因子发生日来自 key 最后一段，例如 `复权:600633:19930705` 的日期为 `19930705`。单条精确查询时，`get` 可能直接返回 value 对象而非长度为 1 的二维数组；调用端若使用精确查询，应兼容这两种形态。读取全量因子时使用 `k2=all:`，即可稳定得到 key/value 对数组。

### 前／后复权计算

`vals` 返回的是不复权行情，服务端没有 `fq=qfq|hfq` 参数。调用端须自行读取 `复权` 表并计算。对每个行情日期 `d`：

```text
F(d)       = 日期不晚于 d 的最后一个 cum；若不存在则为 1.0
F(latest)  = 该代码最后一个 cum；若不存在则为 1.0

前复权 qfq：P_qfq = P_raw × F(d) / F(latest)
后复权 hfq：P_hfq = P_raw × F(d)
```

对 `open`、`high`、`low`、`close`、`pre_close` 使用同一公式；成交量、成交额、市值、估值、换手率等字段保持原值。网页的展示规则会将当天 `pre_close` 覆盖为前一交易日已复权的 `close`；若回测依赖涨跌幅，建议用相邻行的调整后 `close` 自行计算，避免首条记录缺少上一日的问题。

## 5. 可直接复制的最小回测客户端

```python
from __future__ import annotations
import requests

BASE = "http://192.168.1.111:7899/"

def get_daily(code: str, start: str, end: str) -> list[dict]:
    if not (code.isdigit() and len(code) == 6):
        raise ValueError("code 必须是 6 位字符串")
    if not (len(start) == len(end) == 8 and start.isdigit() and end.isdigit()):
        raise ValueError("日 K 日期必须为 YYYYMMDD")

    r = requests.get(BASE, params={
        "cmd": "vals", "t": "日k", "k1": f"key:{code}",
        "k2": f"fwd:{start},{end}",
    }, timeout=(3.05, 30))
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise RuntimeError(f"意外响应：{data!r}")
    return [row for row in data if isinstance(row, dict)]

rows = get_daily("600633", "20260801", "20260806")
```

## 6. 调用限制与排错

- 使用 HTTP 客户端的 `params` / `--data-urlencode`，不要手工拼接未编码的中文参数。
- 连接成功不代表有数据：空数组表示该代码、表或时间范围没有匹配项。
- 返回的 `date` 是 JSON integer，但股票代码 `code` 是 string；不要把代码转为 int，否则深圳、北交所等前导零代码会丢失。
- 服务对浏览器响应 `Access-Control-Allow-Origin: *`；跨机器调用仍须确保 Windows 防火墙允许 **TCP 7899 入站**。文件与打印机共享规则本身不等价于开放 7899。
- 请给请求设置连接与读取超时；全市场 `keys` 和 `all:` 查询返回量可能很大。回测应优先按单代码和日期范围使用 `vals`。
- 本文档的查询接口均为只读。不要依赖或暴露仓库源码中的 `set` 写接口；它不属于已验证的局域网回测调用协议。
