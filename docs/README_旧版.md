<h1 align="center">QuantMind</h1>

<p align="center">
  <strong>AI 驱动的多市场量化交易平台</strong>
</p>

<p align="center">
  数据采集 → 因子挖掘 → 模型训练 → 策略回测 → 智能推理 → 实盘交易
</p>

<p align="center">
  <a href="#-快速开始">快速开始</a> •
  <a href="#-系统架构">系统架构</a> •
  <a href="#-核心功能">核心功能</a> •
  <a href="#-多市场数据">多市场数据</a> •
  <a href="#-ai-能力">AI 能力</a> •
  <a href="#-部署指南">部署指南</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Node.js-20+-green.svg" alt="Node.js">
  <img src="https://img.shields.io/badge/TypeScript-5.x-blue.svg" alt="TypeScript">
  <img src="https://img.shields.io/badge/Qlib-Powered-orange.svg" alt="Qlib">
  <img src="https://img.shields.io/badge/License-AGPL%20v3-blue.svg" alt="License">
</p>

---

## 项目起源

本项目基于 [qusong0627/QuantMind](https://github.com/qusong0627/QuantMind) 分支开发，感谢原作者提供的基础架构和核心思路。

在原项目基础上，本仓库进行了以下主要扩展和整合：

- **全项目单一仓库** — TradingAgents-astock、OpenBB-CN、dexter-finance 三个独立子框架并入主仓库统一管理
- **Qlib 深度集成** — 微软 Qlib 量化框架，Alpha 因子集 + LightGBM/XGBoost 模型训练
- **RD-Agent 因子挖掘** — 微软 RD-Agent 自动化因子进化，支持 A 股/港股/美股/加密四市场
- **TradingAgents 投研** — 多 Agent A 股研究框架（7 个 AI 分析师 + 辩论模块）
- **数据平台** — 统一多市场多数据源接入，17 个数据适配器，151+ 维特征工程
- **模型全生命周期** — 训练 → 版本管理 → 推理 → 信号生成完整闭环
- **策略实验室** — Python SDK 编辑器 + 子进程沙箱回测，7 类内置策略模板
- **实盘交易** — QMT 券商对接、模拟盘验证、风控系统、回放复盘
- **QuantBot 智能助手** — 自然语言交互，意图识别驱动操作
- **量化回放（Replay）** — 历史行情逐笔回放，策略状态复盘

---

## 项目简介

QuantMind 是一个端到端的量化交易平台，集成 Qlib 量化框架、RD-Agent 智能体与 TradingAgents 多 Agent 投研，支持 A 股、港股、美股、区块链、期货五大市场。

**核心能力：**
- **多市场数据管线** — 自动采集、清洗、校准 A/HK/US/Crypto/期货 行情数据
- **AI 因子挖掘** — 基于 RD-Agent 的自动化因子进化（多市场因子集），AlphaAgent 因子编码
- **模型训练与推理** — LightGBM/XGBoost 等模型，支持按市场切换特征与数据源、增量训练、批量推理、信号生成
- **策略生成** — AI 辅助生成 Qlib 策略代码，支持自然语言交互
- **回测引擎** — 基于 Qlib 的高性能回测，多策略对比，批量聚合回测
- **策略实验室** — Python SDK 编辑器，7 类示例策略，子进程沙箱运行
- **投研平台** — 多 Agent 协作的 A 股研究报告生成
- **量化回放** — 历史行情逐笔回放，交易决策复盘

---

## 快速开始

### 环境要求

- Docker & Docker Compose
- 8GB+ 内存（推荐 16GB）
- 50GB+ 磁盘空间（含数据）
- NVIDIA GPU（可选，用于模型训练加速）

### 一键部署

```bash
# 克隆仓库
git clone https://github.com/guge199205-byte/QuantMind-private.git
cd QuantMind-private

# 配置两份环境文件
cp .env.example .env
cp backend/.env.example backend/.env
# .env：PostgreSQL、Redis 与 Compose 配置
# backend/.env：应用密钥、LLM、服务与存储配置

# 启动所有服务
docker compose up -d

# 查看日志
docker compose logs -f quantmind
```

服务启动后：
- **Web 界面**: http://localhost:3080
- **API 文档**: http://localhost:8000/docs
- **引擎服务**: http://localhost:8001
- **行情服务**: http://localhost:8003
- **默认账号**: admin / admin123

> **AI 功能说明**：平台核心（数据/回测/训练/策略）无需 AI key 即可使用。AI 策略生成、因子挖掘、投研分析等 AI 功能需在 `.env` 配置 `AI_IDE_LLM_API_KEY` / `DASHSCOPE_API_KEY` 后才可用（见下方[环境变量配置](#环境变量配置)）。未配置时这些 AI 功能会提示需要 API Key，不影响其他功能。

### 下载市场数据

市场数据存放在 `data/` 目录（`./data:/data` 挂载进容器），五大市场数据中枢：

| 市场 | 数据目录 | 数据源 |
|------|----------|--------|
| **A 股** | `data/quantdb/` | QuantDB 本地 parquet + 17 适配器 |
| **港股** | `data/quanthk/` | Yahoo Finance / akshare |
| **美股** | `data/quantus/` | Yahoo Finance |
| **区块链**（默认屏蔽） | `data/quantbc/` | Binance |
| **期货** | `data/quantfutures/` | akshare |

**方式一：下载预置数据包**（推荐，从 [Releases](https://github.com/guge199205-byte/QuantMind-private/releases)）：

```bash
# 下载并解压到 data/ 目录（与 ./data:/data 挂载对应）
wget https://github.com/guge199205-byte/QuantMind-private/releases/download/v1.0.0-data/quantdb_data.tar.gz
tar xzf quantdb_data.tar.gz -C data/

# 港股、美股同理（quantus_data.tar.gz / quanthk_data.tar.gz）
```

**方式二：从数据源同步**（需网络，按市场）：

```bash
# A 股：从 QuantDB / baostock 等同步
docker exec quantmind python backend/scripts/quantdb_daily_sync.py

# 港股：akshare K 线
docker exec quantmind python backend/scripts/quanthk_daily_sync.py

# 期货
docker exec quantmind python backend/scripts/quantfutures_daily_sync.py
```

> 注意：区块链（quantbc）默认在生产环境屏蔽，无需下载。若需启用，设置 `ENABLE_CRYPTO=true` 后执行 `backend/scripts/quantbc_daily_sync.py`。

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Electron 桌面端 / Web                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐│
│  │ 仪表盘   │ │ 策略向导 │ │ 策略实验室│ │ 回测中心 │ │ 投研平台 ││
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘│
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐│
│  │ 模型管理 │ │ 模型训练 │ │ AI-IDE   │ │ 实盘交易 │ │ QuantBot ││
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘│
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP/WebSocket
┌───────────────────────────┴─────────────────────────────────────┐
│                   API Gateway (quantmind :8000)                  │
└───┬──────────┬──────────┬──────────┬────────────────────────────┘
    │          │          │          │
    ▼          ▼          ▼          ▼
┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
│ api   │ │engine │ │trade  │ │stream │
│ :8000 │ │ :8001 │ │ :8002 │ │ :8003 │
└───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘
    │         │         │         │
    ▼         ▼         ▼         ▼
┌────────────────────────────────────────────────────────────────┐
│   PostgreSQL + Redis + Celery（worker/beat 异步任务）          │
└────────────────────────────────────────────────────────────────┘
```

### 服务职责

| 服务 | 端口 | 职责 |
|------|------|------|
| **api** | 8000 | 用户认证、策略管理、数据平台、模型管理、新闻代理 |
| **engine** | 8001 | Qlib 回测、AI 策略生成、模型推理、Alpha Agent、训练编排 |
| **trade** | 8002 | 订单管理、持仓、风控 |
| **stream** | 8003 | 实时行情、WebSocket 推送 |
| **celery** | - | 异步任务（数据同步、推理、新闻增强） |
| **data-gateway** | 8004 | 多市场数据网关 |
| **dashboard** | 8501 | 数据分析面板（Streamlit） |

### 技术栈

| 层级 | 技术 |
|------|------|
| **前端** | Electron + React + TypeScript + Ant Design + ECharts |
| **后端** | Python + FastAPI + SQLAlchemy + Celery |
| **量化** | Qlib + LightGBM + XGBoost + RD-Agent |
| **数据库** | PostgreSQL + Redis + DuckDB（Parquet 查询） |
| **部署** | Docker Compose + Nginx |

---

## 核心功能

### 1. 数据平台（多市场数据中枢）

统一接入 17 个数据适配器，覆盖 A 股、港股、美股、加密货币：

| 市场 | 数据源适配器 |
|------|-------------|
| **A 股** | QuantDB（本地/远程）、baostock、akshare、efinance、pytdx/eltdx/opentdx、投资数据、simonlin |
| **港股** | yfinance、QuantDB、openbb |
| **美股** | yfinance、simonlin、openbb |
| **加密货币** | Binance、openbb |
| **通用** | Yahoo Finance、OpenBB-CN、TDX API |

- **QuantDB 本地数据中枢** — A 股优先数据源，DuckDB + Parquet 高性能读取
- **151+ 维特征工程** — 动量/波动/流动性/资金流/风格因子
- **Qlib 数据构建** — 自动从 Parquet 生成 Qlib 二进制缓存
- **行情源** — eltdx/opentdx 实时行情，WebSocket 推送

### 2. 模型训练与推理（全生命周期）

**模型训练** — 基于本地 Docker 编排：
- LightGBM / XGBoost / CatBoost / 模型集成（Stacking）
- 训练目标配置、特征选择、参数配置可视化
- 增量训练，训练进度实时日志流
- 资源保护：训练时自动暂停非关键容器释放内存

**模型管理** — 模型注册表：
- 模型版本管理、上传/下载/删除
- 特征目录（Feature Catalog）管理
- 模型评估指标、过拟合检查

**推理服务** — 批量推理引擎：
- 单日推理、批量预测、信号生成
- 批量聚合（Batch Aggregator）多模型投票
- 中性化（Neutralizer）、行业因子（申万行业）
- 交易成本建模、历史缓冲区
- 回测服务联动，信号可视化

### 3. 回测引擎

基于微软 Qlib 的高性能回测：

```python
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.backtest import backtest

strategy = TopkDropoutStrategy(signal=pred_signal, topk=50, n_drop=5)

report, indicator = backtest(
    strategy=strategy,
    start_time="2024-01-01",
    end_time="2024-12-31",
    account=1000000,
)
```

- **快速回测** — 多市场、多策略对比
- **批量回测** — 多模型/多周期批量聚合回测
- **回测参数**：topk 持仓数、n_drop 换仓数、rebalance_period 调仓周期、benchmark 基准

### 4. 策略实验室（Strategy Lab）

Python SDK 驱动的策略编写、回测和持久化平台：

- **Monaco 编辑器** — Python 语法高亮、自动补全
- **7 类内置策略** — 基础 / 趋势 / 反转 / 择时 / 量价 / 横截面 / 多因子
- **子进程沙箱** — 独立进程运行，Redis 实时进度推送
- **结果可视化** — 权益曲线、回撤图、月度热力图、风险指标
- **策略 CRUD** — 保存/加载/更新/删除，持久化到 `/api/v1/strategies`
- **AI 助手** — 自然语言生成策略代码

SDK 核心接口：
```python
ctx.universe    # 股票池
ctx.start       # 回测起始日
ctx.end         # 回测结束日
ctx.cash        # 初始资金

def setup(ctx):     ...   # 初始化
def on_bar(ctx):    ...   # 每日回调
def on_universe(ctx): ... # 选股回调
```

### 5. 实盘交易

- **QMT 券商对接** — 本地优先订单持久化，外部券商提交
- **模拟盘验证** — 下单前模拟盘校验
- **风控系统** — 订单超时扫描、风控规则
- **量化回放（Replay）** — 历史行情逐笔回放，交易决策复盘
  - 手动模式：逐笔勾选、修改数量、跳过
  - 统计报告：自动推进、收益分析
- **实时监控** — 持仓监控、策略状态、交易日志

### 6. QuantBot 智能助手

自然语言交互驱动：
- 策略查询和修改
- 回测执行和结果解读
- 市场行情问答
- 操作指引
- 集成 QwenPaw 智能体（网页 AI 机器人）

---

## Skills（Claude Code 技能包）

QuantMind 提供完整的 **Claude Code / QuantBot 技能包**（`.claude/skills/`），让 AI 助手直接调用平台量化功能：数据分析、因子挖掘、模型训练、策略生成、回测、模拟交易等。技能通过自然语言触发词激活。

**技能清单**（详见 [.claude/skills/README.md](.claude/skills/README.md)）：

| 技能 | 触发词 | 功能 |
|------|--------|------|
| `quantmind-operations` | 模型训练、模型管理、数据更新 | 平台运营操作总指南 |
| `stock-market-analysis` | 分析市场、全市场扫描、行业轮动 | 股票市场深度数据分析与导出 |
| `smart-strategy-stock-picking` | 选股、筛选股票、股票池 | 条件选股（按市场动态加载） |
| `ai-ide-strategy-writing` | 写策略、生成策略、AI-IDE | AI 生成 Qlib 策略代码 |
| `backtest-center` | 回测、策略对比、参数优化 | Qlib 回测中心 |
| `batch-inference-analysis` | 分析批量推理、解读信号 | 批量推理结果分析 |
| `rd-agent-factor-mining` | 挖因子、因子演化、RD-Agent | 多市场因子挖掘 |
| `quantdb-sdk` | quantdb、查询K线、数据集 | QuantDB 数据 SDK |
| `simulation-trading` | 模拟交易、查持仓 | 模拟交易 |

**安装**：解压 `quantmind-operations-skill.zip` 到 `~/.claude/`，或直接使用项目内 `.claude/skills/`。

---

## 多市场数据

### 数据表结构

| 市场 | 表名 | 数据源 | 覆盖范围 |
|------|------|--------|----------|
| A 股 | `stock_daily_latest` | QuantDB + investment_data + baostock | 2010 ~ 今 |
| 港股 | `stock_daily_latest_hk` | Parquet + yfinance | 2020 ~ 今 |
| 美股 | `stock_daily_latest_us` | yfinance | 2020 ~ 今 |
| 加密货币 | `stock_daily_latest_crypto` | Binance API | 2020 ~ 今 |

### 数据管线

```
原始数据源 → PostgreSQL → 技术指标计算 → Qlib bin → H5 文件 → 特征工程 Parquet
```

每个市场包含 35+ 技术指标：
- **均线**: MA5/10/20/60, 距均线偏离度
- **动量**: RSI(6/14), MACD(12/26/9), KDJ(9)
- **波动**: ATR(14/20), 标准差, 下行波动率
- **资金**: VPIN, 量比, 换手率
- **风格**: Beta, 特质波动率, 市值因子

### A 股数据同步

A 股数据通过 `backend/scripts/quantdb_daily_sync.py`（主流程）/ `daily_data_sync.py`（全量同步）自动同步：

```bash
# 手动触发同步
python backend/scripts/quantdb_daily_sync.py

# 全量同步（QuantDB → baostock → akshare → eltdx → PG → Qlib）
python backend/scripts/daily_data_sync.py

# 仅同步行情数据
python backend/scripts/daily_data_sync.py --skip-indicators

# 仅计算指标
python backend/scripts/daily_data_sync.py --indicators-only
```

同步流程：
1. 拉取 QuantDB / investment_data（GitHub releases）
2. 更新 baostock / akshare 日线
3. 合并数据到 PostgreSQL
4. 生成 Qlib bin 格式
5. 计算 35+ 技术指标
6. 生成特征 Parquet（151+ 维）

### 数据管理

管理员可通过 Web 界面管理数据（`管理后台`）：
- **数据管理** → 查看各市场数据状态、触发同步
- **QuantDB 控制台** → QuantDB 本地数据中枢查询
- **数据平台** → 数据状态扫描、特征目录管理
- **RSS 源管理** → 新闻源配置
- **Alpha 因子** → RD-Agent 挖掘结果管理

---

## AI 能力

### 1. AI 策略生成（AI-IDE）

自然语言描述策略需求，AI 自动生成 Qlib 策略代码：

```
用户: 帮我写一个港股动量策略，选 RSI 低于 30 的股票，MA5 金叉 MA20 时买入
AI: [生成完整的 Qlib 策略代码，包含选股、买入、卖出、风控逻辑]
```

支持的市场：
- **CN** — A 股，使用 `/app/db/qlib_data` 数据
- **HK** — 港股，使用 `/app/db/qlib_data/hk_data` 数据
- **US** — 美股，使用 `/app/db/qlib_data/us_data` 数据
- **CRYPTO** — 加密货币，使用 `/app/db/qlib_data/crypto_data` 数据

### 2. RD-Agent 因子挖掘

基于微软 RD-Agent 的自动化因子进化，支持四市场适配器：

```bash
# 启动因子进化
POST /api/v1/alpha-agent/evolve
{
  "market": "hong_kong",
  "iterations": 10
}
```

流程：
1. 从市场数据中提取候选因子
2. 使用 LLM 生成因子假设
3. 回测验证因子有效性
4. 迭代优化，保留有效因子

### 3. AlphaAgent 因子编码

因子进化框架集成（AlphaAgent），支持：
- 因子编码专家系统提示词
- LLM 重试配置（502/429 等临时错误）
- 因子去重（Embedding 向量）

### 4. TradingAgents 投研

多 Agent 协作的 A 股研究框架（7 个 AI 分析师）：

- **基本面分析师** — 财报、估值分析
- **技术分析师** — K 线形态、技术指标
- **消息面分析师** — 新闻、公告解读
- **情绪分析师** — 市场情绪、资金流向
- **风险评估师** — 风险量化、回撤控制
- **辩论模块** — 多空观点碰撞
- **决策模块** — 综合研判，生成报告

12 阶段进度追踪，研究报告支持 Markdown 导出与下载。

### 5. 量化回放（Replay）

历史行情逐笔回放：
- **手动模式** — 逐笔勾选、修改数量、跳过
- **自动模式** — 自动推进行情
- **统计报告** — 回放收益分析、策略状态复盘
- **策略模板** — 下拉菜单快速配置

---

## 项目结构

```
QuantMind/
├── backend/
│   ├── main_oss.py                 # 统一入口（4 服务单镜像）
│   ├── shared/                     # 跨服务共享模块
│   │   ├── db_manager.py           # 数据库连接池
│   │   ├── redis_client.py         # Redis 客户端
│   │   ├── stock_utils.py          # 股票代码工具
│   │   └── trading_calendar.py     # 交易日历
│   ├── services/
│   │   ├── api/                    # API 服务 (:8000)
│   │   │   ├── routers/            # 路由定义
│   │   │   │   └── admin/          # 管理后台
│   │   │   └── user_app/           # 用户认证
│   │   ├── engine/                 # 引擎服务 (:8001)
│   │   │   ├── ai_strategy/        # AI 策略生成
│   │   │   ├── qlib_app/           # Qlib 回测
│   │   │   ├── inference/          # 批量推理引擎
│   │   │   ├── training/           # 本地 Docker 训练编排
│   │   │   ├── data_platform/      # 多市场数据平台
│   │   │   ├── alpha_agent/        # Alpha Agent
│   │   │   ├── rd_agent/           # RD-Agent 因子挖掘
│   │   │   ├── trading_agents/     # TradingAgents 投研
│   │   │   ├── quantbot/           # QuantBot 智能助手
│   │   │   ├── strategy_lab/       # 策略实验室
│   │   │   └── stock_query_app/    # 个股查询
│   │   ├── trade/                  # 交易服务 (:8002)
│   │   ├── stream/                 # 行情服务 (:8003)
│   │   └── ai_ide/                 # AI-IDE 服务
│   └── scripts/
│       ├── quantdb_daily_sync.py   # 每日数据同步（主流程）
│       ├── daily_data_sync.py      # 全量数据同步
│       ├── rebuild_core_parquet_full.py  # 特征 Parquet 重建
│       └── sync_quantdb.py         # QuantDB 同步
├── electron/
│   └── src/
│       ├── features/               # 功能模块
│       │   ├── dashboard/          # 仪表盘
│       │   ├── strategy-wizard/    # 策略向导
│       │   ├── strategy-lab/       # 策略实验室（SDK 编辑器）
│       │   ├── strategy-comparison/# 策略对比
│       │   ├── ai-strategy/        # AI 策略生成
│       │   ├── alpha-research/     # Alpha 研究
│       │   ├── trading-agents/     # 投研平台
│       │   ├── research/           # 研究平台
│       │   ├── quantbot/           # QuantBot 助手
│       │   ├── user-center/        # 用户中心
│       │   ├── news/               # 新闻
│       │   └── admin/              # 管理后台
│       ├── components/             # UI 组件
│       │   ├── backtestCenter/     # 回测中心
│       │   ├── inference/          # 推理面板
│       │   ├── training/           # 训练面板
│       │   ├── market/             # 行情组件
│       │   └── chart/              # K 线图表
│       ├── pages/                  # 页面
│       │   ├── ModelTrainingPage   # 模型训练
│       │   ├── ModelRegistryPage   # 模型管理
│       │   ├── BacktestCenterPage  # 回测中心
│       │   ├── AIIDEPage           # AI-IDE
│       │   └── trading/            # 实盘交易
│       └── config/                 # 配置
├── TradingAgents-astock/           # 多 Agent 投研框架（并入）
├── openbb-cn/                      # OpenBB 中国版数据源（并入）
├── dexter-finance/                 # OpenBB 集成（并入）
├── docker/
│   ├── Dockerfile                  # 后端镜像
│   ├── Dockerfile.web              # 前端 Nginx 镜像
│   └── Dockerfile.data-gateway     # 数据网关镜像
├── db/                             # 数据目录（gitignore）
│   ├── qlib_data/                  # Qlib bin 格式
│   │   ├── cn_data/                # A 股
│   │   ├── hk_data/                # 港股
│   │   ├── us_data/                # 美股
│   │   └── crypto_data/            # 加密货币
│   └── feature_snapshots/          # 特征快照 Parquet
├── config/                         # 配置文件
│   ├── data_sources/               # 数据源路由
│   └── features/                   # 特征目录
└── docker-compose.yml
```

---

## 部署指南

### 生产环境部署

```bash
# 1. 克隆代码
git clone https://github.com/guge199205-byte/QuantMind-private.git
cd QuantMind-private

# 2. 配置两份环境文件
cp .env.example .env
cp backend/.env.example backend/.env

# 根目录 .env：数据库/Redis 容器凭据与映射端口
# backend/.env：应用密钥、管理员、LLM 与服务配置
# 本地 Python 自动连接 127.0.0.1；Compose 容器自动连接 db/redis

# 3. 启动服务
docker compose up -d

# 4. 下载数据
# 从 Releases 下载数据文件并解压到 db/

# 5. 初始化数据库（首次启动自动完成）
# quantmind 容器启动时自动执行 backend/shared/db_init.sql 建表，
# 并自动创建默认 admin 账号。如需手动确认：
docker exec quantmind python -c "from backend.shared.database_manager_v2 import get_session; print('DB OK')"

# 6. 构建股票索引
docker exec quantmind python backend/services/api/scripts/build_stock_index.py
```

### 前端开发

```bash
cd electron
npm install
npm run dev          # Electron 桌面端
npm run dev:web      # Web 浏览器
npm run typecheck    # 类型检查
npm run dashboard:build  # 生产构建
```

### 后端开发

```bash
# 单服务启动（开发）
SERVICE_MODE=api python backend/main_oss.py
SERVICE_MODE=engine python backend/main_oss.py

# 运行测试
python backend/run_tests.py unit
python backend/run_tests.py integration
```

前后端以 Docker / 本地进程混合运行时，请参阅
[前后端容器组合测试](docs/前后端容器组合测试.md)。

---

## 定时任务

| 任务 | 时间 | 说明 |
|------|------|------|
| `quantdb_daily_sync` | 18:00 工作日 | A 股数据同步（主流程） |
| `auto_inference` | 00:00 工作日 | 模型自动推理 |
| `news_enrich` | 每 1 分钟 | 新闻 AI 增强 |
| `news_reload` | 每 10 分钟 | 新闻规则重载 |

---

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `POSTGRES_DB` | PostgreSQL 数据库名（根 `.env`） | `quantmind` |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` | PostgreSQL 容器与后端共享凭据（根 `.env`） | - |
| `REDIS_PASSWORD` | Redis 容器与后端共享密码（根 `.env`） | - |
| `DB_HOST` | 自动推导：本地 `127.0.0.1`，Docker `db`；可用真实环境变量覆盖 | 自动 |
| `REDIS_HOST` | 自动推导：本地 `127.0.0.1`，Docker `redis`；可用真实环境变量覆盖 | 自动 |
| `SECRET_KEY` | 应用密钥 | - |
| `JWT_SECRET_KEY` | JWT 密钥 | - |
| `AI_IDE_LLM_API_KEY` | LLM API Key（全项目统一） | - |
| `AI_IDE_LLM_BASE_URL` | LLM API Base URL | `https://token-plan-cn.xiaomimimo.com/v1` |
| `AI_IDE_LLM_MODEL` | LLM 模型名 | `mimo-v2.5-pro` |
| `DASHSCOPE_API_KEY` | 阿里云百炼 API Key（备用） | - |
| `QUANTDB_API_KEY` | QuantDB 数据源 Key | - |
| `EMBEDDING_API_KEY` | Embedding API Key | - |

---

## 贡献指南

欢迎提交 Issue 和 Pull Request！

```bash
# 1. Fork 仓库
# 2. 创建特性分支
git checkout -b feature/your-feature

# 3. 提交更改
git commit -m "feat: add your feature"

# 4. 推送并创建 PR
git push origin feature/your-feature
```

---

## License

[GNU Affero General Public License v3.0](LICENSE)

---

## 免责声明

> **本项目仅供学习研究与技术演示，不构成任何投资建议。**
>
> - 本系统产出的所有分析报告和交易信号均由 AI 自动生成，可能存在错误或偏差
> - 投资决策请咨询持有中国证监会颁发资质的专业机构
> - 作者不对使用本工具产生的任何投资损失承担责任
> - **股市有风险，投资需谨慎**

---

## 致谢

### 核心框架

- [Qlib](https://github.com/microsoft/qlib) — 微软量化投资平台
- [RD-Agent](https://github.com/microsoft/RD-Agent) — 微软研发智能体
- [AlphaAgent](https://github.com/ModelTC/AlphaAgent) — 因子进化框架
- [TradingAgents-Astock](https://github.com/simonlin1212/TradingAgents-astock) — 多 Agent A 股投研框架（并入）
- [OpenBB-CN](https://github.com/LSY1105/OPENBB-CN) — 开源金融市场数据（并入）
- [LightGBM](https://github.com/microsoft/LightGBM) — 微软梯度提升框架
- [XGBoost](https://github.com/dmlc/xgboost) — 梯度提升框架
- [FastAPI](https://fastapi.tiangolo.com/) — 现代高性能 Web 框架
- [Huntly](https://github.com/lcomplete/huntly) — 财经资讯聚合平台
- [RSSHub](https://github.com/DIYgod/RSSHub) — RSS 源生成工具
- [QwenPaw](https://github.com/agentscope-ai) — 网页 AI 智能体

### 数据源与工具

- [investment_data](https://github.com/chenditc/investment_data) — 开源 A 股历史行情数据
- [baostock](http://baostock.com/) — 免费 A 股行情数据接口
- [akshare](https://akshare.akfamily.xyz/) — 开源财经数据接口
- [efinance](https://github.com/Micro-sheep/efinance) — 东方财富数据接口
- [pytdx](https://github.com/rainx/pytdx) — 通达信行情接口
- [pandas](https://pandas.pydata.org/) / [pyarrow](https://arrow.apache.org/) — 数据处理与 Parquet 格式支持

---

## QQ 群

<p align="center">
  <img src="docs/images/1097406397.png" alt="QuantMind QQ 群二维码" width="260">
</p>

---

<p align="center">
  <strong>QuantMind</strong> — 让量化交易更简单
</p>
