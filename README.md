# QuantAgent: 多智能体 LLM 量化交易系统

> **声明**：本项目是对 [Y-Research-SBU/QuantAgent](https://github.com/Y-Research-SBU/QuantAgent) 的学习与模仿实现，
> 仅用于个人学习和教育目的，不用于商业使用。
> 原作者论文：[QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading](https://arxiv.org/abs/2509.09995)

一个基于 LangChain + LangGraph 的多智能体交易分析系统，整合了技术指标计算、K 线形态识别、趋势通道分析，
最终由决策 Agent 综合输出交易指令（做多 / 做空）。支持 Web 界面操作，适配多种 LLM 供应商。

<p align="center">
  <a href="#快速开始">快速开始</a> ·
  <a href="#系统架构">系统架构</a> ·
  <a href="#代码结构详解">代码详解</a> ·
  <a href="#配置与调参">配置调参</a> ·
  <a href="#安装指南">安装指南</a>
</p>

---

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/lvguanda123/quantagent-.git
cd quantagent-

# 2. 安装环境（选一条执行）
install.bat              # Windows
chmod +x setup.sh && ./setup.sh  # Linux / macOS

# 3. 启动
python web_interface.py
# 浏览器打开 http://127.0.0.1:5000
```

---

## 系统架构

```
                        用户 (Web 界面)
                              |
                              v
                     ┌────────────────────┐
                     │   TradingGraph      │ ← 入口: trading_graph.py
                     │  (LangGraph 编排)    │
                     └────────┬───────────┘
                              |
              ┌───────────────┼───────────────┐
              |               |               |
              v               v               v
     ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
     │ Indicator   │ │  Pattern    │ │   Trend     │
     │   Agent     │ │   Agent     │ │   Agent     │
     │ 技术指标计算 │ │ K线形态识别 │ │ 趋势通道分析 │
     └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
            |               |               |
            └───────────────┼───────────────┘
                            |
                            v
                    ┌───────────────┐
                    │  Decision     │
                    │   Agent       │ ← 综合决策: 做多/做空
                    └───────────────┘
                            |
                            v
                    Web 界面展示结果
```

### 工作流程

1. **用户**在 Web 界面选择股票/币种和时间周期（如 15min、4hour）
2. 系统通过 Yahoo Finance 或 AKShare 获取最近 30 根 K 线数据
3. 三个分析 Agent **并行**运行：
   - Indicator Agent 计算 MACD、RSI 等技术指标
   - Pattern Agent 绘制 K 线图并识别经典形态
   - Trend Agent 绘制趋势通道并分析走势
4. Decision Agent 汇总三份报告，输出最终交易指令（含入场点、止损点）
5. 结果展示在 Web 页面上

---

## 代码结构详解

### 核心文件（每个文件的作用、如何修改）

```
├── trading_graph.py          ← 系统总入口，LLM 工厂
├── graph_setup.py            ← LangGraph 图的组装
├── graph_util.py             ← 工具集（技术指标计算、图片生成）
├── agent_state.py            ← 状态定义
├── default_config.py         ← 所有默认配置
│
├── indicator_agent.py        ← Agent 1: 技术指标分析
├── pattern_agent.py          ← Agent 2: K 线形态识别
├── trend_agent.py            ← Agent 3: 趋势通道分析
├── decision_agent.py         ← Agent 4: 最终交易决策
│
├── web_interface.py          ← Flask Web 界面
├── static_util.py            ← 静态资源工具函数
│
├── data_providers/           ← 数据源抽象层
│   ├── base.py               ←   抽象接口
│   ├── yahoo.py              ←   Yahoo Finance
│   ├── akshare.py            ←   AKShare（A 股）
│   └── qlib.py               ←   Qlib 本地数据
│
├── templates/                ← HTML 模板
│   ├── demo_new.html         ←   主页（选择资产）
│   └── output.html           ←   结果页
│
├── static/                   ← 静态文件（CSS、图片）
├── requirements.txt          ← Python 依赖
├── .env.example              ← 环境变量模板
├── setup.sh / install.bat    ← 一键安装脚本
└── REPRODUCE.md              ← 完整复现指南
```

### 逐个文件详解

#### `trading_graph.py` — 系统总入口

**职责**：初始化 LLM 实例，创建 ToolNode，把所有 Agent 组装成 LangGraph。

**关键类**：
- `TradingGraph.__init__()`：根据配置创建 LLM（支持 OpenAI / Anthropic / Qwen / MiniMax）
- `refresh_llms()`：更新 API Key 后重新初始化
- `update_api_key()`：Web 界面调用此方法切换 Key

**如果你想**：
- 添加新的 LLM 供应商 → 修改 `_create_llm()` 和 `_get_api_key()`
- 更换默认模型 → 修改 `DEFAULT_CONFIG` 中的模型名称

---

#### `graph_setup.py` — LangGraph 图的组装

**职责**：用 LangGraph 的 `StateGraph` 把四个 Agent 连成有向图。

**关键方法**：
- `set_graph()`：创建 Indicator、Pattern、Trend、Decision 四个 Agent 节点，定义数据流向

**如果你想**：
- 添加新的 Agent（如 Risk Agent）→ 在此处创建节点并添加边
- 改变 Agent 执行顺序 → 修改图的边（`START → ... → END`）

---

#### `graph_util.py` — 工具集

**职责**：所有工具函数的集合，供 Agent 调用。

**关键工具**：
- `compute_macd()` / `compute_rsi()` / `compute_stoch()` 等 — 技术指标计算（基于 TA-Lib）
- `generate_kline_image()` — 生成 K 线图片供 Pattern Agent 识别
- `generate_trend_image()` — 生成带趋势通道的图片供 Trend Agent 分析

**如果你想**：
- 增加新的技术指标 → 在此处添加新的 `@tool` 函数
- 修改图片样式（颜色、标注）→ 修改 `generate_*_image()` 中的 matplotlib 代码

---

#### `agent_state.py` — 状态定义

**职责**：定义 LangGraph 中传递的状态数据结构。

**包含字段**：
- `kline_data` — K 线数据（字典格式）
- `indicator_report` — Indicator Agent 的输出
- `pattern_report` — Pattern Agent 的输出
- `trend_report` — Trend Agent 的输出
- `final_trade_decision` — Decision Agent 的最终结果
- `time_frame`、`stock_name` — 用户选择的周期和标的

**如果你想**：
- 在 Agent 之间传递额外数据 → 在此类中添加新字段

---

#### `default_config.py` — 配置中心

**职责**：所有默认配置值，包括模型名称、温度、API Key 等。

**关键参数**：

| 参数 | 含义 | 默认值 |
|------|------|-------|
| `agent_llm_model` | 单个 Agent 使用的模型 | `qwen3.6-plus` |
| `graph_llm_model` | 图编排逻辑使用的模型 | `qwen3.6-plus` |
| `agent_llm_provider` | Agent 的 LLM 供应商 | `anthropic` |
| `graph_llm_provider` | 图的 LLM 供应商 | `anthropic` |
| `agent_llm_temperature` | Agent 回答的随机性 | `0.1`（较低，更稳定）|
| `graph_llm_temperature` | 图逻辑的随机性 | `0.1` |

**如果你想**：
- 更换模型 → 修改 `agent_llm_model` 和 `graph_llm_model`
- 让回答更有创意 → 调高 temperature（0.5~0.7）
- 让回答更确定 → 调低 temperature（0.0~0.1）

---

#### `indicator_agent.py` — 技术指标分析 Agent

**职责**：调用工具计算 MACD、RSI、ROC、Stochastic、Williams %R 五个指标，并用 LLM 解读结果。

**输入**：OHLCV K 线数据 + 时间周期
**输出**：技术指标分析报告（文本）

**如果你想**：
- 增加/减少技术指标 → 修改 `tools` 列表中的函数
- 调整分析提示词 → 修改 `prompt` 模板中的中文说明

---

#### `pattern_agent.py` — K 线形态识别 Agent

**职责**：生成 K 线图，让支持视觉的 LLM（如 GPT-4o、Claude、qwen-vl）识别经典形态（头肩底、双顶、三角形等）。

**输入**：K 线数据 → 生成图片
**输出**：识别到的形态名称和描述

**如果你想**：
- 添加新的形态 → 修改 `pattern_text` 中的形态描述
- 调整图片生成逻辑 → 修改 `generate_kline_image()` 函数

---

#### `trend_agent.py` — 趋势通道分析 Agent

**职责**：生成带趋势通道（上轨 = 近期高点连线，下轨 = 近期低点连线）的 K 线图，让 LLM 分析市场方向。

**输入**：K 线数据 → 生成趋势通道图片
**输出**：趋势报告（方向、斜率、盘整区域）

**如果你想**：
- 改变通道计算方法 → 修改 `generate_trend_image()` 中的趋势线逻辑
- 调整分析提示词 → 修改 `prompt` 中的中文说明

---

#### `decision_agent.py` — 最终交易决策 Agent

**职责**：汇总前三份报告，给出明确的交易指令（LONG 或 SHORT），含入场价、止损价、止盈价。

**输入**：indicator_report + pattern_report + trend_report
**输出**：JSON 格式的交易决策

**如果你想**：
- 允许 HOLD（观望） → 修改 prompt 中 "禁止持有观望" 的约束
- 调整输出格式 → 修改 prompt 和 `_extract_json()` 函数
- 增加风险评估 → 在 prompt 中加入新的评估维度

---

#### `web_interface.py` — Flask Web 界面

**职责**：提供 Web 界面，处理用户请求、获取行情数据、调用 TradingGraph 进行分析。

**主要路由**：
- `/` — 主页（资产选择）
- `/analyze` — 执行分析，返回结果
- `/api_key` — 更新 API Key

**如果你想**：
- 修改页面样式 → 编辑 `templates/` 下的 HTML 和 `static/` 下的 CSS
- 增加新功能 → 添加新的 `@app.route`
- 修改数据获取逻辑 → 编辑 `get_kline_data()` 函数

---

#### `data_providers/` — 数据源抽象层

| 文件 | 数据源 | 适用场景 |
|------|--------|---------|
| `base.py` | 抽象接口 | 定义统一的数据获取接口 |
| `yahoo.py` | Yahoo Finance | 全球股票、加密货币、商品 |
| `akshare.py` | AKShare | A 股、基金、ETF 分钟数据 |
| `qlib.py` | Qlib 本地数据 | 离线回测场景 |

**如果你想**：
- 接入新的数据源（如 Tushare、Binance API）→ 继承 `BaseProvider`，实现对应方法，并在 `registry.py` 中注册

---

## 配置与调参

### 1. 切换 LLM 供应商和模型

在 `default_config.py` 中修改：

```python
# 使用 OpenAI
DEFAULT_CONFIG["agent_llm_provider"] = "openai"
DEFAULT_CONFIG["agent_llm_model"] = "gpt-4o"
DEFAULT_CONFIG["agent_llm_temperature"] = 0.1

# 使用 Anthropic Claude
DEFAULT_CONFIG["agent_llm_provider"] = "anthropic"
DEFAULT_CONFIG["agent_llm_model"] = "claude-sonnet-4-6-20250514"

# 使用 Qwen
DEFAULT_CONFIG["agent_llm_provider"] = "qwen"
DEFAULT_CONFIG["agent_llm_model"] = "qwen3-vl-plus"
```

### 2. 调参指南

| 想调什么 | 改哪里 | 效果 |
|---------|--------|------|
| 模型输出更随机/更稳定 | `agent_llm_temperature` | 高 → 更有创意但更不稳定；低 → 更保守但更一致 |
| 决策更激进/更保守 | `decision_agent.py` 中的 prompt | 修改 prompt 中的风险偏好描述 |
| 分析更多/更少的 K 线 | `web_interface.py` 中的数据获取 | 修改获取的 K 线数量（默认 30 根） |
| 使用不同的技术指标 | `indicator_agent.py` + `graph_util.py` | 在 tools 列表中添加或删除 |
| 添加新的 K 线形态 | `pattern_agent.py` 的 `pattern_text` | 加入形态描述即可 |
| 更换数据源 | `web_interface.py` + `data_providers/` | 选择不同的 Provider |

### 3. 环境变量（推荐方式）

复制模板并填写真实值：

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

也可以在 Web 界面直接输入。

---

## 安装指南

### 前置要求

- Python 3.11+
- Git

### Windows 用户（一键安装）

双击 `install.bat`，脚本会自动创建虚拟环境并安装所有依赖。

### Linux / macOS 用户

```bash
chmod +x setup.sh && ./setup.sh
```

### 手动安装（如果脚本失败）

```bash
# 1. 创建虚拟环境
python -m venv venv

# 2. 激活
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/macOS

# 3. 安装依赖
pip install --upgrade pip
pip install -r requirements.txt
```

### TA-Lib 安装（常见问题）

TA-Lib 需要先安装底层的 C 语言库：

```bash
# Windows：从 https://github.com/ta-lib/ta-lib/releases 下载安装
# macOS
brew install ta-lib
# Ubuntu/Debian
sudo apt-get install -y ta-lib

# 然后重新安装 Python 包
pip install --force-reinstall TA-Lib
```

### 启动

```bash
python web_interface.py
# 浏览器访问 http://127.0.0.1:5000
```

---

## 支持的 LLM 供应商

| 供应商 | 环境变量 | 推荐模型 | 需要视觉能力？ |
|--------|---------|---------|--------------|
| OpenAI | `OPENAI_API_KEY` | GPT-4o / GPT-4o-mini | 是 |
| Anthropic | `ANTHROPIC_API_KEY` | Claude Sonnet / Haiku | 是 |
| 通义千问 | `DASHSCOPE_API_KEY` | qwen3-vl-plus / qwen3-max | 是（视觉） |
| MiniMax | `MINIMAX_API_KEY` | MiniMax-M2.7 | 否 |

> Pattern Agent 和 Trend Agent 需要分析 K 线图片，因此必须使用支持视觉的模型。

---

## 常见问题

### Q: API Key 配置后报 403 错误？
检查 Key 是否正确，以及账户余额是否充足。Qwen 服务端在新加坡，可能有网络延迟。

### Q: TA-Lib 安装失败？
参考上面的 TA-Lib 安装说明，确保 C 库已安装。

### Q: 端口 5000 被占用？
修改 `web_interface.py` 末尾的端口号，或关闭占用该端口的进程。

### Q: 如何添加自己的创新功能？
1. 想新增 Agent → 在 `data_providers/` 同级目录下创建新的 `xxx_agent.py`
2. 在 `graph_setup.py` 中添加节点
3. 在 `agent_state.py` 中添加对应状态字段
4. 在 `trading_graph.py` 中初始化

---

## 引用原始工作

```bibtex
@article{xiong2025quantagent,
  title={QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading},
  author={Fei Xiong and Xiang Zhang and Aosong Feng and Siqi Sun and Chenyu You},
  journal={arXiv preprint arXiv:2509.09995},
  year={2025}
}
```

## 免责声明

本项目仅用于学习和研究目的，不构成任何投资建议。进行实际投资决策前，请务必自行研究并咨询专业财务顾问。
