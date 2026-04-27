# QuantAgent 项目介绍与改动总结

## 项目概述

**QuantAgent** 是一个基于多智能体（Multi-Agent）的大语言模型（LLM）高频交易分析系统。该项目通过组合技术指标分析、K线模式识别和趋势分析，为市场提供全面的交易信号。

---

## 项目结构

```
quantagent/
├── 核心智能体
│   ├── indicator_agent.py    # 指标分析智能体（RSI, MACD, Stoch, ROC, Williams %R）
│   ├── pattern_agent.py     # K线模式识别智能体（16种形态）
│   ├── trend_agent.py        # 趋势分析智能体（支撑/阻力线、趋势拟合）
│   └── decision_agent.py    # 决策智能体（综合判断 LONG/SHORT）
│
├── LangGraph 编排
│   ├── trading_graph.py     # 主交易图（多智能体协调）
│   ├── graph_setup.py       # LangGraph 构建
│   └── agent_state.py       # 智能体状态定义
│
├── 技术指标工具
│   ├── graph_util.py        # TA-Lib 封装，技术指标计算
│   └── static_util.py       # 静态工具（图表生成）
│
├── 数据提供者（本次新增）
│   └── data_providers/
│       ├── __init__.py       # 公共 API 导出
│       ├── base.py            # 抽象基类 + 异常 + 规范化管道
│       ├── registry.py         # 注册表 + 工厂函数
│       ├── akshare.py         # AKShare 数据源
│       ├── yahoo.py           # Yahoo Finance 数据源
│       └── qlib.py            # Qlib 本地数据源
│
├── Web 界面
│   ├── web_interface.py     # Flask Web 服务器
│   └── templates/
│       ├── demo_new.html     # 新版 UI（已更新数据源选择）
│       ├── demo.html         # 旧版 UI
│       └── output.html       # 结果展示页
│
├── 配置
│   ├── default_config.py    # 默认配置
│   └── color_style.py      # 颜色样式
│
├── 测试
│   └── tests/
│       ├── test_minimax_integration.py
│       └── test_minimax_provider.py
│
└── 依赖
    └── requirements.txt
```

---

## 技术栈

- **后端**: Python 3.11, Flask, LangChain, LangGraph
- **LLM 提供商**: OpenAI, Anthropic (Claude), Qwen (DashScope), MiniMax
- **数据源**: AKShare, Yahoo Finance, Qlib (本地)
- **技术指标**: TA-Lib, Pandas, NumPy
- **图表**: Matplotlib, mplfinance

---

## 支持的 LLM 模型

| 提供商 | 模型示例 | 用途 |
|--------|----------|------|
| OpenAI | gpt-4o, gpt-4o-mini | 通用分析 |
| Anthropic | claude-3-5-sonnet-20241022 | 通用分析 |
| Qwen | qwen3.6-plus, qwen3-vl-plus | 文本 + 视觉分析 |
| MiniMax | MiniMax-M2.7 | 通用分析 |

---

## 本次新增改动总结

### 1. 新增文件

| 文件路径 | 文件大小 | 说明 |
|----------|----------|------|
| `data_providers/__init__.py` |` ~50 行 | 公共 API 导出，提供 `list_providers()`, `get_provider()`, `FetchRequest` 等 |
| `data_providers/base.py` | ` ~190 行 | 抽象基类 `BaseDataProvider`，数据规范化管道，自定义异常 |
| `data_providers/registry.py` | ` ~145 行 | 提供者注册表 `ProviderRegistry`，工厂函数模式 |
| `data_providers/akshare.py` | ` ~320 行 | AKShare 数据源实现，支持 A股、指数、美股、期货 |
| `data_providers/yahoo.py` | ` ~130 行 | Yahoo Finance 数据源实现 |
| `data_providers/qlib.py` | ` ~270 行 | Qlib 本地数据源实现（修复 .bin 读取 bug） |

### 2. 修改文件

| 文件路径 | 修改内容 | 说明 |
|----------|----------|------|
| `trading_graph.py` | 修复 `SetGraph` 初始化调用 | 修复参数匹配问题，使用命名参数（`agent_llm`, `pattern_llm`, `trend_llm`, `decision_llm`） |
| `web_interface.py` | 无需修改 | 原本有数据获取方法，本次保持兼容，直接使用 AKShare 数据源 |
| `templates/demo_new.html` | 新增数据源选择 UI | 添加 AKShare/Yahoo/Qlib 三选 UI 和相关 JavaScript 功能 |

### 3. 删除文件（临时测试文件）

| 文件路径 | 说明 |
|----------|------|
| `test_cyb_index.py` | 临时测试文件，已删除 |
| `test_cyb_complete.py` | 临时测试文件，已删除 |
| `add_datasource_ui.py` | 临时脚本，已删除 |
| `add_datasource_css_js.py` | 临时脚本，已删除 |

---

## 详细功能说明

### 数据源抽象层

#### 设计模式

采用工厂模式 + 注册表模式，实现了数据源的完全解耦：

```python
# 使用示例
from data_providers import FetchRequest, get_provider, list_providers

# 列出可用数据源
print(list_providers())  # ['akshare', 'qlib', 'yahoo']

# 获取数据提供者
provider = get_provider('akshare')

# 创建获取请求
request = FetchRequest(
    symbol='SZ399006',  # 创业板指
    interval='1d',
    start_date=datetime(2025, 6, 1),
    end_date=datetime(2026, 4, 19)
)

# 获取数据
df = provider.fetch(request)
# 返回标准化的 DataFrame，包含列：Datetime, Open, High, Low, Close, Volume
```

#### 支持的资产类型

**AKShare 数据源**（免费，推荐）：
- A股：`000001`, `600000`, `600519`, `000858` 等
- A股指数：`SH000001`（上证）, `SZ399001`（深证成指）, `SZ399006`（创业板指）
- 美股：`AAPL`, `TSLA`, `GOOGL`, `MSFT`, `AMZN`, `NVDA`, `META`, `NFLX`
- 美股指数：`SPX`（SPY ETF）, `DJI`（DIA ETF）, `NQ`（QQQ ETF）
- 全球指数：`NKY`（日经 225）, `DAX`, `FTSE`, `HSI`, `KOSPI`
- 国际期货：`GC`（黄金）, `CL`（原油）

**Yahoo Finance 数据源**（实时数据，有限流）：
- 任何 Yahoo Finance 支持的资产代码
- 期货需加 `=F` 后缀（如 `NQ=F`）

**Qlib 数据源**（本地A股数据，无限制）：
- 本地 A股数据，路径：`C:\Users\Administrator\.qlib\qlib_data\cn_data_rolling`
- 资产代码格式：`sh600000`, `sh000300`, `sz000001`

#### 数据规范化管道

所有数据源通过基类的 `_normalize()` 方法统一处理：

1. 列名标准化：映射到 `Datetime`, `Open`, `High`, `Low`, `Close`, `Volume`
2. 数据类型转换：确保数值列为 float64，日期列为 datetime64
3. 数据验证：检查必需列存在，验证价格关系（Low <= High, Open/Close 在范围内）
4. 异常处理：统一使用 `DataFetchError` 异常报告错误

### Web 界面新增功能

#### 1. 数据源选择器 UI

在 `demo_new.html` 的"Data Selection"面板中添加了数据源选择功能：

```html
<!-- 数据源选择器 -->
<div class="data-source-selector">
    <button class="data-source-btn active" data-source="akshare" onclick="selectDataSource(this, 'akshare')">
        <i class="fas fa-globe"></i> AKShare
    </button>
    <button class="data-source-btn" data-source="live" onclick="selectDataSource(this, 'live')">
        <i class="fab fa-yahoo"></i> Yahoo Finance
    </button>
    <button class="data-source-btn" data-source="qlib" onclick="selectDataSource(this, 'qlib')">
        <i class="fas fa-database"></i> Qlib (Local A-Share)
    </button>
</div>
```

#### 2. JavaScript 功能

**数据源选择函数**：
```javascript
function selectDataSource(button, dataSource) {
    // 移除所有按钮的 active 类
    document.querySelectorAll('.data-source-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // 为点击的按钮添加 active 类
    button.classList.add('active');
    selectedDataSource = dataSource;

    // 保存选择到 localStorage
    localStorage.setItem(STORAGE_KEYS.SELECTED_DATA_SOURCE, dataSource);

    // 根据数据源更新自定义资产占位符
    update.updateAssetPlaceholder(dataSource);
}
```

**动态占位符更新**：
```javascript
function updateAssetPlaceholder(dataSource) {
    if (dataSource === 'akshare') {
        placeholder = '输入股票代码（如：000001, 600000, SZ399006）';
        examples = '000001 (平安银行), 600000 (浦发银行), SZ399006 (创业板指)';
    } else if (dataSource === 'qlib') {
        placeholder = '输入Qlib股票代码（如：sh600000, sz000001）';
        examples = 'sh600000 (浦发银行), sz000001 (平安银行), sh000300 (沪深300)';
    } else {
        placeholder = '输入Yahoo Finance符号（如：NQ=F, ZN=F, ^VIX）';
        examples = 'NQ (Nasdaq-100 Futures), ZN (10-Year Treasury)';
    }
}
```

#### 3. CSS 样式

为数据源选择器添加了响应式按钮样式：

```css
.data-source-selector {
    display: flex;
    background: var(--gray-100);
    border-radius: 12px;
    gap: 0.5rem;
    padding: 0.5rem;
}

.data-source-btn {
    flex: 1;
    background: transparent;
    border: none;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.95rem;
    font-weight: 500;
    color: var(--gray-700);
    transition: all 0.2s ease;
}

.data-source-btn:hover {
    background: rgba(86, 39, 216, 0.1);
    color: var(--etrade-purple);
    transform: translateY(-1px);
}

.data-source-btn.active {
    background: var(--etrade-purple-light);
    color: var(--white);
    box-shadow: 0 4px 12px rgba(86, 39, 216, 0.3);
    transform: translateY(-1px);
}
```

---

## 测试验证

所有功能已通过以下测试：

```bash
# 1. TradingGraph 初始化测试
venv\Scripts\python.exe -c "from trading_graph import TradingGraph; tg = TradingGraph(); print('OK')"
结果: OK ✅

# 2. 数据源抽象层测试
venv\Scripts\python.exe -c "from data_providers import list_providers, get_provider; print(list_providers())"
结果: ['akshare', 'qlib', 'yahoo'] ✅

# 3. AKShare 数据获取测试（美股）
venv\Scripts\python.exe -c "from data_providers import FetchRequest, get_provider; from datetime import datetime; df = get_provider('akshare').fetch(FetchRequest('AAPL', '1d', datetime(2025,6,1), datetime(2026,4,15))); print(f'{len(df)} rows')"
结果: 219 rows ✅

# 4. Qlib 数据获取测试（A股）
venv\Scripts\python.exe -c "from data_providers import FetchRequest, get_provider; from datetime import datetime; df = get_provider('qlib').fetch(FetchRequest('sh600000', '1d', datetime(2025,1,1), datetime(2026,3,20))); print(f'{len(df)} rows')"
结果: 292 rows ✅

# 5. Web 界面启动测试
venv\Scripts\python.exe web_interface.py
结果: Flask 服务器正常运行在 http://127.0.0.1:5000 ✅

# 6. pytest 测试
venv\Scripts\python.exe -m pytest tests/ -v
结果: 19 passed, 3 skipped in 2.06s ✅
```

### 创业板指测试示例

```bash
# 测试创业板指（SZ399006）数据获取
venv\Scripts\python.exe -c "
from data_providers import FetchRequest, get_provider
from datetime import datetime, timedelta

df = get_provider('akshare').fetch(
    FetchRequest('SZ399006', '1d', 
               datetime.now() - timedelta(days=120), 
               datetime.now())
)

print(f'数据行数: {len(df)}')
print(f'最新收盘价: {df.Close.iloc[-1]:.2f}')
print(f'期间涨跌: {df.Close.iloc[-1] - df.Close.iloc[0]:+.2f}')
"
```

---

## 使用说明

### 启动 Web 界面

```bash
cd e:\代码运行\quantagent
venv\Scripts\python.exe web_interface.py
```

访问：http://127.0.0.1:5000

### 数据源使用流程

1. 在 Web 界面中选择数据源（默认：AKShare）
2. 选择或输入资产代码：
   - AKShare：输入 A股代码（如 `000001`, `SZ399006`）或美股代码（如 `AAPL`）
   - Yahoo Finance：输入 Yahoo 符号（如 `NQ=F`, `BTC`）
   - Qlib：输入本地股票代码（如 `sh600000`）
3. 选择时间周期（默认：1d）
4. 设置日期范围
5. 点击"Analyze"开始分析

### 编程接口使用

```python
from data_providers import FetchRequest, get_provider, list_providers
from datetime import datetime

# 列出所有可用数据源
providers = list_providers()
print("可用数据源:", providers)

# 使用指定数据源获取数据
provider = get_provider('akshare')
df = provider.fetch(FetchRequest(
    symbol='SZ399006',
    interval='1d',
    start_date=datetime(2025, 6, 1),
    end_date=datetime(2026, 4, 19)
))

# 数据已标准化，可直接用于技术分析
print(df.head())
```

---

## 后续开发建议

1. **LLM 模型配置**：
   - 将 `graph_llm_model` 改为视觉模型（如 `qwen3-vl-plus`）以支持图表分析
   - 当前 `qwen3.6-plus` 不支持图像输入

2. **更多数据源**：
   - 可扩展添加 CSMAR、iFinD 等专业数据源
   - 实现其他免费数据接口

3. **缓存优化**：
   - 为频繁访问的数据添加本地缓存
   - 减少重复的 API 调用

4. **错误处理**：
   - 增强网络异常和数据格式错误的用户提示
   - 添加重试机制

5. **单元测试**：
   - 为 `data_providers` 模块添加完整的单元测试
   - 测试覆盖数据规范化、异常处理等边界情况

6. **Web 功能扩展**：
   - 添加实时数据更新功能
   - 支持历史分析记录查询
   - 添加可视化配置选项

---

## 关键文件依赖关系

```
web_interface.py (Flask 后端)
    ├── trading_graph.py (多智能体协调)
    │   ├── graph_setup.py (LangGraph 构建)
    │   ├── indicator_agent.py
    │   ├── pattern_agent.py
    │   ├── trend_agent.py
    │   ├── decision_agent.py
    │   └── agent_state.py
    └── data_providers/ (数据源抽象层)
        ├── base.py
        ├── registry.py
        ├── akshare.py
        ├── yahoo.py
        └── qlib.py
            └── akshare (Python 库)
            └── yfinance (Python 库)
            └── qlib (Python 库)
```

---

## 配置说明

### 默认配置（default_config.py）

```python
DEFAULT_CONFIG = {
    "agent_llm_model": "qwen3.6-plus",       # 文本模型（指标分析）
    "graph_llm_model": "qwen3.6-plus",       # ⚠️ 需改为视觉模型
    "agent_llm_provider": "openai_compatible", # 或 "openai", "anthropic", "qwen", "minimax"
    "graph_llm_provider_provider": "openai_compatible",
    "agent_llm_temperature": 0.1,
    "graph_llm_temperature": 0.1,
    "api_key": "your-api-key-here",           # DashScope API Key
    "base_url": "https://coding.dashscope.aliyuncs.com/v1",
}
```

### 环境变量配置

```bash
# 设置 API Key
export DASHSCOPE_API_KEY="your-dashscope-key"
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export MINIMAX_API_KEY="your-minimax-key"
```

---

## 版本信息

- **项目名称**: QuantAgent
- **当前版本**: 1.1.0
- **更新日期**: 2026-04-19
- **Python 版本**: 3.11.9
- **主要改动**: 添加多数据源支持（AKShare, Yahoo Finance, Qlib）

---

## 联系方式

如有问题或建议，请通过以下方式联系：

1. GitHub Issues
2. 项目文档
3. 开发者邮箱

---

**注意**：本项目仅供学习和研究使用，不构成投资建议。实际交易存在风险，请谨慎决策。
