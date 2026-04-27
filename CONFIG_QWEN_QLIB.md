# QuantAgent 配置说明（Qwen + Qlib 数据）

## 已完成的配置修改

### 1. 默认配置文件 (`default_config.py`)

```python
DEFAULT_CONFIG = {
    "agent_llm_model": "qwen3.6-plus",          # Qwen Agent 模型
    "graph_llm_model": "qwen3-vl-plus",         # Qwen 视觉模型（用于图表分析）
    "agent_llm_provider": "qwen",               # Agent LLM 供应商
    "graph_llm_provider": "qwen",               # Graph LLM 供应商
    "agent_llm_temperature": 0.1,
    "graph_llm_temperature": 0.1,
    "qwen_api_key": "sk-sp-c7815c43449c42a598cd8717b9b3c053",
    "qwen_base_url": "https://coding.dashscope.aliyuncs.com/v1",  # 阿里云 Base URL
}
```

### 2. TradingGraph 修改 (`trading_graph.py`)

- 添加了 `qwen_base_url` 配置项支持
- ChatQwen 初始化时传入 `base_url` 参数

### 3. Web 接口修改 (`web_interface.py`)

- 添加了 `fetch_qlib_data()` 方法，支持从本地 Qlib 数据目录加载数据
- Qlib 数据目录：`C:\Users\Administrator\.qlib\qlib_data\cn_data_rolling`
- `/api/analyze` 端点现在支持 `data_source="qlib"` 参数
- `update_provider` 端点在切换到 Qwen 时自动配置正确的 base_url 和模型

## 使用方法

### 方法 1: 启动 Web 界面

```bash
cd E:\代码运行\quantagent
conda activate quantagents  # 如果已创建环境
python web_interface.py
```

浏览器访问：`http://127.0.0.1:5000`

### 方法 2: 使用 Python 脚本

**测试配置：**
```bash
python test_qwen_config.py
```

**完整示例（Qlib 数据 + Qwen 模型）：**
```bash
python example_qwen_qlib.py
```

### 方法 3: 在代码中直接使用

```python
from default_config import DEFAULT_CONFIG
from trading_graph import TradingGraph

# 使用默认配置（已配置为 Qwen）
config = DEFAULT_CONFIG.copy()
trading_graph = TradingGraph(config=config)

# 准备数据
initial_state = {
    "kline_data": your_data_dict,
    "analysis_results": None,
    "messages": [],
    "time_frame": "1day",
    "stock_name": "BTC",
}

# 运行分析
final_state = trading_graph.graph.invoke(initial_state)

# 获取结果
print(final_state.get("final_trade_decision"))
print(final_state.get("indicator_report"))
print(final_state.get("pattern_report"))
print(final_state.get("trend_report"))
```

## Qlib 数据加载

### 数据结构要求

Qlib 数据目录结构：
```
C:\Users\Administrator\.qlib\qlib_data\cn_data_rolling/
├── day/
│   ├── features/
│   │   └── 000001/          # 股票代码（小写）
│   │       ├── $open.pkl
│   │       ├── $high.pkl
│   │       ├── $low.pkl
│   │       ├── $close.pkl
│   │       └── $volume.pkl
│   └── calendars.txt        # 交易日历
├── 1min/
├── 5min/
└── ...
```

### 加载 Qlib 数据的 API 调用

```javascript
POST /api/analyze
{
    "data_source": "qlib",       // 使用本地 Qlib 数据
    "asset": "000001",           // 股票代码
    "timeframe": "1d",           // 时间周期
    "start_date": "2024-01-01",
    "end_date": "2024-12-31"
}
```

## 模型配置说明

### Qwen 模型选择

| 用途 | 模型 | 说明 |
|------|------|------|
| Agent LLM | `qwen3.6-plus` | 处理技术指标计算和工具调用 |
| Graph LLM | `qwen3-vl-plus` | 视觉语言模型，分析 K 线图表 |

### Base URL

- 使用阿里云 DashScope 的编码服务：`https://coding.dashscope.aliyuncs.com/v1`
- 该 URL 专门用于代码生成和分析任务

## 环境变量

也可以通过环境变量配置（优先级低于配置文件）：

```bash
export DASHSCOPE_API_KEY="sk-sp-c7815c43449c42a598cd8717b9b3c053"
```

## 依赖安装

```bash
pip install -r requirements.txt
```

如果遇到 TA-Lib 安装问题：
```bash
conda install -c conda-forge ta-lib
```

## 验证配置

运行测试脚本：
```bash
python test_qwen_config.py
```

预期输出：
- ✓ 配置验证通过
- ✓ Qlib 数据目录存在
- ✓ TradingGraph 初始化成功
- ✓ WebTradingAnalyzer 初始化成功

## 常见问题

### 1. API Key 无效
检查：
- API Key 是否正确复制
- 账户是否有足够余额
- 网络是否正常

### 2. Qlib 数据未找到
检查：
- 数据目录路径是否正确
- 股票代码格式（小写，如 `000001`）
- 数据频率是否匹配（`day`, `1min`, `5min` 等）

### 3. 模型调用失败
- 确认 Base URL 正确
- 检查模型名称是否支持
- 查看错误日志获取详细信息

## 下一步

1. **运行测试**：先运行 `test_qwen_config.py` 验证配置
2. **尝试分析**：使用 Web 界面或 Python 脚本进行分析
3. **回测验证**：使用历史数据验证策略效果
4. **参数调优**：根据需要调整模型温度、时间周期等参数
