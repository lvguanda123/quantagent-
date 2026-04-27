# QuantAgent Web 端优化 — 工作进度汇报

## 一、工作目标

在完成 7 项后端核心改动（决策字段、结构化指标存储、JSON 解析修复等）的基础上，**将全部后端能力接入 Web 交互界面**，确保：
- 分析结果能正确跳转到结果页并完整渲染
- 新增字段（入场价、止损价、止盈价、风险收益比）在前端展示
- 技术指标数值表、K线图表、趋势图表均能正常显示
- LLM 输出的决策内容以纯中文 JSON 格式展示

---

## 二、完成的工作清单

### 1. 决策 Agent 全中文输出 (`decision_agent.py`)

**问题**：LLM 返回的决策内容包含英文 JSON 字段名（`"entry_price"`、`"justification"` 等）和英文提示词，前端展示不友好。

**改动**：
- 将系统提示词全部从英文改为中文，包括角色描述、决策策略、输出格式说明
- JSON 模板字段名全部中文化：`"入场价格"`、`"止损价格"`、`"止盈价格"`、`"风险收益比"`、`"理由"`、`"预测周期"`、`"决策"`
- 要求 LLM 直接输出纯 JSON 文本，不再使用 ```json``` 代码块标记
- 解析逻辑 `_extract_json()` 支持**中英双字段名兼容**，确保向后兼容

**涉及文件**：
- `decision_agent.py`（prompt + 解析逻辑）

### 2. Agent 状态扩展 (`agent_state.py`)

**问题**：后端已产生 `risk_reward_ratio`、`justification`、`forecast_horizon` 等字段，但状态定义中缺失。

**改动**：
- 新增 `risk_reward_ratio`、`justification`、`forecast_horizon` 三个 `Annotated` 字段定义

**涉及文件**：
- `agent_state.py`

### 3. Web 接口后端增强 (`web_interface.py`)

**改动**：
- **JSON 代码块清理**：在 `/api/analyze` 路由中增加正则匹配，自动剥离 LLM 返回的 ```json``` 代码块标记，并将 JSON 格式化为美观的缩进格式（`ensure_ascii=False, indent=2`）
- **静态文件服务**：添加 `static_folder="static"` 配置和 `send_from_directory` 导入，使 Flask 能正确提供 Logo 等静态资源
- **Logo 资源部署**：将 `darklogo.png` 复制到 `static/assets/` 目录

**涉及文件**：
- `web_interface.py`（路由逻辑 + Flask 配置）
- `static/assets/darklogo.png`（新增资源文件）

### 4. 输出页面渲染逻辑重写 (`templates/output.html`)

这是改动最大的部分。原 JS 渲染逻辑存在选择器脆弱、部分面板数据无法注入的问题。

**核心问题**：
- 页面通过 Jinja2 模板渲染默认空状态（显示"等待运行分析..."）
- sessionStorage 数据通过 JS 注入，但原选择器依赖 `nth-of-type` 索引，容易错位
- K线图表和趋势图表的 `<img>` 元素 `alt` 属性为英文，与 JS 选择器不匹配
- 指标数值表在无数据时 Jinja2 不渲染表格，JS 也未动态创建

**改动内容**：

| 模块 | 改进方式 |
|------|----------|
| **面板定位** | 新增 `findLargePanelByIcon()` 函数，通过 Font Awesome 图标类（`fa-chart-bar`、`fa-table`、`fa-brain` 等）精准定位，替代不稳定的 nth-of-type 索引 |
| **最终决策** | 通过 `fa-bullseye` 图标定位决策面板，使用 `replace(/</g, '&lt;')` 转义 HTML，防止 JSON 内容被浏览器解析 |
| **交易参数卡片** | 自动移除"等待分析"占位段落，有数据时动态填充入场/止损/止盈/风险收益比四个卡片 |
| **技术指标报告** | 通过 `fa-chart-bar` 图标定位，完整替换报告内容 |
| **技术指标数值表** | Jinja2 无数据时不渲染表格 → JS 检测到有指标数据时**动态创建整个表格**（含表头和数据行） |
| **K线形态识别** | 通过 `fa-brain` → `fa-search` 图标链定位左子面板，替换识别结果和可靠性分析 |
| **趋势分析** | 通过 `fa-chart-line` → `fa-analytics` 图标链定位左子面板，替换趋势报告 |
| **图片 alt 修复** | 将 `<img alt="Pattern Analysis Chart">` 改为 `alt="K线形态分析图"`，与 JS 的 `querySelector('img[alt="..."]')` 选择器精确匹配 |

**涉及文件**：
- `templates/output.html`（JavaScript 渲染逻辑 + HTML alt 属性）

---

## 三、整体工作流

```
用户选择标的 + 时间范围
        ↓
POST /api/analyze
        ↓
Flask 获取数据 → run_analysis()
        ↓
LangGraph 工作流：
  Indicator Agent → Pattern Agent → Trend Agent → Decision Maker
        ↓
final_state 包含：
  indicator_report, pattern_report, trend_report,
  final_trade_decision, entry_price, stop_loss, take_profit,
  rsi, macd, stoch_k, stoch_d, roc, willr, ...
        ↓
web_interface.py 清洗 final_trade_decision（去代码块标记、格式化 JSON）
        ↓
返回 { redirect: "/output", full_results: {...} }
        ↓
前端 JS：sessionStorage.setItem('analysisResults', ...)
        ↓
window.location.href = "/output"
        ↓
/output 路由渲染 output.html（默认空状态）
        ↓
DOMContentLoaded：读取 sessionStorage，JS 动态注入所有面板内容
        ↓
完整结果展示：
  ✅ 分析概要（标的、数据量、时间范围）
  ✅ 最终交易决策（中文 JSON + 做多/做空徽章）
  ✅ 交易参数（入场/止损/止盈/风险收益比卡片）
  ✅ 技术指标分析报告
  ✅ 技术指标数值表
  ✅ K线形态识别 + 图表
  ✅ 趋势分析 + 图表
```

---

## 四、改动文件汇总

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `decision_agent.py` | 修改 | prompt 全中文 + 双字段名兼容解析 |
| `agent_state.py` | 新增字段 | risk_reward_ratio / justification / forecast_horizon |
| `web_interface.py` | 修改 | JSON 清理 + 静态文件配置 + send_from_directory |
| `templates/output.html` | 重写 | sessionStorage 渲染逻辑 + 图片 alt 修复 |
| `static/assets/darklogo.png` | 新增 | Logo 静态资源 |
