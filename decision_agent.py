"""
Agent for making final trade decisions in high-frequency trading (HFT) context.
Combines indicator, pattern, and trend reports to issue a LONG or SHORT order.
"""

import json
import re


def _extract_json(text: str) -> dict | None:
    """Extract JSON block from LLM response text."""
    # Try direct JSON parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find JSON in code block
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return None


def create_final_trade_decider(llm):
    """
    Create a trade decision agent node. The agent uses LLM to synthesize indicator, pattern, and trend reports
    and outputs a final trade decision (LONG or SHORT) with justification and risk-reward ratio.
    """

    def trade_decision_node(state) -> dict:
        indicator_report = state["indicator_report"]
        pattern_report = state["pattern_report"]
        trend_report = state["trend_report"]
        time_frame = state["time_frame"]
        stock_name = state["stock_name"]

        # --- System prompt for LLM ---
        prompt = f"""你是一位高频量化交易（HFT）分析师，当前正在分析 {stock_name} 的 {time_frame} K线图。你的任务是给出**立即执行指令**：**做多（LONG）**或**做空（SHORT）**。⚠️ 由于是HFT环境，禁止持有观望（HOLD）。

你的决策应预测**未来 N 根K线**的市场走势，其中：
- 例如：TIME_FRAME = 15min, N = 1 → 预测未来15分钟。
- TIME_FRAME = 4hour, N = 1 → 预测未来4小时。

请综合以下三份报告的力量、一致性和时效性做出判断：

---

### 1. 技术指标报告：
- 评估动量指标（如MACD、ROC）和振荡指标（如RSI、随机指标、威廉指标）。
- 对**强烈的方向性信号**给予更高权重，如MACD金叉/死叉、RSI背离、极端超买/超卖。
- 除非多个指标方向一致，否则**忽略或降低中性/混合信号的权重**。

---

### 2. 形态报告：
- 仅当形态满足以下条件时才采取行动：
  - 形态**清晰可辨且基本完成**，且
  - **突破或跌破已启动**或概率极高（如长上/下影线、放量、吞没K线）。
- **不要**对早期或猜测性形态采取行动。未获其他报告确认的盘整阶段不可交易。

---

### 3. 趋势报告：
- 分析价格与支撑/阻力线的关系：
  - **向上倾斜的支撑线**表明买盘兴趣。
  - **向下倾斜的阻力线**表明卖压。
  - 如果价格在趋势线之间压缩：
  - **仅在存在强K线或指标确认的共振时才预测突破方向**。
  - **不要**仅凭几何形态假设突破方向。

---

### ✅ 决策策略

1. 仅基于**已确认**的信号行动——避免新兴、推测性或冲突信号。
2. 优先选择**三份报告（指标、形态、趋势）方向一致**的决策。
3. 更重视：
   - 近期强劲动量（如MACD金叉、RSI突破）
   - 明确的价格行为（如突破K线、拒绝影线、支撑反弹）
4. 如果报告存在分歧：
   - 选择**更强且近期有确认**的方向
   - 优先**有动量支撑的信号**而非微弱的振荡指标暗示。
5. ⚖️ 如果市场处于盘整或报告存在分歧：
   - 默认跟随**主导趋势线斜率**（如下降通道中做空）。
   - 不要猜测方向——选择**更有依据**的一侧。
6. 建议合理的**风险收益比**在 **1.2 到 1.8** 之间，基于当前波动率和趋势强度。

---
### 🧠 输出格式（JSON，供系统解析）：

```
{{
"预测周期": "预测下 N 根K线（如15分钟、1小时、1天等）",
"决策": "做多 或 做空",
"入场价格": "浮点数，基于当前价格和支撑/阻力的建议入场价",
"止损价格": "浮点数，止损价以限制下行风险",
"止盈价格": "浮点数，目标出场价",
"理由": "基于报告的简明确认性推理",
"风险收益比": "1.2 到 1.8 之间的浮点数，做多=(止盈-入场)/(入场-止损)，做空=(入场-止盈)/(止损-入场)"
}}

⚠️ 重要：整个输出必须使用中文（除"LONG"/"SHORT"关键字外），包括所有字段名和理由。不要输出```json```代码块标记，直接输出纯JSON文本。

--------
**技术指标报告**
{indicator_report}

**形态报告**
{pattern_report}

**趋势报告**
{trend_report}

        """

        # --- LLM call for decision ---
        response = llm.invoke(prompt)

        # Try to parse structured output and extract trade fields
        result = _extract_json(response.content)
        state_update = {
            "final_trade_decision": response.content,
            "messages": [response],
            "decision_prompt": prompt,
        }

        # Support both Chinese and English field names
        def _get(key_cn: str, key_en: str):
            """Get value from result dict, trying Chinese key first then English key."""
            if result and key_cn in result:
                return result[key_cn]
            if result and key_en in result:
                return result[key_en]
            return None

        # Extract entry_price
        entry = _get("入场价格", "entry_price")
        if entry is not None:
            try:
                state_update["entry_price"] = float(entry)
            except (ValueError, TypeError):
                pass

        # Extract stop_loss
        sl = _get("止损价格", "stop_loss")
        if sl is not None:
            try:
                state_update["stop_loss"] = float(sl)
            except (ValueError, TypeError):
                pass

        # Extract take_profit
        tp = _get("止盈价格", "take_profit")
        if tp is not None:
            try:
                state_update["take_profit"] = float(tp)
            except (ValueError, TypeError):
                pass

        # Extract risk_reward_ratio and justification if present
        rr = _get("风险收益比", "risk_reward_ratio")
        if rr is not None:
            try:
                state_update["risk_reward_ratio"] = float(rr)
            except (ValueError, TypeError):
                pass

        justification = _get("理由", "justification")
        if justification:
            state_update["justification"] = justification

        forecast = _get("预测周期", "forecast_horizon")
        if forecast:
            state_update["forecast_horizon"] = forecast

        decision = _get("决策", "decision")
        if decision:
            state_update["decision"] = decision

        return state_update

    return trade_decision_node
