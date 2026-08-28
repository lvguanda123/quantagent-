"""
Agent for making final trade decisions.

Combines indicator, pattern, and trend reports to issue a LONG / SHORT order.
The prompt is selected at runtime based on the analysis timeframe:

  * sub-hour (``1m``/``5m``/``15m``/``30m``/``60m``/``1h``/...)  → HFT mode,
    forbids holding (观望) and demands an immediate 做多/做空.
  * daily or longer (``1d``/``1w``/``1mo``/...)               → regular analysis
    mode, allows 观望 when signals are weak, asks for entry / SL / TP that
    reflect a longer holding horizon.
"""

import json
import re

import requests
from langchain_core.messages import AIMessage


# Timeframes shorter than this list (and any string ending in m / min / h / hour)
# trigger the HFT prompt.  Anything else falls back to regular analysis.
_LONG_TERM_TIMEFRAMES = frozenset({"1d", "1w", "1wk", "1mo", "1mth", "1y", "1yr",
                                    "day", "week", "month", "year"})


def _to_text(content) -> str:
    """Normalise an LLM response's .content to plain text.

    Some Anthropic-compatible endpoints (e.g. 火山方舟 Ark) return content as a
    list of content blocks like [{"type": "text", "text": "..."}] instead of a
    plain string. Join those blocks so downstream parsing works either way.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content") or ""
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(block))
        return "".join(parts)
    return "" if content is None else str(content)


def _extract_json(text) -> dict | None:
    """Extract JSON block from LLM response text."""
    text = _to_text(text)
    # Try direct JSON parse first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # Try to find JSON in code block
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Fall back to the first {...} object in the text
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _is_high_frequency_timeframe(time_frame: str) -> bool:
    """Return True for sub-hour timeframes that should use the HFT prompt.

    A timeframe is treated as high-frequency when it ends in ``m`` / ``min`` /
    ``h`` / ``hour`` (e.g. ``15min``, ``1h``, ``4hour``). Anything else —
    including unknown strings — is treated as a longer horizon so we never
    accidentally force HFT semantics on a daily chart.
    """
    tf = (time_frame or "").lower().strip()
    if not tf:
        return False
    if tf in _LONG_TERM_TIMEFRAMES:
        return False
    for suffix in ("min", "hour", "m", "h"):
        if tf.endswith(suffix) and len(tf) > len(suffix):
            return True
    return False


def create_final_trade_decider(llm):
    """
    Create a trade decision agent node. The agent uses LLM to synthesize indicator, pattern, and trend reports
    and outputs a final trade decision (LONG or SHORT) with justification and risk-reward ratio.
    """

    def trade_decision_node(state) -> dict:
        # Truncate the three analyst reports so the prompt stays bounded.
        # Full reports can run thousands of chars; the decision only needs the
        # headline signals. Cross-Checker has already compressed the key points.
        indicator_report = (state.get("indicator_report", "") or "")[:800]
        pattern_report = (state.get("pattern_report", "") or "")[:800]
        trend_report = (state.get("trend_report", "") or "")[:800]

        # Consensus fields populated by the Cross-Checker node (may be missing
        # if the graph is run in legacy mode without that node).
        consensus_score = float(state.get("consensus_score", 0.5) or 0.5)
        conflicts = (state.get("conflicts", "") or "").strip()
        core_view = (state.get("core_view", "") or "").strip()
        key_points = (state.get("key_points_summary", "") or "").strip()

        time_frame = state["time_frame"]
        stock_name = state["stock_name"]

        is_sohu = (getattr(llm, "metadata", {}) or {}).get("quantagent_provider") == "sohu"
        is_hft = _is_high_frequency_timeframe(time_frame)

        if is_sohu:
            # Sohu fast-path: stay short even in long-term mode (text budget).
            decision_hint = "仅做多/做空" if is_hft else "可为做多/做空/观望"
            prompt = f"""你是量化交易决策助手。标的 {stock_name} 周期 {time_frame}（{'高频' if is_hft else '常规'}）。
共识评分：{consensus_score:.2f}（1=完全一致）；冲突：{conflicts or '无'}；
核心观点：{core_view}。
技术指标：{indicator_report}
形态：{pattern_report}
趋势：{trend_report}
要点：{key_points}
请输出 JSON：预测周期/决策/入场价格/止损价格/止盈价格/理由/风险收益比。决策{decision_hint}。
"""
        elif is_hft:
            # --- System prompt for LLM (HFT mode: sub-hour timeframes) ---
            prompt = f"""你是高频量化交易（HFT）分析师。当前正在分析 {stock_name} 的 {time_frame} K线，**禁止观望（HOLD）**，必须输出做多或做空。

【共识信息】（由 Cross-Checker 节点提供）
- 共识评分：{consensus_score:.2f} / 1.0（1=完全一致，0=严重冲突）
- 冲突点：{conflicts or '无'}
- 核心观点：{core_view or '无'}
- 共识要点：{key_points or '无'}

【权重建议】
- consensus_score >= 0.8：三份报告强共识，可积极行动
- 0.5 <= consensus_score < 0.8：分歧存在，优先跟随主导趋势
- consensus_score < 0.5：严重冲突，**降低仓位权重**（在理由中说明）

【三份报告】（已截断到 800 字以内）

技术指标：
{indicator_report}

形态分析：
{pattern_report}

趋势分析：
{trend_report}

【决策策略】
1. 仅基于已确认信号行动，避免新兴、推测性或冲突信号。
2. 共识强时按共识方向；冲突时选择有动量支撑的一侧。
3. 风险收益比 1.2~1.8，基于当前波动率和趋势强度。

【输出格式】（仅一个 JSON 对象，无代码块标记、无额外文字）：
{{
"预测周期": "预测下 N 根K线（如15分钟、1小时、1天等）",
"决策": "做多 或 做空",
"入场价格": "浮点数",
"止损价格": "浮点数",
"止盈价格": "浮点数",
"理由": "基于报告和共识的简明确认性推理；如共识低请说明降仓逻辑",
"风险收益比": "1.2 到 1.8 之间的浮点数"
}}

⚠️ 整个输出必须使用中文，包括所有字段名和理由。直接输出纯JSON文本。
"""
        else:
            # --- System prompt for LLM (regular analysis mode: daily / weekly / monthly) ---
            prompt = f"""你是一位常规技术分析师。当前正在分析 {stock_name} 的 {time_frame} K线。请综合三份独立报告给出**做多、做空或观望**的判断（注意：日线/周线等长周期下，**允许观望**——若信号不明确，优先输出"观望"避免强行交易）。

【共识信息】（由 Cross-Checker 节点提供）
- 共识评分：{consensus_score:.2f} / 1.0（1=完全一致，0=严重冲突）
- 冲突点：{conflicts or '无'}
- 核心观点：{core_view or '无'}
- 共识要点：{key_points or '无'}

【决策指引】
- consensus_score >= 0.8 且方向明确：可积极做多/做空
- 0.5 <= consensus_score < 0.8：方向需谨慎，建议只在关键位附近行动，否则观望
- consensus_score < 0.5：严重冲突，**默认输出"观望"**，理由中说明冲突点

【三份报告】（已截断到 800 字以内）

技术指标：
{indicator_report}

形态分析：
{pattern_report}

趋势分析：
{trend_report}

【分析策略】
1. 重视长期趋势的延续性，忽略短期噪音。
2. 关键支撑/阻力位的突破或跌破是重要信号。
3. 形态要求清晰且已确认，模糊形态不交易。
4. 风险收益比 1.2~2.5 均可（长周期可更宽容）。

【输出格式】（仅一个 JSON 对象，无代码块标记、无额外文字）：
{{
"预测周期": "预测的持仓周期（如未来 1-2 周、未来 1 个月）",
"决策": "做多 / 做空 / 观望",
"入场价格": "浮点数；如观望则填 0",
"止损价格": "浮点数；如观望则填 0",
"止盈价格": "浮点数；如观望则填 0",
"理由": "基于三份报告和共识的简明分析；若观望请说明原因",
"风险收益比": "0~3 之间的浮点数；观望填 0"
}}

⚠️ 整个输出必须使用中文，包括所有字段名和理由。直接输出纯JSON文本。
"""

        # --- LLM call for decision ---
        try:
            response = llm.invoke(prompt)
        except requests.exceptions.RequestException as exc:
            if not is_sohu:
                raise
            response = AIMessage(
                content=(
                    "搜狐模型生成最终交易决策时连接超时。"
                    "技术指标、形态与趋势分析已保留，请稍后重新分析以获取最终交易决策。"
                )
            )

        # Try to parse structured output and extract trade fields
        result = _extract_json(response.content)
        response_text = _to_text(response.content)
        state_update = {
            "final_trade_decision": response_text,
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
