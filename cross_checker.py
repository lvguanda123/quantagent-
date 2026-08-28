"""
Cross-Checker agent node for the QuantAgent multi-agent graph.

Sits between the three analyst agents (Indicator / Pattern / Trend) and the
final Decision Maker. Reads the three reports, asks an LLM to produce a
structured consistency judgement, and writes four fields back into the state:

  * consensus_score     0.0~1.0 inter-agent agreement score
  * conflicts          human-readable conflict notes (1-3 sentences)
  * key_points_summary compressed consensus points (≤200 chars)
  * core_view          one-sentence core view (10-30 chars)

The LLM is asked to emit a single JSON object (no code fences). We parse it
with the same robust regex/json-fallback chain used by ``decision_agent`` so
that malformed output degrades to a safe default rather than crashing the
pipeline.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from decision_agent import _extract_json, _to_text

# Fallback report strings emitted by ``graph_setup._resilient_node`` when an
# analyst fails. We use them to detect "this dimension is unavailable" and
# pre-penalise the consensus score so the Decision Maker weights the missing
# report less.
_FALLBACK_MARKERS = (
    "技术指标分析暂不可用",
    "K线形态识别暂不可用",
    "趋势分析暂不可用",
    "AI决策生成失败",
)

_DEFAULT_CONSENSUS = 0.5
_MISSING_DIM_PENALTY = 0.1
_REPORT_TRUNCATE_CHARS = 1200


def _report_present(text) -> bool:
    """Return False if the report is empty or matches a known fallback marker.

    Accepts string or list (some Anthropic-compatible endpoints return content
    as a list of content blocks) — the latter is normalised via ``_to_text``
    before the marker check so a stray ``"[]"`` doesn't trip us up.
    """
    if not text:
        return False
    if not isinstance(text, str):
        text = _to_text(text)
    if not text or not text.strip():
        return False
    for marker in _FALLBACK_MARKERS:
        if marker in text:
            return False
    return True


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def create_cross_checker(llm):
    """Build a Cross-Checker node function. Returns a state-update dict.

    Parameters
    ----------
    llm : LangChain chat model
        A model with ``.invoke([SystemMessage, HumanMessage])``. Should be a
        fast, instruction-following model — this node is on the critical
        path between the three analysts and the Decision Maker.
    """

    def cross_checker_node(state) -> dict:
        indicator_report = state.get("indicator_report", "") or ""
        pattern_report = state.get("pattern_report", "") or ""
        trend_report = state.get("trend_report", "") or ""

        # Truncate per-report to keep the prompt bounded; full reports can
        # run thousands of chars but the cross-check only needs the headlines.
        ir = indicator_report[:_REPORT_TRUNCATE_CHARS]
        pr = pattern_report[:_REPORT_TRUNCATE_CHARS]
        tr = trend_report[:_REPORT_TRUNCATE_CHARS]

        prompt = (
            "你是多智能体金融分析交叉校验器。请对比三份独立分析报告，输出一致性结论。\n\n"
            "校验维度：\n"
            "1. 方向一致性：三份报告给出的多空方向是否一致？\n"
            "2. 关键价位一致性：支撑位/阻力位/目标价/止损价是否数量级一致？\n"
            "3. 时间窗一致性：预测周期是否冲突？\n\n"
            f"输入：\n"
            f"- 技术指标（{len(indicator_report)} 字）：{ir}\n\n"
            f"- 形态（{len(pattern_report)} 字）：{pr}\n\n"
            f"- 趋势（{len(trend_report)} 字）：{tr}\n\n"
            "仅输出一个 JSON 对象（无代码块标记、无额外文字）：\n"
            '{"consensus_score": 0~1 之间的浮点（1=完全一致，0=严重冲突）,'
            ' "conflicts": "1-3 句具体冲突点描述；无冲突填空字符串",'
            ' "key_points_summary": "三份报告共识要点压缩到 150 字以内",'
            ' "core_view": "一句话核心观点，10-30 字，例：『MACD 金叉 + 双底突破 + 上升趋势线，强烈做多』"}\n'
        )

        try:
            response = llm.invoke([
                SystemMessage(content="你是只输出 JSON 的校验器。"),
                HumanMessage(content=prompt),
            ])
        except Exception:
            # The wrapping _resilient_node in graph_setup will catch the
            # exception and emit an error event; here we just return safe
            # defaults so the Decision Maker can still run.
            return {
                "consensus_score": _DEFAULT_CONSENSUS,
                "conflicts": "校验节点调用失败",
                "key_points_summary": "",
                "core_view": "",
                "messages": [],
            }

        result = _extract_json(_to_text(response.content)) or {}
        try:
            consensus_score = float(result.get("consensus_score", _DEFAULT_CONSENSUS))
        except (TypeError, ValueError):
            consensus_score = _DEFAULT_CONSENSUS

        # Penalise consensus when any dimension is unavailable so the
        # Decision Maker knows to weight that report less.
        missing = sum(
            1
            for text in (indicator_report, pattern_report, trend_report)
            if not _report_present(text)
        )
        if missing > 0:
            consensus_score = _clamp01(consensus_score - missing * _MISSING_DIM_PENALTY)

        return {
            "consensus_score": _clamp01(consensus_score),
            "conflicts": str(result.get("conflicts", "") or "").strip(),
            "key_points_summary": str(result.get("key_points_summary", "") or "").strip(),
            "core_view": str(result.get("core_view", "") or "").strip(),
            "messages": [response],
        }

    return cross_checker_node
