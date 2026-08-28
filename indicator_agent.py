"""
Agent for technical indicator analysis in high-frequency trading (HFT) context.
Uses LLM and toolkit to compute and interpret indicators like MACD, RSI, ROC, Stochastic, and Williams %R.

Default path: compute all 5 indicators locally (TA-Lib), then ask the LLM once
to summarise — saves 1-2 round-trips vs the multi-step tool-calling path.
The legacy bind_tools path is kept behind the `indicator_use_tool_calling`
config flag for models that don't accept direct tool inputs.
"""

import copy
import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from decision_agent import _to_text


_INDICATOR_TOOL_NAMES = (
    "compute_macd",
    "compute_rsi",
    "compute_roc",
    "compute_stoch",
    "compute_willr",
)


def _ensure_dict_args(args):
    """Ensure tool call args is a dict, parsing from JSON string if needed."""
    if isinstance(args, str):
        try:
            return json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return {}
    return args


def _is_tool_calling_enabled() -> bool:
    """Read the `indicator_use_tool_calling` flag from DEFAULT_CONFIG.

    Defaults to False (direct compute + single LLM summary). Returning True
    forces the legacy bind_tools multi-step path.
    """
    try:
        from default_config import DEFAULT_CONFIG
    except ImportError:
        return False
    return bool(DEFAULT_CONFIG.get("indicator_use_tool_calling", False))


def _resolve_indicator_tools(toolkit):
    """Return the 5 indicator tool callables in a fixed order."""
    return [
        toolkit.compute_macd,
        toolkit.compute_rsi,
        toolkit.compute_roc,
        toolkit.compute_stoch,
        toolkit.compute_willr,
    ]


def _run_direct_compute(llm, toolkit, state, time_frame):
    """Default path: compute all indicators locally, ask LLM once for a summary.

    Skips the tool-calling loop entirely. TA-Lib runs in <100ms; we then
    pass the last 5 values per indicator to the LLM in a single prompt.
    """
    tools = _resolve_indicator_tools(toolkit)
    tool_results: dict = {}
    for tool_fn in tools:
        tool_results.update(
            tool_fn.invoke({"kline_data": copy.deepcopy(state["kline_data"])})
        )
    compact_results = {
        key: (value[-5:] if isinstance(value, list) else value)
        for key, value in tool_results.items()
    }
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "你是技术指标分析助手。请根据已计算的指标，用中文给出 80-150 字的"
                    "简洁、明确的市场分析，包含：1) 方向倾向（偏多/偏空/震荡）；"
                    "2) 关键阻力支撑价位；3) 风险提示。禁止输出 Markdown 标题。"
                )
            ),
            HumanMessage(
                content=(
                    f"周期：{time_frame}\n最近指标：\n"
                    f"{json.dumps(compact_results, ensure_ascii=False)}"
                )
            ),
        ]
    )
    return {
        "messages": [response],
        "indicator_report": _to_text(response.content),
        **tool_results,
    }


def _run_tool_calling_path(llm, toolkit, state, time_frame):
    """Legacy path: let the LLM call tools via bind_tools, then summarise.

    Kept for models that don't accept direct tool-calling input, or when
    a verbose multi-step reasoning trace is desired.
    """
    tools = _resolve_indicator_tools(toolkit)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a high-frequency trading (HFT) analyst assistant operating under time-sensitive conditions. "
                "You must analyze technical indicators to support fast-paced trading execution.\n\n"
                "You have access to tools: compute_rsi, compute_macd, compute_roc, compute_stoch, and compute_willr. "
                "Use them by providing appropriate arguments like `kline_data` and the respective periods.\n\n"
                f"⚠️ The OHLC data provided is from a {time_frame} intervals, reflecting recent market behavior. "
                "You must interpret this data quickly and accurately.\n\n"
                "Here is the OHLC data:\n{kline_data}.\n\n"
                "Call necessary tools, and analyze the results.\n\n"
                "⚠️ IMPORTANT: Write your entire analysis report in CHINESE (中文). "
                "Use Chinese for all headings, descriptions, and recommendations.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    ).partial(kline_data=json.dumps(state["kline_data"], indent=2))

    chain = prompt | llm.bind_tools(tools)
    messages = state.get("messages", [])
    if not messages:
        messages = [HumanMessage(content="Begin indicator analysis.")]

    # --- Step 1: Ask for tool calls ---
    ai_response = chain.invoke(messages)
    messages.append(ai_response)

    # --- Step 2: Collect tool results ---
    tool_results: dict = {}
    if hasattr(ai_response, "tool_calls") and ai_response.tool_calls:
        for call in ai_response.tool_calls:
            tool_name = call["name"]
            tool_args = _ensure_dict_args(call.get("args", {}))
            tool_args["kline_data"] = copy.deepcopy(state["kline_data"])
            tool_fn = next(t for t in tools if t.name == tool_name)
            tool_result = tool_fn.invoke(tool_args)
            messages.append(
                ToolMessage(tool_call_id=call["id"], content=json.dumps(tool_result))
            )
            tool_results.update(tool_result)

    # --- Step 3: Re-run the chain until the LLM produces a text response ---
    max_iterations = 5
    iteration = 0
    final_response = None

    while iteration < max_iterations:
        iteration += 1
        final_response = chain.invoke(messages)
        messages.append(final_response)
        if not hasattr(final_response, "tool_calls") or not final_response.tool_calls:
            break
        for call in final_response.tool_calls:
            tool_name = call["name"]
            tool_args = _ensure_dict_args(call.get("args", {}))
            tool_args["kline_data"] = copy.deepcopy(state["kline_data"])
            tool_fn = next(t for t in tools if t.name == tool_name)
            tool_result = tool_fn.invoke(tool_args)
            messages.append(
                ToolMessage(tool_call_id=call["id"], content=json.dumps(tool_result))
            )
            tool_results.update(tool_result)

    # Extract content; fall back to the most recent non-tool text message
    if final_response:
        report_content = final_response.content
        if not report_content or (
            isinstance(report_content, str) and not report_content.strip()
        ):
            for msg in reversed(messages):
                if (
                    hasattr(msg, "content")
                    and msg.content
                    and isinstance(msg.content, str)
                    and msg.content.strip()
                    and not hasattr(msg, "tool_calls")
                ):
                    report_content = msg.content
                    break
    else:
        report_content = "Indicator analysis completed, but no detailed report was generated."

    state_update = {
        "messages": messages,
        "indicator_report": _to_text(report_content) if report_content else "Indicator analysis completed.",
    }
    for key in ("rsi", "macd", "macd_signal", "macd_hist", "stoch_k", "stoch_d", "roc", "willr"):
        if key in tool_results:
            state_update[key] = tool_results[key]
    return state_update


def create_indicator_agent(llm, toolkit):
    """
    Create an indicator analysis agent node for HFT.

    The agent computes the 5 standard indicators (RSI / MACD / ROC / Stochastic /
    Williams %R) and asks the LLM to summarise them in 80-150 Chinese chars.
    By default the LLM tool-calling loop is skipped (faster); set
    `indicator_use_tool_calling=True` in `default_config.DEFAULT_CONFIG` to
    fall back to the legacy multi-step bind_tools path.
    """

    def indicator_agent_node(state):
        time_frame = state["time_frame"]
        if _is_tool_calling_enabled():
            return _run_tool_calling_path(llm, toolkit, state, time_frame)
        return _run_direct_compute(llm, toolkit, state, time_frame)

    return indicator_agent_node
