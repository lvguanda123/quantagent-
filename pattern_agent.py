import copy
import json
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from openai import RateLimitError

from decision_agent import _to_text


def _ensure_dict_args(args):
    """Ensure tool call args is a dict, parsing from JSON string if needed."""
    import json
    if isinstance(args, str):
        try:
            return json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return {}
    return args


def invoke_tool_with_retry(tool_fn, tool_args, retries=3, wait_sec=4):
    """
    Invoke a tool function with retries if the result is missing an image.
    """
    for attempt in range(retries):
        result = tool_fn.invoke(tool_args)
        img_b64 = result.get("pattern_image")
        if img_b64:
            return result
        print(
            f"Tool returned no image, retrying in {wait_sec}s (attempt {attempt + 1}/{retries})..."
        )
        time.sleep(wait_sec)
    raise RuntimeError("Tool failed to generate image after multiple retries")


def _model_name(llm) -> str:
    return str(
        getattr(llm, "model_name", "")
        or getattr(llm, "model", "")
        or getattr(llm, "model_id", "")
    ).lower()


def _should_use_text_analysis(llm) -> bool:
    provider_kind = (getattr(llm, "metadata", {}) or {}).get("quantagent_provider")

    model = _model_name(llm)
    # Only models whose name EXPLICITLY signals vision support use image analysis.
    # Everything else falls back to OHLC text analysis — this avoids hanging on
    # endpoints that don't actually support image input (e.g. MiniMax-M3, ark-code-latest).
    vision_markers = (
        "vision",
        "-vl",
        "gpt-4o",
        "gpt-4.1",
        "claude-3",
    )
    if any(marker in model for marker in vision_markers):
        return False

    # Not an explicitly vision-capable model → use text analysis.
    return True


def _is_image_unsupported_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "image_url",
            "vision",
            "image input",
            "expected `text`",
            "unknown variant",
            "multimodal",
        )
    )


def _text_pattern_analysis(graph_llm, state, time_frame, pattern_text):
    compact_data = state["kline_data"]
    bar_count = len(compact_data.get("Close", []))
    response = graph_llm.invoke(
        [
            SystemMessage(
                content="你是K线形态分析助手。请使用中文判断主要形态、方向和风险。"
            ),
            HumanMessage(
                content=(
                    f"周期：{time_frame}\n"
                    f"本次请分析全部 {bar_count} 根OHLC数据，不要只分析最后12根：\n"
                    f"{json.dumps(compact_data, ensure_ascii=False)}\n\n"
                    f"{pattern_text}\n"
                    "请基于完整OHLC数据判断是否存在双底、双顶、突破、反转或整理形态；"
                    "没有明确形态时请直说，并说明支撑、阻力和风险。"
                )
            ),
        ]
    )
    return {
        "messages": [response],
        "pattern_report": _to_text(response.content),
    }


def create_pattern_agent(tool_llm, graph_llm, toolkit):
    """
    Create a pattern recognition agent node for candlestick pattern analysis.
    The agent uses precomputed images from state or falls back to tool generation.
    """

    def pattern_agent_node(state):
        # --- Tool and pattern definitions ---
        tools = [toolkit.generate_kline_image]
        time_frame = state["time_frame"]
        pattern_text = """
        请参考以下经典K线形态（Candlestick Patterns）：

        1. 头肩底（Inverse Head and Shoulders）：三个低点，中间最低，对称结构，通常预示上涨趋势。
        2. 双底（Double Bottom）：两个相似低点，中间反弹，呈"W"形。
        3. 圆弧底（Rounded Bottom）：价格逐渐下跌后逐渐回升，呈"U"形。
        4. 潜伏底（Hidden Base）：水平横盘整理后突然向上突破。
        5. 下降楔形（Falling Wedge）：价格向下收敛，通常向上突破。
        6. 上升楔形（Rising Wedge）：价格上升但逐渐收敛，通常向下突破。
        7. 上升三角形（Ascending Triangle）：支撑线上升，阻力线水平，通常向上突破。
        8. 下降三角形（Descending Triangle）：阻力线下降，支撑线水平，通常向下突破。
        9. 看涨旗形（Bullish Flag）：急涨后短暂向下整理，然后继续上涨。
        10. 看跌旗形（Bearish Flag）：急跌后短暂向上整理，然后继续下跌。
        11. 矩形整理（Rectangle）：价格在水平支撑和阻力之间波动。
        12. 岛形反转（Island Reversal）：两个反向跳空缺口形成孤立价格孤岛。
        13. V形反转（V-shaped Reversal）：急跌后急涨，或反之。
        14. 圆弧顶/圆弧底（Rounded Top/Bottom）：逐渐见顶或见底，呈弧形。
        15. 扩散三角形（Expanding Triangle）：高低点逐渐扩大，波动加剧。
        16. 对称三角形（Symmetrical Triangle）：高低点向顶点收敛，通常随后出现突破。
        """

        # --- Check for precomputed image in state ---
        pattern_image_b64 = state.get("pattern_image")

        if _should_use_text_analysis(graph_llm):
            print("Pattern Agent: model does not support image input, using OHLC text analysis.")
            return _text_pattern_analysis(graph_llm, state, time_frame, pattern_text)

        # --- Retry wrapper for LLM invocation ---
        def invoke_with_retry(call_fn, *args, retries=3, wait_sec=8):
            last_error = None
            for attempt in range(retries):
                try:
                    return call_fn(*args)
                except RateLimitError:
                    last_error = None
                    print(
                        f"Rate limit hit, retrying in {wait_sec}s (attempt {attempt + 1}/{retries})..."
                    )
                    time.sleep(wait_sec)
                except Exception as e:
                    last_error = e
                    if _is_image_unsupported_error(e):
                        raise
                    print(
                        f"Other error: {e}, retrying in {wait_sec}s (attempt {attempt + 1}/{retries})..."
                    )
                    time.sleep(wait_sec)
            raise RuntimeError(f"Max retries exceeded: {last_error}") from last_error

        messages = state.get("messages", [])

        # --- If no precomputed image, fall back to tool generation ---
        if not pattern_image_b64:
            print(
                "No precomputed pattern image found in state, generating with tool..."
            )

            # --- System prompt setup for tool generation ---
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a trading pattern recognition assistant tasked with identifying classical high-frequency trading patterns. "
                        "You have access to tool: generate_kline_image. "
                        "Use it by providing appropriate arguments like `kline_data`\n\n"
                        "Once the chart is generated, compare it to classical pattern descriptions and determine if any known pattern is present.",
                    ),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            ).partial(kline_data=json.dumps(state["kline_data"], indent=2))

            chain = prompt | tool_llm.bind_tools(tools)

            # --- Step 1: First LLM call to determine tool usage ---
            ai_response = invoke_with_retry(chain.invoke, messages)
            messages.append(ai_response)

            # --- Step 2: Handle tool call (generate_kline_image) ---
            if hasattr(ai_response, "tool_calls"):
                for call in ai_response.tool_calls:
                    tool_name = call["name"]
                    tool_args = _ensure_dict_args(call.get("args", {}))
                    # Always provide kline_data
                    tool_args["kline_data"] = copy.deepcopy(state["kline_data"])
                    tool_fn = next(t for t in tools if t.name == tool_name)
                    tool_result = invoke_tool_with_retry(tool_fn, tool_args)
                    pattern_image_b64 = tool_result.get("pattern_image")
                    messages.append(
                        ToolMessage(
                            tool_call_id=call["id"], content=json.dumps(tool_result)
                        )
                    )
        else:
            print("Using precomputed pattern image from state")

        # --- Step 3: Vision analysis with image (precomputed or generated) ---
        if pattern_image_b64:
            image_prompt = [
                {
                    "type": "text",
                    "text": (
                        f"This is a {time_frame} candlestick chart generated from recent OHLC market data.\n\n"
                        f"{pattern_text}\n\n"
                        "Determine whether the chart matches any of the patterns listed. "
                        "Clearly name the matched pattern(s), and explain your reasoning based on structure, trend, and symmetry.\n\n"
                        "⚠️ IMPORTANT: Write your entire analysis report in CHINESE (中文). "
                        "Use Chinese for all headings, descriptions, and recommendations."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{pattern_image_b64}"},
                },
            ]

            # Create messages - ensure HumanMessage has valid content
            # For Anthropic, SystemMessage is extracted separately, but messages array must have at least one message
            human_msg = HumanMessage(content=image_prompt)
            
            # Verify HumanMessage content is valid
            if not human_msg.content:
                raise ValueError("HumanMessage content is empty")
            if isinstance(human_msg.content, list) and len(human_msg.content) == 0:
                raise ValueError("HumanMessage content list is empty")
            
            messages = [
                SystemMessage(
                    content="You are a trading pattern recognition assistant tasked with analyzing candlestick charts."
                ),
                human_msg,
            ]
            
            try:
                final_response = invoke_with_retry(
                    graph_llm.invoke,
                    messages,
                )
            except Exception as e:
                error_str = str(e)
                if _is_image_unsupported_error(e):
                    print("Pattern Agent: image input rejected, falling back to OHLC text analysis.")
                    return _text_pattern_analysis(graph_llm, state, time_frame, pattern_text)
                # Handle Anthropic's "at least one message is required" error
                # This can happen when SystemMessage extraction leaves empty messages array
                if "at least one message" in error_str.lower():
                    # Retry with only HumanMessage (SystemMessage will be lost but Anthropic should work)
                    print("Retrying with HumanMessage only due to Anthropic message conversion issue...")
                    final_response = invoke_with_retry(
                        graph_llm.invoke,
                        [human_msg],
                    )
                else:
                    raise
        else:
            # If no image was generated, fall back to reasoning with messages
            final_response = invoke_with_retry(chain.invoke, messages)

        return {
            "messages": messages + [final_response],
            "pattern_report": _to_text(final_response.content),
        }

    return pattern_agent_node
