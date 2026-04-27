"""
Agent for technical indicator analysis in high-frequency trading (HFT) context.
Uses LLM and toolkit to compute and interpret indicators like MACD, RSI, ROC, Stochastic, and Williams %R.
"""

import copy
import json

from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def _ensure_dict_args(args):
    """Ensure tool call args is a dict, parsing from JSON string if needed."""
    if isinstance(args, str):
        try:
            return json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return {}
    return args


def create_indicator_agent(llm, toolkit):
    """
    Create an indicator analysis agent node for HFT. The agent uses LLM and indicator tools to analyze OHLCV data.
    """

    def indicator_agent_node(state):
        # --- Tool definitions ---
        tools = [
            toolkit.compute_macd,
            toolkit.compute_rsi,
            toolkit.compute_roc,
            toolkit.compute_stoch,
            toolkit.compute_willr,
        ]
        time_frame = state["time_frame"]
        # --- System prompt for LLM ---
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
        # messages = state["messages"]
        messages = state.get("messages", [])
        if not messages:
            messages = [HumanMessage(content="Begin indicator analysis.")]


        # --- Step 1: Ask for tool calls ---
        ai_response = chain.invoke(messages)
        messages.append(ai_response)
        
        # --- Step 2: Collect tool results ---
        tool_results = {}  # Aggregate structured results from tools
        if hasattr(ai_response, "tool_calls") and ai_response.tool_calls:
            for call in ai_response.tool_calls:
                tool_name = call["name"]
                tool_args = _ensure_dict_args(call.get("args", {}))
                # Always provide kline_data
                tool_args["kline_data"] = copy.deepcopy(state["kline_data"])
                # Lookup tool by name
                tool_fn = next(t for t in tools if t.name == tool_name)
                tool_result = tool_fn.invoke(tool_args)
                # Append result as ToolMessage
                messages.append(
                    ToolMessage(
                        tool_call_id=call["id"], content=json.dumps(tool_result)
                    )
                )
                # Collect structured results for state
                tool_results.update(tool_result)

        # --- Step 3: Re-run the chain with tool results ---
        # Keep invoking until we get a text response (not another tool call)
        # This is important for Claude which may make multiple tool calls
        max_iterations = 5  # Prevent infinite loops
        iteration = 0
        final_response = None
        
        while iteration < max_iterations:
            iteration += 1
            final_response = chain.invoke(messages)
            messages.append(final_response)
            
            # If there are no tool calls, we have the final answer
            if not hasattr(final_response, "tool_calls") or not final_response.tool_calls:
                break
            
            # If there are more tool calls, execute them
            for call in final_response.tool_calls:
                tool_name = call["name"]
                tool_args = _ensure_dict_args(call.get("args", {}))
                tool_args["kline_data"] = copy.deepcopy(state["kline_data"])
                tool_fn = next(t for t in tools if t.name == tool_name)
                tool_result = tool_fn.invoke(tool_args)
                messages.append(
                    ToolMessage(
                        tool_call_id=call["id"], content=json.dumps(tool_result)
                    )
                )
                # Collect structured results for state
                tool_results.update(tool_result)

        # Extract content - handle both string and empty content cases
        if final_response:
            report_content = final_response.content
            # If content is empty or None, try to get text from recent messages
            if not report_content or (isinstance(report_content, str) and not report_content.strip()):
                # Check if there's any text content in the messages (skip tool calls)
                for msg in reversed(messages):
                    if (hasattr(msg, 'content') and msg.content and 
                        isinstance(msg.content, str) and msg.content.strip() and 
                        not hasattr(msg, 'tool_calls')):
                        report_content = msg.content
                        break
        else:
            report_content = "Indicator analysis completed, but no detailed report was generated."

        # Build state update with structured fields from tools
        state_update = {"messages": messages, "indicator_report": report_content if report_content else "Indicator analysis completed."}
        # Write tool results to explicit state fields
        if "rsi" in tool_results:
            state_update["rsi"] = tool_results["rsi"]
        if "macd" in tool_results:
            state_update["macd"] = tool_results["macd"]
        if "macd_signal" in tool_results:
            state_update["macd_signal"] = tool_results["macd_signal"]
        if "macd_hist" in tool_results:
            state_update["macd_hist"] = tool_results["macd_hist"]
        if "stoch_k" in tool_results:
            state_update["stoch_k"] = tool_results["stoch_k"]
        if "stoch_d" in tool_results:
            state_update["stoch_d"] = tool_results["stoch_d"]
        if "roc" in tool_results:
            state_update["roc"] = tool_results["roc"]
        if "willr" in tool_results:
            state_update["willr"] = tool_results["willr"]

        return state_update

    return indicator_agent_node
