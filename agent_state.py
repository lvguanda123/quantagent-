from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class IndicatorAgentState(TypedDict):
    """State type for the Indicator Agent including messages, input data, and analysis result."""

    kline_data: Annotated[
        dict, "OHLCV dictionary used for computing technical indicators"
    ]
    time_frame: Annotated[str, "time period for k line data provided"]
    stock_name: Annotated[dict, "stock name for prompt"]

    # Indicator Agent Tools output values (explicitly added per indicator)
    rsi: Annotated[List[float], "Relative Strength Index values"]
    macd: Annotated[List[float], "MACD line values"]
    macd_signal: Annotated[List[float], "MACD signal line values"]
    macd_hist: Annotated[List[float], "MACD histogram values"]
    stoch_k: Annotated[List[float], "Stochastic Oscillator %K values"]
    stoch_d: Annotated[List[float], "Stochastic Oscillator %D values"]
    roc: Annotated[List[float], "Rate of Change values"]
    willr: Annotated[List[float], "Williams %R values"]
    indicator_report: Annotated[
        str, "Final indicator agent summary report to be used by downstream agents"
    ]

    # Pattern Agent
    pattern_image: Annotated[
        str, "Base64-encoded K-line chart for pattern recognition agent use"
    ]
    pattern_image_filename: Annotated[
        str, "Local file path to saved K-line chart image"
    ]
    pattern_image_description: Annotated[
        str, "Brief description of the generated K-line image"
    ]
    pattern_report: Annotated[
        str, "Final pattern agent summary report to be used by downstream agents"
    ]

    # Trend Agent
    trend_image: Annotated[
        str,
        "Base64-encoded trend-annotated candlestick (K-line) chart for trend recognition agent use",
    ]
    trend_image_filename: Annotated[
        str, "Local file path to saved trendline-enhanced K-line chart image"
    ]
    trend_image_description: Annotated[
        str,
        "Brief description of the chart, including presence of support/resistance lines and visual characteristics",
    ]
    trend_report: Annotated[
        str,
        "Final trend analysis summary, describing structure, directional bias, and technical observations for downstream agents",
    ]

    # Final analysis and messaging context
    analysis_results: Annotated[str, "Computed result of the analysis or decision"]
    # `add_messages` reducer lets the three parallel analysts all write to
    # `messages` in the same step; without it, LangGraph raises
    # INVALID_CONCURRENT_GRAPH_UPDATE.
    messages: Annotated[List[BaseMessage], add_messages]
    decision_prompt: Annotated[str, "decision prompt for reflection"]
    final_trade_decision: Annotated[
        str, "Final BUY or SELL decision made after analyzing indicators"
    ]
    entry_price: Annotated[
        float, "Recommended entry price based on current price and support/resistance levels"
    ]
    stop_loss: Annotated[
        float, "Stop-loss price to limit downside risk"
    ]
    take_profit: Annotated[
        float, "Take-profit price for target exit"
    ]
    risk_reward_ratio: Annotated[
        float, "Risk-reward ratio for the trade"
    ]
    justification: Annotated[
        str, "Concise reasoning for the trade decision"
    ]
    forecast_horizon: Annotated[
        str, "Predicted time horizon for the trade"
    ]
    decision: Annotated[
        str, "Trade direction extracted from LLM JSON: 做多 / 做空 / 观望"
    ]

    # Cross-Checker outputs (added for dual-mode + hallucination guard)
    consensus_score: Annotated[
        float, "0~1 inter-agent consistency score; 1 = full agreement"
    ]
    conflicts: Annotated[
        str, "Human-readable conflicts between indicator/pattern/trend reports (1-3 sentences)"
    ]
    key_points_summary: Annotated[
        str, "Compressed consensus points (<=200 chars) for downstream Decision agent"
    ]
    core_view: Annotated[
        str, "One-sentence core view (10-30 chars), e.g. 'MACD 金叉 + 双底突破 + 上升趋势线，强烈做多'"
    ]

    # Process event stream (for expert-mode process visualisation)
    process_events: Annotated[
        List[dict],
        "Ordered list of agent lifecycle events: [{ts, node, status, duration_ms, payload}, ...]",
    ]

    # UI mode selected by user on the home page
    mode: Annotated[
        str, "Report mode selected by user: 'trader' (default) or 'expert'"
    ]
