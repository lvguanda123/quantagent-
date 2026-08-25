import traceback
from typing import Dict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent_state import IndicatorAgentState
from decision_agent import create_final_trade_decider
from graph_util import TechnicalTools
from indicator_agent import create_indicator_agent
from pattern_agent import create_pattern_agent
from trend_agent import create_trend_agent


def _resilient_node(node_name: str, inner_fn, fallback_report: str):
    """Wrap an agent node so that a single-agent failure doesn't crash the whole
    analysis pipeline. On failure, the error is printed and the node returns a
    placeholder report so downstream agents (and the decision maker) can still
    run with the remaining evidence."""

    def wrapped(state):
        try:
            return inner_fn(state)
        except Exception as exc:  # noqa: BLE001
            print(f"[{node_name}] failed: {exc}")
            traceback.print_exc()
            if node_name == "Indicator Agent":
                return {
                    "indicator_report": fallback_report,
                }
            if node_name == "Pattern Agent":
                return {
                    "pattern_report": fallback_report,
                    "messages": state.get("messages", []),
                }
            if node_name == "Trend Agent":
                return {
                    "trend_report": fallback_report,
                    "messages": state.get("messages", []),
                }
            if node_name == "Decision Maker":
                return {
                    "final_trade_decision": (
                        '{"决策": "观望", "入场价格": 0, "止损价格": 0, '
                        '"止盈价格": 0, "理由": "AI决策生成失败，请根据上方三份分析报告自行判断。"}'
                    ),
                    "decision": "观望",
                    "decision_direction": "观望",
                }
            return {}

    return wrapped


class SetGraph:
    def __init__(
        self,
        agent_llm: ChatOpenAI,      # Indicator Agent
        pattern_llm: ChatOpenAI,     # Pattern Agent
        trend_llm: ChatOpenAI,      # Trend Agent
        decision_llm: ChatOpenAI,   # Decision Agent
        toolkit: TechnicalTools,
        # tool_nodes: Dict[str, ToolNode],
    ):
        self.agent_llm = agent_llm
        self.pattern_llm = pattern_llm
        self.trend_llm = trend_llm
        self.decision_llm = decision_llm
        self.toolkit = toolkit
        # self.tool_nodes = tool_nodes

    def set_graph(self):
        # Create analyst nodes
        agent_nodes = {}
        tool_nodes = {}
        all_agents = ["indicator", "pattern", "trend"]

        # create nodes for indicator agent (uses agent_llm for indicator calculations)
        indicator_inner = create_indicator_agent(self.agent_llm, self.toolkit)
        agent_nodes["indicator"] = _resilient_node(
            "Indicator Agent",
            indicator_inner,
            "技术指标分析暂不可用，请参考其他分析维度。",
        )

        # create nodes for pattern agent (uses agent_llm + graph_llm for vision)
        pattern_inner = create_pattern_agent(
            self.agent_llm, self.pattern_llm, self.toolkit
        )
        agent_nodes["pattern"] = _resilient_node(
            "Pattern Agent",
            pattern_inner,
            "K线形态识别暂不可用，请参考其他分析维度。",
        )

        # create nodes for trend agent (uses agent_llm + graph_llm for vision)
        trend_inner = create_trend_agent(
            self.agent_llm, self.trend_llm, self.toolkit
        )
        agent_nodes["trend"] = _resilient_node(
            "Trend Agent",
            trend_inner,
            "趋势分析暂不可用，请参考其他分析维度。",
        )

        # create nodes for decision agent (uses decision_llm for final decision)
        decision_inner = create_final_trade_decider(self.decision_llm)
        decision_agent_node = _resilient_node(
            "Decision Maker",
            decision_inner,
            "",
        )

        # create graph
        graph = StateGraph(IndicatorAgentState)

        # add agent nodes to graph
        for agent_type, cur_node in agent_nodes.items():
            graph.add_node(f"{agent_type.capitalize()} Agent", cur_node)

        # add rest of the nodes
        graph.add_node("Decision Maker", decision_agent_node)

        # set start of graph
        graph.add_edge(START, "Indicator Agent")

        # add edges to graph
        for i, agent_type in enumerate(all_agents):
            current_agent = f"{agent_type.capitalize()} Agent"

            if i == len(all_agents) - 1:
                graph.add_edge(current_agent, "Decision Maker")
            else:

                next_agent = f"{all_agents[i + 1].capitalize()} Agent"
                graph.add_edge(current_agent, next_agent)

        # Decision Maker Process
        graph.add_edge("Decision Maker", END)

        return graph.compile()
