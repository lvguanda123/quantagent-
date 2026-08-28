"""
LangGraph orchestration for the QuantAgent multi-agent trading pipeline.

Pipeline (parallel analysts → cross-check → decision):

    START ──► Indicator Agent ──┐
    START ──► Pattern Agent   ──┼──► Cross-Checker ──► Decision Maker ──► END
    START ──► Trend Agent     ──┘

Three analysts fan out in parallel from ``START`` (LangGraph's built-in join
semantics ensures all three finish before the next node fires). The
Cross-Checker node then judges inter-agent consistency, and the Decision
Maker issues a final LONG/SHORT order.

Each node is wrapped in :func:`_resilient_node` so a single-agent failure
returns a safe fallback instead of crashing the whole graph. When an
:class:`EventCollector` is provided, every node also emits structured
lifecycle events for the expert-mode process visualisation.
"""

import time
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from langgraph.graph import END, START, StateGraph

from agent_state import IndicatorAgentState
from cross_checker import create_cross_checker
from decision_agent import create_final_trade_decider
from graph_util import TechnicalTools
from indicator_agent import create_indicator_agent
from pattern_agent import create_pattern_agent
from trend_agent import create_trend_agent


# ---------------------------------------------------------------------------
# Process event collector (for expert-mode visualisation)
# ---------------------------------------------------------------------------

class EventCollector:
    """Lightweight ordered store for node lifecycle events.

    Events are written by :func:`_resilient_node` and surfaced in the
    final graph state under the ``process_events`` key. The expert output
    template renders them as a process timeline.

    Each event is a dict with the following shape::

        {
            "ts": "2026-08-26T14:23:11.234",   # ISO timestamp with ms
            "node": "Indicator Agent",
            "status": "start" | "end" | "error",
            "duration_ms": 1240,                # only on "end"
            "payload": {...},                    # node-specific summary
        }
    """

    def __init__(self) -> None:
        self.events: List[dict] = []
        self._starts: Dict[str, float] = {}

    def emit(self, node: str, status: str, payload: Optional[dict] = None) -> None:
        event: Dict[str, Any] = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "node": node,
            "status": status,
            "payload": dict(payload or {}),
        }
        if status == "start":
            self._starts[node] = time.time()
        elif status == "end" and node in self._starts:
            event["duration_ms"] = int(
                (time.time() - self._starts.pop(node)) * 1000
            )
        self.events.append(event)


# ---------------------------------------------------------------------------
# Resilient node wrapper
# ---------------------------------------------------------------------------

# Map each node name to (result field, fallback-report default) so we can
# keep the failure paths data-driven instead of one giant if/elif.
_NODE_FALLBACK_REPORTS: Dict[str, str] = {
    "Indicator Agent": "技术指标分析暂不可用，请参考其他分析维度。",
    "Pattern Agent": "K线形态识别暂不可用，请参考其他分析维度。",
    "Trend Agent": "趋势分析暂不可用，请参考其他分析维度。",
}


def _node_fallback(node_name: str, state: dict) -> dict:
    """Return the state update to write when ``node_name`` fails."""
    if node_name == "Indicator Agent":
        return {"indicator_report": _NODE_FALLBACK_REPORTS[node_name]}
    if node_name == "Pattern Agent":
        return {
            "pattern_report": _NODE_FALLBACK_REPORTS[node_name],
        }
    if node_name == "Trend Agent":
        return {
            "trend_report": _NODE_FALLBACK_REPORTS[node_name],
        }
    if node_name == "Cross-Checker":
        return {
            "consensus_score": 0.5,
            "conflicts": "校验节点失败",
            "key_points_summary": "",
            "core_view": "",
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


def _summarize_node_output(node_name: str, out: dict) -> dict:
    """Build a compact payload describing what a node produced (for events)."""
    payload: Dict[str, Any] = {}
    report_field = {
        "Indicator Agent": "indicator_report",
        "Pattern Agent": "pattern_report",
        "Trend Agent": "trend_report",
        "Cross-Checker": "key_points_summary",
        "Decision Maker": "final_trade_decision",
    }.get(node_name)
    if report_field and isinstance(out.get(report_field), str):
        payload["report_chars"] = len(out[report_field])
    if node_name == "Cross-Checker" and "consensus_score" in out:
        payload["consensus_score"] = out["consensus_score"]
    return payload


def _resilient_node(
    node_name: str,
    inner_fn: Callable[[dict], dict],
    collector: Optional[EventCollector] = None,
) -> Callable[[dict], dict]:
    """Wrap an agent node so that a single-agent failure doesn't crash the
    analysis pipeline. On failure, an error event is emitted (if a
    collector is provided) and a placeholder state update is returned so
    downstream nodes can still run with whatever evidence is available.
    """

    def wrapped(state: dict) -> dict:
        if collector is not None:
            collector.emit(node_name, "start")
        try:
            result = inner_fn(state) or {}
        except Exception as exc:  # noqa: BLE001
            print(f"[{node_name}] failed: {exc}")
            traceback.print_exc()
            if collector is not None:
                collector.emit(node_name, "error", {"error": str(exc)})
            return _node_fallback(node_name, state)
        if collector is not None:
            collector.emit(node_name, "end", _summarize_node_output(node_name, result))
        return result

    return wrapped


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

class SetGraph:
    """Build the LangGraph ``StateGraph`` for the trading pipeline.

    Parameters
    ----------
    agent_llm, pattern_llm, trend_llm, decision_llm : ChatModel
        Per-agent LLM instances. ``pattern_llm`` and ``trend_llm`` should be
        vision-capable; the others can be text-only.
    cross_check_llm : ChatModel, optional
        LLM used by the Cross-Checker node. Defaults to ``agent_llm`` when
        ``None`` is passed.
    toolkit : TechnicalTools
        Indicator tool implementations shared across analysts.
    event_collector : EventCollector, optional
        When provided, every node emits lifecycle events to it; the
        resulting ``EventCollector.events`` is written back into the final
        state under ``process_events``.
    """

    _ANALYST_NODES = ("Indicator Agent", "Pattern Agent", "Trend Agent")

    def __init__(
        self,
        agent_llm,
        pattern_llm,
        trend_llm,
        decision_llm,
        toolkit: TechnicalTools,
        cross_check_llm=None,
        event_collector: Optional[EventCollector] = None,
    ):
        self.agent_llm = agent_llm
        self.pattern_llm = pattern_llm
        self.trend_llm = trend_llm
        self.decision_llm = decision_llm
        self.cross_check_llm = cross_check_llm if cross_check_llm is not None else agent_llm
        self.toolkit = toolkit
        self.event_collector = event_collector

    def set_graph(self):
        # Create analyst node functions (each already resilient-wrapped below)
        indicator_inner = create_indicator_agent(self.agent_llm, self.toolkit)
        pattern_inner = create_pattern_agent(
            self.agent_llm, self.pattern_llm, self.toolkit
        )
        trend_inner = create_trend_agent(
            self.agent_llm, self.trend_llm, self.toolkit
        )
        cross_check_inner = create_cross_checker(self.cross_check_llm)
        decision_inner = create_final_trade_decider(self.decision_llm)

        graph = StateGraph(IndicatorAgentState)

        # Each node is wrapped in the resilient + event-emitting decorator
        graph.add_node(
            "Indicator Agent",
            _resilient_node("Indicator Agent", indicator_inner, self.event_collector),
        )
        graph.add_node(
            "Pattern Agent",
            _resilient_node("Pattern Agent", pattern_inner, self.event_collector),
        )
        graph.add_node(
            "Trend Agent",
            _resilient_node("Trend Agent", trend_inner, self.event_collector),
        )
        graph.add_node(
            "Cross-Checker",
            _resilient_node("Cross-Checker", cross_check_inner, self.event_collector),
        )
        graph.add_node(
            "Decision Maker",
            _resilient_node("Decision Maker", decision_inner, self.event_collector),
        )

        # Fan-out: each analyst starts independently from START
        for analyst in self._ANALYST_NODES:
            graph.add_edge(START, analyst)

        # Fan-in: all three analysts must finish before Cross-Checker fires.
        # LangGraph's built-in join semantics (multiple incoming edges on a
        # single node) make this an implicit barrier — no Send / conditional
        # edges required.
        for analyst in self._ANALYST_NODES:
            graph.add_edge(analyst, "Cross-Checker")

        graph.add_edge("Cross-Checker", "Decision Maker")
        graph.add_edge("Decision Maker", END)

        return graph.compile()

    def get_event_collector(self) -> Optional[EventCollector]:
        """Return the collector used to instrument the graph (or None)."""
        return self.event_collector
