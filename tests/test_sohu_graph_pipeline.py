"""End-to-end state test for the Sohu text-only graph path."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from decision_agent import create_final_trade_decider
from default_config import DEFAULT_CONFIG
from trading_graph import TradingGraph


class TestSohuGraphPipeline(unittest.TestCase):
    def test_decision_timeout_preserves_existing_ai_reports(self):
        llm = MagicMock()
        llm.metadata = {"quantagent_provider": "sohu"}
        llm.invoke.side_effect = requests.exceptions.ReadTimeout("timed out")
        decide = create_final_trade_decider(llm)

        result = decide(
            {
                "indicator_report": "技术指标AI分析",
                "pattern_report": "形态AI分析",
                "trend_report": "趋势AI分析",
                "time_frame": "1d",
                "stock_name": "SH000300",
            }
        )

        self.assertIn("连接超时", result["final_trade_decision"])
        self.assertIn("技术指标AI分析", result["decision_prompt"])

    @patch("trading_graph.requests.post")
    def test_all_agent_reports_reach_final_state(self, mock_post):
        responses = [
            "技术指标报告",
            "形态识别报告",
            "趋势分析报告",
            '{"决策":"做多","入场价格":4900,"止损价格":4850,"止盈价格":4975,"理由":"测试"}',
        ]

        def response_for_call(*_args, **_kwargs):
            response = MagicMock(ok=True)
            response.json.return_value = {
                "choices": [
                    {"message": {"content": "", "reasoning_content": responses.pop(0)}}
                ]
            }
            return response

        mock_post.side_effect = response_for_call
        config = DEFAULT_CONFIG.copy()
        config.update(
            {
                "agent_llm_provider": "sohu",
                "graph_llm_provider": "sohu",
                "sohu_api_key": "test-key",
            }
        )
        graph = TradingGraph(config=config)
        graph.toolkit.compute_macd = MagicMock()
        graph.toolkit.compute_macd.invoke.return_value = {
            "macd": [0.1], "macd_signal": [0.05], "macd_hist": [0.05]
        }
        graph.toolkit.compute_rsi = MagicMock()
        graph.toolkit.compute_rsi.invoke.return_value = {"rsi": [52.0]}
        graph.toolkit.compute_roc = MagicMock()
        graph.toolkit.compute_roc.invoke.return_value = {"roc": [1.0]}
        graph.toolkit.compute_stoch = MagicMock()
        graph.toolkit.compute_stoch.invoke.return_value = {
            "stoch_k": [70.0], "stoch_d": [65.0]
        }
        graph.toolkit.compute_willr = MagicMock()
        graph.toolkit.compute_willr.invoke.return_value = {"willr": [-40.0]}
        values = [float(index) for index in range(1, 46)]
        state = {
            "kline_data": {
                "Datetime": [f"2026-05-{(index % 28) + 1:02d}" for index in range(45)],
                "Open": values,
                "High": [value + 1 for value in values],
                "Low": [value - 1 for value in values],
                "Close": values,
            },
            "analysis_results": None,
            "messages": [],
            "time_frame": "1d",
            "stock_name": "SH000300",
            "pattern_image": "",
            "trend_image": "",
        }

        result = graph.graph.invoke(state)

        self.assertEqual(result["indicator_report"], "技术指标报告")
        self.assertEqual(result["pattern_report"], "形态识别报告")
        self.assertEqual(result["trend_report"], "趋势分析报告")
        self.assertIn("做多", result["final_trade_decision"])


if __name__ == "__main__":
    unittest.main()
