"""Unit tests for the Sohu OpenAI-compatible provider."""

import json
import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

for mod_name in ["talib", "langchain_qwq"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from default_config import DEFAULT_CONFIG


class TestSohuProvider(unittest.TestCase):
    def _make_graph(self, config):
        from trading_graph import TradingGraph

        with patch.object(TradingGraph, "_create_llm", return_value=MagicMock()):
            return TradingGraph(config=config)

    def test_default_config_has_separate_key(self):
        self.assertIn("sohu_api_key", DEFAULT_CONFIG)

    def test_web_analyzer_delegates_api_key_update(self):
        from web_interface import WebTradingAnalyzer

        analyzer = object.__new__(WebTradingAnalyzer)
        analyzer.trading_graph = MagicMock()

        analyzer.update_api_key("test-sohu-key", "sohu")

        analyzer.trading_graph.update_api_key.assert_called_once_with(
            "test-sohu-key", "sohu"
        )

    @patch("web_interface.requests.post")
    def test_sohu_key_validation_uses_chat_completions(self, mock_post):
        from web_interface import WebTradingAnalyzer

        mock_post.return_value.ok = True
        analyzer = object.__new__(WebTradingAnalyzer)
        analyzer.config = {"agent_llm_model": "qwen3.6-plus"}

        analyzer.validate_sohu_api_key("test-sohu-key")

        _, kwargs = mock_post.call_args
        self.assertEqual(
            mock_post.call_args.args[0],
            "https://llm-api-test.tv.sohuno.com/v1/chat/completions",
        )
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-sohu-key")
        self.assertEqual(kwargs["json"]["model"], "qwen3.6-plus")

    def test_sohu_indicator_uses_plain_chat_without_tool_binding(self):
        from indicator_agent import create_indicator_agent

        llm = MagicMock()
        llm.metadata = {"quantagent_provider": "sohu"}
        llm.invoke.return_value = MagicMock(content="指标分析完成")
        toolkit = MagicMock()
        toolkit.compute_macd.invoke.return_value = {"macd": [0.1]}
        toolkit.compute_rsi.invoke.return_value = {"rsi": [50.0]}
        toolkit.compute_roc.invoke.return_value = {"roc": [1.0]}
        toolkit.compute_stoch.invoke.return_value = {"stoch_k": [50.0], "stoch_d": [48.0]}
        toolkit.compute_willr.invoke.return_value = {"willr": [-50.0]}
        node = create_indicator_agent(llm, toolkit)
        values = [float(index) for index in range(1, 46)]

        result = node(
            {
                "time_frame": "1d",
                "kline_data": {
                    "Datetime": [f"2026-05-{(index % 28) + 1:02d}" for index in range(45)],
                    "Open": values,
                    "High": [value + 1 for value in values],
                    "Low": [value - 1 for value in values],
                    "Close": values,
                },
                "messages": [],
            }
        )

        llm.bind_tools.assert_not_called()
        llm.invoke.assert_called_once()
        self.assertEqual(result["indicator_report"], "指标分析完成")
        self.assertIn("rsi", result)

    @patch("trading_graph.requests.post")
    def test_sohu_uses_openai_compatible_base_url(self, mock_post):
        from trading_graph import SohuChatModel

        config = DEFAULT_CONFIG.copy()
        config["sohu_api_key"] = "test-sohu-key"
        graph = self._make_graph(config)

        model = graph._create_llm("sohu", "test-model", 0.1)
        self.assertIsInstance(model, SohuChatModel)
        mock_post.return_value.ok = True
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "", "reasoning_content": "分析内容"}}]
        }

        response = model.invoke("分析")

        self.assertEqual(response.content, "分析内容")
        self.assertEqual(
            mock_post.call_args.args[0],
            "https://llm-api-test.tv.sohuno.com/v1/chat/completions",
        )
        self.assertEqual(
            mock_post.call_args.kwargs["headers"]["Authorization"],
            "Bearer test-sohu-key",
        )
        self.assertTrue(mock_post.call_args.kwargs["stream"])
        self.assertEqual(mock_post.call_args.kwargs["timeout"], (10, 180))

    @patch("trading_graph.requests.post")
    def test_sohu_collects_streamed_content(self, mock_post):
        from trading_graph import SohuChatModel

        response = MagicMock(ok=True)
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "内部推理"}}]},
            {"choices": [{"delta": {"content": "分析"}}]},
            {"choices": [{"delta": {"content": "完成"}}]},
        ]
        response.iter_lines.return_value = [
            f"data: {json.dumps(chunk, ensure_ascii=False)}" for chunk in chunks
        ] + ["data: [DONE]"]
        mock_post.return_value = response

        model = SohuChatModel("test-model", "test-key", 0.1)
        result = model.invoke("分析")

        self.assertEqual(result.content, "分析完成")
        response.json.assert_not_called()

    @patch("trading_graph.ChatOpenAI")
    def test_openai_constructor_is_unchanged(self, mock_openai):
        config = DEFAULT_CONFIG.copy()
        config["api_key"] = "test-openai-key"
        graph = self._make_graph(config)

        graph._create_llm("openai", "gpt-test", 0.2)

        mock_openai.assert_called_once_with(
            model="gpt-test",
            temperature=0.2,
            api_key="test-openai-key",
        )

    def test_update_sohu_key_uses_existing_storage_flow(self):
        config = DEFAULT_CONFIG.copy()
        graph = self._make_graph(config)

        with patch.object(graph, "refresh_llms"):
            graph.update_api_key("new-sohu-key", "sohu")

        self.assertEqual(config["sohu_api_key"], "new-sohu-key")
        self.assertEqual(os.environ["SOHU_API_KEY"], "new-sohu-key")

    @patch("web_interface.analyzer.run_analysis")
    @patch("web_interface.analyzer.fetch_data")
    def test_background_analysis_job_completes(self, mock_fetch, mock_run):
        from web_interface import app

        mock_fetch.return_value = pd.DataFrame(
            {
                "Datetime": pd.to_datetime(["2026-06-18"]),
                "Open": [1.0],
                "High": [2.0],
                "Low": [0.5],
                "Close": [1.5],
                "Volume": [100],
            }
        )
        mock_run.return_value = {
            "success": True,
            "final_state": {},
            "asset_name": "SH000300",
            "data_length": 1,
            "pattern_image": "",
            "trend_image": "",
        }

        client = app.test_client()
        response = client.post(
            "/api/analyze/start",
            json={
                "asset": "SH000300",
                "timeframe": "1d",
                "start_date": "2026-05-23",
                "end_date": "2026-06-22",
            },
        )
        self.assertEqual(response.status_code, 202)
        job_id = response.get_json()["job_id"]

        result = None
        for _ in range(50):
            result = client.get(f"/api/analyze/status/{job_id}").get_json()
            if result["status"] != "running":
                break
            time.sleep(0.01)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["result"]["redirect"], "/output")

    def test_api_key_is_only_revealed_on_explicit_request(self):
        from web_interface import analyzer, app

        previous_key = analyzer.config.get("sohu_api_key", "")
        analyzer.config["sohu_api_key"] = "test-visible-sohu-key"
        try:
            client = app.test_client()
            masked = client.get("/api/get-api-key-status?provider=sohu").get_json()
            revealed = client.get(
                "/api/get-api-key-status?provider=sohu&reveal=1"
            ).get_json()
        finally:
            analyzer.config["sohu_api_key"] = previous_key

        self.assertNotIn("api_key", masked)
        self.assertEqual(revealed["api_key"], "test-visible-sohu-key")


if __name__ == "__main__":
    unittest.main()
