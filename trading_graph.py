"""
TradingGraph: Orchestrates the multi-agent trading system using LangChain and LangGraph.
Initializes LLMs, toolkits, and agent nodes for indicator, pattern, and trend analysis.
"""

import json
import os
import re
from typing import Dict

import anthropic
import requests
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_qwq import ChatQwen
from langgraph.prebuilt import ToolNode

from default_config import DEFAULT_CONFIG
from graph_setup import SetGraph
from graph_util import TechnicalTools


class SohuChatModel:
    """Minimal OpenAI-compatible client that preserves Sohu reasoning_content."""

    metadata = {"quantagent_provider": "sohu"}

    def __init__(self, model: str, api_key: str, temperature: float):
        self.model_name = model
        self.api_key = api_key
        self.temperature = temperature

    def invoke(self, messages):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            converted = []
            for message in messages:
                role = {
                    "system": "system",
                    "human": "user",
                    "ai": "assistant",
                }.get(getattr(message, "type", "human"), "user")
                converted.append({"role": role, "content": message.content})
            messages = converted

        try:
            response = requests.post(
                "https://llm-api-test.tv.sohuno.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": 512,
                    "enable_thinking": False,
                    "stream": True,
                },
                stream=True,
                timeout=(10, 180),
            )
        except requests.exceptions.Timeout as exc:
            raise RuntimeError("搜狐模型响应超时，请稍后重试") from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"搜狐模型连接失败：{exc}") from exc

        if not response.ok:
            raise ValueError(f"搜狐模型请求失败（HTTP {response.status_code}）：{response.text}")

        content_parts = []
        reasoning_parts = []
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data:"):
                continue
            data = raw_line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = ((chunk.get("choices") or [{}])[0].get("delta") or {})
            if delta.get("content"):
                content_parts.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning_parts.append(delta["reasoning_content"])

        content = "".join(content_parts) or "".join(reasoning_parts)
        if not content:
            # Keep compatibility with non-streaming responses and test doubles.
            payload = response.json()
            message = (payload.get("choices") or [{}])[0].get("message") or {}
            content = message.get("content") or message.get("reasoning_content") or ""
        if not content:
            raise ValueError("搜狐模型返回成功，但响应中没有 content 或 reasoning_content")
        return AIMessage(content=content)


def normalize_openai_compatible_base_url(base_url: str) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    if normalized == "https://ark.cn-beijing.volces.com/api/coding":
        normalized = "https://ark.cn-beijing.volces.com/api/v3"
    return normalized


def _message_to_dict(message) -> dict:
    role = {
        "system": "system",
        "human": "user",
        "ai": "assistant",
    }.get(getattr(message, "type", "human"), "user")
    content = getattr(message, "content", message)
    return {"role": role, "content": _content_to_text(content)}


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif item.get("type") == "image_url":
                    parts.append("[图像内容已省略：当前高级自定义 HTTP 模式按文本方式调用]")
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _render_template(value: str, context: dict) -> str:
    rendered = value or ""
    for key, replacement in context.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", replacement)
    return rendered


def _extract_path(payload, path: str):
    current = payload
    for part in (path or "").split("."):
        if part == "":
            continue
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _first_text_from_payload(payload) -> str:
    for path in [
        "choices.0.message.content",
        "choices.0.text",
        "data.output",
        "data.content",
        "output",
        "result",
        "message",
        "content",
    ]:
        value = _extract_path(payload, path)
        if isinstance(value, str) and value.strip():
            return value
    return ""


class CustomHttpChatModel:
    """Generic text-first HTTP model for non-OpenAI-compatible providers."""

    metadata = {"quantagent_provider": "custom_http"}

    def __init__(self, config: dict, api_key: str, model: str, temperature: float):
        self.config = config
        self.api_key = api_key
        self.model_name = model
        self.temperature = temperature

    def invoke(self, messages):
        if isinstance(messages, str):
            normalized_messages = [{"role": "user", "content": messages}]
        else:
            normalized_messages = [_message_to_dict(message) for message in messages]

        prompt = "\n\n".join(
            f"{message['role']}: {message['content']}" for message in normalized_messages
        )
        context = {
            "api_key": self.api_key,
            "model": self.model_name,
            "temperature": str(self.temperature),
            "messages_json": json.dumps(normalized_messages, ensure_ascii=False),
            "prompt_json": json.dumps(prompt, ensure_ascii=False),
            "prompt": prompt,
        }

        endpoint_url = (self.config.get("custom_endpoint_url") or "").strip()
        if not endpoint_url:
            raise ValueError("请填写高级自定义接口的完整请求 URL")

        headers_template = self.config.get("custom_headers_template") or "{}"
        body_template = self.config.get("custom_body_template") or "{}"

        try:
            headers = json.loads(_render_template(headers_template, context))
            body = json.loads(_render_template(body_template, context))
        except json.JSONDecodeError as exc:
            raise ValueError(f"高级自定义接口的 Header 或 Body 模板不是合法 JSON：{exc}") from exc

        try:
            response = requests.post(endpoint_url, headers=headers, json=body, timeout=(10, 180))
        except requests.exceptions.Timeout as exc:
            raise RuntimeError("高级自定义接口响应超时，请稍后重试") from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"高级自定义接口连接失败：{exc}") from exc

        if not response.ok:
            error_text = response.text.strip()
            if response.status_code == 404:
                hint = (
                    "请确认完整请求 URL 是该厂商真实可 POST 的模型接口。"
                    "如果使用火山 Ark 的 OpenAI 兼容接口，请切回“OpenAI 兼容接口”，"
                    "Base URL 填 https://ark.cn-beijing.volces.com/api/v3，"
                    "模型名称填控制台实际开通的 endpoint/model id。"
                )
                raise ValueError(
                    f"高级自定义接口请求失败（HTTP 404）：{hint}"
                    + (f" 原始返回：{error_text}" if error_text else "")
                )
            raise ValueError(f"高级自定义接口请求失败（HTTP {response.status_code}）：{error_text}")

        try:
            payload = response.json()
        except ValueError as exc:
            raise ValueError("高级自定义接口返回的不是 JSON，无法解析模型回复") from exc

        response_path = self.config.get("custom_response_path") or ""
        content = ""
        if response_path:
            value = _extract_path(payload, response_path)
            if isinstance(value, str):
                content = value
        if not content:
            content = _first_text_from_payload(payload)
        if not content:
            raise ValueError("高级自定义接口返回成功，但未按返回路径找到文本内容")
        return AIMessage(content=content)


class CustomAnthropicChatModel:
    """Generic Anthropic Messages API compatible client using Bearer auth_token."""

    metadata = {"quantagent_provider": "custom_anthropic"}

    def __init__(self, base_url: str, api_key: str, model: str, temperature: float):
        self.base_url = (base_url or "").strip().rstrip("/")
        self.api_key = api_key
        self.model_name = model
        self.temperature = temperature

    def invoke(self, messages):
        if not self.base_url:
            raise ValueError("请填写 Anthropic 兼容接口的 Base URL")

        if isinstance(messages, str):
            normalized_messages = [{"role": "user", "content": messages}]
        else:
            normalized_messages = [_message_to_dict(message) for message in messages]

        system_parts = []
        anthropic_messages = []
        for message in normalized_messages:
            role = message.get("role") or "user"
            content = message.get("content") or ""
            if role == "system":
                system_parts.append(content)
            elif role == "assistant":
                anthropic_messages.append({"role": "assistant", "content": content})
            else:
                anthropic_messages.append({"role": "user", "content": content})

        if not anthropic_messages:
            anthropic_messages = [{"role": "user", "content": "请开始分析。"}]

        client = anthropic.Anthropic(
            auth_token=self.api_key,
            base_url=self.base_url,
            timeout=180,
            max_retries=1,
        )
        request_payload = {
            "model": self.model_name,
            "messages": anthropic_messages,
            "max_tokens": 1024,
            "temperature": self.temperature,
        }
        if system_parts:
            request_payload["system"] = "\n\n".join(system_parts)

        try:
            response = client.messages.create(**request_payload)
        except anthropic.NotFoundError as exc:
            raise ValueError(
                "Anthropic 兼容接口返回 404：请确认 Base URL、模型名，以及该 Key 是否有权限。"
                f" 原始错误：{exc}"
            ) from exc
        except anthropic.APIStatusError as exc:
            raise ValueError(
                f"Anthropic 兼容接口请求失败（HTTP {exc.status_code}）：{exc.response.text}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Anthropic 兼容接口连接失败：{exc}") from exc

        content = "\n".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        ).strip()
        if not content:
            raise ValueError("Anthropic 兼容接口返回成功，但没有文本内容")
        return AIMessage(content=content)


class TradingGraph:
    """
    Main orchestrator for the multi-agent trading system.
    Sets up LLMs, toolkits, and agent nodes for indicator, pattern, and trend analysis.
    """

    def __init__(self, config=None):
        # --- Configuration and LLMs ---
        self.config = config if config is not None else DEFAULT_CONFIG.copy()

        # Initialize LLMs with provider support
        self.agent_llm = self._create_llm(
            provider=self.config.get("agent_llm_provider", "openai"),
            model=self.config.get("agent_llm_model", "gpt-4o-mini"),
            temperature=self.config.get("agent_llm_temperature", 0.1),
        )
        self.graph_llm = self._create_llm(
            provider=self.config.get("graph_llm_provider", "openai"),
            model=self.config.get("graph_llm_model", "gpt-4o"),
            temperature=self.config.get("graph_llm_temperature", 0.1),
        )
        self.toolkit = TechnicalTools()

        # --- Create tool nodes for each agent ---
        # self.tool_nodes = self._set_tool_nodes()

        # --- Graph logic and setup ---
        self.graph_setup = SetGraph(
            agent_llm=self.agent_llm,       # Indicator Agent
            pattern_llm=self.graph_llm,    # Pattern Agent (uses vision model)
            trend_llm=self.graph_llm,      # Trend Agent (uses vision model)
            decision_llm=self.agent_llm,    # Decision Agent (uses text model)
            toolkit=self.toolkit,
            # tool_nodes=self.tool_nodes,
        )

        # --- The main LangGraph graph object ---
        self.graph = self.graph_setup.set_graph()

    def _get_api_key(self, provider: str = "openai") -> str:
        """
        Get API key with proper validation and error handling.
        
        Args:
            provider: The provider name
        
        Returns:
            str: The API key for the specified provider
            
        Raises:
            ValueError: If API key is missing or invalid
        """
        if provider == "openai":
            # First check if API key is provided in config
            api_key = self.config.get("api_key")
            
            # If not in config, check environment variable
            if not api_key:
                api_key = os.environ.get("OPENAI_API_KEY")
            
            # Validate the API key
            if not api_key:
                raise ValueError(
                    "OpenAI API key not found. Please set it using one of these methods:\n"
                    "1. Set environment variable: export OPENAI_API_KEY='your-key-here'\n"
                    "2. Update the config with: config['api_key'] = 'your-key-here'\n"
                    "3. Use the web interface to update the API key"
                )
            
            if api_key == "your-openai-api-key-here" or api_key == "":
                raise ValueError(
                    "Please replace the placeholder API key with your actual OpenAI API key. "
                    "You can get one from: https://platform.openai.com/api-keys"
                )
        elif provider == "anthropic":
            # First check if API key is provided in config
            api_key = self.config.get("anthropic_api_key")
            
            # If not in config, check environment variable
            if not api_key:
                api_key = os.environ.get("ANTHROPIC_API_KEY")
            
            # Validate the API key
            if not api_key:
                raise ValueError(
                    "Anthropic API key not found. Please set it using one of these methods:\n"
                    "1. Set environment variable: export ANTHROPIC_API_KEY='your-key-here'\n"
                    "2. Update the config with: config['anthropic_api_key'] = 'your-key-here'\n"
                )
            
            if api_key == "":
                raise ValueError(
                    "Please provide your actual Anthropic API key. "
                    "You can get one from: https://console.anthropic.com/"
                )
        elif provider == "qwen":
            # First check if API key is provided in config
            api_key = self.config.get("qwen_api_key")

            # If not in config, check environment variable
            if not api_key:
                api_key = os.environ.get("DASHSCOPE_API_KEY")

            # Validate the API key
            if not api_key:
                raise ValueError(
                    "Qwen API key not found. Please set it using one of these methods:\n"
                    "1. Set environment variable: export DASHSCOPE_API_KEY='your-key-here'\n"
                    "2. Update the config with: config['qwen_api_key'] = 'your-key-here'\n"
                )

            if api_key == "":
                raise ValueError(
                    "Please provide your actual Qwen API key. "
                    "You can get one from: https://dashscope.console.aliyun.com/"
                )
        elif provider == "minimax":
            # First check if API key is provided in config
            api_key = self.config.get("minimax_api_key")

            # If not in config, check environment variable
            if not api_key:
                api_key = os.environ.get("MINIMAX_API_KEY")

            # Validate the API key
            if not api_key:
                raise ValueError(
                    "MiniMax API key not found. Please set it using one of these methods:\n"
                    "1. Set environment variable: export MINIMAX_API_KEY='your-key-here'\n"
                    "2. Update the config with: config['minimax_api_key'] = 'your-key-here'\n"
                    "3. Use the web interface to update the API key"
                )

            if api_key == "":
                raise ValueError(
                    "Please provide your actual MiniMax API key. "
                    "You can get one from: https://platform.minimaxi.com/"
                )
        elif provider == "sohu":
            api_key = self.config.get("sohu_api_key")

            if not api_key:
                api_key = os.environ.get("SOHU_API_KEY")

            if not api_key:
                raise ValueError(
                    "Sohu API key not found. Please set it using one of these methods:\n"
                    "1. Set environment variable: export SOHU_API_KEY='your-key-here'\n"
                    "2. Update the config with: config['sohu_api_key'] = 'your-key-here'\n"
                    "3. Use the web interface to update the API key"
                )
        elif provider == "trial":
            api_key = self.config.get("trial_api_key")

            if not api_key:
                api_key = os.environ.get("QUANTAGENT_TRIAL_API_KEY")

            if not api_key:
                raise ValueError("试用模型 API Key 未配置，请联系管理员。")
        elif provider == "custom":
            api_key = self.config.get("custom_api_key")

            if not api_key:
                api_key = os.environ.get("CUSTOM_OPENAI_API_KEY")

            if not api_key:
                raise ValueError(
                    "Custom AI API key not found. Please provide an API key in settings."
                )
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. Must be 'openai', 'anthropic', "
                "'qwen', 'minimax', 'sohu', 'trial', or 'custom'"
            )
        
        return api_key

    def _create_llm(
        self, provider: str, model: str, temperature: float
    ) -> BaseChatModel:
        """
        Create an LLM instance based on the provider.

        Args:
            provider: The provider name
            model: The model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022", "qwen-vl-max-latest", "MiniMax-M2.7")
            temperature: The temperature setting for the model

        Returns:
            BaseChatModel: An instance of the appropriate LLM class
        """
        api_key = self._get_api_key(provider)

        if provider == "openai":
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=api_key,
            )
        elif provider == "anthropic":
            # ChatAnthropic handles SystemMessage extraction automatically
            # It extracts SystemMessage from the message list and passes it as 'system' parameter
            # The messages array should contain at least one non-SystemMessage
            return ChatAnthropic(
                model=model,
                temperature=temperature,
                api_key=api_key,
                base_url=self.config.get("base_url"),
            )
        elif provider == "qwen":
            return ChatQwen(
                model=model,
                temperature=temperature,
                api_key=api_key,
                max_retries=4,
            )
        elif provider == "minimax":
            # MiniMax uses an OpenAI-compatible API at https://api.minimax.io/v1
            # Temperature must be in (0.0, 1.0] for MiniMax
            clamped_temp = max(0.01, min(temperature, 1.0))
            return ChatOpenAI(
                model=model,
                temperature=clamped_temp,
                api_key=api_key,
                openai_api_base="https://api.minimax.io/v1",
            )
        elif provider == "sohu":
            return SohuChatModel(
                model=model,
                temperature=temperature,
                api_key=api_key,
            )
        elif provider == "trial":
            return ChatOpenAI(
                model=self.config.get("trial_model") or "deepseek-chat",
                temperature=temperature,
                api_key=api_key,
                openai_api_base=normalize_openai_compatible_base_url(
                    self.config.get("trial_base_url") or "https://api.deepseek.com"
                ),
                max_retries=1,
            )
        elif provider == "custom":
            custom_mode = self.config.get("custom_mode", "openai")
            custom_model = self.config.get("custom_model") or model
            if custom_mode == "http":
                return CustomHttpChatModel(
                    config=self.config,
                    api_key=api_key,
                    model=custom_model,
                    temperature=temperature,
                )
            if custom_mode == "anthropic":
                return CustomAnthropicChatModel(
                    base_url=self.config.get("custom_base_url") or "",
                    api_key=api_key,
                    model=custom_model,
                    temperature=temperature,
                )
            base_url = normalize_openai_compatible_base_url(self.config.get("custom_base_url") or "")
            if not base_url:
                raise ValueError("请先填写通用 AI 的 Base URL")
            return ChatOpenAI(
                model=custom_model,
                temperature=temperature,
                api_key=api_key,
                openai_api_base=base_url,
            )
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. Must be 'openai', 'anthropic', "
                "'qwen', 'minimax', 'sohu', 'trial', or 'custom'"
            )

    # def _set_tool_nodes(self) -> Dict[str, ToolNode]:
    #     """
    #     Define tool nodes for each agent type (indicator, pattern, trend).
    #     """
    #     return {
    #         "indicator": ToolNode(
    #             [
    #                 self.toolkit.compute_macd,
    #                 self.toolkit.compute_roc,
    #                 self.toolkit.compute_rsi,
    #                 self.toolkit.compute_stoch,
    #                 self.toolkit.compute_willr,
    #             ]
    #         ),
    #         "pattern": ToolNode(
    #             [
    #                 self.toolkit.generate_kline_image,
    #             ]
    #         ),
    #         "trend": ToolNode([self.toolkit.generate_trend_image]),
    #     }

    def refresh_llms(self):
        """
        Refresh the LLM objects with the current API key from environment.
        This is called when the API key is updated.
        """
        # Recreate LLM objects with current config values
        self.agent_llm = self._create_llm(
            provider=self.config.get("agent_llm_provider", "openai"),
            model=self.config.get("agent_llm_model", "gpt-4o-mini"),
            temperature=self.config.get("agent_llm_temperature", 0.1),
        )
        self.graph_llm = self._create_llm(
            provider=self.config.get("graph_llm_provider", "openai"),
            model=self.config.get("graph_llm_model", "gpt-4o"),
            temperature=self.config.get("graph_llm_temperature", 0.1),
        )

        # Recreate the graph setup with new LLMs
        self.graph_setup = SetGraph(
            agent_llm=self.agent_llm,       # Indicator Agent
            pattern_llm=self.graph_llm,    # Pattern Agent (uses vision model)
            trend_llm=self.graph_llm,      # Trend Agent (uses vision model)
            decision_llm=self.agent_llm,    # Decision Agent (uses text model)
            toolkit=self.toolkit,
            # tool_nodes=self.tool_nodes,
        )

        # Recreate the main graph
        self.graph = self.graph_setup.set_graph()

    def update_api_key(
        self,
        api_key: str,
        provider: str = "openai",
        base_url: str = "",
        model: str = "",
        custom_options: dict | None = None,
    ):
        """
        Update the API key in the config and refresh LLMs.
        This method is called by the web interface when API key is updated.
        
        Args:
            api_key (str): The new API key
            provider (str): The provider name ("openai" or "anthropic"), defaults to "openai"
        """
        if provider == "openai":
            # Update the config with the new API key
            self.config["api_key"] = api_key
            
            # Also update the environment variable for consistency
            os.environ["OPENAI_API_KEY"] = api_key
        elif provider == "anthropic":
            # Update the config with the new API key
            self.config["anthropic_api_key"] = api_key
            
            # Also update the environment variable for consistency
            os.environ["ANTHROPIC_API_KEY"] = api_key
        elif provider == "qwen":
            # Update the config with the new API key
            self.config["qwen_api_key"] = api_key

            # Also update the environment variable for consistency
            os.environ["DASHSCOPE_API_KEY"] = api_key
        elif provider == "minimax":
            # Update the config with the new API key
            self.config["minimax_api_key"] = api_key

            # Also update the environment variable for consistency
            os.environ["MINIMAX_API_KEY"] = api_key
        elif provider == "sohu":
            self.config["sohu_api_key"] = api_key
            os.environ["SOHU_API_KEY"] = api_key
        elif provider == "trial":
            self.config["trial_api_key"] = api_key
            os.environ["QUANTAGENT_TRIAL_API_KEY"] = api_key
        elif provider == "custom":
            custom_options = custom_options or {}
            self.config["custom_api_key"] = api_key
            self.config["custom_mode"] = custom_options.get("mode") or self.config.get("custom_mode", "openai")
            if self.config["custom_mode"] == "openai":
                self.config["custom_base_url"] = normalize_openai_compatible_base_url(base_url)
            else:
                self.config["custom_base_url"] = (base_url or "").strip().rstrip("/")
            self.config["custom_endpoint_url"] = (custom_options.get("endpoint_url") or "").strip()
            self.config["custom_model"] = model or "gpt-4o"
            if custom_options.get("headers_template"):
                self.config["custom_headers_template"] = custom_options["headers_template"]
            if custom_options.get("body_template"):
                self.config["custom_body_template"] = custom_options["body_template"]
            if custom_options.get("response_path"):
                self.config["custom_response_path"] = custom_options["response_path"]
            os.environ["CUSTOM_OPENAI_API_KEY"] = api_key
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. Must be 'openai', 'anthropic', "
                "'qwen', 'minimax', 'sohu', 'trial', or 'custom'"
            )
        
        # Refresh the LLMs with the new API key
        self.refresh_llms()
