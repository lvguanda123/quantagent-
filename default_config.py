DEFAULT_CONFIG = {
    "agent_llm_model": "MiniMax-M3",
    "graph_llm_model": "MiniMax-M3",
    "agent_llm_provider": "anthropic",
    "graph_llm_provider": "anthropic",
    "agent_llm_temperature": 0.1,
    "graph_llm_temperature": 0.1,
    "base_url": "https://ark.cn-beijing.volces.com/api/coding",
    "api_key": "",
    # API keys are NOT committed to source. They are injected at build time from
    # GitHub Secrets into a bundled .env file (desktop), or supplied via a local,
    # git-ignored .env (development). See .github/workflows/build-windows-tauri.yml.
    "anthropic_api_key": "",
    "qwen_api_key": "",
    "minimax_api_key": "",
    "sohu_api_key": "",
    "trial_api_key": "",
    "trial_base_url": "https://api.deepseek.com",
    "trial_model": "deepseek-v4-flash-vision-exp",
    "custom_api_key": "",
    "custom_mode": "openai",
    "custom_base_url": "",
    "custom_endpoint_url": "",
    "custom_model": "gpt-4o",
    "custom_headers_template": '{"Authorization":"Bearer {{api_key}}","Content-Type":"application/json"}',
    "custom_body_template": '{"model":"{{model}}","messages":{{messages_json}},"temperature":{{temperature}}}',
    "custom_response_path": "choices.0.message.content",
    # Indicator Agent: when False, skip the LLM tool-calling loop and let the
    # agent compute all 5 indicators locally (TA-Lib) then call the LLM once
    # to summarise. Saves ~1-2 LLM round-trips. Set to True only when the model
    # in use is unreliable at direct tool-calling or you want a verbose
    # multi-step reasoning trace.
    "indicator_use_tool_calling": False,
}
