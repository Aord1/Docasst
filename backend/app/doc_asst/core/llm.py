from __future__ import annotations

from langchain_openai import ChatOpenAI

from ..config import ENV


def create_chat_model(temperature: float = 0.2) -> ChatOpenAI:
    """
    创建 LangChain ChatOpenAI 实例。
    统一 LLM 调用层：供 create_react_agent 和直接 invoke 使用。
    """
    if not all([ENV.llm_model_id, ENV.llm_api_key, ENV.llm_base_url]):
        raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")
    return ChatOpenAI(
        model=ENV.llm_model_id,
        api_key=ENV.llm_api_key,
        base_url=ENV.llm_base_url,
        temperature=temperature,
        streaming=True,
        timeout=ENV.llm_timeout,
        max_retries=3,
    )


def create_vision_openai_client() -> tuple:
    """
    创建视觉模型用的 OpenAI 原生客户端（供 FileContentTool OCR 使用）。
    返回 (client, model_name)。
    """
    from openai import OpenAI

    api_key = ENV.vision_api_key or ENV.llm_api_key
    base_url = ENV.vision_base_url or ENV.llm_base_url
    model = ENV.vision_model_id or ENV.llm_model_id
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=ENV.llm_timeout)
    return client, model
