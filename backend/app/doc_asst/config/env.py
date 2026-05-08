from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


def _load_project_env() -> None:
    # 从项目根目录加载 .env，避免受启动目录影响。
    env_path = Path(__file__).resolve().parents[4] / ".env"
    load_dotenv(env_path)


_load_project_env()


@dataclass(frozen=True)
class EnvConfig:
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "")
    llm_model_id: str = os.getenv("LLM_MODEL_ID", "")
    llm_timeout: int = int(os.getenv("LLM_TIMEOUT", "60"))
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    serpapi_api_key: str = os.getenv("SERPAPI_API_KEY", "")
    langgraph_postgres_dsn: str = os.getenv("LANGGRAPH_POSTGRES_DSN", "")


ENV = EnvConfig()
