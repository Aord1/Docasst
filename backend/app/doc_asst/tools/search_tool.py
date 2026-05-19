from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Type
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field

from .base_tool import DocasstBaseTool

from dotenv import load_dotenv

load_dotenv()


class SearchArgs(BaseModel):
    query: str = Field(description="搜索查询关键词")
    max_results: int = Field(default=3, description="返回结果数量上限")


class SearchTool(DocasstBaseTool):
    """
    联网搜索工具（优先级固定）：
    1) 先调用 Tavily
    2) Tavily 失败或无结果时回退 SerpApi
    """

    name: str = "web_search"
    description: str = "联网搜索工具：优先 Tavily，失败后回退 SerpApi。用于获取最新事实、外部信息、时效内容。"
    args_schema: Type[BaseModel] = SearchArgs

    tavily_api_key: str = Field(default="", exclude=True)
    serpapi_api_key: str = Field(default="", exclude=True)

    def __init__(self, tavily_api_key: Optional[str] = None, serpapi_api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if tavily_api_key:
            self.tavily_api_key = tavily_api_key
        elif os.getenv("TAVILY_API_KEY"):
            self.tavily_api_key = os.getenv("TAVILY_API_KEY", "")
        if serpapi_api_key:
            self.serpapi_api_key = serpapi_api_key
        elif os.getenv("SERPAPI_API_KEY"):
            self.serpapi_api_key = os.getenv("SERPAPI_API_KEY", "")

    def _execute(self, query: str = "", max_results: int = 3, **kwargs) -> Dict[str, Any]:
        if not query:
            raise ValueError("web_search 工具需要提供 query 参数")

        attempts: List[Dict[str, Any]] = []

        tavily_error = None
        tavily_results: List[Dict[str, Any]] = []
        if self.tavily_api_key:
            try:
                tavily_results = self._search_tavily(query=query, max_results=max_results)
                attempts.append({"provider": "tavily", "ok": True, "message": "调用成功", "count": len(tavily_results)})
            except Exception as exc:  # noqa: BLE001
                tavily_error = str(exc)
                attempts.append({"provider": "tavily", "ok": False, "message": "调用失败", "error": tavily_error})
        else:
            attempts.append({"provider": "tavily", "ok": False, "message": "未配置 TAVILY_API_KEY"})

        if tavily_results:
            return {"provider": "tavily", "query": query, "results": tavily_results, "attempts": attempts}

        serp_error = None
        serp_results: List[Dict[str, Any]] = []
        if self.serpapi_api_key:
            try:
                serp_results = self._search_serpapi(query=query, max_results=max_results)
                attempts.append({"provider": "serpapi", "ok": True, "message": "调用成功", "count": len(serp_results)})
            except Exception as exc:  # noqa: BLE001
                serp_error = str(exc)
                attempts.append({"provider": "serpapi", "ok": False, "message": "调用失败", "error": serp_error})
        else:
            attempts.append({"provider": "serpapi", "ok": False, "message": "未配置 SERPAPI_API_KEY"})

        if serp_results:
            return {"provider": "serpapi", "query": query, "results": serp_results, "attempts": attempts}

        raise RuntimeError(f"所有搜索源都失败。tavily_error={tavily_error}, serpapi_error={serp_error}")

    def compact(self, result: Any) -> str:
        """精简：最多3条结果，每条保留 title + snippet(300字)"""
        if isinstance(result, dict) and "results" in result:
            compact_results = []
            for item in result["results"][:3]:
                compact_results.append({
                    "title": item.get("title", ""),
                    "snippet": str(item.get("snippet", ""))[:300],
                })
            return json.dumps({"ok": True, "results": compact_results}, ensure_ascii=False)
        return json.dumps({"ok": True, "data": result}, ensure_ascii=False, default=str)

    def _search_tavily(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        url = "https://api.tavily.com/search"
        payload = {"api_key": self.tavily_api_key, "query": query, "max_results": max_results, "search_depth": "advanced"}
        body = json.dumps(payload).encode("utf-8")
        request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        response = self._request_json(request)
        items = response.get("results", []) or []
        return [{"title": item.get("title", ""), "url": item.get("url", ""), "snippet": item.get("content", ""), "source": "tavily"} for item in items]

    def _search_serpapi(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        params = {"engine": "google", "q": query, "api_key": self.serpapi_api_key, "num": max_results}
        url = f"https://serpapi.com/search.json?{urlencode(params)}"
        request = Request(url, method="GET")
        response = self._request_json(request)
        items = response.get("organic_results", []) or []
        return [{"title": item.get("title", ""), "url": item.get("link", ""), "snippet": item.get("snippet", ""), "source": "serpapi"} for item in items[:max_results]]

    def _request_json(self, request: Request) -> Dict[str, Any]:
        try:
            with urlopen(request, timeout=30) as response:  # nosec B310
                raw = response.read().decode("utf-8")
                return json.loads(raw)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP 请求失败，状态码={exc.code}，响应体={body}") from exc
        except URLError as exc:
            raise RuntimeError(f"网络请求失败，原因={exc.reason}") from exc
