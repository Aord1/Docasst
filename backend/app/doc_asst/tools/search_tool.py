from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .base_tool import BaseTool

from dotenv import load_dotenv

load_dotenv()


class SearchTool(BaseTool):
    """
    联网搜索工具（优先级固定）：
    1) 先调用 Tavily
    2) Tavily 失败或无结果时回退 SerpApi
    """

    name = "web_search"
    description = "联网搜索工具：优先 Tavily，失败后回退 SerpApi。"

    def __init__(
        self,
        tavily_api_key: Optional[str] = None,
        serpapi_api_key: Optional[str] = None,
    ) -> None:
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        self.serpapi_api_key = serpapi_api_key or os.getenv("SERPAPI_API_KEY")

    def _run(self, **kwargs: Any) -> Dict[str, Any]:
        query = kwargs.get("query")
        max_results = int(kwargs.get("max_results", 5))
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
            return {
                "provider": "tavily",
                "query": query,
                "results": tavily_results,
                "attempts": attempts,
            }

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
            return {
                "provider": "serpapi",
                "query": query,
                "results": serp_results,
                "attempts": attempts,
            }

        raise RuntimeError(
            "所有搜索源都失败或未返回结果。"
            f" tavily_error={tavily_error}, serpapi_error={serp_error}"
        )

    def _search_tavily(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.tavily_api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "advanced",
        }
        body = json.dumps(payload).encode("utf-8")
        request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        response = self._request_json(request)
        items = response.get("results", []) or []
        return [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
                "source": "tavily",
            }
            for item in items
        ]

    def _search_serpapi(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.serpapi_api_key,
            "num": max_results,
        }
        url = f"https://serpapi.com/search.json?{urlencode(params)}"
        request = Request(url, method="GET")
        response = self._request_json(request)
        items = response.get("organic_results", []) or []
        return [
            {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "source": "serpapi",
            }
            for item in items[:max_results]
        ]

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
