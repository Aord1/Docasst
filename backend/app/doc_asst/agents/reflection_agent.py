from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from ..core.agent import BaseAgent


class ReflectionAgent(BaseAgent):
    """
    反思智能体：
    评估当前答案质量，决定是否进入下一轮 ReAct。
    无需工具，通过 ChatOpenAI.invoke() 直接调用 LLM。
    """

    DEFAULT_SYSTEM_PROMPT = (
        "你是质量反思助手。判断当前答案是否达标。"
        "输出仅一个 JSON，字段：\n"
        "- verdict: pass 或 revise\n"
        "- confidence: 0~1 的浮点数\n"
        "- reason: 一句话说明\n"
        "- missing_points: 数组，列出答案中缺失的关键点\n"
        "- next_actions: 数组，列出改进的具体建议\n"
        "\nconfidence 标准："
        "0.9~1.0=完整无缺陷；0.7~0.89=基本合理有局部不足；"
        "0.5~0.69=明显缺陷需修订；0~0.49=严重不合格。"
        "\n规则：confidence<0.7 时 verdict 必须为 revise，且 missing_points 和 next_actions 不能为空。"
    )

    def __init__(
        self,
        llm: ChatOpenAI,
        tools: Optional[List[Any]] = None,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
    ) -> None:
        super().__init__(
            name="reflection_agent",
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
            llm=llm,
            tools=tools,
            temperature=temperature,
        )

    def run(
        self,
        user_input: str,
        memory: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Reflection 无工具，通过 _run_with_tools 内部自动走 ChatOpenAI.invoke() 路径
        reflection_text, _, _, usage = self._run_with_tools(
            user_input=user_input,
            memory=memory,
        )
        parsed = self._parse_reflection_json(reflection_text)
        return {
            "agent": self.name,
            "stage": "reflection",
            "reflection_text": reflection_text,
            "reflection_output": parsed,
            "used_tools": [],
            "usage": usage,
        }

    def _parse_reflection_json(self, reflection_text: str) -> Dict[str, Any]:
        raw = reflection_text.strip()
        if "```" in raw:
            raw = raw.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {
                "verdict": "revise",
                "reason": "反思输出无法解析，默认修订。",
                "confidence": 0.0,
                "missing_points": [],
                "next_actions": [],
            }

        verdict = str(data.get("verdict", "revise")).strip().lower()
        if verdict not in {"pass", "revise"}:
            verdict = "revise"

        missing_points = data.get("missing_points") or data.get("missing", [])
        if not isinstance(missing_points, list):
            missing_points = [str(missing_points)]

        next_actions = data.get("next_actions") or data.get("suggestions") or data.get("actions", [])
        if not isinstance(next_actions, list):
            next_actions = [str(next_actions)]

        return {
            "verdict": verdict,
            "reason": str(data.get("reason", "")),
            "confidence": float(data.get("confidence", 0.0) or 0.0),
            "missing_points": missing_points,
            "next_actions": next_actions,
        }
