from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol

class Tool(Protocol):
    """
    Tool 的最小协议。
    任何工具只要实现 name/description/run，就可以被 Agent 注册和调用。
    """

    name: str
    description: str

    def run(self, **kwargs: Any) -> Any:
        ...


@dataclass
class AgentContext:
    """
    跨节点传递的运行上下文。
    metadata 可放追踪ID、租户信息、调试标记等扩展字段。
    """

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)