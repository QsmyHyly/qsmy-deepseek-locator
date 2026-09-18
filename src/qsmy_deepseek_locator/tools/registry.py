"""工具注册表与执行框架。

职责：
1. 把普通 Python 函数注册成「模型可调用的工具」（复用 tool_schema.build_tool 生成 schema）。
2. 提供给模型的 tools 规格列表（registry.spec()）。
3. 接收模型返回的工具名 + 参数 JSON，找到对应函数并执行（registry.execute()）。
4. 统一处理异常与结果序列化，保证工具出错时也能把信息回填给模型。

典型用法：
    registry = ToolRegistry()
    registry.register(my_func)

    # 模型返回 tool_call 后：
    result = registry.execute(tool_call.name, tool_call.arguments)
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from ..tool_schema import build_tool


@dataclass
class Tool:
    """一个已注册的工具。

    context_params 中的参数不暴露给模型，而是在执行时由运行上下文注入
    （例如当前处理的图片地址 source），避免模型凭空编造这类信息。
    """

    name: str
    func: Callable[..., Any]
    spec: dict
    description: str = ""
    context_params: set[str] = field(default_factory=set)


@dataclass
class ToolResult:
    """一次工具执行的结果。"""

    name: str
    arguments: dict
    ok: bool
    content: str
    raw: Any = None
    error: str | None = None
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "ok": self.ok,
            "content": self.content,
            "error": self.error,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


class ToolRegistry:
    """工具注册表。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # ------------------------------------------------------------------ #
    # 注册
    # ------------------------------------------------------------------ #
    def register(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        context_params: set[str] | None = None,
    ):
        """注册工具，可作为普通调用或装饰器使用。

        - registry.register(func)
        - @registry.register
        - registry.register(func, context_params={"source"})
          source 将不出现在给模型的 schema 中，执行时由上下文注入。
        """

        def _do(f: Callable[..., Any]) -> Callable[..., Any]:
            spec = build_tool(f)
            tool_name = name or spec["function"]["name"]
            if description:
                spec["function"]["description"] = description

            hidden = set(context_params or ())
            params = spec["function"].get("parameters", {})
            for key in hidden:
                params.get("properties", {}).pop(key, None)
                if key in params.get("required", []):
                    params["required"].remove(key)

            self._tools[tool_name] = Tool(
                name=tool_name,
                func=f,
                spec=spec,
                description=spec["function"].get("description", ""),
                context_params=hidden,
            )
            return f

        if func is not None:
            return _do(func)
        return _do

    def register_many(self, *funcs: Callable[..., Any]) -> None:
        for f in funcs:
            self.register(f)

    def register_with_context(
        self,
        func: Callable[..., Any],
        context_params: set[str],
        *,
        name: str | None = None,
        description: str | None = None,
    ):
        """注册一个需要运行上下文注入参数的工具。"""
        return self.register(
            func, name=name, description=description, context_params=context_params
        )

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #
    def spec(self) -> list[dict]:
        """返回可直接传给 chat.completions.create(tools=...) 的列表。"""
        return [t.spec for t in self._tools.values()]

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return list(self._tools)

    def describe(self) -> list[dict]:
        return [
            {"name": t.name, "description": t.description}
            for t in self._tools.values()
        ]

    # ------------------------------------------------------------------ #
    # 执行
    # ------------------------------------------------------------------ #
    def execute(
        self,
        name: str,
        arguments: str | dict | None,
        *,
        context: dict | None = None,
    ) -> ToolResult:
        """执行一次工具调用。

        Args:
            name: 工具名。
            arguments: 模型给出的参数，JSON 字符串或已解析的 dict。
            context: 运行上下文；其中属于该工具 context_params 的键会覆盖模型参数。
        """
        import time

        start = time.perf_counter()

        if isinstance(arguments, str):
            try:
                parsed_args = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                return ToolResult(
                    name=name,
                    arguments={},
                    ok=False,
                    content=f"参数不是合法 JSON：{exc}",
                    error="invalid_arguments_json",
                )
        elif isinstance(arguments, dict):
            parsed_args = arguments
        elif arguments is None:
            parsed_args = {}
        else:
            return ToolResult(
                name=name, arguments={}, ok=False,
                content=f"不支持的工具参数类型：{type(arguments).__name__}",
                error="invalid_arguments_type",
            )

        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                name=name, arguments=parsed_args, ok=False,
                content=f"未注册的工具：{name}。可用工具：{', '.join(self.names()) or '无'}",
                error="unknown_tool",
            )

        # 注入运行上下文（覆盖模型给出的同名参数，避免其编造）
        call_args = dict(parsed_args)
        if context and tool.context_params:
            for key in tool.context_params:
                if key in context and context[key] is not None:
                    call_args[key] = context[key]

        try:
            raw = tool.func(**call_args)
            content = _serialize(raw)
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                name=name, arguments=call_args, ok=True,
                content=content, raw=raw, elapsed_ms=elapsed,
            )
        except TypeError as exc:
            # 通常是模型给出的参数名/数量不匹配
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                name=name, arguments=call_args, ok=False,
                content=f"参数不匹配：{exc}",
                error="bad_arguments", elapsed_ms=elapsed,
            )
        except Exception as exc:  # noqa: BLE001 - 工具异常需回填给模型
            elapsed = (time.perf_counter() - start) * 1000
            detail = traceback.format_exc(limit=2)
            return ToolResult(
                name=name, arguments=call_args, ok=False,
                content=f"工具执行失败：{exc}\n{detail}",
                error="execution_error", elapsed_ms=elapsed,
            )


def _serialize(value: Any) -> str:
    """把工具返回值转成字符串，供 role=tool 消息回填。"""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)
