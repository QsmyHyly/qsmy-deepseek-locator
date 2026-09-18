"""工具执行框架：把普通 Python 函数注册成「模型可调用的工具」。

- registry.ToolRegistry / Tool / ToolResult：注册、schema 生成、执行与异常回填
- builtin.build_default_registry：内置工具集合（默认那 6 个）

只有在用 Agent 循环（qsmy_deepseek_locator.agent）时才需要本子包；
单轮定位（locate / locate_to_file）完全不碰它，也不额外 import 任何东西。
"""

from .builtin import build_default_registry
from .registry import Tool, ToolRegistry, ToolResult

__all__ = ["ToolRegistry", "Tool", "ToolResult", "build_default_registry"]
