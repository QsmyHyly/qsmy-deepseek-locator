"""把本库里的普通函数转换为可加入模型请求的工具列表对象。

（历史上本模块服务于已删除的 point_parser.py，现在统一由 tools/registry.py
在注册工具时调用，构建默认工具集见 tools/builtin.py。）


工具对象的 name / description 取自函数的 __name__ 与 docstring，入参结构取自
函数签名，每个参数的 JSON schema 取自参数的 Annotated 注解，因此名称、描述、
参数结构都只在函数定义处维护一份。

用法：
    from qsmy_deepseek_locator.tool_schema import TOOLS

    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=messages,
        tools=TOOLS,
    )
"""

import inspect
import typing

from .parsing import decode_json_points, extract_coordinates, parse_coordinates

# Python 基础类型 -> JSON Schema 类型
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _arg_schema(hint):
    """由参数注解推导单个入参的 JSON schema。

    - 注解形如 Annotated[基础类型, 元数据]：
      - 元数据是 str，则作为该参数的 description；
      - 元数据是 dict，则直接作为该参数的完整 schema；
    - 注解只是普通类型时，按其类型映射。
    """
    if typing.get_origin(hint) is typing.Annotated:
        base, *meta = typing.get_args(hint)
        schema = {"type": _TYPE_MAP.get(base, "string")}
        for item in meta:
            if isinstance(item, dict):
                schema = dict(item)
            elif isinstance(item, str):
                schema["description"] = item
        return schema
    return {"type": _TYPE_MAP.get(hint, "string")}


def build_tool(func):
    """把函数转换为模型可用的工具对象（名称/描述/入参全部从函数自身读取）。

    - name        来自 func.__name__
    - description 来自函数 docstring 的首行（inspect.getdoc 即清理后的 __doc__）
    - parameters  参数名/顺序/是否必填来自 inspect.signature(func)，
                  每个参数的 schema 来自其 Annotated 注解
    """
    hints = typing.get_type_hints(func, include_extras=True)
    params = inspect.signature(func).parameters
    doc = inspect.getdoc(func) or ""

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc.splitlines()[0] if doc else "",
            "parameters": {
                "type": "object",
                "properties": {name: _arg_schema(hints.get(name)) for name in params},
                "required": [
                    name for name, param in params.items()
                    if param.default is inspect.Parameter.empty
                ],
            },
        },
    }


# 三个工具对象（由 point_parser.py 的函数自动生成）
DECODE_JSON_POINTS_TOOL = build_tool(decode_json_points)
EXTRACT_COORDINATES_TOOL = build_tool(extract_coordinates)
PARSE_COORDINATES_TOOL = build_tool(parse_coordinates)

# 可直接作为 tools 参数传入模型请求的工具列表
TOOLS = [
    DECODE_JSON_POINTS_TOOL,
    EXTRACT_COORDINATES_TOOL,
    PARSE_COORDINATES_TOOL,
]
