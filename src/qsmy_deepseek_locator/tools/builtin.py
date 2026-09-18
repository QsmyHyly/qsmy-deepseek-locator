"""内置工具集合（Agent 循环用；单轮定位不碰这里）。

这些函数会被工具框架生成 JSON Schema 交给模型，模型调用后由 registry 执行。
参数用 Annotated 注解承载描述，名称 / 描述 / 参数结构都只在函数定义处维护一份。

⚠️ **schema 只取 docstring 的首行**（见 tool_schema.build_tool），所以面向模型的约束
必须写在首行里；下面的正文是给人看的。

⚠️ **坐标归一化在工具层做**（下面的 _unitize），**不在 parsing 里做**：
parsing 的那几个函数同时被需要读「模型原始输出」的调用方使用（量「模型到底给的哪个
坐标空间」），一旦在解析核心里归一化，就再也测不出来。
@doc docs/API-NOTES.md#8-坐标口径为什么必须是-0010
（该文档解决"为什么 0.0~1.0 是唯一不变量、提示词写错会怎样"的问题。）
"""

from __future__ import annotations

from typing import Annotated

from ..drawing import COLORS, save_annotated, summarize
from ..images import load_image
from ..parsing import (
    _COORDINATE_LIST_SCHEMA,
    decode_json_points as _decode_json_points,
    extract_coordinates as _extract_coordinates,
    normalize_to_unit,
)
from .registry import ToolRegistry


# --------------------------------------------------------------------------- #
# 坐标归一化的工具层包装
# --------------------------------------------------------------------------- #
def _unitize(data):
    """把解析出来的坐标整体换算到 0.0~1.0，返回 (data, 是否换算过)。

    只改坐标数值，不动容器形状与 label：
    - 最大值 <= 1.0        -> 已是新约定，原样返回；
    - 最大值在 (1.0, 1000] -> 判定为 0~1000 旧刻度，整体除以 1000（无损）；
    - 存在 > 1000 的值     -> 原样返回（很可能是像素坐标），由 check_coordinate_range 报警。

    ⚠️ 为什么放在**工具层**而不是解析核心：见模块头注释。
    """
    if isinstance(data, dict):
        converted_items, converted = normalize_to_unit([data])
        return (converted_items[0] if converted_items else data), converted
    if isinstance(data, list):
        return normalize_to_unit(data)
    return data, False


# --------------------------------------------------------------------------- #
# 包装成「模型友好」的工具函数
# --------------------------------------------------------------------------- #
def decode_json_points(
    text: Annotated[str, "包含坐标 JSON 的文本，可带代码块标记。坐标必须是 0.0~1.0 的相对比例。"],
):
    """解析模型输出文本为坐标对象列表；坐标统一为 0.0~1.0 相对比例（0~1000 旧刻度会自动换算）。"""
    data, _converted = _unitize(_decode_json_points(text))
    return data


def extract_coordinates(
    data: Annotated[list, _COORDINATE_LIST_SCHEMA],
):
    """把坐标列表拆成 bbox / point 两组；坐标统一为 0.0~1.0 相对比例（0~1000 旧刻度会自动换算）。"""
    data, _converted = _unitize(data)
    return _extract_coordinates(data)


def parse_coordinates(
    text: Annotated[str, "包含坐标 JSON 的文本，可带代码块标记。坐标必须是 0.0~1.0 的相对比例。"],
):
    """解析模型输出文本并拆成 bbox / point 两组；坐标统一为 0.0~1.0 相对比例（0~1000 旧刻度会自动换算）。"""
    data, _converted = _unitize(_decode_json_points(text))
    return _extract_coordinates(data)


def get_image_info(
    source: Annotated[str, "图片地址：http(s) URL、本地路径或 data URL。"],
) -> dict:
    """获取图片的宽高与格式信息（这是原图的像素尺寸；不是你实际看到的栅格，不要用它做归一化↔像素换算）。

    ⚠️ 返回的是**原图**的宽高（本地/远端文件本身的像素尺寸），**不是模型这次实际看到的栅格**：
    图片在进模型前会被服务端缩放且不回传缩放后尺寸。
    @doc docs/API-NOTES.md#3-图片-token-与尺寸
    所以这个结果只能用来做「文件元信息」类的判断，**不能拿它把 0.0~1.0 归一化坐标乘回像素** ——
    乘出来的像素值对应不到模型眼里的那张图。
    """
    img = load_image(source)
    return {
        "width": img.size[0],
        "height": img.size[1],
        "mode": img.mode,
        "format": img.format or "unknown",
    }


def annotate_image(
    source: Annotated[str, "图片地址：http(s) URL、本地路径或 data URL。"],
    items: Annotated[
        str,
        "坐标 JSON 文本，元素可含 bbox_2d / point_2d 与 label。"
        "坐标必须是 0.0~1.0 的相对比例（x = 像素x / 图宽，y = 像素y / 图高），"
        "不要给像素值；0~1000 旧刻度会被自动换算。",
    ],
    output_dir: Annotated[
        str | None, "标注图保存目录；由运行上下文注入，不暴露给模型。留空则落在当前工作目录。"
    ] = None,
) -> dict:
    """在图片上绘制坐标标注并保存，返回标注图路径与统计信息（坐标统一按 0.0~1.0 相对比例解读）。"""
    path = save_annotated(source, items, output_dir=output_dir)
    return {
        "annotated_path": str(path),
        "summary": summarize(items),
    }


def list_palette_colors() -> list[str]:
    """列出可用于标注的调色板颜色名称。"""
    return COLORS[:32]


# --------------------------------------------------------------------------- #
# 默认注册表
# --------------------------------------------------------------------------- #
def build_default_registry() -> ToolRegistry:
    """构建包含全部内置工具的注册表（4 个纯函数 + 2 个需要运行上下文的）。"""
    registry = ToolRegistry()
    registry.register_many(
        parse_coordinates,
        decode_json_points,
        extract_coordinates,
        list_palette_colors,
    )
    # 这两个工具需要「当前图片」这类运行上下文：参数由框架注入、不出现在给模型的 schema 里，
    # 避免模型凭空编造图片地址（真实模型确实会这么做）。annotate_image 的输出目录同理 ——
    # 落哪个目录是调用方的决定，不该让模型来挑。
    registry.register_with_context(get_image_info, {"source"})
    registry.register_with_context(annotate_image, {"source", "output_dir"})
    return registry


__all__ = [
    "annotate_image",
    "build_default_registry",
    "decode_json_points",
    "extract_coordinates",
    "get_image_info",
    "list_palette_colors",
    "parse_coordinates",
]

