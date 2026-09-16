"""打标绘制：把 Detection 画回图片上（框 / 点 + 中文标签）。

坐标口径：传进来的 Detection 是 0.0~1.0 归一化比例，这里才换算成像素。
越界值在绘制前会被**夹紧到边界**（而不是抛异常）—— 画在边上总好过整张图打不出来；
但真正的判断（这是不是像素坐标）归 parsing.check_coordinate_range，本模块只负责画。

中文字体是这里最容易被忽略的坑：硬编码 NotoSansCJK 在 Windows 上必然报错，
所以按「Windows -> macOS -> Linux」顺序探测常见中文字体，全都找不到时退回 PIL 内置位图字体
（英文标签仍可读，中文会变成方块，属于「能用但要提醒」的状态）。
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageColor, ImageDraw, ImageFont

from .images import load_image
from .parsing import Detection, parse_detections

# 手写色优先（语义清晰、区分度高），其后追加 PIL 全部命名颜色并去重。
_EXTRA_COLORS = [name for (name, _code) in ImageColor.colormap.items()]
COLORS: list[str] = list(dict.fromkeys([
    "red", "green", "blue", "yellow", "orange", "pink", "purple", "brown", "gray",
    "beige", "turquoise", "cyan", "magenta", "lime", "navy", "maroon", "teal",
    "olive", "coral", "lavender", "violet", "gold", "silver",
] + _EXTRA_COLORS))

# 跨平台中文字体候选。顺序 = 命中概率，别把 Arial 之类不含中文的排前面。
_FONT_CANDIDATES = [
    "msyh.ttc", "msyhbd.ttc",            # Windows 微软雅黑
    "simhei.ttf", "simsun.ttc",          # Windows 黑体 / 宋体
    "NotoSansCJK-Regular.ttc",
    "NotoSansCJKsc-Regular.otf",
    "PingFang.ttc", "Hiragino Sans GB.ttc",   # macOS
    "wqy-microhei.ttc", "wqy-zenhei.ttc",     # Linux
    "DejaVuSans.ttf", "Arial.ttf",
]

_FONT_DIRS = [
    Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path.home() / ".fonts",
    Path("/System/Library/Fonts"),
    Path("/Library/Fonts"),
]

_font_cache: dict[int, Any] = {}


def _find_font_file() -> str | None:
    for name in _FONT_CANDIDATES:
        for directory in _FONT_DIRS:
            candidate = directory / name
            if candidate.exists():
                return str(candidate)
        # 有些发行版把字体放在子目录里，做一次浅层递归
        for directory in _FONT_DIRS:
            if not directory.exists():
                continue
            for found in directory.rglob(name):
                return str(found)
    return None


def resolve_font(size: int = 20) -> Any:
    """取一个尽量支持中文的字体对象（按 size 缓存，避免每条标注都重开文件）。"""
    if size in _font_cache:
        return _font_cache[size]
    path = _find_font_file()
    font: Any
    if path:
        try:
            font = ImageFont.truetype(path, size=size)
        except Exception:  # noqa: BLE001 - 字体文件损坏时退回内置字体，不让打标失败
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()
    _font_cache[size] = font
    return font


def coerce_detections(items: Any) -> list[Detection]:
    """把任意输入统一成 Detection 列表。

    认这些形态：Detection、模型字段字典、坐标 JSON 文本、以及它们的列表。
    坐标文本走 parse_detections，因此旧刻度 0~1000 的兜底在这里同样有效。
    """
    if items is None:
        return []
    if isinstance(items, Detection):
        return [items]
    if isinstance(items, str):
        return parse_detections(items)[0]
    if isinstance(items, dict):
        if "detections" in items and isinstance(items["detections"], list):
            return coerce_detections(items["detections"])
        det = Detection.from_dict(items)
        return [det] if det else []
    out: list[Detection] = []
    for item in items:
        out.extend(coerce_detections(item))
    return out


def _ratio_to_abs(value: float, total: int) -> int:
    """归一化比例 -> 像素，越界夹紧（绘制层的兜底，不改变原始数据）。"""
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        ratio = 0.0
    ratio = min(1.0, max(0.0, ratio))
    return int(round(ratio * total))


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, color: str, font: Any) -> None:
    """画带底色的小标签，保证在任意背景上都读得清（底色亮度决定黑字还是白字）。"""
    try:
        left, top, right, bottom = draw.textbbox(xy, text, font=font)
    except Exception:  # noqa: BLE001 - 某些内置字体没有 textbbox
        draw.text(xy, text, fill=color, font=font)
        return
    pad = 3
    draw.rectangle((left - pad, top - pad, right + pad, bottom + pad), fill=color)
    try:
        rgb = ImageColor.getrgb(color)
        luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        text_color = (0, 0, 0) if luminance > 140 else (255, 255, 255)
    except Exception:  # noqa: BLE001
        text_color = (255, 255, 255)
    draw.text(xy, text, fill=text_color, font=font)


def draw(
    image: Any,
    items: Any,
    *,
    box_width: int = 3,
    point_radius: int = 5,
    font_size: int = 22,
    draw_label: bool = True,
    colors: Sequence[str] | None = None,
) -> Image.Image:
    """在图片上画出 Detection（框 / 点 + 标签），返回新图（不改动入参图）。

    Args:
        image: 路径 / URL / bytes / PIL.Image / data URL。
        items: Detection 列表、模型字段字典列表，或模型输出的坐标 JSON 文本。
        colors: 自定义调色板；留空用内置 COLORS，按目标顺序取色。
    """
    img = load_image(image).convert("RGB")
    detections = coerce_detections(items)
    palette = list(colors) if colors else COLORS
    width, height = img.size

    painter = ImageDraw.Draw(img)
    font = resolve_font(font_size)

    for index, det in enumerate(detections):
        color = palette[index % len(palette)] if palette else "red"
        label = det.label

        if det.bbox is not None:
            x1, y1, x2, y2 = det.bbox
            ax1, ay1 = _ratio_to_abs(x1, width), _ratio_to_abs(y1, height)
            ax2, ay2 = _ratio_to_abs(x2, width), _ratio_to_abs(y2, height)
            if ax1 > ax2:
                ax1, ax2 = ax2, ax1
            if ay1 > ay2:
                ay1, ay2 = ay2, ay1
            painter.rectangle(((ax1, ay1), (ax2, ay2)), outline=color, width=box_width)
            if draw_label and label:
                _draw_label(painter, (ax1, max(0, ay1 - font_size - 6)), label, color, font)

        if det.point is not None:
            cx, cy = _ratio_to_abs(det.point[0], width), _ratio_to_abs(det.point[1], height)
            painter.ellipse(
                [(cx - point_radius, cy - point_radius), (cx + point_radius, cy + point_radius)],
                outline=color, width=box_width,
            )
            painter.ellipse([(cx - 1, cy - 1), (cx + 1, cy + 1)], fill=color)
            if draw_label and label:
                _draw_label(painter, (cx + point_radius + 4, cy + 2), label, color, font)

    return img


def save_annotated(
    image: Any,
    items: Any,
    path: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
    stem: str | None = None,
    **draw_kwargs: Any,
) -> Path:
    """画出标注并保存为 PNG，返回保存路径。

    Args:
        path: 直接给完整文件路径（优先）。
        output_dir / stem: 不给 path 时用它们拼；两者都缺省则落在当前工作目录，
            文件名 annotated_<12位hex>.png。
    """
    annotated = draw(image, items, **draw_kwargs)
    if path is None:
        directory = Path(output_dir) if output_dir else Path.cwd()
        directory.mkdir(parents=True, exist_ok=True)
        name = stem or f"annotated_{uuid.uuid4().hex[:12]}"
        path = directory / f"{name}.png"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(target, format="PNG")
    return target


# 标注图输出支持的扩展名 -> PIL 保存格式。
# 认不出的扩展名一律报错而**不猜**：文件名写着 .jpg、内容却是 PNG 字节，
# 是最难排查的一类问题（下游按后缀读图会直接报「不是 JPEG」）。
_OUTPUT_FORMATS = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".webp": "WEBP",
    ".bmp": "BMP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".gif": "GIF",
}


def resolve_output_path(output: str | Path | None) -> tuple[Path, str]:
    """把用户给的输出路径规范化成 (Path, PIL 保存格式)。

    - 带支持的扩展名（.png/.jpg/.jpeg/.webp/.bmp/.tif/.tiff/.gif）-> 用对应格式保存；
    - 没写扩展名 -> 补 .png（画出来的默认就是 PNG），所以 output="out" 与 "out.png" 同义；
    - 空路径 / 认不出的扩展名 -> ValueError，绝不偷偷换成别的格式。

    大小写不敏感（.PNG 与 .png 等价，保存时显式给 format，不靠 PIL 猜后缀）。
    """
    if output is None or not str(output).strip():
        raise ValueError("必须给标注图的输出路径（含文件名），例如 output='runs/out.png'")
    path = Path(str(output))
    suffix = path.suffix.lower()
    if not suffix:
        path = path.with_suffix(".png")
        suffix = ".png"
    fmt = _OUTPUT_FORMATS.get(suffix)
    if fmt is None:
        raise ValueError(
            f"输出路径的扩展名 {suffix!r} 不支持：{path}\n"
            f"支持 {'/'.join(sorted(_OUTPUT_FORMATS))}；不写扩展名则默认存成 .png。"
        )
    return path, fmt


__all__ = [
    "draw",
    "save_annotated",
    "coerce_detections",
    "resolve_font",
    "resolve_output_path",
    "COLORS",
]
