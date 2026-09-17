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


# 标签底色块四周的留白（px）。3 是历史值，别随手改：它同时决定了标签要在框上方
# 预留多少、以及贴边时能夹出多少余量。
_LABEL_PAD = 3

# 标签避让的最多尝试次数。刻意取小：避让太激进会把标签推到离目标很远的地方，
# 读图的人反而对不上它标的是哪个框 —— 宁可两个标签挨着，也不要它们各奔东西。
_LABEL_AVOID_TRIES = 4

# 不显式给 font_size / box_width / point_radius 时的历史默认值（scale_to_image=False 用）。
_DEFAULT_FONT_SIZE = 22
_DEFAULT_BOX_WIDTH = 3
_DEFAULT_POINT_RADIUS = 5


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: Any) -> tuple[int, int, int, int]:
    """量一段文字，返回 (left, top, width, height)。

    why 连 left/top 一起返回：textbbox 在 (0, 0) 上量出的 left/top 通常不是 0
    （字形自带 ascent/descent 偏移），绘制时必须把这两个偏移减掉，文字才会正正好
    落在底色块里 —— 不减的话中文标签会贴着块的上沿、甚至冒出去一截。
    某些内置位图字体没有 textbbox，退回一个够用的估算值。
    """
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return int(left), int(top), max(1, int(right - left)), max(1, int(bottom - top))
    except Exception:  # noqa: BLE001
        return 0, 0, max(1, len(text) * 8), 16


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """两个矩形是否相交（边贴边不算）。"""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _label_rect(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: Any,
    anchor: tuple[float, float],
    bounds: tuple[int, int] | None = None,
    box: tuple[int, int, int, int] | None = None,
    avoid: Sequence[tuple[int, int, int, int]] | None = None,
) -> tuple[int, int, int, int]:
    """算出标签底色块的最终矩形（画布坐标）。

    这是「标签被切在图片外」那类问题的唯一现场，所以单独抽出来：单测可以直接断言
    矩形，不必去数像素。放置顺序是有讲究的 ——

    1. **贴顶放不下就翻到框内侧**：标签默认画在框上方（ay1 - 字高 - 6），目标贴着
       图片上边时框外根本没地方，单纯夹紧只能把它压在顶上、上沿仍被切掉一条
       （实测 1494x2047 的图上「天空」标签正好撞上这个）；翻进框内则一定在画布里。
    2. **躲开已经画过的标签**：多目标密集时（实测一张图 17 个目标）标签会叠在一起，
       向下挪一个标签高再试，最多 _LABEL_AVOID_TRIES 次。
    3. **四边夹紧**：标签比画布还宽/高时，宁可让它压边，也不把文字裁掉。

    Args:
        anchor: 期望的标签左上角（含底色留白）。
        bounds: 画布 (宽, 高)。给 None 表示不做任何约束（保持旧行为）。
        box: 该目标的框像素坐标 (x1, y1, x2, y2)，用于第 1 步的翻转。
        avoid: 已占用的标签矩形列表，用于第 2 步。

    Returns:
        (x1, y1, x2, y2)；给了 bounds 时保证落在画布内。
    """
    _l, _t, text_w, text_h = _text_size(draw, text, font)
    w, h = text_w + 2 * _LABEL_PAD, text_h + 2 * _LABEL_PAD
    x, y = int(anchor[0]), int(anchor[1])

    if bounds is None:
        return (x, y, x + w, y + h)

    canvas_w, canvas_h = bounds

    # 1. 顶边放不下 -> 翻到框内侧顶部
    if box is not None and y < 0 and box[1] + h <= canvas_h:
        y = box[1]

    # 2. 躲开已画过的标签
    if avoid:
        for _ in range(_LABEL_AVOID_TRIES):
            candidate = (x, y, x + w, y + h)
            if not any(_overlaps(candidate, other) for other in avoid):
                break
            if y + h + 1 > canvas_h:
                break
            y += h + 1

    # 3. 夹进画布：标签比画布宽/高时 min() 得到负数，被外层 max(0, ..) 兜住。
    x = max(0, min(x, canvas_w - w))
    y = max(0, min(y, canvas_h - h))
    return (x, y, x + w, y + h)


def _draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    color: str,
    font: Any,
    *,
    bounds: tuple[int, int] | None = None,
    box: tuple[int, int, int, int] | None = None,
    avoid: Sequence[tuple[int, int, int, int]] | None = None,
) -> tuple[int, int, int, int]:
    """画带底色的小标签，保证在任意背景上都读得清（底色亮度决定黑字还是白字）。

    Args:
        xy: 期望位置（标签左上角）。给了 bounds 时可能被挪动，见 _label_rect。
        bounds / box / avoid: 透传给 _label_rect。

    Returns:
        标签实际占据的矩形，供调用方登记进 avoid。
    """
    glyph_left, glyph_top, _tw, _th = _text_size(draw, text, font)
    rect = _label_rect(draw, text, font, xy, bounds=bounds, box=box, avoid=avoid)

    try:
        rgb = ImageColor.getrgb(color)
        luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        text_color = (0, 0, 0) if luminance > 140 else (255, 255, 255)
    except Exception:  # noqa: BLE001
        text_color = (255, 255, 255)

    draw.rectangle(rect, fill=color)
    # 减掉字形自身的 left/top 偏移，文字才落在块里而不是顶在块外
    draw.text(
        (rect[0] + _LABEL_PAD - glyph_left, rect[1] + _LABEL_PAD - glyph_top),
        text, fill=text_color, font=font,
    )
    return rect


def _auto_style(width: int, height: int) -> tuple[int, int, int]:
    """按图片尺寸推一组 (框线宽, 字号, 点半径)。

    why 按**短边**算：文字是方的，按长边算会让狭长图上的字大到荒唐 —— 实测
    1494x2047 的图按长边能得到 51px 的字，几乎盖住半个画面。分母 40 来自手调经验：
    1600x1200 上得到 30px，与手工调优的 28 很接近；再夹到 12~48 防两端失控。

    什么时候用得上：draw(..., scale_to_image=True)。默认不开，是因为它改变的是
    观感而不是正确性，不该悄悄改掉已有调用方的输出。
    """
    short = max(1, min(width, height))
    size = max(12, min(48, short // 40))
    return max(1, int(round(size / 7.0))), size, max(2, int(round(size / 5.0)))



def draw(
    image: Any,
    items: Any,
    *,
    box_width: int | None = None,
    point_radius: int | None = None,
    font_size: int | None = None,
    draw_label: bool = True,
    colors: Sequence[str] | None = None,
    scale_to_image: bool = False,
) -> Image.Image:
    """在图片上画出 Detection（框 / 点 + 标签），返回新图（不改动入参图）。

    Args:
        image: 路径 / URL / bytes / PIL.Image / data URL。
        items: Detection 列表、模型字段字典列表，或模型输出的坐标 JSON 文本。
        colors: 自定义调色板；留空用内置 COLORS，按目标顺序取色。
        box_width / point_radius / font_size: 显式指定；留 None 时用历史默认
            （线宽 3 / 半径 5 / 字号 22），或按 scale_to_image 自适应。显式值优先级最高。
        scale_to_image: 按图片尺寸自动推线宽 / 字号 / 半径。**大图配小字**是实测踩到的坑：
            3px 的框线画在 4000px 宽的图上细得几乎看不见，22px 的标签同理。
            默认 False —— 它改的是观感而不是正确性，不该悄悄改掉已有调用方的输出。
        draw_label: 关掉可得到只有框、没有文字的干净标注图。

    Returns:
        画好的新图（RGB）。标签默认画在框上方；贴图片边缘时会翻到框内侧、被别的标签
        压住时会向下错开，四边都保证不越界 —— 规则见 _label_rect。
    """
    img = load_image(image).convert("RGB")
    detections = coerce_detections(items)
    palette = list(colors) if colors else COLORS
    width, height = img.size

    if scale_to_image:
        auto_box, auto_font, auto_radius = _auto_style(width, height)
    else:
        auto_box, auto_font, auto_radius = (
            _DEFAULT_BOX_WIDTH, _DEFAULT_FONT_SIZE, _DEFAULT_POINT_RADIUS,
        )
    box_width = auto_box if box_width is None else box_width
    font_size = auto_font if font_size is None else font_size
    point_radius = auto_radius if point_radius is None else point_radius

    painter = ImageDraw.Draw(img)
    font = resolve_font(font_size)
    # 已经画出去的标签矩形，用来给后面的标签让位（见 _label_rect 第 2 步）
    placed: list[tuple[int, int, int, int]] = []

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
                # 这里不再做 max(0, ..)：位置交给 _label_rect 统一决定，它还要负责
                # 「贴顶翻进框内」和「躲开别的标签」两件事。
                placed.append(_draw_label(
                    painter, (ax1, ay1 - font_size - 6), label, color, font,
                    bounds=(width, height), box=(ax1, ay1, ax2, ay2), avoid=placed,
                ))

        if det.point is not None:
            cx, cy = _ratio_to_abs(det.point[0], width), _ratio_to_abs(det.point[1], height)
            painter.ellipse(
                [(cx - point_radius, cy - point_radius), (cx + point_radius, cy + point_radius)],
                outline=color, width=box_width,
            )
            painter.ellipse([(cx - 1, cy - 1), (cx + 1, cy + 1)], fill=color)
            if draw_label and label:
                placed.append(_draw_label(
                    painter, (cx + point_radius + 4, cy + 2), label, color, font,
                    bounds=(width, height), avoid=placed,
                ))

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
    """画出标注并保存，返回保存路径；格式由 path 的扩展名决定。

    Args:
        path: 直接给完整文件路径（优先）。扩展名决定保存格式
            （.png/.jpg/.jpeg/.webp/.bmp/.tif/.tiff/.gif），不写扩展名补 .png，
            认不出的扩展名抛 ValueError —— 理由见 resolve_output_path。
        output_dir / stem: 不给 path 时用它们拼；两者都缺省则落在当前工作目录，
            文件名 annotated_<12位hex>.png。
    """
    annotated = draw(image, items, **draw_kwargs)
    if path is None:
        directory = Path(output_dir) if output_dir else Path.cwd()
        directory.mkdir(parents=True, exist_ok=True)
        name = stem or f"annotated_{uuid.uuid4().hex[:12]}"
        path = directory / f"{name}.png"
    # 走 resolve_output_path，而不是写死 format="PNG"：本函数是公开 API，
    # 调用方给 "out.jpg" 就该拿到 JPEG。写死 PNG 会产出「文件名 .jpg、内容却是 PNG」
    # 的图 —— 正是下面 _OUTPUT_FORMATS 注释里点名的那类最难排查的问题。
    # CLI 的 -o 与 Locator.locate_and_draw 都走这里，所以这一处同时修好三条路径。
    target, fmt = resolve_output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(target, format=fmt)
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
