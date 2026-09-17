"""打标绘制：把 Detection 画回图片上（框 / 点 + 中文标签）。

坐标口径：传进来的 Detection 是 0.0~1.0 归一化比例，这里才换算成像素。
越界值在绘制前会被**夹紧到边界**（而不是抛异常）—— 画在边上总好过整张图打不出来；
但真正的判断（这是不是像素坐标）归 parsing.check_coordinate_range，本模块只负责画。

中文字体是这里最容易被忽略的坑，有两层：

1. **按「字体名优先」探测，不是按目录顺序**：外循环是候选文件名、内循环才是目录
   （见 _find_font_file）。所以决定命中谁的是**名字在 _FONT_CANDIDATES 里的位次**，
   目录只决定它在哪。想加一台机器上的字体，优先把它加进候选名，而不是调目录顺序。
2. **文件名不足以说明它含中文字形**。实测安卓上 /system/fonts/DroidSans.ttf 是
   Roboto-Regular.ttf 的软链 —— 名字像中文字体，实际只有拉丁字形，拿它渲染中文
   只会得到一片豆腐块。所以选定字体后要**真渲染一次做校验**（见 _renders_cjk），
   校验不过就继续往后找。这一步要真的解码字体，代价换来的是「不会静默画方块」。

全都找不到时退回 PIL 内置位图字体（英文标签仍可读，中文会变成方块），并**发一次告警**
（见 _warn_no_cjk）—— 安卓上曾经因为这一步是静默的，图能正常出、只有标签是豆腐块，
排查了很久才发现。要自己指定字体：resolve_font(size, font_path=...) 或环境变量
QSML_FONT_PATH / QSML_FONT_DIR。

@doc README.md#8-常见问题
（该文档解决"中文标签变成方块了怎么办、字体路径怎么给"的问题。）
"""

from __future__ import annotations

import os
import uuid
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageColor, ImageDraw, ImageFont

from .errors import OutputPathError, WriteError
from .images import load_image
from .parsing import Detection, parse_detections

# 手写色优先（语义清晰、区分度高），其后追加 PIL 全部命名颜色并去重。
_EXTRA_COLORS = [name for (name, _code) in ImageColor.colormap.items()]
COLORS: list[str] = list(dict.fromkeys([
    "red", "green", "blue", "yellow", "orange", "pink", "purple", "brown", "gray",
    "beige", "turquoise", "cyan", "magenta", "lime", "navy", "maroon", "teal",
    "olive", "coral", "lavender", "violet", "gold", "silver",
] + _EXTRA_COLORS))

# 跨平台中文字体候选，顺序 = 命中概率（见模块头第 1 条：这是**名字优先**的排序）。
# 开头的五条由安卓贡献：这个库最早只认桌面三大平台，安卓上必然探测失败，
# 于是中文标签静默变成豆腐块 —— 而 /system/fonts/NotoSansCJK-Regular.ttc 其实老早
# 就写在本列表里，**只差一个目录**（见 _FONT_DIRS）。别把不含中文的（Arial 之类）排前面。
_FONT_CANDIDATES = [
    "MiSans-Regular.ttf", "MiSansVF.ttf",     # 安卓 · 小米（VF 是可变字体，PIL 取默认实例）
    "HarmonyOS_Sans_SC_Regular.ttf",          # 安卓 · 华为
    "NotoSansCJK-Regular.ttc",                # 安卓 / Linux 通用（同文件含 SC/TC/JP/KR 多个 face）
    "NotoSansCJKsc-Regular.otf",
    "DroidSansFallback.ttf", "DroidSansFallbackFull.ttf",   # 安卓老 ROM 的兜底中文字体
    "msyh.ttc", "msyhbd.ttc",                 # Windows 微软雅黑
    "simhei.ttf", "simsun.ttc",               # Windows 黑体 / 宋体
    "PingFang.ttc", "Hiragino Sans GB.ttc",   # macOS
    "wqy-microhei.ttc", "wqy-zenhei.ttc",     # Linux
    "DejaVuSans.ttf", "Arial.ttf",            # 兜底：不含中文，但至少是矢量字体
]

# 探测目录。安卓两条放最前（移动端优先），其余是桌面平台的常规位置。
# ⚠️ 目录顺序只在**候选名相同时**才起作用：命中哪个字体由 _FONT_CANDIDATES 的位次决定。
_FONT_DIRS = [
    Path("/system/fonts"),                # 安卓（绝大多数 ROM 的字体都在这）
    Path("/product/fonts"),               # 部分 ROM 把字体放这里
    Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path.home() / ".fonts",
    Path("/System/Library/Fonts"),
    Path("/Library/Fonts"),
]

# 用户显式指定的字体路径（环境变量）。它的**权威性高于探测**：给了就用，即使探测
# 能在别处找到中文字体。理由：能被这条路径影响的人，正是那些「这台机器上哪个字体能用
# 只有我知道」的人（安卓 App、精简过的容器镜像）。
FONT_PATH_ENV = "QSML_FONT_PATH"
# 用户显式指定的字体目录（环境变量）。只影响**探测范围**，找到的字体仍要过渲染校验 ——
# 它解决的是「字体在这台机器上，只是不在我知道的目录里」。
FONT_DIR_ENV = "QSML_FONT_DIR"

# 字体对象缓存：键是 (字体文件路径, 字号)。路径进键是必须的 —— 同一个进程里可能先用
# 默认字体、再被显式指定成另一个文件（App 就是这么干的），只按字号缓存会串味。
# 上限存在的意义：font_path 由调用方给，理论上可以给出很多个不同的值。
_font_cache: dict[tuple[str, int], Any] = {}
_FONT_CACHE_MAX = 32

# 告警只发一次的开关。**刻意不做成"每次降级都发"**：draw 会按多个字号调用 resolve_font，
# 一次打标就能刷出好几条同样的告警，反而把真正该看的那条埋掉。
_warned_no_cjk = False


@lru_cache(maxsize=64)
def _renders_cjk(path: str) -> bool:
    """真渲染一遍，判断这个字体文件有没有中文字形。

    判据（来自安卓 App 的实测，见 drawing.py 模块头第 2 条）：把「中」与**肯定不存在**的
    U+FFFF 各画一遍，逐像素比较 —— 两者位图完全相同，说明「中」也被当成了 .notdef（豆腐块）。

    为什么要额外判一次墨迹量（>20 个亮像素）：「中」与 U+FFFF 不同**也可能**是因为该字体
    给 .notdef 画了个带轮廓的空框，而中文根本没画出来。多这一条，是让判据偏保守 ——
    宁可误判「这个字体不行」去试下一个，也不要误判「行」然后画出方块。

    失败一律返回 False，不抛异常：它跑在探测路径上，探测失败的正确后果是「继续找下一个」。
    """
    try:
        box = Image.new("L", (96, 48), 0)
        painter = ImageDraw.Draw(box)
        hit_font = ImageFont.truetype(path, 32)
        painter.text((4, 4), "中", fill=255, font=hit_font)
        hit = box.tobytes()
        box2 = Image.new("L", (96, 48), 0)
        ImageDraw.Draw(box2).text((4, 4), "\uffff", fill=255, font=hit_font)
        miss = box2.tobytes()
    except Exception:  # noqa: BLE001 - 探测路径，失败就当作「这个字体不行」
        return False
    return hit != miss and sum(1 for value in hit if value > 32) > 20


def _load_truetype(path: str | None, size: int, *, verify: bool) -> Any | None:
    """开字体文件；开了但**确认不含中文字形**时返回 None（表示"换一个试试"）。

    verify=False 时跳过渲染校验，给两条路用：用户显式指定的字体（他的判断优先于我们的），
    以及二次尝试（已经校验过一遍，同一文件不必再渲染）。

    ⚠️ 注意「文件打不开」与「能打开但没中文」是两回事，返回值却都是 None：
    前者说明路径本身不可用，调用方该退回内置字体；后者该继续往后找。区分它们的责任
    在 resolve_font（它是唯一知道"还有没有下一个候选"的地方），这里只如实报告结果。
    """
    if not path:
        return None
    try:
        font = ImageFont.truetype(path, size=size)
    except Exception:  # noqa: BLE001 - 文件损坏 / 不是字体 / 权限不足
        return None
    if verify and not _renders_cjk(path):
        return None
    return font


def _warn_no_cjk() -> None:
    """找不到含中文字形的字体时告警一次（同一进程只吵一次）。

    为什么必须说出来：降级到 load_default() 时**图还是能正常出**，只是中文标签变成方块。
    这种「悄悄坏掉」在安卓上代价极大 —— 用户看到的是「定位成功、标签乱码」，
    第一反应是模型不行或我们解析错了，而真实原因只是少了一个目录。宁可吵，不可静默。
    """
    global _warned_no_cjk
    if _warned_no_cjk:
        return
    _warned_no_cjk = True
    warnings.warn(
        "没有找到含中文字形的字体文件，已退回 PIL 内置位图字体："
        "英文标签可以正常显示，**中文标签会变成方块（豆腐块）**。\n"
        "三种给法任选一种：\n"
        "  1) resolve_font(22, font_path='/path/to/NotoSansCJK-Regular.ttc')；\n"
        "  2) 环境变量 QSML_FONT_PATH=/path/to/字体文件；\n"
        "  3) 环境变量 QSML_FONT_DIR=/path/to/字体目录（库会在里面按候选名找）。\n"
        f"本机探测过的目录：{[str(p) for p in _FONT_DIRS]}",
        UserWarning,
        stacklevel=4,   # 指到调用 resolve_font 的那一行（draw -> resolve_font -> 这里）
    )


def _find_font_file() -> str | None:
    """按「名字优先」找出一个**确认能渲染中文**的字体文件路径；找不到返回 None。

    ⚠️ 这里返回的必须已经过 _renders_cjk 校验：只看文件名会踩到
    /system/fonts/DroidSans.ttf（Roboto 的软链，名字像中文字体，实际只有拉丁字形）。
    唯一的例外是开了 QSML_FONT_PATH —— 那是用户在明确说「就用这个」，不再由本函数替他判断。

    目录来源：QSML_FONT_DIR 给了就只用它（用户明确限定了范围），否则用 _FONT_DIRS。
    """
    explicit = (os.environ.get(FONT_PATH_ENV) or "").strip()
    if explicit and Path(explicit).exists():
        return explicit

    env_dir = (os.environ.get(FONT_DIR_ENV) or "").strip()
    directories = [Path(env_dir)] if env_dir else _FONT_DIRS

    for name in _FONT_CANDIDATES:
        for directory in directories:
            candidate = directory / name
            if candidate.exists() and _renders_cjk(str(candidate)):
                return str(candidate)
        # 有些发行版把字体放在子目录里，做一次浅层递归
        for directory in directories:
            if not directory.exists():
                continue
            for found in directory.rglob(name):
                if _renders_cjk(str(found)):
                    return str(found)
    return None


def resolve_font(size: int = 20, *, font_path: str | Path | None = None) -> Any:
    """取一个尽量支持中文的字体对象（== 同一个字体文件 + 字号**只开一次文件**）。

    字体文件的来源，优先级从高到低（前一个拿不到就试后一个）：

        1. font_path 参数         —— 调用方当场指定，**不再做中文校验**（你的判断优先）
        2. QSML_FONT_PATH         —— 环境变量指定，同上
        3. QSML_FONT_DIR          —— 只限定探测目录，找到的字体仍要过 _renders_cjk
        4. _FONT_DIRS 探测         —— 名字优先，命中后要过 _renders_cjk
        5. PIL 内置位图字体         —— 兜底，此时**会发一次告警**（中文会变方块，见 _warn_no_cjk）

    Args:
        size: 字号（像素）。
        font_path: 直接指定字体文件。给了就用它，连探测都不做 —— 这条路存在的意义是
            「这台机器上哪个字体能用只有我知道」，典型场景是安卓 App（/system/fonts 下的
            字体五花八门，还有 Roboto 软链冒充中文字体）。给一个打不开的路径不算错误，
            会退回探测并最终告警。

    Returns:
        PIL 的字体对象（ImageFont.FreeTypeFont 或内置位图字体）。

    为什么参数是 font_path 而不是 font 对象：缓存按 (路径, 字号) 做键才成立，
    传对象进来的话「同一个字体换个字号」就得重新开文件，而打标路径上字号是常量、
    字体文件却可能很大（NotoSansCJK 是 32MB，安卓上每次重开都是实打实的 IO）。
    """
    key = (str(font_path) if font_path else "", size)
    if key in _font_cache:
        return _font_cache[key]

    if font_path:
        font = _load_truetype(str(font_path), size, verify=False)
        if font is not None:
            return _remember(key, font)

    path = _find_font_file()
    if path:
        # verify=False：_find_font_file 只返回**已经过校验**的路径（或用户显式指定的），
        # 这里再渲染一遍纯属白花 —— 但文件仍可能按这个字号打不开，所以仍要检查返回值。
        font = _load_truetype(path, size, verify=False)
        if font is not None:
            return _remember(key, font)

    _warn_no_cjk()
    return _remember(key, ImageFont.load_default())


def _remember(key: tuple[str, int], font: Any) -> Any:
    """记录缓存（超过上限时清空，而不是做成 LRU —— 这里要的是"别再开文件"，不是命中率）。"""
    if len(_font_cache) >= _FONT_CACHE_MAX:
        _font_cache.clear()
    _font_cache[key] = font
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
    font_path: str | Path | None = None,
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
        font_path: 指定中文字体文件（默认自动探测）。字体探测失败时中文标签会变方块，
            所以「我知道这台机器上哪个字体能用」的场景就该显式传它，别让库去猜 ——
            see resolve_font 的优先级列表。

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
    font = resolve_font(font_size, font_path=font_path)
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
            认不出的扩展名抛 OutputPathError（也是 ValueError）—— 理由见 resolve_output_path。
        output_dir / stem: 不给 path 时用它们拼；两者都缺省则落在当前工作目录，
            文件名 annotated_<12位hex>.png。

    Raises:
        OutputPathError: 路径本身不可用（空、后缀不认识、父目录是个文件……）。
        WriteError: 路径没问题但写不动（磁盘满、只读挂载、没有写权限），
            原始 OSError 在 __cause__ 上。两者都是 LocatorError，一把兜得住。
    """
    annotated = draw(image, items, **draw_kwargs)
    if path is None:
        directory = Path(output_dir) if output_dir else Path.cwd()
        _prepare_dir(directory)
        name = stem or f"annotated_{uuid.uuid4().hex[:12]}"
        path = directory / f"{name}.png"
    # 走 resolve_output_path，而不是写死 format="PNG"：本函数是公开 API，
    # 调用方给 "out.jpg" 就该拿到 JPEG。写死 PNG 会产出「文件名 .jpg、内容却是 PNG」
    # 的图 —— 正是下面 _OUTPUT_FORMATS 注释里点名的那类最难排查的问题。
    # CLI 的 -o 与 Locator.locate_and_draw 都走这里，所以这一处同时修好三条路径。
    target, fmt = resolve_output_path(path)
    _prepare_dir(target.parent)
    try:
        annotated.save(target, format=fmt)
    except (OSError, ValueError) as exc:
        # 落盘是**最后一步**，崩在这里意味着前面那次 API 调用已经花掉了。
        # 所以这里必须把话说全：哪个路径、什么原因、先前有没有成功。
        # 0.1.2 及以前这一行是裸的 annotated.save()，抛出去的是原生 OSError /
        # ValueError（PIL 认不出格式时），只写 except LocatorError 的调用方直接漏网。
        raise WriteError(
            f"标注图写入失败：{target}（{type(exc).__name__}: {exc}）\n"
            "识别已经完成，只是图没落盘；换个可写目录或用绝对路径重试即可，不必再调一次模型。"
        ) from exc
    return target


def _prepare_dir(directory: Path) -> None:
    """建出输出目录，失败时把 OSError 转成 LocatorError（而不是让它裸奔）。

    这是「输出位置」问题的唯一一处兜底，被 save_annotated 和 Locator.locate_to_file 共用 ——
    P2-1 反馈的正是「默认输出位置相对 CWD，安卓上 CWD 是 / 或不可写，于是直接失败」，
    失败可以，但要失败得看得懂：报错里必须带上**具体路径**。
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise WriteError(
            f"输出目录无法使用：{directory}（{type(exc).__name__}: {exc}）\n"
            "常见原因：路径不可写 / 是个文件而不是目录 / 相对路径的解析基准（CWD）不对。"
            "移动端或服务端请直接传绝对路径。"
        ) from exc


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
    - 空路径 / 认不出的扩展名 -> OutputPathError（同时是 ValueError 与 LocatorError），
      绝不偷偷换成别的格式。

    大小写不敏感（.PNG 与 .png 等价，保存时显式给 format，不靠 PIL 猜后缀）。
    """
    if output is None or not str(output).strip():
        raise OutputPathError("必须给标注图的输出路径（含文件名），例如 output='runs/out.png'")
    path = Path(str(output))
    suffix = path.suffix.lower()
    if not suffix:
        path = path.with_suffix(".png")
        suffix = ".png"
    fmt = _OUTPUT_FORMATS.get(suffix)
    if fmt is None:
        raise OutputPathError(
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
    # 字体探测的两个环境变量名，做成常量导出：
    # 调用方（尤其是打包脚本 / App 侧）能 import 到，就不必再手抄字符串。
    "FONT_PATH_ENV",
    "FONT_DIR_ENV",
]
