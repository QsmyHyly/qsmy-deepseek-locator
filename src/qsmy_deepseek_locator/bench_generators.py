# -*- coding: utf-8 -*-
"""评测素材（二）：探测型测试图 —— 圆点阵 / 文字阶梯 / 竖线带 / 分辨率扫描。

**与 bench_shapes.py 的分工**（两类图别混着用）：
    bench_shapes.make_sample 造「已知答案的几何图形图」，测的是**定位准不准**（IoU / 标签）；
    本模块造的图测的是**模型的感知边界在哪** —— 圆点阵量帧几何与坐标约定、文字阶梯量有效
    分辨率、竖线带量最小可分辨线间距、render_scaled 把同一场景铺到不同栅格上做分辨率扫描。

**职责**：只管「怎么画 + 真值是什么」。真值一律是 0.0~1.0 的相对比例（与库的坐标口径一致），
而且都是**算出来的**、不是人标的 —— 这是全程离线评测的前提。

**字体从哪来**：四个画图函数都有一个可选的 font_resolver 参数（缺省用本库 drawing.resolve_font
探测系统字体）。本库不打包字体，所以**自带字体的调用方应当注入自己的**，否则换台机器图就变了 ——
详见 _font() 与 make_marker_image 的 docstring。

**边界**：不调模型、不判分（判分在 bench_score.py）、不认识 Locator。目录定义与提示词属于
调用方（旧演示项目的 objloc/samples/catalog.py），不进本库。

**来历（0.1.3 上游化）**：六个函数原样搬自 deepseek-vision-annotation 的
objloc/samples/generators.py（190 行），逐行搬移，只改了 import 路径与本段说明。
搬它的理由：它是**跨项目共用的评测素材** —— 安卓 App 镜像过同一份代码，而本库自己也要用它
（prompts.py 规定「改提示词必须重跑 bench」，要重跑就得有能暴露问题的图，光有几何图不够）。
"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw

from .bench_shapes import PALETTE, Sample, Shape, _draw_shape
from .drawing import resolve_font


def _font(font_resolver, size):
    """解析「字号 -> 字体对象」：调用方注入了就用它的，否则用本库的探测。

    为什么留这个注入点（2026-09-18 上游化时新增）：本库**不打包字体文件** ——
    安装体积与字体许可都要求如此 —— 默认解析出来的是**当前机器上的**中文字体，
    于是同一段代码在不同机器上画出的文字像素并不相同。而这里造的是**评测素材**，
    「换台机器跑，图还是同一张」是它的基本要求：否则历史评测结论不可比，
    而图看上去完全正常（实测差异 5265 个像素、全部落在文字区域，几何真值一字不差）。
    所以自带字体字节的调用方（旧演示项目、安卓 App）都注入自己的 resolve_font，
    从而与上游化之前**逐字节相同**。
    """
    return (font_resolver or resolve_font)(size)

# --------------------------------------------------------------------------- #
# 生成器 1：彩色圆点阵（量帧几何 / 坐标约定）
# --------------------------------------------------------------------------- #
# 圆点位置（相对坐标），刻意在四角/四边/中心铺开，便于拟合出斜率与截距
MARKER_FRACTIONS = [
    (0.12, 0.14), (0.50, 0.12), (0.88, 0.16),
    (0.14, 0.50), (0.50, 0.50), (0.86, 0.52),
    (0.12, 0.86), (0.50, 0.88), (0.88, 0.84),
]


def make_marker_image(path, width: int, height: int, *, font_resolver=None) -> list[dict]:
    """生成彩色圆点阵，返回真值 [{label, cx, cy, fx, fy, radius}]（像素坐标）。"""
    img = Image.new("RGB", (width, height), (247, 249, 252))
    draw = ImageDraw.Draw(img)
    radius = max(6, int(0.045 * min(width, height)))
    font = _font(font_resolver, max(12, int(min(width, height) * 0.035)))
    names = list(PALETTE.keys())[:len(MARKER_FRACTIONS)]

    truth = []
    for (fx, fy), name in zip(MARKER_FRACTIONS, names):
        cx, cy = fx * width, fy * height
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                     fill=PALETTE[name], outline=(30, 30, 30), width=2)
        draw.text((cx + radius + 6, cy - radius), name, fill=(40, 40, 40), font=font)
        truth.append({"label": name, "cx": cx, "cy": cy, "fx": fx, "fy": fy, "radius": radius})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return truth


def markers_to_gt(truth: list[dict], width: int, height: int) -> list[dict]:
    """把圆点真值转成可评测的 bbox 真值（0.0~1.0，kind=circle）。

    两个刻意选择（都踩过坑）：
    - label **只写颜色名**，不加"圆形"后缀。图上每个点旁边写的就是颜色名，
      MARKER_PROMPT 也只问颜色；若真值要求"红色圆形"而提示词只要颜色，
      标签准确率会恒为 0%——那是真值与提示词打架，不是模型不行。
    - match="center"：圆点直径只有 min(宽,高) 的 9%，在极端扁图上真值框窄到 0.011，
      用 IoU 0.5 判定等于在考"框画得多紧"，而这张图要测的是点位。
      判定实现在 bench_score.py 的 center_hit()（0.1.3 起）。
    """
    gt = []
    for item in truth:
        r = item["radius"]
        gt.append({
            "bbox_2d": [
                round((item["cx"] - r) / width, 4),
                round((item["cy"] - r) / height, 4),
                round((item["cx"] + r) / width, 4),
                round((item["cy"] + r) / height, 4),
            ],
            "label": item["label"],
            "kind": "circle",
            "color_name": item["label"],
            "expect_shape": False,   # 提示词只要颜色，不判形状
            "match": "center",       # 点位判定：真值中心落在预测框内即算命中
        })
    return gt


# --------------------------------------------------------------------------- #
# 生成器 2：文字可读性阶梯（量有效分辨率）
# --------------------------------------------------------------------------- #
# 字号单位是**原始像素**：同一套字号在不同栅格上渲染，模型的最小可读字号会随
# 服务端缩放因子等比变大，由此可反推真实缩放倍数。
TEXT_FONT_SIZES = [64, 48, 36, 28, 22, 16, 12, 9, 7, 5]
_CODE_ALPHABET = "ACDEFGHJKLMNPQRTUVWXY34679"


def random_codes(count: int, seed: int = 20260101) -> list[str]:
    """生成 count 个 4 位随机代码（去掉易混字符，避免判读歧义）。"""
    rng = random.Random(seed)
    return ["".join(rng.choice(_CODE_ALPHABET) for _ in range(4)) for _ in range(count)]


def make_text_image(path, width: int, height: int, sizes, codes, *, font_resolver=None) -> list[dict]:
    """生成字号阶梯图；返回真值 [{index, font_px, code}]。"""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    line_h = height / len(sizes)
    index_font = _font(font_resolver, max(14, int(min(width, height) * 0.022)))
    truth = []
    for i, (size, code) in enumerate(zip(sizes, codes)):
        top = i * line_h
        draw.rectangle([0, top, width - 1, top + line_h - 1], outline=(232, 232, 232), width=2)
        draw.text((width * 0.01, top + line_h / 2 - index_font.size * 0.6),
                  f"{i + 1}", fill=(0, 0, 0), font=index_font)
        font = _font(font_resolver, size)
        draw.text((width * 0.10, top + line_h / 2 - size * 0.62), code, fill=(0, 0, 0), font=font)
        truth.append({"index": i + 1, "font_px": size, "code": code})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return truth


# --------------------------------------------------------------------------- #
# 生成器 3：竖线带（量最小可分辨线间距）
# --------------------------------------------------------------------------- #
BAND_SPECS = [  # (竖线间距 px, 条数) —— 间距递增、条数打乱，防止模型靠猜
    (2, 5), (3, 3), (4, 7), (5, 4), (6, 6),
    (8, 5), (10, 3), (12, 7), (16, 4), (24, 6),
]


def make_band_image(path, width: int, height: int, specs, *, font_resolver=None) -> list[dict]:
    """生成竖线带图片；返回真值 [{index, spacing, lines}]。"""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    band_h = height // len(specs)
    font = _font(font_resolver, max(18, band_h // 4))
    truth = []
    for i, (spacing, lines) in enumerate(specs):
        top = i * band_h
        draw.rectangle([0, top, width - 1, top + band_h - 1], outline=(200, 200, 200), width=2)
        draw.text((20, top + band_h // 2 - band_h // 8), str(i + 1), fill=(0, 0, 0), font=font)
        span = (lines - 1) * spacing
        x0 = width // 2 - span // 2
        for k in range(lines):
            x = x0 + k * spacing
            draw.rectangle([x, top + band_h // 5, x + 1, top + band_h - band_h // 5], fill=(0, 0, 0))
        truth.append({"index": i + 1, "spacing": spacing, "lines": lines})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return truth


# --------------------------------------------------------------------------- #
# 生成器 4：把已知场景等比缩放到任意栅格（分辨率扫描用）
# --------------------------------------------------------------------------- #
def render_scaled(reference: Sample, width: int, height: int, path, *, font_resolver=None) -> Sample:
    """把参考场景等比缩放到目标分辨率重新渲染，返回该尺寸下的 Sample。

    场景先在参考画布（默认 900x720）上生成，再整体缩放渲染 —— 各尺寸的真值（相对比例坐标）
    因此几乎相同，精度差异才只能归因于栅格分辨率 / 服务端重采样，而不是"场景变了"。
    注意必须保持宽高比一致，否则图形会被拉伸。

    ⚠️ **上游化时的实测订正（2026-09-18）**：上游原文（本例与之逐行等价）写的是「真值
    **逐字节相同**」，实测并非如此 —— 因为下面那行 round(v * scale, 1) 先按像素取一位小数、
    之后才除以新宽度归一化。倍数干净的栅格（0.5x / 2x）恰好为 0，而 2400x1920、1280x1024
    这类会露出最大 **1e-4** 的差（归一化坐标，约等于 900px 画布上的 0.09 像素）。
    这个量级对 IoU 判定没有影响（阈值 0.5），所以**保留原行为、只订正措辞**；
    tests/test_bench_generators.py 把边界钉成 ≤1e-4，而不是「完全相等」——
    「完全相同」是个会被后来人当真的错误前提，比那点误差危险得多。
    """
    scale = width / reference.width
    img = Image.new("RGB", (width, height), (247, 249, 252))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width - 1, height - 1], outline=(210, 218, 228), width=2)
    shapes = [
        Shape(kind=s.kind, color_name=s.color_name, rgb=s.rgb,
              bbox_px=tuple(round(v * scale, 1) for v in s.bbox_px))
        for s in reference.shapes
    ]
    for shape in shapes:
        _draw_shape(draw, shape)
    draw.text((16, 12), "#1", fill=(150, 160, 175), font=_font(font_resolver, max(10, int(22 * scale))))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return Sample(name=path.stem, path=str(path), width=width, height=height, shapes=shapes)

