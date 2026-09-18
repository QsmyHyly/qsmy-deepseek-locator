"""测评素材：合成一张「已知答案」的几何图形图，真值顺手算出来。

**职责**：颜色 / 形状词表、Shape 与 Sample 两个数据模型、造图（make_sample）与批量落盘
（make_samples，含 ground_truth.json）。真值是**算出来的**而不是人标的 —— 这正是本库评测
能自动判分、且能做到全程离线的前提。

**边界**：本模块只管「造素材」，不管「判分」（在 bench_score.py），也不管「跑一轮」
（在 benchmark.py）。它不认识 Locator、不调模型；外部依赖只有 Pillow 与
drawing.resolve_font（图片角标那行序号字）。

与拆分前旧文件的对应关系（0.1.2 -> 0.1.3 的等价重构，行为零变化）：
    词表（原 benchmark.py 第 41-104 行）、数据模型（第 107-153 行）、造图（第 156-293 行）
    三段原样搬到这里；benchmark.py 反过来 import 回来，所以
    「from qsmy_deepseek_locator.benchmark import make_sample」这条老路径照旧可用。

⚠️ PALETTE / COLOR_SYNONYMS 是**判分口径的一部分**（bench_score 里的 color_ok / shape_ok
直接依赖它们）。改词表等于改评测口径，历史报告立刻不可比 —— 要改请连同种子一起换一批图。
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw

from .drawing import resolve_font


# --------------------------------------------------------------------------- #
# 颜色与形状词表
# --------------------------------------------------------------------------- #
PALETTE: dict[str, tuple[int, int, int]] = {
    "红色": (214, 48, 49),
    "绿色": (46, 160, 67),
    "蓝色": (52, 96, 219),
    "橙色": (245, 148, 20),
    "紫色": (150, 70, 200),
    "黄色": (232, 197, 20),
    "青色": (32, 178, 190),
    "粉色": (238, 120, 180),
    "灰色": (130, 138, 150),
    "棕色": (140, 90, 50),
}

# 判定用的同义词：模型答「红」和答「红色」都对，不能因为少一个字判错。
# ⚠️ 这张表只用来避免「答对了却判错」，不能拿来兜住错误答案。
COLOR_SYNONYMS: dict[str, list[str]] = {
    "红色": ["红"],
    "绿色": ["绿"],
    "蓝色": ["蓝"],
    "橙色": ["橙", "橘"],
    "紫色": ["紫"],
    "黄色": ["黄"],
    "青色": ["青", "蓝绿"],
    "粉色": ["粉"],
    "灰色": ["灰"],
    "棕色": ["棕", "褐"],
}

SHAPE_CN = {
    "rect": "矩形",
    "circle": "圆形",
    "ellipse": "椭圆形",
    "triangle": "三角形",
}

# 评测用的默认提问。⚠️ 这里**不能出现真值标签本身**（颜色名、形状名），
# 否则模型会照着提示词的词复述，指标虚高。
DEFAULT_BENCH_TARGET = "几何图形"


def shape_ok(kind: str, label: str) -> bool:
    """预测标签是否描述了正确的形状（容忍常见说法）。"""
    if kind == "rect":
        return any(k in label for k in ("矩形", "长方形", "方形", "四边形", "方块"))
    if kind == "circle":
        return "圆" in label and "椭" not in label
    if kind == "ellipse":
        return "椭" in label
    if kind == "triangle":
        return any(k in label for k in ("三角", "角形"))
    return False


def color_ok(color_name: str, label: str) -> bool:
    """预测标签是否描述了正确的颜色。"""
    synonyms = COLOR_SYNONYMS.get(color_name) or [color_name[:1]]
    return any(s in label for s in synonyms)


# --------------------------------------------------------------------------- #
# 文本标签判定（第三套口径，与 color_ok / shape_ok 并列）
# --------------------------------------------------------------------------- #
# 网页元素的名称里经常带这些（"总销售额（今日）"、"加入购物车 >"），不归一化会误判为读错。
_TEXT_NOISE = re.compile(r"[\s，。、,.:：;；!！?？\"'“”‘’()（）\[\]【】<>《》/\\|_\-—~\`·]+")


def _fold_text(value: str) -> str:
    """归一化文本标签：去掉空白与标点、统一小写。"""
    return _TEXT_NOISE.sub("", str(value or "")).lower()


def text_label_ok(gt_label: str, pred_label: str, *, aliases: Iterable[str] = (),
                  min_len: int = 2) -> bool:
    """「文本标签」的判定口径：归一化后相等，或一方包含另一方。

    为什么不能复用 color_ok / shape_ok：那套是给几何图形用的（颜色名 + 形状名），
    而网页截图里的目标标签是**界面上的中文名称**（"总销售额"、"加入购物车"），
    既没有颜色也没有形状，硬套会得到恒为 0 的标签准确率。

    为什么允许"一方包含另一方"：模型常在名称前后补限定语（"KPI 卡片：总销售额"），
    也会把长文案截断（"无线降噪耳机" → "降噪耳机"）——这两种都算读对了。
    为防单字误命中，要求被包含的一方至少 min_len 个字符。

    aliases 是**同一元素的其它合理叫法**（真值里的 aliases，例如搜索框既写"搜索商品"
    也可以叫"搜索框"）。它只用来避免"答对了却判错"，不是用来兜住错误答案的：
    别名必须是"看着这张图的人也可能这么说"的名字。
    ⚠️ 别把模型可能给出的答案整批抄成别名——那样这项指标就失去意义了。

    真值在声明 label_mode="text" 时才会走到这里（见 bench_score.evaluate_sample）。
    """
    a = _fold_text(gt_label)
    if not a:
        return False
    for candidate in (a, *(_fold_text(x) for x in aliases)):
        b = _fold_text(pred_label)
        if not candidate or not b:
            continue
        if candidate == b:
            return True
        if len(candidate) >= min_len and len(b) >= min_len and (candidate in b or b in candidate):
            return True
    return False


def _darken(rgb: tuple[int, int, int], factor: float = 0.65) -> tuple[int, int, int]:
    return tuple(max(0, int(c * factor)) for c in rgb)  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# 数据模型
# --------------------------------------------------------------------------- #
@dataclass
class Shape:
    kind: str
    color_name: str
    rgb: tuple[int, int, int]
    bbox_px: tuple[float, float, float, float]

    @property
    def label(self) -> str:
        return f"{self.color_name}{SHAPE_CN[self.kind]}"

    def to_gt(self, width: int, height: int) -> dict:
        x1, y1, x2, y2 = self.bbox_px
        return {
            "bbox_2d": [
                round(x1 / width, 4), round(y1 / height, 4),
                round(x2 / width, 4), round(y2 / height, 4),
            ],
            "label": self.label,
            "kind": self.kind,
            "color_name": self.color_name,
        }


@dataclass
class Sample:
    name: str
    path: str
    width: int
    height: int
    shapes: list[Shape] = field(default_factory=list)

    def ground_truth(self) -> list[dict]:
        return [s.to_gt(self.width, self.height) for s in self.shapes]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "shapes": [asdict(s) for s in self.shapes],
            "ground_truth": self.ground_truth(),
        }


# --------------------------------------------------------------------------- #
# 造图
# --------------------------------------------------------------------------- #
def _draw_shape(painter: ImageDraw.ImageDraw, shape: Shape) -> None:
    x1, y1, x2, y2 = shape.bbox_px
    box = [x1, y1, x2, y2]
    fill = shape.rgb
    outline = _darken(shape.rgb)
    if shape.kind == "rect":
        painter.rectangle(box, fill=fill, outline=outline, width=5)
    elif shape.kind in ("circle", "ellipse"):
        painter.ellipse(box, fill=fill, outline=outline, width=5)
    elif shape.kind == "triangle":
        cx = (x1 + x2) / 2
        painter.polygon([(cx, y1), (x2, y2), (x1, y2)], fill=fill, outline=outline)
    else:  # pragma: no cover - 词表是常量，不会走到
        raise ValueError(f"未知形状：{shape.kind}")


def make_sample(
    index: int,
    *,
    seed: int | None = None,
    width: int = 900,
    height: int = 720,
    n_shapes: int = 3,
) -> tuple[Sample, Image.Image]:
    """生成一张图和它的真值（不落盘）。

    布局策略：把画布切成 2x2 网格、随机挑格子，尽量不让图形互相重叠 ——
    重叠会让 IoU 匹配产生歧义，测出来的分数就不是「找没找到」而是「怎么切重叠区」了。
    """
    rng = random.Random(seed if seed is not None else index)

    img = Image.new("RGB", (width, height), (247, 249, 252))
    painter = ImageDraw.Draw(img)
    painter.rectangle([0, 0, width - 1, height - 1], outline=(210, 218, 228), width=2)

    cols = rows = 2
    cells = [(c, r) for r in range(rows) for c in range(cols)]
    rng.shuffle(cells)
    cells = cells[:max(1, min(n_shapes, len(cells)))]

    kinds = list(SHAPE_CN.keys())
    colors = list(PALETTE.keys())
    used_colors: set[str] = set()
    shapes: list[Shape] = []
    cell_w, cell_h = width / cols, height / rows
    # ⚠️ 留白必须随画布缩放。写死 60px 时，300x200 这种小画布上
    # (cell_w/2 - 60) 会变成负数，于是框宽为负、PIL 直接抛
    # "y1 must be greater than or equal to y0"（自测抓到的第一个 bug）。
    margin = max(6.0, min(60.0, min(cell_w, cell_h) * 0.12))

    for c, r in cells:
        cx1, cy1 = c * cell_w + margin, r * cell_h + margin
        cx2, cy2 = (c + 1) * cell_w - margin, (r + 1) * cell_h - margin
        avail_w, avail_h = cx2 - cx1, cy2 - cy1
        if avail_w < 8 or avail_h < 8:
            # 画布小到留白都放不下：退回整格，保证可用区域是正数
            cx1, cy1 = c * cell_w, r * cell_h
            cx2, cy2 = (c + 1) * cell_w, (r + 1) * cell_h
            avail_w, avail_h = max(4.0, cx2 - cx1), max(4.0, cy2 - cy1)

        kind = rng.choice(kinds)
        available = [x for x in colors if x not in used_colors] or colors
        color_name = rng.choice(available)
        used_colors.add(color_name)

        # 先定尺寸并夹到单元格内，再按形状微调 —— 顺序反了会把「正方形」重新拉成长方形
        min_side = max(4.0, min(avail_w, avail_h) * 0.4)
        box_w = max(min_side, min(rng.uniform(0.55, 0.9) * avail_w, avail_w))
        box_h = max(min_side, min(rng.uniform(0.55, 0.9) * avail_h, avail_h))

        if kind == "circle":
            side = min(box_w, box_h)
            box_w = box_h = side
        elif kind == "ellipse":
            # 压扁，避免与圆形混淆
            box_h = max(min_side * 0.6, box_h * 0.6)
        elif kind == "rect" and abs(box_w - box_h) < 30:
            # 避免近似正方形（模型可能答「正方形」，形状判定就会误伤）
            box_w = min(box_w + 50, avail_w)

        x1 = rng.uniform(cx1, max(cx1, cx2 - box_w))
        y1 = rng.uniform(cy1, max(cy1, cy2 - box_h))
        x2, y2 = min(x1 + box_w, float(width)), min(y1 + box_h, float(height))
        x1, y1 = max(0.0, min(x1, x2 - 4)), max(0.0, min(y1, y2 - 4))

        shapes.append(Shape(
            kind=kind,
            color_name=color_name,
            rgb=PALETTE[color_name],
            bbox_px=(round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)),
        ))

    for shape in shapes:
        _draw_shape(painter, shape)

    # 左上角写序号：不影响坐标识别，但方便人眼对照
    painter.text((16, 12), f"#{index + 1}", fill=(150, 160, 175), font=resolve_font(22))

    sample = Sample(name=f"sample_{index + 1:02d}", path="", width=width, height=height, shapes=shapes)
    return sample, img


def make_samples(
    count: int = 5,
    *,
    seed: int = 42,
    out_dir: str | Path = "runs/benchmark/images",
    width: int = 900,
    height: int = 720,
    n_shapes: int = 3,
    clean: bool = True,
) -> list[Sample]:
    """批量生成图片与 ground_truth.json，返回已落盘的 Sample 列表。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if clean:
        for old in out.glob("*.png"):
            old.unlink()
        gt_file = out / "ground_truth.json"
        if gt_file.exists():
            gt_file.unlink()

    samples: list[Sample] = []
    for index in range(count):
        sample, img = make_sample(
            index, seed=seed + index * 1000, width=width, height=height, n_shapes=n_shapes
        )
        path = out / f"{sample.name}.png"
        img.save(path, format="PNG")
        sample.path = str(path)
        samples.append(sample)

    with (out / "ground_truth.json").open("w", encoding="utf-8") as fp:
        json.dump([s.to_dict() for s in samples], fp, ensure_ascii=False, indent=2)
    return samples
