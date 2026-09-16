"""准确率评测：程序造图 + 程序判分，不靠人眼。

为什么一个库要自带评测：本库的准确率**几乎完全由提示词口径决定**。
参考项目的 A/B 实测最能说明问题 —— 同一批 15 个目标，只把提示词从
「坐标用归一化值、无需考虑分辨率」改成显式换算公式 + 禁止像素值：

    检出率 40% (6/15) -> 100% (15/15)，平均 IoU 0.655 -> 0.897

也就是「改提示词」这个动作有 60 个百分点的杀伤力。没有自动判分的评测，
改完只能靠肉眼瞄一眼，等于闭着眼睛改。所以 bench 是本库的一等公民，不是附属脚本。

评测怎么做的：

    1. 用代码生成几何图形图（随机颜色 x 形状 x 位置），真值顺手算出来（0.0~1.0 比例）；
    2. 真图送给模型定位；
    3. 按 IoU（默认阈值 0.5）贪心匹配预测框与真值框；
    4. 指标：检出率 = 命中/真值；平均 IoU 只在匹配对上统计；标签准确率 = 颜色对且形状对。

已知边界（别把模型的锅算到评测头上）：极端长宽比（如 2400x300）的密集小目标上，
模型「颜色全认对、位置基本全错」——原因是服务端会把图缩到长边约 1000 再送模型，
小圆点缩完只剩 5~6 像素。那是模型能力的边界，不是本模块的 bug。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw

from .drawing import resolve_font
from .errors import LocatorError
from .locate import Locator
from .parsing import Detection, box_iou

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


# --------------------------------------------------------------------------- #
# 判分
# --------------------------------------------------------------------------- #
def match_predictions(
    preds: Sequence[Detection], gts: Sequence[dict]
) -> list[tuple[int, int, float]]:
    """按 IoU 贪心匹配预测框与真值框，返回 [(pred_idx, gt_idx, iou)]。

    贪心（从 IoU 最大的一对开始吃）而不是匈牙利算法：目标的真值框互不重叠，
    两种解法结果一致，贪心更好读也更好调。
    """
    pairs: list[tuple[float, int, int]] = []
    for pi, pred in enumerate(preds):
        if pred.bbox is None:
            continue
        for gi, gt in enumerate(gts):
            value = box_iou(pred.bbox, gt["bbox_2d"])
            if value > 0:
                pairs.append((value, pi, gi))
    pairs.sort(reverse=True)

    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for value, pi, gi in pairs:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        matches.append((pi, gi, value))
    return matches


def evaluate_sample(
    preds: Sequence[Detection], gts: Sequence[dict], *, iou_threshold: float = 0.5
) -> dict:
    """评估单张图的预测结果。

    命中判定用 IoU >= 阈值；标签判定要求**颜色对且形状对**（两样都写进标签了）。
    """
    bbox_preds = [p for p in preds if p.bbox is not None]
    matches = match_predictions(preds, gts)

    per_target: list[dict] = []
    hits = label_hits = color_hits = shape_hits = 0
    for pi, gi, value in matches:
        # 注意用 preds[pi] 而不是 bbox_preds[pi]：pi 是 preds 的下标，两者不是一回事
        pred_label = preds[pi].label
        gt = gts[gi]
        c_ok = color_ok(gt["color_name"], pred_label)
        s_ok = shape_ok(gt["kind"], pred_label)
        hit = value >= iou_threshold
        hits += int(hit)
        label_hits += int(c_ok and s_ok)
        color_hits += int(c_ok)
        shape_hits += int(s_ok)
        per_target.append({
            "gt_label": gt["label"],
            "pred_label": pred_label,
            "iou": round(value, 3),
            "color_ok": c_ok,
            "shape_ok": s_ok,
            "label_ok": c_ok and s_ok,
            "hit": hit,
        })

    n_gt, n_pred, n_match = len(gts), len(preds), len(matches)
    return {
        "gt_count": n_gt,
        "pred_count": n_pred,
        "pred_bbox_count": len(bbox_preds),
        "pred_point_count": n_pred - len(bbox_preds),
        "matched": n_match,
        "hits": hits,
        "detection_rate": round(hits / n_gt, 3) if n_gt else 0.0,
        "precision": round(hits / n_pred, 3) if n_pred else 0.0,
        "mean_iou": round(sum(v for _, _, v in matches) / n_match, 3) if n_match else 0.0,
        "label_accuracy": round(label_hits / n_match, 3) if n_match else 0.0,
        "color_accuracy": round(color_hits / n_match, 3) if n_match else 0.0,
        "shape_accuracy": round(shape_hits / n_match, 3) if n_match else 0.0,
        "per_target": per_target,
    }


def aggregate(reports: Iterable[dict]) -> dict:
    """汇总多张图的评估结果（检出率按目标总数加权，平均 IoU 在匹配对上取均值）。"""
    reports = list(reports)
    total_gt = sum(r["gt_count"] for r in reports)
    total_pred = sum(r["pred_count"] for r in reports)
    total_hits = sum(r["hits"] for r in reports)
    total_matched = sum(r["matched"] for r in reports)
    pairs = [t for r in reports for t in r["per_target"]]

    def _acc(key: str) -> float:
        if not pairs:
            return 0.0
        return round(sum(int(t[key]) for t in pairs) / len(pairs), 3)

    return {
        "images": len(reports),
        "total_gt": total_gt,
        "total_pred": total_pred,
        "total_hits": total_hits,
        "total_matched": total_matched,
        "detection_rate": round(total_hits / total_gt, 3) if total_gt else 0.0,
        "precision": round(total_hits / total_pred, 3) if total_pred else 0.0,
        "mean_iou": round(sum(t["iou"] for t in pairs) / len(pairs), 3) if pairs else 0.0,
        "label_accuracy": _acc("label_ok"),
        "color_accuracy": _acc("color_ok"),
        "shape_accuracy": _acc("shape_ok"),
        # 「有几张图出现了坐标口径告警」——坐标约定被破坏时这个数会先跳起来
        "images_with_warnings": sum(1 for r in reports if r.get("warnings")),
    }


# --------------------------------------------------------------------------- #
# 跑评测
# --------------------------------------------------------------------------- #
def run_benchmark(
    *,
    count: int = 5,
    n_shapes: int = 3,
    seed: int = 42,
    out_dir: str | Path = "runs/benchmark",
    width: int = 900,
    height: int = 720,
    annotate: bool = False,
    images_only: bool = False,
    target: str | None = None,
    locator: Locator | None = None,
    on_image: Any = None,
) -> dict:
    """跑一轮完整评测，返回报告字典（同时把 report.json 写进 out_dir）。

    Args:
        annotate: 是否把预测结果画成图存到 out_dir/annotated/（出问题时最好查的东西）。
        images_only: 只造图不调模型（不花钱，先看素材）。
        locator: 注入自定义 Locator（自测用假客户端走这条路）。
        on_image: 每张图处理完的回调 on_image(index, sample, result)，CLI 用它打进度。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples = make_samples(
        count, seed=seed, out_dir=out / "images", width=width, height=height, n_shapes=n_shapes
    )

    report: dict[str, Any] = {
        "config": {
            "count": count,
            "n_shapes": n_shapes,
            "seed": seed,
            "width": width,
            "height": height,
            "target": target or DEFAULT_BENCH_TARGET,
            "images_only": images_only,
        },
        "samples": [s.to_dict() for s in samples],
    }
    if images_only:
        report["report"] = None
        _dump(out / "report.json", report)
        return report

    active = locator or Locator()
    per_image: list[dict] = []
    annotated_dir = out / "annotated"
    if annotate:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    for index, sample in enumerate(samples):
        result = active.locate(sample.path, target or DEFAULT_BENCH_TARGET)
        gts = sample.ground_truth()
        evaluated = evaluate_sample(result.detections, gts)
        evaluated.update({
            "name": sample.name,
            "path": sample.path,
            "width": sample.width,
            "height": sample.height,
            "duration_ms": round(result.duration_ms, 1),
            "warnings": list(result.warnings),
        })
        per_image.append(evaluated)

        if annotate:
            from .drawing import save_annotated

            save_annotated(
                sample.path, result.detections, path=annotated_dir / f"{sample.name}_pred.png"
            )
        if on_image is not None:
            on_image(index, sample, result)

    summary = aggregate(per_image)
    summary["mean_duration_ms"] = round(
        sum(r["duration_ms"] for r in per_image) / len(per_image), 1
    ) if per_image else 0.0
    report["report"] = summary
    report["per_image"] = per_image
    _dump(out / "report.json", report)
    return report


def _dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def format_report(summary: dict) -> str:
    """把汇总压成几行人话（CLI 输出）。"""
    if not summary:
        return "（只生成了图片，没有跑模型）"
    return (
        f"图片 {summary['images']} 张 | 真值 {summary['total_gt']} 个 | 预测 {summary['total_pred']} 个\n"
        f"检出率 {summary['detection_rate']:.1%}（{summary['total_hits']}/{summary['total_gt']}）   "
        f"精确率 {summary['precision']:.1%}   平均 IoU {summary['mean_iou']:.3f}\n"
        f"标签准确率 {summary['label_accuracy']:.1%}"
        f"（颜色 {summary['color_accuracy']:.1%} / 形状 {summary['shape_accuracy']:.1%}）\n"
        f"平均耗时 {summary.get('mean_duration_ms', 0) / 1000:.1f}s/张   "
        f"带告警的图片 {summary['images_with_warnings']} 张"
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def bench_main(argv: Sequence[str] | None = None) -> int:
    """评测子命令入口（由 cli.main 转发过来）。"""
    from .cli import _force_utf8

    _force_utf8()

    parser = argparse.ArgumentParser(
        prog="qsmy-deepseek-locator bench",
        description="合成几何图形图的定位准确率评测（真值与判分全自动，不靠人眼）。",
    )
    parser.add_argument("--count", type=int, default=5, help="图片张数（默认 5）")
    parser.add_argument("--n-shapes", type=int, default=3, help="每张图的图形数（默认 3）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（换一批图就换它）")
    parser.add_argument("--width", type=int, default=900, help="画布宽（默认 900）")
    parser.add_argument("--height", type=int, default=720, help="画布高（默认 720）")
    parser.add_argument("--out", default="runs/benchmark", help="产物目录（默认 runs/benchmark）")
    parser.add_argument("--images-only", action="store_true", help="只造图不调模型（不花钱）")
    parser.add_argument("--annotate", action="store_true", help="把预测结果画成图存到 annotated/")
    parser.add_argument("--target", default=None, help=f"提问用词（默认 {DEFAULT_BENCH_TARGET}）")
    parser.add_argument("--system-prompt", default=None, help="覆盖系统提示词（重点测口径时用）")
    parser.add_argument("--model", default=None, help="模型名")
    parser.add_argument("--no-thinking", action="store_true", help="关闭思考模式（更快更省）")
    parser.add_argument("--effort", default=None, help="思考强度 low/medium/high/xhigh/max")
    parser.add_argument("--max-tokens", type=int, default=None, help="输出上限")
    parser.add_argument("--max-side", type=int, default=None, help="发送前缩图的最长边上限")
    parser.add_argument("-q", "--quiet", action="store_true", help="不打印每张图的进度")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    locator = Locator(
        model=args.model,
        system_prompt=args.system_prompt,
        thinking=False if args.no_thinking else None,
        reasoning_effort=args.effort,
        max_tokens=args.max_tokens,
        max_side=args.max_side,
    )

    def progress(index: int, sample: Any, result: Any) -> None:
        if args.quiet:
            return
        sys.stderr.write(
            f"  [{index + 1}] {sample.name}: 预测 {len(result.detections)} 个，"
            f"{result.duration_ms / 1000:.1f}s\n"
        )

    try:
        report = run_benchmark(
            count=args.count,
            n_shapes=args.n_shapes,
            seed=args.seed,
            out_dir=args.out,
            width=args.width,
            height=args.height,
            annotate=args.annotate,
            images_only=args.images_only,
            target=args.target,
            locator=locator,
            on_image=None if args.images_only else progress,
        )
    except LocatorError as exc:
        sys.stderr.write(f"\n错误：{exc}\n")
        return 1

    out = Path(args.out)
    sys.stdout.write(f"图片与真值：{out / 'images'}\n")
    sys.stdout.write(format_report(report.get("report") or {}) + "\n")
    sys.stdout.write(f"报告：{out / 'report.json'}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(bench_main())


__all__ = [
    "PALETTE", "COLOR_SYNONYMS", "SHAPE_CN", "Shape", "Sample",
    "make_sample", "make_samples", "evaluate_sample", "aggregate",
    "match_predictions", "run_benchmark", "format_report", "bench_main",
    "shape_ok", "color_ok", "DEFAULT_BENCH_TARGET",
]
