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

拆分后的职责与边界（0.1.2 -> 0.1.3 的等价重构，行为零变化）：
    本模块只剩**跑一轮 + 报告 + CLI**（run_benchmark / format_report / bench_main）；
    造素材（词表 / Shape / Sample / make_sample / make_samples）搬到了 bench_shapes.py，
    判分（match_predictions / evaluate_sample / aggregate）搬到了 bench_score.py。
    两者都在下面 import 回来，所以
    「from qsmy_deepseek_locator.benchmark import make_sample, evaluate_sample」照旧可用 ——
    benchmark.py 仍是对外那一个门面，README 第 5 节承诺的两个编程接口还在原处。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .errors import LocatorError
from .locate import Locator

# --------------------------------------------------------------------------- #
# 拆分出去、但继续从本模块对外暴露的名字（老 import 路径的兼容层）
#
# 这两个模块是 bench 专用件，不是通用能力，所以名字带 bench_ 前缀：
# 造素材的一批在 bench_shapes.py，判分的一批在 bench_score.py。
# 实现搬走了、名字留在这里，老的 from .benchmark import ... 一行都不用改。
#
# 下面这两条 restore 的是「库内公开件」：它们原本只因被 benchmark.py 用到而顺带可见，
# 但本身是 drawing / parsing 的正式 API（名字不带下划线、在各自模块的 __all__ 里）——
# 切断它们等于凭空制造一次破坏性变更，所以照旧暴露。
# 反过来，原文件里那些**标准库 / PIL 的 import 顺带可见**
# （json / random / dataclass / field / asdict / Iterable / Image / ImageDraw）
# 以及两个下划线私有小工具（_darken / _draw_shape，现在住在 bench_shapes.py）
# 有意**不**保留：它们从来不是本模块的名字，只有 `dir(module)` 会把它们当回事。
# --------------------------------------------------------------------------- #
from .drawing import resolve_font
from .parsing import Detection, box_iou

from .bench_score import aggregate, evaluate_sample, match_predictions
from .bench_shapes import (
    COLOR_SYNONYMS,
    DEFAULT_BENCH_TARGET,
    PALETTE,
    SHAPE_CN,
    Sample,
    Shape,
    color_ok,
    make_sample,
    make_samples,
    shape_ok,
)


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
