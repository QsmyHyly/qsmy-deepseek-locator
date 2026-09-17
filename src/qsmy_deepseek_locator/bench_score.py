"""判分：把「模型给的框」和「真值框」对上，算出检出率 / 平均 IoU / 标签准确率。

**职责**：match_predictions（按 IoU 贪心匹配）、evaluate_sample（单张图的指标）、
aggregate（多张图汇总，检出率按目标总数加权）。

**边界**：本模块不造图、不调模型、不写文件。输入就是两份纯数据（预测 Detection 列表、
真值 dict 列表），所以自测可以直接喂构造好的数据（见 tests/test_benchmark.py）。

与拆分前旧文件的对应关系（0.1.2 -> 0.1.3 的等价重构，行为零变化）：
    这三个函数原本在 benchmark.py 第 296-408 行，现在原样搬到这里；
    benchmark.py 反过来 import 回来，所以
    「from qsmy_deepseek_locator.benchmark import evaluate_sample」这条老路径照旧可用。

⚠️ 返回字典的键名（detection_rate / mean_iou / per_target ...）就是 report.json 的格式，
也是 README 第 5 节那张对比表的来源。改名等于让历史报告与新旧两轮实验没法比。
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .bench_shapes import color_ok, shape_ok
from .parsing import Detection, box_iou


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
