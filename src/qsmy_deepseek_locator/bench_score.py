"""判分：把「模型给的框」和「真值框」对上，算出检出率 / 平均 IoU / 标签准确率。

**职责**：match_predictions（按 IoU 贪心匹配）、evaluate_sample（单张图的指标）、
aggregate（多张图汇总，检出率按目标总数加权）；外加三个为特殊口径服务的件 ——
center_hit（圆形/点目标的命中判定）、rescale_predictions 与 evaluate_any_space
（判断模型到底给的是归一化坐标还是像素坐标）。

**边界**：本模块不造图、不调模型、不写文件。输入就是两份纯数据（预测列表、真值 dict 列表），
所以自测可以直接喂构造好的数据（见 tests/test_benchmark.py）。

**预测的两种表示法都收**（0.1.3 上游化时统一）：本库内部用 Detection（parsing.py 的数据模型），
而下游项目（旧演示项目、安卓 App）直接用解析出来的**原始 dict** 跑评测。两边都不该为对方的
表示法改代码，所以这里统一经 _pred_view() 归一 —— 理由与代价见它的 docstring。

**三处刻意不统一的判定口径**（都踩过坑，别顺手「统一」掉）：
1. **中心点命中**：圆点阵与网页控件用真值里的 match="center"，其余用 IoU 阈值。
   理由见 center_hit()：这两类目标的真值框太小，"框画得多紧"会盖过"有没有找到"。
2. **文本标签**：真值声明 label_mode="text" 时走文本比对（bench_shapes.text_label_ok），
   且形状判定视为通过，否则标签准确率会恒为 0 —— 那是真值与提示词打架，不是模型不行。
3. **不判形状**：真值里 expect_shape=False 时只比颜色。圆点图就是这种：图上只写了颜色名，
   提示词也只问颜色，若还要求形状对，标签准确率同样恒为 0。
   （第 3 条是 0.1.3 上游化时补上的：此前本库的判定不支持它，而库自己造的圆点真值带这个字段 ——
   也就是「库造的图，库自己判不对」。）

与拆分前旧文件的对应关系（0.1.2 -> 0.1.3 的等价重构，行为零变化）：
    这三个函数原本在 benchmark.py 第 296-408 行，现在原样搬到这里；
    benchmark.py 反过来 import 回来，所以
    「from qsmy_deepseek_locator.benchmark import evaluate_sample」这条老路径照旧可用。

⚠️ 返回字典的键名（detection_rate / mean_iou / per_target ...）就是 report.json 的格式，
也是 README 第 5 节那张对比表的来源。改名等于让历史报告与新旧两轮实验没法比。
per_target 里 0.1.3 新增的 label_mode 键是**追加**的（旧消费方不看它也不受影响）。
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .bench_shapes import color_ok, shape_ok, text_label_ok
from .parsing import Detection, box_iou


def _pred_view(pred) -> dict:
    """把一条预测统一看成 dict —— 本库的 Detection 与下游的原始 dict 都收。

    为什么要这一层：本库内部（benchmark.py / 自测）拿的是 Detection 对象，而下游项目
    拿的是 objloc.parsing 解析出来的**原始 dict**（{"bbox_2d": [...], "label": "..."}）。
    评测口径只有一份，输入却有两种 —— 与其让某一方写转换代码（然后两边各自长出一份
    转换逻辑），不如在判分入口一次归一。

    代价说清楚：dict 输入不会被校验（多余键原样留在 _pred_view 的返回里，但判分只读
    bbox_2d / label）。真要让库校验预测的结构，走 parsing.parse_detections 那条路。
    """
    if isinstance(pred, dict):
        return pred
    bbox = getattr(pred, "bbox", None)
    return {
        "bbox_2d": list(bbox) if bbox else None,
        "label": getattr(pred, "label", "") or "",
    }


# --------------------------------------------------------------------------- #
# 几何原语
# --------------------------------------------------------------------------- #
def center_hit(pred_box: Iterable[float], gt_box: Iterable[float],
               max_area_ratio: float = 25.0) -> bool:
    """圆形/点目标专用的命中判定：真值中心落在预测框内，且预测框没有大到离谱。

    为什么不能直接用 IoU 阈值：圆点阵里真值框就是那个圆本身，直径约为
    min(宽,高) 的 9%。在 2400×300 这种极端扁图上，圆点真值框的归一化宽度只有 0.011，
    模型即使把点找准了，只要框画得松一点 IoU 就掉到 0.5 以下——那测的是"框画得多紧"，
    不是"点定位准不准"，而圆点图的用途恰恰是量点位。

    max_area_ratio 用来挡住"框住整张图"的作弊解：预测框面积不得超过真值框的 25 倍。
    """
    px1, py1, px2, py2 = pred_box
    gx1, gy1, gx2, gy2 = gt_box
    cx, cy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
    inside = min(px1, px2) <= cx <= max(px1, px2) and min(py1, py2) <= cy <= max(py1, py2)
    if not inside:
        return False
    pred_area = abs(px2 - px1) * abs(py2 - py1)
    gt_area = abs(gx2 - gx1) * abs(gy2 - gy1)
    if gt_area <= 0:
        return False
    return pred_area <= gt_area * max_area_ratio


# --------------------------------------------------------------------------- #
# 匹配与判分
# --------------------------------------------------------------------------- #
def match_predictions(
    preds: Sequence[Detection], gts: Sequence[dict]
) -> list[tuple[int, int, float]]:
    """按 IoU 贪心匹配预测框与真值框，返回 [(pred_idx, gt_idx, iou)]。

    贪心（从 IoU 最大的一对开始吃）而不是匈牙利算法：目标的真值框互不重叠，
    两种解法结果一致，贪心更好读也更好调。

    preds 可以是 Detection 或原始 dict（见 _pred_view）；返回的 pred_idx 是**传进来那个
    列表**的下标 —— 注意 evaluate_sample 传的是过滤出 bbox 之后的列表，别与原始下标混用。
    """
    pairs: list[tuple[float, int, int]] = []
    for pi, pred in enumerate(preds):
        box = _pred_view(pred)["bbox_2d"]
        if not box:
            continue
        for gi, gt in enumerate(gts):
            value = box_iou(box, gt["bbox_2d"])
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

    命中判定：真值声明 match="center" 时按「真值中心落在预测框内」（见 center_hit），
    其余按 IoU >= 阈值。
    标签判定：真值声明 label_mode="text" 时走文本比对；expect_shape=False 时只比颜色；
    否则要求颜色与形状都对。
    """
    views = [_pred_view(p) for p in preds]
    bbox_preds = [v for v in views if v.get("bbox_2d")]
    matches = match_predictions(bbox_preds, gts)

    per_target: list[dict] = []
    hits = label_hits = color_hits = shape_hits = 0
    for pi, gi, value in matches:
        # 注意用 bbox_preds[pi]：pi 是「过滤后的列表」的下标，与外面 preds 的下标不是一回事
        pred_label = str(bbox_preds[pi].get("label", ""))
        gt = gts[gi]
        text_mode = gt.get("label_mode") == "text"
        if text_mode:
            # 文本标签口径（网页截图等）：目标是界面上的中文名称，没有颜色/形状可言，
            # 直接比文本；s_ok 恒真，避免 color_ok/shape_ok 把标签准确率压成 0。
            c_ok = text_label_ok(gt.get("label", ""), pred_label,
                                 aliases=gt.get("aliases") or [])
            s_ok = True
        else:
            c_ok = color_ok(gt["color_name"], pred_label)
            # 有些真值不要求判形状（例如圆点阵，图里只写了颜色名，提示词也只问颜色），
            # 此时把 shape_ok 视为通过，否则标签准确率会恒为 0——那是真值与提示词打架。
            s_ok = True if gt.get("expect_shape") is False else shape_ok(gt["kind"], pred_label)
        l_ok = c_ok and s_ok
        # 圆形/点目标按"中心点落在框内"判定命中，其余仍用 IoU 阈值，理由见 center_hit()
        hit = (center_hit(bbox_preds[pi]["bbox_2d"], gt["bbox_2d"])
               if gt.get("match") == "center" else value >= iou_threshold)
        hits += int(hit)
        label_hits += int(l_ok)
        color_hits += int(c_ok)
        shape_hits += int(s_ok)
        per_target.append({
            "gt_label": gt["label"],
            "pred_label": pred_label,
            # text 模式下列的 color_ok 实际含义是"文本标签是否对上"，字段名沿用以免破坏既有消费方
            "label_mode": "text" if text_mode else "color_shape",
            "iou": round(value, 3),
            "color_ok": c_ok,
            "shape_ok": s_ok,
            "label_ok": l_ok,
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


def rescale_predictions(preds: Sequence[Detection], width: int, height: int) -> list[dict]:
    """把「像素坐标」的预测换算成 0.0~1.0 的相对比例。

    仅用于诊断：模型若无视约定直接给像素值，这里按原图尺寸换算，
    得到「如果外部帮忙换算」的上限成绩。真实链路不会这样兜底，
    因为服务端缩放后的帧尺寸调用方拿不到（app 里的帧探针脚本给过实测证据）。

    返回的是 dict 列表（evaluate_sample 两种表示法都收，所以不必再转回 Detection）。
    """
    out: list[dict] = []
    for p in preds:
        view = _pred_view(p)
        if not view.get("bbox_2d"):
            continue
        x1, y1, x2, y2 = view["bbox_2d"]
        out.append({
            **view,
            "bbox_2d": [x1 / width, y1 / height, x2 / width, y2 / height],
        })
    return out


def evaluate_any_space(
    preds: Sequence[Detection],
    gts: Sequence[dict],
    width: int,
    height: int,
    *,
    iou_threshold: float = 0.5,
) -> dict:
    """分别按「已归一化」和「像素坐标」两种解释评估，取更优者。

    用于诊断模型是否遵守了 0.0~1.0 相对坐标约定：
    - space == "normalized" 表示模型直接给的就是归一化坐标；
    - space == "pixel"      表示模型给的是像素坐标（约定遵守失败），
      此时 *_pixel 指标代表「如果外部帮忙换算」能达到的准确率上限。
    """
    as_is = evaluate_sample(preds, gts, iou_threshold=iou_threshold)
    as_pixel = evaluate_sample(
        rescale_predictions(preds, width, height), gts, iou_threshold=iou_threshold
    )
    better_pixel = as_pixel["mean_iou"] > as_is["mean_iou"]
    best = as_pixel if better_pixel else as_is
    return {
        "space": "pixel" if better_pixel else "normalized",
        "hits_best": best["hits"],
        "detection_rate_best": best["detection_rate"],
        "mean_iou_best": best["mean_iou"],
        "label_accuracy_best": best["label_accuracy"],
        "report_normalized": as_is,
        "report_pixel": as_pixel,
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
