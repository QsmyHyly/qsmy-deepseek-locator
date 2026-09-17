"""定位结果对象：一次 locate() 拿回来的那份数据，以及「结果为什么是这样」的措辞。

    from qsmy_deepseek_locator import locate
    result = locate("photo.png", "红色圆形")
    result.summary(); result.to_json(); result.find("红"); result.describe()

**职责**：定义 LocateResult，以及「空结果该怎么说」的两条提示语。
**边界**：本模块**不调模型、不读图、不绘制**，也不认识 Locator —— 它只描述「一次调用的产物长什么样」，
以及序列化 / 便捷视图 / 人类可读摘要。谁产生这个对象、怎么产生，是 locate.py 的事。

与拆分前旧文件的对应关系（0.1.2 -> 0.1.3 的等价重构，行为零变化）：
    旧 locate.py 第 54-58 行的两条提示语常量、第 61-76 行的 _returned_empty_array、
    以及第 79-216 行的 LocateResult 原样搬到这里；
    locate.py 反过来从本模块 import 它们，旧的 import 路径（from .locate import LocateResult）
    继续可用，所以对外公开面没有动。

⚠️ 本模块里的**字段名与 to_dict() 的键名是序列化格式的一部分**（result.save() 写出的 JSON、
CLI 的 --print-json），改名等于破坏兼容，别动。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .parsing import Detection, extract_json_block

# 「没找到目标」的两种情形要分开说，否则用户会以为解析坏了：
#   模型给了空数组  -> 它的答案就是「图里没有这个目标」，是正常结果，不需要告警级措辞；
#   正文里没有坐标  -> 多半是没按输出格式来（或回答跑题），这才值得提醒。
_NO_COORD_NOTICE = "正文里没有可解析的坐标。可检查提示词是否明确要求只输出 JSON 数组。"
_EMPTY_ARRAY_NOTICE = "模型返回了空数组：它认为图中没有符合条件的目标。"


def _returned_empty_array(text: str) -> bool:
    """正文里是否**真的**写了一个解析得通的空数组。

    不能用「decode_json_points(text) == []」判断：解析失败时它也返回 []，
    于是「模型答了一段散文、一个坐标都没给」会被误报成「模型说没找到目标」，
    而这两种情况的处理方式完全不同（前者要改提示词，后者是正常结果）。
    所以这里先取出 JSON 片段，再真解析一次，只认解析得通的空列表。
    """
    block = extract_json_block(text)
    if not block:
        return False
    try:
        data = json.loads(block)
    except Exception:  # noqa: BLE001 - 解析不了就不属于「空数组」这个情形
        return False
    return isinstance(data, list) and not data


@dataclass
class LocateResult:
    """一次定位的完整结果。

    detections 是唯一的主数据；其余字段都是**证据**（正文 / 思考 / usage / 告警 / 原始项），
    排查「为什么没找到」时全靠它们。
    """

    detections: list[Detection] = field(default_factory=list)
    text: str = ""
    reasoning: str = ""
    model: str = ""
    prompt: str = ""
    system_prompt: str = ""
    duration_ms: float = 0.0
    usage: dict | None = None
    finish_reason: str | None = None
    warnings: list[str] = field(default_factory=list)
    raw_items: list[dict] = field(default_factory=list)
    image: str = ""
    image_size: tuple[int, int] | None = None
    # 标注图的落盘路径。只有 locate_to_file 会填它，别的入口恒为 None ——
    # 有了这个字段，调用方就不必自己记「我刚才是存到哪了」。
    annotated_path: str | None = None

    # ---- 便捷视图 ---- #
    def __len__(self) -> int:
        return len(self.detections)

    def __iter__(self) -> Iterator[Detection]:
        return iter(self.detections)

    def __getitem__(self, index: int) -> Detection:
        return self.detections[index]

    @property
    def bboxes(self) -> list[tuple[float, float, float, float]]:
        return [d.bbox for d in self.detections if d.bbox is not None]

    @property
    def points(self) -> list[tuple[float, float]]:
        return [d.point for d in self.detections if d.point is not None]

    @property
    def labels(self) -> list[str]:
        return [d.label for d in self.detections]

    @property
    def centers(self) -> list[tuple[float, float]]:
        return [d.center for d in self.detections]

    @property
    def empty(self) -> bool:
        """模型没有给出任何目标（注意：这不等于调用失败）。"""
        return not self.detections

    def find(self, keyword: str) -> list[Detection]:
        """按标签子串筛（例：find("红")）。大小写不敏感，便于粗筛。"""
        key = (keyword or "").strip().lower()
        if not key:
            return list(self.detections)
        return [d for d in self.detections if key in d.label.lower()]

    def summary(self) -> dict:
        """一行式统计，命令行/日志用。"""
        return {
            "total": len(self.detections),
            "bbox_count": len(self.bboxes),
            "point_count": len(self.points),
            "labels": self.labels,
        }

    def to_dict(self, *, include_text: bool = True, include_raw: bool = False) -> dict:
        """转成可 JSON 序列化的字典。

        Args:
            include_text: 是否带上模型正文与思考（默认带；日志里嫌长可以关掉）。
            include_raw: 是否带上解析前的原始坐标项（排查刻度问题时才需要）。
        """
        out: dict[str, Any] = {
            "image": self.image,
            "image_size": list(self.image_size) if self.image_size else None,
            "annotated_path": self.annotated_path,
            "model": self.model,
            "prompt": self.prompt,
            "duration_ms": round(self.duration_ms, 1),
            "finish_reason": self.finish_reason,
            "counts": {
                "total": len(self.detections),
                "bbox_count": len(self.bboxes),
                "point_count": len(self.points),
            },
            "detections": [d.to_dict() for d in self.detections],
            "warnings": list(self.warnings),
            "usage": self.usage,
        }
        if include_text:
            out["text"] = self.text
            out["reasoning"] = self.reasoning
        if include_raw:
            out["raw_items"] = list(self.raw_items)
        return out

    def to_json(self, *, indent: int | None = 2, **kwargs: Any) -> str:
        """序列化成 JSON 文本（ensure_ascii=False，中文标签不会被转义成 \\uXXXX）。"""
        return json.dumps(self.to_dict(**kwargs), ensure_ascii=False, indent=indent)

    def save(self, path: str | Path, *, indent: int | None = 2, **kwargs: Any) -> Path:
        """把 JSON 写到文件，返回写入路径。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(indent=indent, **kwargs), encoding="utf-8")
        return target

    def describe(self) -> str:
        """多行的人类可读摘要（CLI 的默认输出就用它）。"""
        lines = [
            f"模型：{self.model}    耗时：{self.duration_ms / 1000:.1f}s    "
            f"目标：{len(self.detections)} 个    finish_reason：{self.finish_reason}"
        ]
        if self.image_size:
            lines.append(f"图片：{self.image}（{self.image_size[0]}x{self.image_size[1]}）")
        else:
            lines.append(f"图片：{self.image}")
        if self.annotated_path:
            lines.append(f"标注图：{self.annotated_path}")
        for index, det in enumerate(self.detections, 1):
            if det.bbox is not None:
                box = " ".join(f"{v:.4f}" for v in det.bbox)
                lines.append(f"  {index:>2}. [框] {det.label}  bbox_2d=[{box}]")
            else:
                point = det.point or (0.0, 0.0)
                lines.append(f"  {index:>2}. [点] {det.label}  point_2d=[{point[0]:.4f}, {point[1]:.4f}]")
        if not self.detections:
            lines.append("  （没有定位到任何目标）")
        for warning in self.warnings:
            lines.append(f"  ⚠️ {warning}")
        return "\n".join(lines)


__all__ = ["LocateResult"]
