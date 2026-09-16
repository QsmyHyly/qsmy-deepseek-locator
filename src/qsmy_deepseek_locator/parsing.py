"""坐标解析：模型输出文本 -> Detection 列表，并处理坐标刻度问题。

三段职责，按顺序发生（locate() 内部就是这么调的）：

    1. decode_json_points()   容错解析：代码块、前后说明文字、单引号都能吃下（**原样返回，不换算**）
    2. normalize_to_unit()    刻度兜底：整批判定 0~1000 旧刻度，无损除以 1000
    3. check_coordinate_range() 越界告警：提示模型可能给了像素坐标（不静默画错）

坐标约定：0.0~1.0 的相对比例，x 对宽、y 对高，小数位数不设上限。
格式：

    [{"bbox_2d": [x1, y1, x2, y2], "label": "名称"},
     {"point_2d": [x, y], "label": "名称"}]

关于第 2、3 步为什么都要有：deepseek-flash 在定位任务上偶尔会退回 0~1000 刻度
（视觉大模型圈的通用约定，Qwen2-VL 等都用它），与 0~1 是同一空间的 1000 倍线性缩放，
可以无损换回来；但**像素坐标没法自动救** —— 服务端会先把图缩放再送进模型且不回传缩放后尺寸，
模型报的「像素」落在它每次自己编的画布上（实测同一张图三次调用给出 1000x750 / 1000x800 / 1024x768），
所以只能告警、交给调用方判断。
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

# --------------------------------------------------------------------------- #
# 坐标刻度常量
# --------------------------------------------------------------------------- #
COORD_MIN, COORD_MAX = 0.0, 1.0

# 旧刻度：0~1000（千分比）。整批最大值 > 1.0 且 <= 1000 时判定为它，整体除以 1000。
LEGACY_SCALE = 1000.0

# 措辞刻意**不断言原因**：走到这一步只能说明「整批最大值 > 1.0」，至于它究竟是模型退回了
# 0~1000 刻度、还是被覆盖的 system_prompt 要来的像素坐标，库分辨不了 —— 两者在数值上完全重叠
# （800x600 的图，像素坐标全部 <= 1000，与千分比刻度落在同一区间）。
# 断言成「检测到是旧刻度」会在真实原因是后者时把人引向错误方向：那时候我们其实改错了数据，
# 而告警还在说换算成功。所以这里只陈述做了什么 + 怎么自查。
LEGACY_SCALE_NOTICE = (
    "整批坐标的最大值超过 1.0，已按 0~1000 旧刻度的假设统一除以 1000 换算为 0.0~1.0。"
    "若你的 system_prompt 要求过像素坐标（或这段输出不是本库默认提示词产生的），"
    "这一步会把本来正确的像素值改错 —— 请对照结果里的 image_size 复核。"
)

# 数值字段：bbox_2d 是框，point_2d 是点。
BBOX_FIELD = "bbox_2d"
POINT_FIELD = "point_2d"
_COORD_FIELDS = (BBOX_FIELD, POINT_FIELD)

# 归一化路径下唯一会做舍入的地方：旧刻度换算是 350.25 -> 0.35025，
# 保留 6 位足以无损表达千分之一刻度的全部有效小数。
_LEGACY_ROUND = 6


# --------------------------------------------------------------------------- #
# 第一步：容错解析（刻意不做任何换算）
# --------------------------------------------------------------------------- #
def strip_code_fence(text: str) -> str:
    """去掉 markdown 代码块标记，返回块内内容（没有块就原样返回）。"""
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            block = parts[1]
            head = block.lstrip()
            if head[:4].lower() == "json":
                block = head[4:]
            return block.strip()
    return text.strip()


def extract_json_block(text: str) -> str | None:
    """从夹杂说明文字的文本里扫描出第一个完整的 JSON 数组/对象。

    用括号配对扫描而不是正则：模型返回的 label 里可能含括号，正则会被字符串内的括号带偏。
    同时正确处理字符串内的转义，避免把 "a\\" 里的引号当成结束符。
    """
    start = None
    opener = ""
    closer = ""
    for index, char in enumerate(text):
        if char in "[{":
            start = index
            opener = char
            closer = "]" if char == "[" else "}"
            break
    if start is None:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def decode_json_points(text: str) -> Any:
    """把模型输出文本解析成 Python 对象（列表/字典），失败返回空列表。

    依次尝试：直接 json.loads -> 去掉代码块 -> 扫描出 JSON 片段 -> Python 字面量（容忍单引号）。
    **原样返回，不做刻度换算、不做越界修正** —— 换算在第 2 步统一做，
    保持「解析」与「判定」分离，方便对模型原始输出做诊断。
    """
    if not text or not isinstance(text, str):
        return []

    candidates: list[str] = []
    stripped = text.strip()
    candidates.append(stripped)

    unfenced = strip_code_fence(text)
    if unfenced and unfenced not in candidates:
        candidates.append(unfenced)

    extracted = extract_json_block(unfenced or stripped)
    if extracted and extracted not in candidates:
        candidates.append(extracted)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:  # noqa: BLE001 - 换下一个候选继续试
            continue

    try:
        return ast.literal_eval(extracted or unfenced or stripped)
    except Exception:  # noqa: BLE001
        return []


def to_dict_items(data: Any) -> list[dict]:
    """把解析结果统一成「坐标对象列表」（只保留 dict，其余丢弃）。"""
    if isinstance(data, str):
        data = decode_json_points(data)
    if isinstance(data, dict):
        # 形如 {"items": [...]} 的包装也认一下，省得调用方自己拆
        for key in ("items", "results", "detections", "objects"):
            inner = data.get(key)
            if isinstance(inner, list):
                return [d for d in inner if isinstance(d, dict)]
        return [data]
    if isinstance(data, (list, tuple)):
        return [d for d in data if isinstance(d, dict)]
    return []


# --------------------------------------------------------------------------- #
# 第二步：刻度兜底
# --------------------------------------------------------------------------- #
def iter_coord_values(items: Iterable[Any]) -> Iterator[Any]:
    """遍历所有坐标对象里的数值（非数值原样产出，由调用方决定怎么处理）。"""
    for item in items or []:
        if not isinstance(item, dict):
            continue
        for field_name in _COORD_FIELDS:
            coords = item.get(field_name)
            if isinstance(coords, (list, tuple)):
                for value in coords:
                    yield value


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_to_unit(items: Sequence[dict]) -> tuple[list[dict], bool]:
    """把可能的 0~1000 旧刻度整体换算成 0.0~1.0，返回 (新列表, 是否换算过)。

    判定在**整批**坐标上做，避免一半换算一半没换算的错位：

    - 最大值 <= 1.0                    -> 已是新约定，原样返回
    - 1.0 < 最大值 <= 1000             -> 判定为旧刻度，整体除以 1000
    - 存在 > 1000 的值                 -> **不换算**（很可能是像素坐标），
      交给 check_coordinate_range 告警，由调用方判断

    最后那条的歧义无法自动消除：小图上的像素坐标也会落在 0~1000 内，
    与旧刻度从数值上根本分不开，只能靠提示词约束 + 上层告警。
    """
    items = list(items or [])
    values = [v for v in (_as_float(x) for x in iter_coord_values(items)) if v is not None]
    if not values:
        return items, False
    top = max(values)
    if top <= COORD_MAX or top > LEGACY_SCALE:
        return items, False

    converted: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            converted.append(item)
            continue
        new_item = dict(item)
        for field_name in _COORD_FIELDS:
            coords = item.get(field_name)
            if not isinstance(coords, (list, tuple)):
                continue
            scaled: list[Any] = []
            for raw in coords:
                number = _as_float(raw)
                # 非数值项原样保留：它是模型输出的一部分，丢掉会掩盖问题
                scaled.append(round(number / LEGACY_SCALE, _LEGACY_ROUND)
                              if number is not None else raw)
            new_item[field_name] = scaled
        converted.append(new_item)
    return converted, True


def check_coordinate_range(
    items: Iterable[Any], lo: float = COORD_MIN, hi: float = COORD_MAX
) -> list[dict]:
    """检查坐标是否越界，返回告警列表（空列表 = 全部正常）。

    越界的典型含义是「模型给了像素坐标」。这一步只报告、不修改数据：
    调用方可能想留证据、可能想自己按原图尺寸换算，库不该替他决定。
    """
    warnings: list[dict] = []
    for index, item in enumerate(items or []):
        if not isinstance(item, dict):
            continue
        label = item.get("label") or f"#{index}"
        for field_name in _COORD_FIELDS:
            coords = item.get(field_name)
            if not isinstance(coords, (list, tuple)):
                continue
            out_of_range = []
            for value in coords:
                number = _as_float(value)
                if number is None or number < lo or number > hi:
                    out_of_range.append(value)
            if out_of_range:
                warnings.append({
                    "index": index,
                    "label": label,
                    "field": field_name,
                    "coords": list(coords),
                    "out_of_range": out_of_range,
                })
    return warnings


def format_coordinate_warnings(warnings: Sequence[dict]) -> str:
    """把越界告警压成一句给用户看的话（最多点 3 个例子）。"""
    if not warnings:
        return ""
    parts = [
        f"{w['label']} 的 {w['field']} {w['coords']}（越界值 {w['out_of_range']}）"
        for w in warnings[:3]
    ]
    more = "" if len(warnings) <= 3 else f"，另有 {len(warnings) - 3} 处"
    return (
        "检测到坐标超出 0.0~1.0 相对范围：" + "；".join(parts) + more +
        "。模型可能输出了像素坐标（图片会被服务端缩放，像素值不可靠），"
        "建议检查提示词是否明确要求了 0.0~1.0 的相对比例。"
    )


# --------------------------------------------------------------------------- #
# 数据模型
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Detection:
    """一个定位结果：一个框或一个点，外加它的中文名称。

    坐标一律是 **0.0~1.0 的归一化相对比例**（to_pixels() 才换算成像素）。
    为什么不用像素存：服务端会缩放图片且不回传缩放后尺寸，像素值既不可复现也无法还原；
    归一化比例是这个链路上唯一的不变量。
    """

    label: str = ""
    bbox: tuple[float, float, float, float] | None = None
    point: tuple[float, float] | None = None
    # 模型给该目标的原话（例如 "红色圆形"）；留空表示没有额外说明
    raw_label: str = ""

    @property
    def kind(self) -> str:
        """bbox / point。两者同时存在时算 bbox（框比点信息更多）。"""
        return "bbox" if self.bbox is not None else "point"

    @property
    def center(self) -> tuple[float, float]:
        """归一化中心点：框取几何中心，点取自身。"""
        if self.bbox is not None:
            x1, y1, x2, y2 = self.bbox
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        if self.point is not None:
            return self.point
        return (0.0, 0.0)

    @property
    def area(self) -> float:
        """归一化面积（框），点是 0。用于筛掉「框住整张图」的退化结果。"""
        if self.bbox is None:
            return 0.0
        x1, y1, x2, y2 = self.bbox
        return abs(x2 - x1) * abs(y2 - y1)

    def to_dict(self) -> dict:
        """转回模型的原始字段格式，便于落盘 / 与其它工具对接。"""
        out: dict[str, Any] = {}
        if self.bbox is not None:
            out[BBOX_FIELD] = list(self.bbox)
        if self.point is not None:
            out[POINT_FIELD] = list(self.point)
        out["label"] = self.label
        return out

    @classmethod
    def from_dict(cls, item: dict, *, normalize: bool = False) -> "Detection | None":
        """从模型字段格式构造；两个字段都没有、或数值不合法时返回 None。

        Args:
            normalize: True 时把坐标夹紧到 0.0~1.0（画图用，宁可画在边上也不抛异常）；
                False 时**原样保留**（诊断用，越界值本身就是「模型给了像素坐标」的证据）。
        """
        if not isinstance(item, dict):
            return None

        def _coord(raw: Any, size: int) -> tuple[float, ...] | None:
            if not isinstance(raw, (list, tuple)) or len(raw) != size:
                return None
            out = []
            for value in raw:
                number = _as_float(value)
                if number is None:
                    return None
                if normalize:
                    number = min(COORD_MAX, max(COORD_MIN, number))
                out.append(number)
            return tuple(out)  # type: ignore[return-value]

        bbox = _coord(item.get(BBOX_FIELD), 4)
        point = _coord(item.get(POINT_FIELD), 2)
        if bbox is None and point is None:
            return None
        label = str(item.get("label") or "").strip()
        return cls(
            label=label,
            bbox=bbox,      # type: ignore[arg-type]
            point=point,    # type: ignore[arg-type]
            raw_label=str(item.get("raw_label") or "").strip(),
        )

    def to_pixels(self, width: int, height: int) -> dict:
        """换算成像素坐标（画图 / 对接检测框架时用）。

        Returns:
            {"bbox_px": (x1,y1,x2,y2) | None, "point_px": (x,y) | None,
             "center_px": (x,y)}
        """
        def _scale(value: float, total: int) -> int:
            return int(round(value * total))

        bbox_px = None
        if self.bbox is not None:
            x1, y1, x2, y2 = self.bbox
            bbox_px = (
                _scale(min(x1, x2), width), _scale(min(y1, y2), height),
                _scale(max(x1, x2), width), _scale(max(y1, y2), height),
            )
        point_px = None
        if self.point is not None:
            point_px = (_scale(self.point[0], width), _scale(self.point[1], height))
        cx, cy = self.center
        return {
            "bbox_px": bbox_px,
            "point_px": point_px,
            "center_px": (_scale(cx, width), _scale(cy, height)),
        }

    def iou(self, other: "Detection") -> float:
        """与另一个框的 IoU（任一没有框则 0）。"""
        if self.bbox is None or other.bbox is None:
            return 0.0
        return box_iou(self.bbox, other.bbox)

    def contains_point(self, other: "Detection") -> bool:
        """另一个目标的中心点是否落在本框内（点定位任务的命中判定）。"""
        if self.bbox is None:
            return False
        x1, y1, x2, y2 = self.bbox
        cx, cy = other.center
        return min(x1, x2) <= cx <= max(x1, x2) and min(y1, y2) <= cy <= max(y1, y2)


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """两个 [x1,y1,x2,y2] 框的 IoU（自动兼容端点反序）。"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ax1, ax2 = min(ax1, ax2), max(ax1, ax2)
    ay1, ay2 = min(ay1, ay2), max(ay1, ay2)
    bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
    by1, by2 = min(by1, by2), max(by1, by2)

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------- #
# 顶层入口：文本 -> Detection 列表
# --------------------------------------------------------------------------- #
def parse_detections(text: str) -> tuple[list[Detection], list[str], list[dict]]:
    """把模型输出文本解析成 Detection 列表。

    Returns:
        (detections, warnings, raw_items)
        - detections: 已做过刻度换算；越界值**原样保留**（夹紧是画图时才做的事）
        - warnings:   人话告警（旧刻度换算 / 坐标越界 / 有坐标但字段不合法）
        - raw_items:  换算后的原始字典列表，落盘留证、排查问题都用它

    刻意不抛异常：解析失败 = 模型没给坐标，属于正常结果（图里可能真的没有目标），
    由调用方看 detections 是否为空来决定怎么办。
    """
    warnings: list[str] = []
    items = to_dict_items(decode_json_points(text))
    if not items:
        return [], warnings, []

    items, converted = normalize_to_unit(items)
    if converted:
        warnings.append(LEGACY_SCALE_NOTICE)

    out_of_range = check_coordinate_range(items)
    if out_of_range:
        warnings.append(format_coordinate_warnings(out_of_range))

    detections: list[Detection] = []
    dropped = 0
    for item in items:
        det = Detection.from_dict(item)
        if det is None:
            dropped += 1
            continue
        detections.append(det)
    if dropped:
        warnings.append(
            f"有 {dropped} 个目标的坐标字段不合法（bbox_2d 需 4 个数、point_2d 需 2 个数），已跳过。"
        )
    return detections, warnings, items


__all__ = [
    "Detection",
    "parse_detections",
    "decode_json_points",
    "to_dict_items",
    "normalize_to_unit",
    "check_coordinate_range",
    "format_coordinate_warnings",
    "iter_coord_values",
    "strip_code_fence",
    "extract_json_block",
    "box_iou",
    "COORD_MIN",
    "COORD_MAX",
    "LEGACY_SCALE",
    "LEGACY_SCALE_NOTICE",
    "BBOX_FIELD",
    "POINT_FIELD",
]
