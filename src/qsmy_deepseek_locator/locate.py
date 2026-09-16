"""核心入口：一张图 + 一句话 -> 结构化定位结果。

    from qsmy_deepseek_locator import locate, locate_to_file, draw

    result = locate("photo.png", "红色圆形")
    for d in result:
        print(d.label, d.bbox, d.center)

    draw("photo.png", result).save("annotated.png")

只想拿「画好框的图片文件」的话，用本模块最下面那个一行式入口：

    result = locate_to_file("photo.png", "红色圆形", "runs/photo_annotated.png")
    print(result.annotated_path)

一次 locate() 内部发生的事（顺序固定，每一步都能单独拿出来用）：

    1. 图片 -> data URL（images.to_data_url；默认零重编码）
    2. 拼报文（client.build_messages：system 承载坐标口径，user 承载「找什么」）
    3. 调模型（client.DeepSeekVisionClient，恒走流式，可给 on_event 看进度）
    4. 解析正文（parsing.parse_detections：容错 + 旧刻度兜底 + 越界告警）
    5. 打包成 LocateResult（坐标 / 标签 / 耗时 / usage / 告警 / 原始项）

本模块**不做**的三件事，都是有意的：
    - 不静默降级。没有 API Key 就抛 MissingAPIKeyError，不返回假的坐标。
    - 不自动重试「模型没找到目标」这种结果。空结果与调用失败是两回事，前者是正常答案。
    - 不把越界坐标偷偷改小。越界往往意味着模型给了像素坐标，那是一条**关于提示词的情报**，
      悄悄夹紧等于把情报抹掉；夹紧只发生在绘制那一刻（drawing.draw）。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from .client import (
    ChatReply,
    DeepSeekVisionClient,
    EventCallback,
    VisionClient,
    build_messages,
)
from .config import Settings
from .drawing import resolve_output_path
from .errors import LocatorError
from .images import describe_source, source_size, to_data_url
from .parsing import Detection, extract_json_block, parse_detections
from .prompts import DEFAULT_SYSTEM_PROMPT, build_user_prompt

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


class Locator:
    """可复用的定位器：把配置与客户端拿在手里，反复定位多张图。

    典型用法：

        locator = Locator(thinking=False)          # 关掉思考，更快更省 token
        for path in paths:
            result = locator.locate(path, "按钮")
            print(path, result.summary())
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: VisionClient | None = None,
        max_side: int | None = None,
        **overrides: Any,
    ):
        """
        Args:
            settings: 直接给一份完整配置（给了就以它为基础）。
            client: 注入自定义客户端（自测打桩 / 换成别的 OpenAI 兼容服务）。
            max_side: 发送前把图缩到最长边不超过它（省流量；不影响归一化坐标精度）。
            **overrides: 其余键按 Settings 字段名覆盖（api_key / base_url / model / thinking /
                reasoning_effort / image_detail / max_tokens / timeout / system_prompt ...）。
                ⚠️ 值为 None 的键**不覆盖**，所以 thinking=None 表示「沿用」，要关思考请传 False。
        """
        base = settings or Settings.from_env()
        self.settings = base.merged(**overrides)
        self.client: VisionClient = client or DeepSeekVisionClient(self.settings)
        self.max_side = max_side

    # ------------------------------------------------------------------ #
    def locate(
        self,
        image: Any,
        target: str | None = None,
        *,
        prompt: str | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
        image_detail: str | None = None,
        max_tokens: int | None = None,
        max_side: int | None = None,
        client: VisionClient | None = None,
        on_event: EventCallback | None = None,
    ) -> LocateResult:
        """定位一张图里的目标。

        Args:
            image: 图片源：本地路径 / http(s) URL / bytes / PIL.Image / data URL。
            target: 找什么，例如 "红色圆形"、"登录按钮"。为空则用默认话术（识别主要物体）。
            prompt: 直接给完整的 user 消息（给了就忽略 target）。坐标口径不在这里，在 system 消息。
            system_prompt: 覆盖系统提示词（它承载坐标口径，改之前先读 prompts.py 的说明）。
            thinking: 思考模式开关。None = 沿用配置；False = 关掉（更快、更省 token）。
            reasoning_effort: 思考强度 low/medium/high/xhigh/max（仅思考开启时生效）。
            image_detail: 图片精度 low/high/original/auto（默认不带该字段）。
            max_tokens: 输出上限（含思考 token，留空用服务端默认）。
            max_side: 本次发送的缩放上限（覆盖构造参数）。
            client: 本次调用换用别的客户端。
            on_event: 流式事件回调，形如 on_event({"type": "reasoning"|"content"|...})。

        Returns:
            LocateResult。**模型没找到目标时不会抛异常**，而是返回 detections 为空的结果 ——
            与「调用失败」区分开（后者抛 APIError / EmptyResponseError / MissingAPIKeyError）。
        """
        effective = self.settings.merged(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            image_detail=image_detail,
            max_tokens=max_tokens,
            system_prompt=system_prompt or None,
        )
        active_client = client or self.client
        limit = self.max_side if max_side is None else max_side

        user_prompt = prompt if prompt is not None else build_user_prompt(target)
        data_url = to_data_url(image, max_side=limit)
        messages = build_messages(
            user_prompt,
            image_url=data_url,
            system_prompt=effective.system_prompt,
            image_detail=effective.image_detail,
        )

        started = time.perf_counter()
        reply = active_client.complete(messages, settings=effective, on_event=on_event)
        duration_ms = (time.perf_counter() - started) * 1000.0

        detections, warnings, raw_items = parse_detections(reply.text)
        if not detections and reply.text.strip() and not warnings:
            warnings.append(
                _EMPTY_ARRAY_NOTICE if _returned_empty_array(reply.text) else _NO_COORD_NOTICE
            )

        return LocateResult(
            detections=detections,
            text=reply.text,
            reasoning=reply.reasoning,
            model=reply.model or effective.model,
            prompt=user_prompt,
            system_prompt=effective.system_prompt,
            duration_ms=duration_ms,
            usage=reply.usage,
            finish_reason=reply.finish_reason,
            warnings=warnings,
            raw_items=raw_items,
            image=describe_source(image),
            image_size=source_size(image),
        )

    # ------------------------------------------------------------------ #
    def locate_and_draw(
        self,
        image: Any,
        target: str | None = None,
        *,
        output: str | Path | None = None,
        **kwargs: Any,
    ):
        """定位并直接把结果画回图上，返回 (LocateResult, PIL.Image)。

        output 给了就顺手存成 PNG（路径原样返回在 result 里由调用方自己记）。
        这是个便利方法：只要坐标不要图的场景请直接用 locate()；
        只要「画好的图片文件」的场景请用 locate_to_file()（它会校验路径、按后缀定格式，
        并把落盘路径写进 result.annotated_path）。
        """
        from .drawing import draw

        result = self.locate(image, target, **kwargs)
        annotated = draw(image, result.detections)
        if output:
            target_path = Path(output)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            annotated.save(target_path, format="PNG")
        return result, annotated


    # ------------------------------------------------------------------ #
    def locate_to_file(
        self,
        image: Any,
        target: str | None = None,
        output: str | Path | None = None,
        *,
        prompt: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        thinking: bool | None = False,
        reasoning_effort: str | None = None,
        image_detail: str | None = "original",
        use_tools: bool = False,
        max_tokens: int | None = None,
        timeout: float | None = None,
        system_prompt: str | None = None,
        max_side: int | None = None,
        on_event: EventCallback | None = None,
        box_width: int = 3,
        point_radius: int = 5,
        font_size: int = 22,
        draw_label: bool = True,
        colors: Sequence[str] | None = None,
    ) -> LocateResult:
        """定位 + 打标 + 落盘：一次调用把「已经画好框的图片文件」交到你手里。

            from qsmy_deepseek_locator import locate_to_file

            result = locate_to_file("photo.png", "红色圆形", "runs/photo_annotated.png")
            print(result.annotated_path, len(result), result.labels)

        只有前三个参数是位置参数（图片 / 找什么 / 输出路径），其余全是可选关键字参数，
        分三组：模型参数（api_key / thinking / image_detail ...）、绘制参数（colors / ...），
        以及 Locator.locate() 已有的那批。

        Args:
            image: 图片源：本地路径 / http(s) URL / bytes / PIL.Image / data URL。
            target: 要找什么，就是你那句「用户提示词」，例如 "红色圆形"、"登录按钮"。
                本库一贯的命名：**目标描述叫 target**；prompt 是另一回事（见下）。
            output: 标注图的**完整路径（含文件名）**。没写扩展名就补 .png；
                扩展名不在支持列表里抛 ValueError。路径会在调模型**之前**校验 ——
                路径写错不该等到花掉一次 API 调用才发现。父目录不存在会自动创建。
            prompt: 直接给整段 user 消息（给了就忽略 target）。坐标口径在 system 消息里，不在这里。
                ⚠️ 传它就用你这一整段，**本库那句「请找出图中所有的…，逐个输出中文名称与坐标」的
                包装句式会被跳过**（那句是测出来的，实测对召回有影响）。所以除非你确实想自己写
                整段话术，否则请把「找什么」交给 target 或第二个位置参数。
            api_key: 传了就**只用它**，不再看 DEEPSEEK_API_KEY；不传才去读环境变量。
                两处都没有则抛 MissingAPIKeyError（本库不提供「无 Key 返回假坐标」的降级）。
            base_url / model: 接口地址 / 模型名（默认 https://api.deepseek.com 与 deepseek-flash）。
            thinking: 思考模式。默认 False = **显式关闭**（坐标类任务实测不掉准确率、耗时约省一半）；
                传 None 表示「交回配置/环境变量决定」，传 True 才开。
            reasoning_effort: 思考强度 low/medium/high/xhigh/max，仅在思考开启时生效。
            image_detail: 图片精度，默认 "original"。传 None 表示**不发送该字段**。
                注意它并不是「提高定位精度」的开关：官方每张图最多只算 384 token，
                再大的图到服务端照样被缩到约 800x800。
                @doc docs/API-NOTES.md#3-图片-token-与尺寸
                （该文档解决"detail 到底改变了什么、为什么堆分辨率没用"的问题。）
            use_tools: 是否开启工具（Agent）调用。v0.1 **没有实现工具循环**，只能保持 False；
                传 True 会当场抛 NotImplementedError。这是预留参数：宁可报错，
                也不静默忽略 —— 静默忽略会让你以为工具已经开了。
                想自己接工具请走底层：DeepSeekVisionClient.complete(messages, tools=[...])，
                调用结果落在 ChatReply.tool_calls。本库负责透传报文与拼回分片，**不执行工具**。
                @doc docs/API-NOTES.md#61-工具调用也是流式的而且一个字符一个-chunk
                （该文档解决"工具调用的参数为什么要按 index 自己拼、on_event 能拿到什么"的问题。）
            max_tokens: 输出上限（含思考 token）。正文为空时优先调大它。
            timeout: 单次请求超时（秒）。
            max_side: 发送前把图缩到最长边不超过它（归一化坐标不受影响，只省上行流量）。
            on_event: 流式事件回调，可拿来做进度显示。
            box_width / point_radius / font_size / draw_label / colors: 绘制样式，见 drawing.draw。

        Returns:
            LocateResult。「annotated_path」是刚写出的文件路径（相对路径按当前工作目录解析）；
            「detections」为空表示模型没找到目标 —— 那不是失败，图照样落盘（内容等于原图）。

        Note:
            打标画的是**原图**（不是发给模型的那份可能被缩小的副本），
            所以输出图的分辨率始终等于输入图。URL 输入会被下载两次（发模型一次、绘制一次），
            与 locate() 内部 source_size 的行为一致。
        """
        if use_tools:
            raise NotImplementedError(
                "use_tools=True 目前不可用：v0.1 没有实现工具（Agent）调用循环。\n"
                "这个参数是预留给调用点的，传 True 会当场报错而不是被静默忽略 —— "
                "静默忽略会让你以为工具已经开了，那比报错危险得多。\n"
                "需要多轮工具调用请参考 deepseek-vision-annotation 里的 Agent 实现，或等本库后续版本。\n"
                "只想拿到模型的工具调用请求（不执行）可以用底层：\n"
                "    client = DeepSeekVisionClient()\n"
                "    reply = client.complete(messages, tools=[...])\n"
                "    reply.tool_calls  # 已按 index 拼好，arguments 是完整 JSON 串\n"
                "事件流示例见 examples/stream_events.py --tools。"
            )

        # 先校验输出路径再调模型：这是**故意**的顺序，一次 API 调用不该因为路径拼错而白花。
        path, fmt = resolve_output_path(output)

        result = self.locate(
            image,
            target,
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            image_detail=image_detail,
            max_tokens=max_tokens,
            max_side=max_side,
            on_event=on_event,
        )

        from .drawing import draw

        annotated = draw(
            image,
            result.detections,
            box_width=box_width,
            point_radius=point_radius,
            font_size=font_size,
            draw_label=draw_label,
            colors=colors,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(path, format=fmt)
        result.annotated_path = str(path)
        return result


# 模块级便利函数（locate / locate_to_file）里属于「Locator 构造参数」的键名。
# 两个入口共用这一份集合：以后给 Locator 加构造参数时，不会漏掉其中一个入口。
_LOCATOR_KWARGS = frozenset({"settings", "client", "max_side", "max_retries"})


def _split_kwargs(kwargs: dict) -> tuple[dict, dict]:
    """把「一次调用的全部关键字参数」拆成 (Locator 构造参数, 单次调用参数)。"""
    locator_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in _LOCATOR_KWARGS}
    return locator_kwargs, kwargs


def locate(image: Any, target: str | None = None, **kwargs: Any) -> LocateResult:
    """一次性入口：等价于 Locator(**locator_kwargs).locate(image, target, ...)。

    分开两类参数：Locator 的构造参数（settings / client / max_side / max_retries）
    与 locate 的调用参数。函数内部用一次调用把两者都消化掉，方便脚本里一行搞定。

    只想反复调用时请自己建 Locator —— 每次 locate() 都会重新读环境变量并新建客户端。
    """
    locator_kwargs, call_kwargs = _split_kwargs(kwargs)
    locator = Locator(**locator_kwargs)
    return locator.locate(image, target, **call_kwargs)


def locate_to_file(
    image: Any,
    target: str | None = None,
    output: str | Path | None = None,
    **kwargs: Any,
) -> LocateResult:
    """一行式出图入口：图片 + 「找什么」+ 输出路径 -> 标注图直接落盘。

        from qsmy_deepseek_locator import locate_to_file

        # 最省事的写法：三个位置参数，其余全默认
        result = locate_to_file("photo.png", "红色圆形", "runs/photo_annotated.png")

        # 需要时再补可选参数（下面这些键都能直接用，名字与 Locator.locate_to_file 一致）
        result = locate_to_file(
            "photo.png", "登录按钮", "out/btn.png",
            api_key="sk-xxx",        # 传了就不读 DEEPSEEK_API_KEY
            thinking=True,           # 默认 False（显式关闭思考）
            image_detail="high",     # 默认 "original"
            use_tools=False,         # v0.1 只能是 False，传 True 会报错
        )
        print(result.annotated_path)  # 实际写出的文件（没写扩展名时会补 .png）

    参数分三类，函数内部自动分流：

        - Locator 构造参数：settings / client / max_side / max_retries；
        - 单次调用参数：api_key / thinking / image_detail / prompt / model / ... ；
        - 绘制参数：colors / box_width / point_radius / font_size / draw_label。

    反复出图时请自己建一个 Locator 再调用它的 locate_to_file() ——
    这个函数每次都会重新读环境变量并新建客户端（和 locate() 是同一个取舍）。
    """
    locator_kwargs, call_kwargs = _split_kwargs(kwargs)
    locator = Locator(**locator_kwargs)
    return locator.locate_to_file(image, target, output, **call_kwargs)


__all__ = [
    "Locator",
    "LocateResult",
    "locate",
    "locate_to_file",
    "DEFAULT_SYSTEM_PROMPT",
]
