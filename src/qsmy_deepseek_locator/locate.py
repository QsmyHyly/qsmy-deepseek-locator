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

**阻塞与取消（本次改动补）**：locate() 全程同步，最坏情况等 300s × (max_retries+1) = 900s。
调用方要么把它放进后台线程/子进程，要么传 cancel_event=threading.Event 换一个「取消」按钮
（每个流式事件都会查一次开关，set() 之后抛 CancelledError）。本库**不会**主动取消任何请求。

本模块**不做**的三件事，都是有意的：
    - 不静默降级。没有 API Key 就抛 MissingAPIKeyError，不返回假的坐标。
    - 不自动重试「模型没找到目标」这种结果。空结果与调用失败是两回事，前者是正常答案。
    - 不把越界坐标偷偷改小。越界往往意味着模型给了像素坐标，那是一条**关于提示词的情报**，
      悄悄夹紧等于把情报抹掉；夹紧只发生在绘制那一刻（drawing.draw）。

拆分后的职责与边界：
    本模块只剩**动作** —— Locator 类与两个模块级一行式入口，负责「把图变成结果」；
    结果对象本身（LocateResult）与「空结果该怎么说」的两条提示语搬到了 results.py。
    搬走的那部分在拆分前位于本文件第 54-216 行，现在由下面那条 import 拿回来，
    所以 from .locate import LocateResult 这条老路径照旧可用 ——
    外部（含 App 项目 vendor 的镜像）看不到任何差别。

唯一有意的差别：json / dataclass / field / Iterator 这些**标准库 import 的顺带可见**没有保留。
它们从来不是本模块的对外名字（不在 __all__ 里、也没有任何文档提过），保留下来只会让
「这个模块到底提供什么」更难看清。用到的库内公开件（extract_json_block 等）照旧可见。
"""

from __future__ import annotations

import inspect
import os
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Sequence

from .client import (
    ChatReply,
    DeepSeekVisionClient,
    EventCallback,
    VisionClient,
    build_messages,
)
from .config import Settings, api_key_from_env
from .debuglog import coerce_log
# _prepare_dir 是私有的，但在本包里共用它正是「同一个实现只允许有一份」：
# 「输出目录建不出来」的报错措辞与异常类型只该由 drawing 那一处定义。
from .drawing import _prepare_dir, resolve_output_path
from .errors import CancelledError, UnsupportedFeatureError, WriteError
from .images import describe_source, source_size, to_data_url
# extract_json_block 原本只因 results 那段解析逻辑在本文件里而顺带可见（不是本模块的职责）。
# 留下来是为了不切断任何旧 import 路径：本仓库看不出来谁在用，代价只有一行。
from .parsing import Detection, extract_json_block, parse_detections
from .prompts import DEFAULT_SYSTEM_PROMPT, build_user_prompt
# 结果对象与两条「为什么结果是空的」提示语来自 results.py（拆分出来的模块）。
# 在这里 import 有两个用途：Locator.locate() 要用它们；顺便保住旧路径的 re-export 语义。
from .results import (
    _EMPTY_ARRAY_NOTICE,
    _NO_COORD_NOTICE,
    LocateResult,
    _returned_empty_array,
)


# 取消时抛的异常用同一句文案。写成常量是为了让调用方在日志/UI 里能按文案匹配，
# 也免得两个检查点说的话不一样。
_CANCEL_MESSAGE = (
    "调用已被 cancel_event 取消（调用方主动放弃，不是出错）。"
    "已经发出去的请求不会撤回，服务端可能仍在生成 —— 但本库不再等待、也不再解析结果。"
)


def _own_frames_above() -> int:
    """从本函数往上数「还有几个栈帧属于本库」，用于把告警指到**调用方那一行**。

    写死 stacklevel 是不行的：同一个 _complete 可能被 Locator.locate 调、被模块级
    locate() 调、被 CLI 调，深度各不相同。写死的话，经 CLI 进来时告警会指到 cli.py 里
    的一行 —— 用户照着那行去改自己的客户端，改了个寂寞。

    返回 N 表示"本函数之上还有 N 个本库帧"，**不含本函数自己那一帧**（第一个 f_back 就已经
    是调用方了）。调用点用 stacklevel=N+1：+1 回到本函数、+N 走到库内最外那一帧、
    再 +1 才跨出库外 —— 也就是"库内帧数（含本函数）+1"。
    实测：直接调用时 N=0（其余 0 个库内帧，stacklevel=1 就是 user 自己），
    经 Locator.locate 调用时 N=1（stacklevel=2 = locate 的调用方 = user）。
    **这个数是量出来的**：先前写成 +2 会落到 pytest 的 _pytest/python.py，写成 +3 落到 pluggy。
    与 drawing.py 的 _own_frames_above 同源但计数口径差 1，改一处务必想想另一处。
    这个套路与 drawing.py 的 _own_frames_above 同源，但计数口径差 1（那里是从"本函数"开始数），
    改一处务必想想另一处 —— 写反了的表现是告警指回库内部，很难一眼看出来。
    """
    own = os.path.abspath(__file__)
    frame = inspect.currentframe()
    count = 0
    try:
        frame = frame.f_back if frame is not None else None
        while frame is not None and os.path.abspath(frame.f_code.co_filename) == own:
            count += 1
            frame = frame.f_back
    finally:
        del frame
    return count

def _complete(client: Any, messages: list, **kwargs: Any) -> ChatReply:
    """调客户端的 complete()，并对「老式自备客户端不认新形参」做一次优雅降级。

    背景：timeout 是本次改动新加进协议的关键字参数，用来把生效后的读超时传给自备客户端
    （安卓上只有它能用，SDK 那条路装不上）。但**已有的自备客户端是按旧协议写的**，
    签名里没有 timeout —— 直接传就是 TypeError，等于把一次升级变成线上崩溃。

    所以这里只对「TypeError 且报错文本点名 timeout」降级重试一次，并 **warnings.warn**
    说明「你的客户端收不到本次的 timeout」。为什么不静默吞掉：静默意味着用户设的 30 秒
    超时在这台机器上永远不生效，而现象只是「怎么还是卡这么久」，根本归因不到这里。
    """
    try:
        return client.complete(messages, **kwargs)
    except TypeError as exc:
        if "timeout" not in str(exc) or "timeout" not in kwargs:
            raise
        warnings.warn(
            f"{type(client).__name__}.complete() 不接受 timeout 参数，本次调用的读超时"
            f"（{kwargs['timeout']}s）没有传给它，它会用自己构造时设的那个值。"
            "按 VisionClient 协议补一个 timeout=None 形参（忽略即可）就能消除这条告警。",
            UserWarning,
            stacklevel=_own_frames_above() + 1,
        )
        fallback = {k: v for k, v in kwargs.items() if k != "timeout"}
        return client.complete(messages, **fallback)


def _cancellable(
    on_event: EventCallback | None, cancel_event: "threading.Event | None"
) -> EventCallback | None:
    """把 on_event 包一层，在每个流式事件到达时查一次取消开关。

    为什么挂在 on_event 上而不是自己去遍历事件流：事件回调是**已经存在**的注入点，
    库的客户端每收到一个 chunk 就调它一次（实测一次定位 122 条事件），粒度足够细；
    为此再给 VisionClient 协议加一个参数，等于让所有自备客户端都跟着改签名 ——
    那正是 P1-8 刚修完的那类问题，不该再制造一个。

    没传 cancel_event 时**原样返回**（连包装都不包）：不改变任何现有行为，
    自备客户端拿到的还是它自己那个回调对象。

    包装后的回调是同步抛 CancelledError —— 异常会从客户端的事件循环里一路穿出来，
    客户端自身的清理（关闭响应流）由它的 finally 负责，本函数不管这件事。
    """
    if cancel_event is None:
        return on_event

    def _wrapped(event: dict) -> None:
        if cancel_event.is_set():
            raise CancelledError(_CANCEL_MESSAGE)
        if on_event is not None:
            on_event(event)

    return _wrapped


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
        if base.api_key is None:
            # 显式传入的 Settings 没带 Key 时，仍按本模块声明的优先级链回落到环境变量：
            #   函数参数 > Locator 构造参数 > 环境变量 > 内置默认
            # 不这么做的话，Locator(settings=Settings(model="deepseek-flash")) 这种「只想换个
            # 模型」的写法会把环境变量里的 Key 一起丢掉，抛出的 MissingAPIKeyError 还会建议你
            # 「设置 DEEPSEEK_API_KEY」—— 而它其实早就设好了，极具误导性。
            # 只有 api_key 走这条回落：其余字段的 None 是「不发送该参数」的明确语义（见 Settings）。
            base = base.merged(api_key=api_key_from_env())
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
        log_file: Any = None,
        cancel_event: "threading.Event | None" = None,
    ) -> LocateResult:
        """定位一张图里的目标。

        ⚠️ **这是个同步阻塞调用，可能阻塞数分钟**（默认 timeout=300s × (max_retries+1) = 900s
        是最坏情况），**不要在主线程 / UI 线程里直接调** —— 安卓上那样会 ANR，
        桌面 GUI 上会卡住整个窗口。移动端 / GUI 请丢进后台线程或子进程。
        想在阻塞期间能喊停，传 cancel_event（见下）。

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
            client: 本次调用换用别的客户端（给了它就要能接受 log 关键字参数，见下）。
            on_event: 流式事件回调，形如 on_event({"type": "reasoning"|"content"|...})。
            log_file: 调试日志。None = 按配置/环境变量（QSML_LOG_FILE），False = **明确关闭**，
                路径 = 写到该文件（JSONL，追加），True = 自动路径。见 debuglog.py。
                日志记的是网络层：请求体（图片 data URL 已省略）、每个流式事件、
                完整响应体、结构化结果、以及任何异常。**默认全程不开。**
                @doc README.md#71-调试日志把请求体和响应体落盘
                （该文档解决"日志里有什么、怎么读、什么该记什么不该记"的问题。）
                自备 client 时：本库只在日志开启时才会把 log 传给它，所以不用日志的老客户端
                不受影响；一旦开了日志，那个客户端就得接受 log 关键字参数（VisionClient 协议已含）。
            cancel_event: 取消开关（threading.Event）。调用方 set() 之后，本方法会在
                **下一个流式事件到达时**抛 CancelledError。检查点有三处：进入方法时、
                每个流式事件、以及模型返回之后。不传就完全保持旧行为（不检查、不打断）。

        Returns:
            LocateResult。**模型没找到目标时不会抛异常**，而是返回 detections 为空的结果 ——
            与「调用失败」区分开（后者抛 APIError / EmptyResponseError / MissingAPIKeyError）。

        Raises:
            CancelledError: cancel_event 被 set 了（只可能由调用方自己触发）。
            其余异常见 errors.py；本库抛出的东西**总是** LocatorError。
        """
        if cancel_event is not None and cancel_event.is_set():
            # 进来时已经取消了就别发请求了 —— 一次调用可能花掉真金白银，
            # 「点了取消还扣一次钱」是最让人恼火的那种 bug。
            raise CancelledError(_CANCEL_MESSAGE)
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

        # 日志开关：单次参数 > 配置/环境变量。传 False 是一条**明确的关闭**通道，
        # 用于盖掉环境变量里开着的日志（否则「临时不想记」只能改环境变量）。
        log = coerce_log(log_file if log_file is not None else effective.log_file)
        # 只在开启日志时才把 log 传下去：自备客户端不必为了不用日志而改签名。
        log_kwarg = {"log": log} if log is not None else {}

        started = time.perf_counter()
        reply = _complete(
            active_client,
            messages,
            settings=effective,
            on_event=_cancellable(on_event, cancel_event),
            timeout=effective.timeout,
            **log_kwarg,
        )
        duration_ms = (time.perf_counter() - started) * 1000.0
        # 模型返回后、解析之前再查一次：事件回调是"边收边查"，最后一个事件之后
        # 到 complete() 返回之间还有一段时间（拼工具调用、写日志），那段空档不能漏。
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError(_CANCEL_MESSAGE)

        detections, warnings, raw_items = parse_detections(reply.text)
        if not detections and reply.text.strip() and not warnings:
            warnings.append(
                _EMPTY_ARRAY_NOTICE if _returned_empty_array(reply.text) else _NO_COORD_NOTICE
            )

        result = LocateResult(
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
        if log is not None:
            # 结构化结果也留一份：正文/思考在 reply 行里已经有了，这里不重复，
            # 只留解析出来的坐标与告警 —— 排查「模型答了但解析没跟上」时靠它。
            log.write("result", result.to_dict(include_text=False, include_raw=True))
        return result

    # ------------------------------------------------------------------ #
    def locate_and_draw(
        self,
        image: Any,
        target: str | None = None,
        *,
        output: str | Path | None = None,
        box_width: int = 3,
        point_radius: int = 5,
        font_size: int = 22,
        draw_label: bool = True,
        colors: "Sequence[str] | None" = None,
        font_path: "str | Path | None" = None,
        **kwargs: Any,
    ):
        """定位并直接把结果画回图上，返回 (LocateResult, PIL.Image)。

        output 给了就顺手落盘。这是个便利方法：只要坐标不要图的场景请直接用 locate()；
        只要「画好的图片文件」的场景请用 locate_to_file()（它会校验路径、按后缀定格式，
        并把落盘路径写进 result.annotated_path）。

        绘制参数（box_width / point_radius / font_size / draw_label / colors / font_path）
        在这里是**显式形参**，与 Locator.locate_to_file 完全同名同义 —— 以前它们被塞进
        **kwargs 再转给 locate()，而 locate() 没有这些形参，于是要么 TypeError、
        要么（更糟）被默默忽略：`Locator(font_path=…)` 在这条路上曾整条失效，
        中文标签照样画成方块，调用方却以为自己已经指定好了。

        ⚠️ 与 locate() 一样是**同步阻塞**调用，别在主线程里调。
        """
        from .drawing import draw

        result = self.locate(image, target, **kwargs)
        annotated = draw(
            image,
            result.detections,
            box_width=box_width,
            point_radius=point_radius,
            font_size=font_size,
            draw_label=draw_label,
            colors=colors,
            # 与 locate_to_file 同一口径：显式给了就用它，没给才回落到 Settings.font_path。
            font_path=font_path or self.settings.font_path,
        )
        if output:
            # 与 save_annotated / locate_to_file 走同一条路径解析：以前这里写死
            # format="PNG" 且直接 mkdir+save，于是「.jpg 后缀得到 PNG 字节」和
            # 「裸 OSError 漏出 LocatorError 契约」这两个问题在这一条路上都还在。
            target_path, fmt = resolve_output_path(output)
            _prepare_dir(target_path.parent)
            try:
                annotated.save(target_path, format=fmt)
            except (OSError, ValueError) as exc:
                raise WriteError(
                    f"标注图写入失败：{target_path}（{type(exc).__name__}: {exc}）\n"
                    "识别已经完成，只是图没落盘；换个可写目录重试即可，不必再调一次模型。"
                ) from exc
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
        log_file: Any = None,
        box_width: int = 3,
        point_radius: int = 5,
        font_size: int = 22,
        draw_label: bool = True,
        colors: Sequence[str] | None = None,
        font_path: str | Path | None = None,
        cancel_event: "threading.Event | None" = None,
    ) -> LocateResult:
        """定位 + 打标 + 落盘：一次调用把「已经画好框的图片文件」交到你手里。

        ⚠️ **这是个同步阻塞调用，可能阻塞数分钟**（默认 timeout=300s × (max_retries+1) = 900s
        是最坏情况），**不要在主线程 / UI 线程里直接调**。移动端与 GUI 请丢进后台线程，
        并给 cancel_event 留一个「取消」按钮。

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
            use_tools: 是否开启工具（Agent）调用。**这里只能保持 False**：工具循环要多轮编排，
                走 qsmy_deepseek_locator.agent.run_agent（0.2.0 起提供）。传 True 会当场报错，不会被静默忽略；
                传 True 会当场抛 UnsupportedFeatureError（同时也是 NotImplementedError，
                0.1.2 抛的就是它）。这是预留参数：宁可报错，也不静默忽略 ——
                静默忽略会让你以为工具已经开了。
                想自己接工具请走底层：DeepSeekVisionClient.complete(messages, tools=[...])，
                调用结果落在 ChatReply.tool_calls。本库负责透传报文与拼回分片，**不执行工具**。
                @doc docs/API-NOTES.md#61-工具调用也是流式的而且一个字符一个-chunk
                （该文档解决"工具调用的参数为什么要按 index 自己拼、on_event 能拿到什么"的问题。）
            max_tokens: 输出上限（含思考 token）。正文为空时优先调大它。
            timeout: 单次请求超时（秒）。
            max_side: 发送前把图缩到最长边不超过它（归一化坐标不受影响，只省上行流量）。
            on_event: 流式事件回调，可拿来做进度显示。
            log_file: 调试日志（请求体 / 事件流 / 响应体 / 结果 / 异常写成 JSONL）。
                None = 按配置/环境变量，False = 明确关闭，路径 = 写到该文件，True = 自动路径。
                **默认不开。** 日志记的是「发给模型什么、模型回了什么」，见 debuglog.py。
            box_width / point_radius / font_size / draw_label / colors / font_path: 绘制样式与字体，
                见 drawing.draw。font_path 也接受从 Settings.font_path 继承（构造 Locator 时给）。

        Returns:
            LocateResult。「annotated_path」是刚写出的文件路径（相对路径按当前工作目录解析）；
            「detections」为空表示模型没找到目标 —— 那不是失败，图照样落盘（内容等于原图）。

        Note:
            打标画的是**原图**（不是发给模型的那份可能被缩小的副本），
            所以输出图的分辨率始终等于输入图。URL 输入会被下载两次（发模型一次、绘制一次），
            与 locate() 内部 source_size 的行为一致。
        """
        if use_tools:
            # 为什么不做成 locate() 的开关，而是指到另一个入口 —— 见下面这段说明。
            raise UnsupportedFeatureError(
                "use_tools=True 不在这个入口上生效：本方法是**单轮**调用，"
                "而工具（Agent）循环要多轮编排，它在另一个入口上：\n"
                "    from qsmy_deepseek_locator.agent import run_agent\n"
                "    for event in run_agent(messages, tool_context={'source': 图片}):\n"
                "        ...   # round_start / reasoning / content / tool_call / tool_result / done\n"
                "为什么不做成这里的一个布尔开关：Agent 循环每多一轮就多一次计费调用，"
                "把它藏在 use_tools=True 后面，会让「这次要花多少钱、要等多久」"
                "变得无法预期 —— 那必须是调用方显式承接的决定，不该由一个参数偷偷决定。\n"
                "只想拿到模型的工具调用请求（不执行）可以用底层：\n"
                "    client = DeepSeekVisionClient()\n"
                "    reply = client.complete(messages, tools=[...])\n"
                "    reply.tool_calls  # 已按 index 拼好，arguments 是完整 JSON 串\n"
                "事件流示例见 examples/stream_events.py --tools。"
            )


        # 先校验输出路径再调模型：这是**故意**的顺序，一次 API 调用不该因为路径拼错而白花。
        # 父目录也在这里一并建出来，而不是等画完再 mkdir：路径不可写（父级是个文件、
        # 没有写权限）属于「本地输入问题」，不该花掉一次 API 调用才暴露。
        #
        # P2-1 的由来：默认输出位置相对 CWD，而安卓上 CWD 通常是 / 或不可写目录，
        # 「相对路径 + 不可写」叠在一起时调用方拿到的只是一句原生 OSError。
        # 现在路径问题一律是 OutputPathError / WriteError（都是 LocatorError），
        # 报错里带**具体路径**与常见原因，见 drawing._prepare_dir。
        path, fmt = resolve_output_path(output)
        _prepare_dir(path.parent)

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
            log_file=log_file,
            cancel_event=cancel_event,
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
            # 显式给了就用它；没给才回落到 Settings.font_path（可能是构造 Locator 时给的，
            # 也可能是 QSML_FONT_PATH）。resolve_font 自己也读环境变量，这里传下去只是
            # 让「Settings 里的值」这条更明确的路也成立 —— 两者最终指向同一个文件。
            font_path=font_path or self.settings.font_path,
        )
        try:
            annotated.save(path, format=fmt)  # 父目录已在调模型之前建好，见上面那段注释
        except (OSError, ValueError) as exc:
            # 反馈里点名的那处「最终落盘那行没有 try/except」就是这里。
            # 它是最难受的失败位置：钱已经花了、结果也解析出来了，只在最后一步写不进去。
            # 所以文案必须说清「不用再调一次模型」，否则用户第一反应是重跑。
            raise WriteError(
                f"标注图写入失败：{path}（{type(exc).__name__}: {exc}）\n"
                "识别已经完成，只是图没落盘（常见原因：磁盘满、目录只读、同名的目录占了位置）。"
                "换个可写目录或传绝对路径重试即可，**不必再调一次模型**。"
            ) from exc
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

    ⚠️ 与 Locator.locate 一样是**同步阻塞**调用（最坏 timeout × (max_retries+1)），
    别在主线程里调；要能中途停下就传 cancel_event=threading.Event()。
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
            use_tools=False,         # 只能是 False；工具循环走 agent.run_agent
            font_path="/system/fonts/NotoSansCJK-Regular.ttc",   # 指定中文字体（默认自动探测）
            cancel_event=threading.Event(),   # 想要「取消」按钮时给（见 Locator.locate）
        )
        print(result.annotated_path)  # 实际写出的文件（没写扩展名时会补 .png）

    参数分三类，函数内部自动分流：

        - Locator 构造参数：settings / client / max_side / max_retries；
        - 单次调用参数：api_key / thinking / image_detail / prompt / model / ... ；
        - 绘制参数：colors / box_width / point_radius / font_size / draw_label / font_path。

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
