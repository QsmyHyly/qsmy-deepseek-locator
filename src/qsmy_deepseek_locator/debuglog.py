"""调试日志：把网络层的请求体与响应体落成 JSONL 文件（**默认关闭**）。

为什么要它：模型给的结果不对劲时，「它到底收到了什么、回了什么」是唯一能定位问题的东西。
不开日志时本库把报文和响应全丢掉了（只留 LocateResult 里的正文/思考/usage），
排查「是不是图没传上去」「是不是提示词被服务端截了」就只能靠猜。

开启方式（越靠前越优先）：

    locate(..., log_file="runs/logs/run.jsonl")        # 函数参数
    Locator(log_file="runs/logs/run.jsonl")            # 构造参数（同上，等价）
    QSML_LOG_FILE=runs/logs/run.jsonl                  # 环境变量
    qsmy-deepseek-locator img.png -t 红色圆形 --log    # CLI：自动路径 runs/logs/qsml-时间戳.jsonl

写出来是**一行一个 JSON 对象**（JSONL），能直接 grep、能喂给脚本：

    {"ts": "…", "event": "request", "data": {"settings": {...}, "body": {...}}}
    {"ts": "…", "event": "event",   "data": {"type": "reasoning", "text": "…"}}
    {"ts": "…", "event": "chunk",   "data": {...}}      # 只在 chunks=True 时有
    {"ts": "…", "event": "reply",   "data": {"text": "…", "usage": {...}}}
    {"ts": "…", "event": "result",  "data": {"detections": [...], "warnings": [...]}}
    {"ts": "…", "event": "error",   "data": {"type": "APIError", "message": "…"}}

四个刻意的取舍：

1. **图片不写进日志**。报文里的图片是 data URL（base64），一张 1200x900 的图就是几百 KB
   到几 MB，原样落盘会让日志比图还大、也没法看。所以 data URL 一律换成
   占位符「data:image/png;base64,（省略 N 字符）」—— 保留了「类型 + 体积」这两条真正有用的信息。
2. **API Key 不进日志**。配置那一行用 config.redacted() 脱敏（只留首尾几位）。
3. **类实例用最简单的办法转文本**：json.dumps 的 default 兜底依次试
   model_dump / to_dict / __dict__ / str()，不写自定义 Encoder、不逐个类型适配。
   好处是**永远写得出来**，代价是降级到 str() 时只剩一句 repr。
4. **每行独立 open/append，不持有文件句柄**：跨进程、崩溃后追加都不会互相打架，
   也就不用管 close。一次调用百来行的开销相对一次 API 调用可以忽略。

5. **自定义 VisionClient 要自己写这几行**。本模块只提供写入端：request / event / reply /
   error 是 DeepSeekVisionClient 在收发时写的（见 client.py 的 complete / stream），
   result 由 Locator.locate 收尾时补。换成实现 VisionClient 协议的自建客户端时，
   若还想让日志可用，就得在等价位置自己调 log.write —— 否则日志里只会剩一条 result
   （实测：安卓上自建的 requests 客户端正是如此，拿得到结果、看不到模型原始回复，
   排查「是模型给错了还是我解析错了」时抓瞎）。

⚠️ 日志里**有完整的模型输入输出**（提示词、思考过程、坐标）。它适合自己排查，
别默认往公共 CI artifact 或别人的机器上丢。

@doc README.md#71-调试日志把请求体和响应体落盘
（该文档解决"什么时候该开日志、日志里有什么、怎么读"的问题。）
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# 短字符串（路径、URL、普通文本）原样保留，只有 data: 开头且超过这个长度的才省略 ——
# 图片是唯一会长成这样的东西，而这个阈值远大于任何正常的提示词或标签。
_DATA_URL_MIN = 128

# _plain 的递归深度上限：环状引用和病态嵌套宁可截断，也不能把主流程拖死。
_MAX_DEPTH = 8


def _now() -> str:
    """本地时间到毫秒。用本地时间是因为看日志的是人，不是日志系统。"""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    return f"{stamp}.{int(time.time() * 1000) % 1000:03d}"


def _shorten(value: str) -> str:
    """把图片 data URL 换成占位符（保留 MIME 与原始长度）。"""
    if len(value) < _DATA_URL_MIN or not value.startswith("data:"):
        return value
    comma = value.find(",")
    head = value[: comma + 1] if comma != -1 else value[:32]
    return f"{head}（省略 {len(value) - len(head)} 字符）"


def _fallback(obj: Any) -> Any:
    """json.dumps 认不出某个对象时的降级链：model_dump -> to_dict -> __dict__ -> str()。

    openai SDK 的响应对象都有 model_dump()，所以正常路径上拿到的是**真结构**；
    自己写的对象通常能走 __dict__；实在不行 str() 兜底，保证日志永远写得出来。
    """
    for name in ("model_dump", "to_dict", "dict"):
        method = getattr(obj, name, None)
        if callable(method):
            try:
                return method()
            except Exception:  # noqa: BLE001 - 降级路径，失败就继续往下试
                pass
    data = getattr(obj, "__dict__", None)
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if not k.startswith("_")}
    return str(obj)


def _plain(value: Any, depth: int = 0) -> Any:
    """把任意对象变成「json.dumps 一定吃得下」的结构，顺便做 data URL 省略。"""
    if depth > _MAX_DEPTH:
        return "（嵌套过深，已省略）"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _shorten(value)
    if isinstance(value, Mapping):
        return {str(k): _plain(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v, depth + 1) for v in value]
    return _plain(_fallback(value), depth + 1)


class DebugLog:
    """一行一个 JSON 对象的调试日志。

        log = DebugLog("runs/logs/run.jsonl")
        log.write("request", {"body": {...}})

    chunks=True 时连**原始 chunk** 也记（默认只记归一化后的六种事件）：
    归一化事件覆盖了思考/正文/工具调用/用量这些语义内容，够排查九成问题；
    要看「服务端是不是发了奇怪的字段」才需要原始 chunk，代价是行数涨十倍以上
    （实测一次定位 122 条事件 / 2880 个原始 chunk）。
    """

    def __init__(self, path: str | Path, *, chunks: bool = False):
        self.path = Path(path)
        self.chunks = chunks
        self._lock = threading.Lock()
        self._warned = False

    def __repr__(self) -> str:
        return f"DebugLog({str(self.path)!r}, chunks={self.chunks})"

    def write(self, event: str, data: Any = None) -> None:
        """追加一行。**写日志失败绝不影响主流程**：报一次到 stderr，然后闭嘴继续。"""
        try:
            line = json.dumps(
                {"ts": _now(), "event": event, "data": _plain(data)},
                ensure_ascii=False,
                default=_fallback,
            )
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        except Exception as exc:  # noqa: BLE001 - 见 docstring：日志不能反过来搞挂识别
            if not self._warned:
                self._warned = True
                sys.stderr.write(f"[qsml] 调试日志写入失败（不影响识别）：{exc}\n")


def default_log_path() -> Path:
    """自动日志路径：runs/logs/qsml-年月日-时分秒.jsonl。

    落在 runs/ 下是因为它已在 .gitignore 里 —— 日志天然是「跑完就该扔」的东西。
    """
    return Path("runs/logs") / time.strftime("qsml-%Y%m%d-%H%M%S.jsonl")


def coerce_log(value: Any = None, *, chunks: bool = False) -> DebugLog | None:
    """把调用方给的 log_file 参数变成 DebugLog 或 None（None = 不开日志）。

        None      不开，按配置/环境变量
        False     **明确关闭**（用于盖掉环境变量里开着的日志）
        True      开，路径用 default_log_path()
        路径       开，写到这里；空串按「不开」处理（.env 里写 QSML_LOG_FILE= 是常见写法）
        DebugLog  原样使用（chunks 等细节由调用方自己定）

    认不出的类型直接抛 ValueError：日志是个「以为自己开了其实没开」会很难受的东西，
    静默忽略比报错危险。
    """
    if value is None or value is False:
        return None
    if value is True:
        return DebugLog(default_log_path(), chunks=chunks)
    if isinstance(value, DebugLog):
        return value
    if isinstance(value, (str, Path)):
        text = str(value).strip()
        return DebugLog(text, chunks=chunks) if text else None
    raise ValueError(
        f"log_file 只接受 None / True / False / 路径 / DebugLog，收到 {type(value).__name__}"
    )


__all__ = ["DebugLog", "coerce_log", "default_log_path"]
