"""模型客户端：拼报文 + 流式收包，把 DeepSeek 的细节挡在这一层里。

对外只有两个东西：
    ChatReply                     一次调用的完整结果（正文 / 思考 / usage / 结束原因）
    DeepSeekVisionClient.complete 收完整个流，返回 ChatReply（可选 on_event 回调看进度）

为什么**永远走流式**（哪怕调用方只想要最终文本）：
    1. 思考模型先吐 reasoning_content 再吐 content，非流式时这段时间是完全静默的，
       命令行里看着像卡死；流式则能实时显示「模型在想什么」。
    2. on_event 回调让 CLI / GUI 不必自己实现一套流式解包。
    3. 少一条代码路径就少一处两边行为不一致的可能。

必须先知道的两个坑（都是实测踩出来的）：

    a) thinking 参数**不是** Chat Completions 的顶层字段，只能塞进 extra_body；
       reasoning_effort 反而是顶层参数。
    b) 思考 token 与正文共用输出上限：上限被思考吃满时 content 是空串，
       而 HTTP 状态码依然是 200。所以「空正文」要当成明确的错误抛出来，
       而不是当成「图里没有目标」——那是两回事。

@doc docs/API-NOTES.md#4-思考模式与-reasoning_effort
（该文档解决"thinking 参数为什么必须走 extra_body、effort 各档怎么映射"的问题。）

@doc docs/API-NOTES.md#6-流式必须同时读-reasoning_content
（该文档解决"流式 chunk 长什么样、为什么不能只读 delta.content"的问题。）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol

from .config import REASONING_EFFORTS, IMAGE_DETAILS, Settings
from .errors import APIError, EmptyResponseError

# 事件回调签名：on_event({"type": "reasoning"|"content"|"finish", ...})
EventCallback = Callable[[dict], None]


@dataclass
class ChatReply:
    """一次模型调用的结果。"""

    text: str = ""
    reasoning: str = ""
    model: str = ""
    finish_reason: str | None = None
    usage: dict | None = None
    tool_calls: list[dict] = field(default_factory=list)

    @property
    def truncated(self) -> bool:
        """本轮是否因为输出上限被截断（finish_reason == length）。"""
        return self.finish_reason == "length"


class VisionClient(Protocol):
    """客户端协议。测试里注入假客户端只需要实现这一个方法。

    settings 是**本次调用**生效的配置（单次覆盖就是靠它传下来的），None = 用客户端自己的配置。
    """

    def complete(
        self,
        messages: list[dict],
        *,
        settings: "Settings | None" = None,
        on_event: EventCallback | None = None,
    ) -> ChatReply:
        ...


# --------------------------------------------------------------------------- #
# 报文拼装（纯函数，便于单测直接断言，不必联网）
# --------------------------------------------------------------------------- #
def image_part(url: str, detail: str | None = None) -> dict:
    """构造一个 image_url 内容块。

    detail 只接受 low / high / original / auto；空值或不认识的值一律**不带该字段**。
    非法值不抛异常是刻意的：它在「拼一次识别请求」的主路径上，为它抛异常会让整轮识别挂掉，
    而写错一个 detail 最多只是「没生效」。要严格校验的地方是配置层。
    """
    payload: dict[str, Any] = {"url": url}
    if detail and detail in IMAGE_DETAILS:
        payload["detail"] = detail
    return {"type": "image_url", "image_url": payload}


def thinking_payload(enabled: bool | None) -> dict | None:
    """构造 thinking 开关；None 表示**不发送该字段**（由服务端按默认处理，当前默认开启）。"""
    if enabled is None:
        return None
    return {"thinking": {"type": "enabled" if enabled else "disabled"}}


def resolve_thinking(
    settings: Settings, thinking: bool | None, reasoning_effort: str | None = None
) -> tuple[bool | None, str | None]:
    """合并「按次覆盖」与「配置默认」，得到最终生效的 (thinking, effort)。

    规则：
    - thinking 为 None 且配置也没给 -> (None, None)，即两样都不传，服务端自己决定；
    - 显式关闭思考 -> (False, None)，此时 effort 无意义，直接丢掉；
    - 开启思考时 effort 必须在白名单内，非法值一律不传（宁可走服务端默认，也不要 400）。
    """
    enabled = settings.thinking if thinking is None else bool(thinking)
    if enabled is False:
        return False, None
    effort = (settings.reasoning_effort if reasoning_effort is None else reasoning_effort) or ""
    effort = effort.strip().lower()
    effort = effort if effort in REASONING_EFFORTS else None
    return enabled, effort


def build_messages(
    prompt: str,
    *,
    image_url: str | None = None,
    system_prompt: str | None = None,
    image_detail: str | None = None,
) -> list[dict]:
    """拼 system + user 两条消息；有图时 user 用内容块数组（图片只能在 user 消息里）。

    ⚠️ 官方明确：图片放在 system / assistant 消息里会 400，别乱挪。
    """
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if image_url:
        messages.append({
            "role": "user",
            "content": [
                image_part(image_url, image_detail),
                {"type": "text", "text": prompt},
            ],
        })
    else:
        messages.append({"role": "user", "content": prompt})
    return messages


def build_request(settings: Settings, messages: list[dict], *, stream: bool = True) -> dict:
    """拼出 chat.completions.create 的关键字参数（不含客户端本身）。

    单独抽出来是为了让自测能直接断言「报文长什么样」，不用打桩整个 SDK。
    """
    enabled, effort = resolve_thinking(settings, None, None)
    kwargs: dict[str, Any] = {
        "model": settings.model,
        "messages": messages,
        "stream": stream,
    }
    extra = thinking_payload(enabled)
    if extra is not None:
        kwargs["extra_body"] = extra
    if effort:
        kwargs["reasoning_effort"] = effort
    if settings.max_tokens:
        kwargs["max_tokens"] = settings.max_tokens
    if stream:
        # usage 只在显式要求时才随最后一个 chunk 回来；拿不到就是 None，不影响主流程。
        kwargs["stream_options"] = {"include_usage": True}
    return kwargs


# --------------------------------------------------------------------------- #
# 真实客户端
# --------------------------------------------------------------------------- #
class DeepSeekVisionClient:
    """基于 openai SDK 的 DeepSeek 客户端（OpenAI 兼容协议）。

    SDK 的导入放在 __init__ 里：只想用解析 / 打标（不联网）的调用方
    不该因为环境里没装 openai 就 import 失败。
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self._client: Any = None
        # 连接参数变了就重建 SDK 客户端（单次覆盖可能换 base_url / timeout / key）。
        # 按 key 缓存而不是每次 new：OpenAI() 会新建一个 httpx 连接池，每调用一次建一个太浪费。
        self._client_key: tuple | None = None

    def client_for(self, settings: Settings) -> Any:
        key = (settings.api_key, settings.base_url, settings.timeout, settings.max_retries)
        if self._client is not None and self._client_key == key:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - 环境问题
            raise APIError(
                "缺少依赖 openai。安装：pip install openai（或 pip install qsmy-deepseek-locator）"
            ) from exc
        self._client = OpenAI(
            api_key=settings.require_api_key(),
            base_url=settings.base_url,
            timeout=settings.timeout,
            max_retries=settings.max_retries,
        )
        self._client_key = key
        return self._client

    # -- 内部：发起请求，stream_options 不被支持时自动退一步重试 -------- #
    def _create(self, kwargs: dict, settings: Settings) -> Any:
        client = self.client_for(settings)
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            # 有些 OpenAI 兼容服务（含自建代理）不认 stream_options，会直接 400。
            # 这只影响「能不能顺手拿到 usage」，不值得让整轮识别失败，所以去掉重试一次。
            if "stream_options" in str(exc) and "stream_options" in kwargs:
                retry = {k: v for k, v in kwargs.items() if k != "stream_options"}
                try:
                    return client.chat.completions.create(**retry)
                except Exception as exc2:  # noqa: BLE001
                    raise APIError(_format_api_error(exc2, settings)) from exc2
            raise APIError(_format_api_error(exc, settings)) from exc

    def stream(
        self,
        messages: list[dict],
        *,
        settings: Settings | None = None,
        stream: bool = True,
    ) -> Iterator[dict]:
        """逐段产出事件：{"type": "reasoning"|"content"|"finish"|"usage", ...}。"""
        effective = settings or self.settings
        kwargs = build_request(effective, messages, stream=stream)
        raw = self._create(kwargs, effective)
        if not stream:
            yield from _events_from_completion(raw)
            return
        for chunk in raw:
            yield from _events_from_chunk(chunk)

    def complete(
        self,
        messages: list[dict],
        *,
        settings: Settings | None = None,
        on_event: EventCallback | None = None,
    ) -> ChatReply:
        """收完整个流，返回 ChatReply。

        settings 为**本次调用**的配置覆盖（None = 用客户端自身配置）。
        """
        effective = settings or self.settings
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_reason: str | None = None
        usage: dict | None = None
        model = effective.model

        for event in self.stream(messages, settings=effective):
            etype = event.get("type")
            if etype == "reasoning":
                reasoning_parts.append(event["text"])
            elif etype == "content":
                text_parts.append(event["text"])
            elif etype == "finish":
                finish_reason = event.get("reason") or finish_reason
            elif etype == "usage":
                usage = event.get("usage")
            elif etype == "model":
                model = event.get("model") or model
            if on_event is not None:
                on_event(event)

        reply = ChatReply(
            text="".join(text_parts),
            reasoning="".join(reasoning_parts),
            model=model,
            finish_reason=finish_reason,
            usage=usage,
        )
        if not reply.text.strip():
            raise EmptyResponseError(_empty_hint(reply))
        return reply


def _empty_hint(reply: ChatReply) -> str:
    """正文为空时给一句能直接照做的提示（这类问题九成是输出预算被思考吃光）。"""
    reasoning_chars = len(reply.reasoning)
    if reply.truncated:
        return (
            f"模型返回的正文是空的，且本轮因输出上限被截断（finish_reason=length，"
            f"思考内容 {reasoning_chars} 字）。思考 token 与正文共用输出上限，"
            "思考把预算吃光时正文就是空串（HTTP 仍是 200）。"
            "对策：调大 max_tokens，或关掉思考（thinking=False），或降低 reasoning_effort。"
        )
    if reasoning_chars:
        return (
            f"模型返回的正文是空的（思考内容 {reasoning_chars} 字）。"
            "可先调大 max_tokens 或关掉思考（thinking=False）再试；"
            "若仍为空，检查提示词是否要求了输出 JSON。"
        )
    return "模型返回的正文是空的（连思考内容也没有）。请检查模型名与请求参数是否被服务端接受。"


def _format_api_error(exc: Exception, settings: Settings) -> str:
    """把 SDK 异常转成一句能照着排查的话（含状态码与响应体片段）。"""
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    detail = ""
    if isinstance(body, dict):
        message = body.get("message") or body.get("error")
        detail = f"：{message}" if message else ""
    elif body:
        detail = f"：{str(body)[:300]}"
    head = f"调用 {settings.model} 失败"
    if status:
        head += f"（HTTP {status}）"
    return f"{head}{detail}（原始异常：{type(exc).__name__}: {exc}）"


def _events_from_chunk(chunk: Any) -> Iterator[dict]:
    """把一个流式 chunk 拆成事件。

    注意 chunk.choices 可能为空数组：开了 include_usage 时，最后一个只带 usage 的
    chunk 就是这个形状，直接取 choices[0] 会 IndexError。
    """
    usage = getattr(chunk, "usage", None)
    if usage is not None:
        yield {"type": "usage", "usage": _usage_dict(usage)}
    model = getattr(chunk, "model", None)
    if model:
        yield {"type": "model", "model": model}

    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return
    choice = choices[0]
    delta = getattr(choice, "delta", None)

    reasoning = getattr(delta, "reasoning_content", None) if delta is not None else None
    if reasoning:
        yield {"type": "reasoning", "text": reasoning}

    content = getattr(delta, "content", None) if delta is not None else None
    if content:
        yield {"type": "content", "text": content}

    finish = getattr(choice, "finish_reason", None)
    if finish:
        yield {"type": "finish", "reason": finish}


def _events_from_completion(completion: Any) -> Iterator[dict]:
    """非流式响应拆成同样的事件（保留非流式入口，方便对接只支持非流的代理）。"""
    usage = getattr(completion, "usage", None)
    if usage is not None:
        yield {"type": "usage", "usage": _usage_dict(usage)}
    model = getattr(completion, "model", None)
    if model:
        yield {"type": "model", "model": model}
    choices = getattr(completion, "choices", None) or []
    if not choices:
        return
    message = getattr(choices[0], "message", None)
    reasoning = getattr(message, "reasoning_content", None) if message is not None else None
    if reasoning:
        yield {"type": "reasoning", "text": reasoning}
    content = getattr(message, "content", None) if message is not None else None
    if content:
        yield {"type": "content", "text": content}
    finish = getattr(choices[0], "finish_reason", None)
    if finish:
        yield {"type": "finish", "reason": finish}


def _usage_dict(usage: Any) -> dict:
    """把 SDK 的 usage 对象转成普通 dict（不同版本字段名不完全一样，能取多少取多少）。"""
    if isinstance(usage, dict):
        return dict(usage)
    out: dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if value is not None:
            out[key] = value
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", None) if details is not None else None
    if reasoning_tokens is not None:
        out["reasoning_tokens"] = reasoning_tokens
    return out


__all__ = [
    "ChatReply",
    "VisionClient",
    "DeepSeekVisionClient",
    "EventCallback",
    "build_messages",
    "build_request",
    "image_part",
    "thinking_payload",
    "resolve_thinking",
]
