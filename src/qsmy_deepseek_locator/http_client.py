"""裸 HTTP 客户端：只用 requests 直连 DeepSeek，不依赖 openai SDK。

**职责**：实现 VisionClient 协议的第二个官方实现 —— 发一个 stream=True 的 POST，
逐行读 SSE，把每个 data 帧交给 stream_events._events_from_chunk 解包，
收完流拼成 ChatReply。它与 DeepSeekVisionClient 的**对外行为逐条对齐**：
同样的六种事件、同样的 tool_calls 累积、同样的「正文为空但有工具调用不算错」的边界。

**边界**：不拼报文（那是 request_build.build_request 的事）、不解析坐标（parsing）、
不画图（drawing）。本模块只负责中间那层「把 HTTP 字节流变成事件」。

## 什么时候该用它

| 场景 | 用哪个 |
|---|---|
| 普通环境，装了 openai | DeepSeekVisionClient（默认，SDK 帮你重试/超时/连接池） |
| 装不上 openai（安卓 / aarch64 / 无 Rust 工具链的镜像） | **RequestsVisionClient** |
| 想要更少的依赖、能读源码的传输层 | **RequestsVisionClient** |
| 想要 SDK 自带的重试与异常分类 | DeepSeekVisionClient |

为什么 openai 会装不上：它依赖 jiter 与 pydantic-core 两个 **Rust 扩展**，
二者都没有 Android/aarch64 的 wheel（实测 pip install --dry-run openai 报
"Target triple not supported by rustup"）。0.1.3 起 openai 已降级为可选依赖
（pip install qsmy-deepseek-locator[openai]），本模块就是那条「不装它也能用」的路。

## 为什么可以用鸭子类型喂事件解包

_events_from_chunk 全程用 getattr 访问 chunk 字段，不认任何 SDK 的模型类。
所以这里把每个 SSE 帧 json.loads 之后用 SimpleNamespace 递归包一层就能直接喂进去
（见 _to_namespace）—— 不需要为了「解析一个 chunk」而装上整个 pydantic。

## 与 DeepSeekVisionClient 的已知差别（诚实清单）

- **没有自动重试**：SDK 的 max_retries 在这里不存在。网络抖动要靠调用方自己重试。
- **超时口径不同**：这里是 (连接超时 10s, 读超时 timeout)，读超时是「两次数据之间的静默」
  ——与 SDK 的流式口径一致，只是数值固定成了 10s 连接超时。
- **错误文案不同**：非 200 一律抛 APIError，消息里带 HTTP 状态码与响应体前 300 字符。
- **不写 request / event / reply 这几行日志**：传了 log 也只写 error 一行。
  想要完整日志请用 DeepSeekVisionClient，或者自己在等价位置调 log.write（见 debuglog.py 第 5 条）。

@doc README.md#1-安装
（该文档解决"不装 openai 怎么用这个库、两条客户端怎么选"的问题。）
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import requests

from .client import ChatReply, _accumulate_tool_calls, _events_from_chunk
from .config import DEFAULT_BASE_URL, Settings
from .debuglog import DebugLog
from .errors import APIError
from .request_build import build_request

# 连接超时与读超时分开给：连接该快（10s 足够），读超时按 Settings.timeout
# ——那边默认 300s，因为思考模式下首字节可能要等好几秒，生成慢是常态。
_CONNECT_TIMEOUT = 10.0

# 非 200 时响应体截断长度。够看清服务端的错误 JSON，又不至于把整页 HTML 灌进异常里。
_ERROR_BODY_CHARS = 300


def _to_namespace(obj: Any) -> Any:
    """把 dict/list 递归转成属性可访问的对象（模块头「为什么可以用鸭子类型」那节的实现）。"""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(x) for x in obj]
    return obj


def _short(text: str) -> str:
    """错误响应体压成一行（换行会把「错误：…」这类单行输出冲散）。"""
    flat = (text or "").strip().replace("\n", " ")
    return flat if len(flat) <= _ERROR_BODY_CHARS else flat[:_ERROR_BODY_CHARS] + "..."


class RequestsVisionClient:
    """实现 VisionClient 协议的纯 requests 客户端。

        from qsmy_deepseek_locator import Locator, RequestsVisionClient

        client = RequestsVisionClient(api_key="sk-xxx")
        locator = Locator(client=client, thinking=False)
        result = locator.locate("photo.png", "红色圆形")

    也可以只当传输层用（自己声明工具、自己跑工具循环）：

        reply = client.complete(messages, tools=[...], tool_choice="auto")
        reply.tool_calls   # 已按 index 拼好，arguments 是完整 JSON 串

    Args:
        api_key: 必填。空串直接抛 APIError（**不是** ValueError：调用方兜的是 LocatorError）。
        base_url: 接口地址，默认官方站点；自建代理改这里。
        timeout: 读超时（秒）——「两次数据之间的静默」，不是整轮总时长。
        model: 覆盖模型名；None = 用 settings.model（每次调用都取最新的那份配置）。
        session: 复用连接用。默认自建一个 Session：同一个进程连续发多次请求时，
            省掉重复的 TLS 握手（安卓上这一下能省几百毫秒）。
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 90.0,
        model: str | None = None,
        session: "requests.Session | None" = None,
    ) -> None:
        if not api_key or not str(api_key).strip():
            raise APIError(
                "RequestsVisionClient 需要 api_key（空值不接受）。"
                "给法：RequestsVisionClient(api_key=os.environ['DEEPSEEK_API_KEY'])，"
                "或改用 Locator(client=RequestsVisionClient(api_key=...))。"
            )
        self.api_key = str(api_key).strip()
        # 结尾斜杠去掉：调用方写 https://api.deepseek.com/ 也不会拼出 //chat/completions
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.model = model
        self.session = session or requests.Session()

    @property
    def endpoint(self) -> str:
        """完整的 chat/completions 地址。"""
        return self.base_url + "/chat/completions"

    def stream(
        self,
        messages: list[dict],
        *,
        settings: Settings | None = None,
        stream: bool = True,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        log: DebugLog | None = None,
    ):
        """逐段产出事件（与 DeepSeekVisionClient.stream 同一批事件、同一套字段）。

        收进基类不做的原因是两者共用不了任何实现（一个走 SDK、一个走 socket），
        但**事件契约必须一样**，见 client.py 里那张六种事件的表。
        """
        effective = settings or Settings.from_env()
        if log is not None:
            # 与 SDK 版一样，请求体在**发出去之前**落盘。
            from .config import redacted

            log.write("request", {"settings": redacted(effective), "body": build_request(
                effective, messages, stream=stream, tools=tools, tool_choice=tool_choice
            )})
        kwargs = build_request(
            effective, messages, stream=stream, tools=tools, tool_choice=tool_choice
        )
        for event in self._events(kwargs, effective, log=log):
            if log is not None:
                log.write("event", event)
            yield event

    def _events(self, kwargs: dict, settings: Settings, *, log: DebugLog | None = None):
        """发请求 + 读 SSE + 解包成事件（stream() 的实现体，单独拎出来便于阅读）。"""
        payload = self._payload(kwargs, settings)
        headers = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        try:
            resp = self.session.post(
                self.endpoint,
                headers=headers,
                json=payload,
                stream=True,
                timeout=(_CONNECT_TIMEOUT, self.timeout),
            )
        except Exception as exc:  # noqa: BLE001 - requests 的异常种类太多，统一按本库口径抛
            if log is not None:
                log.write("error", {"type": type(exc).__name__, "message": str(exc)})
            raise APIError(
                f"请求 {self.endpoint} 失败（{type(exc).__name__}: {exc}）"
            ) from exc

        try:
            if resp.status_code != 200:
                raise APIError(
                    f"调用 {payload['model']} 失败（HTTP {resp.status_code}）：{_short(resp.text)}"
                )
            for frame in _iter_sse_frames(resp):
                for event in _events_from_chunk(_to_namespace(frame)):
                    yield event
        except APIError:
            raise
        except Exception as exc:  # noqa: BLE001 - 读流中途断连 / 超时
            if log is not None:
                log.write("error", {"type": type(exc).__name__, "message": str(exc)})
            raise APIError(
                f"读取流式响应失败（{type(exc).__name__}: {exc}）"
            ) from exc
        finally:
            # 一定要关：不关的话连接不会回到连接池，连续调用会一直新建连接。
            resp.close()

    def _payload(self, kwargs: dict, settings: Settings) -> dict:
        """把 build_request 的结果翻译成**裸 HTTP 的请求体**。

        ⚠️ 这里是本模块唯一的翻译层，两个坑都在这一层：
        1. extra_body 是 **openai SDK 的概念**（SDK 会把它的键铺到请求体顶层），
           裸 HTTP 里没有这一层，必须自己铺 —— 库的 thinking 开关正是走 extra_body 传的
           （{"thinking": {"type": "disabled"}}），整个塞进 payload 会变成非法字段。
        2. SDK 会替我们过滤掉值为 None 的可选参数，裸 HTTP 不会：所以下面显式挑键，
           而不是 payload.update(kwargs) —— 带上 "tools": null 这种字段，个别服务会直接 400。
        """
        payload: dict[str, Any] = {
            "model": self.model or kwargs["model"],
            "messages": kwargs["messages"],
            "stream": True,
        }
        extra = kwargs.get("extra_body")
        if isinstance(extra, dict):
            payload.update(extra)
        for key in ("reasoning_effort", "max_tokens", "stream_options", "tools", "tool_choice"):
            if kwargs.get(key) is not None:
                payload[key] = kwargs[key]
        return payload

    def complete(
        self,
        messages: list[dict],
        *,
        settings: Settings | None = None,
        on_event: Any = None,
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        log: Any = None,
    ) -> ChatReply:
        """收完整个流，返回 ChatReply（VisionClient 协议的 complete）。

        tools / tool_choice 原样透传给服务端，模型要调用时结果落在 ChatReply.tool_calls；
        **本方法不执行工具**（与 SDK 版一致，见 client.py 的说明）。
        """
        effective = settings or Settings.from_env()
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_acc: dict[int, dict] = {}
        finish_reason: str | None = None
        usage: dict | None = None
        model_name = self.model or effective.model

        for event in self.stream(
            messages, settings=effective, tools=tools, tool_choice=tool_choice, log=log
        ):
            kind = event.get("type")
            if kind == "content":
                text_parts.append(event.get("text") or "")
            elif kind == "reasoning":
                reasoning_parts.append(event.get("text") or "")
            elif kind == "usage":
                usage = event.get("usage")
            elif kind == "model":
                model_name = event.get("model") or model_name
            elif kind == "tool_call":
                _accumulate_tool_calls(tool_acc, event)
            elif kind == "finish":
                finish_reason = event.get("reason")
            if on_event is not None:
                on_event(event)

        reply = ChatReply(
            text="".join(text_parts),
            reasoning="".join(reasoning_parts),
            model=model_name,
            finish_reason=finish_reason,
            usage=usage,
            tool_calls=[tool_acc[index] for index in sorted(tool_acc)],
        )
        if log is not None:
            log.write("reply", reply)
        if not reply.text.strip() and not reply.tool_calls:
            # 与 SDK 版共用同一个判据与提示语：空正文的成因（思考吃光预算）与传输层无关，
            # 只不过本客户端拿不到 finish_reason 以外的细节，_empty_hint 已经覆盖这两种情形。
            from .client import _empty_hint

            hint = _empty_hint(reply)
            if log is not None:
                log.write("error", {"type": "EmptyResponseError", "message": hint})
            from .errors import EmptyResponseError

            raise EmptyResponseError(hint)
        return reply


def _iter_sse_frames(resp: Any):
    """把 requests 的响应逐行读成 SSE 帧的 dict，跳过心跳与解析不了的行。

    只认 "data:" 行：注释行（以 : 开头）是服务端心跳，event:/id: 这些字段 DeepSeek 不发，
    真发了也不影响内容 —— 内容永远是 data。
    decode_unicode=False 是刻意的：让 requests 交原始字节，我们自己按 UTF-8 解，
    免得它按响应头里没写对的中文编码去猜。
    """
    for raw in resp.iter_lines(decode_unicode=False):
        if not raw:
            continue
        line = raw.decode("utf-8", "replace").strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            # 单个帧坏了不该让整轮识别失败：服务端偶尔会发非 JSON 的保活负载。
            continue


__all__ = ["RequestsVisionClient"]
