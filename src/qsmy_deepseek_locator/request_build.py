"""报文拼装：把「一次定位」变成 DeepSeek Chat Completions 能收下的请求体。

**职责**：纯函数拼报文 —— image_part / thinking_payload / resolve_thinking /
build_messages / build_request。它们不碰网络、不碰 SDK，所以自测可以直接断言
「报文长什么样」而不必打桩整个客户端（见 tests/test_locate.py 的 TestMessageBuilding）。

**边界**：本模块只管「请求长什么样」，不管发出去、也不管收回来。
发起请求、拆流式 chunk、拼回 ChatReply 都在 client.py；事件的形状在 stream_events.py。

与拆分前旧文件的对应关系（0.1.2 -> 0.1.3 的等价重构，行为零变化）：
    这五个函数原本定义在 client.py 第 100-201 行，现在原样搬到这里；
    client.py 反过来从本模块 import，所以
    「from qsmy_deepseek_locator.client import build_request」这条老路径照旧可用 ——
    外部（含 App 项目 vendor 的镜像）看不到任何差别。

⚠️ 报文形状是**契约**，不是随手写的：thinking 只能塞进 extra_body、reasoning_effort 反倒是顶层、
stream_options 只在流式时带 —— 这些都是实测出来的。
@doc docs/API-NOTES.md#4-思考模式与-reasoning_effort
（该文档解决"thinking 参数为什么必须走 extra_body、effort 各档怎么映射"的问题。）
"""

from __future__ import annotations

from typing import Any

from .config import IMAGE_DETAILS, REASONING_EFFORTS, Settings


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


def build_request(
    settings: Settings,
    messages: list[dict],
    *,
    stream: bool = True,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
) -> dict:
    """拼出 chat.completions.create 的关键字参数（不含客户端本身）。

    单独抽出来是为了让自测能直接断言「报文长什么样」，不用打桩整个 SDK。

    tools / tool_choice 是**原样透传**给服务端的：本库不校验工具 schema，
    也不替你执行工具（那是调用方的事，见 stream() 的 docstring）。传了才会带上这两个字段。
    """
    enabled, effort = resolve_thinking(settings, None, None)
    kwargs: dict[str, Any] = {
        "model": settings.model,
        "messages": messages,
        "stream": stream,
    }
    if tools:
        kwargs["tools"] = tools
    if tool_choice:
        kwargs["tool_choice"] = tool_choice
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

__all__ = [
    "image_part",
    "thinking_payload",
    "resolve_thinking",
    "build_messages",
    "build_request",
]
