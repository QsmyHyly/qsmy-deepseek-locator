"""流式事件解包：把 SDK 吐出来的 chunk / completion 拆成统一的六种事件。

**职责**：纯解包 —— 从 chunk 里取 reasoning_content / content / tool_calls / usage / model /
finish_reason，拼成 {"type": ...} 事件；以及把「逐字符」吐出来的工具调用按 index 拼回完整对象。

**边界**：本模块不认识 Settings、不发请求、也不决定什么时候写日志。
它只接受「已经拿到的一串 chunk」，所以 client.py 的 stream() 能用，
自测也能直接喂假 chunk（tests/test_client_stream.py 就是这么测的）。
model 事件的去重放在这里，是因为那是「一次流」的性质，不是某个客户端的性质。

与拆分前旧文件的对应关系（0.1.2 -> 0.1.3 的等价重构，行为零变化）：
    这六个函数原本定义在 client.py 第 444-591 行，现在原样搬到这里；
    client.py 反过来从本模块 import，所以
    「from qsmy_deepseek_locator.client import _events_from_chunk」这类老路径照旧可用。

⚠️ 事件名与字段是**对外契约**：on_event 收到的六种事件（reasoning / content / tool_call /
finish / usage / model），README 第 7 节列了表，改名等于破坏兼容。各字段为什么要这样，
见下面各函数身上保留下来的实测注释。
"""

from __future__ import annotations

from typing import Any, Iterator

from .debuglog import DebugLog


def _tool_call_events(tool_calls: Any) -> Iterator[dict]:
    """把 tool_calls 拆成 {"type": "tool_call", ...} 事件。

    ⚠️ 流式下的工具调用是**逐字符**吐的（实测 deepseek-flash 会把 `{"city": "北京"}`
    拆成十几个 chunk，一个 chunk 一个字符；换成本库那个 crop_region 工具则是 47 个分片）：
    第一个分片带 id 与函数名，之后每个分片只带 arguments 的一小段，
    最后一个分片同时带 finish_reason="tool_calls"。
    所以事件里的 arguments 是**增量**，要拿到完整参数得自己按 index 拼
    —— complete() 里的 _accumulate_tool_calls 干的就是这件事。

    @doc docs/API-NOTES.md#61-工具调用也是流式的而且一个字符一个-chunk
    （该文档解决"分片的字段为什么后面会变 null、finish_reason 为什么是 tool_calls"的问题。）

    非流式响应走的是同一个函数：那时 arguments 已经是完整串、index 是 None，
    这里用列表下标补上，于是两条路径的累积逻辑可以完全一样。
    """
    if not tool_calls:
        return
    for position, call in enumerate(tool_calls):
        index = getattr(call, "index", None)
        call_id = getattr(call, "id", None)
        function = getattr(call, "function", None)
        name = getattr(function, "name", None) if function is not None else None
        arguments = getattr(function, "arguments", None) if function is not None else None
        if call_id is None and name is None and not arguments:
            continue  # 全空占位分片（有些服务会发），不发事件
        yield {
            "type": "tool_call",
            "index": position if index is None else index,
            "id": call_id,
            "name": name,
            "arguments": arguments or "",
        }


def _accumulate_tool_calls(acc: dict[int, dict], event: dict) -> None:
    """把 tool_call 事件按 index 拼成完整的工具调用对象（原地修改 acc）。

    合并规则：id / name 只在第一次出现时记下（后面对应字段是 null，别用 null 覆盖掉），
    arguments 一律**追加**。
    """
    index = event.get("index", 0)
    slot = acc.setdefault(index, {
        "id": None,
        "type": "function",
        "function": {"name": None, "arguments": ""},
    })
    if event.get("id"):
        slot["id"] = event["id"]
    if event.get("name"):
        slot["function"]["name"] = event["name"]
    slot["function"]["arguments"] += event.get("arguments") or ""


def _stream_events(chunks: Any, log: DebugLog | None) -> Iterator[dict]:
    """把 chunk 流拆成事件，并顺手（可选）把**原始 chunk** 也记进日志。

    model 去重放在这里：服务端在**每一个** chunk 上都带 model 字段，照单全收的话
    一次调用会甩出上百条一模一样的 model 事件，把 on_event 刷屏
    （第一次跑 examples/stream_events.py 就是这么发现的）。模型名不会中途改变，
    因此只在第一次出现时发一条。
    """
    last_model: str | None = None
    for chunk in chunks:
        if log is not None and log.chunks:
            log.write("chunk", chunk)
        for event in _events_from_chunk(chunk):
            if event.get("type") == "model":
                if event.get("model") == last_model:
                    continue
                last_model = event.get("model")
            yield event


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

    tool_calls = getattr(delta, "tool_calls", None) if delta is not None else None
    yield from _tool_call_events(tool_calls)

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
    tool_calls = getattr(message, "tool_calls", None) if message is not None else None
    yield from _tool_call_events(tool_calls)
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
    "_tool_call_events",
    "_accumulate_tool_calls",
    "_stream_events",
    "_events_from_chunk",
    "_events_from_completion",
    "_usage_dict",
]
