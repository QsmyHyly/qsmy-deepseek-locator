"""客户端流式事件的自测：六种事件各自长什么样、工具调用怎么拼回来。

不联网：`DeepSeekVisionClient._create` 被打桩成「返回一串假的 chunk」，
这样从 build_request -> 拆 chunk -> 拼 ChatReply 的整条链路都能离线跑。

关于工具调用的分片形状，事实来自一次真机探测（deepseek-flash / 流式 / 带 tools）：
    第一个分片 {"index": 0, "id": "call_00_xx", "function": {"name": "get_weather", "arguments": ""}}
    之后每个分片 {"index": 0, "id": null, "function": {"arguments": "<一个字符>"}}
    最后一个分片 {"finish_reason": "tool_calls"}
也就是「一个字符一个 chunk」，所以下面的假数据故意拆得很碎。
"""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest

from qsmy_deepseek_locator.client import (
    ChatReply,
    DeepSeekVisionClient,
    _accumulate_tool_calls,
    _events_from_chunk,
    _events_from_completion,
    _tool_call_events,
    build_request,
)
from qsmy_deepseek_locator.config import Settings
from qsmy_deepseek_locator.errors import EmptyResponseError


# --------------------------------------------------------------------------- #
# 伪造 SDK 对象（只实现被读到的字段）
# --------------------------------------------------------------------------- #
def _call(index=None, id=None, name=None, arguments=None):
    """一个 delta.tool_calls 元素。"""
    return NS(index=index, id=id, type="function" if id else None,
              function=NS(name=name, arguments=arguments))


def _chunk(*, calls=None, content=None, reasoning=None, finish=None, usage=None, model=None):
    """一个流式 chunk。"""
    delta = NS(content=content, reasoning_content=reasoning, tool_calls=calls)
    return NS(choices=[NS(delta=delta, finish_reason=finish)], usage=usage, model=model)


def _usage_chunk(**counts):
    """只带 usage 的收尾 chunk —— choices 是空数组，直接取 choices[0] 会 IndexError。"""
    return NS(choices=[], usage=NS(**counts), model=None)


WEATHER_FRAGMENTS = ['{', '"', 'city', '"', ':', ' ', '"', '北京', '"', '}']


def _tool_call_chunks():
    """一次真实形状的工具调用流：第一个分片带 id/name，剩下逐字符吐参数。"""
    yield _chunk(calls=[_call(index=0, id="call_00_abc", name="get_weather", arguments="")])
    for piece in WEATHER_FRAGMENTS:
        yield _chunk(calls=[_call(index=0, arguments=piece)])
    yield _chunk(content="", finish="tool_calls")


def _client_with(monkeypatch, chunks):
    """造一个客户端，并把它发请求那一步换成「返回预设 chunk」。"""
    client = DeepSeekVisionClient(Settings(api_key="test-key"))
    monkeypatch.setattr(client, "_create", lambda kwargs, settings: iter(chunks))
    return client


# --------------------------------------------------------------------------- #
# 事件层
# --------------------------------------------------------------------------- #
def test_tool_call_events_carry_id_name_and_argument_delta():
    events = list(_tool_call_events([
        _call(index=0, id="call_1", name="get_weather", arguments=""),
        _call(index=0, arguments='{"ci'),
    ]))
    assert events[0] == {
        "type": "tool_call", "index": 0, "id": "call_1",
        "name": "get_weather", "arguments": "",
    }
    # 后面的分片只有 arguments，id / name 是 None —— 累积时**不能**覆盖已有的值
    assert events[1] == {"type": "tool_call", "index": 0, "id": None, "name": None,
                         "arguments": '{"ci'}


def test_tool_call_events_skip_empty_placeholder():
    assert list(_tool_call_events([_call()])) == []
    assert list(_tool_call_events(None)) == []
    assert list(_tool_call_events([])) == []


def test_tool_call_events_index_falls_back_to_position():
    """非流式响应没有 index，用列表下标补，两条路径才能共用累积逻辑。"""
    events = list(_tool_call_events([
        _call(id="a", name="f1", arguments="{}"),
        _call(id="b", name="f2", arguments="{}"),
    ]))
    assert [e["index"] for e in events] == [0, 1]


def test_accumulate_appends_arguments_and_keeps_first_id():
    acc: dict[int, dict] = {}
    for event in _tool_call_events([_call(index=0, id="call_1", name="get_weather", arguments="")]):
        _accumulate_tool_calls(acc, event)
    for piece in WEATHER_FRAGMENTS:
        for event in _tool_call_events([_call(index=0, arguments=piece)]):
            _accumulate_tool_calls(acc, event)
    call = acc[0]
    assert call["id"] == "call_1"
    assert call["type"] == "function"
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "北京"}


def test_chunk_with_tool_calls_yields_tool_call_event():
    events = list(_events_from_chunk(
        _chunk(calls=[_call(index=0, id="c1", name="f", arguments="{}")])
    ))
    assert [e["type"] for e in events] == ["tool_call"]


def test_completion_path_emits_full_arguments():
    """非流式：message.tool_calls 里的 arguments 已经是完整 JSON 串。"""
    message = NS(content=None, reasoning_content=None, tool_calls=[
        NS(index=None, id="c1", type="function",
           function=NS(name="get_weather", arguments='{"city": "北京"}')),
    ])
    completion = NS(choices=[NS(message=message, finish_reason="tool_calls")],
                    usage=None, model="fake")
    events = list(_events_from_completion(completion))
    calls = [e for e in events if e["type"] == "tool_call"]
    assert len(calls) == 1
    assert calls[0]["index"] == 0  # index 为空时按列表下标补
    assert calls[0]["name"] == "get_weather"
    assert calls[0]["arguments"] == '{"city": "北京"}'


# --------------------------------------------------------------------------- #
# 报文：tools / tool_choice 只是原样透传
# --------------------------------------------------------------------------- #
CROP_TOOL = {"type": "function", "function": {"name": "crop_region", "parameters": {}}}


def test_build_request_passes_tools_through():
    kwargs = build_request(Settings(api_key="k"), [{"role": "user", "content": "hi"}],
                           tools=[CROP_TOOL], tool_choice="auto")
    assert kwargs["tools"] == [CROP_TOOL]
    assert kwargs["tool_choice"] == "auto"


def test_build_request_omits_tools_when_absent():
    """没传就不该出现这两个字段 —— 带上空的 tools 数组有的服务会 400。"""
    kwargs = build_request(Settings(api_key="k"), [{"role": "user", "content": "hi"}])
    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs


def test_complete_forwards_tools_to_request(monkeypatch):
    captured: dict = {}
    client = DeepSeekVisionClient(Settings(api_key="test-key"))

    def fake_create(kwargs, settings):
        captured.update(kwargs)
        return iter([_chunk(content="ok", finish="stop")])

    monkeypatch.setattr(client, "_create", fake_create)
    client.complete([{"role": "user", "content": "hi"}], tools=[CROP_TOOL])
    assert captured["tools"] == [CROP_TOOL]


# --------------------------------------------------------------------------- #
# 整条链路：stream() / complete()
# --------------------------------------------------------------------------- #
def test_stream_yields_all_six_event_kinds(monkeypatch):
    chunks = [
        _chunk(reasoning="我想想"),
        _chunk(content="答案"),
        *_tool_call_chunks(),
        _usage_chunk(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    ]
    client = _client_with(monkeypatch, chunks)
    events = list(client.stream([{"role": "user", "content": "hi"}]))
    kinds = [e["type"] for e in events]
    assert kinds.count("reasoning") == 1
    assert kinds.count("content") >= 1
    assert kinds.count("tool_call") == len(WEATHER_FRAGMENTS) + 1
    assert kinds.count("finish") == 1
    assert kinds.count("usage") == 1
    assert kinds.count("model") == 0  # 假数据没给 model 字段


def test_model_event_is_emitted_once(monkeypatch):
    """服务端每个 chunk 都带 model，去重后只该出现一次（否则 on_event 被刷屏）。"""
    chunks = [_chunk(model="deepseek-flash") for _ in range(5)]
    chunks.append(_chunk(content="好", model="deepseek-flash"))
    client = _client_with(monkeypatch, chunks)
    events = list(client.stream([{"role": "user", "content": "hi"}]))
    assert [e["type"] for e in events].count("model") == 1


def test_usage_only_chunk_does_not_crash(monkeypatch):
    """choices 为空数组是合法形状，取 choices[0] 会 IndexError —— 这条守住它。"""
    client = _client_with(monkeypatch, [_usage_chunk(prompt_tokens=1, total_tokens=2)])
    events = list(client.stream([{"role": "user", "content": "hi"}]))
    assert events == [{"type": "usage", "usage": {"prompt_tokens": 1, "total_tokens": 2}}]


def test_complete_rebuilds_tool_calls(monkeypatch):
    client = _client_with(monkeypatch, list(_tool_call_chunks()))
    reply = client.complete([{"role": "user", "content": "北京天气"}])
    assert reply.text == ""
    assert reply.finish_reason == "tool_calls"
    assert len(reply.tool_calls) == 1
    assert reply.tool_calls[0]["id"] == "call_00_abc"
    assert reply.tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(reply.tool_calls[0]["function"]["arguments"]) == {"city": "北京"}


def test_empty_text_with_tool_calls_is_not_an_error(monkeypatch):
    """正文为空但有工具调用 = 正常，不能报 EmptyResponseError。"""
    client = _client_with(monkeypatch, list(_tool_call_chunks()))
    reply = client.complete([{"role": "user", "content": "北京天气"}])
    assert isinstance(reply, ChatReply)


def test_empty_text_without_tool_calls_still_raises(monkeypatch):
    """回归护栏：没有工具调用时，空正文照旧报错。"""
    client = _client_with(monkeypatch, [_chunk(content="", finish="stop")])
    with pytest.raises(EmptyResponseError):
        client.complete([{"role": "user", "content": "hi"}])


def test_parallel_tool_calls_are_kept_separate(monkeypatch):
    """两个工具调用交错分片时按 index 各归各位，顺序稳定。"""
    chunks = [
        _chunk(calls=[_call(index=0, id="c0", name="f0", arguments="")]),
        _chunk(calls=[_call(index=1, id="c1", name="f1", arguments="")]),
        _chunk(calls=[_call(index=0, arguments="{}"), _call(index=1, arguments="{}")]),
        _chunk(content="", finish="tool_calls"),
    ]
    reply = _client_with(monkeypatch, chunks).complete([{"role": "user", "content": "hi"}])
    assert [(c["id"], c["function"]["name"], c["function"]["arguments"])
            for c in reply.tool_calls] == [("c0", "f0", "{}"), ("c1", "f1", "{}")]


def test_on_event_sees_tool_call_events(monkeypatch):
    seen: list[dict] = []
    client = _client_with(monkeypatch, list(_tool_call_chunks()))
    client.complete([{"role": "user", "content": "hi"}], on_event=seen.append)
    assert [e["type"] for e in seen].count("tool_call") == len(WEATHER_FRAGMENTS) + 1


def test_reasoning_and_text_still_accumulate(monkeypatch):
    chunks = [_chunk(reasoning="想"), _chunk(content="好"), _chunk(content="的"),
              _chunk(finish="stop")]
    reply = _client_with(monkeypatch, chunks).complete([{"role": "user", "content": "hi"}])
    assert (reply.reasoning, reply.text, reply.finish_reason) == ("想", "好的", "stop")
    assert reply.tool_calls == []
