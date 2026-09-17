"""裸 HTTP 客户端（P0-2）的自测：不装 openai 也能把整条链路跑完。

反馈文档 P0-2 的结论是「openai 是硬依赖，安卓上装不上」——它依赖 jiter / pydantic-core
两个 Rust 扩展，二者都没有 Android/aarch64 wheel。修法是把 openai 降级为可选依赖，
并把安卓 App 里那个已经端到端跑通的 requests 实现收编进库（http_client.py）。

所以这一组要验三件事：
    1. 报文翻译对不对（extra_body 铺平、None 不进请求体）—— 这是裸 HTTP 唯一的翻译层；
    2. SSE 收包 / 解包 / 拼 ChatReply 与 SDK 版语义一致（含工具调用逐字符分片）；
    3. **真的在"没有 openai"的进程里 import 本库不炸**（子进程模拟，见最后那组）。

全程不联网：session.post 被打桩成「返回一串预设的 SSE 行」。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from qsmy_deepseek_locator import RequestsVisionClient
from qsmy_deepseek_locator.config import Settings
from qsmy_deepseek_locator.errors import (
    APIError,
    EmptyResponseError,
    LocatorError,
    UnsupportedFeatureError,
)
from qsmy_deepseek_locator.request_build import build_request

# 子进程测试要显式把 src 挂到 PYTHONPATH 上（子进程不会读 tests/conftest.py）
SRC = Path(__file__).resolve().parents[1] / "src"


class _FakeResponse:
    """够用就行的假响应：只实现被读到的三个成员。"""

    def __init__(self, lines, status_code=200, text=""):
        self._lines = [line.encode("utf-8") if isinstance(line, str) else line
                       for line in lines]
        self.status_code = status_code
        self.text = text
        self.closed = False

    def iter_lines(self, decode_unicode=False):
        assert decode_unicode is False, "客户端必须自己按 UTF-8 解码（见 _iter_sse_frames）"
        return iter(self._lines)

    def close(self):
        self.closed = True


def _sse(*frames) -> list:
    """把若干 dict 变成 SSE 行，并以 [DONE] 收尾（与真实响应形状一致）。"""
    return ["data: " + json.dumps(f, ensure_ascii=False) for f in frames] + ["data: [DONE]"]


def _delta(content=None, reasoning=None, calls=None, finish=None, model=None):
    """一个流式 data 帧（形状照抄真实响应：choices[0].delta）。"""
    frame = {"choices": [{"delta": {}, "finish_reason": finish}]}
    if content is not None:
        frame["choices"][0]["delta"]["content"] = content
    if reasoning is not None:
        frame["choices"][0]["delta"]["reasoning_content"] = reasoning
    if calls is not None:
        frame["choices"][0]["delta"]["tool_calls"] = calls
    if model is not None:
        frame["model"] = model
    return frame


@pytest.fixture
def capture(monkeypatch):
    """返回 (造客户端 + 截住 HTTP 请求) 的工厂，以及请求记录列表。"""
    calls = []

    def _make(lines, status_code=200, text=""):
        client = RequestsVisionClient(api_key="sk-test")

        def _post(url, headers=None, json=None, stream=None, timeout=None):
            calls.append({"url": url, "headers": headers, "json": json,
                          "stream": stream, "timeout": timeout})
            response = _FakeResponse(lines, status_code=status_code, text=text)
            calls[-1]["response"] = response
            return response

        monkeypatch.setattr(client.session, "post", _post)
        return client

    return _make, calls


def _reply(client, **kwargs):
    return client.complete([{"role": "user", "content": "hi"}],
                           settings=Settings(api_key="sk-test"), **kwargs)


class TestPayloadTranslation:
    """裸 HTTP 唯一的翻译层：openai SDK 的 kwargs -> 真正的请求体。"""

    def test_extra_body_is_flattened_into_the_payload(self, capture):
        """thinking 装在 extra_body 里（SDK 概念），裸 HTTP 必须自己铺到顶层。

        不铺的后果不是报错而是**静默失效**：服务端收不到 thinking，思考模式按默认走，
        调用方以为自己关掉了思考、还在为它付 token。
        """
        make, _calls = capture
        client = make(_sse())
        settings = Settings(api_key="sk-test", thinking=False)
        kwargs = build_request(settings, [{"role": "user", "content": "hi"}])
        payload = client._payload(kwargs, settings)
        assert payload["thinking"] == {"type": "disabled"}
        assert "extra_body" not in payload

    def test_none_optional_fields_are_omitted(self, capture):
        """值为 None 的可选参数不许进请求体（SDK 会替我们过滤，裸 HTTP 不会）。"""
        make, _calls = capture
        client = make(_sse())
        settings = Settings(api_key="sk-test")       # 全部可选参数都是 None
        kwargs = build_request(settings, [{"role": "user", "content": "hi"}])
        payload = client._payload(kwargs, settings)
        for key in ("tools", "tool_choice", "max_tokens", "reasoning_effort"):
            assert key not in payload, key

    def test_tools_and_tool_choice_reach_the_server(self, capture):
        """P1-8 的正向对照：签名里叫得出来，参数就真的发得出去。"""
        make, calls = capture
        client = make(_sse(_delta(content="ok", finish="stop")))
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        _reply(client, tools=tools, tool_choice="auto")
        assert calls[0]["json"]["tools"] == tools
        assert calls[0]["json"]["tool_choice"] == "auto"

    def test_auth_header_and_endpoint(self, capture):
        make, calls = capture
        client = make(_sse(_delta(content="ok")))
        _reply(client)
        assert calls[0]["url"] == "https://api.deepseek.com/chat/completions"
        assert calls[0]["headers"]["Authorization"] == "Bearer sk-test"
        assert calls[0]["stream"] is True
        # 连接超时与读超时分开给（连接用固定的 10s，读用配置的 timeout）
        assert calls[0]["timeout"] == (10.0, client.timeout)


class TestStreamingAssembly:
    def test_content_and_reasoning_are_concatenated(self, capture):
        make, _calls = capture
        client = make(_sse(
            _delta(reasoning="先想"),
            _delta(reasoning="一下"),
            _delta(content="你好"),
            _delta(content="世界"),
            _delta(finish="stop", model="deepseek-flash"),
        ))
        reply = _reply(client)
        assert reply.text == "你好世界"
        assert reply.reasoning == "先想一下"
        assert reply.finish_reason == "stop"
        assert reply.model == "deepseek-flash"

    def test_tool_call_fragments_are_reassembled(self, capture):
        """工具调用是逐字符吐的（实测），裸 HTTP 这条路也必须拼回完整 JSON。"""
        make, _calls = capture
        frames = [_delta(calls=[{"index": 0, "id": "call_1",
                                 "function": {"name": "get_weather", "arguments": ""}}])]
        frames += [_delta(calls=[{"index": 0, "function": {"arguments": ch}}])
                   for ch in '{"city": "北京"}']
        frames.append(_delta(finish="tool_calls"))
        client = make(_sse(*frames))
        reply = _reply(client)
        assert reply.tool_calls[0]["function"]["name"] == "get_weather"
        assert json.loads(reply.tool_calls[0]["function"]["arguments"]) == {"city": "北京"}
        # 有话全说在工具调用里时，空正文**不算错误**（与 SDK 版边界一致）
        assert reply.text == ""

    def test_usage_event_is_picked_up(self, capture):
        make, _calls = capture
        client = make(_sse(
            _delta(content="ok"),
            {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 2,
                                      "total_tokens": 13}},
        ))
        assert _reply(client).usage["total_tokens"] == 13

    def test_on_event_sees_the_same_kinds(self, capture):
        make, _calls = capture
        client = make(_sse(_delta(reasoning="r"), _delta(content="c"), _delta(finish="stop")))
        seen = []
        _reply(client, on_event=seen.append)
        assert [event["type"] for event in seen] == ["reasoning", "content", "finish"]

    def test_response_is_closed_after_a_normal_read(self, capture):
        """收完流必须 close（还回连接池），否则连续调用会一直新建连接。"""
        make, calls = capture
        client = make(_sse(_delta(content="ok")))
        _reply(client)
        assert calls[0]["response"].closed is True

    def test_response_is_closed_and_error_wrapped_when_the_stream_breaks(self, capture):
        """**读流中途炸**（连接被掐）：要转成 APIError，而 finally 仍必须把响应关掉。

        上一版这条其实没生效：它把 _payload 打桩成抛异常，而 _payload 跑在
        session.post **之前** —— 请求压根没发出去，calls2 是空的，finally 那条路
        一个断言都没走到（假绿）。现在改成让 iter_lines **读到一半**才抛。
        """
        make, calls = capture
        client = make([])

        class _BrokenResponse(_FakeResponse):
            """吐一帧之后断线 —— 这就是「读流中途失败」的真实形状。"""

            def iter_lines(self, decode_unicode=False):
                yield b"data: " + json.dumps(_delta(content="前")).encode("utf-8")
                raise ConnectionError("连接被对端掐断")

        broken = _BrokenResponse([])

        def _post(url, headers=None, json=None, stream=None, timeout=None):
            calls.append({"url": url, "json": json, "response": broken})
            return broken

        # 直接替换这个客户端实例上的 post：这一次请求真的会发出去（calls 里有记录），
        # 失败点落在**读流**那一段，正是要覆盖的那条路。
        client.session.post = _post  # type: ignore[method-assign]
        with pytest.raises(APIError) as excinfo:
            _reply(client)
        assert "读取流式响应失败" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, ConnectionError)
        assert broken.closed is True, "读流失败也必须 close（finally 那条路）"


class TestSseRobustness:
    def test_comments_and_broken_frames_are_skipped(self, capture):
        """心跳行（以 : 开头）与坏帧不该让整轮识别失败。"""
        make, _calls = capture
        lines = [": keep-alive", "",
                 "data: {这不是合法 JSON",
                 "data: " + json.dumps(_delta(content="ok", finish="stop")),
                 "data: [DONE]",
                 "data: " + json.dumps(_delta(content="DONE 之后的不该被读到"))]
        client = make(lines)
        assert _reply(client).text == "ok"

    def test_event_prefix_is_ignored(self, capture):
        make, _calls = capture
        lines = ["event: message",
                 "data: " + json.dumps(_delta(content="ok")),
                 "data: [DONE]"]
        assert _reply(make(lines)).text == "ok"

    def test_utf8_is_decoded_by_us(self, capture):
        """中文必须原样还原 —— 客户端刻意让 requests 交原始字节，自己按 UTF-8 解。"""
        make, _calls = capture
        client = make(_sse(_delta(content="蓝色方块")))
        assert _reply(client).text == "蓝色方块"


class TestUnsupportedStreamOff:
    """stream=False 必须**明确报错**，不许静默忽略。

    上一版这个形参只进了 build_request，被 _payload 里写死的 {"stream": True} 覆盖 ——
    传 False 既不报错也不生效，是"参数看起来能用其实没用"的那类坑。
    """

    def test_stream_false_is_rejected_loudly(self, capture):
        make, calls = capture
        client = make(_sse(_delta(content="ok")))
        with pytest.raises(UnsupportedFeatureError) as excinfo:
            list(client.stream([{"role": "user", "content": "hi"}], stream=False))
        assert "只支持流式" in str(excinfo.value)
        assert calls == [], "报错要在发请求之前，别白花一次 API"

    def test_stream_true_still_works(self, capture):
        """负向对照：默认值与显式 True 都必须照常工作。"""
        make, _calls = capture
        client = make(_sse(_delta(content="ok")))
        events = list(client.stream([{"role": "user", "content": "hi"}], stream=True))
        assert any(event["type"] == "content" for event in events)


class TestErrors:
    def test_http_error_becomes_api_error_with_the_body(self, capture):
        make, _calls = capture
        client = make([], status_code=401, text='{"error": {"message": "Authentication Fails"}}')
        with pytest.raises(APIError) as excinfo:
            _reply(client)
        assert "401" in str(excinfo.value)
        assert "Authentication Fails" in str(excinfo.value)

    def test_connection_failure_is_wrapped_with_cause(self, monkeypatch):
        client = RequestsVisionClient(api_key="sk-test")

        def _boom(*args, **kwargs):
            raise ConnectionError("网线被拔了")

        monkeypatch.setattr(client.session, "post", _boom)
        with pytest.raises(APIError) as excinfo:
            _reply(client)
        assert isinstance(excinfo.value.__cause__, ConnectionError)

    def test_empty_text_without_tool_calls_raises(self, capture):
        """与 SDK 版同一条边界：正文与思考都空时要当场说清，不能当成"图里没目标"。"""
        make, _calls = capture
        client = make(_sse(_delta(finish="stop")))
        with pytest.raises(EmptyResponseError):
            _reply(client)

    def test_empty_api_key_is_a_locator_error(self):
        """空 Key 也走本库的异常体系 —— 调用方兜的是 LocatorError。"""
        with pytest.raises(LocatorError):
            RequestsVisionClient(api_key="   ")


class TestWorksWithoutOpenAI:
    """P0-2 的核心断言：**在一个装不上 openai 的环境里，这个库照样能用**。

    用子进程真的把 openai 屏蔽掉，而不是在进程内 monkeypatch —— 进程内改 sys.modules
    只能证明"这一次 import 恰好走了另一条路"，证不了"这个包在没装 openai 的机器上
    装得上、import 得进"。安卓那条反馈要的正是后者。
    """

    SCRIPT = r'''
import sys

class _Blocker:
    """拦住任何 openai 的 import —— 模拟安卓上"根本装不上"的机器。"""
    def find_spec(self, name, path=None, target=None):
        if name == "openai" or name.startswith("openai."):
            raise ImportError("No module named 'openai'（本进程刻意屏蔽它）")
        return None

sys.meta_path.insert(0, _Blocker())
sys.modules.pop("openai", None)

# 1) 顶层包必须能 import
import qsmy_deepseek_locator as lib
assert lib.RequestsVisionClient is not None, "裸 HTTP 客户端必须能直接用"

# 2) 不需要 openai 的公开面照旧可用（含画图、解析、配置）
from qsmy_deepseek_locator import Locator, Settings, draw, Detection, locate_to_file
assert Settings(api_key="sk-x").model
assert Locator(client=lib.RequestsVisionClient(api_key="sk-x")).settings is not None

# 3) **构造** SDK 客户端不该失败：SDK 的 import 是懒的（在 client_for 里），
#    只想用解析/打标、或者只想 new 一个 Locator 的调用方不该被环境问题绊倒。
lib.DeepSeekVisionClient(Settings(api_key="sk-x"))

# 4) 真的要用 SDK 那条路时才失败，而且报错必须给出两条出路
try:
    lib.DeepSeekVisionClient(Settings(api_key="sk-x")).complete(
        [{"role": "user", "content": "hi"}])
except Exception as exc:
    text = str(exc)
    assert "[openai]" in text, text
    assert "RequestsVisionClient" in text, text
else:
    raise AssertionError("没有 openai 却真的把请求发出去了？")

print("NO_OPENAI_OK")
'''

    def test_import_and_use_without_openai(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.run(
            [sys.executable, "-c", self.SCRIPT],
            capture_output=True, text=True, encoding="utf-8", env=env, timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        assert "NO_OPENAI_OK" in proc.stdout

    def test_client_for_error_points_at_both_ways_out(self, monkeypatch):
        """同一条契约在**本进程**里也成立：缺 openai 的报错要给出两条路。

        子进程那条证明"能装上"，这条证明"报错文案对"——两者缺一不可：
        只看子进程的话，文案退化成「pip install openai」也照样绿。
        """
        import builtins
        from qsmy_deepseek_locator.client import DeepSeekVisionClient

        real_import = builtins.__import__

        def _blocked(name, *args, **kwargs):
            if name == "openai" or name.startswith("openai."):
                raise ImportError("No module named 'openai'（本次测试刻意屏蔽）")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocked)
        with pytest.raises(APIError) as excinfo:
            DeepSeekVisionClient(Settings(api_key="sk-test")).client_for(
                Settings(api_key="sk-test"))
        message = str(excinfo.value)
        assert "qsmy-deepseek-locator[openai]" in message
        assert "RequestsVisionClient" in message
