"""agent 主循环的离线自测：打桩客户端，不联网、不花 API。

打桩的意义：真实模型的 tool_calls 是**流式分片**回来的（arguments 一个字符一个 chunk），
这一层最容易写错（拼不齐就 json.loads 失败、参数变半个），而真实调用既慢又要钱。
所以这里用假客户端精确控制「分片怎么切」，把拼装逻辑钉住。
"""

from __future__ import annotations

import json

import pytest

from qsmy_deepseek_locator.agent import (
    collect_final_text,
    collect_items,
    extract_items,
    run_agent,
    run_agent_simple,
)
from qsmy_deepseek_locator.config import Settings


class FakeClient:
    """按预设脚本产出事件的假客户端（契约与 DeepSeekVisionClient.stream 一致）。"""

    def __init__(self, rounds):
        self.rounds = rounds
        self.calls: list[dict] = []

    def stream(self, messages, *, settings=None, tools=None, log=None):
        self.calls.append({
            "settings": settings,
            "tools": tools,
            "messages": [dict(m) for m in messages],
        })
        for event in self.rounds[len(self.calls) - 1]:
            yield event


def _split_tool_call(name: str, args: dict, call_id: str = "call_1"):
    """把一次工具调用切成**逐字符**的增量事件（真实服务端就是这么发的）。"""
    raw = json.dumps(args, ensure_ascii=False)
    events = [{"type": "tool_call", "index": 0, "id": call_id, "name": name, "arguments": ""}]
    events += [{"type": "tool_call", "index": 0, "arguments": ch} for ch in raw]
    return events


def _types(events):
    return [e["type"] for e in events]


class TestRunAgentSingleRound:
    def test_no_tool_calls_finishes_in_one_round(self):
        client = FakeClient([[
            {"type": "reasoning", "text": "看图"},
            {"type": "content", "text": "找到了"},
            {"type": "finish", "reason": "stop"},
        ]])
        events = list(run_agent([{"role": "user", "content": "找"}], client=client))
        assert _types(events) == [
            "round_start", "reasoning", "content", "message", "done",
        ]
        done = events[-1]
        assert done["content"] == "找到了"
        assert done["rounds"] == 1
        assert done["reason"] == "stop"

    def test_messages_history_is_appended(self):
        client = FakeClient([[{"type": "content", "text": "x"}, {"type": "finish", "reason": "stop"}]])
        messages = [{"role": "user", "content": "找"}]
        list(run_agent(messages, client=client))
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "x"


class TestToolCallAccumulation:
    def test_fragmented_arguments_are_joined(self):
        """逐字符分片必须能拼回完整 JSON —— 这是本模块最容易写错的一处。"""
        client = FakeClient([
            _split_tool_call("decode_json_points", {"text": '[{"bbox_2d": [0.1, 0.2, 0.3, 0.4], "label": "红"}]'})
            + [{"type": "finish", "reason": "tool_calls"}],
            [{"type": "content", "text": "好了"}, {"type": "finish", "reason": "stop"}],
        ])
        events = list(run_agent([{"role": "user", "content": "找"}], client=client))

        tool_call = next(e for e in events if e["type"] == "tool_call")
        assert tool_call["name"] == "decode_json_points"
        assert json.loads(tool_call["arguments"]) == {
            "text": '[{"bbox_2d": [0.1, 0.2, 0.3, 0.4], "label": "红"}]'
        }

        tool_result = next(e for e in events if e["type"] == "tool_result")
        assert tool_result["ok"] is True
        assert "bbox_2d" in tool_result["content"]
        assert "0.1" in tool_result["content"]          # 真被工具执行了，不是空转

        # 第二轮的请求里必须带上 assistant(tool_calls) 与 role=tool 两条消息
        second = client.calls[1]["messages"]
        assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in second)
        assert any(m.get("role") == "tool" and m.get("tool_call_id") == "call_1" for m in second)

    def test_tool_context_is_injected_and_overrides_model(self):
        client = FakeClient([
            _split_tool_call("get_image_info", {"source": "https://编造的.example/x.png"})
            + [{"type": "finish", "reason": "tool_calls"}],
            [{"type": "content", "text": "ok"}, {"type": "finish", "reason": "stop"}],
        ])
        # 真图：现画一张，避免依赖任何外部文件
        import os, tempfile
        from qsmy_deepseek_locator.bench_generators import make_marker_image
        work = tempfile.mkdtemp(dir=os.path.expanduser("~"))
        img = os.path.join(work, "m.png")
        make_marker_image(img, 320, 240)

        events = list(run_agent(
            [{"role": "user", "content": "找"}],
            client=client,
            tool_context={"source": img},
        ))
        result = next(e for e in events if e["type"] == "tool_result")
        assert result["ok"] is True, result["content"]
        assert "320" in result["content"] and "240" in result["content"]

    def test_max_rounds_stops_the_loop(self):
        """模型每轮都请求工具时，必须在 max_rounds 处收手（不能无限打转）。"""
        loop_round = _split_tool_call("list_palette_colors", {}) + [
            {"type": "finish", "reason": "tool_calls"}
        ]
        client = FakeClient([loop_round] * 10)
        events = list(run_agent(
            [{"role": "user", "content": "找"}], client=client, max_rounds=3,
        ))
        done = events[-1]
        assert done["type"] == "done"
        assert done["reason"] == "max_rounds"
        assert done["rounds"] == 3
        assert len(client.calls) == 3

    def test_use_tools_false_sends_no_tools(self):
        client = FakeClient([[{"type": "content", "text": "x"}, {"type": "finish", "reason": "stop"}]])
        list(run_agent([{"role": "user", "content": "找"}], client=client, use_tools=False))
        assert client.calls[0]["tools"] is None


class TestThinking:
    def test_reasoning_content_is_returned_only_when_thinking_on(self):
        client = FakeClient([
            [{"type": "reasoning", "text": "想"}, {"type": "content", "text": "答"},
             {"type": "finish", "reason": "tool_calls"}],
            [{"type": "content", "text": "完"}, {"type": "finish", "reason": "stop"}],
        ])
        # 第一轮得先有 tool_calls，第二轮的请求里才可能带 reasoning_content
        client.rounds[0] = _split_tool_call("list_palette_colors", {}) + [
            {"type": "reasoning", "text": "想"},
            {"type": "finish", "reason": "tool_calls"},
        ]
        list(run_agent([{"role": "user", "content": "找"}], client=client, thinking=True))
        assert client.calls[1]["messages"][-2].get("reasoning_content") == "想"

    def test_thinking_off_never_sends_reasoning_content(self):
        client = FakeClient([
            _split_tool_call("list_palette_colors", {}) + [
                {"type": "reasoning", "text": "想"}, {"type": "finish", "reason": "tool_calls"}],
            [{"type": "content", "text": "完"}, {"type": "finish", "reason": "stop"}],
        ])
        list(run_agent([{"role": "user", "content": "找"}], client=client, thinking=False))
        for msg in client.calls[1]["messages"]:
            assert "reasoning_content" not in msg

    def test_thinking_is_merged_from_settings_and_per_call(self):
        """按次覆盖必须压过 settings —— 走的是与单轮定位同一个 merge_thinking。

        ⚠️ 按次**关掉**思考时，effort 会被一并清成 None：关闭思考时它无意义，
        发过去也是废字段。所以这里断言 None，而不是继承来的 "high"。
        """
        client = FakeClient([[{"type": "content", "text": "x"}, {"type": "finish", "reason": "stop"}]])
        settings = Settings(thinking=True, reasoning_effort="high")
        list(run_agent([{"role": "user", "content": "找"}], client=client,
                       settings=settings, thinking=False))
        sent = client.calls[0]["settings"]
        assert sent.thinking is False
        assert sent.reasoning_effort is None

    def test_effort_is_merged_when_given(self):
        client = FakeClient([[{"type": "content", "text": "x"}, {"type": "finish", "reason": "stop"}]])
        list(run_agent([{"role": "user", "content": "找"}], client=client,
                       settings=Settings(), reasoning_effort="low"))
        assert client.calls[0]["settings"].reasoning_effort == "low"


class TestErrors:
    def test_client_exception_becomes_error_event(self):
        class Boom:
            def stream(self, messages, *, settings=None, tools=None, log=None):
                raise RuntimeError("网络断了")
                yield  # pragma: no cover

        events = list(run_agent([{"role": "user", "content": "找"}], client=Boom()))
        assert _types(events) == ["round_start", "error"]
        assert "网络断了" in events[-1]["message"]

    def test_tool_failure_is_reported_not_raised(self):
        client = FakeClient([
            _split_tool_call("get_image_info", {"source": "不存在的图.png"})
            + [{"type": "finish", "reason": "tool_calls"}],
            [{"type": "content", "text": "算了"}, {"type": "finish", "reason": "stop"}],
        ])
        events = list(run_agent([{"role": "user", "content": "找"}], client=client))
        result = next(e for e in events if e["type"] == "tool_result")
        assert result["ok"] is False
        assert events[-1]["type"] == "done"      # 工具失败不该中断整条链路


class TestItemCollection:
    def test_extract_items_from_text(self):
        assert extract_items('[{"bbox_2d": [0, 0, 1, 1], "label": "a"}]') == [
            {"bbox_2d": [0, 0, 1, 1], "label": "a"}
        ]
        assert extract_items("") == []
        assert extract_items("没有 JSON") == []

    def test_collect_items_falls_back_to_tool_results(self):
        """模型把坐标交给工具处理时，最终正文里没有 JSON —— 要能回溯工具结果。"""
        events = [
            {"type": "content", "text": "已经帮你解析好了"},
            {"type": "tool_result", "ok": True,
             "content": json.dumps([{"bbox_2d": [0.1, 0.1, 0.5, 0.5], "label": "红"}], ensure_ascii=False)},
        ]
        items = collect_items(events, "已经帮你解析好了")
        assert items == [{"bbox_2d": [0.1, 0.1, 0.5, 0.5], "label": "红"}]

    def test_collect_items_prefers_final_text(self):
        events = [{"type": "tool_result", "ok": True,
                   "content": json.dumps([{"bbox_2d": [0, 0, 0.1, 0.1], "label": "工具"}])}]
        items = collect_items(events, '[{"bbox_2d": [0.2, 0.2, 0.3, 0.3], "label": "正文"}]')
        assert items[0]["label"] == "正文"

    def test_collect_final_text(self):
        messages = [
            {"role": "assistant", "content": "第一轮"},
            {"role": "tool", "content": "工具结果"},
            {"role": "assistant", "content": "最后一轮"},
        ]
        assert collect_final_text(messages) == "最后一轮"
        assert collect_final_text([{"role": "user", "content": "x"}]) == ""

    def test_run_agent_simple_shape(self):
        client = FakeClient([[{"type": "content", "text": '[{"bbox_2d": [0, 0, 1, 1], "label": "a"}]'},
                              {"type": "finish", "reason": "stop"}]])
        out = run_agent_simple("找", client=client)
        assert out["items"] == [{"bbox_2d": [0, 0, 1, 1], "label": "a"}]
        assert out["text"].startswith("[")
        assert any(e["type"] == "done" for e in out["events"])


class TestCollectItemsReadsToolResults:
    """**回归**：collect_items 必须读得懂本库自带工具的返回结果。

    2026-09-18 上游化时这里断过一次：collect_items 改调严格的 to_dict_items，
    而 parse_coordinates 返回的是成对列表（没有 dict），于是**工具模式下永远收集不到坐标**，
    表现为下游 SSE 里没有 annotated 事件、界面上一片空白，而模型其实答得好好的。
    两个仓库的端到端自测同时红，就是从这里来的。

    这一组刻意**不写死那个 JSON 字符串**，而是真的去调工具、把工具的输出喂进去 ——
    否则工具的返回形态一变，测试照样绿，等于没测（那正是漏掉它的原因）。
    """

    def test_parse_coordinates_output_is_readable(self):
        from qsmy_deepseek_locator.tools.builtin import build_default_registry

        registry = build_default_registry()
        result = registry.execute("parse_coordinates", {"text": '[{"bbox_2d": [180, 240, 430, 620]}]'})
        assert result.ok, result.content

        got = collect_items([{"type": "tool_result", "ok": True, "content": result.content}], "")
        assert got, f"工具结果读不出来：{result.content!r}"
        assert got[0]["bbox_2d"] == [0.18, 0.24, 0.43, 0.62]

    def test_final_text_still_wins(self):
        # 正文里已经有坐标时不必去翻工具结果（既有优先级不能被改坏）
        events = [{"type": "tool_result", "ok": True, "content": '[[[0.9, 0.9, 0.95, 0.95]], ["x"], [], []]'}]
        got = collect_items(events, '[{"bbox_2d": [0.1, 0.1, 0.2, 0.2], "label": "正文"}]')
        assert got == [{"bbox_2d": [0.1, 0.1, 0.2, 0.2], "label": "正文"}]

    def test_no_coordinates_anywhere_is_empty(self):
        # 负向对照：没有任何坐标时必须老实返回空 ——
        # 少了这一条，一个"永远返回点东西"的实现也能过上面两条。
        assert collect_items([{"type": "tool_result", "ok": True, "content": "查到了，图里有三只猫"}], "") == []
        assert collect_items([], "") == []

