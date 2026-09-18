"""报文拼装自测：请求长什么样、以及「thinking 规则只有一份」这件事有没有被守住。

为什么单独钉住 merge_thinking（0.1.3 上游化）：
    这条规则原先在三个项目里各有一份实现（本库一份、旧演示项目 objloc/providers.py 一份、
    安卓 App 的同一份镜像又一份）。抽成不依赖 Settings 的纯函数之后，最容易发生的退化
    **不是新函数写错，而是老路径悄悄绕开它** —— 那样又变回两份。
    所以本文件除了测规则本身，还专门加了「resolve_thinking 必须逐项等于 merge_thinking」
    的参数化对照：只有这条断言在，薄封装才真的只是薄封装。
"""

from __future__ import annotations

import pytest

from qsmy_deepseek_locator.config import Settings
from qsmy_deepseek_locator.request_build import (
    build_messages,
    build_request,
    image_part,
    merge_thinking,
    resolve_thinking,
    thinking_payload,
)


class TestMergeThinking:
    """四条规则逐条钉住；每条都写清它防的是什么。"""

    def test_no_opinion_means_send_nothing(self):
        # 「没表态」不等于「关掉」：两样都不传，服务端自己决定。
        # 若这里被改成 False，等于本库替服务端做了决定，属于契约变更。
        assert merge_thinking(None, None) == (None, None)

    def test_explicit_off_drops_effort(self):
        # 关掉思考时 effort 无意义：留着它会发一个自相矛盾的报文
        assert merge_thinking(False, "high") == (False, None)

    def test_explicit_on_keeps_effort(self):
        assert merge_thinking(True, "high") == (True, "high")

    def test_caller_defaults_are_honoured(self):
        # 下游（旧演示项目 / 安卓 App）就是这样用的：默认值来自它们自己的 Settings
        assert merge_thinking(None, None, default_thinking=True, default_effort="low") == (True, "low")

    def test_per_call_beats_default(self):
        assert merge_thinking(False, None, default_thinking=True, default_effort="low") == (False, None)

    def test_default_off_short_circuits(self):
        assert merge_thinking(None, "high", default_thinking=False) == (False, None)

    @pytest.mark.parametrize("bad", ["ultra", "HIGHEST", "1", "none", "  "])
    def test_illegal_effort_is_dropped_not_forwarded(self, bad):
        # 宁可走服务端默认，也不要 400
        assert merge_thinking(True, bad) == (True, None)

    @pytest.mark.parametrize("raw,expected", [("High", "high"), ("  low ", "low"), ("MAX", "max")])
    def test_effort_is_normalised(self, raw, expected):
        assert merge_thinking(True, raw) == (True, expected)


class TestResolveThinkingIsStillTheSameRule:
    """负向对照：老路径只要绕开 merge_thinking，这里就会红。"""

    @pytest.mark.parametrize("thinking", [None, True, False])
    @pytest.mark.parametrize("effort", [None, "", "high", "HIGH", "  low ", "ultra", "max"])
    def test_matches_merge(self, thinking, effort):
        # 默认值那一维放在循环里穷举（3 档默认开关 × 4 档默认强度 = 12 种），
        # 不再各占一层参数化 —— 否则光这一条对照就会生成两百多个用例，
        # 让「用例数」变成噪音；而它想证伪的只有一件事：老路径有没有绕开新函数。
        for default_thinking in (None, True, False):
            for default_effort in (None, "", "low", "bogus"):
                settings = Settings(
                    api_key="sk-test",
                    thinking=default_thinking,
                    reasoning_effort=default_effort,
                )
                assert resolve_thinking(settings, thinking, effort) == merge_thinking(
                    thinking,
                    effort,
                    default_thinking=settings.thinking,
                    default_effort=settings.reasoning_effort,
                ), (default_thinking, default_effort)


class TestThinkingPayload:
    def test_none_sends_no_field(self):
        # None 表示「请求体里根本没有这个字段」，而不是 {"thinking": {"type": "..."}}
        assert thinking_payload(None) is None

    @pytest.mark.parametrize("enabled,name", [(True, "enabled"), (False, "disabled")])
    def test_shape(self, enabled, name):
        # thinking 只能塞进 extra_body，形状是实测出来的契约
        assert thinking_payload(enabled) == {"thinking": {"type": name}}


class TestImagePart:
    def test_legal_detail_is_kept(self):
        part = image_part("https://x/a.png", "high")
        assert part["image_url"]["detail"] == "high"

    @pytest.mark.parametrize("bad", [None, "", "ultra"])
    def test_illegal_detail_is_silently_dropped(self, bad):
        # 故意不抛：它在「拼一次识别请求」的主路径上，写错 detail 最多只是没生效
        assert "detail" not in image_part("https://x/a.png", bad)["image_url"]


class TestBuildRequest:
    def test_thinking_goes_to_extra_body_not_top_level(self):
        s = Settings(api_key="sk-test", thinking=True, reasoning_effort="high")
        kwargs = build_request(s, build_messages("找红色圆形"))
        assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
        assert kwargs["reasoning_effort"] == "high"
        assert "thinking" not in kwargs

    def test_no_thinking_field_when_unset(self):
        s = Settings(api_key="sk-test", thinking=None)
        kwargs = build_request(s, build_messages("找红色圆形"))
        assert "extra_body" not in kwargs
        assert "reasoning_effort" not in kwargs

    def test_stream_options_only_when_streaming(self):
        s = Settings(api_key="sk-test")
        assert build_request(s, build_messages("x"), stream=True)["stream_options"] == {"include_usage": True}
        assert "stream_options" not in build_request(s, build_messages("x"), stream=False)

