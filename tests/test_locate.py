"""调用链自测：报文拼装 / 结果解析 / 报错语义（全部离线）。"""

from __future__ import annotations

import json

import pytest

from qsmy_deepseek_locator.client import (
    ChatReply,
    build_messages,
    build_request,
    image_part,
    resolve_thinking,
    thinking_payload,
)
from qsmy_deepseek_locator.config import Settings
from qsmy_deepseek_locator.errors import EmptyResponseError, MissingAPIKeyError
from qsmy_deepseek_locator.locate import Locator, locate
from qsmy_deepseek_locator.prompts import DEFAULT_SYSTEM_PROMPT


class TestMessageBuilding:
    def test_image_part_detail_only_when_legal(self):
        assert image_part("u") == {"type": "image_url", "image_url": {"url": "u"}}
        assert image_part("u", "low")["image_url"]["detail"] == "low"
        assert "detail" not in image_part("u", "超高")["image_url"]

    def test_messages_shape(self):
        messages = build_messages("找猫", image_url="data:image/png;base64,AAA",
                                  system_prompt="SYS", image_detail="low")
        assert [m["role"] for m in messages] == ["system", "user"]
        content = messages[1]["content"]
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["detail"] == "low"
        assert content[1] == {"type": "text", "text": "找猫"}

    def test_text_only(self):
        messages = build_messages("你好")
        assert messages[0]["content"] == "你好"


class TestRequest:
    def test_thinking_goes_to_extra_body(self):
        # thinking 不是顶层字段，必须走 extra_body，否则 SDK 直接拒绝
        kwargs = build_request(Settings(thinking=False), [{"role": "user", "content": "x"}])
        assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
        assert "thinking" not in kwargs

    def test_thinking_none_sends_nothing(self):
        kwargs = build_request(Settings(), [{"role": "user", "content": "x"}])
        assert "extra_body" not in kwargs
        assert "reasoning_effort" not in kwargs

    def test_effort_validated(self):
        good = build_request(Settings(thinking=True, reasoning_effort="low"), [])
        assert good["reasoning_effort"] == "low"
        bad = build_request(Settings(thinking=True, reasoning_effort="极高"), [])
        assert "reasoning_effort" not in bad

    def test_no_effort_when_thinking_off(self):
        kwargs = build_request(Settings(thinking=False, reasoning_effort="max"), [])
        assert "reasoning_effort" not in kwargs

    def test_max_tokens_and_stream_options(self):
        kwargs = build_request(Settings(max_tokens=4096), [])
        assert kwargs["max_tokens"] == 4096
        assert kwargs["stream_options"] == {"include_usage": True}
        assert kwargs["stream"] is True

    def test_resolve_thinking_priority(self):
        assert resolve_thinking(Settings(), None) == (None, None)
        assert resolve_thinking(Settings(thinking=True), None) == (True, None)
        assert resolve_thinking(Settings(thinking=True), False) == (False, None)
        assert thinking_payload(None) is None


class TestLocate:
    def test_normal_path(self, fake_client, sample_png):
        client = fake_client(
            '[{"bbox_2d": [0.1, 0.2, 0.3, 0.4], "label": "红色圆形"},'
            ' {"point_2d": [0.5, 0.5], "label": "中心"}]',
            usage={"prompt_tokens": 800, "completion_tokens": 40},
        )
        result = Locator(client=client).locate(sample_png, "几何图形")

        assert len(result) == 2
        assert result.labels == ["红色圆形", "中心"]
        assert result.bboxes == [(0.1, 0.2, 0.3, 0.4)]
        assert result.points == [(0.5, 0.5)]
        assert result.image_size == (400, 300)
        assert result.usage["prompt_tokens"] == 800
        assert result.summary()["total"] == 2
        assert result.empty is False
        assert "红色圆形" in result.describe()
        assert len(result.find("红")) == 1

    def test_prompt_and_system_reach_the_model(self, fake_client, sample_png):
        client = fake_client("[]")
        Locator(client=client).locate(sample_png, "登录按钮")
        messages = client.calls[0]["messages"]
        assert messages[0]["content"] == DEFAULT_SYSTEM_PROMPT
        assert "登录按钮" in messages[1]["content"][1]["text"]
        # 图片必须以 data URL 形式放在 user 消息里
        assert messages[1]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_custom_prompt_wins(self, fake_client, sample_png):
        client = fake_client("[]")
        Locator(client=client).locate(sample_png, "被忽略", prompt="自定义提问")
        assert client.calls[0]["messages"][1]["content"][1]["text"] == "自定义提问"

    def test_settings_override_reaches_client(self, fake_client, sample_png):
        client = fake_client("[]")
        Locator(client=client, model="主模型").locate(sample_png, thinking=False, model="临时模型")
        assert client.calls[0]["settings"].model == "临时模型"
        assert client.calls[0]["settings"].thinking is False

    def test_on_event_streamed(self, fake_client, sample_png):
        client = fake_client('[{"bbox_2d": [0, 0, 1, 1], "label": "x"}]',
                             reasoning="我想想", events=True)
        seen: list[str] = []
        Locator(client=client).locate(sample_png, None, on_event=lambda e: seen.append(e["type"]))
        assert "reasoning" in seen and "content" in seen and "finish" in seen

    def test_legacy_scale_warning_surfaces(self, fake_client, sample_png):
        client = fake_client('[{"bbox_2d": [100, 200, 300, 400], "label": "猫"}]')
        result = Locator(client=client).locate(sample_png)
        assert result.bboxes == [(0.1, 0.2, 0.3, 0.4)]
        assert any("0~1000" in w for w in result.warnings)

    def test_empty_array_is_not_an_error(self, fake_client, sample_png):
        client = fake_client("[]")
        result = Locator(client=client).locate(sample_png)
        assert result.empty
        assert any("空数组" in w for w in result.warnings)

    def test_prose_without_coordinates_warns(self, fake_client, sample_png):
        client = fake_client("这张图里我看不清有什么东西。")
        result = Locator(client=client).locate(sample_png)
        assert result.empty
        assert any("没有可解析的坐标" in w for w in result.warnings)

    def test_result_json_roundtrip(self, fake_client, sample_png, tmp_path):
        client = fake_client('[{"bbox_2d": [0.1, 0.2, 0.3, 0.4], "label": "猫"}]')
        result = Locator(client=client).locate(sample_png, "猫")
        payload = json.loads(result.to_json())
        assert payload["detections"][0]["label"] == "猫"
        assert payload["counts"]["bbox_count"] == 1
        assert payload["image_size"] == [400, 300]
        saved = result.save(tmp_path / "r.json", include_raw=True)
        assert json.loads(saved.read_text(encoding="utf-8"))["raw_items"]

    def test_locate_and_draw(self, fake_client, sample_png, tmp_path):
        client = fake_client('[{"bbox_2d": [0.25, 0.25, 0.75, 0.75], "label": "框"}]')
        result, img = Locator(client=client).locate_and_draw(
            sample_png, "框", output=tmp_path / "a.png"
        )
        assert len(result) == 1
        assert img.getpixel((200, 75)) == (255, 0, 0)
        assert (tmp_path / "a.png").exists()

    def test_module_level_locate(self, fake_client, sample_png):
        client = fake_client("[]")
        result = locate(sample_png, "猫", client=client)
        assert result.empty and result.model == "fake-model"


class TestErrors:
    def test_missing_key_is_explicit(self, sample_png):
        # 这条是本库对外的承诺：没有 Key 就报错，绝不静默给假数据
        with pytest.raises(MissingAPIKeyError) as excinfo:
            Locator().locate(sample_png, "猫")
        assert "DEEPSEEK_API_KEY" in str(excinfo.value)

    def test_empty_content_raises_with_hint(self, monkeypatch):
        from qsmy_deepseek_locator.client import DeepSeekVisionClient

        client = DeepSeekVisionClient(Settings(api_key="sk-test"))

        def fake_stream(messages, **kwargs):
            yield {"type": "reasoning", "text": "想很久"}
            yield {"type": "finish", "reason": "length"}

        monkeypatch.setattr(client, "stream", fake_stream)
        with pytest.raises(EmptyResponseError) as excinfo:
            client.complete([{"role": "user", "content": "x"}])
        message = str(excinfo.value)
        assert "max_tokens" in message and "thinking=False" in message


class TestSettings:
    def test_merged_ignores_none(self):
        base = Settings(model="a", thinking=True)
        assert base.merged(model=None).model == "a"
        assert base.merged(model="b").model == "b"
        assert base.merged(unknown_key="x") is base

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
        monkeypatch.setenv("DEEPSEEK_MODEL", "my-model")
        monkeypatch.setenv("QSML_THINKING", "0")
        monkeypatch.setenv("QSML_REASONING_EFFORT", "MAX")
        monkeypatch.setenv("QSML_IMAGE_DETAIL", "low")
        monkeypatch.setenv("QSML_MAX_TOKENS", "3000")
        settings = Settings.from_env()
        assert settings.api_key == "sk-x"
        assert settings.model == "my-model"
        assert settings.thinking is False
        assert settings.reasoning_effort == "max"
        assert settings.image_detail == "low"
        assert settings.max_tokens == 3000

    def test_bad_env_values_ignored(self, monkeypatch):
        monkeypatch.setenv("QSML_THINKING", "可能吧")
        monkeypatch.setenv("QSML_REASONING_EFFORT", "很高")
        monkeypatch.setenv("QSML_IMAGE_DETAIL", "ultra")
        monkeypatch.setenv("QSML_MAX_TOKENS", "很多")
        settings = Settings.from_env()
        assert settings.thinking is None
        assert settings.reasoning_effort is None
        assert settings.image_detail is None
        assert settings.max_tokens is None

    def test_redacted(self):
        from qsmy_deepseek_locator.config import redacted

        assert "sk-secret-value" not in json.dumps(redacted(Settings(api_key="sk-secret-value-1234")))
