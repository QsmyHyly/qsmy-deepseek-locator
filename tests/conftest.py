"""自测公共设施：源码路径引导 + 假客户端。

全部自测**不联网、不花 API**：真实客户端只在少数几个「报文长什么样」的断言里被间接检查，
模型返回的内容一律由 FakeClient 提供。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

# 让 tests/ 在「未 pip install」时也能直接 import 到 src 下的包
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qsmy_deepseek_locator.client import ChatReply  # noqa: E402


class FakeClient:
    """假客户端：把预设文本当成模型回复，并记录每次调用收到的报文与配置。"""

    def __init__(
        self,
        text: str = "[]",
        *,
        reasoning: str = "",
        finish_reason: str = "stop",
        usage: dict | None = None,
        model: str = "fake-model",
        events: bool = False,
    ):
        self.text = text
        self.reasoning = reasoning
        self.finish_reason = finish_reason
        self.usage = usage
        self.model = model
        self.events = events
        self.calls: list[dict] = []

    def complete(self, messages, *, settings=None, on_event=None, log=None):
        # log 是 v0.2 起的调试日志参数：本库**只在开启日志时**才传它，
        # 所以假客户端收下它即可，不必真的写文件。
        self.calls.append({"messages": messages, "settings": settings, "log": log})
        if on_event is not None and self.events:
            if self.reasoning:
                on_event({"type": "reasoning", "text": self.reasoning[:4]})
                on_event({"type": "reasoning", "text": self.reasoning[4:]})
            for index in range(0, len(self.text), 8):
                on_event({"type": "content", "text": self.text[index:index + 8]})
            on_event({"type": "finish", "reason": self.finish_reason})
        return ChatReply(
            text=self.text,
            reasoning=self.reasoning,
            model=self.model,
            finish_reason=self.finish_reason,
            usage=self.usage,
        )


@pytest.fixture
def fake_client():
    return FakeClient


@pytest.fixture
def sample_png(tmp_path):
    """一张 400x300 的纯白图，可当任何「本地图片」用。"""
    path = tmp_path / "sample.png"
    Image.new("RGB", (400, 300), (255, 255, 255)).save(path, format="PNG")
    return path


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """默认清掉环境变量：本机真的配了 DEEPSEEK_API_KEY，不清掉会误触发真实调用。"""
    for name in (
        "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL",
        "QSML_THINKING", "QSML_REASONING_EFFORT", "QSML_IMAGE_DETAIL",
        "QSML_MAX_TOKENS", "QSML_TIMEOUT", "QSML_MAX_RETRIES", "QSML_SYSTEM_PROMPT",
    ):
        monkeypatch.delenv(name, raising=False)
