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

from qsmy_deepseek_locator import drawing  # noqa: E402
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
        # log 是调试日志参数（0.1.0 就有）：本库**只在开启日志时**才传它，
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


def is_real_font_file(font) -> bool:
    """这个字体对象是**真的开了某个字体文件**，还是 PIL 内置兜底？

    ⚠️ 不能用 isinstance(font, ImageFont.FreeTypeFont) 判断：Pillow 12 起
    load_default() 返回的也是 FreeTypeFont（内置字体被做成了字体对象，path 是个 BytesIO）。
    这个坑值得写下来 —— 它会让「降级了没有」这条断言永远为真，也就是永远测不出问题。
    """
    return isinstance(getattr(font, "path", None), str)


def find_any_cjk_font() -> str | None:
    """在**本机**找一个真的能渲染中文的字体文件（找不到就 skip，不硬编码平台路径）。

    刻意不写死 msyh.ttc 之类的名字：自测要能在 Linux CI 上跑，
    而「本机有没有中文字体」恰恰是这条功能要处理的不确定性本身。
    """
    for name in drawing._FONT_CANDIDATES:
        for directory in drawing._FONT_DIRS:
            candidate = directory / name
            if candidate.exists() and drawing._renders_cjk(str(candidate)):
                return str(candidate)
    for directory in drawing._FONT_DIRS:
        if not directory.is_dir():
            continue
        for suffix in ("*.ttc", "*.ttf", "*.otf"):
            for found in directory.rglob(suffix):
                if drawing._renders_cjk(str(found)):
                    return str(found)
    return None


@pytest.fixture
def cjk_font() -> str:
    """本机某个可用的中文字体路径；本机一个都没有时 skip 掉依赖它的用例。"""
    path = find_any_cjk_font()
    if not path:
        pytest.skip("本机没有任何含中文字形的字体，无法验证「找得到」的那一半")
    return path


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
        "QSML_LOG_FILE",  # 漏了它的话，本机设了该变量自测会真的往那个路径写 JSONL
        # 字体这两个同理：本机真配了 QSML_FONT_PATH 的话，「找不到字体要告警」
        # 那条自测会莫名其妙地不告警（因为它其实找得到）—— 假绿。
        "QSML_FONT_PATH", "QSML_FONT_DIR",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _reset_font_state():
    """每条用例前后都清掉字体缓存与「已经告警过」的开关。

    为什么必须是 autouse 的：缓存与告警去重都是**模块级状态**，任何一条用例碰过它们
    就会影响下一条 —— 尤其是「找不到字体要告警 / 找得到就不许告警」这一对正反断言，
    不重置就变成掷骰子（谁先跑谁说了算）。这正是项目记忆里那条「负向对照」的同类问题。
    """
    def _clear():
        drawing._font_cache.clear()
        drawing._warned_no_cjk = False
        # 用 hasattr 而不是直接调：有的用例会把 _renders_cjk 打桩成一个普通函数
        # （"永远校验不过"），那种桩没有 cache_clear —— 清理逻辑不该因此把用例判红。
        cache_clear = getattr(drawing._renders_cjk, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()

    _clear()
    yield
    _clear()
