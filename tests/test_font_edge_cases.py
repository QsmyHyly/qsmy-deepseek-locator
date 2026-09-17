"""字体来源的收尾边界：显式给了坏路径怎么办、路径优先级怎么排。

这几条是「P1-1 注入字体」这条功能的**退化路径** —— 参数收下了，但指向的东西不可用。
它们的共同要求是：不抛异常、不让整轮打标挂掉，但要退回到一条说得清的路上。
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from PIL import ImageFont

from qsmy_deepseek_locator import drawing
from qsmy_deepseek_locator.drawing import resolve_font
from tests.conftest import is_real_font_file as _is_real_font_file


def test_explicit_missing_path_falls_back_to_discovery(cjk_font, monkeypatch):
    """显式给的路径不存在时，不该直接摆烂 —— 应该继续走正常探测。

    为什么这条重要：调用方传 font_path 的场景（配置里写了个路径、换台机器就没了）
    最容易踩这个。此时**能自己找到就别让用户去修配置**，但也别默默用坏字体。
    """
    monkeypatch.setenv(drawing.FONT_PATH_ENV, cjk_font)
    font = resolve_font(18, font_path="/不存在/的/字体.ttf")
    assert _is_real_font_file(font), "显式路径坏了之后，探测这条路必须接上"


def test_explicit_missing_path_warns_when_nothing_else_works(monkeypatch, tmp_path):
    monkeypatch.setattr(drawing, "_FONT_DIRS", [tmp_path / "空的"])
    monkeypatch.setattr(drawing, "_FONT_CANDIDATES", ["不存在.ttf"])
    monkeypatch.delenv(drawing.FONT_PATH_ENV, raising=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolve_font(18, font_path="/不存在/的/字体.ttf")
    assert any("方块" in str(w.message) for w in caught), "全都不可用却没告警"


def test_broken_font_file_does_not_raise(monkeypatch, tmp_path):
    """文件存在但不是字体（比如一个 .ttf 后缀的文本）—— 不许抛异常，只许告警。"""
    bogus = tmp_path / "假的字体.ttf"
    bogus.write_text("我不是字体", encoding="utf-8")
    monkeypatch.setattr(drawing, "_FONT_DIRS", [tmp_path / "空的"])
    monkeypatch.setattr(drawing, "_FONT_CANDIDATES", ["不存在.ttf"])
    monkeypatch.delenv(drawing.FONT_PATH_ENV, raising=False)
    monkeypatch.delenv(drawing.FONT_DIR_ENV, raising=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        font = resolve_font(18, font_path=str(bogus))
    assert font is not None
    assert any("方块" in str(w.message) for w in caught)


def test_env_path_is_not_render_verified(cjk_font, monkeypatch, tmp_path):
    """QSML_FONT_PATH 是**用户明确的判断**，库不再替他渲染校验。

    判据：拿一个真实存在的字体文件、但把校验函数打桩成"永远不通"，
    如果还照旧用上了它，就说明这条路上确实没做校验（= 用户的判断优先）。
    """
    monkeypatch.setenv(drawing.FONT_PATH_ENV, cjk_font)
    monkeypatch.setattr(drawing, "_renders_cjk", lambda path: False)
    font = resolve_font(18)
    assert _is_real_font_file(font), "显式指定的字体被渲染校验挡下了 —— 与承诺不符"


def test_explicit_argument_is_not_render_verified_either(cjk_font, monkeypatch):
    monkeypatch.setattr(drawing, "_renders_cjk", lambda path: False)
    font = resolve_font(18, font_path=cjk_font)
    assert _is_real_font_file(font)


def test_falsy_font_path_argument_falls_back_to_settings(cjk_font, tmp_path):
    """font_path="" / None 一律"没给" -> 回落到 Settings.font_path。

    这条盯的是 locate.py 里那句 `font_path or self.settings.font_path`：
    用 or 而不是 is not None，于是空串也算"没给"。对路径类参数这样处理是对的
    （空串没有意义），但要**断言**它，否则哪天有人改成 is not None，
    ".env 里写了 QSML_FONT_PATH= 空值 + 显式传了空串" 这种组合就会静默失效。
    """
    from PIL import Image
    from qsmy_deepseek_locator import Locator

    class _Client:
        def complete(self, messages, *, settings=None, on_event=None, log=None):
            from qsmy_deepseek_locator import ChatReply
            return ChatReply(text='[{"bbox_2d": [0.2, 0.2, 0.6, 0.6], "label": "猫"}]',
                             model="fake")

    img = Image.new("RGB", (200, 120), (255, 255, 255))
    out = tmp_path / "o.png"
    locator = Locator(client=_Client(), api_key="sk-test", font_path=cjk_font)
    # font_path=""（空串）在 locate_to_file 里算「没给」-> 回落到 Settings.font_path
    result = locator.locate_to_file(img, "猫", out, font_path="")
    assert result.annotated_path == str(out)
    assert any(Path(key[0]) == Path(cjk_font) for key in drawing._font_cache), \
        list(drawing._font_cache)


def test_settings_font_path_also_wins_over_discovery(cjk_font, monkeypatch, tmp_path):
    """Settings.font_path（构造 Locator 时给）与参数同为"用户明确指定"，同样免检。"""
    monkeypatch.setattr(drawing, "_renders_cjk", lambda path: False)
    from qsmy_deepseek_locator import Locator
    locator = Locator(font_path=cjk_font, api_key="sk-test")
    from PIL import Image
    img = Image.new("RGB", (160, 100), (255, 255, 255))
    drawing.draw(img, [{"bbox_2d": [0.1, 0.1, 0.5, 0.5], "label": "猫"}],
                 font_path=locator.settings.font_path)
    assert any(Path(key[0]) == Path(cjk_font) for key in drawing._font_cache), \
        list(drawing._font_cache)
