"""异常契约与输出路径（P1-3 / P2-1）的自测。

反馈文档 P1-3 的原话是「异常契约不闭合」，点名三处：NotImplementedError（use_tools）、
ValueError（输出路径 / log_file 类型）、OSError（最终落盘那行没有 try/except）。
安卓那边的后果很具体 —— 只能抓 BaseException 兜底，因为它不知道还会漏出什么。

修法的**硬约束是向后兼容**：已有调用方可能正按 except ValueError /
except NotImplementedError 兜这些路径，新增子类不能把它们漏出去。
所以新异常全部多重继承（LocatorError + 原来的那个基类），本文件专门验这件事。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from qsmy_deepseek_locator import Locator, locate_to_file, resolve_output_path, save_annotated
from qsmy_deepseek_locator.debuglog import coerce_log
from qsmy_deepseek_locator.errors import (
    CancelledError,
    LocatorError,
    LogFileTypeError,
    OutputPathError,
    UnsupportedFeatureError,
    WriteError,
)

BOX = '[{"bbox_2d": [0.25, 0.25, 0.75, 0.75], "label": "方块"}]'

# 0.1.2 会从这些路径漏出来的裸异常类型。新子类必须**同时**是两个东西，
# 否则旧代码的 except 子句会静默漏接（代码一行没改，异常却逃出去了）。
_BACKWARD_COMPAT_BASES = {
    UnsupportedFeatureError: (NotImplementedError, LocatorError),
    OutputPathError: (ValueError, LocatorError),
    LogFileTypeError: (ValueError, LocatorError),
    WriteError: (LocatorError,),
    CancelledError: (LocatorError,),
}


class TestExceptionHierarchy:
    @pytest.mark.parametrize("exc_type,bases", list(_BACKWARD_COMPAT_BASES.items()))
    def test_new_errors_keep_both_bases(self, exc_type, bases):
        for base in bases:
            assert issubclass(exc_type, base), "%s 丢了 %s" % (exc_type.__name__, base.__name__)

    def test_write_error_does_not_pretend_to_be_oserror(self):
        """WriteError **刻意不**继承 OSError —— 取舍写在 errors.py 模块头，这里是它的守卫。

        理由一句话：OSError 的 args 语义（errno/strerror）与普通异常混用，会让排查时
        最常看的那个属性变得说不清；而「是本库写的还是文件系统报的」本来也不该由本库
        替调用方判断。原始 OSError 一个不丢，全挂在 __cause__ 上（下面有断言）。
        """
        assert not issubclass(WriteError, OSError)

    def test_everything_the_library_raises_is_a_locator_error(self):
        for exc_type in _BACKWARD_COMPAT_BASES:
            assert issubclass(exc_type, LocatorError)


class TestNotImplementedContract:
    """P1-3 第 1 处：use_tools=True 的报错。"""

    def test_use_tools_raises_the_new_type_but_stays_a_not_implemented_error(
        self, fake_client, sample_png, tmp_path
    ):
        client = fake_client(BOX)
        with pytest.raises(UnsupportedFeatureError) as excinfo:
            locate_to_file(sample_png, "方块", tmp_path / "o.png", use_tools=True, client=client)
        # 老写法（except NotImplementedError）必须照旧抓得住
        assert isinstance(excinfo.value, NotImplementedError)
        assert isinstance(excinfo.value, LocatorError)
        assert client.calls == [], "报错要在调模型之前，别白花一次 API"


class TestOutputPathContract:
    """P1-3 第 2 处 + P2-1：输出路径不可用。"""

    @pytest.mark.parametrize("bad", ["out.tga", "out.png.txt", "", None])
    def test_bad_path_is_both_value_error_and_locator_error(self, bad):
        with pytest.raises(OutputPathError) as excinfo:
            resolve_output_path(bad)
        assert isinstance(excinfo.value, ValueError), "0.1.2 抛的是 ValueError，不能漏接"

    def test_empty_path_message_says_what_to_pass(self):
        with pytest.raises(OutputPathError) as excinfo:
            resolve_output_path(None)
        assert "输出路径" in str(excinfo.value)

    def test_unusable_parent_dir_is_a_locator_error_before_the_model_is_called(
        self, fake_client, sample_png, tmp_path
    ):
        blocker = tmp_path / "占位文件"
        blocker.write_text("我不是目录", encoding="utf-8")
        client = fake_client(BOX)
        with pytest.raises(WriteError) as excinfo:
            locate_to_file(sample_png, "方块", blocker / "o.png", client=client)
        # 原始 OSError 必须在 __cause__ 上（不丢信息），报错文本里要带具体路径
        assert isinstance(excinfo.value.__cause__, OSError)
        assert str(blocker) in str(excinfo.value)
        assert client.calls == []

    def test_mkdir_failure_is_wrapped_not_raw(self, fake_client, sample_png, tmp_path, monkeypatch):
        """用一个与平台无关的方式制造 OSError：把 Path.mkdir 换成必然失败的桩。

        不用 chmod / 只读目录那套：Windows 上对目录 chmod 基本无效，
        那种负向对照在本机会静默失效（假绿）。
        """
        def _boom(self, *args, **kwargs):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(Path, "mkdir", _boom)
        client = fake_client(BOX)
        with pytest.raises(WriteError) as excinfo:
            locate_to_file(sample_png, "方块", tmp_path / "o.png", client=client)
        assert excinfo.value.__cause__.errno == 28
        assert client.calls == []

    def test_final_save_failure_is_wrapped_and_says_no_need_to_recall(
        self, fake_client, sample_png, tmp_path, monkeypatch
    ):
        """P1-3 第 3 处：**最终落盘那行**。钱已经花了、结果也解析完了，只剩写不进去。

        这条路径以前是裸的 annotated.save()：抛原生 OSError，except LocatorError 完全抓不住。
        而且它发生在调模型**之后**，所以文案必须说清「不必再调一次模型」，
        否则用户第一反应是把整轮重跑一遍（再花一次钱）。
        """
        def _boom(self, *args, **kwargs):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(Image.Image, "save", _boom)
        client = fake_client(BOX)
        with pytest.raises(WriteError) as excinfo:
            locate_to_file(sample_png, "方块", tmp_path / "o.png", client=client)
        assert excinfo.value.__cause__.errno == 28
        assert "不必再调一次模型" in str(excinfo.value)
        assert len(client.calls) == 1, "模型确实调过一次（这正是文案要安抚的场景）"

    def test_save_annotated_wraps_oserror(self, sample_png, tmp_path, monkeypatch):
        def _boom(self, *args, **kwargs):
            raise OSError(13, "Permission denied")

        monkeypatch.setattr(Image.Image, "save", _boom)
        with pytest.raises(WriteError) as excinfo:
            save_annotated(sample_png, [{"bbox_2d": [0.1, 0.1, 0.5, 0.5], "label": "x"}],
                           tmp_path / "o.png")
        assert excinfo.value.__cause__.errno == 13
        assert "标注图写入失败" in str(excinfo.value)

    def test_locate_and_draw_uses_the_same_rules(self, fake_client, sample_png, tmp_path, monkeypatch):
        """Locator.locate_and_draw 以前写死 format="PNG" 且直接 save —— 现在与另外两条路同规。"""
        def _boom(self, *args, **kwargs):
            raise OSError(13, "Permission denied")

        monkeypatch.setattr(Image.Image, "save", _boom)
        locator = Locator(client=fake_client(BOX), api_key="sk-test")
        with pytest.raises(WriteError):
            locator.locate_and_draw(sample_png, "方块", output=tmp_path / "o.png")

    def test_locate_and_draw_respects_the_suffix(self, fake_client, sample_png, tmp_path):
        """顺带修掉：这条路以前无论后缀一律存 PNG，.jpg 会得到「名字 .jpg、内容是 PNG」。"""
        locator = Locator(client=fake_client(BOX), api_key="sk-test")
        _, annotated = locator.locate_and_draw(sample_png, "方块", output=tmp_path / "o.jpg")
        target = tmp_path / "o.jpg"
        assert target.exists()
        with Image.open(target) as img:
            assert img.format == "JPEG"


class TestLogFileContract:
    """P1-3 第 2 处的另一半：log_file 类型不认识。"""

    @pytest.mark.parametrize("bad", [123, object(), ["a"], 3.5])
    def test_bad_type_is_both_value_error_and_locator_error(self, bad):
        with pytest.raises(LogFileTypeError) as excinfo:
            coerce_log(bad)
        assert isinstance(excinfo.value, ValueError)

    def test_message_names_the_accepted_types(self):
        with pytest.raises(LogFileTypeError) as excinfo:
            coerce_log(123)
        assert "DebugLog" in str(excinfo.value)

    def test_valid_inputs_still_work(self):
        """负向对照：正常输入不许被新异常误伤。"""
        assert coerce_log(None) is None
        assert coerce_log(False) is None
        assert coerce_log("") is None
        assert coerce_log("runs/x.jsonl") is not None
