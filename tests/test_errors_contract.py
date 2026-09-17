"""异常契约与输出路径（P1-3 / P2-1）的自测。

反馈文档 P1-3 的原话是「异常契约不闭合」，点名三处：NotImplementedError（use_tools）、
ValueError（输出路径 / log_file 类型）、OSError（最终落盘那行没有 try/except）。
安卓那边的后果很具体 —— 只能抓 BaseException 兜底，因为它不知道还会漏出什么。

修法的**硬约束是向后兼容**：已有调用方可能正按 except ValueError /
except NotImplementedError 兜这些路径，新增子类不能把它们漏出去。
所以新异常全部多重继承（LocatorError + 原来的那个基类），本文件专门验这件事。
"""

from __future__ import annotations

import importlib
import re
import warnings
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

    def test_every_error_class_the_tests_exercise_is_a_locator_error(self):
        """扫描自测文件正文里出现的异常类名，每一个都必须是 LocatorError 的子孙。

        上一版只遍历 _BACKWARD_COMPAT_BASES 那 5 个手写名字 —— 它**恰好**看不见
        "某个公开 API 还在抛原生 OSError"这类漏网（LocateResult.save 就曾经这样漏在外面，
        而且是评审读代码发现的，这条断言当时一次都没红）。

        注意候选集只从本库 errors.py 里取：正文里出现的 OSError / ValueError / ConnectionError
        这些**恰恰是被允许抛出来的底层异常**（WriteError.__cause__ 上就挂着 OSError），
        把它们也拉进断言等于要求"本库不许 mention 任何内建异常"，那是另一回事。
        "有没有裸抛"由本文件各条实测（pytest.raises）来钉，不由名字扫描来钉。
        """
        names = set()
        for path in Path(__file__).parent.glob("test_*.py"):
            names.update(re.findall(r"\b[A-Z][A-Za-z]*Error\b", path.read_text(encoding="utf-8")))
        errors_module = importlib.import_module("qsmy_deepseek_locator.errors")
        candidates = {
            name: getattr(errors_module, name)
            for name in names
            if isinstance(getattr(errors_module, name, None), type)
        }
        # 负向对照：扫描本身必须真的扫到了东西（空集合会让这条断言恒真）
        assert len(candidates) >= len(_BACKWARD_COMPAT_BASES), sorted(candidates)
        assert "WriteError" in candidates and "OutputPathError" in candidates
        for name, exc_type in sorted(candidates.items()):
            assert issubclass(exc_type, LocatorError), f"{name} 不是 LocatorError 的子孙"

    def test_the_whole_package_defines_no_stray_error_class(self):
        """遍历**本包每一个模块**，里面定义的异常类一个都不许绕过 LocatorError。

        这条比上面那条宽：名字扫描只能发现"自测提到过"的类，而漏网的那次（results.py 里
        裸抛 OSError）根本没有自己的异常类。这条至少把"新写一个不继承基类的异常类"堵死。
        """
        import pkgutil

        package = importlib.import_module("qsmy_deepseek_locator")
        checked = []
        for info in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f"qsmy_deepseek_locator.{info.name}")
            for attr, value in vars(module).items():
                if (
                    isinstance(value, type)
                    and issubclass(value, BaseException)
                    and value.__module__ == module.__name__
                ):
                    checked.append(f"{module.__name__}.{attr}")
                    assert issubclass(value, LocatorError), f"{checked[-1]} 绕过了 LocatorError"
        assert "qsmy_deepseek_locator.errors.WriteError" in checked
        assert len(checked) >= len(_BACKWARD_COMPAT_BASES)

    def test_readme_exception_table_matches_the_code(self):
        """README 6.1 那张异常表里的每个名字都得真的存在且是 LocatorError。

        文档表是给人看的承诺；表里写一个不存在的类，比不写更糟。
        """
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
        table = readme.split("### 6.1 异常一览")[1].split("##")[0]
        # 只看表格的**第一列**（异常名那一列）：第三列"额外继承"里的 NotImplementedError /
        # ValueError 是内建基类，不该要求本库定义它们。
        listed = set()
        for line in table.splitlines():
            cells = line.split("|")
            if len(cells) >= 3:
                listed.update(re.findall(r"`([A-Za-z]+Error)`", cells[1]))
        assert "WriteError" in listed and "CancelledError" in listed, sorted(listed)
        errors_module = importlib.import_module("qsmy_deepseek_locator.errors")
        for name in sorted(listed):
            exc_type = getattr(errors_module, name, None)
            assert exc_type is not None, f"README 表里的 {name} 在代码里不存在"
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


class TestLocateAndDrawDrawingParams:
    """Locator.locate_and_draw 的绘制参数必须**真的生效**。

    这条是评审抓出来的实锤：它以前把绘制参数塞进 **kwargs 再转给 locate()，
    而 locate() 没有这些形参 —— Locator(font_path=…) 在这条路上整条失效，
    中文标签照样画成方块，调用方却以为自己指定好了；传 font_path 更是直接 TypeError。
    """

    def test_font_path_reaches_the_drawing_step(self, fake_client, sample_png, tmp_path, cjk_font):
        from qsmy_deepseek_locator import drawing

        drawing._font_cache.clear()
        locator = Locator(client=fake_client(BOX), api_key="sk-test", font_path=cjk_font)
        locator.locate_and_draw(sample_png, "方块", output=tmp_path / "o.png")
        assert any(Path(key[0]) == Path(cjk_font) for key in drawing._font_cache), list(drawing._font_cache)

    def test_explicit_font_path_argument_is_accepted(self, fake_client, sample_png, tmp_path, cjk_font):
        """显式传也不能 TypeError（上一版就是这么炸的）。"""
        locator = Locator(client=fake_client(BOX), api_key="sk-test")
        _, annotated = locator.locate_and_draw(
            sample_png, "方块", output=tmp_path / "o.png", font_size=30, font_path=cjk_font
        )
        # 这里原先还断言了 annotated.size == (400, 300) —— 那是**恒真**的（画布尺寸与
        # font_path 无关），删掉；能把"字体真的换上了"钉住的只有同名测试里那条缓存断言。

    def test_other_style_params_are_not_swallowed(self, fake_client, sample_png, tmp_path):
        """box_width 这些也要真的走下去（不能只修 font_path 一条路）。"""
        locator = Locator(client=fake_client(BOX), api_key="sk-test")
        locator.locate_and_draw(sample_png, "方块", output=tmp_path / "a.png", box_width=1)
        locator.locate_and_draw(sample_png, "方块", output=tmp_path / "b.png", box_width=12)
        with Image.open(tmp_path / "a.png") as a, Image.open(tmp_path / "b.png") as b:
            assert a.tobytes() != b.tobytes(), "线宽参数被丢掉了（两张图一模一样）"


class TestTimeoutReachesTheClient:
    """生效后的读超时必须真的传到客户端（评审发现：自备客户端这条整条失效）。

    这是个**功能静默失效**类的问题：RequestsVisionClient 在本次改动之前只读自己构造时的
    timeout，于是 Locator(timeout=30) / QSML_TIMEOUT 在那边毫无作用 —— 安卓上只能用自备
    客户端，那个开关就是唯一能调的超时，而现象只是"我明明设了 30 秒它还是卡了 90 秒"。
    """

    def test_effective_timeout_is_passed_through(self, fake_client, sample_png):
        client = fake_client(BOX)
        Locator(client=client, api_key="sk-test", timeout=7.5).locate(sample_png, "方块")
        assert client.calls[-1]["timeout"] == 7.5

    def test_single_call_override_wins(self, fake_client, sample_png):
        """单次覆盖也要压过构造时那个值（负向对照：不能恒传构造值）。"""
        client = fake_client(BOX)
        Locator(client=client, api_key="sk-test", timeout=7.5).locate(sample_png, "方块", timeout=3.0)
        assert client.calls[-1]["timeout"] == 3.0

    def test_old_style_client_degrades_with_a_visible_warning(self, sample_png):
        """按旧协议写的自备客户端（签名里没有 timeout）不能被这次升级搞崩。

        它收不到本次超时，但**必须继续能用**；同时要告警 —— 静默降级就等于把
        "超时设置不生效"变成一条永远查不出来的现象。
        """
        from qsmy_deepseek_locator import ChatReply

        class _OldClient:
            """照 0.1.2 的协议写的客户端：没有 timeout，也没有 **kwargs。"""

            def complete(self, messages, *, settings=None, on_event=None, log=None):
                self.called = True
                return ChatReply(text=BOX, model="old")

        client = _OldClient()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = Locator(client=client, api_key="sk-test", timeout=12.0).locate(sample_png, "方块")
        assert client.called, "旧式客户端被 TypeError 打挂了"
        assert result.detections, "降级后识别结果不该受影响"
        messages = [str(w.message) for w in caught if "timeout" in str(w.message)]
        assert messages, "静默丢掉了 timeout，用户无从得知"
        assert "12.0" in messages[0]
        assert caught[0].filename == __file__, caught[0].filename


    def test_warning_points_at_the_caller_on_the_module_level_entry(self, sample_png):
        """模块级 locate() 这条入口也要指到用户那一行（栈更深，最容易指错）。"""
        from qsmy_deepseek_locator import ChatReply, locate as module_locate

        class _OldClient:
            def complete(self, messages, *, settings=None, on_event=None, log=None):
                return ChatReply(text=BOX, model="old")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            module_locate(sample_png, "方块", client=_OldClient(), api_key="sk-test")
        timeout_warnings = [w for w in caught if "timeout" in str(w.message)]
        assert timeout_warnings, "模块级入口没有告警"
        assert timeout_warnings[0].filename == __file__, timeout_warnings[0].filename

    def test_warning_points_at_the_caller_on_the_cli_entry(self, sample_png, monkeypatch, capsys):
        """CLI 这条入口同样要指到用户那一行 —— 指到 cli.py 里的话，用户会去改错地方。"""
        from qsmy_deepseek_locator import ChatReply, cli

        class _OldClient:
            def complete(self, messages, *, settings=None, on_event=None, log=None):
                return ChatReply(text=BOX, model="old")

        monkeypatch.setattr(cli, "Locator", lambda **kw: Locator(client=_OldClient(), api_key="sk-test"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cli.main([str(sample_png), "-t", "方块", "--no-draw", "-q"])
        timeout_warnings = [w for w in caught if "timeout" in str(w.message)]
        assert timeout_warnings, "CLI 入口没有告警"
        # CLI 这条路上，库内最外那一帧**就是** cli.py 里的 main()（用户没写调用代码，
        # 他的"代码"就是那条命令行）—— 所以这里断言的是"没有停在库更里面"：
        # 落在 cli.py 或更外层都算对，落回 locate.py / http_client.py 才算指错。
        assert "locate.py" not in timeout_warnings[0].filename, timeout_warnings[0].filename
        capsys.readouterr()


class TestResultJsonContract:
    """公开 API 里**每一条写文件的路**都要在契约内 —— 这一族是评审补出来的欠账。

    LocateResult.save() 曾经是漏网的：调用方按 README 只写 except LocatorError，
    结果 --json 写不进去时看到裸 OSError，还得回去补 catch。
    """

    def test_save_wraps_oserror(self, tmp_path, monkeypatch):
        from qsmy_deepseek_locator import LocateResult

        def _boom(self, *args, **kwargs):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(Path, "write_text", _boom)
        with pytest.raises(WriteError) as excinfo:
            LocateResult().save(tmp_path / "r.json")
        assert isinstance(excinfo.value.__cause__, OSError)
        assert "结果 JSON 写入失败" in str(excinfo.value)

    def test_save_wraps_typeerror_from_unserializable_raw_items(self, tmp_path):
        """raw_items 里塞了不可序列化的东西时，json.dumps 抛的是 TypeError 不是 OSError。

        这条是评审按注释点名实测出来的：注释写着"TypeError/ValueError 也要包"，
        except 子句里却只有 (OSError, ValueError)，于是这种输入照样裸着出去。
        """
        from qsmy_deepseek_locator import LocateResult

        result = LocateResult(raw_items=[object()])
        with pytest.raises(WriteError) as excinfo:
            result.save(tmp_path / "r.json", include_raw=True)
        assert isinstance(excinfo.value.__cause__, TypeError)
        assert "结果 JSON 写入失败" in str(excinfo.value)

    def test_save_still_works_normally(self, tmp_path):
        """负向对照：正常路径不许被新异常误伤。"""
        from qsmy_deepseek_locator import LocateResult
        target = LocateResult().save(tmp_path / "嵌套" / "r.json")
        assert target.exists() and target.read_text(encoding="utf-8")

    def test_cli_reports_json_failure_without_traceback(self, tmp_path, monkeypatch, sample_png, capsys):
        """CLI 的 --json 写不进去时要「错误：…」+ 退出码 1，而不是吐 traceback。

        让识别**真的成功**（把 fake_client 塞进 CLI 造的 Locator），只让最后一次写盘失败 ——
        否则"图片不存在先报错"也会给出退出码 1，那条件断言等于没测（上一版就是这样）。
        """
        from qsmy_deepseek_locator import ChatReply, cli, locate as locate_module

        class _Client:
            def complete(self, messages, *, settings=None, on_event=None, timeout=None, log=None):
                return ChatReply(text=BOX, model="fake")

        monkeypatch.setattr(cli, "Locator", lambda **kw: Locator(client=_Client(), api_key="sk-test"))
        monkeypatch.setattr(
            Path, "write_text",
            lambda self, *a, **k: (_ for _ in ()).throw(OSError(28, "No space left on device")),
        )
        rc = cli.main([
            str(sample_png), "-t", "方块", "--json", str(tmp_path / "r.json"),
            "--no-draw", "-q",
        ])
        assert rc == 1
        err = capsys.readouterr().err
        assert "错误：" in err and "结果 JSON 写入失败" in err
        assert "Traceback" not in err
        # 证明走的确实是这条分支：识别已经跑到"该写文件"这一步了，只是被写盘失败拦下
        assert not (tmp_path / "r.json").exists()
        assert "标注图" not in err, "--no-draw 下不该去画图"


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
