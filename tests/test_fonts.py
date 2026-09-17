"""中文字体探测自测：认不认安卓、名不名字、找不着时告不告警。

这一组针对反馈文档 P0-1 / P1-1 / P1-2 三条，全部对应真机上踩过的坑：

    P0-1  探测目录里没有 /system/fonts，安卓上必然返回 None
    P1-1  resolve_font(size) 只收尺寸，调用方没有正式途径指定字体（只能 monkeypatch）
    P1-2  降级到内置位图字体是**静默**的，中文变方块而图照常出

断言要点是**可证伪**：既有「找不到必须告警」，也有「找得到就不许告警」的负向对照
（只写前者的话，一个"永远告警"的实现也能过）。
"""

from __future__ import annotations

import os
import shutil
import sys
import warnings
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from qsmy_deepseek_locator import drawing
from qsmy_deepseek_locator.config import Settings
from qsmy_deepseek_locator.drawing import resolve_font


# 公共设施（is_real_font_file / find_any_cjk_font / cjk_font 夹具）放在 conftest.py，
# 因为 test_font_edge_cases.py 也要用同一套 —— 「同一个实现只允许有一份」。
from tests.conftest import find_any_cjk_font as _find_any_cjk_font  # noqa: E402
from tests.conftest import is_real_font_file as _is_real_font_file  # noqa: E402


def _silence_cjk_search(monkeypatch, tmp_path):
    """把**所有**字体来源掐掉：探测目录换成空目录、候选名换成不存在的东西。

    必须两头都换：探测是「名字 × 目录」的双层循环，只清掉目录的话，
    一旦以后有人加了绝对路径候选，这条负向对照就会悄悄失效（假绿）。
    """
    monkeypatch.setattr(drawing, "_FONT_DIRS", [tmp_path / "空的字体目录"])
    monkeypatch.setattr(drawing, "_FONT_CANDIDATES", ["这个字体不存在-用来断言告警.ttf"])
    monkeypatch.delenv(drawing.FONT_PATH_ENV, raising=False)
    monkeypatch.delenv(drawing.FONT_DIR_ENV, raising=False)


class TestAndroidDiscovery:
    """P0-1：安卓的字体目录与国产 ROM 的字体名必须都在探测范围里。"""

    def test_android_dirs_come_first(self):
        """安卓两个目录必须排在最前（移动端优先），且路径就是安卓上的真实位置。

        用 as_posix() 而不是 str()：Windows 上 Path("/system/fonts") 会变成反斜杠形式，
        拿 str() 比会在开发机上假红 —— 而这条断言要验的是**写在源码里的那个路径**，
        不是它在本机上的渲染形式。
        """
        dirs = [p.as_posix() for p in drawing._FONT_DIRS]
        assert dirs[0] == "/system/fonts"
        assert dirs[1] == "/product/fonts"

    def test_android_font_names_are_candidates(self):
        """小米 / 华为 / 安卓自带的兜底中文字体都要在候选名单里。"""
        for name in (
            "MiSans-Regular.ttf", "MiSansVF.ttf",
            "HarmonyOS_Sans_SC_Regular.ttf",
            "DroidSansFallback.ttf", "DroidSansFallbackFull.ttf",
        ):
            assert name in drawing._FONT_CANDIDATES, name

    def test_noto_is_still_ahead_of_latin_only_fonts(self):
        """NotoSansCJK 必须排在 Arial 之前 —— 否则会先命中一个没有中文字形的字体。"""
        assert (drawing._FONT_CANDIDATES.index("NotoSansCJK-Regular.ttc")
                < drawing._FONT_CANDIDATES.index("Arial.ttf"))


def _ink_ratio(font, ch: str) -> float:
    """这个字体渲染 ch 时，画布上有多少比例是"墨"（非背景像素）。

    这是**独立于 _renders_cjk 的另一套判据**：它只回答"这个字体能不能画出这个字符"，
    不回答"这个字体有没有中文字形"。两者分开，"用 A 挑样本、用被测判据验结论"才不循环。
    """
    mask = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(mask).text((4, 4), ch, font=font, fill=255)
    ink = sum(1 for pixel in mask.tobytes() if pixel)
    return ink / (64 * 64)


def _renders_ink(path: str, ch: str) -> bool:
    """独立判据的"能出墨"版：打不开、或画出来是空白，都算 False。"""
    try:
        font = ImageFont.truetype(path, size=24)
    except Exception:  # noqa: BLE001 - 探测路径，坏文件不该让自测崩
        return False
    return _ink_ratio(font, ch) > 0.001


def _find_ink_rendering_latin_font() -> str | None:
    """找一个**独立判据**下确定"能出拉丁字形"的字体文件，供负向对照当样本。

    刻意不按 _renders_cjk 挑（那正是上一版的循环）：挑的依据只有两条 ——
    能打开、渲染 "A" 有墨。找不到就返回 None（调用方 skip），不硬编码平台路径。
    注意这里**不要求**它不含中文：那正是接下来要被测的结论，不能在挑样本时预设。
    """
    for name in ("Arial.ttf", "segoeui.ttf", "DejaVuSans.ttf", "Roboto-Regular.ttf", "DroidSans.ttf"):
        for directory in drawing._FONT_DIRS:
            candidate = directory / name
            if candidate.exists() and _renders_ink(str(candidate), "A"):
                return str(candidate)
    return None

class TestRenderVerification:
    """文件名不足以判断有没有中文字形，得真渲染一遍（_renders_cjk）。"""

    def test_latin_only_font_is_rejected(self, tmp_path):
        """负向对照：一个确定只有拉丁字形的字体，必须被判为"不含中文"。

        **选样本的判据必须独立于被测判据**（上一版不是，评审抓到了）：上一版用
        `not drawing._renders_cjk(p)` 去挑样本，再断言 `_renders_cjk(p) is False` ——
        一个"永远返回 False"的坏实现照样全绿，因为选样本这一步自己就把它选出来了。
        现在改成：先用**独立判据**证明样本是个能出墨的字体（渲染 ASCII "A" 有笔画），
        再要求它被中文判据拒绝。坏实现（恒 False）会在"A 出墨"这步先红。
        """
        latin = _find_ink_rendering_latin_font()
        if not latin:
            pytest.skip("本机找不到确定不含中文、但能正常出拉丁字形的字体")
        # 独立性证明：这个字体真的能画东西（不是个坏文件），且画的是拉丁字母
        assert _renders_ink(latin, "A") is True, "样本连 ASCII 都画不出来，测不出东西"
        assert drawing._renders_cjk(latin) is False, f"{latin} 被判成含中文字形了？"


    def test_cjk_font_is_accepted(self, cjk_font):
        """正向对照：真的含中文字形的字体必须被判为可用。"""
        assert drawing._renders_cjk(cjk_font) is True

    def test_broken_path_is_false_not_raise(self):
        """打不开的路径返回 False，不抛异常（探测路径上不许因为一个坏文件整轮挂掉）。"""
        assert drawing._renders_cjk("/确定/不存在/的/字体.ttf") is False

    def test_find_font_file_never_returns_a_latin_only_file(self, cjk_font, tmp_path, monkeypatch):
        """_find_font_file 必须**真的跳过**排在前面但只有拉丁字形的候选。

        上一版只是断言"找回来的那个确实含中文"—— 那是**同义反复**：found 本来就是
        _find_font_file 用同一个判据筛出来的。评审点得对：那种写法测不出"没筛"。
        现在把两个文件按候选顺序摆好（拉丁字体在前、中文字体在后），直接看它能不能
        从第一个身上跨过去 —— 只挑到中文那个才算通过。
        """
        latin = _find_ink_rendering_latin_font()
        if not latin:
            pytest.skip("本机找不到能出拉丁字形的对照字体")

        fonts = tmp_path / "fonts"
        fonts.mkdir()
        names = ["A-First-Latin.ttf", "B-Second-CJK.ttf"]
        shutil.copyfile(latin, fonts / names[0])
        shutil.copyfile(cjk_font, fonts / names[1])
        monkeypatch.setattr(drawing, "_FONT_DIRS", [fonts])
        monkeypatch.setattr(drawing, "_FONT_CANDIDATES", names)
        drawing._renders_cjk.cache_clear()

        found = drawing._find_font_file()
        assert found is not None, "明明摆了中文字体却什么都没找回来"
        assert Path(found).name == names[1], f"挑中了排在前面但只有拉丁字形的 {Path(found).name}"



class TestResolveFontInjection:
    """P1-1：调用方要有正式途径指定字体，而不是 monkeypatch 私有函数。"""

    def test_explicit_font_path_is_used(self, cjk_font):
        font = resolve_font(18, font_path=cjk_font)
        assert _is_real_font_file(font)
        assert Path(font.path) == Path(cjk_font)

    def test_explicit_font_path_beats_environment(self, cjk_font, monkeypatch):
        """参数优先于环境变量：两处都给了时，用的是参数那个。"""
        other = _find_any_cjk_font()
        monkeypatch.setenv(drawing.FONT_PATH_ENV, str(other))
        font = resolve_font(18, font_path=cjk_font)
        assert Path(font.path) == Path(cjk_font)

    def test_env_font_path_is_used(self, cjk_font, monkeypatch):
        """QSML_FONT_PATH 生效 —— 打包 / App 侧不用改代码就能指定字体。"""
        monkeypatch.setenv(drawing.FONT_PATH_ENV, cjk_font)
        font = resolve_font(18)
        assert Path(font.path) == Path(cjk_font)

    def test_env_font_dir_is_searched(self, cjk_font, monkeypatch, tmp_path):
        """QSML_FONT_DIR 能把探测范围限制到指定目录（并沿用候选文件名）。"""
        target = tmp_path / "fonts"
        target.mkdir()
        # 用候选名单里的名字，才能验证"按名字去这个目录里找"
        name = drawing._FONT_CANDIDATES[0]
        shutil.copyfile(cjk_font, target / name)
        monkeypatch.setenv(drawing.FONT_DIR_ENV, str(target))
        monkeypatch.setattr(drawing, "_FONT_CANDIDATES", [name])
        font = resolve_font(18)
        assert Path(font.path) == target / name

    def test_env_font_dir_does_not_blindly_trust_names(self, monkeypatch, tmp_path):
        """限定了目录也**不能**免掉渲染校验：目录里放个没有中文的字体，必须继续往别处找。

        这条是"名字优先 + 渲染校验"两条规则的交点，也是安卓上最容易踩的那一格。
        """
        target = tmp_path / "fonts"
        target.mkdir()
        name = "DroidSans.ttf"
        # 造一个只含拉丁字形的 TTF：拿本机一个不含中文的字体来冒充，找不到就 skip
        latin = None
        for candidate_name in ("Arial.ttf", "DejaVuSans.ttf"):
            for directory in drawing._FONT_DIRS:
                candidate = directory / candidate_name
                if candidate.exists() and not drawing._renders_cjk(str(candidate)):
                    latin = candidate
                    break
            if latin:
                break
        if not latin:
            pytest.skip("本机找不到确定不含中文字形的字体")
        shutil.copyfile(latin, target / name)
        monkeypatch.setenv(drawing.FONT_DIR_ENV, str(target))
        monkeypatch.setattr(drawing, "_FONT_CANDIDATES", [name])
        monkeypatch.setattr(drawing, "_FONT_DIRS", [target])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            font = resolve_font(18)
        assert not _is_real_font_file(font), \
            "把 Roboto 冒充的中文字体当真了 —— 渲染校验没生效"
        assert any("方块" in str(w.message) for w in caught), "没找到中文字体却没告警"

    def test_cache_is_keyed_by_path_and_size(self, cjk_font):
        """同一文件同一字号只开一次；换了字体文件不会串味（键必须含路径）。"""
        first = resolve_font(20, font_path=cjk_font)
        assert resolve_font(20, font_path=cjk_font) is first
        # 键是 (路径, 字号, 来源)。第三个分量是**本次改动加的**：同一个路径"用户显式给的"
        # 与"探测找出来的"要走不同分支（显式的那条不再问环境变量、坏了还要往下探），
        # 用同一个键会让后一次调用拿到前一次分支的结论。
        assert (cjk_font, 20, "explicit") in drawing._font_cache

    def test_draw_actually_uses_the_font_not_just_accepts_it(self, cjk_font, monkeypatch, tmp_path):
        """不止"参数收得下"：真的要换掉字体。

        只看签名的话，一个把 font_path 收下就扔掉的实现照样能过 —— 那样中文标签
        仍会是方块，而用户以为自己已经指定好了。所以这里直接看**画出来的像素**。
        """
        from PIL import Image
        img = Image.new("RGB", (240, 80), (255, 255, 255))
        detection = [{"bbox_2d": [0.1, 0.1, 0.9, 0.6], "label": "中文标签"}]
        with_font = drawing.draw(img, detection, font_size=22, font_path=cjk_font).tobytes()
        # 把所有字体来源掐掉，得到"方块版"渲染（这条路径本来就该告警，这里刻意吞掉它）
        _silence_cjk_search(monkeypatch, tmp_path)
        drawing._font_cache.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            without = drawing.draw(img, detection, font_size=22).tobytes()
        assert with_font != without, "指定字体前后的像素完全一样 —— font_path 被丢掉了"

    def test_locate_to_file_inherits_font_path_from_settings(
        self, fake_client, sample_png, cjk_font, tmp_path
    ):
        """Settings.font_path 一路活到打标：构造 Locator 时给一次就够。

        这是「值必须能活到 draw()」那条取舍的兑现点 —— 中间任何一层漏传，
        这里就看不到那个字体进缓存。
        """
        from qsmy_deepseek_locator import Locator
        out = tmp_path / "o.png"
        locator = Locator(
            client=fake_client('[{"bbox_2d": [0.2, 0.2, 0.8, 0.8], "label": "猫"}]'),
            api_key="sk-test", font_path=cjk_font,
        )
        assert locator.settings.font_path == cjk_font
        result = locator.locate_to_file(sample_png, "猫", out)
        assert result.annotated_path == str(out)
        assert out.exists()
        # 缓存键是 (字体路径, 字号)：能在里面看到它，就说明打标那一步真的用了它
        assert any(Path(key[0]) == Path(cjk_font) for key in drawing._font_cache),             list(drawing._font_cache)

    def test_draw_accepts_font_path(self, cjk_font, monkeypatch, tmp_path):
        """draw() 也要能显式指定字体 —— 而且**真的用上**，不是收下就扔。"""
        from PIL import Image
        img = Image.new("RGB", (120, 80), (255, 255, 255))
        out = drawing.draw(img, [], font_path=cjk_font)
        assert out.size == (120, 80)
        # 只断言尺寸的话，一个把 font_path 收下就丢掉的实现照样能过（那正是评审挑剔的那条"假绿"）。
        # 判据：打标用过的字体进了缓存，而且它就是能渲染中文的那个文件。
        assert any(Path(key[0]) == Path(cjk_font) for key in drawing._font_cache), list(drawing._font_cache)
        _silence_cjk_search(monkeypatch, tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # 这条路径本来就该告警，本用例测的是字体不是告警
            fallback = resolve_font(22)
        assert not _is_real_font_file(fallback), "字源被掐掉后不该还能开出一个真字体文件"
        assert str(getattr(fallback, "path", "")) != cjk_font


class TestMissingFontWarns:
    """P1-2：找不到中文字体时必须告警，而且只有这一种情况才告警。"""

    def test_warns_when_no_cjk_font(self, monkeypatch, tmp_path):
        _silence_cjk_search(monkeypatch, tmp_path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            font = resolve_font(22)
            messages = [str(w.message) for w in caught]
        assert not _is_real_font_file(font), "本该退回内置字体，却开到了某个字体文件"
        assert any("方块" in m for m in messages), messages
        assert any("QSML_FONT_PATH" in m for m in messages), "告警里得给出怎么修"

    def test_warns_only_once_per_process(self, monkeypatch, tmp_path):
        """告警去重：draw 会按不同字号多次调用 resolve_font，一次打标不该刷十几条。"""
        _silence_cjk_search(monkeypatch, tmp_path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolve_font(12)
            resolve_font(22)
            resolve_font(32)
        assert len(caught) == 1, [str(w.message) for w in caught]

    def test_does_not_warn_when_a_cjk_font_exists(self, cjk_font, monkeypatch):
        """**负向对照**：找得到中文字体时一条告警都不许发。

        没有这一条的话，「永远告警」的实现也能通过上面那两条断言。
        """
        monkeypatch.setenv(drawing.FONT_PATH_ENV, cjk_font)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            font = resolve_font(22)
            messages = [str(w.message) for w in caught]
        assert _is_real_font_file(font)
        assert messages == [], messages

    def test_draw_also_warns_end_to_end(self, monkeypatch, tmp_path):
        """端到端：draw(带中文标签) 在没字体时也得告警，而不是只有 resolve_font 单测覆盖。"""
        from PIL import Image
        _silence_cjk_search(monkeypatch, tmp_path)
        img = Image.new("RGB", (200, 120), (255, 255, 255))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            drawing.draw(img, [{"bbox_2d": [0.1, 0.1, 0.5, 0.5], "label": "猫"}])
        assert any("方块" in str(w.message) for w in caught)


class TestWarningPointsAtUserCode:
    """告警的位置必须落在**用户自己写的那一行**。

    为什么单开一族：告警文案再对，位置指到 drawing.py 内部时用户看到的是一句
    「drawing.py:298 有警告」—— 他既不知道哪次调用触发的，也不知道该改哪儿。
    而"算准帧数"这件事恰恰是手写 stacklevel 最容易错的地方（上一版就错了两次），
    所以三条进入路径各留一条断言钉住。
    """

    def test_warns_at_the_direct_call_line(self, monkeypatch, tmp_path):
        _silence_cjk_search(monkeypatch, tmp_path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            expected = sys._getframe().f_lineno + 1
            resolve_font(20)
        assert len(caught) == 1
        assert Path(caught[0].filename) == Path(__file__), caught[0].filename
        assert caught[0].lineno == expected

    def test_warns_at_the_draw_call_line(self, monkeypatch, tmp_path):
        from PIL import Image
        _silence_cjk_search(monkeypatch, tmp_path)
        img = Image.new("RGB", (80, 60), (255, 255, 255))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            expected = sys._getframe().f_lineno + 1
            drawing.draw(img, [])
        assert len(caught) == 1
        assert Path(caught[0].filename) == Path(__file__), caught[0].filename
        assert caught[0].lineno == expected

    def test_warns_at_the_call_line_when_the_given_path_is_broken(self, monkeypatch, tmp_path):
        """显式给了打不开的 font_path：这一支中间多穿一层，最容易指错。"""
        _silence_cjk_search(monkeypatch, tmp_path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            expected = sys._getframe().f_lineno + 1
            resolve_font(20, font_path=str(tmp_path / "没有这个字体.ttf"))
        assert len(caught) == 1
        assert Path(caught[0].filename) == Path(__file__), caught[0].filename
        assert caught[0].lineno == expected


class TestSettingsFontPath:
    """Settings.font_path：字体路径要能跟着配置走（构造 Locator 时给一次就够）。"""

    def test_from_env_reads_the_variable(self, monkeypatch):
        monkeypatch.setenv(drawing.FONT_PATH_ENV, "/tmp/字体.ttf")
        assert Settings.from_env().font_path == "/tmp/字体.ttf"

    def test_env_name_matches_drawing_constant(self):
        """config 里那个字面量必须与 drawing.FONT_PATH_ENV 一模一样。

        config 刻意不 import drawing（不想为了一个字符串把 PIL 拉起来），
        所以两边的一致性只能靠这条断言守着 —— 写岔了就是「设了环境变量没生效」那种鬼故事。
        """
        import re
        source = (Path(__file__).resolve().parents[1] / "src" / "qsmy_deepseek_locator"
                  / "config.py").read_text(encoding="utf-8")
        quoted = re.search(r'font_path=_env_str\("([^"]+)"\)', source)
        assert quoted and quoted.group(1) == drawing.FONT_PATH_ENV

    def test_merged_carries_font_path(self):
        base = Settings(font_path="/tmp/a.ttf")
        assert base.merged(font_path="/tmp/b.ttf").font_path == "/tmp/b.ttf"
        assert base.merged(thinking=False).font_path == "/tmp/a.ttf", "无关字段的覆盖不该抹掉它"

    def test_merged_ignores_none_like_other_fields(self):
        assert Settings(font_path="/tmp/a.ttf").merged(font_path=None).font_path == "/tmp/a.ttf"
