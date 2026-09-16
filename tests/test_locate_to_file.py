"""一行式出图入口 locate_to_file 的自测（全部离线）。

这里重点验的是**副作用与默认值**，而不是解析逻辑（那部分 test_locate.py 已经覆盖）：
文件到底有没有落盘、落在哪、默认开没开思考、传了 api_key 还会不会去读环境变量。
"""

from __future__ import annotations

import json

import pytest
from PIL import Image

from qsmy_deepseek_locator import Locator, locate_to_file, resolve_output_path
from qsmy_deepseek_locator.errors import LocatorError, MissingAPIKeyError

BOX = '[{"bbox_2d": [0.25, 0.25, 0.75, 0.75], "label": "蓝色方块"}]'


class TestOutputPath:
    """路径规范化：支持的后缀 / 补默认后缀 / 认不出就报错。"""

    def test_known_suffixes(self):
        assert resolve_output_path("a/b.png") == (__import__("pathlib").Path("a/b.png"), "PNG")
        assert resolve_output_path("b.JPG")[1] == "JPEG"
        assert resolve_output_path("c.webp")[1] == "WEBP"

    def test_no_suffix_gets_png(self):
        path, fmt = resolve_output_path("out")
        assert path.suffix == ".png" and fmt == "PNG"

    @pytest.mark.parametrize("bad", ["out.tga", "out.png.txt", "", None])
    def test_unusable_path_raises_value_error(self, bad):
        # 宁可当场报错，也不要偷偷换成别的格式：文件名写着 .tga 实际存成 PNG，
        # 下游按后缀读图时报的错会离现场非常远。
        with pytest.raises(ValueError):
            resolve_output_path(bad)


class TestLocateToFile:
    def test_writes_annotated_file(self, fake_client, sample_png, tmp_path):
        client = fake_client(BOX)
        out = tmp_path / "nested" / "annotated.png"   # 父目录不存在，应自动创建
        result = locate_to_file(sample_png, "找方块", out, client=client)

        assert len(result) == 1
        assert result.annotated_path == str(out)
        assert out.exists()
        with Image.open(out) as img:
            assert img.size == (400, 300)          # 画的是原图分辨率
            assert img.getpixel((200, 75)) == (255, 0, 0)

    def test_unusable_parent_dir_fails_before_the_model_is_called(self, fake_client, sample_png, tmp_path):
        """输出目录不可用要在调模型之前就报错，而且得是 LocatorError。

        把父路径先占成一个文件：旧实现要等画完才 mkdir，于是先花掉一次 API 调用，
        抛的还是裸 OSError —— 调用方的 except LocatorError 根本抓不住。
        """
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        client = fake_client("[]")
        with pytest.raises(LocatorError):
            locate_to_file(sample_png, "找方块", blocker / "out.png", client=client)
        assert client.calls == []          # 一次都没调模型

    def test_target_reaches_the_model(self, fake_client, sample_png, tmp_path):
        client = fake_client("[]")
        locate_to_file(sample_png, "登录按钮", tmp_path / "o.png", client=client)
        assert "登录按钮" in client.calls[0]["messages"][1]["content"][1]["text"]

    def test_prompt_overrides_target(self, fake_client, sample_png, tmp_path):
        client = fake_client("[]")
        locate_to_file(sample_png, "被忽略", tmp_path / "o.png",
                       prompt="自定义整段提问", client=client)
        assert client.calls[0]["messages"][1]["content"][1]["text"] == "自定义整段提问"

    def test_suffix_decides_format(self, fake_client, sample_png, tmp_path):
        client = fake_client(BOX)
        out = tmp_path / "o.jpg"
        locate_to_file(sample_png, "方块", out, client=client)
        with Image.open(out) as img:
            assert img.format == "JPEG"            # 后缀说了算，不看后缀就猜是另一回事

    def test_suffixless_output_is_png(self, fake_client, sample_png, tmp_path):
        client = fake_client(BOX)
        result = locate_to_file(sample_png, "方块", tmp_path / "noext", client=client)
        assert result.annotated_path.endswith(".png")
        assert (tmp_path / "noext.png").exists()

    def test_empty_result_still_writes_file(self, fake_client, sample_png, tmp_path):
        # 「模型没找到目标」不是失败：图照样落盘（等于原图），调用方不必额外判断
        client = fake_client("[]")
        out = tmp_path / "empty.png"
        result = locate_to_file(sample_png, "不存在的东西", out, client=client)
        assert result.empty and out.exists()
        assert "标注图" in result.describe()

    def test_annotated_path_in_json(self, fake_client, sample_png, tmp_path):
        client = fake_client(BOX)
        out = tmp_path / "o.png"
        result = locate_to_file(sample_png, "方块", out, client=client)
        assert json.loads(result.to_json())["annotated_path"] == str(out)

    def test_draw_kwargs_apply(self, fake_client, sample_png, tmp_path):
        client = fake_client(BOX)
        out = tmp_path / "o.png"
        locate_to_file(sample_png, "方块", out, client=client, colors=["blue"])
        with Image.open(out) as img:
            assert img.getpixel((200, 75)) == (0, 0, 255)

    def test_locator_method_reuses_client(self, fake_client, sample_png, tmp_path):
        client = fake_client(BOX)
        locator = Locator(client=client)
        locator.locate_to_file(sample_png, "方块", tmp_path / "a.png")
        locator.locate_to_file(sample_png, "方块", tmp_path / "b.png")
        assert len(client.calls) == 2
        assert (tmp_path / "a.png").exists() and (tmp_path / "b.png").exists()


class TestDefaultsAndOverrides:
    """默认值就是本函数的对外承诺，单独一条条钉住。"""

    def test_thinking_off_detail_original_by_default(self, fake_client, sample_png, tmp_path):
        client = fake_client("[]")
        locate_to_file(sample_png, "x", tmp_path / "o.png", client=client)
        settings = client.calls[0]["settings"]
        assert settings.thinking is False          # 默认显式关闭思考
        assert settings.image_detail == "original"  # 默认 original

    def test_thinking_none_falls_back_to_env(self, fake_client, sample_png, tmp_path, monkeypatch):
        monkeypatch.setenv("QSML_THINKING", "1")
        client = fake_client("[]")
        locate_to_file(sample_png, "x", tmp_path / "o.png", client=client, thinking=None)
        assert client.calls[0]["settings"].thinking is True

    def test_passed_api_key_beats_env(self, fake_client, sample_png, tmp_path, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
        client = fake_client("[]")
        locate_to_file(sample_png, "x", tmp_path / "o.png", client=client, api_key="sk-passed")
        assert client.calls[0]["settings"].api_key == "sk-passed"

    def test_env_key_used_when_not_passed(self, fake_client, sample_png, tmp_path, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
        client = fake_client("[]")
        locate_to_file(sample_png, "x", tmp_path / "o.png", client=client)
        assert client.calls[0]["settings"].api_key == "sk-from-env"

    def test_locator_level_kwargs_are_routed(self, fake_client, sample_png, tmp_path):
        # max_retries / max_side 属于 Locator 构造参数，模块级函数要能把它挑出来给构造器
        client = fake_client("[]")
        locate_to_file(sample_png, "x", tmp_path / "o.png", client=client,
                       max_retries=5, max_side=200)
        assert client.calls[0]["settings"].max_retries == 5


class TestRefusals:
    def test_missing_key_raises_and_writes_nothing(self, sample_png, tmp_path):
        out = tmp_path / "o.png"
        with pytest.raises(MissingAPIKeyError):
            locate_to_file(sample_png, "x", out)     # 环境变量已被 conftest 清空
        assert not out.exists()

    def test_bad_output_path_costs_no_api_call(self, fake_client, sample_png, tmp_path):
        client = fake_client("[]")
        with pytest.raises(ValueError):
            locate_to_file(sample_png, "x", tmp_path / "o.tga", client=client)
        assert client.calls == []                    # 路径先校验后调模型

    def test_use_tools_true_fails_loudly(self, fake_client, sample_png, tmp_path):
        # 预留参数：宁可报错也不静默忽略，否则调用方会以为工具已经开了
        client = fake_client("[]")
        out = tmp_path / "o.png"
        with pytest.raises(NotImplementedError) as excinfo:
            locate_to_file(sample_png, "x", out, client=client, use_tools=True)
        assert "工具" in str(excinfo.value)
        assert client.calls == [] and not out.exists()

    def test_use_tools_false_is_the_supported_path(self, fake_client, sample_png, tmp_path):
        client = fake_client("[]")
        result = locate_to_file(sample_png, "x", tmp_path / "o.png",
                                client=client, use_tools=False)
        assert result.annotated_path is not None
