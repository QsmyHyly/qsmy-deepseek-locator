"""命令行自测：参数解析 / 退出码 / 只造图不花钱的路径。"""

from __future__ import annotations

import json

import pytest

from qsmy_deepseek_locator.cli import main


class TestArgumentHandling:
    def test_version_exits_zero(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(["--version"])
        assert excinfo.value.code == 0
        assert "qsmy-deepseek-locator" in capsys.readouterr().out

    def test_missing_image_is_usage_error(self):
        with pytest.raises(SystemExit) as excinfo:
            main([])
        assert excinfo.value.code == 2

    def test_thinking_flags_are_exclusive(self, sample_png):
        with pytest.raises(SystemExit) as excinfo:
            main([str(sample_png), "--thinking", "--no-thinking"])
        assert excinfo.value.code == 2


class TestBenchCommand:
    def test_images_only_returns_zero(self, tmp_path, capsys):
        code = main(["bench", "--images-only", "--count", "1", "--n-shapes", "2", "--out", str(tmp_path)])
        assert code == 0
        out = capsys.readouterr().out
        assert "报告：" in out
        report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
        assert report["report"] is None
        assert len(report["samples"]) == 1
        assert len(report["samples"][0]["ground_truth"]) == 2


class TestLocateCommand:
    def test_missing_api_key_exits_one(self, sample_png, capsys):
        """没有 Key 时必须**明确失败**（退出码 1 + 说清怎么配），而不是吐一堆假坐标。"""
        code = main([str(sample_png), "-t", "猫", "--no-draw", "-q"])
        assert code == 1
        err = capsys.readouterr().err
        assert "DEEPSEEK_API_KEY" in err

    def test_missing_file_exits_one(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")  # 让它走到读图那一步
        code = main([str(tmp_path / "nope.png"), "-t", "猫", "--no-draw", "-q"])
        assert code == 1
        assert "不存在" in capsys.readouterr().err
