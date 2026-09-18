"""探测型测试图自测：图画得出来、真值是算出来的、以及 render_scaled 的那条核心性质。

为什么 render_scaled 的断言最要紧：它全部的价值在于「同一场景在不同栅格上的真值**逐字节
相同**」—— 只有真值相同，各分辨率下测出的精度差异才只能归因于栅格 / 服务端重采样，而不是
「场景变了」。这条性质一旦破了，分辨率扫描的结论就全是错的，而**图本身看上去完全正常**，
所以只能靠测试钉住，肉眼查不出来。
"""

from __future__ import annotations

from PIL import Image

from qsmy_deepseek_locator.bench_generators import (
    BAND_SPECS,
    MARKER_FRACTIONS,
    TEXT_FONT_SIZES,
    make_band_image,
    make_marker_image,
    make_text_image,
    markers_to_gt,
    random_codes,
    render_scaled,
)
from qsmy_deepseek_locator.bench_shapes import PALETTE, Sample, Shape


class TestMarkerImage:
    def test_writes_real_png_and_truth(self, tmp_path):
        p = tmp_path / "m.png"
        truth = make_marker_image(p, 900, 720)
        assert p.exists() and p.stat().st_size > 0
        assert len(truth) == len(MARKER_FRACTIONS)
        with Image.open(p) as im:
            assert im.size == (900, 720)

    def test_labels_are_palette_names_only(self, tmp_path):
        truth = make_marker_image(tmp_path / "m.png", 900, 720)
        for item in truth:
            # 只写颜色名：加了「圆形」后缀会与 MARKER_PROMPT 打架，标签准确率恒为 0%
            assert item["label"] in PALETTE

    def test_positions_follow_fractions(self, tmp_path):
        truth = make_marker_image(tmp_path / "m.png", 800, 600)
        for item, (fx, fy) in zip(truth, MARKER_FRACTIONS):
            assert item["cx"] == fx * 800
            assert item["cy"] == fy * 600


class TestMarkersToGt:
    def test_unit_range_and_center_match(self, tmp_path):
        truth = make_marker_image(tmp_path / "m.png", 900, 720)
        gt = markers_to_gt(truth, 900, 720)
        assert len(gt) == len(truth)
        for item in gt:
            x1, y1, x2, y2 = item["bbox_2d"]
            assert 0.0 <= x1 < x2 <= 1.0
            assert 0.0 <= y1 < y2 <= 1.0
            assert item["match"] == "center"        # 点位判定：圆点直径只有短边的 9%
            assert item["expect_shape"] is False    # 提示词只问颜色，不判形状
            assert item["color_name"] == item["label"]


class TestTextImage:
    def test_ladder_truth(self, tmp_path):
        codes = random_codes(len(TEXT_FONT_SIZES))
        truth = make_text_image(tmp_path / "t.png", 900, 720, TEXT_FONT_SIZES, codes)
        assert [t["font_px"] for t in truth] == list(TEXT_FONT_SIZES)
        assert [t["code"] for t in truth] == codes
        assert [t["index"] for t in truth] == list(range(1, len(TEXT_FONT_SIZES) + 1))


class TestRandomCodes:
    def test_deterministic(self):
        assert random_codes(5, seed=42) == random_codes(5, seed=42)

    def test_shape_and_alphabet(self):
        codes = random_codes(50)
        assert all(len(c) == 4 for c in codes)
        # 易混字符必须不在字母表里，否则判读歧义会被算到模型头上
        for ch in "0O1I2Z5S8B":
            assert all(ch not in c for c in codes)


class TestBandImage:
    def test_truth_matches_specs(self, tmp_path):
        truth = make_band_image(tmp_path / "b.png", 900, 720, BAND_SPECS)
        assert [(t["spacing"], t["lines"]) for t in truth] == list(BAND_SPECS)
        assert [t["index"] for t in truth] == list(range(1, len(BAND_SPECS) + 1))


class TestRenderScaled:
    @staticmethod
    def _reference() -> Sample:
        names = list(PALETTE.items())[:3]
        return Sample(
            name="ref", path="ref.png", width=900, height=720,
            shapes=[
                Shape(kind="rect", color_name=names[0][0], rgb=names[0][1],
                      bbox_px=(100.0, 80.0, 300.0, 220.0)),
                Shape(kind="circle", color_name=names[1][0], rgb=names[1][1],
                      bbox_px=(500.0, 400.0, 700.0, 600.0)),
                Shape(kind="triangle", color_name=names[2][0], rgb=names[2][1],
                      bbox_px=(650.0, 100.0, 860.0, 300.0)),
            ],
        )

    def test_ground_truth_only_differs_by_rounding(self, tmp_path):
        # 上游原文声称「逐字节相同」，实测有 1e-4 量级的舍入（来源 round(v*scale, 1)）：
        # 倍数干净的栅格看不出来，2400x1920 / 1280x1024 才会露。这里钉成 ≤1e-4 而不是「相等」——
        # 把错误前提写进断言，等于替后来人担保一件不成立的事。
        ref = self._reference()
        base = ref.ground_truth()
        for w, h in [(450, 360), (1800, 1440), (2400, 1920), (1280, 1024)]:
            out = render_scaled(ref, w, h, tmp_path / ("s%d.png" % w))
            gt = out.ground_truth()
            assert len(gt) == len(base)
            for want, got in zip(base, gt):
                for x, y in zip(want["bbox_2d"], got["bbox_2d"]):
                    assert abs(x - y) <= 1e-4, (w, h, want, got)
            assert (out.width, out.height) == (w, h)
            assert len(out.shapes) == len(ref.shapes)

    def test_scaled_image_is_written(self, tmp_path):
        p = tmp_path / "s.png"
        render_scaled(self._reference(), 1800, 1440, p)
        with Image.open(p) as im:
            assert im.size == (1800, 1440)



class TestFontInjection:
    """字体注入点必须一直有效 —— 它保的是「评测素材跨机器可复现」。

    库不打包字体，默认解析出的是当前机器的系统字体；自带字体的调用方（旧演示项目、
    安卓 App）靠注入自己的 resolve_font，才能让图与历史版本逐字节相同。
    这条参数一旦被后来人当成"没人用的可选参数"删掉，那两边都会**静默**换成系统字体，
    图看上去完全正常，只有历史评测结论不再可比。
    """

    def test_injected_resolver_receives_font_sizes(self, tmp_path):
        from qsmy_deepseek_locator.drawing import resolve_font as lib_resolve

        seen = []

        def fake(size):
            seen.append(size)
            return lib_resolve(size)

        make_marker_image(tmp_path / "m.png", 200, 160, font_resolver=fake)
        assert seen, "注入的字体解析函数根本没被调用（参数被绕过了？）"

    def test_text_image_injects_for_both_fonts(self, tmp_path):
        from qsmy_deepseek_locator.drawing import resolve_font as lib_resolve

        seen = []

        def fake(size):
            seen.append(size)
            return lib_resolve(size)

        make_text_image(tmp_path / "t.png", 300, 240, [40, 20], ["AAAA", "BBBB"], font_resolver=fake)
        # 一张阶梯图上有两种字：行号与代码正文，都必须走注入的解析
        assert len(seen) >= 3, seen

