"""解析层自测：容错解析 / 刻度兜底 / 越界告警 / Detection 模型。"""

from __future__ import annotations

import pytest

from qsmy_deepseek_locator.parsing import (
    Detection,
    LEGACY_SCALE_NOTICE,
    box_iou,
    check_coordinate_range,
    decode_json_points,
    normalize_to_unit,
    parse_detections,
    to_dict_items,
)


class TestDecode:
    def test_plain_array(self):
        assert decode_json_points('[{"bbox_2d": [0.1, 0.2, 0.3, 0.4], "label": "猫"}]') == [
            {"bbox_2d": [0.1, 0.2, 0.3, 0.4], "label": "猫"}
        ]

    def test_fenced_json(self):
        text = '好的，结果如下：\n```json\n[{"point_2d": [0.5, 0.5], "label": "点"}]\n```\n以上。'
        assert decode_json_points(text)[0]["label"] == "点"

    def test_prose_around(self):
        text = '我找到了：\n[{"bbox_2d": [0, 0, 1, 1], "label": "整图"}]\n希望有帮助。'
        assert decode_json_points(text)[0]["label"] == "整图"

    def test_brackets_inside_label(self):
        # label 里带方括号/引号时，正则扫描会跑偏；这里验证括号配对扫描是对的
        text = r'[{"bbox_2d": [0.1, 0.1, 0.2, 0.2], "label": "按钮[主] \"确定\""}]'
        assert decode_json_points(text)[0]["label"] == '按钮[主] "确定"'

    def test_single_quotes(self):
        assert decode_json_points("[{'point_2d': [0.3, 0.4], 'label': 'x'}]")[0]["label"] == "x"

    def test_garbage(self):
        assert decode_json_points("这张图里没有你要找的东西。") == []
        assert decode_json_points("") == []
        assert decode_json_points(None) == []

    def test_wrapped_dict(self):
        assert to_dict_items({"items": [{"bbox_2d": [0, 0, 1, 1]}]})[0]["bbox_2d"] == [0, 0, 1, 1]


class TestNormalize:
    def test_unit_scale_untouched(self):
        items = [{"bbox_2d": [0.1, 0.2, 0.3, 0.4], "label": "a"}]
        out, converted = normalize_to_unit(items)
        assert converted is False
        assert out[0]["bbox_2d"] == [0.1, 0.2, 0.3, 0.4]

    def test_legacy_scale_divided(self):
        items = [{"bbox_2d": [100, 200, 350.25, 400], "label": "a"}]
        out, converted = normalize_to_unit(items)
        assert converted is True
        assert out[0]["bbox_2d"] == [0.1, 0.2, 0.35025, 0.4]

    def test_pixel_values_left_alone(self):
        # 超过 1000 的值不换算：那更像像素坐标，自动除以 1000 反而会掩盖问题
        items = [{"bbox_2d": [1440, 900, 2000, 1500]}]
        out, converted = normalize_to_unit(items)
        assert converted is False
        assert out[0]["bbox_2d"] == [1440, 900, 2000, 1500]

    def test_mixed_batch_decided_as_whole(self):
        # 整批判定：只要有一个 > 1，全批一起换算，不会出现一半一半
        items = [{"bbox_2d": [0.5, 0.5, 0.6, 0.6]}, {"point_2d": [500, 500]}]
        out, converted = normalize_to_unit(items)
        assert converted is True
        assert out[0]["bbox_2d"] == [0.0005, 0.0005, 0.0006, 0.0006]
        assert out[1]["point_2d"] == [0.5, 0.5]


class TestRange:
    def test_in_range_is_quiet(self):
        assert check_coordinate_range([{"bbox_2d": [0.1, 0.2, 0.3, 0.4]}]) == []

    def test_pixel_coords_flagged(self):
        warnings = check_coordinate_range([{"bbox_2d": [100, 200, 300, 400], "label": "猫"}])
        assert len(warnings) == 1
        assert warnings[0]["label"] == "猫"
        assert warnings[0]["out_of_range"] == [100, 200, 300, 400]


class TestParseDetections:
    def test_normal(self):
        dets, warnings, raw = parse_detections(
            '[{"bbox_2d": [0.1, 0.2, 0.3, 0.4], "label": "猫"},'
            ' {"point_2d": [0.5, 0.6], "label": "眼睛"}]'
        )
        assert [d.kind for d in dets] == ["bbox", "point"]
        assert dets[0].label == "猫"
        assert warnings == []
        assert len(raw) == 2

    def test_legacy_scale_warns(self):
        dets, warnings, _ = parse_detections('[{"bbox_2d": [100, 200, 300, 400], "label": "猫"}]')
        assert dets[0].bbox == (0.1, 0.2, 0.3, 0.4)
        assert LEGACY_SCALE_NOTICE in warnings[0]

    def test_malformed_item_dropped_with_warning(self):
        dets, warnings, _ = parse_detections(
            '[{"bbox_2d": [0.1, 0.2, 0.3], "label": "三点框"}, {"point_2d": [0.5, 0.5], "label": "ok"}]'
        )
        assert len(dets) == 1
        assert any("不合法" in w for w in warnings)

    def test_empty_text(self):
        dets, warnings, raw = parse_detections("")
        assert dets == [] and warnings == [] and raw == []


class TestDetection:
    def test_kind_and_center(self):
        box = Detection(label="框", bbox=(0.2, 0.4, 0.6, 0.8))
        assert box.kind == "bbox"
        assert box.center == pytest.approx((0.4, 0.6))
        assert box.area == pytest.approx(0.16)

        dot = Detection(label="点", point=(0.25, 0.75))
        assert dot.kind == "point"
        assert dot.center == (0.25, 0.75)
        assert dot.area == 0.0

    def test_roundtrip(self):
        original = {"bbox_2d": [0.1, 0.2, 0.3, 0.4], "label": "猫"}
        det = Detection.from_dict(original)
        assert det is not None
        assert det.to_dict() == original

    def test_from_dict_rejects_bad(self):
        assert Detection.from_dict({"label": "没有坐标"}) is None
        assert Detection.from_dict({"bbox_2d": [0.1, 0.2, "x", 0.4]}) is None
        assert Detection.from_dict("不是字典") is None

    def test_normalize_clamps(self):
        det = Detection.from_dict({"bbox_2d": [-0.2, 0.5, 1.5, 0.9]}, normalize=True)
        assert det.bbox == (0.0, 0.5, 1.0, 0.9)

    def test_to_pixels(self):
        det = Detection(label="框", bbox=(0.25, 0.25, 0.75, 0.75))
        px = det.to_pixels(400, 300)
        assert px["bbox_px"] == (100, 75, 300, 225)
        assert px["center_px"] == (200, 150)
        assert Detection(point=(0.5, 0.5)).to_pixels(400, 300)["point_px"] == (200, 150)

    def test_iou_and_contains(self):
        a = Detection(bbox=(0.0, 0.0, 1.0, 1.0))
        b = Detection(bbox=(0.0, 0.0, 0.5, 0.5))
        assert a.iou(b) == pytest.approx(0.25)
        assert a.iou(Detection(point=(0.5, 0.5))) == 0.0
        assert a.contains_point(Detection(point=(0.5, 0.5)))
        assert not a.contains_point(Detection(point=(1.5, 0.5)))

    def test_box_iou_reversed_edges(self):
        assert box_iou([0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]) == pytest.approx(1.0)
