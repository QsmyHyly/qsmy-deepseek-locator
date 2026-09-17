"""评测模块自测：造图 / 判分 / 汇总 / 端到端（离线，用假客户端）。"""

from __future__ import annotations

import json

from PIL import Image

from qsmy_deepseek_locator.benchmark import (
    PALETTE,
    aggregate,
    color_ok,
    evaluate_sample,
    make_sample,
    make_samples,
    match_predictions,
    run_benchmark,
    shape_ok,
)
from qsmy_deepseek_locator.locate import Locator
from qsmy_deepseek_locator.parsing import Detection


def _gt(bbox, label, kind, color_name):
    return {"bbox_2d": list(bbox), "label": label, "kind": kind, "color_name": color_name}


class TestLabelRules:
    def test_color_synonyms(self):
        assert color_ok("红色", "红色的圆形")
        assert color_ok("红色", "红圆")
        assert not color_ok("红色", "蓝色矩形")

    def test_shape_rules(self):
        assert shape_ok("rect", "长方形")
        assert shape_ok("circle", "圆形")
        assert not shape_ok("circle", "椭圆形")  # 椭圆不能算圆
        assert shape_ok("ellipse", "椭圆")
        assert shape_ok("triangle", "三角形")


class TestMakeSamples:
    def test_shapes_inside_canvas(self, tmp_path):
        sample, img = make_sample(0, seed=1, width=600, height=400, n_shapes=3)
        assert img.size == (600, 400)
        assert len(sample.shapes) == 3
        for shape in sample.shapes:
            x1, y1, x2, y2 = shape.bbox_px
            assert 0 <= x1 < x2 <= 600
            assert 0 <= y1 < y2 <= 400

    def test_gt_normalized_and_matches_pixels(self, tmp_path):
        """真值必须与画面像素一致：按真值框取中心像素，颜色应当就是真值说的那个颜色。

        这条是防「真值写错却看不出来」的：真值错了，评测分数就全是假的。
        """
        sample, img = make_sample(1, seed=5, width=800, height=600, n_shapes=4)
        for shape, gt in zip(sample.shapes, sample.ground_truth()):
            x1, y1, x2, y2 = gt["bbox_2d"]
            assert 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0
            cx = int((x1 + x2) / 2 * 800)
            cy = int((y1 + y2) / 2 * 600)
            assert img.getpixel((cx, cy)) == shape.rgb

    def test_colors_unique_within_image(self):
        sample, _ = make_sample(2, seed=9, n_shapes=3)
        names = [s.color_name for s in sample.shapes]
        assert len(set(names)) == len(names)

    def test_make_samples_writes_files(self, tmp_path):
        samples = make_samples(2, seed=3, out_dir=tmp_path, width=300, height=200, n_shapes=2)
        assert len(samples) == 2
        assert (tmp_path / "sample_01.png").exists()
        payload = json.loads((tmp_path / "ground_truth.json").read_text(encoding="utf-8"))
        assert len(payload) == 2
        assert payload[0]["ground_truth"][0]["color_name"] in PALETTE

    def test_deterministic_for_same_seed(self):
        a, img_a = make_sample(0, seed=11, n_shapes=2)
        b, img_b = make_sample(0, seed=11, n_shapes=2)
        assert [s.bbox_px for s in a.shapes] == [s.bbox_px for s in b.shapes]
        assert img_a.tobytes() == img_b.tobytes()


class TestScoring:
    def test_perfect_prediction(self):
        gts = [_gt((0.1, 0.1, 0.3, 0.3), "红色矩形", "rect", "红色")]
        preds = [Detection(label="红色矩形", bbox=(0.1, 0.1, 0.3, 0.3))]
        report = evaluate_sample(preds, gts)
        assert report["detection_rate"] == 1.0
        assert report["mean_iou"] == 1.0
        assert report["label_accuracy"] == 1.0
        assert report["precision"] == 1.0

    def test_half_overlap_misses_iou_threshold(self):
        gts = [_gt((0.0, 0.0, 0.4, 0.4), "红色矩形", "rect", "红色")]
        preds = [Detection(label="红色矩形", bbox=(0.2, 0.0, 0.6, 0.4))]
        report = evaluate_sample(preds, gts)
        assert report["hits"] == 0          # IoU 只有 1/3，低于 0.5
        assert 0.3 < report["mean_iou"] < 0.34

    def test_wrong_label_still_counts_as_hit(self):
        # 检出与读标签是两件事：框对但颜色认错，检出率照样 100%，标签准确率 0
        gts = [_gt((0.1, 0.1, 0.3, 0.3), "红色矩形", "rect", "红色")]
        preds = [Detection(label="蓝色矩形", bbox=(0.1, 0.1, 0.3, 0.3))]
        report = evaluate_sample(preds, gts)
        assert report["detection_rate"] == 1.0
        assert report["label_accuracy"] == 0.0
        assert report["color_accuracy"] == 0.0
        assert report["shape_accuracy"] == 1.0

    def test_extra_predictions_lower_precision(self):
        gts = [_gt((0.1, 0.1, 0.3, 0.3), "红色矩形", "rect", "红色")]
        preds = [
            Detection(label="红色矩形", bbox=(0.1, 0.1, 0.3, 0.3)),
            Detection(label="幻觉目标", bbox=(0.6, 0.6, 0.9, 0.9)),
        ]
        report = evaluate_sample(preds, gts)
        assert report["detection_rate"] == 1.0
        assert report["precision"] == 0.5

    def test_point_predictions_not_matched(self):
        # 评测按框算 IoU：只给点的预测无法参与匹配（本库的 bench 就是测框的）
        gts = [_gt((0.1, 0.1, 0.3, 0.3), "红色矩形", "rect", "红色")]
        report = evaluate_sample([Detection(label="红色矩形", point=(0.2, 0.2))], gts)
        assert report["hits"] == 0
        assert report["pred_point_count"] == 1

    def test_greedy_matching_is_one_to_one(self):
        preds = [
            Detection(label="a", bbox=(0.0, 0.0, 0.5, 0.5)),
            Detection(label="b", bbox=(0.01, 0.01, 0.5, 0.5)),
        ]
        gts = [_gt((0.0, 0.0, 0.5, 0.5), "a", "rect", "红色")]
        matches = match_predictions(preds, gts)
        assert len(matches) == 1  # 一个真值只能被一个预测吃掉

    def test_aggregate(self):
        gts = [_gt((0.1, 0.1, 0.3, 0.3), "红色矩形", "rect", "红色")]
        perfect = evaluate_sample([Detection(label="红色矩形", bbox=(0.1, 0.1, 0.3, 0.3))], gts)
        empty = evaluate_sample([], gts)
        summary = aggregate([perfect, empty])
        assert summary["total_gt"] == 2
        assert summary["total_hits"] == 1
        assert summary["detection_rate"] == 0.5
        assert summary["label_accuracy"] == 1.0


class TestRunBenchmark:
    def test_images_only(self, tmp_path):
        report = run_benchmark(count=2, n_shapes=2, seed=4, out_dir=tmp_path, images_only=True)
        assert report["report"] is None
        assert (tmp_path / "report.json").exists()
        assert (tmp_path / "images" / "sample_01.png").exists()

    def test_end_to_end_with_fake_client(self, tmp_path):
        """假客户端照抄真值当答案 —— 这是评测链路本身的「满分可达性」验收。"""
        first = run_benchmark(count=1, n_shapes=2, seed=7, out_dir=tmp_path / "gen", images_only=True)
        gt = first["samples"][0]["ground_truth"]
        text = json.dumps(
            [{"bbox_2d": g["bbox_2d"], "label": g["label"]} for g in gt], ensure_ascii=False
        )

        class OneShot:
            def complete(self, messages, *, settings=None, on_event=None, timeout=None):
                from qsmy_deepseek_locator.client import ChatReply

                return ChatReply(text=text, model="fake", finish_reason="stop")

        report = run_benchmark(
            count=1, n_shapes=2, seed=7, out_dir=tmp_path / "run",
            locator=Locator(client=OneShot()), annotate=True,
        )
        summary = report["report"]
        assert summary["total_gt"] == 2
        assert summary["detection_rate"] == 1.0
        assert summary["label_accuracy"] == 1.0
        assert summary["mean_iou"] >= 0.99     # 真值坐标落盘时保留了 4 位小数
        assert (tmp_path / "run" / "annotated" / "sample_01_pred.png").exists()
        assert (tmp_path / "run" / "report.json").exists()

    def test_missing_target_prompt_feature(self, tmp_path):
        # 只造图时不该碰配置（不带 Key 也必须能跑）
        report = run_benchmark(count=1, n_shapes=1, out_dir=tmp_path, images_only=True)
        assert report["config"]["target"]
