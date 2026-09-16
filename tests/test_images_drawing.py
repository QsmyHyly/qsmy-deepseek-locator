"""图像加载 / 编码 / 打标自测。"""

from __future__ import annotations

import base64
import functools
import http.server
import io
import socketserver
import threading

import pytest
from PIL import Image

from qsmy_deepseek_locator.drawing import coerce_detections, draw, save_annotated
from qsmy_deepseek_locator.errors import ImageLoadError
from qsmy_deepseek_locator.images import (
    describe_source,
    encode_data_url,
    load_image,
    source_size,
    to_data_url,
)
from qsmy_deepseek_locator.parsing import Detection


class TestLoadImage:
    def test_from_path(self, sample_png):
        img = load_image(sample_png)
        assert img.size == (400, 300)

    def test_from_bytes(self, sample_png):
        assert load_image(sample_png.read_bytes()).size == (400, 300)

    def test_from_pil_copy(self, sample_png):
        original = Image.open(sample_png)
        loaded = load_image(original)
        assert loaded.size == original.size
        assert loaded is not original  # 必须返回副本，不能把调用方的图改了

    def test_from_data_url(self, sample_png):
        url = to_data_url(sample_png)
        assert load_image(url).size == (400, 300)

    def test_missing_file(self, tmp_path):
        with pytest.raises(ImageLoadError, match="不存在"):
            load_image(tmp_path / "nope.png")

    def test_bad_bytes(self):
        with pytest.raises(ImageLoadError, match="无法解码"):
            load_image(b"this is not an image")


@pytest.fixture
def image_server(tmp_path):
    """在 127.0.0.1 上起一个只读小服务器，用来离线验证「按 URL 读图」。

    为什么要自建：URL 读图是 `requests` 那条代码路径，不测就等于没验。
    打外部网站会引入网络波动，本地起一个端口为 0（系统分配）的服务器最稳。
    """
    payload = io.BytesIO()
    Image.new("RGB", (120, 90), (12, 34, 56)).save(payload, format="PNG")
    (tmp_path / "remote.png").write_bytes(payload.getvalue())

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as server:
        server.RequestHandlerClass.log_message = lambda *a, **k: None  # 别把访问日志吐进测试输出
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}/remote.png"
        finally:
            server.shutdown()
            thread.join(timeout=5)


class TestRemoteUrl:
    def test_load_from_url(self, image_server):
        assert load_image(image_server).size == (120, 90)

    def test_to_data_url_from_url(self, image_server):
        url = to_data_url(image_server)
        assert url.startswith("data:image/png;base64,")
        with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as img:
            assert img.size == (120, 90)

    def test_source_size_from_url(self, image_server):
        assert source_size(image_server) == (120, 90)

    def test_404_raises_image_load_error(self, image_server):
        with pytest.raises(ImageLoadError):
            load_image(image_server.replace("remote.png", "missing.png"))

    def test_describe_url_is_short(self, image_server):
        text = describe_source(image_server)
        assert len(text) < 60 and "127.0.0.1" in text


class TestDataUrl:
    def test_png_bytes_passthrough(self, sample_png):
        """不需要缩放时应当**零重编码**：原始字节原样 base64 上去。"""
        raw = sample_png.read_bytes()
        url = to_data_url(sample_png)
        assert url.startswith("data:image/png;base64,")
        assert base64.b64decode(url.split(",", 1)[1]) == raw

    def test_max_side_shrinks(self, sample_png):
        url = to_data_url(sample_png, max_side=100)
        payload = base64.b64decode(url.split(",", 1)[1])
        with Image.open(io.BytesIO(payload)) as img:
            assert max(img.size) == 100

    def test_shrink_keeps_aspect(self):
        img = Image.new("RGB", (800, 400), (10, 20, 30))
        url = to_data_url(img, max_side=200)
        payload = base64.b64decode(url.split(",", 1)[1])
        with Image.open(io.BytesIO(payload)) as out:
            assert out.size == (200, 100)

    def test_no_upscale(self):
        img = Image.new("RGB", (100, 50), (0, 0, 0))
        payload = base64.b64decode(to_data_url(img, max_side=400).split(",", 1)[1])
        with Image.open(io.BytesIO(payload)) as out:
            assert out.size == (100, 50)

    def test_alpha_keeps_png(self):
        img = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
        assert to_data_url(img, max_side=5).startswith("data:image/png;base64,")

    def test_plain_image_becomes_jpeg_when_reencoding(self):
        assert encode_data_url(Image.new("RGB", (8, 8))).startswith("data:image/jpeg;base64,")

    def test_bad_source_raises(self):
        with pytest.raises(ImageLoadError):
            to_data_url("不存在的路径.png", max_side=100)


class TestHelpers:
    def test_source_size(self, sample_png):
        assert source_size(sample_png) == (400, 300)
        assert source_size(b"garbage") is None

    def test_describe_hides_base64(self, sample_png):
        text = describe_source(to_data_url(sample_png))
        assert text.startswith("<data URL") and len(text) < 40
        assert describe_source(b"12345") == "<bytes 5B>"


class TestCoerce:
    def test_accepts_json_text(self):
        dets = coerce_detections('[{"bbox_2d": [100, 200, 300, 400], "label": "猫"}]')
        assert dets[0].bbox == (0.1, 0.2, 0.3, 0.4)  # 旧刻度兜底同样在这里生效

    def test_accepts_mixed(self):
        items = [Detection(bbox=(0.1, 0.1, 0.2, 0.2)), {"point_2d": [0.5, 0.5], "label": "点"}, "[]"]
        assert len(coerce_detections(items)) == 2

    def test_none(self):
        assert coerce_detections(None) == []


class TestDraw:
    def test_bbox_hits_expected_pixels(self):
        # 400x300 白底，画 (0.25,0.25)-(0.75,0.75) 的红框：上边中点应当变红
        img = Image.new("RGB", (400, 300), (255, 255, 255))
        out = draw(img, [Detection(label="框", bbox=(0.25, 0.25, 0.75, 0.75))])
        assert out.getpixel((200, 75)) == (255, 0, 0)
        assert out.getpixel((200, 150)) == (255, 255, 255)  # 框内不填充

    def test_point_drawn(self):
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        out = draw(img, [Detection(label="点", point=(0.5, 0.5))])
        assert out.getpixel((100, 100)) == (255, 0, 0)

    def test_out_of_range_clamped_not_crash(self):
        img = Image.new("RGB", (100, 100), (255, 255, 255))
        out = draw(img, [Detection(label="越界", bbox=(-1.0, -1.0, 3.0, 3.0))])
        assert out.size == (100, 100)

    def test_accepts_model_text(self):
        img = Image.new("RGB", (100, 100), (255, 255, 255))
        out = draw(img, '[{"point_2d": [500, 500], "label": "中点"}]')
        assert out.getpixel((50, 50)) == (255, 0, 0)

    def test_chinese_label_does_not_crash(self):
        img = Image.new("RGB", (300, 200), (255, 255, 255))
        out = draw(img, [Detection(label="红色圆形", bbox=(0.2, 0.2, 0.6, 0.6))])
        assert out.size == (300, 200)

    def test_save_annotated(self, sample_png, tmp_path):
        out = save_annotated(sample_png, [Detection(label="猫", bbox=(0.1, 0.1, 0.5, 0.5))],
                             path=tmp_path / "sub" / "out.png")
        assert out.exists()
        with Image.open(out) as img:
            assert img.size == (400, 300)

    def test_save_annotated_auto_name(self, sample_png, tmp_path):
        out = save_annotated(sample_png, [], output_dir=tmp_path, stem="固定名")
        assert out.name == "固定名.png"

    def test_save_annotated_follows_suffix(self, sample_png, tmp_path):
        """扩展名决定格式：写 .jpg 就得是 JPEG 字节，不能挂羊头卖狗肉。

        这条曾经是坏的：save_annotated 写死 format="PNG"，于是 -o out.jpg
        会产出「文件名 .jpg、内容却是 PNG」的图，下游按后缀读图直接报错。
        """
        out = save_annotated(sample_png, [Detection(label="猫", bbox=(0.1, 0.1, 0.5, 0.5))],
                             path=tmp_path / "out.jpg")
        assert out.read_bytes()[:2] == b"\xff\xd8"        # JPEG 魔数
        with Image.open(out) as img:
            assert img.format == "JPEG"

    def test_save_annotated_rejects_unknown_suffix(self, sample_png, tmp_path):
        """认不出的后缀要报错，不能偷偷存成 PNG。"""
        with pytest.raises(ValueError):
            save_annotated(sample_png, [], path=tmp_path / "out.txt")
