"""图像加载与编码：把「任意图片源」统一成 PIL 图或 data URL。

支持四类输入（下游 load_image / to_data_url 都认同一套）：

    - 本地路径（str 或 pathlib.Path）
    - http(s) URL
    - bytes（原始图片字节）
    - PIL.Image.Image
    - data URL（data:image/png;base64,...）

编码策略（to_data_url）—— 刻意分两条路，别小看这个取舍：

    ① 不需要缩放、源本身就是图片字节（路径 / URL / bytes / data URL）
       -> **直接把原始字节 base64 上去**，一个像素都不重新编码。
          位置精度虽然靠的是相对比例、与压缩无关，但重编码本身是白花的 CPU 和画质，
          而且会让「本库发的报文」和「已验证过的那条链路」产生差异，不利于排查。
    ② 需要重编码（传进来的是 PIL 图，或显式给了 max_side 要缩图）
       -> 有透明度用 PNG，否则用 JPEG(quality 默认 90)。

max_side 是给大图准备的省流量开关（例：手机拍的 4000x3000 照片转 base64 有十几 MB）。
坐标是相对比例，缩放**不影响定位结果**，所以这个开关是安全的。
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from .errors import ImageLoadError

# 后缀 -> mime。只用于「把已有字节标个类型」，认不出就退回 PNG。
_SUFFIX_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}

# 字节魔数 -> mime（用于 bytes 输入，不猜后缀）
_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)


def sniff_mime(data: bytes) -> str:
    """按魔数判断图片类型，认不出返回 image/png（后续解码失败会如实报错）。"""
    for magic, mime in _MAGIC:
        if data.startswith(magic):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _text_source(source: Any) -> str:
    return str(source)


def load_bytes(source: Any, *, timeout: float = 60.0) -> tuple[bytes, str]:
    """把任意「非 PIL」源读成 (原始字节, mime)。不重编码、不解码。"""
    if isinstance(source, Path):
        source = str(source)

    if isinstance(source, bytes):
        return source, sniff_mime(source)

    if isinstance(source, bytearray):
        data = bytes(source)
        return data, sniff_mime(data)

    if not isinstance(source, str):
        raise ImageLoadError(f"不认识的图片源类型：{type(source).__name__}")

    text = source.strip()
    if text.startswith("data:"):
        header, _, b64 = text.partition(",")
        mime = header[5:].split(";")[0] or "image/png"
        try:
            return base64.b64decode(b64), mime
        except Exception as exc:  # noqa: BLE001
            raise ImageLoadError(f"data URL 解码失败：{exc}") from exc

    if text.startswith(("http://", "https://")):
        try:
            resp = requests.get(text, timeout=timeout)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise ImageLoadError(f"下载图片失败：{text}（{exc}）") from exc
        return resp.content, resp.headers.get("Content-Type", "").split(";")[0] or sniff_mime(resp.content)

    path = Path(text)
    if not path.exists():
        raise ImageLoadError(f"图片文件不存在：{text}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ImageLoadError(f"读取图片失败：{text}（{exc}）") from exc
    return data, _SUFFIX_MIME.get(path.suffix.lower(), sniff_mime(data))


def load_image(source: Any, *, timeout: float = 60.0) -> Image.Image:
    """加载成 PIL 图（**原样保留**源图的模式：RGB / RGBA / L / P 都可能）。

    为什么这里不做 convert("RGB")：**本函数的职责只是"把源读成 PIL 图"，不是"准备一张能画的图"**，
    用哪张图做什么是调用方的事。透明度该不该丢、什么时候丢，取决于下游要干什么 ——
    打标那条路确实要 RGB（drawing.draw 第一行就 convert("RGB")，因为 ImageDraw 的
    抗锯齿与色彩混合在带 alpha 的图上是另一套行为），但"下载/读取"这一步就把信息扔掉，
    会让「读出来交给别的库处理」的调用方拿不到原本的通道。

    ⚠️ 旧注释曾写「保留 RGBA 是因为转 RGB 会糊成黑底」，那句话与实现不符 ——
    drawing.draw 的第一行就是 .convert("RGB")，糊不糊根本轮不到这里管；真正的理由如上。
    """
    if isinstance(source, Image.Image):
        return source.copy()

    data, _mime = load_bytes(source, timeout=timeout)
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # noqa: BLE001
        raise ImageLoadError(f"图片内容无法解码（不是有效图片，或格式不受支持）：{exc}") from exc
    return img


def source_size(source: Any, *, timeout: float = 60.0) -> tuple[int, int] | None:
    """尽量便宜地拿到图片宽高；拿不到返回 None。

    本地路径走 PIL 的懒加载头解析，不会解码整张图；URL 则必须下载完整字节（没法只读头）。
    """
    if isinstance(source, Image.Image):
        return source.size
    try:
        data, _mime = load_bytes(source, timeout=timeout)
        with Image.open(io.BytesIO(data)) as img:
            return img.size
    except Exception:  # noqa: BLE001 - 宽高只是附带信息，拿不到不影响主流程
        return None


def describe_source(source: Any) -> str:
    """给日志/结果里用的简短描述，避免把整段 base64 打进日志。"""
    if isinstance(source, Image.Image):
        return f"<PIL.Image {source.width}x{source.height} {source.mode}>"
    if isinstance(source, (bytes, bytearray)):
        return f"<bytes {len(source)}B>"
    text = _text_source(source)
    if text.startswith("data:"):
        return f"<data URL {len(text)}B>"
    return text if len(text) <= 200 else text[:200] + "..."


def _shrink(img: Image.Image, max_side: int | None) -> Image.Image:
    """等比缩小到最长边不超过 max_side（只缩不放）。"""
    if not max_side or max_side <= 0:
        return img
    width, height = img.size
    longest = max(width, height)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    # 向内取整，保证结果不超过 max_side
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return img.resize(size, Image.LANCZOS)


def encode_data_url(
    img: Image.Image, *, jpeg_quality: int = 90, fmt: str | None = None
) -> str:
    """把 PIL 图编码成 data URL（有 alpha 用 PNG，否则 JPEG，也可用 fmt 强制）。"""
    buffer = io.BytesIO()
    chosen = (fmt or "").upper()
    if not chosen:
        chosen = "PNG" if img.mode in ("RGBA", "LA", "P") else "JPEG"
    if chosen == "JPEG":
        rgb = img.convert("RGB")
        rgb.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        mime = "image/jpeg"
    else:
        img.save(buffer, format="PNG")
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def to_data_url(
    source: Any,
    *,
    max_side: int | None = None,
    jpeg_quality: int = 90,
    timeout: float = 60.0,
) -> str:
    """把任意图片源转成可直接放进 image_url 的 data URL。

    Args:
        source: 路径 / URL / bytes / data URL / PIL.Image。
        max_side: 最长边上限；None = 不缩放。缩放不影响归一化坐标的精度，只影响上传体积。
        jpeg_quality: 重编码成 JPEG 时的质量（仅在需要重编码时才有意义）。

    为什么默认不缩放：官方每张图最多只算 384 token，大图在服务端照样会被缩到约 800x800，
    所以缩放在这里省的是**上行流量**而不是 token。默认保持原图，行为可预期；
    真嫌慢/嫌大就传 max_side=1600 这类值。
    """
    if isinstance(source, Image.Image):
        return encode_data_url(_shrink(source, max_side), jpeg_quality=jpeg_quality)

    raw, mime = load_bytes(source, timeout=timeout)
    if max_side is None:
        # 零重编码路径：原始字节直接上行
        return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"

    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.load()
            return encode_data_url(_shrink(img, max_side), jpeg_quality=jpeg_quality)
    except Exception as exc:  # noqa: BLE001
        raise ImageLoadError(f"图片内容无法解码：{exc}") from exc


__all__ = [
    "load_image",
    "load_bytes",
    "to_data_url",
    "encode_data_url",
    "source_size",
    "describe_source",
    "sniff_mime",
]
