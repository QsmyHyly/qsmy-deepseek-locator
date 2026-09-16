"""qsmy-deepseek-locator：用 DeepSeek 视觉模型做物体定位。

一句话：给它一张图和一句「找什么」，拿回 0.0~1.0 的归一化坐标（框 / 点）与中文名称，
需要的话再把框和标签画回图上。

最常用的几个名字：

    from qsmy_deepseek_locator import locate, locate_to_file, draw, Locator, Detection

    result = locate("photo.png", "红色圆形")     # 一次调用，拿坐标
    draw("photo.png", result).save("out.png")    # 只想自己掌控绘制时

    # 一行出图：图片 + 「找什么」+ 输出路径，回来时文件已经在了
    result = locate_to_file("photo.png", "红色圆形", "runs/photo_annotated.png")
    print(result.annotated_path)

设计上刻意与「旧演示项目」不同的两点（对外承诺，别在后来的改动里悄悄破坏）：

    1. **没有 Key 就报错，不静默给假数据**。演示项目无 Key 时进 Mock 模式对页面友好，
       但对库是危险的：用户会拿着假坐标当真结果。
    2. **坐标只有一种口径：0.0~1.0 相对比例，小数位不设上限**。0~1000 旧刻度会被自动
       除以 1000 换算并留下告警，像素坐标则一律告警而不猜测 —— 详见 parsing.py 的模块说明。
"""

from importlib.metadata import PackageNotFoundError, version as _version

from .client import (
    ChatReply,
    DeepSeekVisionClient,
    VisionClient,
    build_messages,
    build_request,
)
from .config import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    IMAGE_DETAILS,
    REASONING_EFFORTS,
    Settings,
)
from .drawing import (
    COLORS,
    coerce_detections,
    draw,
    resolve_font,
    resolve_output_path,
    save_annotated,
)
from .errors import (
    APIError,
    EmptyResponseError,
    ImageLoadError,
    LocatorError,
    MissingAPIKeyError,
)
from .images import encode_data_url, load_image, source_size, to_data_url
from .locate import LocateResult, Locator, locate, locate_to_file
from .parsing import (
    BBOX_FIELD,
    POINT_FIELD,
    Detection,
    box_iou,
    check_coordinate_range,
    decode_json_points,
    normalize_to_unit,
    parse_detections,
)
from .prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT, build_user_prompt

try:  # 版本号只有一个来源：pyproject.toml
    __version__ = _version("qsmy-deepseek-locator")
except PackageNotFoundError:  # 未安装（直接从源码 import）时的兜底
    __version__ = "0.1.0"

__all__ = [
    # 核心
    "locate",
    "locate_to_file",
    "Locator",
    "LocateResult",
    "Detection",
    # 打标
    "draw",
    "save_annotated",
    "coerce_detections",
    "resolve_font",
    "resolve_output_path",
    "COLORS",
    # 配置与客户端
    "Settings",
    "DeepSeekVisionClient",
    "VisionClient",
    "ChatReply",
    "build_messages",
    "build_request",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_TIMEOUT",
    "REASONING_EFFORTS",
    "IMAGE_DETAILS",
    # 图片
    "load_image",
    "to_data_url",
    "encode_data_url",
    "source_size",
    # 解析
    "parse_detections",
    "decode_json_points",
    "normalize_to_unit",
    "check_coordinate_range",
    "box_iou",
    "BBOX_FIELD",
    "POINT_FIELD",
    # 提示词
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_USER_PROMPT",
    "build_user_prompt",
    # 异常
    "LocatorError",
    "MissingAPIKeyError",
    "ImageLoadError",
    "APIError",
    "EmptyResponseError",
    # 元信息
    "__version__",
]
