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

依赖只有三个：Pillow 与 requests 是必装，**openai 现在是可选依赖**
（pip install qsmy-deepseek-locator[openai]）。不装 openai 也能完整使用本库 ——
换成自带的自备客户端即可，它只用 requests：

    from qsmy_deepseek_locator import Locator, RequestsVisionClient
    locator = Locator(client=RequestsVisionClient(api_key="sk-xxx"))
    result = locator.locate("photo.png", "红色圆形")

openai 之所以可选，是因为它依赖的 jiter / pydantic-core 都是 Rust 扩展，
安卓（aarch64）上没有 wheel，装了也白装 —— 而「在一个装不上 SDK 的环境里用这个库」
是完全正当的用法。

⚠️ **所有入口都是同步阻塞调用**，一次 locate 最坏可能等 900s（300s 超时 × 3 次尝试）。
别在主线程 / UI 线程里调；要能中途喊停就传 cancel_event=threading.Event，
它在下一个流式事件到达时抛 CancelledError。

想看「到底发出去什么、模型回了什么」，给任何入口传 log_file（**默认不开**）：

    locate("photo.png", "红色圆形", log_file="runs/logs/run.jsonl")

请求体 / 事件流 / 响应体 / 解析结果 / 异常会写成一行一个 JSON 的 JSONL 文件，
读法与取舍见 debuglog.py 的模块说明。
"""

from importlib.metadata import PackageNotFoundError, version as _version

from .client import (
    ChatReply,
    DeepSeekVisionClient,
    VisionClient,
    build_messages,
    build_request,
)
# 裸 HTTP 客户端：不依赖 openai，只依赖 requests。给装不上 openai 的环境用
# （安卓 / aarch64），用法与 DeepSeekVisionClient 完全一样，见 http_client.py 的对照表。
from .http_client import RequestsVisionClient
from .config import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    IMAGE_DETAILS,
    REASONING_EFFORTS,
    Settings,
)
from .debuglog import DebugLog, coerce_log, default_log_path
from .drawing import (
    COLORS,
    coerce_detections,
    draw,
    find_cjk_font,
    resolve_font,
    resolve_output_path,
    save_annotated,
)
from .errors import (
    APIError,
    CancelledError,
    EmptyResponseError,
    ImageLoadError,
    LocatorError,
    LogFileTypeError,
    MissingAPIKeyError,
    OutputPathError,
    UnsupportedFeatureError,
    WriteError,
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
# 报文拼装的三个纯函数（0.1.3 上游化）：下游若有自己的配置对象，走 merge_thinking 这条
# **不依赖 Settings** 的路径；image_part / thinking_payload 是「拼一次请求」的两块积木。
# 它们本来就是公开行为，只是此前没进 __all__ —— 结果三个项目各抄了一份，见 request_build.py。
from .request_build import image_part, merge_thinking, thinking_payload

# 版本号有两个来源，必须一起改：
#   装了包 -> importlib.metadata 从 dist-info 读（真来源是 pyproject.toml 的 [project] version）；
#   没装包 -> 下面这个兜底字面量。**从源码 import 的情形比想象中常见**：直接跑仓库里的脚本、
#   或者像 App 项目那样把源码 vendor 进去，二者都拿不到 metadata，只能走兜底。
#   顺带一提：src/ 下若留着上次构建的 egg-info（.gitignore 里那类残留），metadata 会先读到它，
#   于是 __version__ 报到上一版去 —— 那种情况请删掉/重建 egg-info，不是这里的兜底出了问题。
# 0.1.2 发版时就漏改了兜底值，于是那些人看到的 __version__ 是谎报的 0.1.1 ——
# 而这恰恰是最需要版本号准确的时候（排查现场第一句话就是「你用的哪个版本」）。
# 「发版时记得改」已经被证明靠不住了，所以 tests/test_version.py 会把兜底值与
# pyproject.toml 的 version 钉在一起：只改一处，测试就红。
try:  # 版本号只有一个来源：pyproject.toml
    __version__ = _version("qsmy-deepseek-locator")
except PackageNotFoundError:  # 未安装（直接从源码 import）时的兜底
    __version__ = "0.2.0"

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
    "find_cjk_font",
    "resolve_font",
    "resolve_output_path",
    "COLORS",
    # 配置与客户端
    "Settings",
    "DeepSeekVisionClient",
    "RequestsVisionClient",   # 纯 requests，不需要 openai
    "VisionClient",
    "ChatReply",
    "build_messages",
    "build_request",
    "image_part",
    "thinking_payload",
    "merge_thinking",     # 不依赖 Settings 的 thinking 规则本体，给下游复用
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_TIMEOUT",
    "REASONING_EFFORTS",
    "IMAGE_DETAILS",
    # 调试日志（默认关闭）
    "DebugLog",
    "coerce_log",
    "default_log_path",
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
    # 异常（**本库抛出的东西总是 LocatorError**，本次改动后闭合）
    "LocatorError",
    "MissingAPIKeyError",
    "ImageLoadError",
    "APIError",
    "EmptyResponseError",
    "UnsupportedFeatureError",
    "OutputPathError",
    "WriteError",
    "LogFileTypeError",
    "CancelledError",
    # 元信息
    "__version__",
]
