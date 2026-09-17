"""配置：环境变量 + 内置默认，全部收在 Settings 一个不可变快照里。

设计取舍（和旧演示项目不同的地方）：

- **没有全局单例、没有可热改的配置文件**。旧项目有 config.local.json（网页上点过的开关），
  那是给「服务端常驻进程 + 浏览器界面」用的；本库是被人 import 的，模块级单例会让
  「改了环境变量没生效」「两个调用方互相污染」这类问题极难排查。
  所以 Settings 是 frozen dataclass，由 from_env() 现取，按次覆盖走 Locator(...) / locate(...)。
- **环境变量只做默认值**，显式传参永远优先。优先级：函数参数 > Locator 构造参数 > 环境变量 > 内置默认。
- 变量名沿用 DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL（与旧项目和官方 SDK 生态一致，
  用户不必再学一套），本库自己的开关统一加 QSML_ 前缀。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from typing import Any, Mapping

from .prompts import DEFAULT_SYSTEM_PROMPT

# 默认接口地址与模型。deepseek-flash 就是 DeepSeek-V4.1-Flash；
# 官方另一档 deepseek-v4-pro **不支持图像理解**，换成它整条识别链路直接失效（图片会被静默丢弃，
# HTTP 仍返回 200，只是 prompt_tokens 不涨）。官方来源：
# https://api-docs.deepseek.com/zh-cn/quick_start/pricing
# @doc docs/API-NOTES.md#7-非视觉模型不会报错只会静默丢图
#（该文档解决"怎么判断图片到底有没有被模型接收"的问题。）
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"

# 单次请求超时（秒）。**故意给得宽松**，因为：
#   1. 本库恒走流式，超时口径是「两次数据之间的静默」而不是整轮总时长 ——
#      思考模式下首字节可能好几秒才来，生成慢是常态（实测 6 目标的图思考开时花了 7.4s）；
#   2. 这里同时也是连接/写入超时，网络差的时候 120s 会误杀本来能成的请求。
# 代价要说清：服务端真挂住时，一次调用最多等 timeout × (max_retries + 1) —— 300s × 3 是上限。
# 想更激进/更保守都行：Settings(timeout=...)、Locator(timeout=...)、或环境变量 QSML_TIMEOUT。
DEFAULT_TIMEOUT = 300.0

# reasoning_effort 的合法取值。不在此集合内一律不传，避免服务端 400；
# 空值表示「不传该参数」，由服务端按自己的默认档处理（当前是 high）。
REASONING_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})

# image_url 内容块可选的 detail 字段：
#   low = 推理前缩到 512x512（更快更省 token）；high / original = 保留原图；auto = 服务端决定。
# ⚠️ 它不是「提高定位精度」的开关 —— 官方每张图最多只算 384 token，大图无论如何都会被缩到约 800x800。
#    默认 None = 根本不发送这个字段，报文与已验证过的旧链路逐字节一致。
# @doc docs/API-NOTES.md#3-图片-token-与尺寸
#（该文档解决"detail 到底改变了什么、为什么堆分辨率没用"的问题。）
IMAGE_DETAILS = ("low", "high", "original", "auto")

_TRUTHY = {"1", "true", "yes", "on", "y", "t"}
_FALSY = {"0", "false", "no", "off", "n", "f"}


def _env_str(name: str) -> str | None:
    """读字符串环境变量；空串按「没设置」处理（.env 里留空是常见写法）。"""
    raw = os.getenv(name)
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def _env_float(name: str) -> float | None:
    raw = _env_str(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _env_int(name: str) -> int | None:
    raw = _env_str(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _env_bool(name: str) -> bool | None:
    """三态布尔：未设置 -> None（表示「不显式传」），能认的值 -> True/False，
    认不出的一律当没设置（宁可让服务端决定，也不要猜错把思考关掉）。"""
    raw = _env_str(name)
    if raw is None:
        return None
    low = raw.lower()
    if low in _TRUTHY:
        return True
    if low in _FALSY:
        return False
    return None


def _env_effort() -> str | None:
    raw = _env_str("QSML_REASONING_EFFORT")
    if raw is None:
        return None
    low = raw.lower()
    return low if low in REASONING_EFFORTS else None


def _env_detail() -> str | None:
    raw = _env_str("QSML_IMAGE_DETAIL")
    if raw is None:
        return None
    low = raw.lower()
    return low if low in IMAGE_DETAILS else None


def api_key_from_env() -> str | None:
    """只读环境变量 DEEPSEEK_API_KEY（没设就是 None）。

    单独拎出来，是因为 Key 是唯一「None 表达不了任何语义」的配置项：其余字段的 None 都表示
    「这个参数不发」，而 Key 没有「不发」这种状态。所以显式传入的 Settings 即使 api_key=None，
    也只说明「这份配置没带 Key」，不该把「环境变量」这一层一起关掉。
    回落逻辑在 Locator.__init__，用它的地方都有注释说明为什么。
    """
    return _env_str("DEEPSEEK_API_KEY")


@dataclass(frozen=True)
class Settings:
    """一次调用要用到的全部配置快照（不可变）。

    字段全是「最终生效值」：None 表示**不发送该参数**，而不是「用某个默认值」。
    这个区别很重要 —— 服务端自己会重试、也会有自己的默认档，我们不替它做决定。
    """

    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = 2
    # 思考模式开关：None = 不传（服务端默认开启），True/False = 显式开关。
    thinking: bool | None = None
    reasoning_effort: str | None = None
    image_detail: str | None = None
    # 输出上限，含思考 token。None = 不传。content 为空时优先怀疑它太小。
    max_tokens: int | None = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    # 调试日志路径。None = 不开（默认），字符串 = 写到该文件（JSONL，追加）。
    # 这里只收路径，别收 DebugLog 对象 —— Settings 是要能序列化、能比较的配置，
    # 「日志写到哪」是配置，「日志对象怎么构造」不是。转换见 debuglog.coerce_log()。
    log_file: str | None = None
    # 中文标签要用的字体文件路径。None = 自动探测（见 drawing.resolve_font）。
    #
    # 为什么它该进 Settings：调用方一旦自己建 Locator，就会用 Locator(font_path=...) 或
    # Settings 传配置，而字体在**打标阶段**才用得上 —— 值必须能一路活到 draw()，
    # 中途每一层都手工透传一遍才叫真的容易漏（App 那边原先只能 monkeypatch 私有函数）。
    # 它跟着 merged() 走，所以 Locator(settings=..., font_path=...) 这类写法也能生效。
    #
    # 注意它**不是**「发给模型」的参数：不进报文，只影响画标签那一步。
    font_path: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        """按环境变量构造（未设置的项用内置默认）。"""
        retries = _env_int("QSML_MAX_RETRIES")
        return cls(
            api_key=_env_str("DEEPSEEK_API_KEY"),
            base_url=_env_str("DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL,
            model=_env_str("DEEPSEEK_MODEL") or DEFAULT_MODEL,
            timeout=_env_float("QSML_TIMEOUT") or DEFAULT_TIMEOUT,
            max_retries=2 if retries is None else retries,
            thinking=_env_bool("QSML_THINKING"),
            reasoning_effort=_env_effort(),
            image_detail=_env_detail(),
            max_tokens=_env_int("QSML_MAX_TOKENS"),
            system_prompt=_env_str("QSML_SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT,
            log_file=_env_str("QSML_LOG_FILE"),
            # 环境变量名与 drawing.FONT_PATH_ENV 是同一个串（QSML_FONT_PATH）。
            # 这里写字面量而不 import drawing：config 是被所有模块 import 的底座，
            # 让它反过来 import 一个要拉起 PIL 的模块，会让「只想用解析/提示词」的人
            # 也被迫加载 Pillow —— 那正是 P0-2 要避免的那类依赖绑架。
            # 两边一旦写岔，tests/test_fonts.py 会红。
            font_path=_env_str("QSML_FONT_PATH"),
        )

    def merged(self, **overrides: Any) -> "Settings":
        """返回一份新配置，只覆盖显式给了值的项（None 视为「没给」）。

        为什么 None 不算覆盖：调用方经常把「我没意见」写成 None（例如 thinking=None
        表示沿用服务端默认）。若把 None 也覆盖进去，就会把上面几层已经定好的值抹掉。
        想真正清空某个值，用 replace(settings, field=None) 显式表达。
        """
        valid = {f.name for f in fields(self)}
        clean = {k: v for k, v in overrides.items() if v is not None and k in valid}
        if not clean:
            return self
        return replace(self, **clean)

    def require_api_key(self) -> str:
        """取 API Key，没有就抛 MissingAPIKeyError（含怎么配的提示）。"""
        from .errors import MissingAPIKeyError

        if self.api_key and self.api_key.strip():
            return self.api_key.strip()
        raise MissingAPIKeyError(
            "未配置 DeepSeek API Key。三种给法任选一种：\n"
            "  1) 环境变量 DEEPSEEK_API_KEY=sk-xxx\n"
            "  2) locator = Locator(api_key='sk-xxx')\n"
            "  3) locate(image, target, api_key='sk-xxx')\n"
            "本库刻意不提供「无 Key 时返回假数据」的降级，以免把假坐标当成真实结果。"
        )


def redacted(settings: Mapping[str, Any] | Settings) -> dict:
    """把配置转成可安全打印的字典（Key 只留首尾，避免进日志）。"""
    data = dict(settings) if isinstance(settings, Mapping) else dict(vars(settings))
    key = data.get("api_key")
    if isinstance(key, str) and key:
        data["api_key"] = f"{key[:6]}...{key[-4:]}" if len(key) > 12 else "***"
    return data


__all__ = [
    "Settings",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_TIMEOUT",
    "REASONING_EFFORTS",
    "IMAGE_DETAILS",
    "redacted",
]
