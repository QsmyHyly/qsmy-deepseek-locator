"""异常类型。

为什么单独开一个模块：调用方需要能把「Key 没配」「图片读不了」「模型返回的正文是空的」
这三类问题分开处理（前两类是本地配置/输入问题，第三类往往是输出预算不够）。
全部继承 LocatorError，方便一把兜住。

刻意**不做**的事：不在缺 Key 时静默降级成假数据。旧演示项目（deepseek-vision-annotation）
无 Key 时会进 Mock 模式，那对演示页很友好，但对一个库是危险的：
用户会拿到一堆看起来正常的坐标，却以为真的调用了模型。本库缺 Key 就报错。
"""

from __future__ import annotations


class LocatorError(Exception):
    """本库所有异常的基类。"""


class MissingAPIKeyError(LocatorError):
    """没有可用的 API Key。

    取 Key 的顺序：locate(api_key=...) > Locator(api_key=...) > 环境变量 DEEPSEEK_API_KEY。
    三者都没有就抛这个，而不是偷偷走离线假数据。
    """


class ImageLoadError(LocatorError):
    """图片无法加载：路径不存在、URL 下载失败、字节内容不是有效图片等。"""


class APIError(LocatorError):
    """调用模型接口失败（网络、鉴权、限流、服务端 5xx 等）。"""


class EmptyResponseError(LocatorError):
    """接口调用成功，但模型返回的正文（content）是空的。

    最常见的成因不是模型坏了，而是**思考 token 吃光了输出预算**：
    DeepSeek 是思考模型，先产 reasoning_content 再产 content，
    两者共用同一个输出上限；上限被思考占满时 content 就是空串，而 HTTP 状态码仍是 200。
    对策：调大 max_tokens，或关掉思考模式（thinking=False），或降低 reasoning_effort。

    @doc docs/API-NOTES.md#5-思考-token-会吃掉-max_tokens
    （该文档解决"为什么正文为空却是 HTTP 200、以及三条对策的实测数据"的问题。）
    """


__all__ = [
    "LocatorError",
    "MissingAPIKeyError",
    "ImageLoadError",
    "APIError",
    "EmptyResponseError",
]
