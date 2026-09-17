"""异常类型。

为什么单独开一个模块：调用方需要能把「Key 没配」「图片读不了」「模型返回的正文是空的」
这三类问题分开处理（前两类是本地配置/输入问题，第三类往往是输出预算不够）。
全部继承 LocatorError，方便一把兜住。

刻意**不做**的事：不在缺 Key 时静默降级成假数据。旧演示项目（deepseek-vision-annotation）
无 Key 时会进 Mock 模式，那对演示页很友好，但对一个库是危险的：
用户会拿到一堆看起来正常的坐标，却以为真的调用了模型。本库缺 Key 就报错。

**契约闭合（本次改动）**：在此之前，除了下面这些 LocatorError 之外还会漏出三类裸异常 ——
`NotImplementedError`（use_tools=True）、`ValueError`（输出路径后缀 / log_file 类型）、
`OSError`（标注图最终落盘那行没有 try/except）。只写 `except LocatorError` 的调用方
会在最后一步落盘上崩掉，而异常类型也不在文档承诺里 —— 安卓 App 那边正是因此只能抓
`BaseException` 兜底。现在这三类全部收进本模块：

    裸异常                            现在抛的                    多重继承
    NotImplementedError               UnsupportedFeatureError     LocatorError, NotImplementedError
    ValueError（输出路径）             OutputPathError            LocatorError, ValueError
    ValueError（log_file 类型）        LogFileTypeError           LocatorError, ValueError
    OSError（最终落盘 / 建目录）        WriteError                 LocatorError（见下）

**为什么前面三个子类要做多重继承**：只继承 LocatorError 的话，「原来的 `except ValueError`
抓不住」就成了一次**静默的行为破坏** —— 调用方代码一行没改，异常却从 except 的缝里漏出去。
多重继承同时满足两边：老写法照旧抓到，新写法 `except LocatorError` 一把兜住，
`isinstance(exc, ValueError)` 这类判断也仍然成立。代价是继承图稍微绕一点，
换来的是一条硬承诺：**这个库抛出的东西，总是 LocatorError**。

**唯一的例外是文件系统的 OSError，刻意不给它留退路**。WriteError 只继承 LocatorError，
不继承 OSError：一来 OSError 是本库最不擅长判断的一类错误（磁盘满、只读挂载、权限、
路径长度……原因全在调用方那边，本库只能转述），并进来没给调用方任何新信息；
二来 OSError 的 `__init__` 有自己的一套 args 语义（errno / strerror），混着用会让异常对象的
`args` 说不清，而排查时看的恰恰是它。原始 OSError **一个都不会丢** —— 一律挂在 `__cause__` 上。

这条硬承诺的**适用范围要写清楚**：它覆盖识别与出图 API（定位、打标、落盘、读图、调接口、
取消、写结果 JSON）。本地评测工具（benchmark / bench）写 runs/ 下的中间产物时仍是原生
OSError —— 那跑在开发者自己机器上，失败时就该看到完整系统错误，包一层反而挡信息（见
benchmark.py 模块头）。别把「总是」理解成"包括我拿来改代码的那把锤子"。

@doc README.md#6-api-速查
（该文档解决"这个库会抛哪些异常、每个该在哪一层兜"的问题。）
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


class UnsupportedFeatureError(LocatorError, NotImplementedError):
    """调用了本版本还没有实现的功能（当前只有一处：locate_to_file(use_tools=True)）。

    它是**预留参数**的报错，不是「参数传错了」：v0.1 没有实现工具（Agent）调用循环，
    而 use_tools 留在签名里，是为了将来接上时不必改调用方。静默忽略它会让用户以为
    工具已经开了 —— 那比报错危险得多，所以这里宁可当场炸。

    继承 NotImplementedError 是为了向后兼容：0.1.2 抛的就是它（见模块头）。
    """


class OutputPathError(LocatorError, ValueError):
    """标注图的输出路径**本身**不可用：空路径，或者扩展名认不出（只认 _FORMATS 里那几个）。

    只管「路径这个字符串不对，改参数就行」这一档。至于「路径没问题但写不进去」
    （父目录建不出来、磁盘满、只读）一律归 WriteError —— 两类问题的修法不同，
    混在一起调用方就分不出该改参数还是改环境。原先两者都是裸的，见 WriteError。

    继承 ValueError 同样是向后兼容：以前 resolve_output_path 抛的就是 ValueError，
    已有调用方可能正按 `except ValueError` 兜它。
    """


class WriteError(LocatorError):
    """写文件失败（标注图落盘、输出目录创建）。

    与 OutputPathError 分开，是因为两者对应**不同的修法**：OutputPathError 是「路径本身不对，
    改参数」，WriteError 是「路径对但写不动，改环境」——磁盘满、只读挂载、没有写权限。

    刻意**不**继承 OSError，理由见模块头最后一段。原始 OSError 永远挂在 `__cause__` 上。
    """


class LogFileTypeError(LocatorError, ValueError):
    """log_file 参数的类型不认识（只接受 None / True / False / 路径 / DebugLog）。

    单独成类而不是随便抛个 ValueError：日志是个「以为自己开了、其实没开」会非常难受的东西，
    所以类型不认识时宁可报错也不静默忽略（见 debuglog.coerce_log）；给它一个能按名字兜的类型，
    调用方就不必去比对报错文本。
    """


class CancelledError(LocatorError):
    """调用被 cancel_event 取消了。

    只在调用方**自己**传了 cancel_event 时才可能出现 —— 本库不会主动取消任何请求。
    语义是「调用方改主意了」，不是「出错了」：重试没有意义，清理后直接返回即可。
    它继承 LocatorError 但**不**继承 InterruptedError / KeyboardInterrupt，
    免得被别的兜底逻辑当成系统级中断处理。
    """


__all__ = [
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
]
