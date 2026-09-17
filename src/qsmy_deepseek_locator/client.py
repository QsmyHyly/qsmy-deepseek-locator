"""模型客户端：拼报文 + 流式收包，把 DeepSeek 的细节挡在这一层里。

对外有三个东西：
    ChatReply                     一次调用的完整结果（正文 / 思考 / 工具调用 / usage / 结束原因）
    DeepSeekVisionClient.complete 收完整个流，返回 ChatReply（可选 on_event 回调看进度）
    DeepSeekVisionClient.stream   逐段产出事件（最细的一层，自己收自己拼）

三层粒度（一条龙 locate_to_file / 结构化+进度 locate / 只要事件流）见
@doc README.md#7-看过程流式事件
（该文档解决"想实时看思考、正文、工具调用时该调哪个入口"的问题。）
可跑示例：examples/stream_events.py。

为什么**永远走流式**（哪怕调用方只想要最终文本）：
    1. 思考模型先吐 reasoning_content 再吐 content，非流式时这段时间是完全静默的，
       命令行里看着像卡死；流式则能实时显示「模型在想什么」。
    2. on_event 回调让 CLI / GUI 不必自己实现一套流式解包。
    3. 少一条代码路径就少一处两边行为不一致的可能。
    4. **流式还是「不超时」的关键**：超时是按「两次数据之间的静默」算的，
       而不是整轮请求的总时长 —— 只要 chunk 不断到来，一次调用就能远远超过 timeout 继续跑。
       实测（同一张图、同一份报文、timeout=2s、max_retries=0）：
           流式   -> 总耗时 7.42s 正常返回（2880 个 chunk，相邻 chunk 最大间隔 507ms）
           非流式 -> 2.14s 就被 APITimeoutError 打断
       也就是说：关掉流式，本来能成的请求会直接失败。
       ⚠️ 但静默期照样算超时：模型迟迟不吐第一个字时仍然会被打断，所以 timeout 别设得太贴。

必须先知道的两个坑（都是实测踩出来的）：

    a) thinking 参数**不是** Chat Completions 的顶层字段，只能塞进 extra_body；
       reasoning_effort 反而是顶层参数。
    b) 思考 token 与正文共用输出上限：上限被思考吃满时 content 是空串，
       而 HTTP 状态码依然是 200。所以「空正文」要当成明确的错误抛出来，
       而不是当成「图里没有目标」——那是两回事。

@doc docs/API-NOTES.md#4-思考模式与-reasoning_effort
（该文档解决"thinking 参数为什么必须走 extra_body、effort 各档怎么映射"的问题。）

@doc docs/API-NOTES.md#6-流式必须同时读-reasoning_content
（该文档解决"流式 chunk 长什么样、为什么不能只读 delta.content"的问题。）

拆分后的职责与边界（0.1.2 -> 0.1.3 的等价重构，行为零变化）：
    本模块只剩**客户端本身** —— ChatReply / VisionClient 协议 / DeepSeekVisionClient，
    外加两条错误措辞（_empty_hint / _format_api_error）。
    拼报文那批纯函数（image_part / thinking_payload / resolve_thinking / build_messages /
    build_request，原本在本文件第 100-201 行）搬到了 request_build.py；
    chunk 到事件的解包（_tool_call_events / _accumulate_tool_calls / _stream_events /
    _events_from_chunk / _events_from_completion / _usage_dict，原本在第 444-591 行）
    搬到了 stream_events.py。
    两者都在下面 import 回来，所以 from .client import build_request 这类老路径
    （含 App 项目 vendor 的镜像）照旧可用 —— **client.py 仍然是对外那一个门面**。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol

# IMAGE_DETAILS / REASONING_EFFORTS 在拆分前就随「拼报文」那批一起从这里可见，
# 保留它们是为了不改动任何一个旧 import 路径（谁在用没法从本仓库看出来）。
from .config import IMAGE_DETAILS, REASONING_EFFORTS, Settings, redacted
from .debuglog import DebugLog
from .errors import APIError, EmptyResponseError, LocatorError

# --------------------------------------------------------------------------- #
# 拆分出去、但继续从本模块对外暴露的名字（老 import 路径的兼容层）
#
# 这些名字在拆分前就定义在 client.py 里：拼报文的一批是纯函数（便于单测直接断言），
# 解包的一批是下划线开头的内部件（自测也直接 import 它们喂假 chunk）。
# 实现搬走了，名字留在这里 —— 少动一处 import，就少一个镜像漂移的机会。
# --------------------------------------------------------------------------- #
from .request_build import (
    build_messages,
    build_request,
    image_part,
    resolve_thinking,
    thinking_payload,
)
from .stream_events import (
    _accumulate_tool_calls,
    _events_from_chunk,
    _events_from_completion,
    _stream_events,
    _tool_call_events,
    _usage_dict,
)


# 事件回调签名：六种事件，字段见 DeepSeekVisionClient.stream() 的 docstring。
#   reasoning / content  模型在想什么、写了什么（片段）
#   tool_call            模型要调哪个工具、参数拼到哪了（片段）
#   finish / usage / model  结束原因、用量、服务端实际使用的模型名
EventCallback = Callable[[dict], None]


@dataclass
class ChatReply:
    """一次模型调用的结果。"""

    text: str = ""
    reasoning: str = ""
    model: str = ""
    finish_reason: str | None = None
    usage: dict | None = None
    # 模型请求调用的工具（OpenAI 格式，已按 index 拼回完整对象）；没调工具就是空列表。
    # 这个字段非空时 text 通常就是空串 —— 那是「模型把话全说在工具调用里了」，
    # 不是模型抽风，所以 complete() 不会把它当成空正文报错。
    tool_calls: list[dict] = field(default_factory=list)

    @property
    def truncated(self) -> bool:
        """本轮是否因为输出上限被截断（finish_reason == length）。"""
        return self.finish_reason == "length"


class VisionClient(Protocol):
    """客户端协议。测试里注入假客户端只需要实现这一个方法。

    settings 是**本次调用**生效的配置（单次覆盖就是靠它传下来的），None = 用客户端自己的配置。

    ⚠️ **签名要和实现一样宽**（0.1.3 修）。协议原先只声明 messages/settings/on_event/log
    四个参数，而实现 DeepSeekVisionClient.complete() 还接受 tools / tool_choice —— 同一份
    文件里协议比实现窄。后果不是报错，而是**静默丢功能**：按协议写签名的自备客户端
    （安卓那台机器上只能自备，因为 openai 装不上）会把这两个参数吃掉，于是用户明明打开了
    「工具」开关，模型却永远收不到工具清单；现象与「模型这次恰好不想调工具」无法区分。
    所以这里把两个参数补进来 —— 纯增量，对已有实现与调用方都没有影响。
    """

    def complete(
        self,
        messages: list[dict],
        *,
        settings: "Settings | None" = None,
        on_event: EventCallback | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        log: DebugLog | None = None,
    ) -> ChatReply:
        ...

    # 注：log **只在开启日志时**才会被传进来（见 locate.py 里那个 log_kwarg）。
    # 这样自备客户端、自己实现了本协议的老代码，不用日志功能时完全不受影响。
    # tools / tool_choice 同理：本库自己的 locate() 这一路不传它们（不替调用方声明工具），
    # 只有直接调 complete() 的调用方才用得上。


class DeepSeekVisionClient:
    """基于 openai SDK 的 DeepSeek 客户端（OpenAI 兼容协议）。

    SDK 的导入放在 __init__ 里：只想用解析 / 打标（不联网）的调用方
    不该因为环境里没装 openai 就 import 失败。
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self._client: Any = None
        # 连接参数变了就重建 SDK 客户端（单次覆盖可能换 base_url / timeout / key）。
        # 按 key 缓存而不是每次 new：OpenAI() 会新建一个 httpx 连接池，每调用一次建一个太浪费。
        self._client_key: tuple | None = None

    def client_for(self, settings: Settings) -> Any:
        key = (settings.api_key, settings.base_url, settings.timeout, settings.max_retries)
        if self._client is not None and self._client_key == key:
            return self._client
        # 先要 Key 再 import openai：两者都缺时，「没设 DEEPSEEK_API_KEY」是用户当场
        # 就能自己修的那一个，而「没装 openai」是环境问题。反过来写会让最该看到的那条
        # 报错被另一条整个盖住 —— 实测在安卓上（装不了 openai）缺 Key 的提示完全看不见，
        # 只看到「缺少依赖 openai」，而装 openai 恰恰是那台机器上做不到的事。
        api_key = settings.require_api_key()
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - 环境问题
            # openai 自 0.1.3 起是**可选依赖**（它的 jiter / pydantic-core 是 Rust 扩展，
            # 安卓/aarch64 上没有 wheel，装了也白装）。所以这里要给出两条都走得通的路，
            # 而不是一句「装 openai」——在装不上的机器上，那句话等于没说。
            raise APIError(
                "缺少依赖 openai（自 0.1.3 起它是可选依赖）。两种解法任选一种：\n"
                "  1) 装它：pip install qsmy-deepseek-locator[openai]\n"
                "  2) 不装：改用本库自带的裸 HTTP 客户端 ——\n"
                "     from qsmy_deepseek_locator import RequestsVisionClient\n"
                "     locator = Locator(client=RequestsVisionClient(api_key='sk-xxx'))\n"
                "     它只依赖 requests，与 openai 完全无关（安卓 App 就是这么跑的）。"
            ) from exc
        self._client = OpenAI(
            api_key=api_key,
            base_url=settings.base_url,
            timeout=settings.timeout,
            max_retries=settings.max_retries,
        )
        self._client_key = key
        return self._client

    # -- 内部：发起请求，stream_options 不被支持时自动退一步重试 -------- #
    def _create(self, kwargs: dict, settings: Settings) -> Any:
        client = self.client_for(settings)
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            # 有些 OpenAI 兼容服务（含自建代理）不认 stream_options，会直接 400。
            # 这只影响「能不能顺手拿到 usage」，不值得让整轮识别失败，所以去掉它重试一次。
            #
            # 重试条件收窄到「HTTP 400（或拿不到状态码）+ 错误文本点名 stream_options」：
            # 只看子串的话，连 500 / 401 这种根本不可能是「参数不认识」的错误也会被重发一次。
            # 被拒的请求本身不产生模型开销，所以这不是钱的问题，而是**白等一个来回**、
            # 并且让真正的错误（鉴权、超时）晚一轮才浮出来。
            # status 为 None 时仍按老规矩试一次：个别自建服务抛的不是 SDK 的 HTTP 异常，
            # 拿不到状态码，宁可多试一次也别把它们挡在门外。
            status = getattr(exc, "status_code", None)
            if (
                "stream_options" in kwargs
                and (status is None or status == 400)
                and "stream_options" in str(exc)
            ):
                retry = {k: v for k, v in kwargs.items() if k != "stream_options"}
                try:
                    return client.chat.completions.create(**retry)
                except Exception as exc2:  # noqa: BLE001
                    # 两次都失败时把第一次也带上：重试后的报错常常只是同一个问题换个说法，
                    # 只报第二次会让「到底哪里不对」这条线索断掉（实测：第一次 400 说不认
                    # stream_options，第二次 401 说 Key 无效，只报后者会让人去查 Key）。
                    raise APIError(
                        f"{_format_api_error(exc2, settings)}"
                        f"（去掉 stream_options 重试前的那次失败：{type(exc).__name__}: {exc}）"
                    ) from exc2
            raise APIError(_format_api_error(exc, settings)) from exc

    def stream(
        self,
        messages: list[dict],
        *,
        settings: Settings | None = None,
        stream: bool = True,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        log: DebugLog | None = None,
    ) -> Iterator[dict]:
        """逐段产出事件（生成器）。六种事件与各自的字段：

            {"type": "reasoning", "text": str}                 思考内容的一个片段
            {"type": "content",   "text": str}                 正文的一个片段
            {"type": "tool_call", "index": int,                工具调用的一个片段
                                  "id": str | None,            首个分片才有 id 和函数名
                                  "name": str | None,
                                  "arguments": str}            参数的**增量**，不是完整 JSON
            {"type": "finish",    "reason": str}               结束原因 stop / length / tool_calls
            {"type": "usage",     "usage": dict}               用量（只在最后一个 chunk，可能要不到）
            {"type": "model",     "model": str}                服务端实际使用的模型名

        这是全库最细的一层：想边收边显示（思考 / 正文 / 工具调用分栏、实时打字机效果），
        直接用这个生成器，或者给 complete() / locate() 传 on_event —— 两条路拿到的是同一批事件。

        三个必须知道的点：
        - text 类片段的**切分是任意的**（按 token 而非按字/句），拼起来才是完整内容；
        - tool_call 的 arguments 同理，且流式下一个字符就是一个 chunk，
          要拿到可 json.loads 的完整参数必须按 index 自己拼（complete() 已经帮你拼好）；
        - usage 可能永远不来（个别兼容服务不认 stream_options），别等它。

        log 给 DebugLog 时，请求体、每个事件、以及任何异常都会追进那个日志文件
        （见 debuglog.py）。**直接调用本方法时不会写 reply/result 行** ——
        那两行需要「一次调用已经结束」或「解析结果」才知道，分别由 complete() 与 locate() 写。
        """
        effective = settings or self.settings
        kwargs = build_request(
            effective, messages, stream=stream, tools=tools, tool_choice=tool_choice
        )
        if log is not None:
            # 请求体在**发出去之前**落盘：这样即使请求打不通，日志里也有完整报文可看。
            # 图片 data URL 会被 debuglog 换成占位符，API Key 由 redacted() 脱敏。
            log.write("request", {"settings": redacted(effective), "body": kwargs})
        try:
            raw = self._create(kwargs, effective)
            events = (
                _events_from_completion(raw) if not stream else _stream_events(raw, log)
            )
            for event in events:
                if log is not None:
                    log.write("event", event)
                yield event
        except Exception as exc:  # noqa: BLE001 - 记一笔，再按本库的口径抛出去
            if log is not None:
                log.write("error", {"type": type(exc).__name__, "message": str(exc)})
            # 流**中途**出错（服务端断连、读超时、流里回一个 error 事件）时，SDK 抛的是它
            # 自己的异常类型，与本库的 APIError 不是同一个类；而调用方（含本库 CLI）都按
            # LocatorError 兜底 —— 不在这里统一，用户拿到的是一个裸栈而不是「错误：…」。
            # errors.py 里 APIError 的定义本来就写着「网络、鉴权、限流、服务端 5xx」，
            # 这属于兑现那条承诺；原始异常挂在 __cause__ 上，没被吃掉。
            if isinstance(exc, LocatorError):  # 已经是本库异常（如 _create 抛的）原样放行
                raise
            raise APIError(_format_api_error(exc, effective)) from exc

    def complete(
        self,
        messages: list[dict],
        *,
        settings: Settings | None = None,
        on_event: EventCallback | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        log: DebugLog | None = None,
    ) -> ChatReply:
        """收完整个流，返回 ChatReply。

        settings 为**本次调用**的配置覆盖（None = 用客户端自身配置）。
        tools 原样透传给服务端，模型要调用时结果落在 ChatReply.tool_calls；
        **本方法不执行工具**，要不要跑、跑完怎么把结果发回去，都是调用方的事。
        log 给 DebugLog 时，除了 stream() 那些行，还会追一条 reply（完整响应体）
        与一条 error（空正文这类失败）。
        """
        effective = settings or self.settings
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_acc: dict[int, dict] = {}
        finish_reason: str | None = None
        usage: dict | None = None
        model = effective.model

        for event in self.stream(
            messages, settings=effective, tools=tools, tool_choice=tool_choice, log=log
        ):
            etype = event.get("type")
            if etype == "reasoning":
                reasoning_parts.append(event["text"])
            elif etype == "content":
                text_parts.append(event["text"])
            elif etype == "tool_call":
                _accumulate_tool_calls(tool_acc, event)
            elif etype == "finish":
                finish_reason = event.get("reason") or finish_reason
            elif etype == "usage":
                usage = event.get("usage")
            elif etype == "model":
                model = event.get("model") or model
            if on_event is not None:
                on_event(event)

        reply = ChatReply(
            text="".join(text_parts),
            reasoning="".join(reasoning_parts),
            model=model,
            finish_reason=finish_reason,
            usage=usage,
            tool_calls=[tool_acc[index] for index in sorted(tool_acc)],
        )
        if log is not None:
            # 响应体：拼好的完整结果（正文 / 思考 / 工具调用 / usage / 结束原因）。
            log.write("reply", reply)
        # 有工具调用时正文为空是**正常**的（模型把话都说在 tool_calls 里了，finish_reason
        # 会是 tool_calls），不能按「空正文」报错；既没正文又没工具调用才是真出问题。
        if not reply.text.strip() and not reply.tool_calls:
            hint = _empty_hint(reply)
            if log is not None:
                log.write("error", {"type": "EmptyResponseError", "message": hint})
            raise EmptyResponseError(hint)
        return reply


def _empty_hint(reply: ChatReply) -> str:
    """正文为空时给一句能直接照做的提示（这类问题九成是输出预算被思考吃光）。"""
    reasoning_chars = len(reply.reasoning)
    if reply.truncated:
        return (
            f"模型返回的正文是空的，且本轮因输出上限被截断（finish_reason=length，"
            f"思考内容 {reasoning_chars} 字）。思考 token 与正文共用输出上限，"
            "思考把预算吃光时正文就是空串（HTTP 仍是 200）。"
            "对策：调大 max_tokens，或关掉思考（thinking=False），或降低 reasoning_effort。"
        )
    if reasoning_chars:
        return (
            f"模型返回的正文是空的（思考内容 {reasoning_chars} 字）。"
            "可先调大 max_tokens 或关掉思考（thinking=False）再试；"
            "若仍为空，检查提示词是否要求了输出 JSON。"
        )
    return "模型返回的正文是空的（连思考内容也没有）。请检查模型名与请求参数是否被服务端接受。"


def _format_api_error(exc: Exception, settings: Settings) -> str:
    """把 SDK 异常转成一句能照着排查的话（含状态码与响应体片段）。"""
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    detail = ""
    if isinstance(body, dict):
        message = body.get("message") or body.get("error")
        detail = f"：{message}" if message else ""
    elif body:
        detail = f"：{str(body)[:300]}"
    head = f"调用 {settings.model} 失败"
    if status:
        head += f"（HTTP {status}）"
    return f"{head}{detail}（原始异常：{type(exc).__name__}: {exc}）"


__all__ = [
    "ChatReply",
    "VisionClient",
    "DeepSeekVisionClient",
    "EventCallback",
    "build_messages",
    "build_request",
    "image_part",
    "thinking_payload",
    "resolve_thinking",
]
