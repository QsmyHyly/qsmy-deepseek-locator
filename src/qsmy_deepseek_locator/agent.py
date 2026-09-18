"""Agent 主循环：流式输出 + 工具执行（**可选能力**，单轮定位完全不经过这里）。

流程：
    1. 调用模型（流式），把 reasoning / content / tool_call 增量实时吐出；
    2. 若本轮出现 tool_calls，交给 ToolRegistry 逐个执行，把 role=tool 的结果
       追加进消息历史，然后进入下一轮；
    3. 直到模型不再请求工具，或达到 max_tool_rounds 轮上限。

对外只暴露一个事件生成器 run_agent(...)：CLI / 网页 / 安卓 App 消费同一套事件，
所以「流式展示」与「工具执行」在本库只有一份实现。

事件类型（与下游约定好的，不要随意改）：
    round_start / reasoning / content / tool_call / tool_result / message / done / error

⚠️ **两个同名但不同义的 tool_call，别搞混**：
    - 客户端（client.stream）的 tool_call 是**增量** —— 一个字符一个 chunk，要按 index 自己拼；
    - 本模块 yield 的 tool_call 是**拼好的完整一次调用**，给前端展示用。
    名字撞车是历史原因（下游的 providers 把增量那一层叫 tool_call_delta），
    本模块内部先按 index 累积、再对外发一次完整的。

⚠️ **为什么单独成模块、不进包顶层**：agent 依赖工具框架与内置工具集，
它一被 import 就会连带拉起那些东西。单轮定位（locate / locate_to_file）是绝大多数
用法，不该为用不上的能力付出 import 代价，所以这里要显式
`from qsmy_deepseek_locator.agent import run_agent`。

@doc docs/API-NOTES.md#61-工具调用也是流式的而且一个字符一个-chunk
（该文档解决"工具调用的 arguments 为什么必须按 index 自己拼"的问题。）
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Iterator

from .client import DeepSeekVisionClient
from .config import Settings
from .parsing import decode_json_points, to_items
from .request_build import build_messages, merge_thinking
from .tools import ToolRegistry, build_default_registry

__all__ = [
    "build_messages",
    "collect_final_text",
    "collect_items",
    "extract_items",
    "run_agent",
    "run_agent_simple",
]


def _accumulate_tool_calls(pending: dict[int, dict]) -> list[dict]:
    """把流式增量拼装成 OpenAI 格式的 tool_calls 列表。

    arguments 是**分片字符串**，必须按 index 累加后才是完整 JSON（见模块头注释）。
    """
    calls = []
    for index in sorted(pending):
        entry = pending[index]
        if not entry.get("name"):
            continue
        calls.append({
            "id": entry.get("id") or f"call_{index}",
            "type": "function",
            "function": {
                "name": entry["name"],
                "arguments": entry.get("arguments") or "{}",
            },
        })
    return calls


def run_agent(
    messages: list[dict],
    *,
    client: Any | None = None,
    registry: ToolRegistry | None = None,
    settings: Settings | None = None,
    max_rounds: int | None = None,
    use_tools: bool = True,
    tool_context: dict | None = None,
    thinking: bool | None = None,
    reasoning_effort: str | None = None,
    log: Any | None = None,
) -> Iterator[dict]:
    """执行 agent 循环，逐个产出事件字典。

    Args:
        client: 任何实现了 `stream(messages, *, settings, tools, log)` 的客户端；
            留空用 DeepSeekVisionClient。本库两个客户端都满足这个契约。
        registry: 工具注册表；留空用 build_default_registry()。
        tool_context: 运行上下文，用于注入工具中不暴露给模型的参数
            （如 {"source": 当前图片地址}）。
        thinking / reasoning_effort: **按次覆盖**；None 表示用 settings 里的值
            （而 settings 自己又是一条 环境变量 > 内置默认 的小链）。
            合并规则与单轮定位共用同一个 merge_thinking，不会出现两套口径。
        log: DebugLog 或路径；给了就把每次请求体与事件追进去。

    事件类型：
        round_start / reasoning / content / tool_call / tool_result / message / done / error
    """
    settings = settings or Settings.from_env()
    client = client or DeepSeekVisionClient(settings)
    # 先合并成确定的值，保证「真实客户端 / 自定义客户端 / 网络层」看到的是同一个决定。
    enabled, effort = merge_thinking(
        thinking,
        reasoning_effort,
        default_thinking=settings.thinking,
        default_effort=settings.reasoning_effort,
    )
    # client.stream() 的 thinking 是从 Settings 里读的（没有按次形参），
    # 所以把合并结果包成一个临时 Settings 传下去 —— 与单轮定位走的是同一条路。
    call_settings = replace(settings, thinking=enabled, reasoning_effort=effort)
    registry = registry or build_default_registry()
    rounds = max_rounds or settings.max_tool_rounds
    tools = registry.spec() if use_tools else None

    final_content = ""
    for round_index in range(rounds):
        yield {"type": "round_start", "index": round_index, "thinking": bool(enabled)}

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        pending_tool_calls: dict[int, dict] = {}
        finish_reason: str | None = None

        try:
            for event in client.stream(
                messages, settings=call_settings, tools=tools, log=log
            ):
                etype = event.get("type")
                if etype == "reasoning":
                    reasoning_parts.append(event["text"])
                    yield {"type": "reasoning", "text": event["text"]}
                elif etype == "content":
                    content_parts.append(event["text"])
                    yield {"type": "content", "text": event["text"]}
                elif etype == "tool_call":
                    # 这一层是**增量**，按 index 累积（见模块头注释）
                    idx = event.get("index", 0)
                    entry = pending_tool_calls.setdefault(
                        idx, {"id": "", "name": "", "arguments": ""}
                    )
                    if event.get("id"):
                        entry["id"] = event["id"]
                    if event.get("name"):
                        entry["name"] = event["name"]
                    if event.get("arguments"):
                        entry["arguments"] += event["arguments"]
                elif etype == "finish":
                    finish_reason = event.get("reason")
        except Exception as exc:  # noqa: BLE001 - 网络/SDK 异常要反馈到前端，不能抛穿
            yield {"type": "error", "message": f"模型调用失败：{exc}"}
            return

        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts)
        tool_calls = _accumulate_tool_calls(pending_tool_calls)

        assistant_message: dict[str, Any] = {"role": "assistant", "content": content or None}
        # 带 tools 时官方要求把历史轮次的 reasoning_content 原样回传，
        # 否则模型会丢掉上一轮的思考上下文（官方文档甚至说会 400）。
        # 这里**只在确实有思考内容时**回传：关闭思考时本来就没有该字段，
        # 而不是补一个空串 —— 空串是否被接受没有实测依据，不冒这个险。
        if reasoning and enabled:
            assistant_message["reasoning_content"] = reasoning
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        messages.append(assistant_message)
        yield {"type": "message", "message": assistant_message, "finish_reason": finish_reason}

        if not tool_calls:
            final_content = content
            yield {
                "type": "done",
                "reason": finish_reason or "stop",
                "content": final_content,
                "rounds": round_index + 1,
                "thinking": bool(enabled),
                "messages": messages,
            }
            return

        # ---- 执行工具 ----
        for call in tool_calls:
            fn = call["function"]
            name = fn["name"]
            raw_args = fn["arguments"]
            yield {
                "type": "tool_call",
                "id": call["id"],
                "name": name,
                "arguments": raw_args,
            }
            result = registry.execute(name, raw_args, context=tool_context)
            yield {
                "type": "tool_result",
                "id": call["id"],
                "name": name,
                "ok": result.ok,
                "content": result.content,
                "elapsed_ms": round(result.elapsed_ms, 1),
            }
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": result.content,
            })

    yield {
        "type": "done",
        "reason": "max_rounds",
        "content": final_content,
        "rounds": rounds,
        "thinking": bool(enabled),
        "messages": messages,
    }


def extract_items(text: str) -> list[dict]:
    """从模型最终文本中抽取坐标对象列表，供标注使用。"""
    if not text:
        return []
    data = decode_json_points(text)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    return []


def collect_items(events: list[dict], final_text: str = "") -> list[dict]:
    """从事件流与最终文本中收集可标注的坐标对象。

    优先用最终正文里的 JSON；若没有，则回溯工具执行结果
    （模型可能把坐标交给 parse_coordinates 之类的工具去处理）。
    """
    items = [d for d in extract_items(final_text) if "bbox_2d" in d or "point_2d" in d]
    if items:
        return items

    for event in reversed(events):
        if event.get("type") != "tool_result" or not event.get("ok"):
            continue
        try:
            data = json.loads(event.get("content") or "")
        except Exception:  # noqa: BLE001 - 工具结果不一定是 JSON，跳过即可
            continue
        # 必须用 to_items 而不是 to_dict_items：模型常把坐标交给 parse_coordinates
        # 去处理，而那个工具的返回值是「成对列表」，里面没有 dict —— 用严格的
        # to_dict_items 会静默得到空列表（见 parsing.to_items 的 docstring）。
        got = [d for d in to_items(data) if "bbox_2d" in d or "point_2d" in d]
        if got:
            return got
    return []


def collect_final_text(messages: list[dict]) -> str:
    """取最后一条 assistant 正文。"""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return str(msg["content"])
    return ""


def run_agent_simple(prompt: str, image: str | None = None, **kwargs) -> dict:
    """非流式便捷入口：执行完整 agent 循环并返回最终结果。"""
    settings = kwargs.pop("settings", None) or Settings.from_env()
    messages = build_messages(
        prompt,
        image_url=image,
        system_prompt=settings.system_prompt,
        image_detail=settings.image_detail,
    )
    events = list(run_agent(messages, settings=settings, **kwargs))
    done = next((e for e in events if e["type"] == "done"), None)
    text = done.get("content", "") if done else ""
    return {
        "events": events,
        "text": text,
        "items": extract_items(text),
        "messages": done.get("messages") if done else messages,
    }

