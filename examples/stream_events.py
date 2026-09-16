"""逐条看模型在干什么：流式事件长什么样、怎么接。

跑法（先配好 DEEPSEEK_API_KEY）：

    python examples/stream_events.py                      # 定位请求的事件流（逐条带时间戳）
    python examples/stream_events.py --tools              # 带 tools 的请求：工具调用是怎么一片片吐出来的
    python examples/stream_events.py --raw                # 连事件的 JSON 原样打出来
    python examples/stream_events.py path/to/img.png "红色圆形"

两种情况都会真实调用 API（联网、要花钱）。产物写进 runs/example/（.gitignore 已忽略）。

本库的粒度分三层，按需要挑：

    1. locate_to_file(...)                        一条龙：图进 -> 标注图出（README 3.1）
    2. Locator.locate(..., on_event=cb)           要结构化结果 + 想看进度（本示例 default 模式）
    3. DeepSeekVisionClient.stream(messages, ...) 只要事件流，自己收自己拼（本示例 --tools 模式）

六种事件（字段见 client.stream() 的 docstring）：
    reasoning  思考片段      content  正文片段      tool_call  工具调用片段
    finish     结束原因      usage    用量          model      服务端实际用的模型名
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

from qsmy_deepseek_locator import (
    DeepSeekVisionClient,
    Locator,
    build_messages,
    to_data_url,
)

OUTPUT_DIR = Path("runs/example")

# 一个「工具」的声明：本库只负责把它透传给服务端，**不负责执行**。
# 这里声明的是「裁剪一块区域」，参数用本库一贯的 0.0~1.0 归一化坐标。
CROP_TOOL = {
    "type": "function",
    "function": {
        "name": "crop_region",
        "description": "把图中指定矩形区域裁剪出来另存为一张图",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "这块区域是什么物体，中文"},
                "x1": {"type": "number", "description": "左上角 x，0.0~1.0 归一化"},
                "y1": {"type": "number", "description": "左上角 y，0.0~1.0 归一化"},
                "x2": {"type": "number", "description": "右下角 x，0.0~1.0 归一化"},
                "y2": {"type": "number", "description": "右下角 y，0.0~1.0 归一化"},
            },
            "required": ["name", "x1", "y1", "x2", "y2"],
        },
    },
}


def _force_utf8() -> None:
    """Windows 控制台默认可能是 GBK，中文事件会炸，先切成 UTF-8。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass


def make_example_image(path: Path) -> Path:
    """现画一张 900x700 的示例图：一个红圆、一个蓝矩形、一个绿三角。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (900, 700), (247, 249, 252))
    painter = ImageDraw.Draw(img)
    painter.ellipse([120, 120, 320, 320], fill=(214, 48, 49))
    painter.rectangle([520, 140, 780, 330], fill=(52, 96, 219))
    painter.polygon([(450, 600), (620, 400), (790, 600)], fill=(46, 160, 67))
    img.save(path, format="PNG")
    return path


class EventPrinter:
    """把事件流打成人类看得懂的样子，并顺手统计。

    两个刻意的地方：
    - 思考/正文**按片段原样续着打**（这才是真的流式观感），只在切换阶段时换行加标题；
    - tool_call 反而要**逐片显示**，因为「一个字符一个 chunk」正是它最容易踩坑的地方。
    """

    def __init__(self, *, raw: bool = False, show_text: bool = True):
        self.raw = raw
        self.show_text = show_text
        self.start = time.perf_counter()
        self.counts: dict[str, int] = {}
        self.first: dict[str, float] = {}
        self._phase: str | None = None
        self.calls: dict[int, dict] = {}

    def _stamp(self) -> str:
        return f"{time.perf_counter() - self.start:6.2f}s"

    def _note(self, kind: str) -> None:
        """记下每类事件第一次到达的时刻 —— 它就是这类内容的首字延迟。"""
        self.counts[kind] = self.counts.get(kind, 0) + 1
        self.first.setdefault(kind, time.perf_counter() - self.start)

    def __call__(self, event: dict) -> None:
        kind = event.get("type", "?")
        self._note(kind)
        if self.raw:
            print(f"[{self._stamp()}] {json.dumps(event, ensure_ascii=False)}")
            return
        if kind == "reasoning":
            self._switch("reasoning", "\n--- 思考过程 ---")
            if self.show_text:
                print(event.get("text", ""), end="", flush=True)
        elif kind == "content":
            self._switch("content", "\n--- 正文 ---")
            if self.show_text:
                print(event.get("text", ""), end="", flush=True)
        elif kind == "tool_call":
            self._switch("tool_call", "\n--- 工具调用分片 ---")
            index = event.get("index")
            slot = self.calls.setdefault(index, {"id": None, "name": None, "arguments": ""})
            if event.get("id"):
                slot["id"] = event["id"]
            if event.get("name"):
                slot["name"] = event["name"]
            slot["arguments"] += event.get("arguments") or ""
            print(
                f"\n[{self._stamp()}] #{index} "
                f"id={event.get('id')} name={event.get('name')} "
                f"arguments+={event.get('arguments')!r}",
                end="", flush=True,
            )
        elif kind == "usage":
            print(f"\n[{self._stamp()}] usage={event.get('usage')}")
        elif kind == "model":
            print(f"[{self._stamp()}] model={event.get('model')}")
        elif kind == "finish":
            print(f"\n[{self._stamp()}] finish_reason={event.get('reason')}")

    def _switch(self, phase: str, title: str) -> None:
        if self._phase != phase:
            if self._phase is not None:
                print()
            print(f"{title}  ({self._stamp()})", flush=True)
            self._phase = phase

    def summary(self) -> None:
        print("\n" + "=" * 62)
        print(f"事件总数：{sum(self.counts.values())}")
        for kind, count in self.counts.items():
            print(f"  {kind:<10} {count:>4} 条   首次到达 {self.first[kind]:.2f}s")
        print(f"总耗时：{time.perf_counter() - self.start:.2f}s")
        if self.calls:
            print("拼回来的工具调用（arguments 是逐字符攒的，这里已能 json 解析）：")
            for index, slot in sorted(self.calls.items()):
                args = slot["arguments"]
                try:
                    parsed = json.loads(args)
                except json.JSONDecodeError:
                    parsed = f"<还不是合法 JSON，差一点：{args!r}>"
                print(f"  #{index} {slot['name']}({slot['id']}) -> {parsed}")


def run_locate(image: str, target: str | None, printer: EventPrinter) -> int:
    """第 2 层：要结构化结果，同时用 on_event 看过程。"""
    print(f"图片：{image}\n提问：{target or '（默认：识别主要物体）'}")
    result = Locator(thinking=False).locate(image, target, on_event=printer)
    printer.summary()
    print("\n" + result.describe())
    print("\n注：locate 这一路**不会**有 tool_call 事件 —— 它不带 tools 参数。")
    print("    想看工具调用分片，跑：python examples/stream_events.py --tools")
    return 0


def run_tools_demo(image: str, printer: EventPrinter) -> int:
    """第 3 层：自己拼报文、自己发请求，看模型怎么一片片吐出工具调用。

    这里用的是 DeepSeekVisionClient.stream() —— 全库最细的一层。
    注意它**只负责把事件给你**：工具是你声明、你执行的，往返要自己接
    （本库 v0.1 没有 Agent 循环，见 README「已知边界」）。
    """
    client = DeepSeekVisionClient()
    messages = build_messages(
        "请调用 crop_region 工具，框出图中红色圆形所在区域。必须调用工具，不要只用文字回答。",
        image_url=to_data_url(image),
        image_detail="original",
    )
    print(f"图片：{image}\n请求：带 tools=[crop_region] 的流式请求\n")
    for event in client.stream(messages, tools=[CROP_TOOL], tool_choice="auto"):
        printer(event)
    printer.summary()
    print("\n拿到 tool_calls 之后要做什么，由你决定 —— 本库到「事件的 JSON」为止。")
    return 0


def main(argv: list[str]) -> int:
    _force_utf8()
    raw = "--raw" in argv
    argv = [a for a in argv if a != "--raw"]
    want_tools = "--tools" in argv
    argv = [a for a in argv if a != "--tools"]

    if argv:
        image = argv[0]
        target = argv[1] if len(argv) > 1 else None
    else:
        image = str(make_example_image(OUTPUT_DIR / "input.png"))
        target = "几何图形"

    printer = EventPrinter(raw=raw)
    if want_tools:
        return run_tools_demo(image, printer)
    return run_locate(image, target, printer)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
