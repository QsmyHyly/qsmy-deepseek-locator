"""命令行入口。

两个用法（同一个可执行文件）：

    qsmy-deepseek-locator 图片.png -t "红色圆形" -o out.png
    qsmy-deepseek-locator bench --count 5 --n-shapes 3

约定（写给以后改这个文件的人）：

- **stdout 只放结果，进度一律走 stderr**。这样 --print-json 的输出可以直接管道给 jq / 别的程序，
  不会被「[思考中…]」这类进度文字污染。
- 退出码：0 成功；1 运行期错误（缺 Key / 图片读不了 / 接口报错）；2 命令行用法错误（argparse 自带）。
- Windows 上先把 stdout/stderr 切成 UTF-8：中文标签在没有改过编码的终端里会直接抛
  UnicodeEncodeError，而那是本库最常见的输出内容。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .config import IMAGE_DETAILS, REASONING_EFFORTS
from .debuglog import default_log_path
from .errors import LocatorError
from .locate import Locator, LocateResult


def _force_utf8() -> None:
    """把标准输出/错误切成 UTF-8（Windows 控制台默认可能是 GBK，中文标签会炸）。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001 - 不支持就算了，不影响主流程
            pass


def _make_progress(quiet: bool, show_reasoning: bool, show_text: bool):
    """构造流式进度回调（写 stderr）。返回 None 表示调用方不需要回调。"""
    if quiet:
        return None
    state = {"phase": "", "wrote": False}

    def on_event(event: dict) -> None:
        etype = event.get("type")
        if etype == "reasoning":
            if show_reasoning:
                if state["phase"] != "reasoning":
                    sys.stderr.write("\n--- 思考过程 ---\n")
                    state["phase"] = "reasoning"
                    state["wrote"] = True
                sys.stderr.write(event.get("text", ""))
            elif state["phase"] != "reasoning":
                sys.stderr.write("[思考中…] ")
                state["phase"] = "reasoning"
            sys.stderr.flush()
        elif etype == "content":
            if show_text:
                if state["phase"] != "content":
                    sys.stderr.write("\n--- 模型输出 ---\n")
                    state["phase"] = "content"
                sys.stderr.write(event.get("text", ""))
            elif state["phase"] != "content":
                sys.stderr.write("[生成中…] ")
                state["phase"] = "content"
            sys.stderr.flush()
        elif etype == "finish" and state["wrote"]:
            sys.stderr.write("\n")

    return on_event


def _default_out(image: str) -> Path:
    """没给 -o 时的默认标注图路径：图片名 + _annotated.png，落在当前目录。"""
    text = str(image)
    if text.startswith(("http://", "https://", "data:")):
        return Path("annotated.png")
    return Path(Path(text).stem + "_annotated.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qsmy-deepseek-locator",
        description="用 DeepSeek 视觉模型做物体定位：一张图 + 一句「找什么」，得到归一化坐标与中文标签。",
        epilog="子命令 bench：跑合成图的准确率评测。例：qsmy-deepseek-locator bench --count 5",
    )
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("image", help="图片：本地路径 / http(s) URL")
    parser.add_argument("-t", "--target", default=None, help='找什么，例如 "红色圆形"、"登录按钮"')
    parser.add_argument("-p", "--prompt", default=None, help="直接给完整的用户消息（给了就忽略 --target）")
    parser.add_argument("-o", "--out", default=None, help="标注图保存路径（默认 <图片名>_annotated.png）")
    parser.add_argument("--no-draw", action="store_true", help="只输出坐标，不生成标注图")
    parser.add_argument("--json", dest="json_path", default=None, help="把完整结果连同证据写成 JSON 文件")
    parser.add_argument("--print-json", action="store_true", help="把结果 JSON 打到 stdout（替代人类可读摘要）")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--thinking", dest="thinking", action="store_true", default=None,
                       help="强制开启思考模式（默认由服务端决定）")
    group.add_argument("--no-thinking", dest="thinking", action="store_false",
                       help="关闭思考模式：更快、更省 token（坐标类任务实测不掉准确率）")

    parser.add_argument("--effort", choices=sorted(REASONING_EFFORTS), default=None,
                        help="思考强度（仅思考开启时生效）")
    parser.add_argument("--detail", choices=list(IMAGE_DETAILS), default=None,
                        help="图片精度 low/high/original/auto（默认不发送该字段）")
    parser.add_argument("--model", default=None, help="模型名（默认 deepseek-flash）")
    parser.add_argument("--base-url", default=None, help="接口地址（默认官方；也可指向自建兼容服务）")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="输出上限（含思考 token）。正文为空时优先调大它")
    parser.add_argument("--timeout", type=float, default=None, help="单次请求超时（秒）")
    parser.add_argument("--max-side", type=int, default=None, help="发送前把图缩到最长边不超过该值")
    parser.add_argument("--system-prompt", default=None, help="覆盖系统提示词（承载坐标口径，慎改）")
    parser.add_argument("--log-file", dest="log_file", default=None, metavar="PATH",
                        help="把网络层请求体/响应体写成 JSONL 调试日志（默认不开）")
    parser.add_argument("--log", dest="log_auto", action="store_true",
                        help="开调试日志，路径自动取 runs/logs/qsml-<时间戳>.jsonl")
    parser.add_argument("--show-reasoning", action="store_true", help="把思考过程实时打到 stderr")
    parser.add_argument("--show-text", action="store_true", help="把模型正文实时打到 stderr")
    parser.add_argument("-q", "--quiet", action="store_true", help="不打印任何进度")
    return parser


def locate_main(argv: Sequence[str]) -> int:
    args = build_parser().parse_args(list(argv))

    locator = Locator(
        model=args.model,
        base_url=args.base_url,
        timeout=args.timeout,
        thinking=args.thinking,
        reasoning_effort=args.effort,
        image_detail=args.detail,
        max_tokens=args.max_tokens,
        system_prompt=args.system_prompt,
        max_side=args.max_side,
    )

    log_file = args.log_file or (str(default_log_path()) if args.log_auto else None)
    if not args.quiet and log_file:
        # 先说日志写到哪，再干活：这次的输出就是给人「照着文件去翻」用的
        sys.stderr.write(f"调试日志：{log_file}\n")

    try:
        result = locator.locate(
            args.image,
            args.target,
            prompt=args.prompt,
            log_file=log_file,
            on_event=_make_progress(args.quiet, args.show_reasoning, args.show_text),
        )
    except LocatorError as exc:
        # 只兜本库自己的异常：别的异常（含用户中断）照常抛出去，便于定位问题
        sys.stderr.write(f"\n错误：{exc}\n")
        return 1

    if not args.quiet:
        sys.stderr.write("\n")

    if args.print_json:
        sys.stdout.write(result.to_json() + "\n")
    else:
        sys.stdout.write(result.describe() + "\n")

    if args.json_path:
        saved = result.save(args.json_path, include_raw=True)
        sys.stderr.write(f"结果 JSON：{saved}\n")

    if not args.no_draw:
        from .drawing import save_annotated

        out = Path(args.out) if args.out else _default_out(args.image)
        try:
            path = save_annotated(args.image, result.detections, path=out)
            sys.stderr.write(f"标注图：{path}\n")
        except LocatorError as exc:
            sys.stderr.write(f"标注图保存失败：{exc}\n")
            return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    _force_utf8()
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "bench":
        from .benchmark import bench_main

        return bench_main(args[1:])
    return locate_main(args)


__all__ = ["main", "locate_main", "build_parser"]
