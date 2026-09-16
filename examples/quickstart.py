"""最小可跑示例：定位 -> 打印 -> 打标出图。

跑法（先把 DEEPSEEK_API_KEY 配好）：

    python examples/quickstart.py                      # 用内置生成的示例图
    python examples/quickstart.py path/to/your.png "画面里的人"

它会花一次 API 调用（联网、要花钱）。
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

from qsmy_deepseek_locator import Locator, draw, save_annotated


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


def _progress(event: dict) -> None:
    """流式进度：只打阶段提示，别把几百个增量刷满屏。"""
    kind = event.get("type")
    if kind == "reasoning":
        print(".", end="", flush=True)
    elif kind == "content":
        print("\n[正文] ", end="", flush=True)


def main(argv: list[str]) -> int:
    if len(argv) >= 1:
        image = argv[0]
        target = argv[1] if len(argv) > 1 else None
    else:
        image = str(make_example_image(Path("examples/example_input.png")))
        target = "几何图形"

    # thinking=False：关掉思考模式，更快更省 token（定位类任务实测不掉准确率）
    locator = Locator(thinking=False)

    print(f"图片：{image}")
    print(f"提问：{target or '（默认：识别主要物体）'}")
    result = locator.locate(image, target, on_event=_progress)

    print()
    print(result.describe())
    print()
    print(f"usage：{result.usage}")

    # 1) 直接拿 PIL 图自己处理
    annotated = draw(image, result)

    # 2) 或者让库帮你落盘
    saved = save_annotated(image, result, path="examples/example_annotated.png")
    print(f"标注图：{saved}（{annotated.width}x{annotated.height}）")

    # 3) 结构化结果也能直接存成 JSON（include_raw 连解析前的原始项一起留证）
    json_path = result.save("examples/example_result.json", include_raw=True)
    print(f"结果 JSON：{json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
