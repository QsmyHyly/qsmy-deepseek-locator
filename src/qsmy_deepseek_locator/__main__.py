"""支持 python -m qsmy_deepseek_locator 的直接调用（等价于命令行入口）。"""

from .cli import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
