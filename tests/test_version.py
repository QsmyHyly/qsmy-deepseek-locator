"""版本号一致性自检：__init__.py 里那个兜底字面量必须跟着 pyproject.toml 走。

为什么要为「一个字符串」专门写测试：版本号有两个来源 ——
装了包时走 importlib.metadata（真来源是 pyproject.toml 的 [project] version），
没装包时走 __init__.py 里写死的兜底字面量。**0.1.2 发版时就漏改了兜底值**，
后果是从源码 import 的使用者（没 pip install、或像 App 项目那样把源码 vendor 进去）
看到 __version__ 谎报 0.1.1 —— 而那正是最需要版本号准确的场合（排查现场第一句话
就是「你用的哪个版本」）。

「发版时记得改」已经被证明靠不住，所以这里把它钉成断言：兜底字面量 == pyproject 的 version。
只改 pyproject 不改兜底，本文件立刻红 —— 这才是真正防住下一次漏改的东西。

⚠️ 断言的是**源码里的字面量**，不是运行时的 __version__：本机开发环境里常常留着上一版的
元数据（本仓库就是 —— .venv312 里装着 0.1.0 的 dist-info，src/ 下还留着 0.1.1 的 egg-info），
此时 __version__ 会诚实地报出那个旧版本。那是元数据残留的锅、不是兜底的锅，
拿它来断言反而测不到我们要防的东西。
运行路径另有 test_fallback_used_when_metadata_is_missing 兜住。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INIT_PY = PROJECT_ROOT / "src" / "qsmy_deepseek_locator" / "__init__.py"
PYPROJECT = PROJECT_ROOT / "pyproject.toml"


def _pyproject_version() -> str:
    """取 pyproject.toml 里 [project] 表的 version。

    刻意不用 tomllib：它要 Python 3.11+，而本库 requires-python >= 3.9，测试也得跟着能跑。
    """
    text = PYPROJECT.read_text(encoding="utf-8")
    section = re.search(r"^\[project\]\s*$", text, re.MULTILINE)
    assert section is not None, "pyproject.toml 里找不到 [project] 表"
    rest = text[section.end():]
    # 只在这个表内找：下一个表头就是边界，否则会读到 [project.urls] 之类
    nxt = re.search(r"^\[", rest, re.MULTILINE)
    body = rest[: nxt.start()] if nxt else rest
    match = re.search(r'^version\s*=\s*"([^"]+)"', body, re.MULTILINE)
    assert match is not None, "pyproject.toml 的 [project] 里找不到 version"
    return match.group(1)


def _literal_fallback_version() -> str:
    """从 __init__.py 源码里取出 PackageNotFoundError 分支赋给 __version__ 的那个字面量。

    用 ast 而不是正则：正则会被文档字符串、注释里的引号带偏，而这里要的恰恰是
    「异常分支里那一个」——取错了这个测试就变成摆设。
    """
    tree = ast.parse(INIT_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for handler in node.handlers:
            if handler.type is None or "PackageNotFoundError" not in ast.unparse(handler.type):
                continue
            for stmt in handler.body:
                if not isinstance(stmt, ast.Assign):
                    continue
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == "__version__":
                        return str(ast.literal_eval(stmt.value))
    raise AssertionError(
        "__init__.py 里没找到 PackageNotFoundError 分支给 __version__ 的兜底赋值"
        "（有意的改动请连同 tests/test_version.py 一起改）"
    )


class TestVersionFallback:
    def test_fallback_literal_matches_pyproject(self):
        """核心断言：改了 pyproject 的 version 却忘了改兜底字面量 -> 这里红。"""
        assert _literal_fallback_version() == _pyproject_version()

    def test_fallback_used_when_metadata_is_missing(self, monkeypatch):
        """兜底分支真的会被走到：把 metadata 关掉后，__version__ 就是那个兜底值。

        上一条测的是「字面量对不对」，这一条测的是「这个字面量确实是未安装时的出口」——
        两条合起来才堵得住「改了字面量但代码走的根本不是它」。
        """
        import importlib
        import importlib.metadata as metadata

        import qsmy_deepseek_locator as pkg

        def _not_installed(name):
            raise metadata.PackageNotFoundError(name)

        monkeypatch.setattr(metadata, "version", _not_installed)
        try:
            assert importlib.reload(pkg).__version__ == _pyproject_version()
        finally:
            # 还原并 reload 一次：否则包会带着「未安装」时的 __version__ 留在 sys.modules 里。
            # 这里复用的是同一个模块对象（reload 不会换对象），所以别处持有的引用不受影响。
            monkeypatch.undo()
            importlib.reload(pkg)
