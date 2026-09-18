#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""发版前置校验：**本地源码**与**PyPI 上同名同版本的产物**，公开 API 是否一致。

## 为什么要有它（2026-09-18 的真实事故）

`to_items` 是在 0.2.0 **发布之后**才补进库的（commit 5d059fb），版本号没跟着升。
后果是 PyPI 上的 0.2.0 与源码里的 0.2.0 **不是同一份东西**：下游
`pip install qsmy-deepseek-locator` 装到的库没有 to_items，于是
`from qsmy_deepseek_locator.parsing import to_items` 直接 ImportError，
整个下游项目连模块都 import 不了。

本机没暴露，是因为下游是按「源码在旁边、挂 PYTHONPATH」跑的 ——
**兜底把人骗过去了**，直到有人真的去验「按 README 装一遍会怎样」。

同类前科还有一次：0.1.2 发版漏改 `__init__.py` 的兜底字面量，使用者看到的
`__version__` 谎报 0.1.1。两次都是「发版时记得改」这种靠人记的约定失效。
兜底字面量那次已经被 tests/test_version.py 钉死了；**产物与源码不一致这次，
钉在这儿**。

## 它查什么

按模块逐一对齐「公开符号集合」，任一模块的符号集合或模块清单对不上就红：

- 模块清单：本地有而产物没有的模块（新文件忘了打包 / 没发版）
- 公开符号：模块里少了的名字（像 `to_items` 那种）
- 产物有而本地没有的（本地删了符号却没发版，同样是不一致）

公开符号的口径：模块有 `__all__` 就用 `__all__`；没有就取顶层
`def` / `async def` / `class` 与模块级赋值里不以 `_` 开头的名字。
**刻意不比函数体**：docstring 与格式化改动不该让发版检查变红，那是噪音。

## 用法

    python tools/check_release.py              # 本地版本 vs PyPI 同版本
    python tools/check_release.py --self-test  # 负向对照：伪造不一致，必须报红
    python tools/check_release.py --offline    # 不联网，只做本地自洽检查

退出码：0 = 一致（或该版本尚未发布，属发版前的正常状态）；1 = 不一致。**不一致就是事故**：
已经有人能 pip 装到那个版本了，它会与你的源码行为不同。

⚠️ 判定「尚未发布」是正常的：发版前本地版本总会比 PyPI 新，这时本脚本只报
「本地 X 未发布，最新已发布 Y」并列出新增符号，不算失败。真正要拦的是
**同名同版本却不一致**。
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "qsmy_deepseek_locator"
PYPROJECT = ROOT / "pyproject.toml"
PKG = "qsmy-deepseek-locator"


# --------------------------------------------------------------------- 版本
def pyproject_version() -> str:
    """读 [project] 表的 version。不用 tomllib：本库 requires-python >= 3.9。"""
    text = PYPROJECT.read_text(encoding="utf-8")
    section = re.search(r"^\[project\]\s*$", text, re.MULTILINE)
    if section is None:
        raise SystemExit("pyproject.toml 里找不到 [project] 表")
    rest = text[section.end():]
    nxt = re.search(r"^\[", rest, re.MULTILINE)
    body = rest[: nxt.start()] if nxt else rest
    m = re.search(r'^version\s*=\s*"([^"]+)"', body, re.MULTILINE)
    if m is None:
        raise SystemExit("pyproject.toml 的 [project] 里找不到 version")
    return m.group(1)


def published_versions() -> list:
    """问索引要已发布版本列表。取不到（离线 / 索引不可达）时返回空列表。

    用 pip 而不是直接打 HTTP：pip 已经知道本机的索引配置（公司镜像 / 代理），
    另起一套 HTTP 会绕过那些配置，反而在别人的机器上失灵。
    """
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "index", "versions", PKG],
            capture_output=True, text=True, timeout=90,
        )
    except Exception:
        return []
    out = (r.stdout or "") + (r.stderr or "")
    m = re.search(r"Available versions:\s*(.+)", out)
    if not m:
        return []
    return [v.strip() for v in m.group(1).split(",") if v.strip()]


# --------------------------------------------------------------------- 符号
def public_symbols(py_path: pathlib.Path) -> set:
    """取一个模块的公开符号集合。语法错误时返回空集（比抛异常更好报错定位）。"""
    try:
        tree = ast.parse(py_path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()

    for node in tree.body:  # 有 __all__ 就用它，那是作者对外的正式承诺
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            try:
                return {str(x) for x in ast.literal_eval(node.value)}
            except Exception:
                break

    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
    return {n for n in names if not n.startswith("_")}


def symbols_in(pkg_dir: pathlib.Path) -> dict:
    """{模块相对路径(点号形式) : 公开符号集合}。跳过私有模块与 __main__。"""
    out = {}
    for py in sorted(pkg_dir.rglob("*.py")):
        rel = py.relative_to(pkg_dir).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if "private" in parts or any(p.startswith("_") and p != "__init__" for p in parts):
            continue
        # 包根的 __init__.py：parts 会被削空，用 __init__ 当名字，别留个空白模块名
        mod = ".".join(parts) or "__init__"
        if mod == "__main__":
            continue
        out[mod] = public_symbols(py)
    return out


# --------------------------------------------------------------------- 产物
def download(version: str, dest: pathlib.Path) -> pathlib.Path | None:
    """把已发布产物拽下来，返回包文件路径；拿不到返回 None。"""
    for extra in (["--only-binary", ":all:"], []):  # 先试 wheel，不行再试 sdist
        r = subprocess.run(
            [sys.executable, "-m", "pip", "download", f"{PKG}=={version}",
             "--no-deps", "-d", str(dest), "--timeout", "30"] + extra,
            capture_output=True, text=True, timeout=180,
        )
        got = [p for p in dest.glob("*") if p.suffix in (".whl", ".gz")]
        if got:
            return got[0]
    return None


def extract_symbols(pkg_file: pathlib.Path, dest: pathlib.Path) -> dict:
    """从 wheel / sdist 里解出 {模块: 公开符号}，不 import 它（避免污染当前进程）。"""
    if pkg_file.name.endswith(".whl"):
        with zipfile.ZipFile(pkg_file) as z:
            for n in z.namelist():
                if n.startswith("qsmy_deepseek_locator/") and n.endswith(".py"):
                    target = dest / n
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(z.read(n))
    else:
        import tarfile
        with tarfile.open(pkg_file) as tf:
            for m in tf.getmembers():
                if "qsmy_deepseek_locator/" in m.name and m.name.endswith(".py"):
                    rel = m.name.split("qsmy_deepseek_locator/", 1)[1]
                    target = dest / "qsmy_deepseek_locator" / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    data = tf.extractfile(m)
                    if data is not None:
                        target.write_bytes(data.read())
    pkg = dest / "qsmy_deepseek_locator"
    return symbols_in(pkg) if pkg.exists() else {}


# --------------------------------------------------------------------- 比对
def diff_symbols(local: dict, remote: dict) -> list:
    """返回不一致的描述行；空列表 = 一致。"""
    problems = []
    for mod in sorted(set(local) - set(remote)):
        problems.append(f"模块只在本地有（没打进产物？）: {mod}")
    for mod in sorted(set(remote) - set(local)):
        problems.append(f"模块只在产物里有（本地删了没发版？）: {mod}")
    for mod in sorted(set(local) & set(remote)):
        gone = local[mod] - remote[mod]
        extra = remote[mod] - local[mod]
        if gone:
            problems.append(f"{mod}: 产物里缺这些公开符号 -> {', '.join(sorted(gone))}")
        if extra:
            problems.append(f"{mod}: 产物里多出这些公开符号 -> {', '.join(sorted(extra))}")
    return problems


def self_test() -> int:
    """负向对照：比对函数在「一致」和「不一致」两种输入下必须给出不同结论。

    没有这一段，这个脚本可能在两种情况下都打印「一致」而没人发现 ——
    那正是它要防的那类事故。**伪造的不一致必须真的被抓住**才说明检查有效。
    """
    print("[self-test] 先用本地源码自身做「一致」的对照")
    one = symbols_in(SRC)
    if not one:
        print("  失败：本地源码一个模块都没读到")
        return 1
    if diff_symbols(one, dict(one)):
        print("  失败：自己跟自己比竟然不一致 ->", diff_symbols(one, dict(one)))
        return 1
    print(f"  ok：{len(one)} 个模块自身比对为一致")

    print("[self-test] 再删掉一个公开符号，必须被抓出来")
    target_mod = next((m for m in sorted(one) if one[m]), None)
    if target_mod is None:
        print("  失败：找不到有公开符号的模块可供伪造")
        return 1
    victim = sorted(one[target_mod])[0]
    broken = {m: set(v) for m, v in one.items()}
    broken[target_mod] = set(one[target_mod]) - {victim}
    got = diff_symbols(broken, one)
    if not got:
        print(f"  失败：删掉 {target_mod}.{victim} 之后竟然还判为一致 —— 检查是空气")
        return 1
    if victim not in " ".join(got):
        print(f"  失败：报错了但没点名 {victim} -> {got}")
        return 1
    print(f"  ok：抓到了 -> {got[0]}")

    print("[self-test] 再伪造一个「只在产物里有」的模块，也必须抓到")
    extra_mod = dict(one)
    extra_mod["_fake_module_for_test"] = set()
    got2 = diff_symbols(one, extra_mod)
    if not got2:
        print("  失败：多出来的模块没被发现")
        return 1
    print(f"  ok：抓到了 -> {got2[0]}")

    print("self-test 全部通过：这个检查在「一致」时不误报、在「不一致」时真报")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="发版前置校验：本地源码 vs PyPI 同一版本的公开 API")
    ap.add_argument("--self-test", action="store_true", help="负向对照：伪造不一致，必须被抓住")
    ap.add_argument("--offline", action="store_true", help="不联网，只做本地自洽检查")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    version = pyproject_version()
    local = symbols_in(SRC)
    print(f"本地版本 {version}：{len(local)} 个模块，"
          f"{sum(len(v) for v in local.values())} 个公开符号")

    if args.offline:
        print("--offline：跳过与已发布产物的比对。")
        print("⚠️ 这次跑**没有**回答『发出去的那份和这份一样吗』—— 那正是本脚本存在的理由，"
              "别把这次通过当成发版放行。")
        return 0

    published = published_versions()
    if not published:
        print("拿不到已发布版本列表（离线 / 索引不可达）。")
        print("⚠️ 同上：这次跑没有校验产物一致性，别当成发版放行。")
        return 0
    print(f"索引上的已发布版本：{', '.join(published)}")

    if version not in published:
        latest = published[0]
        print()
        print(f"本地版本 {version} 尚未发布 —— 这是发版前的正常状态，不算失败。")
        print(f"（索引上最新是 {latest}）")
        print()
        print("本次改动相对已发布版本新增/变化的公开符号，供发版说明参考：")
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            f = download(latest, tmp)
            if f is None:
                print("  拿不到已发布产物，无法列出差异。")
                return 0
            remote = extract_symbols(f, tmp / "x")
            noted = False
            for line in diff_symbols(local, remote):
                print("  " + line)
                noted = True
            if not noted:
                print("  （公开 API 无差异 —— 本次是行为修复，发版说明里写明修了什么）")
        return 0

    # 同名同版本：这时**必须**逐字节等价的公开 API，否则已经有人装到了不一样的东西
    print()
    print(f"本地版本 {version} 已经发布过了，逐模块比对公开 API …")
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        f = download(version, tmp)
        if f is None:
            print("拿不到已发布产物，无法比对（不算失败，但也没验到）。")
            return 0
        remote = extract_symbols(f, tmp / "x")

    problems = diff_symbols(local, remote)
    if not problems:
        print(f"一致：索引上的 {version} 与本地源码公开 API 相同。")
        return 0

    print()
    print("!! 不一致：索引上已经有", version, "了，但它与你手上的源码不是同一份东西。")
    print("   下游 pip 装到的是那一份，行为会和你现在测的不一样。")
    print("   修法：**升一个版本号**再发（不要原地覆盖已发布版本），然后重跑本脚本。")
    print()
    for p in problems:
        print("   - " + p)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
