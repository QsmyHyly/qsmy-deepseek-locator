"""文档交叉引用自检：代码里的 @doc 必须指得到地方。

约定（与参考项目一致）：注释里引用独立文档时统一写

    @doc docs/API-NOTES.md#5-思考-token-会吃掉-max_tokens

锚点是**人类可读标签**，不是 GitHub 自动生成的 slug（自动 slug 会吃掉点号、下划线），
所以比对时按「忽略标点与大小写」的宽松口径来，只要指得到那节标题就算过。

本文件同时反查一件事：**docs/ 下不许有孤儿文档** —— 没有任何代码引用它的文档，
要么是漏了引用，要么它本该是代码里的注释。

⚠️ 只扫 `src/` 下的 .py：`@doc` 是**代码注释**的约定，README 里那句
「代码里都用 @doc docs/API-NOTES.md#<锚点> 指回对应小节」是在**介绍**这个约定，
不是一条引用 —— 扫描范围放宽到 README 时，它会当成锚点「<锚点>」误报（第一次跑就撞上了）。
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DOCS_DIR = PROJECT_ROOT / "docs"

# @doc <路径>#<锚点>；路径不含空白，锚点可含中文/连字符/下划线
_DOCREF = re.compile(r"@doc\s+([^\s#]+)#(\S+)")
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)


def _fold(text: str) -> str:
    """忽略标点、空白与大小写（只留字母数字与中日韩字符）。"""
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text.lower())


def _iter_source_files():
    """只扫包源码：@doc 引用必须出现在代码注释里（见文件头说明）。"""
    for path in sorted(SRC_DIR.rglob("*.py")):
        yield path


def _collect_refs() -> list[tuple[Path, str, str]]:
    refs = []
    for path in _iter_source_files():
        text = path.read_text(encoding="utf-8")
        for target, anchor in _DOCREF.findall(text):
            refs.append((path, target, anchor))
    return refs


class TestDocRefs:
    def test_has_refs(self):
        # 如果一条都没有，说明引用被删光了，这个自检就变成了摆设
        assert _collect_refs(), "没有找到任何 @doc 引用"

    def test_targets_exist_and_anchors_match(self):
        problems = []
        for path, target, anchor in _collect_refs():
            doc = PROJECT_ROOT / target
            if not doc.exists():
                problems.append(f"{path.name} 引用的 {target} 不存在")
                continue
            headings = [_fold(h) for h in _HEADING.findall(doc.read_text(encoding="utf-8"))]
            needle = _fold(anchor)
            if not any(h == needle or h.startswith(needle) or needle.startswith(h) for h in headings):
                problems.append(f"{path.name} 的锚点 #{anchor} 在 {target} 里找不到对应标题")
        assert not problems, "文档引用断了：\n" + "\n".join(problems)

    def test_no_orphan_docs(self):
        referenced = {target for _, target, _ in _collect_refs()}
        orphans = [
            str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
            for p in DOCS_DIR.rglob("*.md")
            if str(p.relative_to(PROJECT_ROOT)).replace("\\", "/") not in referenced
        ]
        assert not orphans, f"docs/ 下有没人引用的孤儿文档：{orphans}"
