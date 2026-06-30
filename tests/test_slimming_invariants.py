"""瘦身不变量回归测试：保护 docs/SLIMMING_NOTES.md 记录的裁剪结果不被回退。

覆盖：
- 核心 / SQLite 记忆 / prebuilt 关键符号可导入；
- langgraph 运行时元数据不再依赖 langgraph-sdk；
- 远程执行模块（pregel.remote / _remote_run_stream）已删除；
- 保留源码中不存在对 langgraph_sdk 的真实 import；
- runtime.BaseUser 是本地 runtime_checkable Protocol，行为正确；
- checkpoint-sqlite 已声明 orjson（codex 复审 P1 修复点）。
"""

from __future__ import annotations

import ast
import importlib
import importlib.metadata as md
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
KEPT_LIBS = ["langgraph", "checkpoint", "checkpoint-sqlite", "prebuilt"]


def test_core_symbols_importable():
    from langgraph.graph import END, START, StateGraph  # noqa: F401
    from langgraph.prebuilt import (  # noqa: F401
        InjectedState,
        InjectedStore,
        ToolNode,
        create_react_agent,
        tools_condition,
    )
    from langgraph.runtime import BaseUser, Runtime  # noqa: F401


def test_sqlite_memory_symbols_importable():
    from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: F401
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: F401
    from langgraph.store.sqlite import SqliteStore  # noqa: F401
    from langgraph.store.sqlite.aio import AsyncSqliteStore  # noqa: F401


def test_langgraph_does_not_require_sdk():
    reqs = md.requires("langgraph") or []
    offenders = [r for r in reqs if r.replace("_", "-").startswith("langgraph-sdk")]
    assert not offenders, f"langgraph 仍声明 langgraph-sdk 依赖: {offenders}"


def test_checkpoint_sqlite_declares_orjson():
    # store/sqlite 无条件 import orjson；依赖必须声明，否则干净安装会断（codex P1）。
    reqs = md.requires("langgraph-checkpoint-sqlite") or []
    assert any(
        r.split()[0].split(">")[0].split("=")[0].strip() == "orjson" for r in reqs
    ), f"checkpoint-sqlite 未声明 orjson 依赖: {reqs}"


def test_remote_modules_removed():
    for mod in (
        "langgraph.pregel.remote",
        "langgraph.pregel._remote_run_stream",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(mod)


def _iter_kept_source_files():
    for lib in KEPT_LIBS:
        pkg = REPO_ROOT / "libs" / lib / "langgraph"
        if pkg.exists():
            yield from pkg.rglob("*.py")


def test_no_residual_sdk_imports_in_source():
    """静态扫描：保留源码中不应存在 `import langgraph_sdk` / `from langgraph_sdk ...`。

    只看真实 import 语句（AST），不误伤 docstring/注释里的文字提及。
    """
    offenders: list[str] = []
    for path in _iter_kept_source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(a.name.split(".")[0] == "langgraph_sdk" for a in node.names):
                    offenders.append(f"{path}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root == "langgraph_sdk":
                    offenders.append(f"{path}:{node.lineno}")
    assert not offenders, f"发现残留 langgraph_sdk import: {offenders}"


def test_no_remotegraph_references_in_source():
    """保留源码不应再出现 RemoteGraph / RemoteException 标识符。"""
    offenders: list[str] = []
    for path in _iter_kept_source_files():
        text = path.read_text(encoding="utf-8")
        if "RemoteGraph" in text or "RemoteException" in text:
            offenders.append(str(path))
    assert not offenders, f"发现残留 RemoteGraph/RemoteException 引用: {offenders}"


def test_baseuser_protocol_behaviour():
    from langgraph.runtime import BaseUser

    class GoodUser:
        is_authenticated = True
        display_name = "robot-operator"
        identity = "op-1"

    class BadUser:
        identity = "op-2"  # 缺 is_authenticated / display_name

    assert isinstance(GoodUser(), BaseUser)
    assert not isinstance(BadUser(), BaseUser)


def test_baseuser_exported():
    import langgraph.runtime as rt

    assert "BaseUser" in rt.__all__
