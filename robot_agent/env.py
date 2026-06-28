"""应用层环境变量：从仓库根目录的 `.env` 加载到 `os.environ`（不覆盖已有变量）。"""

from __future__ import annotations

import os
import re
from pathlib import Path

_ENV_LOADED = False
_ENV_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _repo_root() -> Path:
    """应用层包所在目录的上一级即 monorepo 根（含 `.env.example`）。"""
    return Path(__file__).resolve().parent.parent


def find_dotenv(*, start: Path | None = None) -> Path | None:
    """查找 `.env`；默认只使用仓库根，显式 `start` 时才向上查找。"""
    if start is None:
        candidate = _repo_root() / ".env"
        return candidate if candidate.is_file() else None

    current = start.resolve()
    for directory in (current, *current.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].lstrip()
    if "=" not in stripped:
        return None
    key, _, raw_value = stripped.partition("=")
    key = key.strip()
    if not _ENV_KEY_RE.fullmatch(key):
        return None
    value = _strip_quotes(raw_value.strip())
    return key, value


def load_env(*, path: Path | str | None = None, override: bool = False) -> Path | None:
    """把 `.env` 键值注入 `os.environ`。

    - 默认只读取仓库根 `.env`；也可显式传入 `path`。
    - `override=False`（默认）：已存在于环境中的键不被 `.env` 覆盖（shell 导出优先）。
    - 文件不存在时静默返回 `None`；成功加载返回 `.env` 路径。
    - 幂等：同一进程内重复调用不会重复读盘（除非传入不同 `path`）。
    """
    global _ENV_LOADED
    dotenv_path = Path(path) if path is not None else find_dotenv()
    if dotenv_path is None or not dotenv_path.is_file():
        return None

    if _ENV_LOADED and path is None:
        return dotenv_path

    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_dotenv_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if override or key not in os.environ:
            os.environ[key] = value

    if path is None:
        _ENV_LOADED = True
    return dotenv_path


def ensure_env_loaded() -> Path | None:
    """若尚未加载则读取 `.env`；供 `robot_agent` 包导入或 LLM 工厂调用。"""
    return load_env()
