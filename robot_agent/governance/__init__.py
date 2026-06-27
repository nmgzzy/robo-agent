"""治理层（设计 §8.4 / §8.6，实现计划 §P7 / §P9）。

- **记忆治理 (P7)**：`compact_namespace` / `compact_all` / `make_compaction_hook`——对 Store
  namespace 做去重、LLM 冲突消解、衰减，长跑不被脏记忆拖垮（AC-6）。
- 安全/对齐策略层 (P9) 后续在本包补充（宪章硬约束、工具权限、限流、审计）。
"""

from __future__ import annotations

from robot_agent.governance.compaction import (
    CompactionReport,
    compact_all,
    compact_namespace,
    make_compaction_hook,
)

__all__ = [
    "CompactionReport",
    "compact_all",
    "compact_namespace",
    "make_compaction_hook",
]
