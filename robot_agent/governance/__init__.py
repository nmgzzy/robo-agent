"""治理层（设计 §8.4 / §8.6，实现计划 §P7 / §P9）。

- **记忆治理 (P7)**：`compact_namespace` / `compact_all` / `make_compaction_hook`——对 Store
  namespace 做去重、LLM 冲突消解、衰减，长跑不被脏记忆拖垮（AC-6）。
- **安全/对齐策略层 (P9)**：`GovernancePolicy`（宪章硬约束 + 工具权限 + 限幅 + 限频 + 审计），
  在工具封装层动作下发前硬性校验，违反直接拒绝并记 `AuditLog`。
"""

from __future__ import annotations

from robot_agent.governance.compaction import (
    CompactionReport,
    compact_all,
    compact_namespace,
    make_compaction_hook,
)
from robot_agent.governance.policy import (
    AmplitudeLimit,
    AuditEntry,
    AuditLog,
    GovernancePolicy,
    ToolPermission,
)

__all__ = [
    "AmplitudeLimit",
    "AuditEntry",
    "AuditLog",
    "CompactionReport",
    "GovernancePolicy",
    "ToolPermission",
    "compact_all",
    "compact_namespace",
    "make_compaction_hook",
]
