"""Default system prompt template.

We keep the prompt in Python (not a .md file under .claude/) so it can be
unit-tested and composed from config. The template leaves a slot for the
tool manifest description so the agent knows what it has without us having
to repeat each tool's JSON schema.

Design goals:
- Chinese user-facing answers (per user preference)
- Prefer tool calls over guessing (especially for KB/DB)
- Cite sources when retrieval is involved
- Be concise — one turn can easily exceed 4k tokens without discipline
"""
from __future__ import annotations

from typing import Iterable

from datamind.core.tools import ToolSpec


_SYSTEM_TEMPLATE = """你是 DataMind 智能助手。你可以访问多种工具来回答用户的问题，请根据问题类型选择合适的工具。

# 能力总览
你拥有以下几类工具:
{tool_groups}

# 工具使用原则
1. **优先调用工具**: 当问题涉及知识库、数据库、图谱、用户历史、运维/代码审查等领域时，必须先调用相应工具获取事实，再给出答案。不要凭记忆编造。
2. **循序渐进**: 先用搜索类工具（kb_search / graph_search_entities / skill_search / memory_recall）定位相关信息，再用细粒度工具（skill_get / graph_traverse / db_query_sql）拿详细内容。
3. **组合使用**: 一个问题可能需要多个工具配合。例如"给我看看 Shanghai 的员工工资情况并对比项目 Alpha 的预算"—— 需要先 db_query_nl 查员工，再 db_query_nl 查项目。
4. **安全约束**:
   - 数据库默认只读；不要请求 INSERT/UPDATE/DELETE/DDL —— 系统会拒绝。
   - 图谱的 upsert 和 memory 的 save/forget 是写操作，只在用户明确要求时使用。
5. **引用来源**: 使用 kb_search / db_query_nl / graph_traverse 后，简要说明结论来自哪个源（文件名 / 表名 / 实体路径）。

# 回答风格
- 默认用中文回答，保持简洁。
- 需要呈现结构化数据（表、路径、列表）时使用 Markdown。
- 遇到无法回答或工具返回空结果的情况，直接说"没有找到相关信息"并说明尝试了哪些工具。
"""


def _group(specs: Iterable[ToolSpec]) -> dict[str, list[ToolSpec]]:
    g: dict[str, list[ToolSpec]] = {}
    for s in specs:
        label = s.metadata.get("group", "other")
        g.setdefault(label, []).append(s)
    return g


_GROUP_LABEL = {
    "kb": "知识库 (kb_*)",
    "db": "数据库 (db_*)",
    "graph": "图谱 (graph_*)",
    "memory": "长期记忆 (memory_*)",
    "skill.knowledge": "知识型技能 (skill_*)",
    "skill.code": "通用小工具",
    "ingest": "数据导入 (kb_add_* / db_import_* / graph_add_*)",
    "other": "其他",
}


def build_system_prompt(specs: Iterable[ToolSpec]) -> str:
    grouped = _group(specs)
    lines: list[str] = []
    for label in [
        "kb", "graph", "db", "skill.knowledge", "skill.code", "memory", "ingest"
    ]:
        if label not in grouped:
            continue
        friendly = _GROUP_LABEL.get(label, label)
        names = ", ".join(s.name for s in grouped[label])
        lines.append(f"- {friendly}: {names}")
    for label, specs_list in grouped.items():
        if label in {
            "kb", "graph", "db", "skill.knowledge", "skill.code", "memory", "ingest"
        }:
            continue
        names = ", ".join(s.name for s in specs_list)
        lines.append(f"- {_GROUP_LABEL.get(label, label)}: {names}")
    tool_groups = "\n".join(lines) if lines else "(暂无可用工具)"
    return _SYSTEM_TEMPLATE.format(tool_groups=tool_groups)


__all__ = ["build_system_prompt"]
