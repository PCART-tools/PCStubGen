from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# 提取参数在 Python 函数签名中的位置语义。
ExtractedArgumentKind = Literal[
    "positional_or_keyword",
    "keyword_only",
    "var_positional",
    "var_keyword",
]


@dataclass
class ExtractedArgument:
    """单个参数的提取结果。"""

    name: str
    type_name: str | None = None
    default_value: str | None = None
    # 与 IRArgumentKind 对齐的轻量语义标记。
    kind: ExtractedArgumentKind = "positional_or_keyword"


@dataclass
class ExtractedSignature:
    """单条函数签名（参数列表 + 返回值类型）。"""

    arguments: list[ExtractedArgument] = field(default_factory=list)
    return_type_name: str | None = None


@dataclass
class ExtractedFunction:
    """按 PyMethodDef 条目聚合的函数提取结果。"""

    py_name: str
    c_name: str
    method_flags: list[str] = field(default_factory=list)
    signatures: list[ExtractedSignature] = field(default_factory=list)
    # 记录来源，便于调试与去重。
    source_file: str | None = None
    method_table: str | None = None


@dataclass
class ExtractedModule:
    """按 PyModuleDef 聚合的模块级提取结果。"""

    name: str
    lookup_names: set[str] = field(default_factory=set)
    functions: dict[str, list[ExtractedFunction]] = field(default_factory=dict)