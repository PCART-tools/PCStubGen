from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ....ir import IRArgumentKind

if TYPE_CHECKING:
    from clang.cindex import Cursor


@dataclass
class ExtractedArgument:
    """单个参数的提取结果。"""

    name: str
    type_name: str | None = None
    default_value: str | None = None
    kind: IRArgumentKind = IRArgumentKind.POSITIONAL_OR_KEYWORD


@dataclass
class ExtractedSignature:
    """单条函数签名（参数列表 + 返回值类型）。"""

    arguments: list[ExtractedArgument] = field(default_factory=list)
    return_type_name: str | None = None


@dataclass
class ExtractedFunction:
    """按 PyMethodDef 条目聚合的函数提取结果。"""

    ml_name: str
    function_cursor: Cursor = field(repr=False, compare=False)
    ml_flags: int = 0
    signatures: list[ExtractedSignature] = field(default_factory=list)


@dataclass
class ExtractedModule:
    """按 PyModuleDef 聚合的模块级提取结果。"""

    name: str
    lookup_names: set[str] = field(default_factory=set)
    functions: dict[str, ExtractedFunction] = field(default_factory=dict)
