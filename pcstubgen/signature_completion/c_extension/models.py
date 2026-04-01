from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ...ir import IRArgumentKind
from ...type_system.types import Type

if TYPE_CHECKING:
    from clang.cindex import Cursor


@dataclass
class CArgument:
    """
    单个参数的提取结果。

    `has_default` 用于区分“必选参数”和“存在默认值但默认值文本未知”：
    - `has_default=False` 时，参数没有默认值，`default_value` 应为 `None`
    - `has_default=True` 且 `default_value is None` 时，表示存在默认值但未解析出文本
    - `has_default=True` 且 `default_value` 为字符串时，表示已解析出默认值文本
    """

    name: str
    type: Type | None = None
    default_value: str | None = None
    has_default: bool = False
    kind: IRArgumentKind = IRArgumentKind.POSITIONAL_OR_KEYWORD


@dataclass
class CSignature:
    """单条函数签名（参数列表 + 返回值类型）。"""

    arguments: list[CArgument] = field(default_factory=list)
    return_type: Type | None = None


@dataclass
class CFunction:
    """按 PyMethodDef 条目聚合的函数提取结果。"""

    ml_name: str
    function_cursor: Cursor = field(repr=False, compare=False)
    ml_flags: int = 0
    signatures: list[CSignature] = field(default_factory=list)


@dataclass
class CModule:
    """按 PyModuleDef 聚合的模块级提取结果。"""

    name: str
    functions: dict[str, CFunction] = field(default_factory=dict)
