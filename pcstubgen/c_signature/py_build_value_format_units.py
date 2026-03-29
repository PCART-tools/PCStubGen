from __future__ import annotations

"""`Py_BuildValue` parser 共用的 format unit 定义。"""

from dataclasses import dataclass

from .types import AnyType, NamedType, Type, UnionType


@dataclass(frozen=True)
class _FormatUnitSpec:
    """描述单个 `Py_BuildValue` 格式单元的消费方式与类型语义。"""

    unit: str
    c_arg_count: int
    value_type: Type
    object_type_arg_offset: int | None = None


_STR_OR_NONE_TYPE = UnionType((NamedType("str"), NamedType("None")))
_BYTES_OR_NONE_TYPE = UnionType((NamedType("bytes"), NamedType("None")))

_FORMAT_UNIT_SPECS: tuple[_FormatUnitSpec, ...] = (
    _FormatUnitSpec("s#", 2, _STR_OR_NONE_TYPE),
    _FormatUnitSpec("y#", 2, _BYTES_OR_NONE_TYPE),
    _FormatUnitSpec("z#", 2, _STR_OR_NONE_TYPE),
    _FormatUnitSpec("u#", 2, _STR_OR_NONE_TYPE),
    _FormatUnitSpec("U#", 2, _STR_OR_NONE_TYPE),
    _FormatUnitSpec("O&", 2, AnyType(), object_type_arg_offset=0),
    _FormatUnitSpec("s", 1, _STR_OR_NONE_TYPE),
    _FormatUnitSpec("y", 1, _BYTES_OR_NONE_TYPE),
    _FormatUnitSpec("z", 1, _STR_OR_NONE_TYPE),
    _FormatUnitSpec("u", 1, _STR_OR_NONE_TYPE),
    _FormatUnitSpec("U", 1, _STR_OR_NONE_TYPE),
    _FormatUnitSpec("i", 1, NamedType("int")),
    _FormatUnitSpec("b", 1, NamedType("int")),
    _FormatUnitSpec("h", 1, NamedType("int")),
    _FormatUnitSpec("l", 1, NamedType("int")),
    _FormatUnitSpec("B", 1, NamedType("int")),
    _FormatUnitSpec("H", 1, NamedType("int")),
    _FormatUnitSpec("I", 1, NamedType("int")),
    _FormatUnitSpec("k", 1, NamedType("int")),
    _FormatUnitSpec("L", 1, NamedType("int")),
    _FormatUnitSpec("K", 1, NamedType("int")),
    _FormatUnitSpec("n", 1, NamedType("int")),
    _FormatUnitSpec("c", 1, NamedType("bytes")),
    _FormatUnitSpec("C", 1, NamedType("str")),
    _FormatUnitSpec("d", 1, NamedType("float")),
    _FormatUnitSpec("f", 1, NamedType("float")),
    _FormatUnitSpec("D", 1, NamedType("complex")),
    _FormatUnitSpec("O", 1, AnyType(), object_type_arg_offset=0),
    _FormatUnitSpec("S", 1, AnyType(), object_type_arg_offset=0),
    _FormatUnitSpec("N", 1, AnyType(), object_type_arg_offset=0),
)
