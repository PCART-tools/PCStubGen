from __future__ import annotations

"""`Py_BuildValue` parser 共用的 format unit 定义。"""

from dataclasses import dataclass

from .....type_models import AnyType, RawType, Type, UnionType


@dataclass(frozen=True)
class _FormatUnitSpec:
    """描述单个 `Py_BuildValue` 格式单元的消费方式与类型语义。"""

    unit: str
    c_arg_count: int
    value_type: Type
    object_type_arg_offset: int | None = None


_STR_OR_NONE_TYPE = UnionType((RawType.str_, RawType.none_))
_BYTES_OR_NONE_TYPE = UnionType((RawType.bytes_, RawType.none_))

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
    _FormatUnitSpec("i", 1, RawType.int_),
    _FormatUnitSpec("b", 1, RawType.int_),
    _FormatUnitSpec("h", 1, RawType.int_),
    _FormatUnitSpec("l", 1, RawType.int_),
    _FormatUnitSpec("B", 1, RawType.int_),
    _FormatUnitSpec("H", 1, RawType.int_),
    _FormatUnitSpec("I", 1, RawType.int_),
    _FormatUnitSpec("k", 1, RawType.int_),
    _FormatUnitSpec("L", 1, RawType.int_),
    _FormatUnitSpec("K", 1, RawType.int_),
    _FormatUnitSpec("n", 1, RawType.int_),
    _FormatUnitSpec("c", 1, RawType.bytes_),
    _FormatUnitSpec("C", 1, RawType.str_),
    _FormatUnitSpec("d", 1, RawType.float_),
    _FormatUnitSpec("f", 1, RawType.float_),
    _FormatUnitSpec("D", 1, RawType.complex_),
    _FormatUnitSpec("O", 1, AnyType(), object_type_arg_offset=0),
    _FormatUnitSpec("S", 1, AnyType(), object_type_arg_offset=0),
    _FormatUnitSpec("N", 1, AnyType(), object_type_arg_offset=0),
)
