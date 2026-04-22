from __future__ import annotations

"""`PyArg_ParseTuple*` 系列 parser 共用的 format unit 定义。"""

from dataclasses import dataclass

from .....type_models import RawType, Type, UnionType


@dataclass(frozen=True)
class _FormatUnitSpec:
    """描述单个格式单元如何映射到 Python 参数。"""

    unit: str
    c_arg_count: int
    decl_ref_offset: int
    type: Type
    type_object_arg_offset: int | None = None
    converter_arg_offset: int | None = None


_BUFFER_TYPE = RawType("collections.abc.Buffer", imports=("collections.abc",))
_STR_OR_BYTES_OR_BYTEARRAY_TYPE = UnionType(
    (RawType.str_, RawType.bytes_, RawType.bytearray_)
)
_STR_OR_BUFFER_TYPE = UnionType((RawType.str_, _BUFFER_TYPE))
_STR_OR_BUFFER_OR_NONE_TYPE = UnionType((RawType.str_, _BUFFER_TYPE, RawType.none_))
_STR_OR_NONE_TYPE = UnionType((RawType.str_, RawType.none_))
_BYTES_OR_BYTEARRAY_TYPE = UnionType((RawType.bytes_, RawType.bytearray_))


_FORMAT_UNIT_SPECS: tuple[_FormatUnitSpec, ...] = (
    _FormatUnitSpec("es#", 3, 1, RawType.str_),
    _FormatUnitSpec("et#", 3, 1, _STR_OR_BYTES_OR_BYTEARRAY_TYPE),
    _FormatUnitSpec("s*", 1, 0, _STR_OR_BUFFER_TYPE),
    _FormatUnitSpec("s#", 2, 0, _STR_OR_BUFFER_TYPE),
    _FormatUnitSpec("z*", 1, 0, _STR_OR_BUFFER_OR_NONE_TYPE),
    _FormatUnitSpec("z#", 2, 0, _STR_OR_BUFFER_OR_NONE_TYPE),
    _FormatUnitSpec("y*", 1, 0, _BUFFER_TYPE),
    _FormatUnitSpec("y#", 2, 0, _BUFFER_TYPE),
    _FormatUnitSpec("es", 2, 1, RawType.str_),
    _FormatUnitSpec("et", 2, 1, _STR_OR_BYTES_OR_BYTEARRAY_TYPE),
    _FormatUnitSpec("w*", 1, 0, _BUFFER_TYPE),
    _FormatUnitSpec("O!", 2, 1, RawType.object_, type_object_arg_offset=0),
    _FormatUnitSpec("O&", 2, 1, RawType.object_, converter_arg_offset=0),
    _FormatUnitSpec("s", 1, 0, RawType.str_),
    _FormatUnitSpec("z", 1, 0, _STR_OR_NONE_TYPE),
    _FormatUnitSpec("y", 1, 0, _BUFFER_TYPE),
    _FormatUnitSpec("S", 1, 0, RawType.bytes_),
    _FormatUnitSpec("Y", 1, 0, RawType.bytearray_),
    _FormatUnitSpec("U", 1, 0, RawType.str_),
    _FormatUnitSpec("b", 1, 0, RawType.int_),
    _FormatUnitSpec("B", 1, 0, RawType.int_),
    _FormatUnitSpec("h", 1, 0, RawType.int_),
    _FormatUnitSpec("H", 1, 0, RawType.int_),
    _FormatUnitSpec("i", 1, 0, RawType.int_),
    _FormatUnitSpec("I", 1, 0, RawType.int_),
    _FormatUnitSpec("l", 1, 0, RawType.int_),
    _FormatUnitSpec("k", 1, 0, RawType.int_),
    _FormatUnitSpec("L", 1, 0, RawType.int_),
    _FormatUnitSpec("K", 1, 0, RawType.int_),
    _FormatUnitSpec("n", 1, 0, RawType.int_),
    _FormatUnitSpec("c", 1, 0, _BYTES_OR_BYTEARRAY_TYPE),
    _FormatUnitSpec("C", 1, 0, RawType.str_),
    _FormatUnitSpec("f", 1, 0, RawType.float_),
    _FormatUnitSpec("d", 1, 0, RawType.float_),
    _FormatUnitSpec("D", 1, 0, RawType.complex_),
    _FormatUnitSpec("O", 1, 0, RawType.object_),
    _FormatUnitSpec("p", 1, 0, RawType.bool_),
)
