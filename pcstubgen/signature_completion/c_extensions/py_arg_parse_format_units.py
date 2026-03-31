from __future__ import annotations

"""`PyArg_ParseTuple*` 系列 parser 共用的 format unit 定义。"""

from dataclasses import dataclass

from ...type_system.types import RawType, Type


@dataclass(frozen=True)
class _FormatUnitSpec:
    """描述单个格式单元如何映射到 Python 参数。"""

    unit: str
    type: Type
    c_arg_count: int
    decl_ref_offset: int
    object_type_arg_offset: int | None = None


_FORMAT_UNIT_SPECS: tuple[_FormatUnitSpec, ...] = (
    _FormatUnitSpec("es#", RawType("str"), 3, 1),
    _FormatUnitSpec("et#", RawType("str | bytes | bytearray"), 3, 1),
    _FormatUnitSpec("s*", RawType("str | collections.abc.Buffer", imports=("collections.abc",)), 1, 0),
    _FormatUnitSpec("s#", RawType("str | collections.abc.Buffer", imports=("collections.abc",)), 2, 0),
    _FormatUnitSpec(
        "z*",
        RawType("str | collections.abc.Buffer | None", imports=("collections.abc",)),
        1,
        0,
    ),
    _FormatUnitSpec(
        "z#",
        RawType("str | collections.abc.Buffer | None", imports=("collections.abc",)),
        2,
        0,
    ),
    _FormatUnitSpec("y*", RawType("collections.abc.Buffer", imports=("collections.abc",)), 1, 0),
    _FormatUnitSpec("y#", RawType("collections.abc.Buffer", imports=("collections.abc",)), 2, 0),
    _FormatUnitSpec("es", RawType("str"), 2, 1),
    _FormatUnitSpec("et", RawType("str | bytes | bytearray"), 2, 1),
    _FormatUnitSpec("w*", RawType("collections.abc.Buffer", imports=("collections.abc",)), 1, 0),
    _FormatUnitSpec("O!", RawType("object"), 2, 1, object_type_arg_offset=0),
    _FormatUnitSpec("O&", RawType("object"), 2, 1, object_type_arg_offset=0),
    _FormatUnitSpec("s", RawType("str"), 1, 0),
    _FormatUnitSpec("z", RawType("str | None"), 1, 0),
    _FormatUnitSpec("y", RawType("collections.abc.Buffer", imports=("collections.abc",)), 1, 0),
    _FormatUnitSpec("S", RawType("bytes"), 1, 0),
    _FormatUnitSpec("Y", RawType("bytearray"), 1, 0),
    _FormatUnitSpec("U", RawType("str"), 1, 0),
    _FormatUnitSpec("b", RawType("int"), 1, 0),
    _FormatUnitSpec("B", RawType("int"), 1, 0),
    _FormatUnitSpec("h", RawType("int"), 1, 0),
    _FormatUnitSpec("H", RawType("int"), 1, 0),
    _FormatUnitSpec("i", RawType("int"), 1, 0),
    _FormatUnitSpec("I", RawType("int"), 1, 0),
    _FormatUnitSpec("l", RawType("int"), 1, 0),
    _FormatUnitSpec("k", RawType("int"), 1, 0),
    _FormatUnitSpec("L", RawType("int"), 1, 0),
    _FormatUnitSpec("K", RawType("int"), 1, 0),
    _FormatUnitSpec("n", RawType("int"), 1, 0),
    _FormatUnitSpec("c", RawType("bytes | bytearray"), 1, 0),
    _FormatUnitSpec("C", RawType("str"), 1, 0),
    _FormatUnitSpec("f", RawType("float"), 1, 0),
    _FormatUnitSpec("d", RawType("float"), 1, 0),
    _FormatUnitSpec("D", RawType("complex"), 1, 0),
    _FormatUnitSpec("O", RawType("object"), 1, 0),
    _FormatUnitSpec("p", RawType("object"), 1, 0),
)
