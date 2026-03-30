from __future__ import annotations

"""`PyArg_ParseTuple*` 系列 parser 共用的 format unit 定义。"""

from dataclasses import dataclass

from .types import RawType, Type


def _raw(text: str, *, imports: list[str] | None = None) -> RawType:
    return RawType(text, imports=imports)


@dataclass(frozen=True)
class _FormatUnitSpec:
    """描述单个格式单元如何映射到 Python 参数。"""

    unit: str
    type: Type
    c_arg_count: int
    decl_ref_offset: int
    object_type_arg_offset: int | None = None


_FORMAT_UNIT_SPECS: tuple[_FormatUnitSpec, ...] = (
    _FormatUnitSpec("es#", _raw("str"), 3, 1),
    _FormatUnitSpec("et#", _raw("str | bytes | bytearray"), 3, 1),
    _FormatUnitSpec("s*", _raw("str | collections.abc.Buffer", imports=["collections.abc"]), 1, 0),
    _FormatUnitSpec("s#", _raw("str | collections.abc.Buffer", imports=["collections.abc"]), 2, 0),
    _FormatUnitSpec(
        "z*",
        _raw("str | collections.abc.Buffer | None", imports=["collections.abc"]),
        1,
        0,
    ),
    _FormatUnitSpec(
        "z#",
        _raw("str | collections.abc.Buffer | None", imports=["collections.abc"]),
        2,
        0,
    ),
    _FormatUnitSpec("y*", _raw("collections.abc.Buffer", imports=["collections.abc"]), 1, 0),
    _FormatUnitSpec("y#", _raw("collections.abc.Buffer", imports=["collections.abc"]), 2, 0),
    _FormatUnitSpec("es", _raw("str"), 2, 1),
    _FormatUnitSpec("et", _raw("str | bytes | bytearray"), 2, 1),
    _FormatUnitSpec("w*", _raw("collections.abc.Buffer", imports=["collections.abc"]), 1, 0),
    _FormatUnitSpec("O!", _raw("object"), 2, 1, object_type_arg_offset=0),
    _FormatUnitSpec("O&", _raw("object"), 2, 1, object_type_arg_offset=0),
    _FormatUnitSpec("s", _raw("str"), 1, 0),
    _FormatUnitSpec("z", _raw("str | None"), 1, 0),
    _FormatUnitSpec("y", _raw("collections.abc.Buffer", imports=["collections.abc"]), 1, 0),
    _FormatUnitSpec("S", _raw("bytes"), 1, 0),
    _FormatUnitSpec("Y", _raw("bytearray"), 1, 0),
    _FormatUnitSpec("U", _raw("str"), 1, 0),
    _FormatUnitSpec("b", _raw("int"), 1, 0),
    _FormatUnitSpec("B", _raw("int"), 1, 0),
    _FormatUnitSpec("h", _raw("int"), 1, 0),
    _FormatUnitSpec("H", _raw("int"), 1, 0),
    _FormatUnitSpec("i", _raw("int"), 1, 0),
    _FormatUnitSpec("I", _raw("int"), 1, 0),
    _FormatUnitSpec("l", _raw("int"), 1, 0),
    _FormatUnitSpec("k", _raw("int"), 1, 0),
    _FormatUnitSpec("L", _raw("int"), 1, 0),
    _FormatUnitSpec("K", _raw("int"), 1, 0),
    _FormatUnitSpec("n", _raw("int"), 1, 0),
    _FormatUnitSpec("c", _raw("bytes | bytearray"), 1, 0),
    _FormatUnitSpec("C", _raw("str"), 1, 0),
    _FormatUnitSpec("f", _raw("float"), 1, 0),
    _FormatUnitSpec("d", _raw("float"), 1, 0),
    _FormatUnitSpec("D", _raw("complex"), 1, 0),
    _FormatUnitSpec("O", _raw("object"), 1, 0),
    _FormatUnitSpec("p", _raw("object"), 1, 0),
)
