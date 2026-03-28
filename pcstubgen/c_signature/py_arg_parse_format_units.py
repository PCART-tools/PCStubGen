from __future__ import annotations

"""`PyArg_ParseTuple*` 系列 parser 共用的 format unit 定义。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class _FormatUnitSpec:
    """描述单个格式单元如何映射到 Python 参数。"""

    unit: str
    type_name: str
    c_arg_count: int
    decl_ref_offset: int
    object_type_arg_offset: int | None = None


_FORMAT_UNIT_SPECS: tuple[_FormatUnitSpec, ...] = (
    _FormatUnitSpec("es#", "str", 3, 1),
    _FormatUnitSpec("et#", "str | bytes | bytearray", 3, 1),
    _FormatUnitSpec("s*", "str | collections.abc.Buffer", 1, 0),
    _FormatUnitSpec("s#", "str | collections.abc.Buffer", 2, 0),
    _FormatUnitSpec("z*", "str | collections.abc.Buffer | None", 1, 0),
    _FormatUnitSpec("z#", "str | collections.abc.Buffer | None", 2, 0),
    _FormatUnitSpec("y*", "collections.abc.Buffer", 1, 0),
    _FormatUnitSpec("y#", "collections.abc.Buffer", 2, 0),
    _FormatUnitSpec("es", "str", 2, 1),
    _FormatUnitSpec("et", "str | bytes | bytearray", 2, 1),
    _FormatUnitSpec("w*", "collections.abc.Buffer", 1, 0),
    _FormatUnitSpec("O!", "object", 2, 1, object_type_arg_offset=0),
    _FormatUnitSpec("O&", "object", 2, 1, object_type_arg_offset=0),
    _FormatUnitSpec("s", "str", 1, 0),
    _FormatUnitSpec("z", "str | None", 1, 0),
    _FormatUnitSpec("y", "collections.abc.Buffer", 1, 0),
    _FormatUnitSpec("S", "bytes", 1, 0),
    _FormatUnitSpec("Y", "bytearray", 1, 0),
    _FormatUnitSpec("U", "str", 1, 0),
    _FormatUnitSpec("b", "int", 1, 0),
    _FormatUnitSpec("B", "int", 1, 0),
    _FormatUnitSpec("h", "int", 1, 0),
    _FormatUnitSpec("H", "int", 1, 0),
    _FormatUnitSpec("i", "int", 1, 0),
    _FormatUnitSpec("I", "int", 1, 0),
    _FormatUnitSpec("l", "int", 1, 0),
    _FormatUnitSpec("k", "int", 1, 0),
    _FormatUnitSpec("L", "int", 1, 0),
    _FormatUnitSpec("K", "int", 1, 0),
    _FormatUnitSpec("n", "int", 1, 0),
    _FormatUnitSpec("c", "bytes | bytearray", 1, 0),
    _FormatUnitSpec("C", "str", 1, 0),
    _FormatUnitSpec("f", "float", 1, 0),
    _FormatUnitSpec("d", "float", 1, 0),
    _FormatUnitSpec("D", "complex", 1, 0),
    _FormatUnitSpec("O", "object", 1, 0),
    _FormatUnitSpec("p", "object", 1, 0),
)
