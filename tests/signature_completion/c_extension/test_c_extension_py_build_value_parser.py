from __future__ import annotations

"""`Py_BuildValue` 类型树解析、规范化与渲染测试。"""

from typing import cast

import pytest
from clang.cindex import Cursor

from pcstubgen.type_models import (
    AnyType,
    DictType,
    ListType,
    RawType as NamedType,
    TupleType,
    Type,
    UnionType,
)
from pcstubgen.signature_completion.c_extension.signatures.py_build_value.parser import (
    PyBuildValueTypeParser,
    PyBuildValueTypeParserError,
)


def _fake_args(count: int) -> list[Cursor]:
    """构造指定数量的假实参游标。"""
    return [cast(Cursor, object()) for _ in range(count)]


def _parse(
    format_string: str,
    arg_count: int,
    *,
    infer_object_type_func=None,
) -> Type:
    """解析格式串并返回原始类型树。"""
    return PyBuildValueTypeParser(
        format_string,
        _fake_args(arg_count),
        infer_object_type_func=infer_object_type_func or (lambda cursor: NamedType("Resolved")),
    ).parse()


def _canonical_render(
    format_string: str,
    arg_count: int,
    *,
    infer_object_type_func=None,
) -> str:
    """执行 parse -> canonicalize -> render 全流程。"""
    return _parse(
        format_string,
        arg_count,
        infer_object_type_func=infer_object_type_func,
    ).canonicalize().render()


@pytest.mark.parametrize(
    ("format_string", "arg_count", "expected"),
    [
        ("", 0, NamedType("None")),
        ("()", 0, TupleType(())),
        ("(i)", 1, TupleType((NamedType("int"),))),
        (
            "[Oi]",
            2,
            ListType(UnionType((NamedType("Resolved"), NamedType("int")))),
        ),
        (
            "{Oiis}",
            4,
            DictType(
                UnionType((NamedType("Resolved"), NamedType("int"))),
                UnionType(
                    (
                        NamedType("int"),
                        UnionType((NamedType("str"), NamedType("None"))),
                    )
                ),
            ),
        ),
    ],
)
def test_parse_returns_expected_raw_type_tree(
    format_string: str,
    arg_count: int,
    expected: Type,
) -> None:
    """解析阶段应返回未经规范化的原始类型树。"""
    assert _parse(
        format_string,
        arg_count,
        infer_object_type_func=lambda cursor: NamedType("Resolved"),
    ) == expected


@pytest.mark.parametrize(
    ("format_string", "arg_count", "expected"),
    [
        ("", 0, "None"),
        (" \t , : ", 0, "None"),
        ("()", 0, "tuple[()]"),
        ("[]", 0, "list[typing.Any]"),
        ("{}", 0, "dict[typing.Any, typing.Any]"),
        ("(i)", 1, "tuple[int,]"),
        ("[szuU]", 4, "list[None | str]"),
        ("{syUy}", 4, "dict[None | str, None | bytes]"),
        ("i", 1, "int"),
        ("b", 1, "int"),
        ("h", 1, "int"),
        ("l", 1, "int"),
        ("B", 1, "int"),
        ("H", 1, "int"),
        ("I", 1, "int"),
        ("k", 1, "int"),
        ("L", 1, "int"),
        ("K", 1, "int"),
        ("n", 1, "int"),
        ("d", 1, "float"),
        ("f", 1, "float"),
        ("D", 1, "complex"),
        ("C", 1, "str"),
        ("s", 1, "None | str"),
        ("s#", 2, "None | str"),
        ("z", 1, "None | str"),
        ("z#", 2, "None | str"),
        ("u", 1, "None | str"),
        ("u#", 2, "None | str"),
        ("U", 1, "None | str"),
        ("U#", 2, "None | str"),
        ("y", 1, "None | bytes"),
        ("y#", 2, "None | bytes"),
        ("c", 1, "bytes"),
        (
            "([i{sz}](s#y#){isfs})",
            11,
            "tuple[list[dict[None | str, None | str] | int], tuple[None | str, None | bytes], dict[float | int, None | str]]",
        ),
    ],
)
def test_parse_canonicalize_render_returns_expected_type_string(
    format_string: str,
    arg_count: int,
    expected: str,
) -> None:
    """完整流程应保持既有对外字符串行为。"""
    assert _canonical_render(format_string, arg_count) == expected


@pytest.mark.parametrize(
    ("format_string", "arg_count", "expected"),
    [
        ("O", 1, "Resolved"),
        ("O&", 2, "Resolved"),
        ("S", 1, "Resolved"),
        ("N", 1, "Resolved"),
        ("[Oi]", 2, "list[Resolved | int]"),
        ("{Oi}", 2, "dict[Resolved, int]"),
        ("{iO}", 2, "dict[int, Resolved]"),
        ("{Oiis}", 4, "dict[Resolved | int, None | int | str]"),
        (
            "(i, [sz], {s:i, s:[f]}, y#, O&)",
            11,
            "tuple[int, list[None | str], dict[None | str, int | list[float]], None | bytes, Resolved]",
        ),
    ],
)
def test_parse_canonicalize_render_returns_expected_type_string_for_resolved_object_units(
    format_string: str,
    arg_count: int,
    expected: str,
) -> None:
    assert _canonical_render(
        format_string,
        arg_count,
        infer_object_type_func=lambda cursor: NamedType("Resolved"),
    ) == expected


def test_parse_raises_with_chinese_message_for_unpaired_dictionary_format() -> None:
    with pytest.raises(PyBuildValueTypeParserError, match="key/value"):
        _parse("{sis}", 3)


@pytest.mark.parametrize(("format_string",), [("O",), ("S",), ("N",)])
def test_parse_uses_object_slot_cursor_for_object_like_units(format_string: str) -> None:
    """`O`、`S`、`N` 应将自己的对象槽位交给类型解析函数。"""
    object_cursor = cast(Cursor, object())
    seen: list[Cursor] = []

    def infer_object_type(cursor: Cursor) -> Type:
        """记录解析器看到的游标并返回固定类型。"""
        seen.append(cursor)
        return NamedType("Resolved")

    parser = PyBuildValueTypeParser(
        format_string,
        [object_cursor],
        infer_object_type_func=infer_object_type,
    )

    assert parser.parse() == NamedType("Resolved")
    assert seen == [object_cursor]


def test_parse_uses_converter_cursor_for_o_ampersand_resolver() -> None:
    """`O&` 应将 converter 游标交给对象类型解析函数。"""
    converter_cursor = cast(Cursor, object())
    data_cursor = cast(Cursor, object())
    seen: list[Cursor] = []

    def infer_object_type(cursor: Cursor) -> Type:
        """记录解析器看到的游标并返回固定类型。"""
        seen.append(cursor)
        return NamedType("Converted")

    parser = PyBuildValueTypeParser(
        "O&",
        [converter_cursor, data_cursor],
        infer_object_type_func=infer_object_type,
    )

    assert parser.parse() == NamedType("Converted")
    assert seen == [converter_cursor]


def test_parse_uses_resolved_converter_type_in_nested_o_ampersand_structure() -> None:
    """嵌套结构里的 `O&` 也应保留 converter 解析结果到类型树中。"""
    converter_cursor = cast(Cursor, object())
    data_cursor = cast(Cursor, object())
    seen: list[Cursor] = []

    def infer_object_type(cursor: Cursor) -> Type:
        """记录解析器看到的游标并返回固定类型。"""
        seen.append(cursor)
        return NamedType("Converted")

    parser = PyBuildValueTypeParser(
        "([O&])",
        [converter_cursor, data_cursor],
        infer_object_type_func=infer_object_type,
    )

    assert parser.parse() == TupleType(
        (ListType(UnionType((NamedType("Converted"),))),)
    )
    assert seen == [converter_cursor]


@pytest.mark.parametrize(("format_string", "arg_count"), [("O", 1), ("O&", 2), ("S", 1), ("N", 1)])
def test_parse_falls_back_to_any_when_object_type_inference_raises(
    format_string: str,
    arg_count: int,
) -> None:
    assert _parse(
        format_string,
        arg_count,
        infer_object_type_func=lambda cursor: (_ for _ in ()).throw(RuntimeError("boom")),
    ) == AnyType()


@pytest.mark.parametrize(
    ("format_string", "arg_count"),
    [
        ("q", 0),
        ("(i", 1),
        ("{sis}", 3),
        ("p", 1),
        ("s#", 1),
        ("i", 2),
        ("i#", 2),
        ("s&", 2),
        ("S&", 2),
        ("N&", 2),
    ],
)
def test_parse_raises_for_invalid_format(format_string: str, arg_count: int) -> None:
    """非法格式串应在解析阶段直接抛错。"""
    with pytest.raises(PyBuildValueTypeParserError):
        _parse(format_string, arg_count)
