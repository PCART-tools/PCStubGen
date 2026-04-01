from __future__ import annotations

"""`Py_BuildValue` 类型树解析、规范化与渲染测试。"""

from typing import cast

import pytest
from clang.cindex import Cursor

from pcstubgen.type_system import (
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


def _parse(format_string: str, arg_count: int) -> Type:
    """解析格式串并返回原始类型树。"""
    return PyBuildValueTypeParser(format_string, _fake_args(arg_count)).parse()


def _canonical_render(format_string: str, arg_count: int) -> str:
    """执行 parse -> canonicalize -> render 全流程。"""
    return _parse(format_string, arg_count).canonicalize().render()


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        (AnyType(), "typing.Any"),
        (NamedType("int"), "int"),
        (UnionType(()), "Never"),
        (UnionType((NamedType("int"),)), "int"),
        (
            TupleType((NamedType("int"), UnionType((NamedType("str"),)))),
            "tuple[int, str]",
        ),
        (ListType(UnionType((NamedType("int"),))), "list[int]"),
        (
            DictType(
                UnionType((NamedType("str"),)),
                UnionType((NamedType("int"),)),
            ),
            "dict[str, int]",
        ),
    ],
)
def test_render_type_returns_expected_string(node: Type, expected: str) -> None:
    """显式渲染接口应按当前节点结构输出文本。"""
    assert node.render() == expected


def test_canonicalize_type_turns_empty_union_into_never() -> None:
    """空 union 在规范化后应保持为空 union 这一唯一 `Never` 表示。"""
    canonical = UnionType(()).canonicalize()

    assert canonical == UnionType(())


def test_canonicalize_type_propagates_any_across_union_members() -> None:
    """联合类型中只要出现显式 `Any`，整体就应规范化为 `Any`。"""
    canonical = UnionType(
        (AnyType(), NamedType("int"))
    ).canonicalize()

    assert canonical == AnyType()


def test_canonicalize_type_drops_never_from_union_members() -> None:
    """联合类型规范化时应移除空 union 这一 `Never` 幺元成员。"""
    canonical = UnionType(
        (UnionType(()), NamedType("int"))
    ).canonicalize()

    assert canonical == NamedType("int")


def test_canonicalize_type_short_circuits_nested_any() -> None:
    """多层嵌套联合中出现 `Any` 时应直接短路为显式 `Any`。"""
    canonical = UnionType(
        (
            NamedType("str"),
            UnionType((NamedType("int"), AnyType())),
        )
    ).canonicalize()

    assert canonical == AnyType()


def test_canonicalize_type_flattens_deduplicates_and_sorts_unions() -> None:
    """规范化阶段应展平、去重并按渲染键排序联合类型成员。"""
    canonical = UnionType(
        (
            NamedType("str"),
            UnionType((NamedType("float"), NamedType("int"))),
            NamedType("bool"),
            UnionType(
                (
                    NamedType("int"),
                    UnionType((NamedType("bool"), UnionType(()))),
                )
            ),
        )
    ).canonicalize()

    assert canonical == UnionType(
        (
            NamedType("bool"),
            NamedType("float"),
            NamedType("int"),
            NamedType("str"),
        )
    )


def test_canonicalize_type_node_recurses_into_container_members() -> None:
    """规范化阶段应递归处理容器内部的联合类型。"""
    canonical = ListType(
        UnionType(
            (
                NamedType("str"),
                UnionType((NamedType("int"), NamedType("bool"))),
                NamedType("bool"),
            )
        )
    ).canonicalize()

    assert canonical == ListType(
        UnionType((NamedType("bool"), NamedType("int"), NamedType("str")))
    )


def test_canonicalize_type_node_folds_single_union_member_inside_container() -> None:
    """容器槽位在规范化后可直接持有非 union 节点。"""
    canonical = DictType(
        UnionType((NamedType("str"),)),
        UnionType((NamedType("int"),)),
    ).canonicalize()

    assert canonical == DictType(NamedType("str"), NamedType("int"))


def test_canonicalize_type_node_deduplicates_structurally_equal_container_members() -> None:
    """结构相同的容器成员在 union 里应按节点相等性去重。"""
    canonical = UnionType(
        (
            ListType(NamedType("int")),
            ListType(NamedType("int")),
        )
    ).canonicalize()

    assert canonical == ListType(NamedType("int"))


def test_canonicalize_type_node_keeps_empty_union_when_all_members_are_never() -> None:
    """嵌套 union 仅包含 `Never` 时应回到空 union。"""
    canonical = UnionType(
        (
            UnionType(()),
            UnionType((UnionType(()),)),
        )
    ).canonicalize()

    assert canonical == UnionType(())


@pytest.mark.parametrize(
    ("format_string", "arg_count", "expected"),
    [
        ("", 0, NamedType("None")),
        (" \t , : ", 0, NamedType("None")),
        ("()", 0, TupleType(())),
        ("[]", 0, ListType(UnionType(()))),
        (
            "{}",
            0,
            DictType(
                UnionType(()),
                UnionType(()),
            ),
        ),
        ("(i)", 1, TupleType((NamedType("int"),))),
        (
            "[szuU]",
            4,
            ListType(
                UnionType(
                    (
                        UnionType((NamedType("str"), NamedType("None"))),
                        UnionType((NamedType("str"), NamedType("None"))),
                        UnionType((NamedType("str"), NamedType("None"))),
                        UnionType((NamedType("str"), NamedType("None"))),
                    )
                )
            ),
        ),
        (
            "{Oi}",
            2,
            DictType(
                UnionType((AnyType(),)),
                UnionType((NamedType("int"),)),
            ),
        ),
        (
            "{Oiis}",
            4,
            DictType(
                UnionType((AnyType(), NamedType("int"))),
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
    assert _parse(format_string, arg_count) == expected


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
        ("[Oi]", 2, "list[typing.Any]"),
        ("{Oi}", 2, "dict[typing.Any, int]"),
        ("{iO}", 2, "dict[int, typing.Any]"),
        ("{Oiis}", 4, "dict[typing.Any, None | int | str]"),
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
        ("O", 1, "typing.Any"),
        ("O&", 2, "typing.Any"),
        ("S", 1, "typing.Any"),
        ("N", 1, "typing.Any"),
        (
            "(i, [sz], {s:i, s:[f]}, y#, O&)",
            11,
            "tuple[int, list[None | str], dict[None | str, int | list[float]], None | bytes, typing.Any]",
        ),
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


def test_parse_raises_with_chinese_message_for_unpaired_dictionary_format() -> None:
    with pytest.raises(
        PyBuildValueTypeParserError,
        match=r"Dictionary format 必须包含成对的 key/value。",
    ):
        _parse("{sis}", 3)


@pytest.mark.parametrize(("format_string",), [("O",), ("S",), ("N",)])
def test_parse_uses_object_slot_cursor_for_object_like_units(format_string: str) -> None:
    """`O`、`S`、`N` 应将自己的对象槽位交给类型解析函数。"""
    object_cursor = cast(Cursor, object())
    seen: list[Cursor] = []

    def resolve_object_type(cursor: Cursor) -> Type:
        """记录解析器看到的游标并返回固定类型。"""
        seen.append(cursor)
        return NamedType("Resolved")

    parser = PyBuildValueTypeParser(
        format_string,
        [object_cursor],
        resolve_object_type_func=resolve_object_type,
    )

    assert parser.parse() == NamedType("Resolved")
    assert seen == [object_cursor]


def test_parse_uses_converter_cursor_for_o_ampersand_resolver() -> None:
    """`O&` 应将 converter 游标交给对象类型解析函数。"""
    converter_cursor = cast(Cursor, object())
    data_cursor = cast(Cursor, object())
    seen: list[Cursor] = []

    def resolve_object_type(cursor: Cursor) -> Type:
        """记录解析器看到的游标并返回固定类型。"""
        seen.append(cursor)
        return NamedType("Converted")

    parser = PyBuildValueTypeParser(
        "O&",
        [converter_cursor, data_cursor],
        resolve_object_type_func=resolve_object_type,
    )

    assert parser.parse() == NamedType("Converted")
    assert seen == [converter_cursor]


def test_parse_uses_resolved_converter_type_in_nested_o_ampersand_structure() -> None:
    """嵌套结构里的 `O&` 也应保留 converter 解析结果到类型树中。"""
    converter_cursor = cast(Cursor, object())
    data_cursor = cast(Cursor, object())
    seen: list[Cursor] = []

    def resolve_object_type(cursor: Cursor) -> Type:
        """记录解析器看到的游标并返回固定类型。"""
        seen.append(cursor)
        return NamedType("Converted")

    parser = PyBuildValueTypeParser(
        "([O&])",
        [converter_cursor, data_cursor],
        resolve_object_type_func=resolve_object_type,
    )

    assert parser.parse() == TupleType(
        (ListType(UnionType((NamedType("Converted"),))),)
    )
    assert seen == [converter_cursor]


def test_collect_imports_returns_recursive_dependency_set() -> None:
    node = DictType(
        NamedType("numpy.ndarray", imports=("numpy",)),
        UnionType((AnyType(), NamedType("collections.abc.Buffer", imports=("collections.abc",)))),
    )

    assert node.collect_imports() == {"numpy", "collections.abc", "typing"}


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

