from __future__ import annotations

"""`Py_BuildValue` 类型树解析、规范化与渲染测试。"""

from typing import cast

import pytest
from clang.cindex import Cursor

from core.node_visitors.c_signature_extraction.core.py_build_value_type_nodes import (
    AnyTypeNode,
    DictTypeNode,
    ListTypeNode,
    NamedTypeNode,
    TupleTypeNode,
    TypeNode,
    UnionTypeNode,
)
from core.node_visitors.c_signature_extraction.core.py_build_value_type_parser import (
    PyBuildValueTypeParser,
    PyBuildValueTypeParserError,
)


def _fake_args(count: int) -> list[Cursor]:
    """构造指定数量的假实参游标。"""
    return [cast(Cursor, object()) for _ in range(count)]


def _parse(format_string: str, arg_count: int) -> TypeNode:
    """解析格式串并返回原始类型树。"""
    return PyBuildValueTypeParser(format_string, _fake_args(arg_count)).parse()


def _canonical_render(format_string: str, arg_count: int) -> str:
    """执行 parse -> canonicalize -> render 全流程。"""
    return _parse(format_string, arg_count).canonicalize().render()


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        (AnyTypeNode(), "Any"),
        (NamedTypeNode("int"), "int"),
        (UnionTypeNode(()), "Never"),
        (UnionTypeNode((NamedTypeNode("int"),)), "int"),
        (
            TupleTypeNode((NamedTypeNode("int"), UnionTypeNode((NamedTypeNode("str"),)))),
            "tuple[int, str]",
        ),
        (ListTypeNode(UnionTypeNode((NamedTypeNode("int"),))), "list[int]"),
        (
            DictTypeNode(
                UnionTypeNode((NamedTypeNode("str"),)),
                UnionTypeNode((NamedTypeNode("int"),)),
            ),
            "dict[str, int]",
        ),
    ],
)
def test_render_type_node_returns_expected_string(node: TypeNode, expected: str) -> None:
    """显式渲染接口应按当前节点结构输出文本。"""
    assert node.render() == expected


def test_canonicalize_type_node_turns_empty_union_into_never() -> None:
    """空 union 在规范化后应保持为空 union 这一唯一 `Never` 表示。"""
    canonical = UnionTypeNode(()).canonicalize()

    assert canonical == UnionTypeNode(())


def test_union_type_node_is_empty_reports_whether_members_exist() -> None:
    """空 union 判定应只取决于成员是否为空。"""
    assert UnionTypeNode(()).is_empty() is True
    assert UnionTypeNode((NamedTypeNode("int"),)).is_empty() is False


def test_canonicalize_type_node_propagates_any_across_union_members() -> None:
    """联合类型中只要出现显式 `Any`，整体就应规范化为 `Any`。"""
    canonical = UnionTypeNode(
        (AnyTypeNode(), NamedTypeNode("int"))
    ).canonicalize()

    assert canonical == AnyTypeNode()


def test_canonicalize_type_node_drops_never_from_union_members() -> None:
    """联合类型规范化时应移除空 union 这一 `Never` 幺元成员。"""
    canonical = UnionTypeNode(
        (UnionTypeNode(()), NamedTypeNode("int"))
    ).canonicalize()

    assert canonical == NamedTypeNode("int")


def test_canonicalize_type_node_short_circuits_nested_any() -> None:
    """多层嵌套联合中出现 `Any` 时应直接短路为显式 `Any`。"""
    canonical = UnionTypeNode(
        (
            NamedTypeNode("str"),
            UnionTypeNode((NamedTypeNode("int"), AnyTypeNode())),
        )
    ).canonicalize()

    assert canonical == AnyTypeNode()


def test_canonicalize_type_node_flattens_deduplicates_and_sorts_unions() -> None:
    """规范化阶段应展平、去重并按渲染键排序联合类型成员。"""
    canonical = UnionTypeNode(
        (
            NamedTypeNode("str"),
            UnionTypeNode((NamedTypeNode("float"), NamedTypeNode("int"))),
            NamedTypeNode("bool"),
            UnionTypeNode(
                (
                    NamedTypeNode("int"),
                    UnionTypeNode((NamedTypeNode("bool"), UnionTypeNode(()))),
                )
            ),
        )
    ).canonicalize()

    assert canonical == UnionTypeNode(
        (
            NamedTypeNode("bool"),
            NamedTypeNode("float"),
            NamedTypeNode("int"),
            NamedTypeNode("str"),
        )
    )


def test_canonicalize_type_node_recurses_into_container_members() -> None:
    """规范化阶段应递归处理容器内部的联合类型。"""
    canonical = ListTypeNode(
        UnionTypeNode(
            (
                NamedTypeNode("str"),
                UnionTypeNode((NamedTypeNode("int"), NamedTypeNode("bool"))),
                NamedTypeNode("bool"),
            )
        )
    ).canonicalize()

    assert canonical == ListTypeNode(
        UnionTypeNode((NamedTypeNode("bool"), NamedTypeNode("int"), NamedTypeNode("str")))
    )


def test_canonicalize_type_node_folds_single_union_member_inside_container() -> None:
    """容器槽位在规范化后可直接持有非 union 节点。"""
    canonical = DictTypeNode(
        UnionTypeNode((NamedTypeNode("str"),)),
        UnionTypeNode((NamedTypeNode("int"),)),
    ).canonicalize()

    assert canonical == DictTypeNode(NamedTypeNode("str"), NamedTypeNode("int"))


def test_canonicalize_type_node_deduplicates_structurally_equal_container_members() -> None:
    """结构相同的容器成员在 union 里应按节点相等性去重。"""
    canonical = UnionTypeNode(
        (
            ListTypeNode(NamedTypeNode("int")),
            ListTypeNode(NamedTypeNode("int")),
        )
    ).canonicalize()

    assert canonical == ListTypeNode(NamedTypeNode("int"))


def test_canonicalize_type_node_keeps_empty_union_when_all_members_are_never() -> None:
    """嵌套 union 仅包含 `Never` 时应回到空 union。"""
    canonical = UnionTypeNode(
        (
            UnionTypeNode(()),
            UnionTypeNode((UnionTypeNode(()),)),
        )
    ).canonicalize()

    assert canonical == UnionTypeNode(())


@pytest.mark.parametrize(
    ("format_string", "arg_count", "expected"),
    [
        ("", 0, NamedTypeNode("None")),
        (" \t , : ", 0, NamedTypeNode("None")),
        ("()", 0, TupleTypeNode(())),
        ("[]", 0, ListTypeNode(UnionTypeNode(()))),
        (
            "{}",
            0,
            DictTypeNode(
                UnionTypeNode(()),
                UnionTypeNode(()),
            ),
        ),
        ("(i)", 1, TupleTypeNode((NamedTypeNode("int"),))),
        (
            "[szuU]",
            4,
            ListTypeNode(
                UnionTypeNode(
                    (
                        UnionTypeNode((NamedTypeNode("str"), NamedTypeNode("None"))),
                        UnionTypeNode((NamedTypeNode("str"), NamedTypeNode("None"))),
                        UnionTypeNode((NamedTypeNode("str"), NamedTypeNode("None"))),
                        UnionTypeNode((NamedTypeNode("str"), NamedTypeNode("None"))),
                    )
                )
            ),
        ),
        (
            "{Oi}",
            2,
            DictTypeNode(
                UnionTypeNode((AnyTypeNode(),)),
                UnionTypeNode((NamedTypeNode("int"),)),
            ),
        ),
        (
            "{Oiis}",
            4,
            DictTypeNode(
                UnionTypeNode((AnyTypeNode(), NamedTypeNode("int"))),
                UnionTypeNode(
                    (
                        NamedTypeNode("int"),
                        UnionTypeNode((NamedTypeNode("str"), NamedTypeNode("None"))),
                    )
                ),
            ),
        ),
    ],
)
def test_parse_returns_expected_raw_type_tree(
    format_string: str,
    arg_count: int,
    expected: TypeNode,
) -> None:
    """解析阶段应返回未经规范化的原始类型树。"""
    assert _parse(format_string, arg_count) == expected


@pytest.mark.parametrize(
    ("format_string", "arg_count", "expected"),
    [
        ("", 0, "None"),
        (" \t , : ", 0, "None"),
        ("()", 0, "tuple[()]"),
        ("[]", 0, "list[Any]"),
        ("{}", 0, "dict[Any, Any]"),
        ("(i)", 1, "tuple[int,]"),
        ("[szuU]", 4, "list[None | str]"),
        ("[Oi]", 2, "list[Any]"),
        ("{Oi}", 2, "dict[Any, int]"),
        ("{iO}", 2, "dict[int, Any]"),
        ("{Oiis}", 4, "dict[Any, None | int | str]"),
        ("{syUy}", 4, "dict[None | str, None | bytes]"),
        ("b", 1, "int"),
        ("f", 1, "float"),
        ("D", 1, "complex"),
        ("p", 1, "bool"),
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
        ("O", 1, "Any"),
        ("O&", 2, "Any"),
        ("S", 1, "Any"),
        ("N", 1, "Any"),
        (
            "(i, [sz], {s:i, s:[f]}, y#, O&)",
            11,
            "tuple[int, list[None | str], dict[None | str, int | list[float]], None | bytes, Any]",
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


@pytest.mark.parametrize(
    ("format_string", "arg_count", "expected"),
    [
        ("", 0, "None"),
        ("i", 1, "int"),
        ("iii", 3, "tuple[int, int, int]"),
        ("s", 1, "None | str"),
        ("y", 1, "None | bytes"),
        ("ss", 2, "tuple[None | str, None | str]"),
        ("s#", 2, "None | str"),
        ("y#", 2, "None | bytes"),
        ("()", 0, "tuple[()]"),
        ("(i)", 1, "tuple[int,]"),
        ("(ii)", 2, "tuple[int, int]"),
        ("(i,i)", 2, "tuple[int, int]"),
        ("[i,i]", 2, "list[int]"),
        ("{s:i,s:i}", 4, "dict[None | str, int]"),
        (
            "((ii)(ii)) (ii)",
            6,
            "tuple[tuple[tuple[int, int], tuple[int, int]], tuple[int, int]]",
        ),
    ],
)
def test_parse_canonicalize_render_matches_documented_py_buildvalue_examples(
    format_string: str,
    arg_count: int,
    expected: str,
) -> None:
    """文档样例在当前项目的类型推断语义下应保持稳定。"""
    assert _canonical_render(format_string, arg_count) == expected


def test_parse_value_returns_list_node_for_empty_list_format() -> None:
    """直接解析值时应为列表格式返回列表节点。"""
    parser = PyBuildValueTypeParser("[]", [])

    list_node = parser._parse_value()

    assert isinstance(list_node, ListTypeNode)
    assert isinstance(list_node.element, UnionTypeNode)
    assert list_node.element == UnionTypeNode(())


def test_parse_value_returns_dict_node_for_empty_dict_format() -> None:
    """直接解析值时应为字典格式返回字典节点。"""
    parser = PyBuildValueTypeParser("{}", [])

    dict_node = parser._parse_value()

    assert isinstance(dict_node, DictTypeNode)
    assert isinstance(dict_node.key, UnionTypeNode)
    assert isinstance(dict_node.value, UnionTypeNode)
    assert dict_node.key == UnionTypeNode(())
    assert dict_node.value == UnionTypeNode(())


def test_parse_uses_converter_cursor_for_o_ampersand_resolver() -> None:
    """`O&` 应将 converter 游标交给对象类型解析函数。"""
    converter_cursor = cast(Cursor, object())
    data_cursor = cast(Cursor, object())
    seen: list[Cursor] = []

    def resolve_object_type(cursor: Cursor) -> TypeNode:
        """记录解析器看到的游标并返回固定类型。"""
        seen.append(cursor)
        return NamedTypeNode("Converted")

    parser = PyBuildValueTypeParser(
        "O&",
        [converter_cursor, data_cursor],
        resolve_object_type_func=resolve_object_type,
    )

    assert parser.parse() == NamedTypeNode("Converted")
    assert seen == [converter_cursor]


def test_parse_uses_resolved_converter_type_in_nested_o_ampersand_structure() -> None:
    """嵌套结构里的 `O&` 也应保留 converter 解析结果到类型树中。"""
    converter_cursor = cast(Cursor, object())
    data_cursor = cast(Cursor, object())
    seen: list[Cursor] = []

    def resolve_object_type(cursor: Cursor) -> TypeNode:
        """记录解析器看到的游标并返回固定类型。"""
        seen.append(cursor)
        return NamedTypeNode("Converted")

    parser = PyBuildValueTypeParser(
        "([O&])",
        [converter_cursor, data_cursor],
        resolve_object_type_func=resolve_object_type,
    )

    assert parser.parse() == TupleTypeNode(
        (ListTypeNode(UnionTypeNode((NamedTypeNode("Converted"),))),)
    )
    assert seen == [converter_cursor]


@pytest.mark.parametrize(
    ("format_string", "arg_count"),
    [
        ("q", 0),
        ("(i", 1),
        ("{sis}", 3),
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
