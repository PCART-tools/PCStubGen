from __future__ import annotations

from pcstubgen.signature_completion.c_extension.signatures.return_type_maps import FUNCTION_NAME_TO_TYPE
from pcstubgen.types import RawType, UnionType
from tests._c_extension_test_support import *


def test_function_name_to_type_uses_type_instances() -> None:
    assert FUNCTION_NAME_TO_TYPE["PyLong_FromLong"] == RawType("int")


@pytest.mark.parametrize(
    ("token_name", "expected"),
    [
        ("_Py_NoneStruct", RawType("None")),
        ("_Py_TrueStruct", RawType("bool")),
        ("_Py_FalseStruct", RawType("bool")),
    ],
)
def test_infer_expr_type_detects_addressed_object_returns(token_name: str, expected: RawType) -> None:
    inferred = signature_rules_module.infer_expr_type(_address_of(token_name))

    assert inferred == expected


@pytest.mark.parametrize(
    "token_name",
    [
        "Py_RETURN_NONE",
        "Py_RETURN_TRUE",
        "Py_RETURN_FALSE",
        "Py_RETURN_NAN",
        "Py_RETURN_INF",
    ],
)
def test_infer_expr_type_returns_none_for_preserved_macro_tokens(token_name: str) -> None:
    macro_expr = _macro_expr(token_name)

    inferred = signature_rules_module.infer_expr_type(macro_expr)

    assert inferred is None


def test_infer_expr_type_returns_none_when_macro_name_is_not_exposed_by_ast() -> None:
    macro_expr = _FakeNode(
        kind=clang.cindex.CursorKind.UNEXPOSED_EXPR,
        children=[_FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR)],
    )

    inferred = signature_rules_module.infer_expr_type(macro_expr)

    assert inferred is None


@pytest.mark.parametrize(
    ("call_name", "expected"),
    [
        ("PyBool_FromLong", RawType("bool")),
        ("PyLong_FromLong", RawType("int")),
        ("PyFloat_FromDouble", RawType("float")),
        ("PyComplex_FromDoubles", RawType("complex")),
        ("PyUnicode_FromString", RawType("str")),
        ("PyUnicode_AsUTF8String", RawType("bytes")),
        ("PyByteArray_FromObject", RawType("bytearray")),
        ("PySlice_New", RawType("slice")),
        ("PyMemoryView_FromObject", RawType("memoryview")),
        ("PyTuple_New", RawType("tuple")),
        ("PyList_New", RawType("list")),
        ("PyDict_New", RawType("dict")),
        ("PySet_New", RawType("set")),
        ("PyFrozenSet_New", RawType("frozenset")),
        ("PyList_AsTuple", RawType("tuple")),
        ("PyDict_Items", RawType("list")),
    ],
)
def test_infer_expr_type_detects_exact_factory_mappings(call_name: str, expected: RawType) -> None:
    inferred = signature_rules_module.infer_expr_type(
        _call_expr(call_name, _identifier_node("arg"))
    )

    assert inferred == expected


def test_infer_expr_type_parses_py_buildvalue() -> None:
    inferred = signature_rules_module.infer_expr_type(
        _call_expr(
            "Py_BuildValue",
            _string_literal("(is)"),
            _identifier_node("count"),
            _identifier_node("name"),
        )
    )

    assert inferred == TupleType(
        (
            RawType("int"),
            UnionType((RawType("None"), RawType("str"))),
        )
    )


def test_infer_expr_type_canonicalizes_py_buildvalue_container_unions() -> None:
    """`Py_BuildValue` 推断结果应在渲染前先规范化容器内部的联合类型。"""
    inferred = signature_rules_module.infer_expr_type(
        _call_expr(
            "Py_BuildValue",
            _string_literal("[si]"),
            _identifier_node("name"),
            _identifier_node("count"),
        )
    )

    assert inferred == ListType(
        UnionType(
            (
                RawType("None"),
                RawType("int"),
                RawType("str"),
            )
        )
    )


def test_infer_expr_type_resolves_py_buildvalue_object_slots() -> None:
    inferred = signature_rules_module.infer_expr_type(
        _call_expr(
            "Py_BuildValue",
            _string_literal("(O)"),
            _call_expr("PyLong_FromLong", _identifier_node("value")),
        )
    )

    assert inferred == TupleType((RawType("int"),))


def test_infer_expr_type_keeps_py_buildvalue_object_slots_as_any_when_unknown() -> None:
    inferred = signature_rules_module.infer_expr_type(
        _call_expr(
            "Py_BuildValue",
            _string_literal("(O)"),
            _call_expr("CustomFactory", _identifier_node("value")),
        )
    )

    assert inferred == TupleType((AnyType(),))


def test_infer_expr_type_keeps_py_buildvalue_o_ampersand_as_any_when_converter_unknown() -> None:
    inferred = signature_rules_module.infer_expr_type(
        _call_expr(
            "Py_BuildValue",
            _string_literal("(O&)"),
            _identifier_node("converter"),
            _identifier_node("value"),
        )
    )

    assert inferred == TupleType((AnyType(),))


def test_infer_expr_type_unwraps_transparent_wrappers_and_casts() -> None:
    wrapped_expr = _wrap(
        clang.cindex.CursorKind.UNEXPOSED_EXPR,
        _wrap(
            clang.cindex.CursorKind.PAREN_EXPR,
            _FakeNode(
                kind=clang.cindex.CursorKind.CSTYLE_CAST_EXPR,
                children=[
                    _identifier_node("PyObject"),
                    _call_expr("PyUnicode_AsUTF8String", _identifier_node("value")),
                ],
            ),
        ),
    )

    inferred = signature_rules_module.infer_expr_type(wrapped_expr)

    assert inferred == RawType("bytes")


def test_infer_expr_type_returns_none_when_conditional_branches_are_unknown() -> None:
    inferred = signature_rules_module.infer_expr_type(
        _conditional_expr(
            _macro_expr("Py_RETURN_NONE"),
            _call_expr("CustomFactory", _identifier_node("left")),
            _identifier_node("UnknownToken"),
        )
    )

    assert inferred is None


def test_infer_expr_type_returns_none_for_unsupported_expr() -> None:
    inferred = signature_rules_module.infer_expr_type(
        _call_expr("CustomFactory", _identifier_node("value"))
    )

    assert inferred is None


@pytest.mark.parametrize(
    ("token_name", "expected"),
    [
        ("_Py_NoneStruct", "None"),
        ("_Py_TrueStruct", "bool"),
        ("_Py_FalseStruct", "bool"),
    ],
)
def test_return_type_detects_addressed_object_returns(token_name: str, expected: str) -> None:
    cursor = _fake_function_cursor_with_children(_return_stmt(_address_of(token_name)))

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred is not None
    assert inferred.render() == expected


@pytest.mark.parametrize(
    "token_name",
    [
        "Py_RETURN_NONE",
        "Py_RETURN_TRUE",
        "Py_RETURN_FALSE",
        "Py_RETURN_NAN",
        "Py_RETURN_INF",
    ],
)
def test_return_type_returns_none_for_preserved_macro_tokens(token_name: str) -> None:
    macro_expr = _macro_expr(token_name)
    cursor = _fake_function_cursor_with_children(_return_stmt(macro_expr))

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred is None


@pytest.mark.parametrize(
    ("call_name", "expected"),
    [
        ("PyBool_FromLong", "bool"),
        ("PyLong_FromLong", "int"),
        ("PyFloat_FromDouble", "float"),
        ("PyComplex_FromDoubles", "complex"),
        ("PyUnicode_FromString", "str"),
        ("PyUnicode_AsUTF8String", "bytes"),
        ("PyByteArray_FromObject", "bytearray"),
        ("PySlice_New", "slice"),
        ("PyMemoryView_FromObject", "memoryview"),
        ("PyTuple_New", "tuple"),
        ("PyList_New", "list"),
        ("PyDict_New", "dict"),
        ("PySet_New", "set"),
        ("PyFrozenSet_New", "frozenset"),
        ("PyList_AsTuple", "tuple"),
        ("PyDict_Items", "list"),
    ],
)
def test_return_type_detects_exact_factory_mappings(call_name: str, expected: str) -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr(call_name, _identifier_node("arg")))
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred is not None
    assert inferred.render() == expected


def test_return_type_parses_py_buildvalue() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(
            _call_expr(
                "Py_BuildValue",
                _string_literal("(is)"),
                _identifier_node("count"),
                _identifier_node("name"),
            )
        )
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred is not None
    assert inferred.render() == "tuple[int, None | str]"


def test_return_type_unwraps_transparent_wrappers_and_casts() -> None:
    wrapped_expr = _wrap(
        clang.cindex.CursorKind.UNEXPOSED_EXPR,
        _wrap(
            clang.cindex.CursorKind.PAREN_EXPR,
            _FakeNode(
                kind=clang.cindex.CursorKind.CSTYLE_CAST_EXPR,
                children=[
                    _identifier_node("PyObject"),
                    _call_expr("PyUnicode_AsUTF8String", _identifier_node("value")),
                ],
            ),
        ),
    )
    cursor = _fake_function_cursor_with_children(_return_stmt(wrapped_expr))

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred is not None
    assert inferred.render() == "bytes"


def test_return_type_deduplicates_and_canonicalizes_order() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_address_of("_Py_NoneStruct")),
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value"))),
        _return_stmt(
            _call_expr(
                "Py_BuildValue",
                _string_literal("i"),
                _identifier_node("value"),
            )
        ),
        _return_stmt(_address_of("_Py_FalseStruct")),
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred is not None
    assert inferred.render() == "None | bool | int"


def test_return_type_detects_conditional_expr() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(
            _conditional_expr(
                _identifier_node("cond"),
                _call_expr("PyLong_FromLong", _identifier_node("value")),
                _call_expr("PyFloat_FromDouble", _identifier_node("value")),
            )
        )
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred is not None
    assert inferred.render() == "float | int"


def test_return_type_deduplicates_members_already_present_in_union_return() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(
            _conditional_expr(
                _identifier_node("cond"),
                _call_expr("PyLong_FromLong", _identifier_node("value")),
                _call_expr("PyFloat_FromDouble", _identifier_node("value")),
            )
        ),
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value"))),
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred is not None
    assert inferred.render() == "float | int"


def test_return_type_returns_none_for_unsupported_returns() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("CustomFactory", _identifier_node("value"))),
        _return_stmt(),
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred is None


def test_return_type_returns_none_when_function_has_no_return() -> None:
    cursor = _fake_function_cursor_with_children()

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred is None


def test_infer_argument_lists_parses_pyarg_parsetuple() -> None:
    count_decl = _var_decl("count", _int_literal("1"))
    label_decl = _var_decl("label", _identifier_node("Py_None"))
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("i|z"),
            _address_of("count", referenced=count_decl),
            _address_of("label", referenced=label_decl),
        )
    )

    inferred = signature_rules_module.infer_argument_lists(cursor)

    assert inferred == [
        [
            _arg("count", "int"),
            _arg(
                "label",
                UnionType((RawType("str"), RawType("None"))),
                default_value="None",
                has_default=True,
            ),
        ]
    ]


def test_resolve_object_type_for_pyarg_reads_name_from_extent_source_text(tmp_path: Path) -> None:
    source = tmp_path / "object_type_from_extent.c"
    source.write_text(
        "\n".join(
            [
                "/* 中文注释 */",
                "PyArg_ParseTuple(args, \"O!\", (&PyUnicode_Type), &value);",
            ]
        ),
        encoding="utf-8",
    )
    cursor = _FakeNode(
        kind=clang.cindex.CursorKind.UNARY_OPERATOR,
        extent=_extent_for_source_snippet(source, "(&PyUnicode_Type)"),
    )

    inferred = signature_rules_module._resolve_object_type_for_pyarg(cursor)

    assert inferred is not None
    assert inferred.render() == "str"


def test_infer_argument_lists_keeps_object_fallback_for_unknown_o_bang_type(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unknown_object_type_from_extent.c"
    source.write_text(
        "\n".join(
            [
                "/* 中文注释 */",
                "PyArg_ParseTuple(args, \"O!\", (&UnknownRuntimeType), &value);",
            ]
        ),
        encoding="utf-8",
    )
    value_decl = _var_decl("value")
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("O!"),
            _FakeNode(
                kind=clang.cindex.CursorKind.UNARY_OPERATOR,
                extent=_extent_for_source_snippet(source, "(&UnknownRuntimeType)"),
            ),
            _address_of("value", referenced=value_decl),
        )
    )

    inferred = signature_rules_module.infer_argument_lists(cursor)

    assert inferred == [[_arg("value", "object")]]


def test_infer_argument_lists_joins_decl_ref_names_for_tuple_arguments() -> None:
    left_decl = _var_decl("left")
    right_decl = _var_decl("right")
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("(ii)"),
            _address_of("left", referenced=left_decl),
            _address_of("right", referenced=right_decl),
        )
    )

    inferred = signature_rules_module.infer_argument_lists(cursor)

    assert inferred == [[_arg("left_right", "tuple[int, int]")]]


def test_infer_argument_lists_returns_empty_when_no_supported_pyarg_calls_exist() -> None:
    cursor = _fake_function_cursor_with_children(
        _call_expr("PyLong_FromLong", _identifier_node("value"))
    )

    inferred = signature_rules_module.infer_argument_lists(cursor)

    assert inferred == []


def test_infer_argument_lists_parses_pyarg_parsetuple_sizet_alias() -> None:
    value_decl = _var_decl("value", _int_literal("0"))
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "_PyArg_ParseTuple_SizeT",
            _identifier_node("args"),
            _string_literal("i"),
            _address_of("value", referenced=value_decl),
        )
    )

    inferred = signature_rules_module.infer_argument_lists(cursor)

    assert inferred == [[_arg("value", "int")]]


def test_infer_argument_lists_parses_pyarg_parsetuple_and_keywords_sizet_alias() -> None:
    kwlist_decl = _var_decl("kwlist", _init_list(_string_literal("x"), _null_ptr_literal()))
    x_decl = _var_decl("x", _float_literal("0.0"))
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "_PyArg_ParseTupleAndKeywords_SizeT",
            _identifier_node("args"),
            _identifier_node("kwds"),
            _string_literal("|d"),
            _token_identifier_node("kwlist", referenced=kwlist_decl),
            _address_of("x", referenced=x_decl),
        )
    )

    inferred = signature_rules_module.infer_argument_lists(cursor)

    assert inferred == [[_arg("x", "float", has_default=True)]]


def test_infer_argument_lists_raises_for_parse_tuple_and_keywords_without_valid_kwlist() -> None:
    invalid_kwlist_decl = _var_decl("kwlist", _init_list(_identifier_node("bad"), _null_ptr_literal()))
    x_decl = _var_decl("x", _float_literal("0.0"))
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTupleAndKeywords",
            _identifier_node("args"),
            _identifier_node("kwds"),
            _string_literal("|d"),
            _token_identifier_node("kwlist", referenced=invalid_kwlist_decl),
            _address_of("x", referenced=x_decl),
        )
    )

    with pytest.raises(RuntimeError, match="kwlist"):
        signature_rules_module.infer_argument_lists(cursor)


def test_infer_argument_lists_raises_when_argument_name_cannot_be_resolved() -> None:
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("i"),
            _int_literal("0"),
        )
    )

    with pytest.raises(RuntimeError, match="C 参数槽位"):
        signature_rules_module.infer_argument_lists(cursor)


def test_infer_argument_lists_raises_when_parse_tuple_format_string_is_not_literal() -> None:
    value_decl = _var_decl("value")
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _identifier_node("fmt"),
            _address_of("value", referenced=value_decl),
        )
    )

    with pytest.raises(RuntimeError, match="format string"):
        signature_rules_module.infer_argument_lists(cursor)


def test_infer_argument_lists_keeps_matching_pyarg_calls() -> None:
    first_decl = _var_decl("value")
    second_decl = _var_decl("value")
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("i"),
            _address_of("value", referenced=first_decl),
        ),
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("i"),
            _address_of("value", referenced=second_decl),
        ),
    )

    inferred = signature_rules_module.infer_argument_lists(cursor)

    assert inferred == [
        [_arg("value", "int")],
        [_arg("value", "int")],
    ]


def test_infer_signature_returns_signature_with_inferred_return_type() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value")))
    )

    inferred = signature_rules_module.infer_signature(
        ExtractedFunction(ml_name="foo", function_cursor=cursor)
    )

    assert inferred == [
        ExtractedSignature(
            arguments=[],
            return_type=RawType("int"),
        )
    ]


def test_infer_signature_returns_signature_with_inferred_arguments_when_return_is_unknown() -> None:
    value_decl = _var_decl("value", _int_literal("0"))
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("i"),
            _address_of("value", referenced=value_decl),
        ),
        _return_stmt(_call_expr("CustomFactory", _identifier_node("value"))),
    )

    inferred = signature_rules_module.infer_signature(
        ExtractedFunction(ml_name="foo", function_cursor=cursor)
    )

    assert inferred == [ExtractedSignature(arguments=[_arg("value", "int")])]


def test_infer_signature_returns_empty_when_return_type_is_unknown() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("CustomFactory", _identifier_node("value")))
    )

    inferred = signature_rules_module.infer_signature(
        ExtractedFunction(ml_name="foo", function_cursor=cursor)
    )

    assert inferred == []


def test_infer_signature_returns_empty_when_return_type_is_macro_expr() -> None:
    cursor = _fake_function_cursor_with_children(_return_stmt(_macro_expr("Py_RETURN_NONE")))

    inferred = signature_rules_module.infer_signature(
        ExtractedFunction(ml_name="foo", function_cursor=cursor)
    )

    assert inferred == []


def test_infer_signature_merges_inferred_arguments_and_return_type() -> None:
    value_decl = _var_decl("value", _int_literal("0"))
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("i"),
            _address_of("value", referenced=value_decl),
        ),
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value"))),
    )

    inferred = signature_rules_module.infer_signature(
        ExtractedFunction(ml_name="foo", function_cursor=cursor)
    )

    assert inferred == [
        ExtractedSignature(
            arguments=[_arg("value", "int")],
            return_type=RawType("int"),
        )
    ]


def test_infer_signature_preserves_arguments_when_return_type_is_macro_expr() -> None:
    value_decl = _var_decl("value", _int_literal("0"))
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("i"),
            _address_of("value", referenced=value_decl),
        ),
        _return_stmt(_macro_expr("Py_RETURN_NONE")),
    )

    inferred = signature_rules_module.infer_signature(
        ExtractedFunction(ml_name="foo", function_cursor=cursor)
    )

    assert inferred == [ExtractedSignature(arguments=[_arg("value", "int")])]


def test_infer_signature_uses_meth_noargs_without_pyarg_parse() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("CustomFactory", _identifier_node("value")))
    )

    inferred = signature_rules_module.infer_signature(
        ExtractedFunction(
            ml_name="foo",
            function_cursor=cursor,
            ml_flags=METH_NOARGS,
        )
    )

    assert inferred == [ExtractedSignature(arguments=[])]


def test_infer_signature_keeps_return_type_for_meth_noargs() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value")))
    )

    inferred = signature_rules_module.infer_signature(
        ExtractedFunction(
            ml_name="foo",
            function_cursor=cursor,
            ml_flags=METH_NOARGS,
        )
    )

    assert inferred == [
        ExtractedSignature(
            arguments=[],
            return_type=RawType("int"),
        )
    ]


def test_infer_signature_uses_meth_o_argument_shape_without_pyarg_parse() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("CustomFactory", _identifier_node("value")))
    )

    inferred = signature_rules_module.infer_signature(
        ExtractedFunction(
            ml_name="foo",
            function_cursor=cursor,
            ml_flags=METH_O,
        )
    )

    assert inferred == [
        ExtractedSignature(
            arguments=[
                _arg(
                    "arg",
                    "object",
                    kind=IRArgumentKind.POSITIONAL_ONLY,
                )
            ]
        )
    ]


def test_infer_signature_keeps_return_type_for_meth_o() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value")))
    )

    inferred = signature_rules_module.infer_signature(
        ExtractedFunction(
            ml_name="foo",
            function_cursor=cursor,
            ml_flags=METH_O,
        )
    )

    assert inferred == [
        ExtractedSignature(
            arguments=[
                _arg(
                    "arg",
                    "object",
                    kind=IRArgumentKind.POSITIONAL_ONLY,
                )
            ],
            return_type=RawType("int"),
        )
    ]


def test_infer_expr_type_raises_when_conditional_operator_children_count_is_invalid() -> None:
    expr = _FakeNode(
        kind=clang.cindex.CursorKind.CONDITIONAL_OPERATOR,
        children=[_identifier_node("cond"), _identifier_node("a")],
    )

    with pytest.raises(RuntimeError, match="CONDITIONAL_OPERATOR"):
        signature_rules_module.infer_expr_type(cast(clang.cindex.Cursor, expr))
