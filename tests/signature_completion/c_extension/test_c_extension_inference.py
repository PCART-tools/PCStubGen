from __future__ import annotations

import re
from pathlib import Path
from typing import cast

import clang.cindex
import pytest

from pcstubgen.models import ArgumentKind, Signature
from pcstubgen.signature_completion.c_extension.method_flags import (
    METH_FASTCALL,
    METH_KEYWORDS,
    METH_NOARGS,
    METH_O,
    METH_VARARGS,
)
from pcstubgen.signature_completion.c_extension.signatures import inference as signature_rules_module
from pcstubgen.type_models import AnyType, ListType, RawType, TupleType, UnionType
from tests._c_extension_test_support import (
    _FakeNode,
    _address_of,
    _arg,
    _assignment,
    _call_expr,
    _conditional_expr,
    _extent_for_source_snippet,
    _fake_function_cursor_with_children,
    _float_literal,
    _identifier_node,
    _init_list,
    _int_literal,
    _macro_expr,
    _null_ptr_literal,
    _return_stmt,
    _string_literal,
    _token_identifier_node,
    _var_decl,
    _wrap,
)


def _location_text(text: str) -> object:
    class _Location:
        def __str__(self) -> str:
            return text

    return _Location()


@pytest.fixture(autouse=True)
def _patch_fake_clang_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """让 fake cursor 支持本组测试依赖的 clang helper。"""
    real_cursor_get_text = signature_rules_module.get_cursor_text

    def fake_cursor_get_text(cursor: object) -> str:
        extent = getattr(cursor, "extent", None)
        if isinstance(extent, str):
            return extent
        if hasattr(extent, "start") and hasattr(extent, "end"):
            start = extent.start
            end = extent.end
            if start.file is not None:
                source_bytes = Path(start.file.name).read_bytes()
                return source_bytes[start.offset:end.offset].decode(
                    "utf-8",
                    errors="ignore",
                )
        return real_cursor_get_text(cast(clang.cindex.Cursor, cursor))

    monkeypatch.setattr(
        signature_rules_module,
        "get_cursor_text",
        fake_cursor_get_text,
    )
    real_get_call_expr_source_name = signature_rules_module.get_call_expr_source_name

    def fake_get_call_expr_source_name(cursor: object) -> str:
        if isinstance(cursor, _FakeNode):
            tokens = list(cursor.get_tokens())
            if not tokens:
                raise RuntimeError(f"调用表达式起点缺少 token, cursor: {cursor.location}")
            return tokens[0].spelling
        return real_get_call_expr_source_name(cast(clang.cindex.Cursor, cursor))

    monkeypatch.setattr(
        signature_rules_module,
        "get_call_expr_source_name",
        fake_get_call_expr_source_name,
    )

    def fake_cursor_binary_operator_kind(cursor: object) -> int:
        operator_kind = cast(_FakeNode, cursor).binary_operator_kind
        assert operator_kind is not None
        return operator_kind

    monkeypatch.setattr(
        signature_rules_module,
        "get_cursor_binary_operator_kind",
        fake_cursor_binary_operator_kind,
    )


def test_infer_expr_type_raises_when_macro_name_is_not_exposed_by_ast() -> None:
    macro_expr = _FakeNode(
        kind=clang.cindex.CursorKind.UNEXPOSED_EXPR,
        children=[_FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR)],
    )

    with pytest.raises(RuntimeError, match="AST 节点提取名称|对象返回标识符"):
        signature_rules_module.infer_expr_type(macro_expr)


def test_infer_expr_type_keeps_raw_py_buildvalue_container_union_shape() -> None:
    """`Py_BuildValue` 直接推断时保留 parser 原始类型树，顶层再统一规范化。"""
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
                UnionType(
                    (
                        RawType("str"),
                        RawType("None"),
                    )
                ),
                RawType("int"),
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


def test_infer_expr_type_falls_back_to_any_when_py_buildvalue_object_slots_are_unknown() -> None:
    inferred = signature_rules_module.infer_expr_type(
        _call_expr(
            "Py_BuildValue",
            _string_literal("(O)"),
            _call_expr("CustomFactory", _identifier_node("value")),
        )
    )

    assert inferred == TupleType((AnyType(),))


def test_infer_expr_type_falls_back_to_any_when_py_buildvalue_o_ampersand_converter_is_unknown() -> None:
    inferred = signature_rules_module.infer_expr_type(
        _call_expr(
            "Py_BuildValue",
            _string_literal("(O&)"),
            _identifier_node("converter"),
            _identifier_node("value"),
        )
    )

    assert inferred == TupleType((AnyType(),))


def test_infer_expr_type_skips_unknown_conditional_branches() -> None:
    inferred = signature_rules_module.infer_expr_type(
        _conditional_expr(
            _macro_expr("Py_RETURN_NONE"),
            _call_expr("CustomFactory", _identifier_node("left")),
            _identifier_node("UnknownToken"),
        )
    )

    assert inferred == UnionType(())


@pytest.mark.parametrize("call_name", ["PyErr_NoMemory", "PyErr_Format"])
def test_infer_expr_type_detects_error_return_factories(call_name: str) -> None:
    inferred = signature_rules_module.infer_expr_type(
        _call_expr(
            call_name,
            _identifier_node("exception"),
            _string_literal("message"),
        )
    )

    assert inferred == UnionType(())


def test_infer_expr_type_does_not_treat_pyerr_prefix_as_error_return() -> None:
    with pytest.raises(RuntimeError, match="返回值工厂调用"):
        signature_rules_module.infer_expr_type(
            _call_expr(
                "PyErr_NewException",
                _string_literal("module.CustomError"),
                _null_ptr_literal(),
                _null_ptr_literal(),
            )
        )


def test_infer_expr_type_raises_for_unsupported_expr() -> None:
    expr_cursor = _call_expr("CustomFactory", _identifier_node("value"))
    expr_cursor.location = _location_text("factory.c:10:2")

    with pytest.raises(
        RuntimeError,
        match=rf"返回值工厂调用.*{re.escape('factory.c:10:2')}",
    ):
        signature_rules_module.infer_expr_type(expr_cursor)


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
def test_return_type_returns_any_for_preserved_macro_tokens(token_name: str) -> None:
    macro_expr = _macro_expr(token_name)
    cursor = _fake_function_cursor_with_children(_return_stmt(macro_expr))

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred == AnyType()


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


@pytest.mark.parametrize(
    "call_name",
    [
        "PyArray_ContiguousFromObject",
        "PyArray_Arange",
        "PyArray_SimpleNew",
        "PyArray_FROMANY",
    ],
)
def test_infer_expr_type_detects_numpy_ndarray_factories(call_name: str) -> None:
    inferred = signature_rules_module.infer_expr_type(
        _call_expr(
            call_name,
            _identifier_node("arg"),
            _identifier_node("dims"),
            _identifier_node("typenum"),
        )
    )

    assert inferred.render() == "numpy.ndarray"


def test_infer_expr_type_uses_call_start_token_for_function_like_macro_call() -> None:
    call_cursor = _call_expr(
        "PyArray_ContiguousFromObject",
        _identifier_node("source"),
        _identifier_node("typenum"),
        _int_literal("0"),
        _int_literal("0"),
    )
    callee_cursor = next(call_cursor.get_children())
    callee_cursor.extent = "PyArray_ContiguousFromObject(source, typenum, 0, 0)"

    inferred = signature_rules_module.infer_expr_type(call_cursor)

    assert inferred.render() == "numpy.ndarray"


def test_return_type_detects_pycapsule_new_factory() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(
            _call_expr(
                "PyCapsule_New",
                _identifier_node("pointer"),
                _identifier_node("name"),
                _identifier_node("destructor"),
            )
        )
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred is not None
    assert inferred.render() == "types.CapsuleType"


def test_return_type_traces_local_decl_ref_initialized_from_factory() -> None:
    cobj_decl = _var_decl(
        "cobj",
        _call_expr(
            "PyCapsule_New",
            _identifier_node("pointer"),
            _identifier_node("name"),
            _identifier_node("destructor"),
        ),
    )
    cursor = _fake_function_cursor_with_children(
        cobj_decl,
        _return_stmt(_token_identifier_node("cobj", referenced=cobj_decl)),
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred is not None
    assert inferred.render() == "types.CapsuleType"


def test_py_buildvalue_traces_local_decl_ref_assigned_after_zero_initializer() -> None:
    cobj_decl = _var_decl("cobj", _null_ptr_literal())
    cursor = _fake_function_cursor_with_children(
        cobj_decl,
        _assignment(
            "cobj",
            _call_expr(
                "PyCapsule_New",
                _identifier_node("pointer"),
                _identifier_node("name"),
                _identifier_node("destructor"),
            ),
            referenced=cobj_decl,
        ),
        _return_stmt(
            _call_expr(
                "Py_BuildValue",
                _string_literal("iN"),
                _identifier_node("changed"),
                _token_identifier_node("cobj", referenced=cobj_decl),
            )
        ),
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred is not None
    assert inferred.render() == "tuple[int, types.CapsuleType]"


def test_py_buildvalue_skips_chained_null_assignment_before_factory_assignment() -> None:
    arr_decl = _var_decl("arr")
    other_decl = _var_decl("other")
    cursor = _fake_function_cursor_with_children(
        arr_decl,
        other_decl,
        _assignment(
            "arr",
            _assignment("other", _null_ptr_literal(), referenced=other_decl),
            referenced=arr_decl,
        ),
        _assignment(
            "arr",
            _call_expr(
                "PyArray_SimpleNew",
                _identifier_node("ndim"),
                _identifier_node("dims"),
                _identifier_node("typenum"),
            ),
            referenced=arr_decl,
        ),
        _return_stmt(
            _call_expr(
                "Py_BuildValue",
                _string_literal("N"),
                _token_identifier_node("arr", referenced=arr_decl),
            )
        ),
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred is not None
    assert inferred.render() == "numpy.ndarray"


def test_py_buildvalue_traces_numpy_factory_local_decl_ref_for_n_slot() -> None:
    arr_decl = _var_decl(
        "arr",
        _call_expr(
            "PyArray_SimpleNew",
            _identifier_node("ndim"),
            _identifier_node("dims"),
            _identifier_node("typenum"),
        ),
    )
    cursor = _fake_function_cursor_with_children(
        arr_decl,
        _return_stmt(
            _call_expr(
                "Py_BuildValue",
                _string_literal("N"),
                _token_identifier_node("arr", referenced=arr_decl),
            )
        ),
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred is not None
    assert inferred.render() == "numpy.ndarray"


def test_py_buildvalue_traces_numpy_factory_local_decl_ref_for_o_slot() -> None:
    arr_decl = _var_decl(
        "arr",
        _call_expr(
            "PyArray_FROMANY",
            _identifier_node("source"),
            _identifier_node("typenum"),
            _identifier_node("min_depth"),
            _identifier_node("max_depth"),
            _identifier_node("requirements"),
        ),
    )
    cursor = _fake_function_cursor_with_children(
        arr_decl,
        _return_stmt(
            _call_expr(
                "Py_BuildValue",
                _string_literal("O"),
                _token_identifier_node("arr", referenced=arr_decl),
            )
        ),
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred is not None
    assert inferred.render() == "numpy.ndarray"


def test_return_type_traces_local_assignment_without_tokens() -> None:
    value_decl = _var_decl("value", _null_ptr_literal())
    assignment = _assignment(
        "value",
        _call_expr("PyLong_FromLong", _identifier_node("raw_value")),
        referenced=value_decl,
    )
    assignment._tokens = []
    cursor = _fake_function_cursor_with_children(
        value_decl,
        assignment,
        _return_stmt(_token_identifier_node("value", referenced=value_decl)),
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred == RawType("int")


def test_return_type_accepts_multiple_local_assignments_when_types_converge() -> None:
    value_decl = _var_decl("value", _null_ptr_literal())
    cursor = _fake_function_cursor_with_children(
        value_decl,
        _assignment(
            "value",
            _call_expr("PyLong_FromLong", _identifier_node("left")),
            referenced=value_decl,
        ),
        _assignment(
            "value",
            _call_expr("PyLong_FromLong", _identifier_node("right")),
            referenced=value_decl,
        ),
        _return_stmt(_token_identifier_node("value", referenced=value_decl)),
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred == RawType("int")


def test_return_type_rejects_local_assignments_when_types_diverge() -> None:
    value_decl = _var_decl("value", _null_ptr_literal())
    cursor = _fake_function_cursor_with_children(
        value_decl,
        _assignment(
            "value",
            _call_expr("PyLong_FromLong", _identifier_node("left")),
            referenced=value_decl,
        ),
        _assignment(
            "value",
            _call_expr("PyFloat_FromDouble", _identifier_node("right")),
            referenced=value_decl,
        ),
        _return_stmt(_token_identifier_node("value", referenced=value_decl)),
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred == AnyType()


def test_return_type_rejects_local_decl_ref_when_only_zero_candidates_exist() -> None:
    value_decl = _var_decl("value", _null_ptr_literal())
    cursor = _fake_function_cursor_with_children(
        value_decl,
        _return_stmt(_token_identifier_node("value", referenced=value_decl)),
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred == AnyType()


def test_return_type_does_not_trace_global_decl_ref() -> None:
    value_decl = _var_decl(
        "value",
        _call_expr("PyLong_FromLong", _identifier_node("value")),
    )
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_token_identifier_node("value", referenced=value_decl)),
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred == AnyType()


def test_return_type_does_not_trace_static_local_decl_ref() -> None:
    value_decl = _var_decl(
        "value",
        _call_expr("PyLong_FromLong", _identifier_node("value")),
        storage_class=clang.cindex.StorageClass.STATIC,
    )
    cursor = _fake_function_cursor_with_children(
        value_decl,
        _return_stmt(_token_identifier_node("value", referenced=value_decl)),
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred == AnyType()


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


def test_return_type_drops_error_return_factory_branch() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("PyErr_NoMemory")),
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value"))),
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred == RawType("int")


def test_return_type_drops_error_return_factory_conditional_branch() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(
            _conditional_expr(
                _identifier_node("cond"),
                _call_expr("PyErr_NoMemory"),
                _call_expr("PyLong_FromLong", _identifier_node("value")),
            )
        )
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred == RawType("int")


def test_return_type_returns_any_for_unsupported_returns() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("CustomFactory", _identifier_node("value"))),
        _return_stmt(),
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred == AnyType()


def test_return_type_returns_any_when_function_has_no_return() -> None:
    cursor = _fake_function_cursor_with_children()

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred == AnyType()


def test_return_type_skips_failed_return_expr_and_keeps_successful_returns() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("CustomFactory", _identifier_node("value"))),
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value"))),
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred == RawType("int")


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
                default_value="...",
            ),
        ]
    ]


def test_infer_type_object_type_for_pyarg_reads_name_from_extent_source_text(tmp_path: Path) -> None:
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

    inferred = signature_rules_module._infer_type_object_type_for_pyarg(cursor)

    assert inferred is not None
    assert inferred.render() == "str"


def test_infer_type_object_type_for_pyarg_reads_numpy_name_from_extent_source_text(
    tmp_path: Path,
) -> None:
    source = tmp_path / "numpy_object_type_from_extent.c"
    source.write_text(
        "\n".join(
            [
                "/* 中文注释 */",
                "PyArg_ParseTuple(args, \"O!\", (&PyArray_Type), &value);",
            ]
        ),
        encoding="utf-8",
    )
    cursor = _FakeNode(
        kind=clang.cindex.CursorKind.UNARY_OPERATOR,
        extent=_extent_for_source_snippet(source, "(&PyArray_Type)"),
    )

    inferred = signature_rules_module._infer_type_object_type_for_pyarg(cursor)

    assert inferred is not None
    assert inferred.render() == "numpy.ndarray"


def test_infer_type_object_type_for_pyarg_propagates_extent_source_text_read_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cursor = _FakeNode(
        kind=clang.cindex.CursorKind.UNARY_OPERATOR,
        extent="boom",
    )
    monkeypatch.setattr(
        signature_rules_module,
        "get_cursor_text",
        lambda node: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        signature_rules_module._infer_type_object_type_for_pyarg(cursor)


def test_infer_converter_type_for_pyarg_reads_numpy_converter_name_from_extent_source_text(
    tmp_path: Path,
) -> None:
    source = tmp_path / "numpy_converter_from_extent.c"
    source.write_text(
        "\n".join(
            [
                "/* 中文注释 */",
                "PyArg_ParseTuple(args, \"O&\", NI_ObjectToInputArray, &value);",
            ]
        ),
        encoding="utf-8",
    )
    cursor = _identifier_node("converter")
    cursor.extent = _extent_for_source_snippet(source, "NI_ObjectToInputArray")

    inferred = signature_rules_module._infer_converter_type_for_pyarg(cursor)

    assert inferred is not None
    assert inferred.render() == "numpy.ndarray"


def test_infer_converter_type_for_pyarg_reads_optional_numpy_converter_name_from_extent_source_text(
    tmp_path: Path,
) -> None:
    source = tmp_path / "optional_numpy_converter_from_extent.c"
    source.write_text(
        "\n".join(
            [
                "/* 中文注释 */",
                "PyArg_ParseTuple(args, \"O&\", NI_ObjectToOptionalInputArray, &value);",
            ]
        ),
        encoding="utf-8",
    )
    cursor = _identifier_node("converter")
    cursor.extent = _extent_for_source_snippet(source, "NI_ObjectToOptionalInputArray")

    inferred = signature_rules_module._infer_converter_type_for_pyarg(cursor)

    assert inferred is not None
    assert inferred.render() == "numpy.ndarray | None"


def test_infer_converter_type_for_pyarg_reads_tuple_converter_name_from_extent_source_text(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tuple_converter_from_extent.c"
    source.write_text(
        "\n".join(
            [
                "/* 中文注释 */",
                "PyArg_ParseTuple(args, \"O&\", PyArray_IntpConverter, &value);",
            ]
        ),
        encoding="utf-8",
    )
    cursor = _identifier_node("converter")
    cursor.extent = _extent_for_source_snippet(source, "PyArray_IntpConverter")

    inferred = signature_rules_module._infer_converter_type_for_pyarg(cursor)

    assert inferred is not None
    assert inferred.render() == "tuple[int, ...]"


def test_infer_argument_lists_falls_back_to_object_for_unknown_o_bang_type(
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


def test_infer_argument_lists_falls_back_to_object_for_unknown_o_ampersand_converter(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unknown_converter_from_extent.c"
    source.write_text(
        "\n".join(
            [
                "/* 中文注释 */",
                "PyArg_ParseTuple(args, \"O&\", UnknownConverter, &value);",
            ]
        ),
        encoding="utf-8",
    )
    value_decl = _var_decl("value")
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("O&"),
            _FakeNode(
                kind=clang.cindex.CursorKind.DECL_REF_EXPR,
                extent=_extent_for_source_snippet(source, "UnknownConverter"),
            ),
            _address_of("value", referenced=value_decl),
        )
    )

    inferred = signature_rules_module.infer_argument_lists(cursor)

    assert inferred == [[_arg("value", "object")]]


def test_infer_argument_lists_falls_back_to_unknown_default_value_when_default_parse_fails() -> None:
    label_decl = _var_decl("label", _identifier_node("UNSUPPORTED_DEFAULT"))
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("|z"),
            _address_of("label", referenced=label_decl),
        )
    )

    inferred = signature_rules_module.infer_argument_lists(cursor)

    assert inferred == [
        [
            _arg(
                "label",
                UnionType((RawType("str"), RawType("None"))),
                default_value="...",
            )
        ]
    ]


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


def test_infer_argument_lists_parses_pyarg_parsetuple_and_keywords_sizet_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(signature_rules_module, "evaluate_cursor", lambda _: 0.0)
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

    assert inferred == [[_arg("x", "float", default_value="0.0")]]


def test_infer_argument_lists_raises_for_parse_tuple_and_keywords_without_valid_kwlist() -> None:
    invalid_entry = _identifier_node("bad")
    invalid_entry.location = _location_text("kwlist.c:7:9")
    invalid_kwlist_decl = _var_decl("kwlist", _init_list(invalid_entry, _null_ptr_literal()))
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

    with pytest.raises(
        RuntimeError,
        match=rf"kwlist.*{re.escape('kwlist.c:7:9')}",
    ):
        signature_rules_module.infer_argument_lists(cursor)


def test_infer_argument_lists_raises_when_argument_name_cannot_be_resolved() -> None:
    invalid_slot = _int_literal("0")
    invalid_slot.location = _location_text("parse_tuple.c:12:8")
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("i"),
            invalid_slot,
        )
    )

    with pytest.raises(
        RuntimeError,
        match=rf"C 参数槽位.*{re.escape('parse_tuple.c:12:8')}",
    ):
        signature_rules_module.infer_argument_lists(cursor)


def test_infer_argument_lists_raises_when_parse_tuple_format_string_is_not_literal() -> None:
    value_decl = _var_decl("value")
    format_cursor = _identifier_node("fmt")
    format_cursor.location = _location_text("parse_tuple.c:20:6")
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            format_cursor,
            _address_of("value", referenced=value_decl),
        )
    )

    with pytest.raises(
        RuntimeError,
        match=rf"字符串字面量.*{re.escape('parse_tuple.c:20:6')}",
    ):
        signature_rules_module.infer_argument_lists(cursor)


def test_render_default_expr_raises_with_cursor_location_for_unsupported_expr() -> None:
    expr_cursor = _call_expr("PyLong_FromLong", _identifier_node("value"))
    expr_cursor.location = _location_text("default_value.c:15:3")
    target_decl = _var_decl("value")

    with pytest.raises(
        RuntimeError,
        match=rf"不支持的默认值表达式类型.*{re.escape('default_value.c:15:3')}",
    ):
        signature_rules_module._render_default_expr(expr_cursor, target_decl)


def test_render_default_expr_uses_evaluated_float_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[_FakeNode] = []
    cursor = _float_literal("1e+06")
    target_decl = _var_decl("value")

    monkeypatch.setattr(signature_rules_module, "is_nullptr_or_zero", lambda _: False)
    monkeypatch.setattr(
        signature_rules_module,
        "evaluate_cursor",
        lambda received_cursor: observed.append(received_cursor) or 1000000.0,
    )

    assert signature_rules_module._render_default_expr(cursor, target_decl) == "1000000.0"
    assert observed == [cursor]


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

    inferred = signature_rules_module.infer_signature(cursor)

    assert inferred == [
        Signature(
            args=[],
            return_type=RawType("int"),
        )
    ]


def test_infer_signature_returns_known_arguments_when_return_type_is_unknown() -> None:
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

    inferred = signature_rules_module.infer_signature(cursor)

    assert inferred == [Signature(args=[_arg("value", "int")], return_type=AnyType())]


def test_infer_signature_returns_any_when_return_type_is_unknown() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("CustomFactory", _identifier_node("value")))
    )

    inferred = signature_rules_module.infer_signature(cursor)

    assert inferred == [Signature(return_type=AnyType())]


def test_infer_signature_returns_any_when_return_type_is_macro_expr() -> None:
    cursor = _fake_function_cursor_with_children(_return_stmt(_macro_expr("Py_RETURN_NONE")))

    inferred = signature_rules_module.infer_signature(cursor)

    assert inferred == [Signature(return_type=AnyType())]


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
        cursor
    )

    assert inferred == [
        Signature(
            args=[_arg("value", "int")],
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

    inferred = signature_rules_module.infer_signature(cursor)

    assert inferred == [Signature(args=[_arg("value", "int")], return_type=AnyType())]


def test_infer_signature_uses_meth_noargs_without_pyarg_parse() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("CustomFactory", _identifier_node("value")))
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_NOARGS,
    )

    assert inferred == [Signature(args=[], return_type=AnyType())]


def test_infer_signature_keeps_return_type_for_meth_noargs() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value")))
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_NOARGS,
    )

    assert inferred == [
        Signature(
            args=[],
            return_type=RawType("int"),
        )
    ]


def test_infer_signature_uses_meth_o_argument_shape_without_pyarg_parse() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("CustomFactory", _identifier_node("value")))
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_O,
    )

    assert inferred == [
        Signature(
            args=[
                _arg(
                    "arg",
                    "object",
                    kind=ArgumentKind.POSITIONAL_ONLY,
                )
            ],
            return_type=AnyType(),
        )
    ]


def test_infer_signature_keeps_return_type_for_meth_o() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value")))
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_O,
    )

    assert inferred == [
        Signature(
            args=[
                _arg(
                    "arg",
                    "object",
                    kind=ArgumentKind.POSITIONAL_ONLY,
                )
            ],
            return_type=RawType("int"),
        )
    ]


def test_infer_minimal_signatures_supports_varargs_and_keywords() -> None:
    inferred = signature_rules_module.infer_minimal_signatures(
        METH_VARARGS | METH_KEYWORDS,
        return_type=RawType("int"),
    )

    assert inferred == [
        Signature(
            args=[
                _arg("args", "object", kind=ArgumentKind.VAR_POSITIONAL),
                _arg("kwargs", "object", kind=ArgumentKind.VAR_KEYWORD),
            ],
            return_type=RawType("int"),
        )
    ]


def test_infer_minimal_signatures_supports_fastcall_and_keywords() -> None:
    inferred = signature_rules_module.infer_minimal_signatures(
        METH_FASTCALL | METH_KEYWORDS,
        return_type=RawType("int"),
    )

    assert inferred == [
        Signature(
            args=[
                _arg("args", "object", kind=ArgumentKind.VAR_POSITIONAL),
                _arg("kwargs", "object", kind=ArgumentKind.VAR_KEYWORD),
            ],
            return_type=RawType("int"),
        )
    ]
