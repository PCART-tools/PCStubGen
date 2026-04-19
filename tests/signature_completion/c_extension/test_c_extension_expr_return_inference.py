from __future__ import annotations

import re

import clang.cindex
import pytest

from pcstubgen.signature_completion.c_extension.signatures import inferencer as signature_rules_module
from pcstubgen.type_models import AnyType, ListType, RawType, TupleType, UnionType
from tests._c_extension_test_support import (
    _FakeNode,
    _FakeToken,
    _address_of,
    _arg,
    _assignment,
    _call_expr,
    _conditional_expr,
    _fake_function_cursor_with_children,
    _identifier_node,
    _int_literal,
    _macro_expr,
    _null_ptr_literal,
    _return_stmt,
    _string_literal,
    _token_identifier_node,
    _var_decl,
    _wrap,
    _location_text,
    patch_inference_clang_helpers,
)


@pytest.fixture(autouse=True)
def _patch_fake_clang_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_inference_clang_helpers(monkeypatch, signature_rules_module)


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

