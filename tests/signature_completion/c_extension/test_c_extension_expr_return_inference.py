from __future__ import annotations

import re

import clang.cindex
import pytest

from pcstubgen.signature_completion.c_extension.signatures import inferencer as signature_rules_module
from pcstubgen.type_models import AnyType, ListType, RawType, TupleType, UnionType
from tests._c_extension_test_support import (
    _FakeCanonicalType,
    _FakeNode,
    _address_of,
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
                        RawType.str_,
                        RawType.none_,
                    )
                ),
                RawType.int_,
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

    assert inferred == TupleType((RawType.int_,))

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
    ],
)
def test_return_type_maps_representative_runtime_tokens(
    token_name: str,
    expected: str,
) -> None:
    cursor = _fake_function_cursor_with_children(_return_stmt(_address_of(token_name)))

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred is not None
    assert inferred.render() == expected

def test_return_type_returns_any_for_preserved_macro_token() -> None:
    macro_expr = _macro_expr("Py_RETURN_TRUE")
    cursor = _fake_function_cursor_with_children(_return_stmt(macro_expr))

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred == AnyType()

@pytest.mark.parametrize(
    ("call_name", "expected"),
    [
        ("PyLong_FromLong", "int"),
        ("PyUnicode_AsUTF8String", "bytes"),
        ("PyList_New", "list"),
        ("PyInt_FromLong", "int"),
        ("PyInt_FromSsize_t", "int"),
        ("Bytes_FromString", "bytes"),
        ("conn_text_from_chars", "str"),
    ],
)
def test_return_type_maps_representative_known_factory_calls(
    call_name: str,
    expected: str,
) -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr(call_name, _identifier_node("arg")))
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred is not None
    assert inferred.render() == expected


def test_return_type_maps_torch_variable_wrap() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("THPVariable_Wrap", _identifier_node("tensor")))
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred == RawType("torch.Tensor", imports=("torch",))


def test_return_type_maps_torch_variable_wrap_with_type() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(
            _call_expr(
                "THPVariable_WrapWithType",
                _identifier_node("tensor"),
                _identifier_node("cls"),
            )
        )
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred == UnionType((RawType.none_, RawType("torch.Tensor", imports=("torch",)))).canonicalize()


@pytest.mark.parametrize(
    ("call_name", "expected"),
    [
        ("THPVariable_is_nonzero", "bool"),
        ("THPUtils_packInt64", "int"),
        ("THPUtils_packDoubleAsInt", "int"),
    ],
)
def test_return_type_maps_torch_direct_helpers(call_name: str, expected: str) -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr(call_name, _identifier_node("value")))
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred is not None
    assert inferred.render() == expected


@pytest.mark.parametrize(
    ("cpp_type_spelling", "expected", "expected_imports"),
    [
        ("at::Tensor", "torch.Tensor", {"torch"}),
        ("bool", "bool", set()),
        ("int64_t", "int", set()),
        ("double", "float", set()),
        ("at::Stream", "torch.Stream", {"torch"}),
        ("c10::ScalarType", "torch.dtype", {"torch"}),
        ("c10::Layout", "torch.layout", {"torch"}),
        ("c10::QScheme", "torch.qscheme", {"torch"}),
        ("c10::ArrayRef<at::Tensor>", "tuple[torch.Tensor, ...]", {"torch"}),
        ("c10::ArrayRef<long>", "tuple[int, ...]", set()),
    ],
)
def test_return_type_maps_torch_wrap_overloads(
    cpp_type_spelling: str,
    expected: str,
    expected_imports: set[str],
) -> None:
    value = _identifier_node("value")
    value.type = _FakeCanonicalType(None, cpp_type_spelling)
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("wrap", value))
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred is not None
    assert inferred.render() == expected
    assert inferred.collect_imports() == expected_imports

@pytest.mark.parametrize(
    "call_name",
    [
        "PyObject_Call",
        "PyObject_CallFunction",
        "PyObject_CallFunctionObjArgs",
    ],
)
def test_infer_expr_type_keeps_pyobject_call_family_unmapped(call_name: str) -> None:
    with pytest.raises(RuntimeError, match=rf"无法识别的返回值工厂调用: {call_name}"):
        signature_rules_module.infer_expr_type(
            _call_expr(
                call_name,
                _identifier_node("callable"),
                _identifier_node("arg"),
            )
        )

def test_return_type_detects_generic_alias_factory_collects_types_import() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(
            _call_expr(
                "Py_GenericAlias",
                _identifier_node("cls"),
                _identifier_node("args"),
            )
        )
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred == RawType("types.GenericAlias", imports=("types",))
    assert inferred.collect_imports() == {"types"}

def test_infer_expr_type_maps_representative_numpy_factory_call() -> None:
    inferred = signature_rules_module.infer_expr_type(
        _call_expr(
            "PyArray_SimpleNew",
            _identifier_node("arg"),
            _identifier_node("dims"),
            _identifier_node("typenum"),
        )
    )

    assert inferred.render() == "numpy.ndarray"

def test_return_type_detects_numpy_dtype_factory() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(
            _call_expr(
                "PyArray_DescrNewByteorder",
                _identifier_node("descr"),
                _identifier_node("newendian"),
            )
        )
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred is not None
    assert inferred.render() == "numpy.dtype"

@pytest.mark.parametrize(
    ("call_name", "expected", "expected_imports"),
    [
        ("PyArray_NewCopy", "numpy.ndarray", {"numpy"}),
        ("PyArray_NewFromDescr", "numpy.ndarray", {"numpy"}),
        ("PyArray_ToList", "list", set()),
        ("PyArray_ToString", "bytes", set()),
        ("PyArray_DescrFromType", "numpy.dtype", {"numpy"}),
    ],
)
def test_return_type_detects_representative_new_numpy_factory_mappings(
    call_name: str,
    expected: str,
    expected_imports: set[str],
) -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(
            _call_expr(
                call_name,
                _identifier_node("arg"),
                _identifier_node("other"),
            )
        )
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred is not None
    assert inferred.render() == expected
    assert inferred.collect_imports() == expected_imports

def test_return_type_detects_numpy_helper_int_factory() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("pylong_from_int128", _identifier_node("value")))
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred is not None
    assert inferred.render() == "int"

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

    assert inferred == RawType.int_

def test_return_type_uses_last_assignment_before_return_when_types_converge() -> None:
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

    assert inferred == RawType.int_

def test_return_type_uses_last_assignment_before_return_when_types_diverge() -> None:
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

    assert inferred == RawType.float_

def test_return_type_does_not_revisit_earlier_assignment_when_last_assignment_is_null() -> None:
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
            _null_ptr_literal(),
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

def test_return_type_traces_global_decl_ref_initializer() -> None:
    value_decl = _var_decl(
        "value",
        _call_expr("PyLong_FromLong", _identifier_node("value")),
    )
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_token_identifier_node("value", referenced=value_decl)),
    )

    inferred = signature_rules_module.infer_return_type(cursor)

    assert inferred == RawType.int_

def test_return_type_traces_static_local_decl_ref_initializer() -> None:
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

    assert inferred == RawType.int_

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

    assert inferred == RawType.int_

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

    assert inferred == RawType.int_

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

    assert inferred == RawType.int_
