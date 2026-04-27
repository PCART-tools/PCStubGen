from __future__ import annotations

import subprocess
import sysconfig
from pathlib import Path

import clang.cindex
import pytest

from pcstubgen.models import ArgumentKind, Signature
from pcstubgen.signature_completion.c_extension.method_flags import (
    METH_FASTCALL,
    METH_CLASS,
    METH_KEYWORDS,
    METH_NOARGS,
    METH_O,
    METH_STATIC,
    METH_VARARGS,
)
from pcstubgen.signature_completion.c_extension import inferencer as signature_rules_module
from pcstubgen.type_models import AnyType, RawType, UnionType
from tests._c_extension_test_support import (
    _FakeCanonicalType,
    _FakeNode,
    _address_of,
    _array_subscript,
    _arg,
    _assignment,
    _call_expr,
    _c_style_cast_expr,
    _fake_function_cursor_with_children,
    _init_list,
    _identifier_node,
    _int_literal,
    _null_ptr_literal,
    _param_decl,
    _return_stmt,
    _string_literal,
    _token_identifier_node,
    _var_decl,
    patch_inference_clang_helpers,
)


@pytest.fixture(autouse=True)
def _patch_fake_clang_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_inference_clang_helpers(monkeypatch, signature_rules_module)


def test_infer_signature_does_not_parse_arguments_without_flags() -> None:
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

    inferred = signature_rules_module.infer_signature(cursor)

    assert inferred == [Signature(args=[], return_type=RawType.int_)]


def test_infer_signature_inserts_self_for_method_meth_noargs() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value")))
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_NOARGS,
        owner_class=object,
    )

    assert inferred == [Signature(args=[_arg("self")], return_type=RawType.int_)]


def test_infer_signature_returns_self_for_instance_receiver() -> None:
    conn_decl = _param_decl("conn")
    cursor = _fake_function_cursor_with_children(
        conn_decl,
        _return_stmt(_c_style_cast_expr(_token_identifier_node("conn", referenced=conn_decl))),
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_NOARGS,
        owner_class=object,
    )

    assert inferred == [Signature(args=[_arg("self")], return_type=RawType.self_)]
    assert inferred[0].return_type.collect_imports() == {"typing"}


def test_infer_signature_returns_self_through_local_alias() -> None:
    self_decl = _param_decl("self")
    dummy_decl = _param_decl("dummy")
    rv_decl = _var_decl("rv", _null_ptr_literal())
    cursor = _fake_function_cursor_with_children(
        self_decl,
        dummy_decl,
        rv_decl,
        _call_expr("Py_INCREF", _token_identifier_node("self", referenced=self_decl)),
        _assignment(
            "rv",
            _c_style_cast_expr(_token_identifier_node("self", referenced=self_decl)),
            referenced=rv_decl,
        ),
        _return_stmt(_token_identifier_node("rv", referenced=rv_decl)),
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_O,
        owner_class=object,
    )

    assert inferred == [
        Signature(
            args=[
                _arg("self", kind=ArgumentKind.POSITIONAL_ONLY),
                _arg("arg", "object", kind=ArgumentKind.POSITIONAL_ONLY),
            ],
            return_type=RawType.self_,
        )
    ]


def test_infer_signature_inserts_cls_for_classmethod_meth_noargs() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value")))
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_NOARGS | METH_CLASS,
        owner_class=object,
    )

    assert inferred == [Signature(args=[_arg("cls")], return_type=RawType.int_)]


def test_infer_signature_does_not_return_self_for_classmethod_receiver() -> None:
    cls_decl = _param_decl("type")
    cursor = _fake_function_cursor_with_children(
        cls_decl,
        _return_stmt(_token_identifier_node("type", referenced=cls_decl)),
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_NOARGS | METH_CLASS,
        owner_class=object,
    )

    assert inferred == [Signature(args=[_arg("cls")], return_type=AnyType())]


def test_infer_signature_skips_receiver_for_staticmethod_meth_noargs() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value")))
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_NOARGS | METH_STATIC,
        owner_class=object,
    )

    assert inferred == [Signature(args=[], return_type=RawType.int_)]


def test_infer_signature_does_not_return_self_for_staticmethod_first_param() -> None:
    null_self_decl = _param_decl("self")
    cursor = _fake_function_cursor_with_children(
        null_self_decl,
        _return_stmt(_token_identifier_node("self", referenced=null_self_decl)),
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_NOARGS | METH_STATIC,
        owner_class=object,
    )

    assert inferred == [Signature(args=[], return_type=AnyType())]


def test_infer_signature_inserts_self_for_method_meth_o() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value")))
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_O,
        owner_class=object,
    )

    assert inferred == [
        Signature(
            args=[
                _arg("self", kind=ArgumentKind.POSITIONAL_ONLY),
                _arg("arg", "object", kind=ArgumentKind.POSITIONAL_ONLY),
            ],
            return_type=RawType.int_,
        )
    ]


def test_infer_signature_inserts_cls_for_classmethod_meth_o() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value")))
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_O | METH_CLASS,
        owner_class=object,
    )

    assert inferred == [
        Signature(
            args=[
                _arg("cls", kind=ArgumentKind.POSITIONAL_ONLY),
                _arg("arg", "object", kind=ArgumentKind.POSITIONAL_ONLY),
            ],
            return_type=RawType.int_,
        )
    ]


def test_infer_signature_skips_receiver_for_staticmethod_meth_o() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value")))
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_O | METH_STATIC,
        owner_class=object,
    )

    assert inferred == [
        Signature(
            args=[_arg("arg", "object", kind=ArgumentKind.POSITIONAL_ONLY)],
            return_type=RawType.int_,
        )
    ]


def test_infer_signature_ignores_body_parse_for_meth_o() -> None:
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
        cursor,
        flags=METH_O | METH_CLASS,
        owner_class=object,
    )

    assert inferred == [
        Signature(
            args=[
                _arg("cls", kind=ArgumentKind.POSITIONAL_ONLY),
                _arg("arg", "object", kind=ArgumentKind.POSITIONAL_ONLY),
            ],
            return_type=RawType.int_,
        )
    ]


def test_infer_signature_refines_meth_o_argument_from_type_check() -> None:
    self_decl = _param_decl("self")
    value_decl = _param_decl("value")
    cursor = _fake_function_cursor_with_children(
        self_decl,
        value_decl,
        _call_expr(
            "PyTuple_Check",
            _token_identifier_node("value", referenced=value_decl),
        ),
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value"))),
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_O,
        owner_class=object,
    )

    assert inferred == [
        Signature(
            args=[
                _arg("self", kind=ArgumentKind.POSITIONAL_ONLY),
                _arg(
                    "arg",
                    RawType("tuple[typing.Any, ...]", imports=("typing",)),
                    kind=ArgumentKind.POSITIONAL_ONLY,
                ),
            ],
            return_type=RawType.int_,
        )
    ]


def test_infer_signature_keeps_parse_tuple_result_for_meth_varargs_method() -> None:
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
        cursor,
        flags=METH_VARARGS,
        owner_class=object,
    )

    assert inferred == [
        Signature(
            args=[
                _arg("self", kind=ArgumentKind.POSITIONAL_ONLY),
                _arg("value", "int", kind=ArgumentKind.POSITIONAL_ONLY),
            ],
            return_type=RawType.int_,
        )
    ]


def test_infer_signature_falls_back_to_varargs_for_meth_varargs() -> None:
    value_decl = _var_decl("value", _int_literal("0"))
    kwlist_decl = _var_decl("kwlist", _init_list(_string_literal("value"), _null_ptr_literal()))
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTupleAndKeywords",
            _identifier_node("args"),
            _identifier_node("kwds"),
            _string_literal("i"),
            _token_identifier_node("kwlist", referenced=kwlist_decl),
            _address_of("value", referenced=value_decl),
        )
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_VARARGS,
        owner_class=object,
    )

    assert inferred == [
        Signature(
            args=[
                _arg("self"),
                _arg("args", "object", kind=ArgumentKind.VAR_POSITIONAL),
            ],
            return_type=AnyType(),
        )
    ]


def test_infer_signature_keeps_parse_tuple_and_keywords_result_for_classmethod() -> None:
    value_decl = _var_decl("value", _int_literal("0"))
    kwlist_decl = _var_decl("kwlist", _init_list(_string_literal("value"), _null_ptr_literal()))
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTupleAndKeywords",
            _identifier_node("args"),
            _identifier_node("kwds"),
            _string_literal("i"),
            _token_identifier_node("kwlist", referenced=kwlist_decl),
            _address_of("value", referenced=value_decl),
        ),
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value"))),
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_VARARGS | METH_KEYWORDS | METH_CLASS,
        owner_class=object,
    )

    assert inferred == [
        Signature(
            args=[_arg("cls"), _arg("value", "int")],
            return_type=RawType.int_,
        )
    ]


def test_infer_signature_keeps_parse_tuple_result_and_appends_kwargs() -> None:
    value_decl = _var_decl("value", _int_literal("0"))
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("i"),
            _address_of("value", referenced=value_decl),
        )
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_VARARGS | METH_KEYWORDS | METH_CLASS,
        owner_class=object,
    )

    assert inferred == [
        Signature(
            args=[
                _arg("cls", kind=ArgumentKind.POSITIONAL_ONLY),
                _arg("value", "int", kind=ArgumentKind.POSITIONAL_ONLY),
            ],
            return_type=AnyType(),
        )
    ]


def test_infer_signature_keeps_parse_tuple_and_keywords_results_together() -> None:
    tuple_value_decl = _var_decl("tuple_value", _int_literal("0"))
    keywords_value_decl = _var_decl("keywords_value", _int_literal("0"))
    kwlist_decl = _var_decl("kwlist", _init_list(_string_literal("value"), _null_ptr_literal()))
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("i"),
            _address_of("tuple_value", referenced=tuple_value_decl),
        ),
        _call_expr(
            "PyArg_ParseTupleAndKeywords",
            _identifier_node("args"),
            _identifier_node("kwds"),
            _string_literal("i"),
            _token_identifier_node("kwlist", referenced=kwlist_decl),
            _address_of("keywords_value", referenced=keywords_value_decl),
        ),
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("tuple_value"))),
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_VARARGS | METH_KEYWORDS | METH_CLASS,
        owner_class=object,
    )

    assert inferred == [
        Signature(
            args=[
                _arg("cls", kind=ArgumentKind.POSITIONAL_ONLY),
                _arg("tuple_value", "int", kind=ArgumentKind.POSITIONAL_ONLY),
            ],
            return_type=RawType.int_,
        ),
        Signature(
            args=[_arg("cls"), _arg("value", "int")],
            return_type=RawType.int_,
        ),
    ]


def test_infer_signature_uses_fastcall_skeleton_with_receiver() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value")))
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_FASTCALL | METH_KEYWORDS,
        owner_class=object,
    )

    assert inferred == [
        Signature(
            args=[
                _arg("self"),
                _arg("args", "object", kind=ArgumentKind.VAR_POSITIONAL),
                _arg("kwargs", "object", kind=ArgumentKind.VAR_KEYWORD),
            ],
            return_type=RawType.int_,
        )
    ]


def test_infer_signature_uses_fastcall_skeleton_without_keywords() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value")))
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_FASTCALL,
        owner_class=object,
    )

    assert inferred == [
        Signature(
            args=[
                _arg("self"),
                _arg("args", "object", kind=ArgumentKind.VAR_POSITIONAL),
            ],
            return_type=RawType.int_,
        )
    ]


def test_infer_signature_uses_npy_parse_arguments_for_fastcall_keywords(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(signature_rules_module, "evaluate_cursor", lambda _: 0)
    obj_decl = _var_decl("obj")
    copy_decl = _var_decl("copy", _int_literal("0"))
    copy_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.INT)
    out_decl = _var_decl("out", _null_ptr_literal())
    out_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.POINTER)
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "npy_parse_arguments",
            _string_literal("argmax"),
            _address_of("__argparse_cache"),
            _identifier_node("args"),
            _identifier_node("len_args"),
            _identifier_node("kwnames"),
            _string_literal("obj"),
            _null_ptr_literal(),
            _address_of("obj", referenced=obj_decl),
            _string_literal("|copy"),
            _FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR, extent="PyArray_BoolConverter"),
            _address_of("copy", referenced=copy_decl),
            _string_literal("$out"),
            _FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR, extent="PyArray_OutputConverter"),
            _address_of("out", referenced=out_decl),
            _null_ptr_literal(),
            _null_ptr_literal(),
            _null_ptr_literal(),
        ),
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value"))),
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_FASTCALL | METH_KEYWORDS,
        owner_class=object,
    )

    assert inferred == [
        Signature(
            args=[
                _arg("self"),
                _arg("obj", "object"),
                _arg("copy", "bool", default_value="False"),
                _arg(
                    "out",
                    UnionType((RawType("numpy.ndarray", imports=("numpy",)), RawType.none_)),
                    default_value="...",
                    kind=ArgumentKind.KEYWORD_ONLY,
                ),
            ],
            return_type=RawType.int_,
        )
    ]


def test_infer_signature_uses_npy_parse_arguments_with_macro_expanded_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(signature_rules_module, "evaluate_cursor", lambda _: 0)
    shape_decl = _var_decl("shape")
    order_decl = _var_decl("order", _int_literal("0"))
    order_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.INT)
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "npy_parse_arguments",
            _string_literal("empty"),
            _address_of("__argparse_cache"),
            _identifier_node("args"),
            _identifier_node("len_args"),
            _identifier_node("kwnames"),
            _string_literal("shape"),
            _FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR, extent="PyArray_IntpConverter"),
            _address_of("shape", referenced=shape_decl),
            _string_literal("|order"),
            _FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR, extent="PyArray_BoolConverter"),
            _address_of("order", referenced=order_decl),
            _null_ptr_literal(),
            _null_ptr_literal(),
            _null_ptr_literal(),
        ),
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value"))),
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_FASTCALL | METH_KEYWORDS,
        owner_class=None,
    )

    assert inferred == [
        Signature(
            args=[
                _arg("shape", UnionType((RawType.int_, RawType("tuple[int, ...]")))),
                _arg("order", "bool", default_value="False"),
            ],
            return_type=RawType.int_,
        )
    ]


def test_infer_signature_uses_npy_parse_arguments_empty_name_as_positional_only() -> None:
    d1_decl = _var_decl("d1")
    d2_decl = _var_decl("d2")
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "npy_parse_arguments",
            _string_literal("promote_types"),
            _address_of("__argparse_cache"),
            _identifier_node("args"),
            _identifier_node("len_args"),
            _null_ptr_literal(),
            _string_literal(""),
            _null_ptr_literal(),
            _address_of("d1", referenced=d1_decl),
            _string_literal(""),
            _null_ptr_literal(),
            _address_of("d2", referenced=d2_decl),
            _null_ptr_literal(),
            _null_ptr_literal(),
            _null_ptr_literal(),
        ),
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value"))),
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_FASTCALL,
        owner_class=None,
    )

    assert inferred == [
        Signature(
            args=[
                _arg("d1", "object", kind=ArgumentKind.POSITIONAL_ONLY),
                _arg("d2", "object", kind=ArgumentKind.POSITIONAL_ONLY),
            ],
            return_type=RawType.int_,
        )
    ]


def test_infer_signature_uses_npy_parse_arguments_object_fallback_for_unknown_converter() -> None:
    axis_decl = _var_decl("axis")
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "npy_parse_arguments",
            _string_literal("take"),
            _address_of("__argparse_cache"),
            _identifier_node("args"),
            _identifier_node("len_args"),
            _identifier_node("kwnames"),
            _string_literal("axis"),
            _FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR, extent="UnknownConverter"),
            _address_of("axis", referenced=axis_decl),
            _null_ptr_literal(),
            _null_ptr_literal(),
            _null_ptr_literal(),
        ),
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value"))),
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_FASTCALL | METH_KEYWORDS,
        owner_class=None,
    )

    assert inferred == [
        Signature(
            args=[_arg("axis", "object")],
            return_type=RawType.int_,
        )
    ]


def test_infer_signature_accepts_renamed_npy_parse_arguments_inputs() -> None:
    value_decl = _var_decl("value")
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "npy_parse_arguments",
            _string_literal("einsum"),
            _address_of("__argparse_cache"),
            _identifier_node("positional_args"),
            _identifier_node("arg_count"),
            _identifier_node("keyword_names"),
            _string_literal("value"),
            _null_ptr_literal(),
            _address_of("value", referenced=value_decl),
            _null_ptr_literal(),
            _null_ptr_literal(),
            _null_ptr_literal(),
        ),
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value"))),
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_FASTCALL | METH_KEYWORDS,
        owner_class=object,
    )

    assert inferred == [
        Signature(
            args=[_arg("self"), _arg("value", "object")],
            return_type=RawType.int_,
        )
    ]


def test_infer_signature_maps_numpy_array_like_dtype_and_copy_converters() -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(signature_rules_module, "evaluate_cursor", lambda _: 0)
    prototype_decl = _var_decl("prototype")
    dt_info_decl = _var_decl("dt_info", _init_list(_null_ptr_literal(), _null_ptr_literal()))
    subok_decl = _var_decl("subok", _int_literal("0"))
    subok_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.INT)
    shape_decl = _var_decl("shape")
    device_decl = _var_decl("device", _identifier_node("NPY_DEVICE_CPU"))
    device_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.INT)
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "npy_parse_arguments",
            _string_literal("array"),
            _address_of("__argparse_cache"),
            _identifier_node("args"),
            _identifier_node("len_args"),
            _identifier_node("kwnames"),
            _string_literal("prototype"),
            _FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR, extent="PyArray_Converter"),
            _address_of("prototype", referenced=prototype_decl),
            _string_literal("|dtype"),
            _FakeNode(
                kind=clang.cindex.CursorKind.DECL_REF_EXPR,
                extent="PyArray_DTypeOrDescrConverterOptional",
            ),
            _address_of("dt_info", referenced=dt_info_decl),
            _string_literal("|subok"),
            _FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR, extent="PyArray_PythonPyIntFromInt"),
            _address_of("subok", referenced=subok_decl),
            _string_literal("|shape"),
            _FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR, extent="PyArray_OptionalIntpConverter"),
            _address_of("shape", referenced=shape_decl),
            _string_literal("$device"),
            _FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR, extent="PyArray_DeviceConverterOptional"),
            _address_of("device", referenced=device_decl),
            _null_ptr_literal(),
            _null_ptr_literal(),
            _null_ptr_literal(),
        ),
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value"))),
    )

    try:
        inferred = signature_rules_module.infer_signature(
            cursor,
            flags=METH_FASTCALL | METH_KEYWORDS,
            owner_class=None,
        )
    finally:
        monkeypatch.undo()

    assert inferred == [
        Signature(
            args=[
                _arg("prototype", RawType("numpy.typing.ArrayLike", imports=("numpy.typing",))),
                _arg(
                    "dtype",
                    UnionType((RawType("numpy.typing.DTypeLike", imports=("numpy.typing",)), RawType.none_)),
                    default_value="...",
                ),
                _arg("subok", "int", default_value="0"),
                _arg(
                    "shape",
                    UnionType((RawType.int_, RawType("tuple[int, ...]"), RawType.none_)),
                    default_value="...",
                ),
                _arg(
                    "device",
                    RawType('typing.Literal["cpu"] | None', imports=("typing",)),
                    default_value="...",
                    kind=ArgumentKind.KEYWORD_ONLY,
                ),
            ],
            return_type=RawType.int_,
        )
    ]


def test_infer_signature_maps_numpy_copy_converter_and_int_defaults() -> None:
    copy_decl = _var_decl("copy", _int_literal("1"))
    copy_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.INT)
    ndmin_decl = _var_decl("ndmin", _int_literal("0"))
    ndmin_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.INT)
    ndmax_decl = _var_decl("ndmax", _int_literal("64"))
    ndmax_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.INT)
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "npy_parse_arguments",
            _string_literal("array"),
            _address_of("__argparse_cache"),
            _identifier_node("args"),
            _identifier_node("len_args"),
            _identifier_node("kwnames"),
            _string_literal("$copy"),
            _FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR, extent="PyArray_CopyConverter"),
            _address_of("copy", referenced=copy_decl),
            _string_literal("$ndmin"),
            _FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR, extent="PyArray_PythonPyIntFromInt"),
            _address_of("ndmin", referenced=ndmin_decl),
            _string_literal("$ndmax"),
            _FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR, extent="PyArray_PythonPyIntFromInt"),
            _address_of("ndmax", referenced=ndmax_decl),
            _null_ptr_literal(),
            _null_ptr_literal(),
            _null_ptr_literal(),
        ),
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value"))),
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_FASTCALL | METH_KEYWORDS,
        owner_class=None,
    )

    assert inferred == [
        Signature(
            args=[
                _arg(
                    "copy",
                    UnionType((RawType.bool_, RawType('typing.Literal[False, True, 2]', imports=("typing",)), RawType.none_)),
                    default_value="...",
                    kind=ArgumentKind.KEYWORD_ONLY,
                ),
                _arg("ndmin", "int", default_value="...", kind=ArgumentKind.KEYWORD_ONLY),
                _arg("ndmax", "int", default_value="...", kind=ArgumentKind.KEYWORD_ONLY),
            ],
            return_type=RawType.int_,
        )
    ]


def test_infer_signature_maps_numpy_business_day_converters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(signature_rules_module, "evaluate_cursor", lambda _: 0)
    kwlist_decl = _var_decl(
        "kwlist",
        _init_list(
            _string_literal("dates"),
            _string_literal("offsets"),
            _string_literal("roll"),
            _string_literal("weekmask"),
            _string_literal("holidays"),
            _null_ptr_literal(),
        ),
    )
    roll_decl = _var_decl("roll", _identifier_node("NPY_BUSDAY_RAISE"))
    weekmask_decl = _var_decl("weekmask")
    holidays_decl = _var_decl("holidays", _null_ptr_literal())
    holidays_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.POINTER)
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTupleAndKeywords",
            _identifier_node("args"),
            _identifier_node("kwds"),
            _string_literal("OO|O&O&O&:busday_offset"),
            _token_identifier_node("kwlist", referenced=kwlist_decl),
            _identifier_node("dates_in"),
            _identifier_node("offsets_in"),
            _FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR, extent="PyArray_BusDayRollConverter"),
            _address_of("roll", referenced=roll_decl),
            _FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR, extent="PyArray_WeekMaskConverter"),
            _address_of("weekmask", referenced=weekmask_decl),
            _FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR, extent="PyArray_HolidaysConverter"),
            _address_of("holidays", referenced=holidays_decl),
        ),
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value"))),
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_VARARGS | METH_KEYWORDS,
        owner_class=None,
    )

    assert inferred == [
        Signature(
            args=[
                _arg("dates", "object"),
                _arg("offsets", "object"),
                _arg(
                    "roll",
                    RawType(
                        'typing.Literal["raise", "nat", "forward", "following", "backward", "preceding", "modifiedfollowing", "modifiedpreceding"]',
                        imports=("typing",),
                    ),
                    default_value="...",
                ),
                _arg("weekmask", RawType("numpy.typing.ArrayLike", imports=("numpy.typing",)), default_value="..."),
                _arg(
                    "holidays",
                    UnionType((RawType("numpy.typing.ArrayLike", imports=("numpy.typing",)), RawType.none_)),
                    default_value="...",
                ),
            ],
            return_type=RawType.int_,
        )
    ]


def test_infer_signature_maps_numpy_trim_converter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(signature_rules_module, "evaluate_cursor", lambda _: 0)
    trim_decl = _var_decl("trim")
    trim_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.INT)
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "npy_parse_arguments",
            _string_literal("format_float_positional"),
            _address_of("__argparse_cache"),
            _identifier_node("args"),
            _identifier_node("len_args"),
            _identifier_node("kwnames"),
            _string_literal("|trim"),
            _FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR, extent="trimmode_converter"),
            _address_of("trim", referenced=trim_decl),
            _null_ptr_literal(),
            _null_ptr_literal(),
            _null_ptr_literal(),
        ),
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value"))),
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_FASTCALL | METH_KEYWORDS,
        owner_class=None,
    )

    assert inferred == [
        Signature(
            args=[
                _arg(
                    "trim",
                    RawType('typing.Literal["k", ".", "0", "-"]', imports=("typing",)),
                    default_value="...",
                ),
            ],
            return_type=RawType.int_,
        )
    ]


def test_infer_signature_maps_numpy_errmode_converter() -> None:
    all_mode_decl = _var_decl("all_mode", _int_literal("-1"))
    all_mode_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.INT)
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "npy_parse_arguments",
            _string_literal("_seterrobj"),
            _address_of("__argparse_cache"),
            _identifier_node("args"),
            _identifier_node("len_args"),
            _identifier_node("kwnames"),
            _string_literal("$all"),
            _FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR, extent="errmodeconverter"),
            _address_of("all_mode", referenced=all_mode_decl),
            _null_ptr_literal(),
            _null_ptr_literal(),
            _null_ptr_literal(),
        ),
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value"))),
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_FASTCALL | METH_KEYWORDS,
        owner_class=None,
    )

    assert inferred == [
        Signature(
            args=[
                _arg(
                    "all",
                    RawType(
                        'typing.Literal["ignore", "warn", "raise", "call", "print", "log"] | None',
                        imports=("typing",),
                    ),
                    default_value="...",
                    kind=ArgumentKind.KEYWORD_ONLY,
                ),
            ],
            return_type=RawType.int_,
        )
    ]


def test_infer_signature_keeps_multiple_npy_parse_argument_lists() -> None:
    left_decl = _var_decl("left")
    right_decl = _var_decl("right")
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "npy_parse_arguments",
            _string_literal("func"),
            _address_of("__argparse_cache"),
            _identifier_node("args"),
            _identifier_node("len_args"),
            _identifier_node("kwnames"),
            _string_literal("left"),
            _null_ptr_literal(),
            _address_of("left", referenced=left_decl),
            _null_ptr_literal(),
            _null_ptr_literal(),
            _null_ptr_literal(),
        ),
        _call_expr(
            "npy_parse_arguments",
            _string_literal("func"),
            _address_of("__argparse_cache"),
            _identifier_node("args"),
            _identifier_node("len_args"),
            _identifier_node("kwnames"),
            _string_literal("right"),
            _null_ptr_literal(),
            _address_of("right", referenced=right_decl),
            _null_ptr_literal(),
            _null_ptr_literal(),
            _null_ptr_literal(),
        ),
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value"))),
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_FASTCALL | METH_KEYWORDS,
        owner_class=None,
    )

    assert inferred == [
        Signature(
            args=[_arg("left", "object")],
            return_type=RawType.int_,
        ),
        Signature(
            args=[_arg("right", "object")],
            return_type=RawType.int_,
        ),
    ]


def test_infer_object_type_from_call_ignores_non_decl_ref_refined_cursor() -> None:
    args_decl = _var_decl("args")
    target_decl = _var_decl("value")
    call_expr = _call_expr(
        "PyTuple_Check",
        _array_subscript("args", _int_literal("0"), referenced=args_decl),
    )
    inferencer = signature_rules_module.Inferencer(
        _fake_function_cursor_with_children(call_expr),
        0,
        None,
    )

    inferred = inferencer._infer_object_type_from_call(call_expr, target_decl)

    assert inferred is None


def test_infer_object_type_from_call_ignores_different_target_decl() -> None:
    value_decl = _var_decl("value")
    other_decl = _var_decl("other")
    call_expr = _call_expr(
        "PyUnicode_Check",
        _token_identifier_node("value", referenced=value_decl),
    )
    inferencer = signature_rules_module.Inferencer(
        _fake_function_cursor_with_children(call_expr),
        0,
        None,
    )

    inferred = inferencer._infer_object_type_from_call(call_expr, other_decl)

    assert inferred is None


def test_infer_signature_keeps_pyarg_tuple_lists_with_unrelated_array_subscript_check() -> None:
    args_decl = _var_decl("args")
    left_decl = _var_decl("left")
    right_decl = _var_decl("right")
    only_decl = _var_decl("only")
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyTuple_Check",
            _array_subscript("args", _int_literal("0"), referenced=args_decl),
        ),
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("(OO)"),
            _address_of("left", referenced=left_decl),
            _address_of("right", referenced=right_decl),
        ),
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("(O)"),
            _address_of("only", referenced=only_decl),
        ),
        _return_stmt(_token_identifier_node("_Py_NoneStruct")),
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_VARARGS,
        owner_class=None,
    )

    assert inferred == [
        Signature(
            args=[
                _arg(
                    "left_right",
                    RawType("tuple[object, object]"),
                    kind=ArgumentKind.POSITIONAL_ONLY,
                )
            ],
            return_type=RawType.none_,
        ),
        Signature(
            args=[
                _arg(
                    "only",
                    RawType("tuple[object,]"),
                    kind=ArgumentKind.POSITIONAL_ONLY,
                )
            ],
            return_type=RawType.none_,
        ),
    ]


def test_infer_signature_keeps_npy_parse_keyword_arguments_with_unrelated_array_subscript_check() -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(signature_rules_module, "evaluate_cursor", lambda _: 0)
    args_decl = _var_decl("args")
    out_obj_decl = _var_decl("out_obj", _null_ptr_literal())
    out_obj_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.POINTER)
    order_decl = _var_decl("order", _identifier_node("NPY_KEEPORDER"))
    order_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.INT)
    casting_decl = _var_decl("casting", _identifier_node("NPY_SAFE_CASTING"))
    casting_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.INT)
    dtype_decl = _var_decl("dtype", _null_ptr_literal())
    dtype_decl.type = _FakeCanonicalType(clang.cindex.TypeKind.POINTER)
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyBytes_Check",
            _array_subscript("args", _int_literal("0"), referenced=args_decl),
        ),
        _call_expr(
            "npy_parse_arguments",
            _string_literal("einsum"),
            _address_of("__argparse_cache"),
            _identifier_node("args"),
            _identifier_node("len_args"),
            _identifier_node("kwnames"),
            _string_literal("$out"),
            _null_ptr_literal(),
            _address_of("out_obj", referenced=out_obj_decl),
            _string_literal("$order"),
            _FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR, extent="PyArray_OrderConverter"),
            _address_of("order", referenced=order_decl),
            _string_literal("$casting"),
            _FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR, extent="PyArray_CastingConverter"),
            _address_of("casting", referenced=casting_decl),
            _string_literal("$dtype"),
            _FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR, extent="PyArray_DescrConverter2"),
            _address_of("dtype", referenced=dtype_decl),
            _null_ptr_literal(),
            _null_ptr_literal(),
            _null_ptr_literal(),
        ),
        _call_expr(
            "PyArray_Check",
            _token_identifier_node("out_obj", referenced=out_obj_decl),
        ),
        _return_stmt(_token_identifier_node("_Py_NoneStruct")),
    )

    try:
        inferred = signature_rules_module.infer_signature(
            cursor,
            flags=METH_FASTCALL | METH_KEYWORDS,
            owner_class=None,
        )
    finally:
        monkeypatch.undo()

    assert inferred == [
        Signature(
            args=[
                _arg(
                    "out",
                    RawType("numpy.ndarray", imports=("numpy",)),
                    default_value="...",
                    kind=ArgumentKind.KEYWORD_ONLY,
                ),
                _arg(
                    "order",
                    UnionType((
                        RawType('typing.Literal["K", "A", "C", "F"]', imports=("typing",)),
                        RawType.none_,
                    )),
                    default_value="...",
                    kind=ArgumentKind.KEYWORD_ONLY,
                ),
                _arg(
                    "casting",
                    RawType(
                        'typing.Literal["no", "equiv", "safe", "same_kind", "unsafe"]',
                        imports=("typing",),
                    ),
                    default_value="...",
                    kind=ArgumentKind.KEYWORD_ONLY,
                ),
                _arg(
                    "dtype",
                    UnionType((RawType("numpy.typing.DTypeLike", imports=("numpy.typing",)), RawType.none_)),
                    default_value="...",
                    kind=ArgumentKind.KEYWORD_ONLY,
                ),
            ],
            return_type=RawType.none_,
        )
    ]


def test_infer_signature_keeps_npy_parse_arguments_with_unrelated_member_ref_check() -> None:
    obj_decl = _var_decl("obj")
    str_decl = _var_decl("str")
    member_ref = _FakeNode(
        kind=clang.cindex.CursorKind.MEMBER_REF_EXPR,
        spelling="tp_dict",
        tokens=[],
    )
    cursor = _fake_function_cursor_with_children(
        _call_expr("PyDict_CheckExact", member_ref),
        _call_expr(
            "npy_parse_arguments",
            _string_literal("add_docstring"),
            _address_of("__argparse_cache"),
            _identifier_node("args"),
            _identifier_node("len_args"),
            _null_ptr_literal(),
            _string_literal(""),
            _null_ptr_literal(),
            _address_of("obj", referenced=obj_decl),
            _string_literal(""),
            _null_ptr_literal(),
            _address_of("str", referenced=str_decl),
            _null_ptr_literal(),
            _null_ptr_literal(),
            _null_ptr_literal(),
        ),
        _call_expr(
            "PyUnicode_Check",
            _token_identifier_node("str", referenced=str_decl),
        ),
        _return_stmt(_token_identifier_node("_Py_NoneStruct")),
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_FASTCALL,
        owner_class=None,
    )

    assert inferred == [
        Signature(
            args=[
                _arg("obj", "object", kind=ArgumentKind.POSITIONAL_ONLY),
                _arg("str", "str", kind=ArgumentKind.POSITIONAL_ONLY),
            ],
            return_type=RawType.none_,
        )
    ]


def _parse_function_cursor_with_python_headers(
    source_path: Path,
) -> clang.cindex.Cursor:
    resource_dir = subprocess.check_output(
        ["clang", "-print-resource-dir"],
        text=True,
    ).strip()
    parse_args = [
        "-x",
        "c",
        "--std=c11",
        "-I",
        sysconfig.get_path("include"),
        "-resource-dir",
        resource_dir,
    ]

    translation_unit = clang.cindex.Index.create().parse(str(source_path), args=parse_args)
    return next(
        cursor
        for cursor in translation_unit.cursor.get_children()
        if cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL and cursor.spelling == "demo"
    )


@pytest.mark.libclang
def test_infer_signature_refines_meth_o_argument_for_pyarray_check_exact_macro(
    tmp_path: Path,
) -> None:
    source = tmp_path / "macro_exact_signature.c"
    source.write_text(
        "\n".join(
            [
                "#include <Python.h>",
                "int fake_pyarray_check_exact(PyObject *obj);",
                "#define PyArray_CheckExact(obj) fake_pyarray_check_exact(obj)",
                "PyObject *PyLong_FromLong(long);",
                "PyObject *demo(PyObject *self, PyObject *arg) {",
                "    if (PyArray_CheckExact(arg)) {",
                "        return PyLong_FromLong(1);",
                "    }",
                "    return PyLong_FromLong(0);",
                "}",
            ]
        ),
        encoding="utf-8",
    )

    func_cursor = _parse_function_cursor_with_python_headers(source)

    inferred = signature_rules_module.infer_signature(
        func_cursor,
        flags=METH_O,
        owner_class=object,
    )

    assert inferred == [
        Signature(
            args=[
                _arg("self", kind=ArgumentKind.POSITIONAL_ONLY),
                _arg(
                    "arg",
                    RawType("numpy.ndarray", imports=("numpy",)),
                    kind=ArgumentKind.POSITIONAL_ONLY,
                ),
            ],
            return_type=RawType.int_,
        )
    ]
