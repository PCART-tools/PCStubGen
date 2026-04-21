from __future__ import annotations

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
from pcstubgen.signature_completion.c_extension.signatures import inferencer as signature_rules_module
from pcstubgen.type_models import AnyType, RawType, UnionType
from tests._c_extension_test_support import (
    _FakeCanonicalType,
    _FakeNode,
    _address_of,
    _arg,
    _call_expr,
    _fake_function_cursor_with_children,
    _init_list,
    _identifier_node,
    _int_literal,
    _null_ptr_literal,
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

    assert inferred == [Signature(args=[], return_type=RawType("int"))]


def test_infer_signature_inserts_self_for_method_meth_noargs() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value")))
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_NOARGS,
        is_method=True,
    )

    assert inferred == [Signature(args=[_arg("self")], return_type=RawType("int"))]


def test_infer_signature_inserts_cls_for_classmethod_meth_noargs() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value")))
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_NOARGS | METH_CLASS,
        is_method=True,
    )

    assert inferred == [Signature(args=[_arg("cls")], return_type=RawType("int"))]


def test_infer_signature_skips_receiver_for_staticmethod_meth_noargs() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value")))
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_NOARGS | METH_STATIC,
        is_method=True,
    )

    assert inferred == [Signature(args=[], return_type=RawType("int"))]


def test_infer_signature_inserts_self_for_method_meth_o() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value")))
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_O,
        is_method=True,
    )

    assert inferred == [
        Signature(
            args=[
                _arg("self", kind=ArgumentKind.POSITIONAL_ONLY),
                _arg("arg", "object", kind=ArgumentKind.POSITIONAL_ONLY),
            ],
            return_type=RawType("int"),
        )
    ]


def test_infer_signature_inserts_cls_for_classmethod_meth_o() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value")))
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_O | METH_CLASS,
        is_method=True,
    )

    assert inferred == [
        Signature(
            args=[
                _arg("cls", kind=ArgumentKind.POSITIONAL_ONLY),
                _arg("arg", "object", kind=ArgumentKind.POSITIONAL_ONLY),
            ],
            return_type=RawType("int"),
        )
    ]


def test_infer_signature_skips_receiver_for_staticmethod_meth_o() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value")))
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_O | METH_STATIC,
        is_method=True,
    )

    assert inferred == [
        Signature(
            args=[_arg("arg", "object", kind=ArgumentKind.POSITIONAL_ONLY)],
            return_type=RawType("int"),
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
        is_method=True,
    )

    assert inferred == [
        Signature(
            args=[
                _arg("cls", kind=ArgumentKind.POSITIONAL_ONLY),
                _arg("arg", "object", kind=ArgumentKind.POSITIONAL_ONLY),
            ],
            return_type=RawType("int"),
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
        is_method=True,
    )

    assert inferred == [
        Signature(
            args=[_arg("self"), _arg("value", "int")],
            return_type=RawType("int"),
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
        is_method=True,
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
        is_method=True,
    )

    assert inferred == [
        Signature(
            args=[_arg("cls"), _arg("value", "int")],
            return_type=RawType("int"),
        )
    ]


def test_infer_signature_falls_back_to_varargs_and_keywords() -> None:
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
        is_method=True,
    )

    assert inferred == [
        Signature(
            args=[
                _arg("cls"),
                _arg("args", "object", kind=ArgumentKind.VAR_POSITIONAL),
                _arg("kwargs", "object", kind=ArgumentKind.VAR_KEYWORD),
            ],
            return_type=AnyType(),
        )
    ]


def test_infer_signature_uses_fastcall_skeleton_with_receiver() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value")))
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_FASTCALL | METH_KEYWORDS,
        is_method=True,
    )

    assert inferred == [
        Signature(
            args=[
                _arg("self"),
                _arg("args", "object", kind=ArgumentKind.VAR_POSITIONAL),
                _arg("kwargs", "object", kind=ArgumentKind.VAR_KEYWORD),
            ],
            return_type=RawType("int"),
        )
    ]


def test_infer_signature_uses_fastcall_skeleton_without_keywords() -> None:
    cursor = _fake_function_cursor_with_children(
        _return_stmt(_call_expr("PyLong_FromLong", _identifier_node("value")))
    )

    inferred = signature_rules_module.infer_signature(
        cursor,
        flags=METH_FASTCALL,
        is_method=True,
    )

    assert inferred == [
        Signature(
            args=[
                _arg("self"),
                _arg("args", "object", kind=ArgumentKind.VAR_POSITIONAL),
            ],
            return_type=RawType("int"),
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
        is_method=True,
    )

    assert inferred == [
        Signature(
            args=[
                _arg("self"),
                _arg("obj", "object"),
                _arg("copy", "bool", default_value="False"),
                _arg(
                    "out",
                    UnionType((RawType("numpy.ndarray", imports=("numpy",)), RawType("None"))),
                    default_value="...",
                    kind=ArgumentKind.KEYWORD_ONLY,
                ),
            ],
            return_type=RawType("int"),
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
        is_method=False,
    )

    assert inferred == [
        Signature(
            args=[
                _arg("shape", "tuple[int, ...]"),
                _arg("order", "bool", default_value="False"),
            ],
            return_type=RawType("int"),
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
        is_method=False,
    )

    assert inferred == [
        Signature(
            args=[
                _arg("d1", "object", kind=ArgumentKind.POSITIONAL_ONLY),
                _arg("d2", "object", kind=ArgumentKind.POSITIONAL_ONLY),
            ],
            return_type=RawType("int"),
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
        is_method=False,
    )

    assert inferred == [
        Signature(
            args=[_arg("axis", "object")],
            return_type=RawType("int"),
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
        is_method=True,
    )

    assert inferred == [
        Signature(
            args=[_arg("self"), _arg("value", "object")],
            return_type=RawType("int"),
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
        is_method=False,
    )

    assert inferred == [
        Signature(
            args=[_arg("left", "object")],
            return_type=RawType("int"),
        ),
        Signature(
            args=[_arg("right", "object")],
            return_type=RawType("int"),
        ),
    ]
