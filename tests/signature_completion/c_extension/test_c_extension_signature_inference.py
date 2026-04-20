from __future__ import annotations

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
from pcstubgen.type_models import AnyType, RawType
from tests._c_extension_test_support import (
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
