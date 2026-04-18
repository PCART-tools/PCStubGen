from __future__ import annotations

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
from pcstubgen.type_models import AnyType, RawType
from tests._c_extension_test_support import (
    _address_of,
    _arg,
    _call_expr,
    _fake_function_cursor_with_children,
    _identifier_node,
    _int_literal,
    _macro_expr,
    _return_stmt,
    _string_literal,
    _var_decl,
    patch_inference_clang_helpers,
)


@pytest.fixture(autouse=True)
def _patch_fake_clang_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_inference_clang_helpers(monkeypatch, signature_rules_module)


def test_infer_signature_uses_minimal_signature_when_parse_tuple_is_skipped() -> None:
    invalid_slot = _int_literal("0")
    cursor = _fake_function_cursor_with_children(
        _call_expr(
            "PyArg_ParseTuple",
            _identifier_node("args"),
            _string_literal("i"),
            invalid_slot,
        )
    )

    inferred = signature_rules_module.infer_signature(cursor, flags=METH_VARARGS)

    assert inferred == [
        Signature(
            args=[
                _arg(
                    "args",
                    "object",
                    kind=ArgumentKind.VAR_POSITIONAL,
                )
            ],
            return_type=AnyType(),
        )
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

