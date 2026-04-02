from __future__ import annotations

from tests._c_extension_test_support import *


def test_c_signature_engine_resolve_init_list_expr_supports_positional_entries(tmp_path: Path) -> None:
    field_names = ("a", "b", "c")
    first = _string_literal("first")
    second = _int_literal("2")

    resolved = _resolve_INIT_LIST_EXPR(_init_list(first, second), field_names)

    assert resolved == {"a": first, "b": second}


def test_c_signature_engine_resolve_init_list_expr_supports_designated_entries(tmp_path: Path) -> None:
    field_names = ("a", "b", "c")
    second = _int_literal("2")
    third = _string_literal("third")

    resolved = _resolve_INIT_LIST_EXPR(
        _init_list(
            _designated_initializer("b", second),
            _designated_initializer("c", third),
        ),
        field_names,
    )

    assert resolved == {"b": second, "c": third}


def test_c_signature_engine_resolve_init_list_expr_supports_mixed_entries(tmp_path: Path) -> None:
    field_names = ("a", "b", "c", "d")
    first = _string_literal("first")
    third = _string_literal("third")
    fourth = _int_literal("4")

    resolved = _resolve_INIT_LIST_EXPR(
        _init_list(
            first,
            _designated_initializer("c", third),
            fourth,
        ),
        field_names,
    )

    assert resolved == {"a": first, "c": third, "d": fourth}


def test_c_signature_engine_resolve_init_list_expr_advances_positional_index_after_designated(
    tmp_path: Path,
) -> None:
    field_names = ("a", "b", "c", "d")
    second = _string_literal("second")
    third = _int_literal("3")
    fourth = _int_literal("4")

    resolved = _resolve_INIT_LIST_EXPR(
        _init_list(
            _designated_initializer("b", second),
            third,
            fourth,
        ),
        field_names,
    )

    assert resolved == {"b": second, "c": third, "d": fourth}


def test_c_signature_engine_resolve_init_list_expr_ignores_unknown_designated_field(tmp_path: Path) -> None:
    field_names = ("a", "b", "c")
    unknown = _string_literal("skip")
    first = _int_literal("1")

    resolved = _resolve_INIT_LIST_EXPR(
        _init_list(
            _designated_initializer("missing", unknown),
            first,
        ),
        field_names,
    )

    assert resolved == {"a": first}


def test_c_signature_engine_resolve_init_list_expr_last_value_wins_for_duplicate_field(tmp_path: Path) -> None:
    field_names = ("a", "b", "c")
    first = _int_literal("1")
    second = _int_literal("2")

    resolved = _resolve_INIT_LIST_EXPR(
        _init_list(
            first,
            _designated_initializer("a", second),
        ),
        field_names,
    )

    assert resolved == {"a": second}


def test_c_signature_engine_resolve_init_list_expr_keeps_nested_init_list_as_value(tmp_path: Path) -> None:
    field_names = ("a", "b")
    nested = _init_list(_int_literal("1"), _int_literal("2"))

    resolved = _resolve_INIT_LIST_EXPR(
        _init_list(_designated_initializer("b", nested)),
        field_names,
    )

    assert resolved == {"b": nested}


def test_c_signature_engine_extracts_pymethod_fields_from_ast_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_fake_eval_int(monkeypatch)
    is_sentinel, extracted = _collect_PyMethodDef_INIT_LIST_EXPR(
        init_list_expr=_init_list(
            _ml_name_field("add"),
            _ml_meth_field("simple_add"),
            _ml_flags_identifier_field("METH_VARARGS"),
            _string_literal("doc"),
        ),
    )

    assert is_sentinel is False
    assert extracted is not None
    assert extracted.ml_name == "add"
    assert extracted.ml_flags == METH_VARARGS
    assert extracted.function_cursor is not None
    assert extracted.signatures == []


def test_c_signature_engine_extracts_cast_wrapped_ml_meth_from_ast(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_fake_eval_int(monkeypatch)
    is_sentinel, extracted = _collect_PyMethodDef_INIT_LIST_EXPR(
        init_list_expr=_init_list(
            _ml_name_field("distance"),
            _ml_meth_cast_field("Point_distance"),
            _ml_flags_identifier_field("METH_VARARGS"),
            _string_literal("doc"),
        ),
    )

    assert is_sentinel is False
    assert extracted is not None
    assert extracted.ml_name == "distance"
    assert extracted.ml_flags == METH_VARARGS
    assert extracted.function_cursor is not None


def test_c_signature_engine_extracts_combined_flags_from_ast_field(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_fake_eval_int(monkeypatch)
    is_sentinel, extracted = _collect_PyMethodDef_INIT_LIST_EXPR(
        init_list_expr=_init_list(
            _ml_name_field("kw"),
            _ml_meth_field("kw_impl"),
            _ml_flags_identifier_field("METH_VARARGS", "METH_KEYWORDS"),
            _string_literal("doc"),
        ),
    )

    assert is_sentinel is False
    assert extracted is not None
    assert extracted.ml_flags == (METH_VARARGS | METH_KEYWORDS)


def test_c_signature_engine_keeps_empty_flags_when_ast_field_is_unparseable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_eval_int(monkeypatch)
    is_sentinel, extracted = _collect_PyMethodDef_INIT_LIST_EXPR(
        init_list_expr=_init_list(
            _ml_name_field("add"),
            _ml_meth_field("simple_add"),
            _identifier_node("flag_var"),
            _string_literal("doc"),
        ),
    )

    assert is_sentinel is False
    assert extracted is not None
    assert extracted.ml_flags == 0


def test_c_signature_engine_extract_pymethoddef_init_list_expr_marks_sentinel(tmp_path: Path) -> None:
    is_sentinel, extracted = _collect_PyMethodDef_INIT_LIST_EXPR(
        init_list_expr=_init_list(_null_ptr_literal()),
    )

    assert is_sentinel is True
    assert extracted is None


def test_c_signature_engine_extract_pymethoddef_init_list_expr_discards_entry_without_function_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证缺失 `ml_meth` 引用时当前条目会被直接丢弃。"""
    _patch_fake_eval_int(monkeypatch)
    is_sentinel, extracted = _collect_PyMethodDef_INIT_LIST_EXPR(
        init_list_expr=_init_list(
            _ml_name_field("missing"),
            _FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR),
            _ml_flags_identifier_field("METH_VARARGS"),
            _string_literal("doc"),
        ),
    )

    assert is_sentinel is False
    assert extracted is None


def test_c_signature_engine_extract_method_table_stops_at_sentinel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    method_1 = _init_list(
        _FakeNode(kind=clang.cindex.CursorKind.STRING_LITERAL, tokens=[_FakeToken(clang.cindex.TokenKind.LITERAL, '"a"')]),
        _FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR),
        _int_literal("1"),
        _FakeNode(kind=clang.cindex.CursorKind.STRING_LITERAL, tokens=[_FakeToken(clang.cindex.TokenKind.LITERAL, '"doc"')]),
    )
    method_2 = _init_list(
        _FakeNode(kind=clang.cindex.CursorKind.STRING_LITERAL, tokens=[_FakeToken(clang.cindex.TokenKind.LITERAL, '"b"')]),
        _FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR),
        _int_literal("1"),
        _FakeNode(kind=clang.cindex.CursorKind.STRING_LITERAL, tokens=[_FakeToken(clang.cindex.TokenKind.LITERAL, '"doc"')]),
    )
    supported_sentinel = _init_list(_null_ptr_literal())
    non_sentinel = _init_list(_identifier_node("nullptr"))
    calls: list[_FakeNode] = []

    def fake_extract(
        *,
        init_list_expr: _FakeNode,
        definition_resolver: DefinitionResolver,
    ) -> tuple[bool, SimpleNamespace | None]:
        _ = definition_resolver
        calls.append(init_list_expr)
        if init_list_expr is supported_sentinel:
            return True, None
        return False, SimpleNamespace(ml_name=f"entry_{len(calls)}")

    monkeypatch.setattr(module_table_module, "collect_pymethoddef_init_list_expr", fake_extract)
    monkeypatch.setattr(module_table_module, "is_PyMethodDef_array_definition", lambda cursor: True)

    should_break_array = _FakeNode(
        kind=clang.cindex.CursorKind.VAR_DECL,
        children=[
            _FakeNode(
                kind=clang.cindex.CursorKind.INIT_LIST_EXPR,
                children=[method_1, supported_sentinel, method_2],
            ),
        ],
    )
    output = _collect_method_table(
        should_break_array,
        module_name="pkg.mod",
    )
    assert calls == [method_1, supported_sentinel]
    assert list(output) == ["entry_1"]

    calls.clear()
    output.clear()

    should_not_break_array = _FakeNode(
        kind=clang.cindex.CursorKind.VAR_DECL,
        children=[
            _FakeNode(
                kind=clang.cindex.CursorKind.INIT_LIST_EXPR,
                children=[method_1, non_sentinel, method_2],
            ),
        ],
    )
    output = _collect_method_table(
        should_not_break_array,
        module_name="pkg.mod",
    )
    assert calls == [method_1, non_sentinel, method_2]
    assert list(output) == ["entry_1", "entry_2", "entry_3"]

