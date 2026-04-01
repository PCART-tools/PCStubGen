from __future__ import annotations

from tests._c_extension_test_support import *


def test_c_signature_resolver_preserves_extracted_fields_and_source_comment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "foo_impl.c"
    snippet = "\n".join(
        [
            "static PyObject* foo_impl(PyObject* self, PyObject* args) {",
            "    return (PyObject*)0;",
            "}",
        ]
    )
    source.write_text(snippet, encoding="utf-8", newline="\n")
    func_cursor = cast(
        clang.cindex.Cursor,
        _FakeNode(
            kind=clang.cindex.CursorKind.FUNCTION_DECL,
            spelling="foo_impl",
            extent=_extent_for_source_snippet(source, snippet),
        ),
    )
    _patch_c_signature_extractor(
        monkeypatch,
        modules=_module_fixture(
            functions={
                "foo": ExtractedFunction(
                    ml_name="foo",
                    function_cursor=func_cursor,
                    ml_flags=METH_VARARGS,
                    signatures=[
                        ExtractedSignature(
                            arguments=[
                                _arg("self", "object"),
                                _arg("x", "int"),
                                _arg("flag", "bool", default_value="False", has_default=True),
                            ],
                            return_type=RawType("int"),
                        )
                    ],
                )
            }
        ),
    )
    resolver = CSignatureResolver(source_root=tmp_path)
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo")],
    )

    resolved = resolver.resolve_function(module=module, func=module.functions[0])

    assert resolved is not None
    signature = resolved.signatures[0]
    assert [arg.name for arg in signature.arguments] == ["self", "x", "flag"]
    assert signature.arguments[0].type is not None and signature.arguments[0].type.render() == "object"
    assert signature.arguments[1].type is not None and signature.arguments[1].type.render() == "int"
    assert signature.arguments[2].type is not None and signature.arguments[2].type.render() == "bool"
    assert signature.arguments[2].default_value == "False"
    assert signature.arguments[2].has_default is True
    assert signature.return_type is not None
    assert signature.return_type.render() == "int"
    assert resolved.c_inferred_source_comment == snippet


def test_c_signature_resolver_skips_missing_extent_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_c_signature_extractor(
        monkeypatch,
        modules=_module_fixture(
            functions={
                "foo": ExtractedFunction(
                    ml_name="foo",
                    function_cursor=_fake_function_cursor("foo"),
                    ml_flags=METH_VARARGS,
                    signatures=[ExtractedSignature(arguments=[_arg("value", "int")])],
                )
            }
        ),
    )
    resolver = CSignatureResolver(source_root=tmp_path)
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo")],
    )

    resolved = resolver.resolve_function(module=module, func=module.functions[0])

    assert resolved is not None
    assert resolved.c_inferred_source_comment is None


def test_c_signature_resolver_returns_none_for_methods_and_python_modules(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_c_signature_extractor(
        monkeypatch,
        modules=_module_fixture(
            functions={
                "foo": ExtractedFunction(
                    ml_name="foo",
                    function_cursor=_fake_function_cursor("foo"),
                    ml_flags=METH_VARARGS,
                    signatures=[ExtractedSignature(arguments=[_arg("x", "int")])],
                )
            }
        ),
    )
    resolver = CSignatureResolver(source_root=tmp_path)
    extension_module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo")],
    )
    python_module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.PYTHON,
        functions=[_unknown_function("foo")],
    )

    assert (
        resolver.resolve_function(
            module=extension_module,
            func=extension_module.functions[0],
            is_method=True,
        )
        is None
    )
    assert resolver.resolve_function(module=python_module, func=python_module.functions[0]) is None


def test_c_signature_resolver_matches_exact_module_before_leaf_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_c_signature_extractor(
        monkeypatch,
        modules={
            "pkg.first": ExtractedModule(
                name="pkg.first",
                functions={
                    "foo": ExtractedFunction(
                        ml_name="foo",
                        function_cursor=_fake_function_cursor("foo"),
                        ml_flags=METH_VARARGS,
                        signatures=[ExtractedSignature(arguments=[_arg("x", "int")])],
                    )
                },
            ),
            "pkg.second": ExtractedModule(
                name="pkg.second",
                functions={
                    "foo": ExtractedFunction(
                        ml_name="foo",
                        function_cursor=_fake_function_cursor("foo"),
                        ml_flags=METH_VARARGS,
                        signatures=[ExtractedSignature(arguments=[_arg("value", "float")])],
                    )
                },
            ),
        },
    )
    resolver = CSignatureResolver(source_root=tmp_path)
    first_module = IRModule(
        full_name=QualifiedName.from_str("pkg.first"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo")],
    )
    second_module = IRModule(
        full_name=QualifiedName.from_str("pkg.second"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo")],
    )

    first = resolver.resolve_function(module=first_module, func=first_module.functions[0])
    second = resolver.resolve_function(module=second_module, func=second_module.functions[0])

    assert first is not None
    assert second is not None
    assert first.signatures[0].arguments[0].name == "x"
    assert first.signatures[0].arguments[0].type is not None
    assert first.signatures[0].arguments[0].type.render() == "int"
    assert second.signatures[0].arguments[0].name == "value"
    assert second.signatures[0].arguments[0].type is not None
    assert second.signatures[0].arguments[0].type.render() == "float"


def test_c_signature_resolver_falls_back_to_unique_leaf_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_c_signature_extractor(
        monkeypatch,
        modules={
            "mod": ExtractedModule(
                name="mod",
                functions={
                    "foo": ExtractedFunction(
                        ml_name="foo",
                        function_cursor=_fake_function_cursor("foo"),
                        ml_flags=METH_VARARGS,
                        signatures=[ExtractedSignature(arguments=[_arg("value", "float")])],
                    )
                },
            ),
        },
    )
    resolver = CSignatureResolver(source_root=tmp_path)
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo")],
    )

    resolved = resolver.resolve_function(module=module, func=module.functions[0])

    assert resolved is not None
    assert resolved.signatures[0].arguments[0].name == "value"
    assert resolved.signatures[0].arguments[0].type is not None
    assert resolved.signatures[0].arguments[0].type.render() == "float"


def test_c_signature_resolver_returns_none_for_ambiguous_leaf_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_c_signature_extractor(
        monkeypatch,
        modules={
            "one": ExtractedModule(
                name="mod",
                functions={
                    "foo": ExtractedFunction(
                        ml_name="foo",
                        function_cursor=_fake_function_cursor("foo"),
                        ml_flags=METH_VARARGS,
                        signatures=[ExtractedSignature(arguments=[_arg("x", "int")])],
                    )
                },
            ),
            "two": ExtractedModule(
                name="mod",
                functions={
                    "foo": ExtractedFunction(
                        ml_name="foo",
                        function_cursor=_fake_function_cursor("foo"),
                        ml_flags=METH_VARARGS,
                        signatures=[ExtractedSignature(arguments=[_arg("y", "float")])],
                    )
                },
            ),
        },
    )
    resolver = CSignatureResolver(source_root=tmp_path)
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo")],
    )

    assert resolver.resolve_function(module=module, func=module.functions[0]) is None


def test_c_signature_resolver_returns_none_when_function_is_missing_or_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_c_signature_extractor(
        monkeypatch,
        modules=_module_fixture(
            functions={
                "bar": ExtractedFunction(
                    ml_name="bar",
                    function_cursor=_fake_function_cursor("bar"),
                    ml_flags=METH_VARARGS,
                    signatures=[ExtractedSignature(arguments=[_arg("x", "int")])],
                ),
                "baz": ExtractedFunction(
                    ml_name="baz",
                    function_cursor=_fake_function_cursor("baz"),
                    ml_flags=METH_VARARGS,
                    signatures=[],
                ),
            }
        ),
    )
    resolver = CSignatureResolver(source_root=tmp_path)
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo"), _unknown_function("baz")],
    )

    assert resolver.resolve_function(module=module, func=module.functions[0]) is None
    assert resolver.resolve_function(module=module, func=module.functions[1]) is None


def test_c_signature_resolver_propagates_extraction_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_raising_c_signature_extractor(monkeypatch, RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        CSignatureResolver(source_root=tmp_path)


def test_c_signature_resolver_passes_clang_options_to_extractor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _record_collect_modules(
        source_root: Path,
        *,
        include: list[str] = (),
        include_directory: list[Path] = (),
        c_std: str = "c11",
        cpp_std: str = "c++17",
    ) -> dict[str, ExtractedModule]:
        captured["source_root"] = source_root
        captured["include"] = list(include)
        captured["include_directory"] = list(include_directory)
        captured["c_std"] = c_std
        captured["cpp_std"] = cpp_std
        return {}

    import pcstubgen.signature_completion.c_extension.source as resolver_module

    monkeypatch.setattr(
        resolver_module,
        "collect_modules",
        _record_collect_modules,
    )

    CSignatureResolver(
        source_root=tmp_path,
        include=["Python.h"],
        include_directory=[Path("C:/MyInclude")],
        c_std="c99",
        cpp_std="c++20",
    )

    assert captured["source_root"] == tmp_path
    assert captured["include"] == ["Python.h"]
    assert captured["include_directory"] == [Path("C:/MyInclude")]
    assert captured["c_std"] == "c99"
    assert captured["cpp_std"] == "c++20"

