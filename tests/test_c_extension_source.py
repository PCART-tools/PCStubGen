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
    resolver = CSignatureResolver(source=tmp_path)
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo")],
    )

    resolved = resolver.resolve_function(module, module.functions[0])

    assert resolved is not None
    signatures, source_comment = resolved
    signature = signatures[0]
    assert [arg.name for arg in signature.args] == ["self", "x", "flag"]
    assert signature.args[0].type is not None and signature.args[0].type.render() == "object"
    assert signature.args[1].type is not None and signature.args[1].type.render() == "int"
    assert signature.args[2].type is not None and signature.args[2].type.render() == "bool"
    assert signature.args[2].default_value == "False"
    assert signature.args[2].has_default is True
    assert signature.return_type is not None
    assert signature.return_type.render() == "int"
    assert source_comment == snippet


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
    resolver = CSignatureResolver(source=tmp_path)
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo")],
    )

    resolved = resolver.resolve_function(module, module.functions[0])

    assert resolved is not None
    _, source_comment = resolved
    assert source_comment is None


def test_c_signature_resolver_accepts_zero_argument_signature(
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
                    ml_flags=METH_NOARGS,
                    signatures=[ExtractedSignature(arguments=[])],
                )
            }
        ),
    )
    resolver = CSignatureResolver(source=tmp_path)
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo")],
    )

    resolved = resolver.resolve_function(module, module.functions[0])

    assert resolved is not None
    signatures, _ = resolved
    assert signatures == [IRSignature(args=[])]


def test_c_signature_resolver_raises_for_methods_and_python_modules(
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
    resolver = CSignatureResolver(source=tmp_path)
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

    with pytest.raises(RuntimeError, match="暂不支持方法"):
        resolver.resolve_function(
            extension_module,
            extension_module.functions[0],
            is_method=True,
        )

    with pytest.raises(RuntimeError, match="不是扩展模块"):
        resolver.resolve_function(python_module, python_module.functions[0])


def test_c_signature_resolver_matches_leaf_module_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_c_signature_extractor(
        monkeypatch,
        modules={
            "first": ExtractedModule(
                name="first",
                functions={
                    "foo": ExtractedFunction(
                        ml_name="foo",
                        function_cursor=_fake_function_cursor("foo"),
                        ml_flags=METH_VARARGS,
                        signatures=[ExtractedSignature(arguments=[_arg("x", "int")])],
                    )
                },
            ),
            "second": ExtractedModule(
                name="second",
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
    resolver = CSignatureResolver(source=tmp_path)
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

    first = resolver.resolve_function(first_module, first_module.functions[0])
    second = resolver.resolve_function(second_module, second_module.functions[0])

    assert first is not None
    assert second is not None
    first_signatures, _ = first
    second_signatures, _ = second
    assert first_signatures[0].args[0].name == "x"
    assert first_signatures[0].args[0].type is not None
    assert first_signatures[0].args[0].type.render() == "int"
    assert second_signatures[0].args[0].name == "value"
    assert second_signatures[0].args[0].type is not None
    assert second_signatures[0].args[0].type.render() == "float"


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
    resolver = CSignatureResolver(source=tmp_path)
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo")],
    )

    resolved = resolver.resolve_function(module, module.functions[0])

    assert resolved is not None
    signatures, _ = resolved
    assert signatures[0].args[0].name == "value"
    assert signatures[0].args[0].type is not None
    assert signatures[0].args[0].type.render() == "float"


def test_c_signature_resolver_raises_for_ambiguous_leaf_name(
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
    resolver = CSignatureResolver(source=tmp_path)
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo")],
    )

    with pytest.raises(RuntimeError, match="未匹配到唯一C模块"):
        resolver.resolve_function(module, module.functions[0])


def test_c_signature_resolver_raises_when_function_is_missing_or_empty(
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
    resolver = CSignatureResolver(source=tmp_path)
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo"), _unknown_function("baz")],
    )

    with pytest.raises(RuntimeError, match="未找到函数 foo"):
        resolver.resolve_function(module, module.functions[0])

    with pytest.raises(RuntimeError, match="没有可用签名"):
        resolver.resolve_function(module, module.functions[1])


def test_c_signature_resolver_propagates_extraction_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_raising_c_signature_extractor(monkeypatch, RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        CSignatureResolver(source=tmp_path)


def test_c_signature_resolver_passes_compilation_database_to_extractor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _record_collect_modules(
        compilation_database: Path,
    ) -> dict[str, ExtractedModule]:
        captured["compilation_database"] = compilation_database
        return {}

    import pcstubgen.signature_completion.c_extension.source as resolver_module

    monkeypatch.setattr(
        resolver_module,
        "collect_modules",
        _record_collect_modules,
    )

    CSignatureResolver(compilation_database=tmp_path / "compile_commands.json")

    assert captured["compilation_database"] == tmp_path / "compile_commands.json"
