from __future__ import annotations

from tests._c_signature_test_support import *


def test_c_ast_visitor_rewrites_module_function_without_normalizing_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    func = _unknown_function("foo")
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[func],
    )
    _patch_c_signature_extractor(
        monkeypatch,
        modules=_module_fixture(
            functions={
                "foo": ExtractedFunction(
                    ml_name="foo",
                    function_cursor=_fake_function_cursor("foo"),
                    ml_flags=METH_VARARGS,
                    signatures=[
                        ExtractedSignature(
                            arguments=[
                                _arg("self", "object"),
                                _arg("x", "int"),
                                _arg("flag", "bool", default_value="False", has_default=True),
                            ],
                            return_type=_raw("int"),
                        )
                    ],
                )
            }
        ),
    )

    visitor = CSignatureVisitor(
        source_root=tmp_path,
    )
    visitor.visit_module(module)

    rewritten = module.functions[0]
    assert len(rewritten.signatures) == 1
    signature = rewritten.signatures[0]
    assert [arg.name for arg in signature.args] == ["self", "x", "flag"]
    assert signature.args[0].type is not None and signature.args[0].type.render() == "object"
    assert signature.args[1].type is not None and signature.args[1].type.render() == "int"
    assert signature.args[2].type is not None and signature.args[2].type.render() == "bool"
    assert signature.args[2].default_value is not None
    assert signature.args[2].default_value == "False"
    assert signature.args[2].has_default is True
    assert signature.return_type is not None
    assert signature.return_type.render() == "int"
    assert rewritten.c_inferred_source_comment is None
    assert visitor._stats.total_unknown_signatures == 1
    assert visitor._stats.success == 1


def test_c_ast_visitor_records_c_inferred_source_comment_when_enabled(
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

    func = _unknown_function("foo")
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[func],
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
                            arguments=[_arg("value", "int")]
                        )
                    ],
                )
            }
        ),
    )

    visitor = CSignatureVisitor(
        source_root=tmp_path,
        include_c_inferred_source_comment=True,
    )
    visitor.visit_module(module)

    assert module.functions[0].c_inferred_source_comment == snippet


def test_c_ast_visitor_skips_c_inferred_source_comment_when_extent_text_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    func = _unknown_function("foo")
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[func],
    )
    _patch_c_signature_extractor(
        monkeypatch,
        modules=_module_fixture(
            functions={
                "foo": ExtractedFunction(
                    ml_name="foo",
                    function_cursor=_fake_function_cursor("foo"),
                    ml_flags=METH_VARARGS,
                    signatures=[
                        ExtractedSignature(
                            arguments=[_arg("value", "int")]
                        )
                    ],
                )
            }
        ),
    )

    visitor = CSignatureVisitor(
        source_root=tmp_path,
        include_c_inferred_source_comment=True,
    )
    visitor.visit_module(module)

    assert module.functions[0].c_inferred_source_comment is None


def test_c_ast_visitor_preserves_has_default_without_default_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    func = _unknown_function("foo")
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[func],
    )
    _patch_c_signature_extractor(
        monkeypatch,
        modules=_module_fixture(
            functions={
                "foo": ExtractedFunction(
                    ml_name="foo",
                    function_cursor=_fake_function_cursor("foo"),
                    ml_flags=METH_VARARGS,
                    signatures=[
                        ExtractedSignature(
                            arguments=[
                                _arg("flag", "bool", has_default=True)
                            ]
                        )
                    ],
                )
            }
        ),
    )

    visitor = CSignatureVisitor(source_root=tmp_path)
    visitor.visit_module(module)

    signature = module.functions[0].signatures[0]
    assert signature.args[0].has_default is True
    assert signature.args[0].default_value is None


def test_c_ast_visitor_preserves_raw_argument_and_return_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    func = _unknown_function("foo")
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[func],
    )
    _patch_c_signature_extractor(
        monkeypatch,
        modules=_module_fixture(
            functions={
                "foo": ExtractedFunction(
                    ml_name="foo",
                    function_cursor=_fake_function_cursor("foo"),
                    ml_flags=METH_VARARGS,
                    signatures=[
                        ExtractedSignature(
                            arguments=[
                                _arg("value", "  int  ", default_value="  keep_raw()  ", has_default=True)
                            ],
                            return_type=_raw("  bool  "),
                        )
                    ],
                )
            }
        ),
    )

    visitor = CSignatureVisitor(source_root=tmp_path)
    visitor.visit_module(module)

    signature = module.functions[0].signatures[0]
    assert signature.args[0].type is not None and signature.args[0].type.render() == "  int  "
    assert signature.args[0].default_value == "  keep_raw()  "
    assert signature.return_type is not None and signature.return_type.render() == "  bool  "


def test_c_ast_visitor_preserves_extracted_argument_kinds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    func = _unknown_function("foo")
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[func],
    )
    _patch_c_signature_extractor(
        monkeypatch,
        modules=_module_fixture(
            functions={
                "foo": ExtractedFunction(
                    ml_name="foo",
                    function_cursor=_fake_function_cursor("foo"),
                    ml_flags=METH_VARARGS | METH_KEYWORDS,
                    signatures=[
                        ExtractedSignature(
                            arguments=[
                                ExtractedArgument(
                                    name="value",
                                    kind=IRArgumentKind.KEYWORD_ONLY,
                                ),
                                ExtractedArgument(
                                    name="args",
                                    kind=IRArgumentKind.VAR_POSITIONAL,
                                ),
                                ExtractedArgument(
                                    name="kwargs",
                                    kind=IRArgumentKind.VAR_KEYWORD,
                                ),
                            ]
                        )
                    ],
                )
            }
        ),
    )

    visitor = CSignatureVisitor(source_root=tmp_path)
    visitor.visit_module(module)

    signature = module.functions[0].signatures[0]
    assert [arg.kind for arg in signature.args] == [
        IRArgumentKind.KEYWORD_ONLY,
        IRArgumentKind.VAR_POSITIONAL,
        IRArgumentKind.VAR_KEYWORD,
    ]


def test_c_ast_visitor_keeps_known_function_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    visitor = CSignatureVisitor(
        source_root=tmp_path,
    )
    func = IRFunction(
        name="foo",
        signatures=[_signature(args=[IRArgument(name="x", kind=IRArgumentKind.POSITIONAL_OR_KEYWORD)])],
    )
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[func],
    )
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

    visitor.visit_module(module)

    assert module.functions[0] is func
    assert func.signatures[0].args[0].name == "x"
    assert func.signatures[0].args[0].type is None
    assert func.c_inferred_source_comment is None
    assert visitor._stats.total_unknown_signatures == 0
    assert visitor._stats.success == 0


def test_c_ast_visitor_records_missing_function_match_stats(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo")],
    )
    _patch_c_signature_extractor(
        monkeypatch,
        modules=_module_fixture(
            functions={
                "bar": ExtractedFunction(
                    ml_name="bar",
                    function_cursor=_fake_function_cursor("bar"),
                    ml_flags=METH_VARARGS,
                    signatures=[ExtractedSignature(arguments=[_arg("x", "int")])],
                )
            }
        ),
    )

    visitor = CSignatureVisitor(source_root=tmp_path)
    visitor.visit_module(module)

    assert module.functions[0].signatures == []
    assert visitor._stats.total_unknown_signatures == 1
    assert visitor._stats.success == 0
    assert visitor._stats.missing_module_match == 0
    assert visitor._stats.missing_function_match == 1
    assert visitor._stats.matched_function_without_signatures == 0


def test_c_ast_visitor_records_matched_function_without_signatures_stats(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo")],
    )
    _patch_c_signature_extractor(
        monkeypatch,
        modules=_module_fixture(
            functions={
                "foo": ExtractedFunction(
                    ml_name="foo",
                    function_cursor=_fake_function_cursor("foo"),
                    ml_flags=METH_VARARGS,
                    signatures=[],
                )
            }
        ),
    )

    visitor = CSignatureVisitor(source_root=tmp_path)
    visitor.visit_module(module)

    assert module.functions[0].signatures == []
    assert visitor._stats.total_unknown_signatures == 1
    assert visitor._stats.success == 0
    assert visitor._stats.missing_module_match == 0
    assert visitor._stats.missing_function_match == 0
    assert visitor._stats.matched_function_without_signatures == 1


def test_c_ast_visitor_records_empty_extraction_as_missing_module_match(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo")],
    )
    _patch_c_signature_extractor(monkeypatch, modules={})

    visitor = CSignatureVisitor(source_root=tmp_path)
    visitor.visit_module(module)

    assert module.functions[0].signatures == []
    assert visitor._stats.total_unknown_signatures == 1
    assert visitor._stats.success == 0
    assert visitor._stats.missing_module_match == 1
    assert visitor._stats.missing_function_match == 0
    assert visitor._stats.matched_function_without_signatures == 0


def test_c_ast_visitor_matches_candidates_by_module_before_function_name(
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
    visitor = CSignatureVisitor(
        source_root=tmp_path,
    )
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

    visitor.visit_module(first_module)
    visitor.visit_module(second_module)

    assert [arg.name for arg in first_module.functions[0].signatures[0].args] == ["x"]
    assert first_module.functions[0].signatures[0].args[0].type is not None and first_module.functions[0].signatures[0].args[0].type.render() == "int"
    assert [arg.name for arg in second_module.functions[0].signatures[0].args] == ["value"]
    assert second_module.functions[0].signatures[0].args[0].type is not None and second_module.functions[0].signatures[0].args[0].type.render() == "float"


def test_c_ast_visitor_falls_back_to_unique_leaf_module_match(
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
                        signatures=[
                            ExtractedSignature(
                                arguments=[_arg("value", "float")]
                            )
                        ],
                    )
                },
            ),
        },
    )
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo")],
    )
    visitor = CSignatureVisitor(source_root=tmp_path)

    visitor.visit_module(module)

    assert [arg.name for arg in module.functions[0].signatures[0].args] == ["value"]
    assert module.functions[0].signatures[0].args[0].type is not None and module.functions[0].signatures[0].args[0].type.render() == "float"
    assert visitor._stats.success == 1


def test_c_ast_visitor_rejects_ambiguous_leaf_module_match_without_global_fallback(
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
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo")],
    )
    visitor = CSignatureVisitor(
        source_root=tmp_path,
    )

    visitor.visit_module(module)

    assert module.functions[0].signatures == []
    assert visitor._stats.total_unknown_signatures == 1
    assert visitor._stats.missing_module_match == 1
    assert visitor._stats.missing_function_match == 0


def test_c_ast_visitor_overwrites_existing_return_with_raw_inferred_return(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    func = IRFunction(
        name="foo",
        signatures=[],
        doc="original doc",
    )
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[func],
    )
    _patch_c_signature_extractor(
        monkeypatch,
        modules=_module_fixture(
            functions={
                "foo": ExtractedFunction(
                    ml_name="foo",
                    function_cursor=_fake_function_cursor("foo"),
                    ml_flags=METH_VARARGS,
                    signatures=[
                        ExtractedSignature(
                            arguments=[_arg("x", "int")],
                            return_type=_raw("typing.Optional[int]", imports=["typing"]),
                        )
                    ],
                )
            }
        ),
    )

    visitor = CSignatureVisitor(
        source_root=tmp_path,
    )
    visitor.visit_module(module)

    rewritten = module.functions[0]
    assert rewritten.signatures[0].return_type is not None
    assert rewritten.signatures[0].return_type.render() == "typing.Optional[int]"


def test_c_ast_visitor_skips_python_modules(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.PYTHON,
        functions=[_unknown_function("foo")],
    )
    extractor = _patch_c_signature_extractor(
        monkeypatch,
        modules=_module_fixture(
            functions={
                "foo": ExtractedFunction(
                    ml_name="foo",
                    function_cursor=_fake_function_cursor("foo"),
                    signatures=[ExtractedSignature(arguments=[_arg("x", "int")])],
                )
            }
        ),
    )
    visitor = CSignatureVisitor(
        source_root=tmp_path,
    )
    visitor.visit_module(module)

    assert module.functions[0].signatures == []
    assert extractor.called == 0


def test_c_ast_visitor_propagates_signature_extraction_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo")],
    )
    _patch_raising_c_signature_extractor(monkeypatch, RuntimeError("boom"))

    visitor = CSignatureVisitor(
        source_root=tmp_path,
    )

    with pytest.raises(RuntimeError, match="boom"):
        visitor.visit_module(module)


def test_c_ast_visitor_passes_clang_options_to_extractor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {
        "extract_calls": 0,
        "include": None,
    }

    def _record_extract_c_signature_modules(
        source_root: Path,
        *,
        include: list[str] = (),
        include_directory: list[Path] = (),
        c_std: str = "c11",
        cpp_std: str = "c++17",
    ) -> dict[str, ExtractedModule]:
        captured["extract_calls"] = int(captured["extract_calls"]) + 1
        captured["source_root"] = source_root
        captured["include"] = list(include)
        captured["include_directory"] = list(include_directory)
        captured["c_std"] = c_std
        captured["cpp_std"] = cpp_std
        return {}

    import pcstubgen.visitors.c_signature_visitor as visitor_module

    monkeypatch.setattr(
        visitor_module,
        "extract_c_signature_modules",
        _record_extract_c_signature_modules,
    )

    visitor = CSignatureVisitor(
        source_root=tmp_path,
        include=["Python.h"],
        include_directory=[Path("C:/MyInclude")],
        c_std="c99",
        cpp_std="c++20",
    )

    assert captured["extract_calls"] == 0

    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
    )
    visitor.visit_module(module)
    visitor.visit_module(module)

    assert captured["extract_calls"] == 1
    assert captured["source_root"] == tmp_path
    assert captured["include"] == ["Python.h"]
    assert captured["include_directory"] == [Path("C:/MyInclude")]
    assert captured["c_std"] == "c99"
    assert captured["cpp_std"] == "c++20"
