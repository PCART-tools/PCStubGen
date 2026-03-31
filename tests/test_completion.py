from __future__ import annotations

from tests._c_signature_test_support import *


def test_completer_prefers_c_over_docstring_and_writes_source_comment(
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

    def foo(value: int) -> int:
        raise NotImplementedError

    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[
            _unknown_function(
                "foo",
                doc="foo(value: str) -> str\n\nparsed from docstring",
                runtime_function=foo,
            )
        ],
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
                            arguments=[_arg("value", "int")],
                            return_type=RawType("bool"),
                        )
                    ],
                )
            }
        ),
    )

    summary = SignatureCompleter(
        StubGenerationOptions(
            source_root=tmp_path,
            include_c_inferred_source_comment=True,
        )
    ).run(module)

    parsed = module.functions[0]
    assert [arg.name for arg in parsed.signatures[0].args] == ["value"]
    assert [arg.type.render() if arg.type is not None else None for arg in parsed.signatures[0].args] == [
        "int"
    ]
    assert parsed.signatures[0].return_type is not None
    assert parsed.signatures[0].return_type.render() == "bool"
    assert parsed.signatures[0].doc == "foo(value: str) -> str\n\nparsed from docstring"
    assert parsed.c_inferred_source_comment == snippet
    assert summary.total_functions == 1
    assert summary.c_resolved == 1
    assert summary.docstring_resolved == 0
    assert summary.inspect_resolved == 0
    assert summary.unresolved == 0


def test_completer_falls_back_to_docstring_when_c_has_no_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[
            _unknown_function(
                "cdist_minkowski",
                doc=(
                    "cdist_minkowski(x: object, y: object, w: object = None, "
                    "out: object = None, p: typing.SupportsFloat = 2.0) -> numpy.ndarray"
                ),
            )
        ],
    )
    _patch_c_signature_extractor(monkeypatch, modules={})

    summary = SignatureCompleter(StubGenerationOptions(source_root=tmp_path)).run(module)

    parsed = module.functions[0]
    assert [arg.name for arg in parsed.signatures[0].args] == ["x", "y", "w", "out", "p"]
    assert parsed.signatures[0].return_type is not None
    assert parsed.signatures[0].return_type.render() == "numpy.ndarray"
    assert summary.total_functions == 1
    assert summary.c_resolved == 0
    assert summary.docstring_resolved == 1
    assert summary.inspect_resolved == 0
    assert summary.unresolved == 0


def test_completer_uses_inspect_as_last_fallback_and_skips_c_for_methods(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Builder:
        @classmethod
        def build(cls, value: int) -> str:
            raise NotImplementedError

    _patch_c_signature_extractor(
        monkeypatch,
        modules=_module_fixture(
            functions={
                "build": ExtractedFunction(
                    ml_name="build",
                    function_cursor=_fake_function_cursor("build"),
                    ml_flags=METH_VARARGS,
                    signatures=[
                        ExtractedSignature(
                            arguments=[_arg("from_c", "bool")],
                            return_type=RawType("bool"),
                        )
                    ],
                )
            }
        ),
    )
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        classes=[
            IRClass(
                name="Builder",
                methods=[
                    IRMethod(
                        function=_unknown_function(
                            "build",
                            runtime_function=Builder.build,
                        ),
                        decorator=None,
                    )
                ],
            )
        ],
    )

    summary = SignatureCompleter(StubGenerationOptions(source_root=tmp_path)).run(module)

    parsed = module.classes[0].methods[0].function
    assert [arg.name for arg in parsed.signatures[0].args] == ["cls", "value"]
    assert [arg.type.render() if arg.type is not None else None for arg in parsed.signatures[0].args] == [
        None,
        "int",
    ]
    assert parsed.signatures[0].return_type is not None
    assert parsed.signatures[0].return_type.render() == "str"
    assert summary.total_functions == 1
    assert summary.c_resolved == 0
    assert summary.docstring_resolved == 0
    assert summary.inspect_resolved == 1
    assert summary.unresolved == 0


def test_completer_skips_known_signatures_and_counts_unresolved() -> None:
    def fallback(value: int) -> int:
        raise NotImplementedError

    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.PYTHON,
        functions=[
            IRFunction(
                name="known",
                signatures=[_signature(args=[IRArgument(name="value")])],
            ),
            _unknown_function("missing"),
            _unknown_function("fallback", runtime_function=fallback),
        ],
    )

    summary = SignatureCompleter(StubGenerationOptions()).run(module)

    assert summary.total_functions == 3
    assert summary.skipped_known_signatures == 1
    assert summary.c_resolved == 0
    assert summary.docstring_resolved == 0
    assert summary.inspect_resolved == 1
    assert summary.unresolved == 1
    assert module.functions[0].signatures[0].args[0].name == "value"
    assert module.functions[1].signatures == []
    assert [arg.name for arg in module.functions[2].signatures[0].args] == ["value"]


def test_completer_run_recreates_summary_for_each_invocation() -> None:
    completer = SignatureCompleter(StubGenerationOptions())

    first_module = IRModule(
        full_name=QualifiedName.from_str("pkg.first"),
        module_type=IRModuleType.PYTHON,
        functions=[_unknown_function("missing")],
    )
    second_module = IRModule(
        full_name=QualifiedName.from_str("pkg.second"),
        module_type=IRModuleType.PYTHON,
        functions=[],
    )

    first_summary = completer.run(first_module)
    second_summary = completer.run(second_module)

    assert first_summary.total_functions == 1
    assert first_summary.unresolved == 1
    assert second_summary.total_functions == 0
    assert second_summary.unresolved == 0
    assert second_summary is not first_summary
