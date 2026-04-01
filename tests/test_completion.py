from __future__ import annotations

from io import StringIO

from loguru import logger

from tests._c_extension_test_support import *


def test_summary_str_uses_chinese_labels() -> None:
    summary = SignatureCompletionResult(
        total_functions=6,
        c_completed=2,
        docstring_completed=1,
        uncompleted=3,
    )

    assert str(summary) == (
        "签名补全结果: 函数总数=6, "
        "C源码补全=2, 文档字符串补全=1, 未补全=3"
    )


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

    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[
            _unknown_function(
                "foo",
                doc="foo(value: str) -> str\n\nparsed from docstring",
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
    assert parsed.doc == "foo(value: str) -> str\n\nparsed from docstring"
    assert parsed.c_inferred_source_comment == snippet
    assert summary.total_functions == 1
    assert summary.c_completed == 1
    assert summary.docstring_completed == 0
    assert summary.uncompleted == 0


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

    log_output = StringIO()
    sink_id = logger.add(log_output, format="{message}")
    try:
        summary = SignatureCompleter(StubGenerationOptions(source_root=tmp_path)).run(module)
    finally:
        logger.remove(sink_id)

    parsed = module.functions[0]
    assert [arg.name for arg in parsed.signatures[0].args] == ["x", "y", "w", "out", "p"]
    assert parsed.signatures[0].return_type is not None
    assert parsed.signatures[0].return_type.render() == "numpy.ndarray"
    assert summary.total_functions == 1
    assert summary.c_completed == 0
    assert summary.docstring_completed == 1
    assert summary.uncompleted == 0
    assert "通过docstring补全成功" in log_output.getvalue()
    assert "补全失败" not in log_output.getvalue()


def test_completer_skips_source_comment_when_option_disabled(
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
                            arguments=[_arg("value", "int")],
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
        functions=[_unknown_function("foo")],
    )

    SignatureCompleter(
        StubGenerationOptions(
            source_root=tmp_path,
            include_c_inferred_source_comment=False,
        )
    ).run(module)

    parsed = module.functions[0]
    assert [arg.name for arg in parsed.signatures[0].args] == ["value"]
    assert parsed.c_inferred_source_comment is None


def test_completer_skips_c_for_methods_and_leaves_unresolved_without_docstring(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
                        function=_unknown_function("build"),
                        decorator=None,
                    )
                ],
            )
        ],
    )

    log_output = StringIO()
    sink_id = logger.add(log_output, format="{message}")
    try:
        summary = SignatureCompleter(StubGenerationOptions(source_root=tmp_path)).run(module)
    finally:
        logger.remove(sink_id)

    parsed = module.classes[0].methods[0].function
    assert parsed.signatures == []
    assert summary.total_functions == 1
    assert summary.c_completed == 0
    assert summary.docstring_completed == 0
    assert summary.uncompleted == 1
    assert "c_reason: C源码补全暂不支持方法。" in log_output.getvalue()
    assert "docstring_reason: docstring为空或缺失，无法解析签名。" in log_output.getvalue()


def test_completer_preserves_function_doc_when_signature_stays_unresolved() -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.PYTHON,
        functions=[
            _unknown_function(
                "fallback",
                doc="plain fallback docs",
            )
        ],
    )

    summary = SignatureCompleter(StubGenerationOptions()).run(module)

    parsed = module.functions[0]
    assert parsed.doc == "plain fallback docs"
    assert parsed.signatures == []
    assert summary.uncompleted == 1


def test_completer_uses_docstring_when_available_for_python_module() -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.PYTHON,
        functions=[
            _unknown_function(
                "fallback",
                doc="fallback(value: str) -> bool\n\nparsed from docstring",
            )
        ],
    )

    summary = SignatureCompleter(StubGenerationOptions()).run(module)

    parsed = module.functions[0]
    assert [arg.name for arg in parsed.signatures[0].args] == ["value"]
    assert parsed.signatures[0].args[0].type is not None
    assert parsed.signatures[0].args[0].type.render() == "str"
    assert parsed.signatures[0].return_type is not None
    assert parsed.signatures[0].return_type.render() == "bool"
    assert summary.docstring_completed == 1
    assert summary.uncompleted == 0


def test_completer_logs_explicit_reasons_when_both_paths_return_no_signature() -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.PYTHON,
        functions=[
            _unknown_function(
                "fallback",
                doc="plain fallback docs",
            )
        ],
    )

    log_output = StringIO()
    sink_id = logger.add(log_output, format="{message}")
    try:
        summary = SignatureCompleter(StubGenerationOptions()).run(module)
    finally:
        logger.remove(sink_id)

    assert summary.uncompleted == 1
    assert "c_reason: 未配置 source_root，未启用C源码补全。" in log_output.getvalue()
    assert "docstring_reason: docstring首行不是可解析的签名声明。" in log_output.getvalue()


def test_completer_logs_docstring_parse_error_in_final_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[
            _unknown_function(
                "foo",
                doc="foo(a=1=2) -> int",
            )
        ],
    )
    _patch_c_signature_extractor(monkeypatch, modules={})

    log_output = StringIO()
    sink_id = logger.add(log_output, format="{message}")
    try:
        summary = SignatureCompleter(StubGenerationOptions(source_root=tmp_path)).run(module)
    finally:
        logger.remove(sink_id)

    assert summary.uncompleted == 1
    assert "c_reason: 未匹配到唯一C模块: pkg.mod" in log_output.getvalue()
    assert "docstring_reason: ValueError: 参数默认值声明中包含多个 '='。" in log_output.getvalue()


def test_completer_keeps_known_signatures_and_counts_unresolved() -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.PYTHON,
        functions=[
            IRFunction(
                name="known",
                signatures=[_signature(args=[IRArgument(name="value")])],
            ),
            _unknown_function("missing"),
            _unknown_function("fallback"),
        ],
    )

    summary = SignatureCompleter(StubGenerationOptions()).run(module)

    assert summary.total_functions == 3
    assert summary.c_completed == 0
    assert summary.docstring_completed == 0
    assert summary.uncompleted == 3
    assert module.functions[0].signatures[0].args[0].name == "value"
    assert module.functions[1].signatures == []
    assert module.functions[2].signatures == []


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
    assert first_summary.uncompleted == 1
    assert second_summary.total_functions == 0
    assert second_summary.uncompleted == 0
    assert second_summary is not first_summary
