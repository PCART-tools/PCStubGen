from __future__ import annotations

from pathlib import Path

import pytest

from pcstubgen2.NodeVisitors.CSignatureInference.CSignatureExtraction import CSignatureExtractionEngine
from pcstubgen2.NodeVisitors.CSignatureInference.CSignatureExtraction.Models import (
    ExtractedArgument,
    ExtractedFunction,
    ExtractedSignature,
)
from pcstubgen2.ErrorCollector import ErrorCollector
from pcstubgen2.IR import (
    IRArgument,
    IRArgumentKind,
    IRClass,
    IRFunction,
    IRMethod,
    IRModule,
    IRModuleType,
    QualifiedName,
    ResolvedType,
)
from pcstubgen2.NodeVisitors.CSignatureInference.CAstSignatureInferenceVisitor import (
    CAstSignatureInferenceVisitor,
)
from pcstubgen2.NodeVisitors.DocStringSignatureParserVisitor import (
    DocStringSignatureParserVisitor,
)
from pcstubgen2.NodeVisitors.Fixes import InferMethodModifierVisitor
from pcstubgen2.Pipeline import Pipeline


def _generic_signature() -> list[IRArgument]:
    return [
        IRArgument(name="args", kind=IRArgumentKind.VAR_POSITIONAL),
        IRArgument(name="kwargs", kind=IRArgumentKind.VAR_KEYWORD),
    ]


class _FakeExtractor:
    def __init__(self, data: dict[str, list[ExtractedFunction]]) -> None:
        self.data = data
        self.called = 0

    def extract(self) -> dict[str, list[ExtractedFunction]]:
        self.called += 1
        return self.data


def _get_packaged_libclang_path() -> str | None:
    import clang

    native_dir = Path(clang.__file__).resolve().parent / "native"
    for filename in ("libclang.dll", "libclang.so", "libclang.dylib"):
        candidate = native_dir / filename
        if candidate.exists():
            return str(candidate)
    return None


def test_c_ast_visitor_rewrites_module_function_and_drops_self() -> None:
    func = IRFunction(name="foo", args=_generic_signature())
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.C,
        functions=[func],
    )
    extractor = _FakeExtractor(
        {
            "foo": [
                ExtractedFunction(
                    py_name="foo",
                    c_name="c_foo",
                    method_flags=["METH_VARARGS"],
                    signatures=[
                        ExtractedSignature(
                            arguments=[
                                ExtractedArgument(name="self", type_name="object"),
                                ExtractedArgument(name="x", type_name="int"),
                                ExtractedArgument(name="flag", type_name="bool", default_value="False"),
                            ],
                            return_type_name="int",
                        )
                    ],
                )
            ]
        }
    )

    visitor = CAstSignatureInferenceVisitor(
        error_collector=ErrorCollector(),
        c_source_root=None,
        signature_inference_scope="c_modules",
        extractor=extractor,
    )
    visitor.visit_module(module)

    rewritten = module.functions[0]
    assert [arg.name for arg in rewritten.args] == ["x", "flag"]
    assert str(rewritten.args[0].annotation) == "int"
    assert str(rewritten.args[1].annotation) == "bool"
    assert rewritten.args[1].default is not None
    assert rewritten.args[1].default.repr == "False"
    assert rewritten.return_annotation is not None
    assert str(rewritten.return_annotation) == "int"


def test_c_ast_visitor_keeps_existing_return_when_inferred_return_invalid() -> None:
    func = IRFunction(
        name="foo",
        args=_generic_signature(),
        return_annotation=ResolvedType(name=QualifiedName.from_str("bytes")),
    )
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.C,
        functions=[func],
    )
    extractor = _FakeExtractor(
        {
            "foo": [
                ExtractedFunction(
                    py_name="foo",
                    c_name="c_foo",
                    method_flags=["METH_VARARGS"],
                    signatures=[
                        ExtractedSignature(
                            arguments=[ExtractedArgument(name="x", type_name="int")],
                            return_type_name="typing.Optional[int]",
                        )
                    ],
                )
            ]
        }
    )

    visitor = CAstSignatureInferenceVisitor(
        error_collector=ErrorCollector(),
        c_source_root=None,
        signature_inference_scope="c_modules",
        extractor=extractor,
    )
    visitor.visit_module(module)

    rewritten = module.functions[0]
    assert rewritten.return_annotation is not None
    assert str(rewritten.return_annotation) == "bytes"


def test_c_ast_visitor_generates_overloads_for_methods() -> None:
    method = IRMethod(function=IRFunction(name="build", args=_generic_signature()), decorator=None)
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.C,
        classes=[IRClass(name="C", methods=[method])],
    )

    extractor = _FakeExtractor(
        {
            "build": [
                ExtractedFunction(
                    py_name="build",
                    c_name="c_build",
                    method_flags=["METH_VARARGS"],
                    signatures=[
                        ExtractedSignature(
                            arguments=[
                                ExtractedArgument(name="self", type_name="object"),
                                ExtractedArgument(name="count", type_name="int"),
                            ]
                        ),
                        ExtractedSignature(
                            arguments=[
                                ExtractedArgument(name="self", type_name="object"),
                                ExtractedArgument(name="count", type_name="int"),
                                ExtractedArgument(name="scale", type_name="float", default_value="1.0"),
                            ]
                        ),
                    ],
                )
            ]
        }
    )

    visitor = CAstSignatureInferenceVisitor(
        error_collector=ErrorCollector(),
        c_source_root=None,
        signature_inference_scope="c_modules",
        extractor=extractor,
    )
    visitor.visit_module(module)

    methods = module.classes[0].methods
    assert len(methods) == 2
    assert [m.function.args[0].name for m in methods] == ["self", "self"]
    assert [str(m.function.args[0].annotation) for m in methods] == ["object", "object"]
    assert [str(m.function.args[1].annotation) for m in methods] == ["int", "int"]
    assert methods[1].function.args[2].annotation is not None
    assert str(methods[1].function.args[2].annotation) == "float"
    assert all(m.function.decorators == ["typing.overload"] for m in methods)
    assert methods[1].function.args[2].default is not None
    assert methods[1].function.args[2].default.repr == "1.0"


def test_scope_c_modules_only_skips_python_modules() -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.PYTHON,
        functions=[IRFunction(name="foo", args=_generic_signature())],
    )
    extractor = _FakeExtractor(
        {
            "foo": [
                ExtractedFunction(
                    py_name="foo",
                    c_name="c_foo",
                    signatures=[ExtractedSignature(arguments=[ExtractedArgument(name="x", type_name="int")])],
                )
            ]
        }
    )
    visitor = CAstSignatureInferenceVisitor(
        error_collector=ErrorCollector(),
        c_source_root=None,
        signature_inference_scope="c_modules",
        extractor=extractor,
    )
    visitor.visit_module(module)

    assert module.functions[0].is_generic_signature()
    assert extractor.called == 0


def test_doc_parser_runs_before_c_ast_visitor_in_pipeline() -> None:
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.C,
        functions=[
            IRFunction(
                name="foo",
                args=_generic_signature(),
                doc="foo(a: int, b: int = 0) -> int",
            )
        ],
    )
    extractor = _FakeExtractor(
        {
            "foo": [
                ExtractedFunction(
                    py_name="foo",
                    c_name="c_foo",
                    signatures=[ExtractedSignature(arguments=[ExtractedArgument(name="x", type_name="int")])],
                )
            ]
        }
    )
    pipeline = Pipeline(
        [
            DocStringSignatureParserVisitor(error_collector=ErrorCollector()),
            CAstSignatureInferenceVisitor(
                error_collector=ErrorCollector(),
                c_source_root=None,
                signature_inference_scope="c_modules",
                extractor=extractor,
            ),
        ]
    )
    pipeline.run(module)

    parsed = module.functions[0]
    assert [arg.name for arg in parsed.args] == ["a", "b"]
    assert [str(arg.annotation) for arg in parsed.args] == ["int", "int"]
    assert extractor.called == 1


def test_infer_method_modifier_after_c_ast_visitor() -> None:
    method = IRMethod(
        function=IRFunction(name="make", args=_generic_signature()),
        decorator="staticmethod",
    )
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.C,
        classes=[IRClass(name="Builder", methods=[method])],
    )
    extractor = _FakeExtractor(
        {
            "make": [
                ExtractedFunction(
                    py_name="make",
                    c_name="c_make",
                    method_flags=["METH_CLASS"],
                    signatures=[
                        ExtractedSignature(
                            arguments=[
                                ExtractedArgument(name="cls", type_name="type"),
                                ExtractedArgument(name="n", type_name="int"),
                            ]
                        )
                    ],
                )
            ]
        }
    )

    Pipeline(
        [
            CAstSignatureInferenceVisitor(
                error_collector=ErrorCollector(),
                c_source_root=None,
                signature_inference_scope="c_modules",
                extractor=extractor,
            ),
            InferMethodModifierVisitor(),
        ]
    ).run(module)

    rewritten = module.classes[0].methods[0]
    assert [arg.name for arg in rewritten.function.args] == ["cls", "n"]
    assert [str(arg.annotation) for arg in rewritten.function.args] == ["type", "int"]
    assert rewritten.decorator == "classmethod"


def test_c_signature_extraction_engine_parses_minimal_c_file(tmp_path: Path) -> None:
    pytest.importorskip("clang.cindex")
    libclang_path = _get_packaged_libclang_path()
    if libclang_path is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "mini_ext.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "#define METH_VARARGS 1",
                "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                "static PyObject* add_impl(PyObject* self, PyObject* args) {",
                "    int a = 0;",
                "    int b = 0;",
                "    if (!PyArg_ParseTuple(args, \"ii\", &a, &b)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyMethodDef Methods[] = {",
                "    {\"add\", add_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractionEngine(
        source_root=tmp_path,
        clang_library_path=libclang_path,
        clang_parse_args=["-std=c11"],
    )
    extracted = engine.extract()

    assert "add" in extracted
    add_candidates = extracted["add"]
    assert add_candidates
    first = add_candidates[0]
    assert first.py_name == "add"
    assert first.signatures
    assert [arg.name for arg in first.signatures[0].arguments] == ["self", "a", "b"]
    assert [arg.type_name for arg in first.signatures[0].arguments] == ["object", "int", "int"]
    assert first.signatures[0].return_type_name is None


def test_c_signature_engine_infers_return_type_from_py_buildvalue(tmp_path: Path) -> None:
    pytest.importorskip("clang.cindex")
    libclang_path = _get_packaged_libclang_path()
    if libclang_path is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "mini_buildvalue_ext.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "#define METH_VARARGS 1",
                "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                "PyObject* Py_BuildValue(const char* fmt, ...);",
                "static PyObject* make_impl(PyObject* self, PyObject* args) {",
                "    int value = 0;",
                "    if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return Py_BuildValue(\"i\", value);",
                "}",
                "static PyMethodDef Methods[] = {",
                "    {\"make\", make_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractionEngine(
        source_root=tmp_path,
        clang_library_path=libclang_path,
        clang_parse_args=["-std=c11"],
    )
    extracted = engine.extract()

    assert "make" in extracted
    first = extracted["make"][0]
    assert first.signatures
    assert first.signatures[0].return_type_name == "int"


def test_c_signature_engine_falls_back_to_object_on_conflicting_return_types(tmp_path: Path) -> None:
    pytest.importorskip("clang.cindex")
    libclang_path = _get_packaged_libclang_path()
    if libclang_path is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "mini_conflict_return_ext.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "#define METH_VARARGS 1",
                "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                "PyObject* PyLong_FromLong(long v);",
                "PyObject* PyBool_FromLong(long v);",
                "static PyObject* pick_impl(PyObject* self, PyObject* args) {",
                "    int flag = 0;",
                "    if (!PyArg_ParseTuple(args, \"i\", &flag)) {",
                "        return (PyObject*)0;",
                "    }",
                "    if (flag) {",
                "        return PyLong_FromLong(1);",
                "    }",
                "    return PyBool_FromLong(0);",
                "}",
                "static PyMethodDef Methods[] = {",
                "    {\"pick\", pick_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractionEngine(
        source_root=tmp_path,
        clang_library_path=libclang_path,
        clang_parse_args=["-std=c11"],
    )
    extracted = engine.extract()

    assert "pick" in extracted
    first = extracted["pick"][0]
    assert first.signatures
    assert first.signatures[0].return_type_name == "object"


def test_c_ast_visitor_passes_clang_options_to_extractor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class _RecorderExtractor:
        def __init__(
            self,
            source_root: str | Path,
            *,
            clang_library_path: str | None = None,
            clang_parse_args: list[str] | None = None,
        ) -> None:
            captured["source_root"] = source_root
            captured["clang_library_path"] = clang_library_path
            captured["clang_parse_args"] = list(clang_parse_args) if clang_parse_args is not None else None

        def extract(self) -> dict[str, list[ExtractedFunction]]:
            return {}

    import pcstubgen2.NodeVisitors.CSignatureInference.CAstSignatureInferenceVisitor as visitor_module

    monkeypatch.setattr(visitor_module, "CSignatureExtractionEngine", _RecorderExtractor)

    visitor = CAstSignatureInferenceVisitor(
        error_collector=ErrorCollector(),
        c_source_root=tmp_path,
        signature_inference_scope="c_modules",
        clang_library_path="C:/fake/libclang.dll",
        clang_parse_args=["-std=c11", "-DMY_FLAG=1"],
    )
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.C,
    )
    visitor.visit_module(module)

    assert captured["source_root"] == tmp_path
    assert captured["clang_library_path"] == "C:/fake/libclang.dll"
    assert captured["clang_parse_args"] == ["-std=c11", "-DMY_FLAG=1"]


def test_c_signature_engine_defaults_to_c11_parse_arg(tmp_path: Path) -> None:
    class _FakeConfig:
        loaded = False
        configured_path: str | None = None

        @classmethod
        def set_library_file(cls, path: str) -> None:
            cls.configured_path = path

    class _FakeClang:
        Config = _FakeConfig

    engine = CSignatureExtractionEngine(source_root=tmp_path)
    engine._clang = _FakeClang

    assert engine._ensure_clang_ready() is True
    assert engine._clang_parse_args is not None
    assert "-std=c11" in engine._clang_parse_args


def test_c_signature_engine_skips_non_parser_calls_in_token_params(tmp_path: Path) -> None:
    engine = CSignatureExtractionEngine(source_root=tmp_path)

    assert (
        engine._set_token_params(
            func_cursor=object(),
            meth_flags=["METH_VARARGS"],
            token_list=["Py_BuildValue", '"i"', "value"],
        )
        is None
    )
    assert (
        engine._set_token_params(
            func_cursor=object(),
            meth_flags=["METH_VARARGS", "METH_KEYWORDS"],
            token_list=["PyArg_NoKeywords", "kwargs"],
        )
        is None
    )


def test_c_signature_engine_prefers_same_file_function_definition(tmp_path: Path) -> None:
    engine = CSignatureExtractionEngine(source_root=tmp_path)
    preferred_file = str(tmp_path / "module_a.c")
    other_file = str(tmp_path / "module_b.c")

    class _FakeLocation:
        def __init__(self, file: str) -> None:
            self.file = file

    class _FakeFunctionCursor:
        def __init__(self, *, file: str, is_definition: bool) -> None:
            self.location = _FakeLocation(file=file)
            self._is_definition = is_definition

        def is_definition(self) -> bool:
            return self._is_definition

    from_other_file = _FakeFunctionCursor(file=other_file, is_definition=True)
    in_same_file_decl = _FakeFunctionCursor(file=preferred_file, is_definition=False)
    in_same_file_def = _FakeFunctionCursor(file=preferred_file, is_definition=True)

    selected = engine._select_function_cursor(
        [from_other_file, in_same_file_decl, in_same_file_def],
        preferred_file=preferred_file,
    )

    assert selected is in_same_file_def


def test_c_ast_visitor_drops_leading_self_for_static_method() -> None:
    method = IRMethod(function=IRFunction(name="build", args=_generic_signature()), decorator=None)
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.C,
        classes=[IRClass(name="Builder", methods=[method])],
    )
    extractor = _FakeExtractor(
        {
            "build": [
                ExtractedFunction(
                    py_name="build",
                    c_name="c_build",
                    method_flags=["METH_STATIC"],
                    signatures=[
                        ExtractedSignature(
                            arguments=[
                                ExtractedArgument(name="self", type_name="object"),
                                ExtractedArgument(name="count", type_name="int"),
                            ]
                        )
                    ],
                )
            ]
        }
    )

    Pipeline(
        [
            CAstSignatureInferenceVisitor(
                error_collector=ErrorCollector(),
                c_source_root=None,
                signature_inference_scope="c_modules",
                extractor=extractor,
            ),
            InferMethodModifierVisitor(),
        ]
    ).run(module)

    rewritten = module.classes[0].methods[0]
    assert [arg.name for arg in rewritten.function.args] == ["count"]
    assert str(rewritten.function.args[0].annotation) == "int"
    assert rewritten.decorator == "staticmethod"


def test_c_signature_engine_decodes_combined_numeric_method_flags(tmp_path: Path) -> None:
    engine = CSignatureExtractionEngine(source_root=tmp_path)

    assert engine._decode_meth_literal_flags("3") == ["METH_VARARGS", "METH_KEYWORDS"]
    assert engine._decode_meth_literal_flags("0x21U") == ["METH_VARARGS", "METH_STATIC"]


def test_c_signature_engine_recognizes_c_style_end_array_element(tmp_path: Path) -> None:
    engine = CSignatureExtractionEngine(source_root=tmp_path)

    class _FakeCursorKind:
        CXX_NULL_PTR_LITERAL_EXPR = object()

    class _FakeTokenKind:
        IDENTIFIER = object()
        LITERAL = object()

    class _FakeClang:
        CursorKind = _FakeCursorKind
        TokenKind = _FakeTokenKind

    class _FakeToken:
        def __init__(self, kind: object, spelling: str) -> None:
            self.kind = kind
            self.spelling = spelling

    class _FakeElement:
        def __init__(self, tokens: list[_FakeToken], children: list[object]) -> None:
            self._tokens = tokens
            self._children = children

        def get_tokens(self) -> list[_FakeToken]:
            return self._tokens

        def get_children(self) -> list[object]:
            return self._children

    engine._clang = _FakeClang

    c_style_end = _FakeElement(
        tokens=[
            _FakeToken(_FakeTokenKind.LITERAL, "0"),
            _FakeToken(_FakeTokenKind.LITERAL, "0"),
            _FakeToken(_FakeTokenKind.LITERAL, "0"),
            _FakeToken(_FakeTokenKind.LITERAL, "0"),
        ],
        children=[],
    )
    not_end = _FakeElement(
        tokens=[
            _FakeToken(_FakeTokenKind.LITERAL, '"add"'),
            _FakeToken(_FakeTokenKind.IDENTIFIER, "add_impl"),
        ],
        children=[],
    )

    assert engine._is_end_array_element(c_style_end) is True
    assert engine._is_end_array_element(not_end) is False


def test_c_signature_engine_parses_keywords_with_non_kwlist_name(tmp_path: Path) -> None:
    engine = CSignatureExtractionEngine(source_root=tmp_path)

    args = engine._set_token_params(
        func_cursor=object(),
        meth_flags=["METH_VARARGS", "METH_KEYWORDS"],
        token_list=[
            "PyArg_ParseTupleAndKeywords",
            "args",
            "kwargs",
            '"iO!"',
            "keywords",
            "count",
            "expected_type",
            "value",
        ],
    )

    assert args is not None
    assert [arg.name for arg in args] == ["count", "expected_type", "value"]
    assert [arg.type_name for arg in args] == ["int", "object", "object"]

