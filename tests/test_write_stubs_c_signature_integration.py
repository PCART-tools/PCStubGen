from __future__ import annotations

from tests._c_signature_test_support import *


def test_write_stubs_propagates_extract_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pcstubgen as stubgen_module
    from pcstubgen.stub_generation_options import StubGenerationOptions

    ir_module = IRModule(
        full_name=QualifiedName.from_str("pkg"),
        module_type=IRModuleType.EXTENSION,
        functions=[_unknown_function("foo")],
        classes=[
            IRClass(
                name="Builder",
                methods=[IRMethod(function=_unknown_function("build"), decorator=None)],
            )
        ],
        sub_modules=[
            IRModule(
                full_name=QualifiedName.from_str("pkg.child"),
                module_type=IRModuleType.EXTENSION,
                functions=[_unknown_function("bar")],
            )
        ],
    )
    monkeypatch.setattr(stubgen_module, "build_module", lambda path, module: ir_module)
    _patch_raising_c_signature_extractor(monkeypatch, RuntimeError("boom"))

    options = StubGenerationOptions(
        enable_docstring_signature_parser=False,
        source_root=tmp_path,
    )
    with pytest.raises(RuntimeError, match="boom"):
        stubgen_module.write_stubs("math", tmp_path, options=options)


def test_write_stubs_passes_c_inferred_source_comment_option(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pcstubgen as stubgen_module

    captured: dict[str, object] = {}
    ir_module = IRModule(full_name=QualifiedName.from_str("pkg.mod"))

    class FakeCSignatureVisitor:
        def __init__(
            self,
            *,
            source_root: Path,
            include: list[str] = (),
            include_directory: list[Path] = (),
            c_std: str = "c11",
            cpp_std: str = "c++17",
            include_c_inferred_source_comment: bool = False,
        ) -> None:
            captured["visitor_source_root"] = source_root
            captured["visitor_include"] = list(include)
            captured["visitor_include_directory"] = list(include_directory)
            captured["visitor_c_std"] = c_std
            captured["visitor_cpp_std"] = cpp_std
            captured["visitor_include_c_inferred_source_comment"] = (
                include_c_inferred_source_comment
            )

        def visit_module(self, node: IRModule) -> None:
            _ = node

        def visit_class(self, node: IRClass, module: IRModule) -> None:
            _ = (node, module)

        def visit_function(self, node: IRFunction, module: IRModule) -> None:
            _ = (node, module)

        def visit_method(self, node: IRMethod, module: IRModule) -> None:
            _ = (node, module)

        def log_summary(self) -> None:
            captured["visitor_log_summary_called"] = True

    class FakeStubRenderer:
        def __init__(
            self,
            include_docstrings: bool = True,
            include_module_type_comment: bool = False,
            include_c_inferred_source_comment: bool = False,
        ) -> None:
            captured["printer_include_docstrings"] = include_docstrings
            captured["printer_include_module_type_comment"] = include_module_type_comment
            captured["printer_include_c_inferred_source_comment"] = (
                include_c_inferred_source_comment
            )

        def print_module(self, node: IRModule) -> list[str]:
            _ = node
            return []

    class FakeWriter:
        def write(
            self,
            module: IRModule,
            renderer: FakeStubRenderer,
            to: Path,
        ) -> None:
            captured["written_module"] = module
            captured["written_renderer"] = renderer
            captured["written_to"] = to

    monkeypatch.setattr(stubgen_module, "build_module", lambda path, module: ir_module)
    monkeypatch.setattr(
        stubgen_module,
        "CSignatureVisitor",
        FakeCSignatureVisitor,
    )
    monkeypatch.setattr(stubgen_module, "StubRenderer", FakeStubRenderer)

    options = StubGenerationOptions(
        source_root=tmp_path,
        include=["Python.h"],
        include_directory=[tmp_path / "include"],
        c_std="c99",
        cpp_std="c++20",
        include_docstrings=False,
        include_module_type_comment=True,
        include_c_inferred_source_comment=True,
    )

    stubgen_module.write_stubs(
        "math",
        tmp_path,
        options=options,
        _writer=FakeWriter(),
    )

    assert captured["visitor_source_root"] == tmp_path
    assert captured["visitor_include"] == ["Python.h"]
    assert captured["visitor_include_directory"] == [tmp_path / "include"]
    assert captured["visitor_c_std"] == "c99"
    assert captured["visitor_cpp_std"] == "c++20"
    assert captured["visitor_include_c_inferred_source_comment"] is True
    assert captured["visitor_log_summary_called"] is True
    assert captured["printer_include_docstrings"] is False
    assert captured["printer_include_module_type_comment"] is True
    assert captured["printer_include_c_inferred_source_comment"] is True
    assert captured["written_module"] is ir_module
    assert captured["written_to"] == tmp_path


def test_doc_parser_preserves_rewritten_signature_without_c_ast_candidates(
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
    extractor = _patch_c_signature_extractor(monkeypatch, modules={})

    Pipeline(
        [
            DocstringSignatureVisitor(),
            CSignatureVisitor(
                source_root=tmp_path,
            ),
        ]
    ).run(module)

    parsed = module.functions[0]
    assert [arg.name for arg in parsed.signatures[0].args] == ["x", "y", "w", "out", "p"]
    assert parsed.signatures[0].return_type_name == "numpy.ndarray"
    assert extractor.called == 1
