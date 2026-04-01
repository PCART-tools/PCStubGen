from __future__ import annotations

from tests._c_extension_test_support import *


def test_write_stubs_propagates_extract_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pcstubgen.api as stubgen_module

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

    options = StubGenerationOptions(source_root=tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        stubgen_module.write_stubs("math", tmp_path, options=options)


def test_write_stubs_passes_options_to_completer_and_logs_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pcstubgen.api as stubgen_module

    captured: dict[str, object] = {}
    ir_module = IRModule(full_name=QualifiedName.from_str("pkg.mod"))

    class FakeSummary:
        def __str__(self) -> str:
            return (
                "签名补全汇总: 函数总数=1, 跳过已有签名=0, C源码补全=0, "
                "文档字符串补全=0, 运行时反射补全=1, 未补全=0"
            )

    class FakeStubRenderer:
        def __init__(
            self,
            include_docstrings: bool = True,
            include_module_type_comment: bool = False,
            include_c_inferred_source_comment: bool = False,
        ) -> None:
            captured["renderer_include_docstrings"] = include_docstrings
            captured["renderer_include_module_type_comment"] = include_module_type_comment
            captured["renderer_include_c_inferred_source_comment"] = (
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

    class FakeSignatureCompleter:
        def __init__(self, options: StubGenerationOptions) -> None:
            captured["completion_options"] = options

        def run(self, module: IRModule) -> FakeSummary:
            captured["completed_module"] = module
            return FakeSummary()

    monkeypatch.setattr(stubgen_module, "build_module", lambda path, module: ir_module)
    monkeypatch.setattr(stubgen_module, "SignatureCompleter", FakeSignatureCompleter)
    monkeypatch.setattr(stubgen_module, "StubRenderer", FakeStubRenderer)
    monkeypatch.setattr(stubgen_module, "logger", SimpleNamespace(info=lambda *args: None))

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
        writer=FakeWriter(),
    )

    assert captured["completed_module"] is ir_module
    assert captured["completion_options"] is options
    assert captured["renderer_include_docstrings"] is False
    assert captured["renderer_include_module_type_comment"] is True
    assert captured["renderer_include_c_inferred_source_comment"] is True
    assert captured["written_module"] is ir_module
    assert isinstance(captured["written_renderer"], FakeStubRenderer)
    assert captured["written_to"] == tmp_path
