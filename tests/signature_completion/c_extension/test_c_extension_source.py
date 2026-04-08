from __future__ import annotations

import clang.cindex
import pytest

from pcstubgen.ir_modules import IRArgumentKind, IRFunction, IRModule, IRModuleType, QualifiedName
from pcstubgen.signature_completion.c_extension.dwarf import ResolvedAddressLocation
from pcstubgen.signature_completion.c_extension.models import CSignature
from pcstubgen.signature_completion.c_extension.modules.method_flags import (
    METH_KEYWORDS,
    METH_NOARGS,
    METH_O,
    METH_VARARGS,
)
from pcstubgen.signature_completion.c_extension.runtime import RuntimePyMethodDef
from pcstubgen.signature_completion.c_extension.source import CExtensionSource
from pcstubgen.types import AnyType, RawType
from tests._c_extension_test_support import _FakeNode, _arg, _extent_for_source_snippet


def test_c_extension_source_prefers_ast_inference_and_preserves_source_comment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    source_path = tmp_path / "foo_impl.c"
    snippet = "\n".join(
        [
            "static PyObject* foo_impl(PyObject* self, PyObject* args) {",
            "    return (PyObject*)0;",
            "}",
        ]
    )
    source_path.write_text(snippet, encoding="utf-8", newline="\n")
    function_cursor = _FakeNode(
        kind=clang.cindex.CursorKind.FUNCTION_DECL,
        spelling="foo_impl",
        extent=_extent_for_source_snippet(source_path, snippet),
    )

    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.resolve_runtime_pymethoddef",
        lambda handle: RuntimePyMethodDef(
            name="foo",
            method_address=0x1234,
            flags=METH_VARARGS,
            doc=None,
            handle=handle,
        ),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.resolve_address_location",
        lambda address: ResolvedAddressLocation(
            binary_path=tmp_path / "sample.so",
            relative_address=address,
            source_path=source_path,
            source_line=1,
        ),
    )
    monkeypatch.setattr(
        CExtensionSource,
        "_resolve_function_cursor",
        lambda self, **kwargs: function_cursor,
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.inference.infer_signature",
        lambda c_function: [
            CSignature(
                arguments=[_arg("value", "int")],
                return_type=RawType("bool"),
            )
        ],
    )

    source = CExtensionSource(compilation_database=tmp_path / "compile_commands.json")
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[IRFunction(name="foo", runtime_handle=object())],
    )

    signatures, source_comment = source.resolve_function(module, module.functions[0])

    assert signatures[0].args[0].name == "value"
    assert signatures[0].args[0].type is not None
    assert signatures[0].args[0].type.render() == "int"
    assert signatures[0].return_type is not None
    assert signatures[0].return_type.render() == "bool"
    assert source_comment == snippet


def test_c_extension_source_uses_minimal_noargs_signature_without_ast(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.resolve_runtime_pymethoddef",
        lambda handle: RuntimePyMethodDef(
            name="foo",
            method_address=0x1234,
            flags=METH_NOARGS,
            doc=None,
            handle=handle,
        ),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.resolve_address_location",
        lambda address: ResolvedAddressLocation(
            binary_path=tmp_path / "sample.so",
            relative_address=address,
        ),
    )

    source = CExtensionSource()
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[IRFunction(name="foo", runtime_handle=object())],
    )

    signatures, source_comment = source.resolve_function(module, module.functions[0])

    assert signatures[0].args == []
    assert source_comment is None


def test_c_extension_source_uses_minimal_o_signature_without_ast(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.resolve_runtime_pymethoddef",
        lambda handle: RuntimePyMethodDef(
            name="foo",
            method_address=0x1234,
            flags=METH_O,
            doc=None,
            handle=handle,
        ),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.resolve_address_location",
        lambda address: ResolvedAddressLocation(
            binary_path=tmp_path / "sample.so",
            relative_address=address,
        ),
    )

    source = CExtensionSource()
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[IRFunction(name="foo", runtime_handle=object())],
    )

    signatures, _ = source.resolve_function(module, module.functions[0])

    assert signatures[0].args[0].name == "arg"
    assert signatures[0].args[0].kind is IRArgumentKind.POSITIONAL_ONLY
    assert signatures[0].args[0].type == AnyType()


def test_c_extension_source_uses_minimal_varargs_keywords_shape_without_ast(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.resolve_runtime_pymethoddef",
        lambda handle: RuntimePyMethodDef(
            name="foo",
            method_address=0x1234,
            flags=METH_VARARGS | METH_KEYWORDS,
            doc=None,
            handle=handle,
        ),
    )
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.resolve_address_location",
        lambda address: ResolvedAddressLocation(
            binary_path=tmp_path / "sample.so",
            relative_address=address,
        ),
    )

    source = CExtensionSource()
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[IRFunction(name="foo", runtime_handle=object())],
    )

    signatures, _ = source.resolve_function(module, module.functions[0])

    assert [arg.name for arg in signatures[0].args] == ["args", "kwargs"]
    assert signatures[0].args[0].kind is IRArgumentKind.VAR_POSITIONAL
    assert signatures[0].args[1].kind is IRArgumentKind.VAR_KEYWORD


def test_c_extension_source_supports_method_descriptors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        "pcstubgen.signature_completion.c_extension.source.resolve_address_location",
        lambda address: ResolvedAddressLocation(
            binary_path=tmp_path / "sample.so",
            relative_address=address,
        ),
    )

    source = CExtensionSource()
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.EXTENSION,
        functions=[IRFunction(name="append", runtime_handle=list.__dict__["append"])],
    )

    signatures, _ = source.resolve_function(module, module.functions[0])

    assert signatures[0].args[0].name == "arg"
    assert signatures[0].args[0].kind is IRArgumentKind.POSITIONAL_ONLY


def test_c_extension_source_rejects_non_extension_modules() -> None:
    source = CExtensionSource()
    module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        module_type=IRModuleType.PYTHON,
        functions=[IRFunction(name="foo", runtime_handle=object())],
    )

    with pytest.raises(RuntimeError, match="不是扩展模块"):
        source.resolve_function(module, module.functions[0])
