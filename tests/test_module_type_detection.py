from __future__ import annotations

import importlib.machinery
import types

from pcstubgen.ir import IRModuleType, QualifiedName
from pcstubgen.module_build import build_module


def _module_with_loader(
    name: str,
    loader: object,
    *,
    file_path: str | None = None,
) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = types.SimpleNamespace(
        name=name,
        loader=loader,
        submodule_search_locations=None,
    )
    if file_path is not None:
        module.__file__ = file_path
    return module


def test_detect_module_type_uses_loader_mapping() -> None:
    source_loader = importlib.machinery.SourceFileLoader("demo_source", "demo_source.py")
    sourceless_loader = importlib.machinery.SourcelessFileLoader(
        "demo_sourceless", "demo_sourceless.pyc"
    )
    extension_loader = importlib.machinery.ExtensionFileLoader(
        "demo_extension", "demo_extension.pyd"
    )

    cases = [
        ("demo_builtin", importlib.machinery.BuiltinImporter, IRModuleType.BUILTIN),
        ("demo_extension", extension_loader, IRModuleType.EXTENSION),
        ("demo_sourceless", sourceless_loader, IRModuleType.PYTHON),
        ("demo_source", source_loader, IRModuleType.PYTHON),
    ]

    for module_name, loader, expected in cases:
        module = _module_with_loader(module_name, loader)
        ir_module = build_module(QualifiedName.from_str(module_name), module)
        assert ir_module.module_type == expected


def test_detect_module_type_returns_unknown_for_unrecognized_loader() -> None:
    native_module = _module_with_loader("native_mod", loader=None, file_path="native_mod.pyd")
    native_ir = build_module(QualifiedName.from_str("native_mod"), native_module)
    assert native_ir.module_type == IRModuleType.UNKNOWN

    py_module = _module_with_loader("py_mod", loader=None, file_path="py_mod.py")
    py_ir = build_module(QualifiedName.from_str("py_mod"), py_module)
    assert py_ir.module_type == IRModuleType.UNKNOWN
