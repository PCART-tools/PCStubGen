from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

from core import write_stubs
from core.ErrorCollector import ErrorCollector
from core.IR import QualifiedName
from core.ModuleBuilder import ModuleBuilder
from core.StubGenerationOptions import StubGenerationOptions


def test_module_builder_keeps_only_tree_functions_and_methods() -> None:
    module = types.ModuleType("pkg")

    def root_function(*args, **kwargs):
        return args, kwargs

    root_function.__module__ = "pkg"
    root_function.__doc__ = "root_function(x: int) -> int"

    class RootClass:
        class_attr = 123

        def method(*args, **kwargs):
            return args, kwargs

        @property
        def prop(self):
            return 1

    RootClass.__module__ = "pkg"
    RootClass.method.__doc__ = "method(self, y: int) -> int"

    sub = types.ModuleType("pkg.sub")

    def sub_function(*args, **kwargs):
        return args, kwargs

    sub_function.__module__ = "pkg.sub"
    sub.sub_function = sub_function

    module.root_function = root_function
    module.alias_function = root_function
    module.RootClass = RootClass
    module.sub = sub
    module.VALUE = 10

    ir_module = ModuleBuilder(ErrorCollector()).build_module(
        QualifiedName.from_str("pkg"), module
    )

    assert [func.name for func in ir_module.functions] == ["root_function"]
    assert [cls.name for cls in ir_module.classes] == ["RootClass"]
    assert [sub_mod.Name for sub_mod in ir_module.sub_modules] == ["sub"]

    root_cls = ir_module.classes[0]
    assert [method.function.name for method in root_cls.methods] == ["method"]
    assert root_cls.classes == []

    # 精简后的 IR 不再携带变量/属性/字段等节点
    assert not hasattr(ir_module, "variables")
    assert not hasattr(root_cls, "properties")
    assert not hasattr(root_cls, "fields")


def test_write_stubs_outputs_core_structure_only(tmp_path: Path) -> None:
    package_dir = tmp_path / "demo_pkg"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text(
        "\n".join(
            [
                "VALUE = 123",
                "",
                "def root_function(*args, **kwargs):",
                "    \"\"\"root_function(x: int) -> int\"\"\"",
                "    return 0",
                "",
                "class RootClass:",
                "    CLASS_FIELD = 3",
                "",
                "    def method(*args, **kwargs):",
                "        \"\"\"method(self, y: int) -> int\"\"\"",
                "        return 0",
                "",
                "    @property",
                "    def prop(self):",
                "        return 1",
                "",
                "from . import sub",
            ]
        ),
        encoding="utf-8",
    )
    (package_dir / "sub.py").write_text(
        "\n".join(
            [
                "def sub_function(*args, **kwargs):",
                "    \"\"\"sub_function(name: str) -> str\"\"\"",
                "    return name",
            ]
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "stubs"
    out_dir_with_comment = tmp_path / "stubs_with_comment"
    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    try:
        write_stubs("demo_pkg", out_dir, options=StubGenerationOptions())
        write_stubs(
            "demo_pkg",
            out_dir_with_comment,
            options=StubGenerationOptions(include_module_type_comment=True),
        )
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("demo_pkg", None)
        sys.modules.pop("demo_pkg.sub", None)

    root_stub = (out_dir / "demo_pkg" / "__init__.pyi").read_text(encoding="utf-8")
    sub_stub = (out_dir / "demo_pkg" / "sub.pyi").read_text(encoding="utf-8")
    root_stub_with_comment = (
        out_dir_with_comment / "demo_pkg" / "__init__.pyi"
    ).read_text(encoding="utf-8")
    sub_stub_with_comment = (out_dir_with_comment / "demo_pkg" / "sub.pyi").read_text(
        encoding="utf-8"
    )

    assert not root_stub.startswith("# module type:")
    assert not sub_stub.startswith("# module type:")

    assert "from . import sub" in root_stub
    assert "def root_function(x: int) -> int:" in root_stub
    assert "class RootClass:" in root_stub
    assert "def method(self, y: int) -> int:" in root_stub
    assert "VALUE" not in root_stub
    assert "CLASS_FIELD" not in root_stub
    assert "prop" not in root_stub

    assert "def sub_function(name: str) -> str:" in sub_stub
    assert root_stub_with_comment.startswith("# module type: python\n")
    assert sub_stub_with_comment.startswith("# module type: python\n")
