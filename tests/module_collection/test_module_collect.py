from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
import types

import pytest

from pcstubgen.ir_modules import QualifiedName
import pcstubgen.module_collect as module_collect_module
from pcstubgen.module_collect import collect_function, collect_module


def test_module_collect_collect_function_keeps_doc_without_completing_signatures() -> None:
    def sample(value: int, flag: bool = False) -> int:
        """sample doc"""
        raise NotImplementedError

    parsed = collect_function(QualifiedName.from_str("pkg.mod.sample"), sample)

    assert parsed.doc == "sample doc"
    assert parsed.signatures == []
    assert parsed.runtime_handle is sample


def test_module_collect_keeps_only_tree_functions_and_methods() -> None:
    module = types.ModuleType("math")
    module.sin = math.sin
    module.sin_alias = math.sin

    class RootClass:
        class_attr = 123

        @property
        def prop(self):
            return 1

    RootClass.__module__ = "math"
    RootClass.sin = math.sin
    RootClass.sin_alias = math.sin

    module.RootClass = RootClass
    module.sub = types.ModuleType("math.sub")
    module.VALUE = 10

    ir_module = collect_module(QualifiedName.from_str("math"), module)

    assert [func.name for func in ir_module.functions] == ["sin"]
    assert [cls.name for cls in ir_module.classes] == ["RootClass"]
    assert ir_module.sub_modules == []

    root_cls = ir_module.classes[0]
    assert [method.function.name for method in root_cls.methods] == ["sin"]
    assert root_cls.methods[0].function.runtime_handle is math.sin
    assert root_cls.methods[0].runtime_owner is RootClass
    assert root_cls.classes == []

    # 精简后的 IR 不再携带变量/属性/字段等节点
    assert not hasattr(ir_module, "variables")
    assert not hasattr(root_cls, "properties")
    assert not hasattr(root_cls, "fields")


def test_module_collect_collects_extension_method_descriptors() -> None:
    ir_class = module_collect_module.collect_class(
        QualifiedName.from_str("builtins.list"),
        list,
        module_type=module_collect_module.IRModuleType.EXTENSION,
    )

    method_names = [method.function.name for method in ir_class.methods]

    assert "append" in method_names
    append_method = next(method for method in ir_class.methods if method.function.name == "append")
    assert type(append_method.function.runtime_handle).__name__ == "method_descriptor"
    assert append_method.runtime_owner is list


def test_module_collect_discovers_direct_submodules_from_package_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_package_file(
        tmp_path / "samplepkg" / "__init__.py",
        "def root() -> int:\n    return 1\n",
    )
    _write_package_file(
        tmp_path / "samplepkg" / "public_mod.py",
        "def public_func() -> int:\n    return 1\n",
    )
    _write_package_file(
        tmp_path / "samplepkg" / "_private_mod.py",
        "def private_func() -> int:\n    return 1\n",
    )
    _write_package_file(
        tmp_path / "samplepkg" / "subpackage" / "__init__.py",
        "def nested() -> int:\n    return 1\n",
    )

    module = _import_module_from_tmp("samplepkg", tmp_path, monkeypatch)

    ir_module = collect_module(QualifiedName.from_str("samplepkg"), module)

    assert ir_module.functions == []
    assert [sub_mod.full_name.name for sub_mod in ir_module.sub_modules] == [
        "_private_mod",
        "public_mod",
        "subpackage",
    ]


def test_module_collect_discovers_hidden_private_subpackage_not_exposed_by_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_package_file(
        tmp_path / "hiddenpkg" / "__init__.py",
        "__all__ = ['public_mod']\n"
        "def __dir__():\n"
        "    return __all__\n",
    )
    _write_package_file(
        tmp_path / "hiddenpkg" / "public_mod.py",
        "VALUE = 1\n",
    )
    _write_package_file(
        tmp_path / "hiddenpkg" / "_hidden" / "__init__.py",
        "VALUE = 2\n",
    )

    module = _import_module_from_tmp("hiddenpkg", tmp_path, monkeypatch)

    assert "_hidden" not in dir(module)

    ir_module = collect_module(QualifiedName.from_str("hiddenpkg"), module)

    assert [sub_mod.full_name.name for sub_mod in ir_module.sub_modules] == [
        "_hidden",
        "public_mod",
    ]


def test_module_collect_ignores_module_attributes_not_on_package_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_package_file(
        tmp_path / "attrpkg" / "__init__.py",
        "import types\n"
        "external = types.ModuleType('attrpkg.external_alias')\n",
    )

    module = _import_module_from_tmp("attrpkg", tmp_path, monkeypatch)

    ir_module = collect_module(QualifiedName.from_str("attrpkg"), module)

    assert ir_module.sub_modules == []


def test_module_collect_treats_single_file_module_as_non_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = tmp_path / "singlemod.py"
    module_path.write_text("VALUE = 1\n", encoding="utf-8")

    module = _import_module_from_tmp("singlemod", tmp_path, monkeypatch)

    ir_module = collect_module(QualifiedName.from_str("singlemod"), module)

    assert ir_module.is_package is False
    assert ir_module.sub_modules == []


def test_module_collect_supports_namespace_packages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace_root = tmp_path / "namespacepkg"
    namespace_root.mkdir(parents=True, exist_ok=True)
    _write_package_file(
        namespace_root / "child.py",
        "VALUE = 1\n",
    )

    module = _import_module_from_tmp("namespacepkg", tmp_path, monkeypatch)

    ir_module = collect_module(QualifiedName.from_str("namespacepkg"), module)

    assert ir_module.is_package is True
    assert [sub_mod.full_name.name for sub_mod in ir_module.sub_modules] == ["child"]


def test_module_collect_skips_submodule_when_dependency_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_package_file(
        tmp_path / "optionalpkg" / "__init__.py",
        "",
    )
    _write_package_file(
        tmp_path / "optionalpkg" / "healthy.py",
        "VALUE = 1\n",
    )
    _write_package_file(
        tmp_path / "optionalpkg" / "needs_missing_dep.py",
        "import definitely_missing_dependency\n",
    )

    module = _import_module_from_tmp("optionalpkg", tmp_path, monkeypatch)
    error_records: list[tuple[str, str, str]] = []

    def fake_error(
        message: str,
        module_name: str,
        error: BaseException,
    ) -> None:
        _ = message
        error_records.append((module_name, type(error).__name__, str(error)))

    monkeypatch.setattr(module_collect_module.logger, "error", fake_error)

    ir_module = collect_module(QualifiedName.from_str("optionalpkg"), module)

    assert [sub_mod.full_name.name for sub_mod in ir_module.sub_modules] == ["healthy"]
    assert error_records == [
        (
            "optionalpkg.needs_missing_dep",
            "ModuleNotFoundError",
            "No module named 'definitely_missing_dependency'",
        )
    ]


def test_module_collect_skips_submodule_when_import_raises_os_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_package_file(
        tmp_path / "oserrorpkg" / "__init__.py",
        "",
    )
    _write_package_file(
        tmp_path / "oserrorpkg" / "healthy.py",
        "VALUE = 1\n",
    )
    _write_package_file(
        tmp_path / "oserrorpkg" / "broken.py",
        "raise OSError('dll load failed')\n",
    )

    module = _import_module_from_tmp("oserrorpkg", tmp_path, monkeypatch)
    error_records: list[tuple[str, str, str]] = []

    def fake_error(
        message: str,
        module_name: str,
        error: BaseException,
    ) -> None:
        _ = message
        error_records.append((module_name, type(error).__name__, str(error)))

    monkeypatch.setattr(module_collect_module.logger, "error", fake_error)

    ir_module = collect_module(QualifiedName.from_str("oserrorpkg"), module)

    assert [sub_mod.full_name.name for sub_mod in ir_module.sub_modules] == ["healthy"]
    assert error_records == [
        ("oserrorpkg.broken", "OSError", "dll load failed")
    ]


def _write_package_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _import_module_from_tmp(
    module_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> types.ModuleType:
    monkeypatch.syspath_prepend(str(tmp_path))
    _clear_modules(module_name)
    importlib.invalidate_caches()
    return importlib.import_module(module_name)


def _clear_modules(module_name: str) -> None:
    stale_modules = [
        loaded_name
        for loaded_name in sys.modules
        if loaded_name == module_name or loaded_name.startswith(f"{module_name}.")
    ]
    for loaded_name in stale_modules:
        sys.modules.pop(loaded_name, None)
