from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

import pcstubgen.module_collector as module_collector_module
from pcstubgen.module_collector import ModuleCollector
from pcstubgen.models import QualifiedName


def test_module_collector_discovers_direct_submodules_from_package_path(
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

    _prepare_module_import("samplepkg", tmp_path, monkeypatch)
    module_node = ModuleCollector().run("samplepkg")

    assert module_node.functions == []
    assert [sub_mod.full_name.name for sub_mod in module_node.sub_modules] == [
        "_private_mod",
        "public_mod",
        "subpackage",
    ]


def test_module_collector_discovers_hidden_private_subpackage_not_exposed_by_dir(
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

    _prepare_module_import("hiddenpkg", tmp_path, monkeypatch)
    module_node = ModuleCollector().run("hiddenpkg")

    assert [sub_mod.full_name.name for sub_mod in module_node.sub_modules] == [
        "_hidden",
        "public_mod",
    ]


def test_module_collector_ignores_module_attributes_not_on_package_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_package_file(
        tmp_path / "attrpkg" / "__init__.py",
        "import types\n"
        "external = types.ModuleType('attrpkg.external_alias')\n",
    )

    _prepare_module_import("attrpkg", tmp_path, monkeypatch)
    module_node = ModuleCollector().run("attrpkg")

    assert module_node.sub_modules == []


def test_module_collector_ignores_plain_members_without_module_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_package_file(
        tmp_path / "plainattrpkg" / "__init__.py",
        "VALUE = {'answer': 42}\n",
    )

    _prepare_module_import("plainattrpkg", tmp_path, monkeypatch)
    module_node = ModuleCollector().run("plainattrpkg")

    assert module_node.functions == []
    assert module_node.classes == []
    assert module_node.sub_modules == []


def test_module_collector_treats_single_file_module_as_non_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = tmp_path / "singlemod.py"
    module_path.write_text("VALUE = 1\n", encoding="utf-8")

    _prepare_module_import("singlemod", tmp_path, monkeypatch)
    module_node = ModuleCollector().run("singlemod")

    assert module_node.is_package is False
    assert module_node.sub_modules == []


def test_module_collector_supports_namespace_packages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace_root = tmp_path / "namespacepkg"
    namespace_root.mkdir(parents=True, exist_ok=True)
    _write_package_file(
        namespace_root / "child.py",
        "VALUE = 1\n",
    )

    _prepare_module_import("namespacepkg", tmp_path, monkeypatch)
    module_node = ModuleCollector().run("namespacepkg")

    assert module_node.is_package is True
    assert [sub_mod.full_name.name for sub_mod in module_node.sub_modules] == ["child"]


@pytest.mark.parametrize(
    ("package_name", "broken_module_name", "broken_source", "expected_error_type"),
    [
        (
            "optionalpkg",
            "needs_missing_dep",
            "import definitely_missing_dependency\n",
            ModuleNotFoundError,
        ),
        (
            "oserrorpkg",
            "broken",
            "raise OSError('dll load failed')\n",
            OSError,
        ),
    ],
)
def test_module_collector_skips_submodule_when_submodule_import_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    package_name: str,
    broken_module_name: str,
    broken_source: str,
    expected_error_type: type[BaseException],
) -> None:
    _write_package_file(
        tmp_path / package_name / "__init__.py",
        "",
    )
    _write_package_file(
        tmp_path / package_name / "healthy.py",
        "VALUE = 1\n",
    )
    _write_package_file(
        tmp_path / package_name / f"{broken_module_name}.py",
        broken_source,
    )

    _prepare_module_import(package_name, tmp_path, monkeypatch)
    error_records: list[tuple[str, BaseException]] = []

    def fake_error(
        message: str,
        module_name: str,
        error: BaseException,
    ) -> None:
        _ = message
        error_records.append((module_name, error))

    monkeypatch.setattr(module_collector_module.logger, "error", fake_error)

    module_node = ModuleCollector().run(package_name)

    assert [sub_mod.full_name.name for sub_mod in module_node.sub_modules] == ["healthy"]
    assert len(error_records) == 1
    assert error_records[0][0] == f"{package_name}.{broken_module_name}"
    assert isinstance(error_records[0][1], expected_error_type)


def test_module_collector_does_not_swallow_base_exception_from_submodule_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_package_file(
        tmp_path / "interruptpkg" / "__init__.py",
        "",
    )
    _write_package_file(
        tmp_path / "interruptpkg" / "broken.py",
        "raise KeyboardInterrupt('stop')\n",
    )
    _write_package_file(
        tmp_path / "interruptpkg" / "healthy.py",
        "VALUE = 1\n",
    )

    _prepare_module_import("interruptpkg", tmp_path, monkeypatch)

    with pytest.raises(KeyboardInterrupt, match="stop"):
        ModuleCollector().run("interruptpkg")


def test_module_collector_collects_cpython_method_descriptor_from_builtin_type() -> None:
    class_node = ModuleCollector()._collect_class(
        QualifiedName.from_str("builtins.list"),
        list,
    )

    append_method = next(
        method for method in class_node.methods if method.function.name == "append"
    )

    assert append_method.decorator is None
    assert append_method.function.runtime_handle is list.__dict__["append"]
    assert "__new__" not in {method.function.name for method in class_node.methods}


def test_module_collector_collects_cpython_classmethod_descriptor_from_builtin_type() -> None:
    class_node = ModuleCollector()._collect_class(
        QualifiedName.from_str("builtins.dict"),
        dict,
    )

    fromkeys_method = next(
        method for method in class_node.methods if method.function.name == "fromkeys"
    )

    assert fromkeys_method.decorator == "classmethod"
    assert fromkeys_method.function.runtime_handle is dict.__dict__["fromkeys"]


def test_module_collector_collects_cpython_staticmethod_from_builtin_type() -> None:
    class_node = ModuleCollector()._collect_class(
        QualifiedName.from_str("builtins.str"),
        str,
    )

    maketrans_method = next(
        method for method in class_node.methods if method.function.name == "maketrans"
    )

    assert maketrans_method.decorator == "staticmethod"
    assert maketrans_method.function.runtime_handle is str.__dict__["maketrans"].__func__
    assert maketrans_method.function.doc is not None
    assert "translation table" in maketrans_method.function.doc
    assert "staticmethod(function)" not in maketrans_method.function.doc


def _write_package_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _prepare_module_import(
    module_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(tmp_path))
    _clear_modules(module_name)
    importlib.invalidate_caches()


def _clear_modules(module_name: str) -> None:
    stale_modules = [
        loaded_name
        for loaded_name in sys.modules
        if loaded_name == module_name or loaded_name.startswith(f"{module_name}.")
    ]
    for loaded_name in stale_modules:
        sys.modules.pop(loaded_name, None)
