from __future__ import annotations

import importlib
import sys
import typing
from pathlib import Path
import types

import pytest

from pcstubgen.ir import QualifiedName
import pcstubgen.module_builder as module_builder_module
from pcstubgen.module_builder import build_function, build_module


def test_module_builder_keeps_raw_annotation_strings() -> None:
    def sample(a: int, b: list[int]) -> typing.Optional[int]:
        raise NotImplementedError

    parsed = build_function(QualifiedName.from_str("pkg.mod.sample"), sample)

    assert len(parsed.signatures) == 1
    signature = parsed.signatures[0]
    assert [arg.type_name for arg in signature.args] == ["int", "list[int]"]
    assert signature.return_type_name == "typing.Optional[int]"


def test_module_builder_keeps_default_values_as_strings() -> None:
    def sample(
        flag: bool = False,
        values: tuple[int, int] = (1, 2),
    ) -> None:
        raise NotImplementedError

    parsed = build_function(QualifiedName.from_str("pkg.mod.sample"), sample)

    assert len(parsed.signatures) == 1
    signature = parsed.signatures[0]
    assert [arg.default_value for arg in signature.args] == ["False", "(1, 2)"]
    assert [arg.has_default for arg in signature.args] == [True, True]


def test_module_builder_uses_empty_signatures_when_inspect_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def sample() -> None:
        raise NotImplementedError

    def _raise_signature_error(obj: object) -> object:
        """模拟 inspect.signature 失败。"""
        raise TypeError(f"cannot inspect {obj!r}")

    monkeypatch.setattr(module_builder_module.inspect, "signature", _raise_signature_error)

    parsed = build_function(QualifiedName.from_str("pkg.mod.sample"), sample)

    assert parsed.signatures == []


def test_module_builder_discovers_direct_submodules_from_package_path(
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

    ir_module = build_module(QualifiedName.from_str("samplepkg"), module)

    assert [func.name for func in ir_module.functions] == ["root"]
    assert [sub_mod.full_name.name for sub_mod in ir_module.sub_modules] == [
        "_private_mod",
        "public_mod",
        "subpackage",
    ]


def test_module_builder_discovers_hidden_private_subpackage_not_exposed_by_dir(
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

    ir_module = build_module(QualifiedName.from_str("hiddenpkg"), module)

    assert [sub_mod.full_name.name for sub_mod in ir_module.sub_modules] == [
        "_hidden",
        "public_mod",
    ]


def test_module_builder_ignores_module_attributes_not_on_package_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_package_file(
        tmp_path / "attrpkg" / "__init__.py",
        "import types\n"
        "external = types.ModuleType('attrpkg.external_alias')\n",
    )

    module = _import_module_from_tmp("attrpkg", tmp_path, monkeypatch)

    ir_module = build_module(QualifiedName.from_str("attrpkg"), module)

    assert ir_module.sub_modules == []


def test_module_builder_treats_single_file_module_as_non_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = tmp_path / "singlemod.py"
    module_path.write_text("VALUE = 1\n", encoding="utf-8")

    module = _import_module_from_tmp("singlemod", tmp_path, monkeypatch)

    ir_module = build_module(QualifiedName.from_str("singlemod"), module)

    assert ir_module.is_package is False
    assert ir_module.sub_modules == []


def test_module_builder_supports_namespace_packages(
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

    ir_module = build_module(QualifiedName.from_str("namespacepkg"), module)

    assert ir_module.is_package is True
    assert [sub_mod.full_name.name for sub_mod in ir_module.sub_modules] == ["child"]


def test_module_builder_skips_submodule_when_dependency_is_missing(
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
    warning_messages: list[tuple[str, str, str | None]] = []

    def fake_warning(
        message: str,
        module_name: str,
        error_type: str,
        missing_dependency: str | None,
        error: str,
    ) -> None:
        warning_messages.append((message, error_type, missing_dependency))

    monkeypatch.setattr(module_builder_module.logger, "warning", fake_warning)

    ir_module = build_module(QualifiedName.from_str("optionalpkg"), module)

    assert [sub_mod.full_name.name for sub_mod in ir_module.sub_modules] == ["healthy"]
    assert warning_messages == [
        (
            "跳过子模块, module: {}, error_type: {}, missing_dependency: {}, error: {}",
            "ModuleNotFoundError",
            "definitely_missing_dependency",
        )
    ]


def test_module_builder_skips_submodule_when_import_raises_os_error(
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
    warning_messages: list[tuple[str, str, str | None]] = []

    def fake_warning(
        message: str,
        module_name: str,
        error_type: str,
        missing_dependency: str | None,
        error: str,
    ) -> None:
        warning_messages.append((message, error_type, missing_dependency))

    monkeypatch.setattr(module_builder_module.logger, "warning", fake_warning)

    ir_module = build_module(QualifiedName.from_str("oserrorpkg"), module)

    assert [sub_mod.full_name.name for sub_mod in ir_module.sub_modules] == ["healthy"]
    assert warning_messages == [
        (
            "跳过子模块, module: {}, error_type: {}, missing_dependency: {}, error: {}",
            "OSError",
            None,
        )
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
