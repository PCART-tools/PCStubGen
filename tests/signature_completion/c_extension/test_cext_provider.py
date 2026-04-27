from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pcstubgen.models import QualifiedName
from pcstubgen.signature_completion.c_extension import provider as provider_module
from pcstubgen.signature_completion.c_extension.dwarfdump import LookupResult
from pcstubgen.signature_completion.completion_models import (
    SignatureCompletionContext,
    UnsupportedSignatureCompletion,
)


def test_get_func_cursor_and_flags_uses_dwarf_manager(
    monkeypatch,
    tmp_path: Path,
) -> None:
    binary_path = tmp_path / "sample.so"
    source_path = tmp_path / "sample.c"
    expected_cursor = object()
    calls: list[tuple[Path, int]] = []
    binary_path.write_bytes(b"")

    class FakeDwarfManager:
        def lookup(self, binary_path_arg: Path, relative_address: int) -> LookupResult:
            """记录 provider 发起的 DWARF 查询。"""
            calls.append((binary_path_arg, relative_address))
            return LookupResult(source_path, "foo_impl", "foo_linkage")

    class FakeClangFunctionLocator:
        def get_function_cursor(
            self,
            source_path_arg: Path,
            function_name: str,
            linkage_name: str | None,
        ) -> object:
            """校验 DWARF 结果被传给 libclang 函数定位器。"""
            assert source_path_arg == source_path
            assert function_name == "foo_impl"
            assert linkage_name == "foo_linkage"
            return expected_cursor

    monkeypatch.setattr(
        provider_module.runtime,
        "read_c_extension_function_runtime_info",
        lambda handle: SimpleNamespace(address=0x1234, flags=8),
    )
    monkeypatch.setattr(
        provider_module.dladdr,
        "get_binary_and_ra",
        lambda address: (binary_path, 0x234),
    )

    provider = object.__new__(provider_module.CExtensionProvider)
    provider._function_cursor_locator = FakeClangFunctionLocator()
    provider._dwarf_manager = FakeDwarfManager()

    func_cursor, flags = provider.get_func_cursor_and_flags(object())

    assert func_cursor is expected_cursor
    assert flags == 8
    assert calls == [(binary_path, 0x234)]


def test_match_rejects_cython_pickle_method_descriptor(monkeypatch) -> None:
    class Sample:
        """用于模拟 Cython 扩展类。"""

    method = SimpleNamespace(__name__="__reduce_cython__", __objclass__=Sample)
    monkeypatch.setattr(
        provider_module.runtime,
        "is_c_extension_instance_method",
        lambda member: member is method,
    )
    monkeypatch.setattr(
        provider_module.runtime,
        "is_c_extension_class_method",
        lambda member: False,
    )
    monkeypatch.setattr(
        provider_module.runtime,
        "is_c_extension_static_method",
        lambda member: False,
    )

    assert provider_module.CExtensionProvider.match(method, Sample) is False


def test_get_rejects_pythran_wrapall_cursor(monkeypatch) -> None:
    provider = object.__new__(provider_module.CExtensionProvider)
    runtime_handle = object()
    func_cursor = SimpleNamespace(spelling="__pythran_wrapall_group_dense")

    monkeypatch.setattr(
        provider,
        "_analyze_member",
        lambda member: (runtime_handle, None, None),
    )
    monkeypatch.setattr(
        provider,
        "get_func_cursor_and_flags",
        lambda handle: (func_cursor, 0),
    )

    context = SignatureCompletionContext(
        module_name=QualifiedName.from_str("pkg.mod"),
        func_name="group_dense",
        member=object(),
    )

    with pytest.raises(UnsupportedSignatureCompletion):
        provider.get(context)


def test_get_func_cursor_and_flags_rejects_current_interpreter_method_descriptor(
    monkeypatch,
) -> None:
    provider = object.__new__(provider_module.CExtensionProvider)
    provider._function_cursor_locator = object()
    provider._dwarf_manager = object()

    monkeypatch.setattr(
        provider_module.runtime,
        "read_c_extension_function_runtime_info",
        lambda handle: SimpleNamespace(address=0x1234, flags=8),
    )
    monkeypatch.setattr(
        provider_module.dladdr,
        "get_binary_and_ra",
        lambda address: (Path(sys.executable), 0x234),
    )

    with pytest.raises(UnsupportedSignatureCompletion):
        provider.get_func_cursor_and_flags(SimpleNamespace(__name__="__format__"))
