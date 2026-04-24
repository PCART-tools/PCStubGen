from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pcstubgen.signature_completion.c_extension import provider as provider_module
from pcstubgen.signature_completion.c_extension.dwarfdump import LookupResult


def test_get_func_cursor_and_flags_uses_dwarf_manager(
    monkeypatch,
    tmp_path: Path,
) -> None:
    binary_path = tmp_path / "sample.so"
    source_path = tmp_path / "sample.c"
    expected_cursor = object()
    calls: list[tuple[Path, int]] = []

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
