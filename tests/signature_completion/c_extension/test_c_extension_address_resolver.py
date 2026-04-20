from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

from pcstubgen.signature_completion.c_extension import dladdr as resolver_module


def test_get_binary_and_ra_returns_binary_path_and_relative_address(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    binary_path = tmp_path / "sample.so"
    base_address = 0x1000
    address = 0x1234

    def fake_dladdr(pointer: ctypes.c_void_p, info_ptr: object) -> int:
        _ = pointer
        info_ptr._obj.dli_fname = str(binary_path).encode()
        info_ptr._obj.dli_fbase = base_address
        return 1

    monkeypatch.setattr(resolver_module, "_dladdr", fake_dladdr)

    resolved_binary_path, relative_address = resolver_module.get_binary_and_ra(address)

    assert resolved_binary_path == binary_path.resolve()
    assert relative_address == address - base_address


def test_get_binary_and_ra_rejects_unresolved_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resolver_module, "_dladdr", lambda *args: 0)

    with pytest.raises(RuntimeError, match="无法定位函数地址所属共享库"):
        resolver_module.get_binary_and_ra(0x1234)


def test_get_binary_and_ra_rejects_incomplete_shared_library_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resolver_module, "_dladdr", lambda *args: 1)

    with pytest.raises(RuntimeError, match="共享库位置信息不完整"):
        resolver_module.get_binary_and_ra(0x1234)
