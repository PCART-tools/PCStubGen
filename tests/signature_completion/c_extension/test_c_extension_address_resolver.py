from __future__ import annotations

from pathlib import Path

import pytest

from pcstubgen.signature_completion.c_extension import dladdr as resolver_module
from pcstubgen.signature_completion.c_extension.dwarfdump import LookupResult


def test_get_symbolized_address_location_delegates_to_dwarfdump_lookup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    binary_path = tmp_path / "sample.so"
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        resolver_module,
        "_get_binary_and_ra",
        lambda address: (binary_path, 0x234),
    )
    monkeypatch.setattr(
        resolver_module.dwarfdump,
        "lookup",
        lambda binary_path_arg, relative_address_arg: captured.update(
            binary_path=binary_path_arg,
            relative_address=relative_address_arg,
        )
        or LookupResult(
            compilation_unit_path=(tmp_path / "sample.c").resolve(),
            function_name="foo_impl",
            linkage_name="_Z8foo_implv",
        ),
    )

    result = resolver_module.get_func_file_location(0x1234)

    assert captured == {
        "binary_path": binary_path,
        "relative_address": 0x234,
    }
    assert result.compilation_unit_path == (tmp_path / "sample.c").resolve()
    assert result.function_name == "foo_impl"
    assert result.linkage_name == "_Z8foo_implv"


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
