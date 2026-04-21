from __future__ import annotations

import pytest

from pcstubgen import runtime as runtime_module
from pcstubgen.signature_completion.c_extension.method_flags import (
    METH_CLASS,
    METH_O,
    METH_STATIC,
)


@pytest.mark.parametrize(
    ("handle", "expected_flags"),
    [
        (len, METH_O),
        (dict.__dict__["fromkeys"], METH_CLASS),
    ],
)
def test_read_cpython_function_runtime_info_supports_representative_handles(
    handle: object,
    expected_flags: int,
) -> None:
    info = runtime_module.read_c_extension_function_runtime_info(handle)

    assert info.address != 0
    assert info.flags & expected_flags == expected_flags


@pytest.mark.parametrize(
    ("attr_name", "replacement"),
    [
        (
            "_pycfunction_get_function",
            lambda handle: (_ for _ in ()).throw(SystemError("bad argument")),
        ),
        (
            "_pycfunction_get_flags",
            lambda handle: (_ for _ in ()).throw(SystemError("bad argument")),
        ),
    ],
)
def test_read_cpython_function_runtime_info_propagates_cpython_api_errors(
    monkeypatch: pytest.MonkeyPatch,
    attr_name: str,
    replacement: object,
) -> None:
    monkeypatch.setattr(runtime_module, "_pycfunction_get_function", lambda handle: 0x1234)
    monkeypatch.setattr(runtime_module, "_pycfunction_get_flags", lambda handle: 8)
    monkeypatch.setattr(runtime_module, attr_name, replacement)

    with pytest.raises(SystemError, match="bad argument"):
        runtime_module.read_c_extension_function_runtime_info(len)


def test_read_cpython_function_runtime_info_rejects_zero_function_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_module, "_pycfunction_get_function", lambda handle: 0)
    monkeypatch.setattr(runtime_module, "_pycfunction_get_flags", lambda handle: 8)

    with pytest.raises(RuntimeError, match="C函数地址为空"):
        runtime_module.read_c_extension_function_runtime_info(len)


def test_read_cpython_function_runtime_info_rejects_unsupported_handle() -> None:
    with pytest.raises(RuntimeError, match="不支持的 CPython 函数对象"):
        runtime_module.read_c_extension_function_runtime_info(object())


def test_runtime_recognizes_pybind11_handle_kinds(monkeypatch: pytest.MonkeyPatch) -> None:
    pybind_record_type = type("pybind_record", (), {})
    pybind_record_type.__module__ = "pybind11_builtins"

    fake_instance_method_type = type("instancemethod", (), {})
    fake_instance_method_type.__module__ = "builtins"

    fake_instance_method = fake_instance_method_type()
    fake_instance_method.__self__ = pybind_record_type()

    fake_builtin = len
    fake_staticmethod = staticmethod(fake_builtin)
    monkeypatch.setattr(
        runtime_module,
        "is_pybind11_bound",
        lambda handle: handle is fake_builtin or handle is fake_instance_method,
    )

    assert runtime_module.is_pybind11_module_function(fake_builtin) is True
    assert runtime_module.is_pybind11_instance_method(fake_instance_method) is True
    assert runtime_module.is_pybind11_static_method(fake_staticmethod) is True
    assert runtime_module.is_pybind11_module_function(object()) is False
    assert runtime_module.is_pybind11_instance_method(object()) is False
    assert runtime_module.is_pybind11_static_method(object()) is False
