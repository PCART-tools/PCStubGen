from __future__ import annotations

import ctypes

import pytest

from pcstubgen.signature_completion.pybind11 import runtime_introspection

_pybind11_runtime = pytest.importorskip(
    "pcstubgen.signature_completion.pybind11._pybind11_runtime"
)


def _new_instance_method(function: object) -> object:
    new_instance_method = ctypes.pythonapi.PyInstanceMethod_New
    new_instance_method.argtypes = [ctypes.py_object]
    new_instance_method.restype = ctypes.py_object
    return new_instance_method(function)


def test_runtime_introspection_returns_none_for_plain_object() -> None:
    with pytest.raises(RuntimeError, match="不是 PyCFunction"):
        runtime_introspection.extract_pybind11_signatures(object())


def test_runtime_introspection_returns_none_for_builtin_function() -> None:
    with pytest.raises(RuntimeError, match="不支持的 pybind11 self 布局"):
        _pybind11_runtime.extract_signatures(len)


def test_runtime_introspection_handles_instance_method_wrapper() -> None:
    handle = _new_instance_method(len)

    with pytest.raises(RuntimeError, match="不支持的 pybind11 self 布局"):
        _pybind11_runtime.extract_signatures(handle)


def test_runtime_introspection_returns_native_result(
    monkeypatch,
) -> None:
    fake_module = type(
        "_FakePybind11Runtime",
        (),
        {"extract_signatures": staticmethod(lambda obj: ["(value: int) -> int"])},
    )()
    monkeypatch.setattr(runtime_introspection, "_pybind11_runtime", fake_module)

    assert runtime_introspection.extract_pybind11_signatures(object()) == [
        "(value: int) -> int"
    ]
