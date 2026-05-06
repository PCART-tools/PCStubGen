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


@pytest.mark.parametrize("self_obj", [object(), len.__self__])
def test_runtime_introspection_identifies_non_pybind11_self_as_false(
    self_obj: object,
) -> None:
    assert _pybind11_runtime.is_pybind11_self(self_obj) is False


def test_runtime_introspection_supports_real_scipy_pybind11_function() -> None:
    scipy = pytest.importorskip("scipy")
    _ = scipy
    pocketfft = pytest.importorskip("scipy.fft._pocketfft.pypocketfft")

    assert _pybind11_runtime.is_pybind11_self(pocketfft.c2c.__self__) is True
    signatures = runtime_introspection.extract_pybind11_signatures(pocketfft.c2c)

    assert signatures
    assert signatures[0].startswith("(")
    assert "->" in signatures[0]
