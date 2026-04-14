from __future__ import annotations

import pytest

from pcstubgen.signature_completion.c_extension import runtime as runtime_module


def test_read_builtin_function_runtime_info_rejects_unsupported_handle() -> None:
    with pytest.raises(RuntimeError, match="不支持的 builtin function 对象"):
        runtime_module.read_builtin_function_runtime_info(object())


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
def test_read_builtin_function_runtime_info_wraps_cpython_api_errors(
    monkeypatch: pytest.MonkeyPatch,
    attr_name: str,
    replacement: object,
) -> None:
    monkeypatch.setattr(runtime_module, "_pycfunction_get_function", lambda handle: 0x1234)
    monkeypatch.setattr(runtime_module, "_pycfunction_get_flags", lambda handle: 8)
    monkeypatch.setattr(runtime_module, attr_name, replacement)

    with pytest.raises(RuntimeError, match="读取 builtin function 运行时信息失败"):
        runtime_module.read_builtin_function_runtime_info(len)


def test_read_builtin_function_runtime_info_rejects_zero_function_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_module, "_pycfunction_get_function", lambda handle: 0)
    monkeypatch.setattr(runtime_module, "_pycfunction_get_flags", lambda handle: 8)

    with pytest.raises(RuntimeError, match="C函数地址为空"):
        runtime_module.read_builtin_function_runtime_info(len)


def test_supports_builtin_function_inference_rejects_pybind11_bound_builtin(
) -> None:
    class _PybindBoundSelf:
        __module__ = "pybind11_builtins.fake_module"

    _BuiltinFunctionLike = type(
        "builtin_function_or_method",
        (),
        {"__module__": "builtins"},
    )

    class _FakeBuiltinHandle(_BuiltinFunctionLike):
        __self__ = _PybindBoundSelf()

    assert runtime_module.is_cpython_builtin(_FakeBuiltinHandle()) is False
