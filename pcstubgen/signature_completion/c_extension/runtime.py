from __future__ import annotations

import ctypes
from dataclasses import dataclass


@dataclass(frozen=True)
class BuiltinFunctionRuntimeInfo:
    address: int
    flags: int


_pycfunction_get_function = ctypes.pythonapi.PyCFunction_GetFunction
_pycfunction_get_function.argtypes = [ctypes.py_object]
_pycfunction_get_function.restype = ctypes.c_void_p

_pycfunction_get_flags = ctypes.pythonapi.PyCFunction_GetFlags
_pycfunction_get_flags.argtypes = [ctypes.py_object]
_pycfunction_get_flags.restype = ctypes.c_int


def supports_builtin_function_inference(handle: object) -> bool:
    """判断运行时对象是否支持 builtin function C 源码签名推导。"""
    handle_type = type(handle)
    return (
        handle_type.__module__ == "builtins"
        and handle_type.__name__ == "builtin_function_or_method"
        and not _is_pybind11_bound_builtin(handle)
    )


def read_builtin_function_runtime_info(handle: object) -> BuiltinFunctionRuntimeInfo:
    """读取 builtin function 的入口地址与调用约定。"""
    if not supports_builtin_function_inference(handle):
        raise RuntimeError(f"不支持的 builtin function 对象: {type(handle).__name__}")

    try:
        method_address = int(_pycfunction_get_function(handle))
        flags = int(_pycfunction_get_flags(handle))
    except (SystemError, TypeError) as ex:
        raise RuntimeError("读取 builtin function 运行时信息失败。") from ex

    if method_address == 0:
        raise RuntimeError("读取 builtin function 运行时信息失败: C函数地址为空。")

    return BuiltinFunctionRuntimeInfo(
        address=method_address,
        flags=flags,
    )


def _is_pybind11_bound_builtin(handle: object) -> bool:
    self_obj = getattr(handle, "__self__", None)
    if self_obj is None:
        return False

    self_type = type(self_obj)
    return self_type.__module__.startswith("pybind11_builtins")
