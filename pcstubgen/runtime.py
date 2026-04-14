from __future__ import annotations

import ctypes
import types
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


def is_cpython_builtin(handle: object) -> bool:
    """判断运行时对象是否支持 builtin function C 源码签名推导。"""
    return isinstance(handle, types.BuiltinFunctionType) and is_module_bound(handle)

def is_module_bound(handle: object) -> bool:
    self_obj = getattr(handle, "__self__", None)
    if self_obj is not None and isinstance(self_obj, types.ModuleType):
        return True
    return False

def is_pybind11_builtin(handle: object) -> bool:
    """判断运行时函数句柄是否为 pybind11 绑定函数。"""
    return isinstance(handle, types.BuiltinFunctionType) and is_pybind11_bound(handle)


def is_pybind11_bound(handle: object) -> bool:
    self_obj = getattr(handle, "__self__", None)
    if self_obj is not None and type(self_obj).__module__ == "pybind11_builtins":
        return True
    return False

def read_builtin_function_runtime_info(handle: object) -> BuiltinFunctionRuntimeInfo:
    """读取 builtin function 的入口地址与调用约定。"""
    if not is_cpython_builtin(handle):
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