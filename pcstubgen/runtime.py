from __future__ import annotations

import ctypes
import types
from dataclasses import dataclass


@dataclass(frozen=True)
class BuiltinFunctionRuntimeInfo:
    address: int
    flags: int


class _PyMethodDef(ctypes.Structure):
    _fields_ = [
        ("ml_name", ctypes.c_void_p),
        ("ml_meth", ctypes.c_void_p),
        ("ml_flags", ctypes.c_int),
        ("ml_doc", ctypes.c_void_p),
    ]


class _PyDescrObject(ctypes.Structure):
    _fields_ = [
        ("ob_refcnt", ctypes.c_ssize_t),
        ("ob_type", ctypes.c_void_p),
        ("d_type", ctypes.c_void_p),
        ("d_name", ctypes.c_void_p),
        ("d_qualname", ctypes.c_void_p),
    ]


class _PyMethodDescrObject(ctypes.Structure):
    _fields_ = [
        ("d_common", _PyDescrObject),
        ("d_method", ctypes.POINTER(_PyMethodDef)),
        ("vectorcall", ctypes.c_void_p),
    ]


_pycfunction_get_function = ctypes.pythonapi.PyCFunction_GetFunction
_pycfunction_get_function.argtypes = [ctypes.py_object]
_pycfunction_get_function.restype = ctypes.c_void_p

_pycfunction_get_flags = ctypes.pythonapi.PyCFunction_GetFlags
_pycfunction_get_flags.argtypes = [ctypes.py_object]
_pycfunction_get_flags.restype = ctypes.c_int


def is_cpython_builtin(handle: object) -> bool:
    """判断运行时对象是否支持 CPython C 扩展签名推导。"""
    if is_pybind11_builtin(handle):
        return False
    if isinstance(handle, types.BuiltinFunctionType):
        return True
    return is_cpython_method_descriptor(handle)


def is_cpython_method_descriptor(handle: object) -> bool:
    """判断运行时对象是否为 CPython 方法或类方法描述器。"""
    return isinstance(
        handle,
        (
            types.MethodDescriptorType,
            types.ClassMethodDescriptorType,
        ),
    )

def is_module_bound(handle: object) -> bool:
    """判断 builtin function 是否绑定到模块对象。"""
    self_obj = getattr(handle, "__self__", None)
    if self_obj is not None and isinstance(self_obj, types.ModuleType):
        return True
    return False

def is_pybind11_builtin(handle: object) -> bool:
    """判断运行时函数句柄是否为 pybind11 绑定函数。"""
    return isinstance(handle, types.BuiltinFunctionType) and is_pybind11_bound(handle)


def is_pybind11_bound(handle: object) -> bool:
    """判断 builtin function 是否绑定到 pybind11 对象。"""
    self_obj = getattr(handle, "__self__", None)
    if self_obj is not None and type(self_obj).__module__ == "pybind11_builtins":
        return True
    return False


def read_cpython_function_runtime_info(handle: object) -> BuiltinFunctionRuntimeInfo:
    """读取 CPython C 扩展函数句柄的入口地址与调用约定。"""
    if isinstance(handle, types.BuiltinFunctionType):
        return _read_builtin_function_runtime_info(handle)

    if is_cpython_method_descriptor(handle):
        return _read_method_descriptor_runtime_info(handle)

    raise RuntimeError(f"不支持的 CPython 函数对象: {type(handle).__name__}")


def read_builtin_function_runtime_info(handle: object) -> BuiltinFunctionRuntimeInfo:
    """读取 CPython C 扩展函数句柄的入口地址与调用约定。"""
    if not is_cpython_builtin(handle):
        raise RuntimeError(f"不支持的 builtin function 对象: {type(handle).__name__}")

    return read_cpython_function_runtime_info(handle)


def _read_builtin_function_runtime_info(
    handle: types.BuiltinFunctionType,
) -> BuiltinFunctionRuntimeInfo:
    """读取 builtin function 的入口地址与调用约定。"""
    try:
        method_address = int(_pycfunction_get_function(handle))
        flags = int(_pycfunction_get_flags(handle))
    except (SystemError, TypeError) as ex:
        raise RuntimeError("读取 builtin function 运行时信息失败。") from ex

    return _build_runtime_info(
        method_address=method_address,
        flags=flags,
        error_prefix="读取 builtin function 运行时信息失败",
    )


def _read_method_descriptor_runtime_info(
    handle: object,
) -> BuiltinFunctionRuntimeInfo:
    """读取方法描述器保存的入口地址与调用约定。"""
    try:
        descriptor = _PyMethodDescrObject.from_address(id(handle))
        method_definition = descriptor.d_method.contents
    except (TypeError, ValueError) as ex:
        raise RuntimeError("读取 method descriptor 运行时信息失败。") from ex
    except Exception as ex:
        raise RuntimeError("读取 method descriptor 运行时信息失败。") from ex

    return _build_runtime_info(
        method_address=int(method_definition.ml_meth),
        flags=int(method_definition.ml_flags),
        error_prefix="读取 method descriptor 运行时信息失败",
    )


def _build_runtime_info(
    *,
    method_address: int,
    flags: int,
    error_prefix: str,
) -> BuiltinFunctionRuntimeInfo:
    """构造运行时信息并校验函数地址。"""
    if method_address == 0:
        raise RuntimeError(f"{error_prefix}: C函数地址为空。")

    return BuiltinFunctionRuntimeInfo(
        address=method_address,
        flags=flags,
    )
