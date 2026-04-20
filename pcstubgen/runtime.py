from __future__ import annotations

import ctypes
import types
from dataclasses import dataclass


@dataclass(frozen=True)
class CExtensionFunctionRuntimeInfo:
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


def is_c_extension_module_function(handle: object) -> bool:
    return isinstance(handle, types.BuiltinFunctionType) and not is_pybind11_bound(handle)

def is_c_extension_instance_method(handle: object) -> bool:
    return isinstance(handle, types.MethodDescriptorType)

def is_c_extension_static_method(handle: object) -> bool:
    return isinstance(handle, staticmethod) and is_c_extension_module_function(handle.__func__)

def is_c_extension_class_method(handle: object) -> bool:
    return isinstance(handle, types.ClassMethodDescriptorType)

def is_pybind11_module_function(handle: object) -> bool:
    return isinstance(handle, types.BuiltinFunctionType) and is_pybind11_bound(handle)

def is_pybind11_instance_method(handle: object) -> bool:
    return type(handle).__module__ == "builtins" and type(handle).__name__ == "instancemethod" and is_pybind11_bound(handle)

def is_pybind11_static_method(handle: object) -> bool:
    return isinstance(handle, staticmethod) and is_pybind11_module_function(handle.__func__)


def is_pybind11_bound(handle: object) -> bool:
    """判断 builtin function 是否绑定到 pybind11 对象。"""
    self_obj = getattr(handle, "__self__", None)
    if self_obj is not None and type(self_obj).__module__ == "pybind11_builtins":
        return True
    return False


def read_c_extension_function_runtime_info(handle: object) -> CExtensionFunctionRuntimeInfo:
    """读取 CPython C 扩展函数句柄的入口地址与调用约定。"""
    if is_c_extension_module_function(handle):
        return _read_pycfunction_runtime_info(handle)

    if is_c_extension_instance_method(handle) or is_c_extension_class_method(handle):
        return _read_method_descriptor_runtime_info(handle)

    raise RuntimeError(f"不支持的 CPython 函数对象: {type(handle).__name__}")


def _read_pycfunction_runtime_info(handle: object) -> CExtensionFunctionRuntimeInfo:
    """读取 builtin function 的入口地址与调用约定。"""
    address = int(_pycfunction_get_function(handle))
    flags = int(_pycfunction_get_flags(handle))
    if address == 0:
        raise RuntimeError("读取 builtin function 运行时信息失败: C函数地址为空。")

    return CExtensionFunctionRuntimeInfo(address, flags)


def _read_method_descriptor_runtime_info(handle: object) -> CExtensionFunctionRuntimeInfo:
    """读取方法描述器保存的入口地址与调用约定。"""
    descriptor = _PyMethodDescrObject.from_address(id(handle))
    method_definition = descriptor.d_method.contents
    address = int(method_definition.ml_meth)
    flags = int(method_definition.ml_flags)
    if address == 0:
        raise RuntimeError("读取 method descriptor 运行时信息失败: C函数地址为空。")

    return CExtensionFunctionRuntimeInfo(address, flags)
