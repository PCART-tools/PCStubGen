from __future__ import annotations

import ctypes
from dataclasses import dataclass


@dataclass(frozen=True)
class BuiltinFunctionRuntimeInfo:
    address: int
    flags: int


class _PyMethodDef(ctypes.Structure):
    _fields_ = [
        ("ml_name", ctypes.c_char_p),
        ("ml_meth", ctypes.c_void_p),
        ("ml_flags", ctypes.c_int),
        ("ml_doc", ctypes.c_char_p),
    ]


class _PyCFunctionObject(ctypes.Structure):
    _fields_ = [
        ("ob_refcnt", ctypes.c_ssize_t),
        ("ob_type", ctypes.c_void_p),
        ("m_ml", ctypes.POINTER(_PyMethodDef)),
        ("m_self", ctypes.py_object),
        ("m_module", ctypes.py_object),
        ("m_weakreflist", ctypes.c_void_p),
        ("vectorcall", ctypes.c_void_p),
    ]


def supports_builtin_function_inference(handle: object) -> bool:
    """判断运行时对象是否支持 builtin function C 源码签名推导。"""
    return _runtime_type_key(handle) == (
        "builtins",
        "builtin_function_or_method",
    ) and not _is_pybind11_bound_builtin(handle)


def read_builtin_function_runtime_info(handle: object) -> BuiltinFunctionRuntimeInfo:
    """读取 builtin function 的入口地址与调用约定。"""
    if not supports_builtin_function_inference(handle):
        raise RuntimeError(f"不支持的 builtin function 对象: {type(handle).__name__}")

    method_def = _read_builtin_function_methoddef(handle)
    method_address = int(method_def.ml_meth)
    if method_address == 0:
        raise RuntimeError("PyMethodDef.ml_meth 为空。")

    return BuiltinFunctionRuntimeInfo(
        address=method_address,
        flags=int(method_def.ml_flags),
    )


def _runtime_type_key(handle: object) -> tuple[str, str]:
    handle_type = type(handle)
    return handle_type.__module__, handle_type.__name__


def _read_builtin_function_methoddef(handle: object) -> _PyMethodDef:
    try:
        method_ptr = _PyCFunctionObject.from_address(id(handle)).m_ml
    except (TypeError, ValueError) as ex:
        raise RuntimeError("读取 PyCFunctionObject 失败。") from ex
    if not bool(method_ptr):
        raise RuntimeError("PyCFunctionObject.m_ml 为空。")
    return method_ptr.contents


def _is_pybind11_bound_builtin(handle: object) -> bool:
    self_obj = getattr(handle, "__self__", None)
    if self_obj is None:
        return False

    self_type = type(self_obj)
    return self_type.__module__.startswith("pybind11_builtins")
