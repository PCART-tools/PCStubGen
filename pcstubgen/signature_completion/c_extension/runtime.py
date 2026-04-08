from __future__ import annotations

import ctypes
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimePyMethodDef:
    method_address: int
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


class _PyMethodDescrObject(ctypes.Structure):
    _fields_ = [
        ("ob_refcnt", ctypes.c_ssize_t),
        ("ob_type", ctypes.c_void_p),
        ("d_type", ctypes.c_void_p),
        ("d_name", ctypes.py_object),
        ("d_qualname", ctypes.py_object),
        ("d_method", ctypes.POINTER(_PyMethodDef)),
        ("vectorcall", ctypes.c_void_p),
    ]


class _WrapperBase(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("offset", ctypes.c_int),
        ("function", ctypes.c_void_p),
        ("wrapper", ctypes.c_void_p),
        ("doc", ctypes.c_char_p),
        ("flags", ctypes.c_int),
        ("name_strobj", ctypes.py_object),
    ]


class _PyWrapperDescrObject(ctypes.Structure):
    _fields_ = [
        ("ob_refcnt", ctypes.c_ssize_t),
        ("ob_type", ctypes.c_void_p),
        ("d_type", ctypes.c_void_p),
        ("d_name", ctypes.py_object),
        ("d_qualname", ctypes.py_object),
        ("d_base", ctypes.POINTER(_WrapperBase)),
        ("d_wrapped", ctypes.c_void_p),
    ]


def resolve_runtime_pymethoddef(handle: object) -> RuntimePyMethodDef:
    """从受支持的 CPython 运行时对象读取函数入口元信息。"""
    handle = _unwrap_runtime_handle(handle)
    runtime_type = _runtime_type_key(handle)

    if runtime_type == ("builtins", "builtin_function_or_method"):
        _reject_if_pybind11_builtin(handle)
        return _build_runtime_methoddef(
            method_def=_read_cfunction_methoddef(handle),
        )

    if runtime_type == ("builtins", "method_descriptor"):
        return _build_runtime_methoddef(
            method_def=_read_method_descriptor_methoddef(handle),
        )

    if runtime_type == ("builtins", "wrapper_descriptor"):
        return _build_runtime_wrapperdef(
            wrapper_object=_read_wrapper_descriptor(handle),
        )

    raise RuntimeError(f"不支持的C函数对象类型: {type(handle).__name__}")


def _build_runtime_methoddef(
    *,
    method_def: _PyMethodDef,
) -> RuntimePyMethodDef:
    method_address = int(method_def.ml_meth)
    if method_address == 0:
        raise RuntimeError("PyMethodDef.ml_meth 为空。")

    return RuntimePyMethodDef(
        method_address=method_address,
        flags=int(method_def.ml_flags),
    )


def _build_runtime_wrapperdef(
    *,
    wrapper_object: _PyWrapperDescrObject,
) -> RuntimePyMethodDef:
    base_ptr = wrapper_object.d_base
    if not bool(base_ptr):
        raise RuntimeError("PyWrapperDescrObject.d_base 为空。")

    method_address = int(wrapper_object.d_wrapped)
    if method_address == 0:
        raise RuntimeError("PyWrapperDescrObject.d_wrapped 为空。")

    return RuntimePyMethodDef(
        method_address=method_address,
        # wrapper_descriptor 使用 CPython slot wrapper flags，不是 PyMethodDef.ml_flags。
        flags=0,
    )


def _unwrap_runtime_handle(handle: object) -> object:
    while True:
        runtime_type = _runtime_type_key(handle)
        if runtime_type not in {
            ("builtins", "staticmethod"),
            ("builtins", "classmethod"),
            ("builtins", "instancemethod"),
        }:
            return handle

        wrapped = getattr(handle, "__func__", None)
        if wrapped is None:
            raise RuntimeError(f"{type(handle).__name__} 缺少 __func__。")
        handle = wrapped


def _runtime_type_key(handle: object) -> tuple[str, str]:
    handle_type = type(handle)
    return handle_type.__module__, handle_type.__name__


def _read_cfunction_methoddef(handle: object) -> _PyMethodDef:
    try:
        method_ptr = _PyCFunctionObject.from_address(id(handle)).m_ml
    except (TypeError, ValueError) as ex:
        raise RuntimeError("读取 PyCFunctionObject 失败。") from ex
    if not bool(method_ptr):
        raise RuntimeError("PyCFunctionObject.m_ml 为空。")
    return method_ptr.contents


def _read_method_descriptor_methoddef(handle: object) -> _PyMethodDef:
    try:
        method_ptr = _PyMethodDescrObject.from_address(id(handle)).d_method
    except (TypeError, ValueError) as ex:
        raise RuntimeError("读取 PyMethodDescrObject 失败。") from ex
    if not bool(method_ptr):
        raise RuntimeError("PyMethodDescrObject.d_method 为空。")
    return method_ptr.contents


def _read_wrapper_descriptor(handle: object) -> _PyWrapperDescrObject:
    try:
        return _PyWrapperDescrObject.from_address(id(handle))
    except (TypeError, ValueError) as ex:
        raise RuntimeError("读取 PyWrapperDescrObject 失败。") from ex


def _reject_if_pybind11_builtin(handle: object) -> None:
    self_obj = getattr(handle, "__self__", None)
    if self_obj is None:
        return

    self_type = type(self_obj)
    if self_type.__module__.startswith("pybind11_builtins"):
        raise RuntimeError(
            f"不支持的C函数运行时目标: {self_type.__module__}.{self_type.__name__}"
        )
