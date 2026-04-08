from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RuntimePyMethodDef:
    name: str
    method_address: int
    flags: int
    doc: str | None
    handle: Any = field(repr=False, compare=False)


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


def resolve_runtime_pymethoddef(handle: object) -> RuntimePyMethodDef:
    """从运行时对象读取 `PyMethodDef` 元信息。"""
    import inspect

    if inspect.isbuiltin(handle):
        return _build_runtime_methoddef(
            handle=handle,
            method_def=_PyCFunctionObject.from_address(id(handle)).m_ml.contents,
        )

    if inspect.ismethoddescriptor(handle):
        return _build_runtime_methoddef(
            handle=handle,
            method_def=_PyMethodDescrObject.from_address(id(handle)).d_method.contents,
        )

    raise RuntimeError(f"不支持的C函数对象类型: {type(handle).__name__}")


def _build_runtime_methoddef(
    *,
    handle: object,
    method_def: _PyMethodDef,
) -> RuntimePyMethodDef:
    method_address = int(method_def.ml_meth)
    if method_address == 0:
        raise RuntimeError("PyMethodDef.ml_meth 为空。")

    return RuntimePyMethodDef(
        name=_decode_c_string(method_def.ml_name) or getattr(handle, "__name__", "<unnamed>"),
        method_address=method_address,
        flags=int(method_def.ml_flags),
        doc=_decode_c_string(method_def.ml_doc),
        handle=handle,
    )


def _decode_c_string(value: bytes | None) -> str | None:
    if value is None:
        return None
    decoded = value.decode("utf-8", errors="replace")
    if not decoded:
        return None
    return decoded
