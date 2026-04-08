from __future__ import annotations

from typing import Any, cast

import pytest

from pcstubgen.signature_completion.c_extension.runtime import resolve_runtime_pymethoddef


def test_runtime_supports_builtin_function_or_method() -> None:
    result = resolve_runtime_pymethoddef(object.__dict__["__new__"])

    assert result.name == "__new__"
    assert result.method_address != 0


def test_runtime_supports_method_descriptor() -> None:
    result = resolve_runtime_pymethoddef(list.__dict__["append"])

    assert result.name == "append"
    assert result.method_address != 0


def test_runtime_supports_wrapper_descriptor() -> None:
    result = resolve_runtime_pymethoddef(object.__dict__["__init__"])

    assert result.name == "__init__"
    assert result.method_address != 0
    assert result.flags == 0
    assert result.doc is not None
    assert result.doc.startswith("__init__(")


def test_runtime_unwraps_staticmethod_before_classification() -> None:
    class StaticBox:
        method = staticmethod(len)

    result = resolve_runtime_pymethoddef(StaticBox.__dict__["method"])

    assert result.name == "len"
    assert result.method_address != 0


def test_runtime_unwraps_classmethod_before_classification() -> None:
    class ClassBox:
        method = classmethod(cast(Any, dict.fromkeys))

    result = resolve_runtime_pymethoddef(ClassBox.__dict__["method"])

    assert result.name == "fromkeys"
    assert result.method_address != 0


def test_runtime_rejects_pybind11_instancemethod_target() -> None:
    scipy_fmm_core = pytest.importorskip("scipy.io._fast_matrix_market._fmm_core")

    with pytest.raises(RuntimeError, match="pybind11_builtins"):
        resolve_runtime_pymethoddef(scipy_fmm_core.header.__dict__["__init__"])


def test_runtime_rejects_cython_runtime_objects() -> None:
    scipy_messagestream = pytest.importorskip("scipy._lib.messagestream")

    with pytest.raises(RuntimeError, match="cython_function_or_method"):
        resolve_runtime_pymethoddef(scipy_messagestream.MessageStream.__dict__["get"])
