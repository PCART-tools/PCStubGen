from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .completer import Pybind11Completer
    from .inferencer import parse_args_str, parse_pybind11_signature

__all__ = [
    "Pybind11Completer",
    "parse_args_str",
    "parse_pybind11_signature",
]


def __getattr__(name: str) -> object:
    """按需导出 pybind11 子模块符号，避免循环导入。"""
    if name == "Pybind11Completer":
        from .completer import Pybind11Completer

        return Pybind11Completer

    if name == "parse_args_str":
        from .inferencer import parse_args_str

        return parse_args_str

    if name == "parse_pybind11_signature":
        from .inferencer import parse_pybind11_signature

        return parse_pybind11_signature

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
