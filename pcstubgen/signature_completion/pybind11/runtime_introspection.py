from __future__ import annotations

from loguru import logger

from . import _pybind11_runtime


def extract_pybind11_signatures(obj: object) -> list[str]:
    """提取 pybind11 overload chain 上的单条签名文本。"""
    signatures = _pybind11_runtime.extract_signatures(obj)
    logger.debug(
        "pybind11 runtime extractor 成功提取签名, type: {}, overloads: {}",
        type(obj).__name__,
        len(signatures),
    )
    return signatures
