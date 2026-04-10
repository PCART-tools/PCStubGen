from __future__ import annotations

from ctypes import POINTER, byref, c_char_p, c_int, c_uint
from typing import Any

import clang.cindex
from clang.cindex import Index, TranslationUnit, TranslationUnitLoadError


def _get_parse_translation_unit2_full_argv() -> Any:
    """返回已注册 ctypes 签名的 clang_parseTranslationUnit2FullArgv。"""
    parse_translation_unit2_full_argv = (
        clang.cindex.conf.lib.clang_parseTranslationUnit2FullArgv
    )
    parse_translation_unit2_full_argv.argtypes = [
        Index,
        clang.cindex.c_interop_string,
        POINTER(c_char_p),
        c_int,
        POINTER(clang.cindex._CXUnsavedFile),
        c_uint,
        c_uint,
        POINTER(clang.cindex.c_object_p),
    ]
    parse_translation_unit2_full_argv.restype = c_uint
    return parse_translation_unit2_full_argv


def parse_translation_unit_full_argv(
    index: Index,
    arguments: list[str],
    *,
    options: int = 0,
    source_filename: str | None = None,
) -> TranslationUnit:
    """使用完整 argv 调用 libclang 解析 translation unit。"""
    if not arguments:
        raise TranslationUnitLoadError("Error parsing translation unit: empty argv.")

    parse_translation_unit2_full_argv = _get_parse_translation_unit2_full_argv()
    arguments_array = (c_char_p * len(arguments))(
        *[clang.cindex.b(argument) for argument in arguments]
    )
    translation_unit_pointer = clang.cindex.c_object_p()
    error_code = parse_translation_unit2_full_argv(
        index,
        source_filename,
        arguments_array,
        len(arguments),
        None,
        0,
        options,
        byref(translation_unit_pointer),
    )
    if error_code != 0 or not translation_unit_pointer:
        raise TranslationUnitLoadError(
            f"Error parsing translation unit. libclang error code: {error_code}"
        )
    return TranslationUnit(translation_unit_pointer, index=index)
