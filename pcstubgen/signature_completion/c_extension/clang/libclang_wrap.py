from __future__ import annotations

from ctypes import POINTER, byref, c_char_p, c_int, c_size_t, c_uint, c_void_p, string_at

import clang.cindex
from clang.cindex import File, Index, SourceLocation, TranslationUnit, TranslationUnitLoadError


_parse_translation_unit2_full_argv = clang.cindex.conf.lib.clang_parseTranslationUnit2FullArgv
_parse_translation_unit2_full_argv.argtypes = [
    Index,
    clang.cindex.c_interop_string,
    POINTER(c_char_p),
    c_int,
    POINTER(clang.cindex._CXUnsavedFile),
    c_uint,
    c_uint,
    POINTER(clang.cindex.c_object_p),
]
_parse_translation_unit2_full_argv.restype = c_uint

_get_file_location = clang.cindex.conf.lib.clang_getFileLocation
_get_file_location.argtypes = [
    SourceLocation,
    POINTER(clang.cindex.c_object_p),
    POINTER(c_uint),
    POINTER(c_uint),
    POINTER(c_uint),
]
_get_file_location.restype = None

_get_file_contents = clang.cindex.conf.lib.clang_getFileContents
_get_file_contents.argtypes = [
    TranslationUnit,
    File,
    POINTER(c_size_t),
]
_get_file_contents.restype = c_void_p


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

    arguments_array = (c_char_p * len(arguments))(
        *[clang.cindex.b(argument) for argument in arguments]
    )
    translation_unit_pointer = clang.cindex.c_object_p()
    error_code = _parse_translation_unit2_full_argv(
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


def get_file_location(
    location: SourceLocation,
) -> tuple[File | None, int, int, int]:
    """返回源码位置对应的文件、行、列和字节偏移。"""
    file_pointer = clang.cindex.c_object_p()
    line = c_uint()
    column = c_uint()
    offset = c_uint()
    _get_file_location(
        location,
        byref(file_pointer),
        byref(line),
        byref(column),
        byref(offset),
    )

    file = None if not file_pointer else File(file_pointer)
    return file, int(line.value), int(column.value), int(offset.value)


def get_file_contents(
    translation_unit: TranslationUnit,
    file: File,
) -> bytes:
    """返回 translation unit 已加载的文件完整字节内容。"""
    size = c_size_t()
    buffer_pointer = _get_file_contents(
        translation_unit,
        file,
        byref(size),
    )
    if not buffer_pointer:
        raise RuntimeError(f"未从 translation unit 获取到文件内容: {file.name}")
    return string_at(buffer_pointer, size.value)
