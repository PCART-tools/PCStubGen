"""
libclang api的低层wrap
"""
from __future__ import annotations

from ctypes import (
    POINTER,
    byref,
    c_char_p,
    c_double,
    c_int,
    c_longlong,
    c_size_t,
    c_uint,
    c_ulonglong,
    c_void_p,
    string_at,
)

import clang.cindex
from clang.cindex import Cursor, File, Index, SourceLocation, TranslationUnit, TranslationUnitLoadError


_CX_EVAL_UNEXPOSED = 0
_CX_EVAL_INT = 1
_CX_EVAL_FLOAT = 2
_CX_EVAL_OBJC_STR_LITERAL = 3
_CX_EVAL_STR_LITERAL = 4
_CX_EVAL_CFSTR = 5
_CX_EVAL_OTHER = 6

CX_BINARY_OPERATOR_ASSIGN = 22

_STRING_EVAL_RESULT_KINDS = {
    _CX_EVAL_OBJC_STR_LITERAL,
    _CX_EVAL_STR_LITERAL,
    _CX_EVAL_CFSTR,
}

_EVAL_RESULT_KIND_NAMES = {
    _CX_EVAL_UNEXPOSED: "UNEXPOSED",
    _CX_EVAL_INT: "INT",
    _CX_EVAL_FLOAT: "FLOAT",
    _CX_EVAL_OBJC_STR_LITERAL: "OBJC_STR_LITERAL",
    _CX_EVAL_STR_LITERAL: "STR_LITERAL",
    _CX_EVAL_CFSTR: "CFSTR",
    _CX_EVAL_OTHER: "OTHER",
}


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

_cursor_evaluate = clang.cindex.conf.lib.clang_Cursor_Evaluate
_cursor_evaluate.argtypes = [Cursor]
_cursor_evaluate.restype = c_void_p

_eval_result_get_kind = clang.cindex.conf.lib.clang_EvalResult_getKind
_eval_result_get_kind.argtypes = [c_void_p]
_eval_result_get_kind.restype = c_int

_eval_result_get_as_long_long = clang.cindex.conf.lib.clang_EvalResult_getAsLongLong
_eval_result_get_as_long_long.argtypes = [c_void_p]
_eval_result_get_as_long_long.restype = c_longlong

_eval_result_get_as_unsigned = clang.cindex.conf.lib.clang_EvalResult_getAsUnsigned
_eval_result_get_as_unsigned.argtypes = [c_void_p]
_eval_result_get_as_unsigned.restype = c_ulonglong

_eval_result_is_unsigned_int = clang.cindex.conf.lib.clang_EvalResult_isUnsignedInt
_eval_result_is_unsigned_int.argtypes = [c_void_p]
_eval_result_is_unsigned_int.restype = c_uint

_eval_result_get_as_double = clang.cindex.conf.lib.clang_EvalResult_getAsDouble
_eval_result_get_as_double.argtypes = [c_void_p]
_eval_result_get_as_double.restype = c_double

_eval_result_get_as_str = clang.cindex.conf.lib.clang_EvalResult_getAsStr
_eval_result_get_as_str.argtypes = [c_void_p]
_eval_result_get_as_str.restype = c_char_p

_eval_result_dispose = clang.cindex.conf.lib.clang_EvalResult_dispose
_eval_result_dispose.argtypes = [c_void_p]
_eval_result_dispose.restype = None

_get_cursor_binary_operator_kind = clang.cindex.conf.lib.clang_getCursorBinaryOperatorKind
_get_cursor_binary_operator_kind.argtypes = [Cursor]
_get_cursor_binary_operator_kind.restype = c_uint


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


def evaluate_cursor(cursor: Cursor) -> int | float | str:
    """求值 cursor 对应的 Python 标量结果。"""
    eval_result_pointer = _cursor_evaluate(cursor)
    if not eval_result_pointer:
        raise RuntimeError(f"cursor 无法求值, cursor: {cursor.location}")

    try:
        kind = int(_eval_result_get_kind(eval_result_pointer))
        if kind == _CX_EVAL_INT:
            if _eval_result_is_unsigned_int(eval_result_pointer):
                return int(_eval_result_get_as_unsigned(eval_result_pointer))
            return int(_eval_result_get_as_long_long(eval_result_pointer))

        if kind == _CX_EVAL_FLOAT:
            return float(_eval_result_get_as_double(eval_result_pointer))
        if kind in _STRING_EVAL_RESULT_KINDS:
            string_value = _eval_result_get_as_str(eval_result_pointer)
            if string_value is None:
                raise RuntimeError(
                    "libclang 返回了空字符串求值结果, "
                    f"kind: {_eval_result_kind_name(kind)}, cursor: {cursor.location}"
                )
            return string_value.decode("utf-8")

        if kind in {_CX_EVAL_UNEXPOSED, _CX_EVAL_OTHER}:
            raise RuntimeError(
                "cursor 求值结果不受支持, "
                f"kind: {_eval_result_kind_name(kind)}, cursor: {cursor.location}"
            )

        raise RuntimeError(
            f"未知的 libclang 求值结果类型: {kind}, cursor: {cursor.location}"
        )
    finally:
        _eval_result_dispose(eval_result_pointer)


def get_cursor_binary_operator_kind(cursor: Cursor) -> int:
    """返回二元操作符 cursor 对应的 `CXBinaryOperatorKind`。"""
    return int(_get_cursor_binary_operator_kind(cursor))


def _eval_result_kind_name(kind: int) -> str:
    """返回求值结果 kind 的稳定名称。"""
    return _EVAL_RESULT_KIND_NAMES.get(kind, str(kind))
