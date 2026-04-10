from __future__ import annotations

import contextlib

import clang
from clang.cindex import Diagnostic, Index, TranslationUnit

from .compilation_database import MyCompileCommand
from .libclang_parse import parse_translation_unit_full_argv


def diagnostic_severity_to_str(severity: int) -> str:
    """把 libclang severity 严重程度数值转换成可读名称。"""
    match severity:
        case clang.cindex.Diagnostic.Ignored:
            return "IGNORED"
        case clang.cindex.Diagnostic.Note:
            return "NOTE"
        case clang.cindex.Diagnostic.Warning:
            return "WARNING"
        case clang.cindex.Diagnostic.Error:
            return "ERROR"
        case clang.cindex.Diagnostic.Fatal:
            return "FATAL"
        case _:
            return f"SEVERITY_{severity}"


def diagnostic_to_str(diagnostic: Diagnostic) -> str:
    """将单条 clang diagnostic 格式化为稳定的一行文本。"""
    severity = diagnostic_severity_to_str(diagnostic.severity)
    location = diagnostic.location
    message = diagnostic.spelling
    return f"[{severity}] {location}: {message}"


def has_error_diagnostics(diagnostics: list[Diagnostic]) -> bool:
    """判断 diagnostics 中是否包含 Error/Fatal 级别。"""
    for diagnostic in diagnostics:
        if diagnostic.severity >= clang.cindex.Diagnostic.Error:
            return True
    return False


def parse(
    index: Index,
    compile_command: MyCompileCommand,
) -> TranslationUnit:
    """解析单个编译数据库条目为 clang translation unit。"""
    with contextlib.chdir(compile_command.directory):
        translation_unit = parse_translation_unit_full_argv(
            index,
            compile_command.arguments,
        )
    return translation_unit
