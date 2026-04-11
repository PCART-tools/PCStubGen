from __future__ import annotations

import contextlib
import functools
from pathlib import Path
import subprocess

import clang
from clang.cindex import CompileCommand, Diagnostic, Index, TranslationUnit

from .libclang_parse_wrap import parse_translation_unit_full_argv


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
    compile_command: CompileCommand,
) -> TranslationUnit:
    """解析单个编译数据库条目为 clang translation unit。"""
    arguments = list(compile_command.arguments)
    resource_dir = try_get_clang_resource_dir()
    if resource_dir is not None:
        arguments.extend(["-resource-dir", str(resource_dir)])

    with contextlib.chdir(Path(str(compile_command.directory)).resolve()):
        translation_unit = parse_translation_unit_full_argv(
            index,
            arguments,
        )
    return translation_unit

@functools.cache
def try_get_clang_resource_dir() -> Path | None:
    """尝试解析 clang resource dir，失败时返回 None。"""
    try:
        resource_dir_text = subprocess.check_output(
            ["clang", "-print-resource-dir"],
            text=True,
        ).strip()
    except BaseException:
        return None

    if not resource_dir_text:
        return None

    return Path(resource_dir_text)
