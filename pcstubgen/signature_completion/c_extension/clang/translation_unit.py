from __future__ import annotations

import contextlib
import functools
import subprocess

import clang
from clang.cindex import Diagnostic, Index, TranslationUnit

from .compilation_database import CompilationCommand


@functools.cache
def detect_clang_resource_dir() -> str | None:
    """调用系统 clang 自动探测 resource dir。"""
    completed_process = subprocess.run(
        ["clang", "-print-resource-dir"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed_process.returncode != 0:
        return None

    resource_dir = completed_process.stdout.strip()
    if not resource_dir:
        return None
    return resource_dir


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


def build_effective_parse_args(compilation_command: CompilationCommand) -> list[str]:
    """构造实际传给 libclang parse 的参数列表。"""
    effective_parse_args = list(compilation_command.parse_args)
    resource_dir = detect_clang_resource_dir()
    if resource_dir is not None:
        effective_parse_args.extend(["-resource-dir", resource_dir])
    return effective_parse_args


def parse(
    index: Index,
    compilation_command: CompilationCommand,
    *,
    effective_parse_args: list[str] | None = None,
) -> TranslationUnit:
    """解析单个编译数据库条目为 clang translation unit。"""
    if effective_parse_args is None:
        effective_parse_args = build_effective_parse_args(compilation_command)

    with contextlib.chdir(compilation_command.working_directory):
        translation_unit = index.parse(
            str(compilation_command.file_path),
            args=effective_parse_args,
        )
    return translation_unit
