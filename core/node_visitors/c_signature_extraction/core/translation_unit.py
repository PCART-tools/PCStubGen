from __future__ import annotations

import posixpath
import re
import sysconfig
from pathlib import Path

import clang
from clang.cindex import Diagnostic, Index, TranslationUnit
from loguru import logger

from .constants import CPP_SOURCE_SUFFIXES, NATIVE_SOURCE_SUFFIXES

_FILE_NOT_FOUND_RE = re.compile(r"'([^']+)' file not found")


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
    filename = location.file.name
    line = location.line
    column = location.column
    message = diagnostic.spelling
    return f"[{severity}] {filename}:{line}:{column}: {message}"


def format_diagnostics_message(
    *,
    file_path: Path,
    parse_args: list[str],
    diagnostics: list[Diagnostic],
) -> str:
    """格式化 translation unit 的诊断日志块，便于统一输出和排查。"""
    lines = [
        "翻译单元诊断信息",
        f"文件路径: {file_path}",
        f"解析参数: {parse_args}",
        "诊断:",
    ]
    lines.extend(f"- {diagnostic_to_str(diagnostic)}" for diagnostic in diagnostics)
    return "\n".join(lines)


def normalize_include_literal(include_literal: str) -> str:
    """规范化报错里的头文件字面量，便于后续路径匹配。"""
    normalized = include_literal.replace("\\", "/").strip()
    if not normalized:
        return ""
    normalized = posixpath.normpath(normalized)
    if normalized == ".":
        return ""
    return normalized


def extract_missing_include_literals(diagnostics: list[Diagnostic]) -> list[str]:
    """从 clang 错误诊断中提取缺失头文件名。"""
    missing: set[str] = set()
    for diagnostic in diagnostics:
        if diagnostic.severity != clang.cindex.Diagnostic.Fatal:
            continue
        message = str(diagnostic.spelling)
        match = _FILE_NOT_FOUND_RE.search(message)
        if match is None:
            continue
        include_literal = normalize_include_literal(match.group(1))
        if not include_literal:
            continue
        missing.add(include_literal)
    return sorted(missing)


def inject_python_include_directories(include_directories: list[str]) -> list[str]:
    """向 include 目录列表注入当前 Python 头文件目录。"""
    directories = list(include_directories)
    include_candidates = [
        sysconfig.get_path("include"),
        sysconfig.get_path("platinclude"),
    ]
    for include_dir in include_candidates:
        if not include_dir or include_dir in directories:
            continue
        directories.append(include_dir)
    return directories


def resolve_missing_include_dir(source_root: Path, *, include_literal: str) -> Path | None:
    """在源码树内搜索缺失头文件，找到首个匹配的 include 目录后立即返回。"""
    include_root_depth = len(tuple(part for part in include_literal.split("/") if part)) - 1

    for header_path in source_root.rglob(include_literal):
        if not header_path.is_file():
            continue
        if include_root_depth >= len(header_path.parents):
            continue
        return header_path.parents[include_root_depth]
    return None


def append_include_args(clang_include_directory: list[str], include_args: list[str]) -> list[str]:
    """将新发现的 include 目录追加到 clang 参数中，并返回实际新增项。"""
    added: list[str] = []
    for include_dir in include_args:
        if include_dir in clang_include_directory or include_dir in added:
            continue
        clang_include_directory.append(include_dir)
        added.append(include_dir)
    return added


def discover_missing_include_args(
    *,
    file_path: Path,
    diagnostics: list[Diagnostic],
    source_root: Path,
    clang_include_directory: list[str],
) -> list[str]:
    """基于缺失头文件诊断自动补全 clang include 目录。"""
    resolved_pairs: list[tuple[str, str]] = []
    missing_literals = extract_missing_include_literals(diagnostics)
    for include_literal in missing_literals:
        include_dir = resolve_missing_include_dir(source_root, include_literal=include_literal)
        if include_dir is None:
            continue
        resolved_pairs.append((include_literal, str(include_dir)))

    added = append_include_args(
        clang_include_directory,
        [include_dir for _, include_dir in resolved_pairs],
    )
    if not added:
        return added

    for include_literal, include_dir in resolved_pairs:
        if include_dir not in added:
            continue
        logger.info(
            "补全clang include path: {}, parse 文件: {}, include 字面量: {}",
            include_dir,
            file_path,
            include_literal,
        )
    return added


def find_candidate_files(source_root: Path) -> list[Path]:
    """查找包含 `PyModuleDef` 定义线索的 C/C++ 源文件。"""
    result: list[Path] = []
    for path in source_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in NATIVE_SOURCE_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "PyModuleDef" in text:
                result.append(path)
    result.sort()
    return result


def get_std_value_for_file(
    file_path: Path,
    *,
    clang_c_std: str,
    clang_cpp_std: str,
) -> str:
    """按后缀为源码文件选择 C 或 C++ 标准值。"""
    if file_path.suffix.lower() in CPP_SOURCE_SUFFIXES:
        return clang_cpp_std
    return clang_c_std


def build_clang_parse_args(
    file_path: Path,
    *,
    clang_include: list[str],
    clang_include_directory: list[str],
    clang_c_std: str,
    clang_cpp_std: str,
) -> list[str]:
    """为指定源码文件构建 clang 解析参数列表。"""
    parse_args: list[str] = []
    std_value = get_std_value_for_file(
        file_path,
        clang_c_std=clang_c_std,
        clang_cpp_std=clang_cpp_std,
    )
    parse_args.extend(["--std", std_value])
    for include_value in clang_include:
        parse_args.extend(["--include", include_value])
    for include_dir in clang_include_directory:
        parse_args.extend(["--include-directory", include_dir])
    return parse_args


def has_error_diagnostics(diagnostics: list[Diagnostic]) -> bool:
    """判断 diagnostics 中是否包含 Error/Fatal 级别。"""
    for diagnostic in diagnostics:
        if diagnostic.severity >= clang.cindex.Diagnostic.Error:
            return True
    return False


def parse_translation_unit(
    index: Index,
    file_path: Path,
    *,
    source_root: Path,
    clang_include: list[str],
    clang_include_directory: list[str],
    clang_c_std: str,
    clang_cpp_std: str,
) -> TranslationUnit:
    """解析单个源码文件为 clang translation unit。"""
    translation_unit: TranslationUnit | None = None
    diagnostics: list[Diagnostic] = []
    parse_args: list[str] = []
    for _ in range(10):
        parse_args = build_clang_parse_args(
            file_path,
            clang_include=clang_include,
            clang_include_directory=clang_include_directory,
            clang_c_std=clang_c_std,
            clang_cpp_std=clang_cpp_std,
        )
        translation_unit = index.parse(str(file_path), args=parse_args)
        diagnostics = translation_unit.diagnostics
        added = discover_missing_include_args(
            file_path=file_path,
            diagnostics=diagnostics,
            source_root=source_root,
            clang_include_directory=clang_include_directory,
        )
        if not added:
            break

    if has_error_diagnostics(diagnostics):
        logger.warning(
            format_diagnostics_message(
                file_path=file_path,
                parse_args=parse_args,
                diagnostics=diagnostics,
            )
        )
    return translation_unit
