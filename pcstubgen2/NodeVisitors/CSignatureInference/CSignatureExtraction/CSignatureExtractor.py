from __future__ import annotations

import logging
import re
import sysconfig
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

import clang
import clang.cindex
from clang.cindex import (
    Cursor,
    CursorKind,
    Diagnostic,
    Index,
    TokenKind,
    TranslationUnit,
    TypeKind
)

from . import ClangEval
from .Constants import (
    CPP_SOURCE_SUFFIXES,
    FORMAT_TYPE_MAP,
    METH_TYPE_LITERAL_MAP,
    NATIVE_SOURCE_SUFFIXES,
    PY_BUILDVALUE_SINGLE_MARKER_TYPE_MAP,
    RETURN_CALL_PREFIX_TYPE_MAP,
    RETURN_MACRO_TYPE_MAP,
    RETURN_TOKEN_TYPE_MAP,
    UNRELATED_TOKENS,
)
from .Models import (
    ExtractedArgument,
    ExtractedClass,
    ExtractedFunction,
    ExtractedModule,
    ExtractedSignature,
)

logger = logging.getLogger(__name__)


AUTO_INCLUDE_MISSING_HEADER_RE = re.compile(r"'([^']+)' file not found")


SignatureArgumentKey: TypeAlias = tuple[str, str | None, str | None, str]
SignatureKey: TypeAlias = tuple[str | None, tuple[SignatureArgumentKey, ...]]
FunctionDedupKey: TypeAlias = tuple[str, str | None, str | None, tuple[SignatureKey, ...]]


@dataclass
class _DiscoveredMethodTable:
    cursor: Cursor
    name: str
    source_file: str | None
    functions: list[ExtractedFunction] = field(default_factory=list)


@dataclass
class _DiscoveredModuleDef:
    cursor: Cursor
    name: str
    module_name: str | None
    methods_table_cursor: Cursor | None
    source_file: str | None


@dataclass
class _DiscoveredTypeDef:
    cursor: Cursor
    name: str
    tp_name: str | None
    methods_table_cursor: Cursor | None
    source_file: str | None


@dataclass
class _RegisteredTypeRef:
    exported_name: str | None
    type_def: _DiscoveredTypeDef

transparent_cursor_kinds = {
    # 没有暴露给python libclang的表达式
    # 比如ImplicitCastExpr、RecoveryExpr（解析错误的时候产生）
    CursorKind.UNEXPOSED_EXPR,
    CursorKind.PAREN_EXPR,  # (expr)
    CursorKind.CSTYLE_CAST_EXPR,  # (T)expr
    CursorKind.CXX_STATIC_CAST_EXPR,  # static_cast<T>(expr)
    CursorKind.CXX_REINTERPRET_CAST_EXPR,  # reinterpret_cast<T>(expr)
    CursorKind.CXX_CONST_CAST_EXPR,  # const_cast<T>(expr)
    CursorKind.CXX_FUNCTIONAL_CAST_EXPR,  # T(expr)
}

cast_cursor_kinds = {
    CursorKind.CSTYLE_CAST_EXPR,
    CursorKind.CXX_STATIC_CAST_EXPR,
    CursorKind.CXX_REINTERPRET_CAST_EXPR,
    CursorKind.CXX_CONST_CAST_EXPR,
    CursorKind.CXX_FUNCTIONAL_CAST_EXPR,
}

array_type_kinds = {
    TypeKind.CONSTANTARRAY,  # 固定长度数组，如 `int values[8]`
    TypeKind.INCOMPLETEARRAY,  # 不完整数组，如 `extern int values[]`
    TypeKind.VARIABLEARRAY,  # 变长数组，如 `int values[n]`
    TypeKind.DEPENDENTSIZEDARRAY,  # 依赖表达式推导长度的数组，多见于模板/泛型上下文
}

cpp_nullptr_literal_cursor_kinds = {
    CursorKind.CXX_NULL_PTR_LITERAL_EXPR,  # nullptr
    CursorKind.GNU_NULL_EXPR,  # GNU 扩展 __null
}

record_cursor_kinds = {
    CursorKind.STRUCT_DECL,
    CursorKind.UNION_DECL,
    CursorKind.CLASS_DECL,
}

_CPP_STRING_LITERAL_RE = re.compile(r'^(?:u8|u|U|L)?"(.*)"$', re.DOTALL)


def _unwrap_transparent(cursor: Cursor) -> Cursor:
    while cursor.kind in transparent_cursor_kinds:
        children = list(cursor.get_children())
        if not children:
            break
        if cursor.kind in cast_cursor_kinds:
            cursor = children[-1]
        else:
            cursor = children[0]
    return cursor


# llvm-project issue #68340
def _is_integer_literal_value(cursor: Cursor, value: int) -> bool:
    """
    是0整数字面量，NULL如果展开为((void*)0)也会走这里
    """
    if cursor.kind != CursorKind.INTEGER_LITERAL:
        return False
    ret = ClangEval.eval_int(cursor)
    if ret is None:
        return False
    return ret == value


def _is_PyMethodDef_array_sentinel(element: Cursor) -> bool:
    """判断 `PyMethodDef` 条目是否为数组结束哨兵。只要ml_name语义上为0就判定为哨兵"""

    def _is_null_token(node: Cursor) -> bool:
        return any(str(token.spelling) == "NULL" for token in node.get_tokens())

    def _is_semantic_zero(node: Cursor) -> bool:
        target = _unwrap_transparent(node)
        if target.kind in cpp_nullptr_literal_cursor_kinds:
            return True
        if _is_integer_literal_value(target, 0):
            return True
        return _is_null_token(target)

    if element.kind != CursorKind.INIT_LIST_EXPR:
        return False

    fields = list(element.get_children())

    # {}
    if not fields:
        return True

    return _is_semantic_zero(fields[0])


def _get_diagnostic_severity_name(severity: int) -> str:
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


def _format_single_diagnostic(diagnostic: Diagnostic) -> str:
    """将单条 clang diagnostic 格式化为稳定的一行文本。"""
    severity = _get_diagnostic_severity_name(diagnostic.severity)
    location: clang.cindex.SourceLocation = diagnostic.location
    diag_file = location.file.name
    line = location.line
    column = location.column
    message = diagnostic.spelling
    return f"[{severity}] {diag_file}:{line}:{column}: {message}"


def _format_diagnostics_message(
        *,
        file_path: Path,
        parse_args: list[str],
        diagnostics: list[Diagnostic],
) -> str:
    """格式化包含 error/fatal diagnostics 的日志块。"""
    lines = [
        f"Translation unit diagnostics",
        f"  file_path: {file_path}",
        f"  suffix: {file_path.suffix.lower() or '<none>'}",
        f"  parse_args: {parse_args!r}",
        "  diagnostics:",
    ]
    lines.extend(f"    {_format_single_diagnostic(diag)}" for diag in diagnostics)
    return "\n".join(lines)


def _get_packaged_libclang_path() -> str | None:
    """从 `clang` 包的 `native` 目录探测可用的 `libclang` 动态库。"""
    native_dir = Path(clang.__file__).resolve().parent / "native"
    for filename in ("libclang.dll", "libclang.so", "libclang.dylib"):
        candidate = native_dir / filename
        if candidate.exists():
            return str(candidate)
    return None

def _is_PyMethodDef_array_definition(cursor: Cursor) -> bool:
    """判断节点是否为 `PyMethodDef[]`。"""
    if cursor.type.kind in array_type_kinds and cursor.is_definition():
        elem_type = cursor.type.get_array_element_type()
        if elem_type.spelling in {"PyMethodDef", "struct PyMethodDef"}:
            return True
    return False


class CSignatureExtractor:
    """
    基于 libclang 的 C 签名提取引擎。

    该引擎从 `PyInit_*` 入口函数出发，沿 clang AST 的 `referenced` 关系
    定位 `PyModuleDef` / `PyTypeObject` / `PyMethodDef`，再结合 `PyArg_*`
    调用和格式串规则推断 Python 侧参数信息。
    """

    def __init__(
            self,
            source_root: Path,
            *,
            clang_include: list[str] = (),
            clang_include_directory: list[str] = (),
            clang_c_std: str = "c11",
            clang_cpp_std: str = "c++17",
    ) -> None:
        """初始化提取器并准备惰性缓存。"""
        self.source_root = source_root
        self._clang_include = [str(include_value) for include_value in clang_include]
        self._clang_include_directory = [str(include_path) for include_path in clang_include_directory]
        self._clang_c_std = clang_c_std
        self._clang_cpp_std = clang_cpp_std
        self._cache_modules: dict[str, ExtractedModule] | None = None

    def extract_modules(self) -> dict[str, ExtractedModule]:
        """执行模块级签名提取主流程。"""
        if self._cache_modules is not None:
            return self._cache_modules

        if not self.source_root.exists():
            logger.warning("source_root does not exist: %s", self.source_root)
            self._cache_modules = {}
            return self._cache_modules

        if not self._ensure_clang_ready():
            self._cache_modules = {}
            return self._cache_modules

        self._clang_include_directory = self._inject_python_include_directories(self._clang_include_directory)

        source_files = self._find_candidate_files()
        if not source_files:
            self._cache_modules = {}
            return self._cache_modules

        index = Index.create()

        translation_units: list[TranslationUnit] = []
        for file_path in source_files:
            tu = self._parse_translation_unit(index=index, file_path=file_path)
            if tu is None:
                continue
            translation_units.append(tu)

        discovered_modules: dict[str, ExtractedModule] = {}
        for tu in translation_units:
            try:
                modules = self._discover_translation_unit(tu.cursor)
            except AssertionError as ex:
                logger.exception("AssertionError", exc_info=ex)
                continue
            for module in modules:
                existing = discovered_modules.get(module.name)
                if existing is None:
                    discovered_modules[module.name] = module
                    continue
                self._merge_extracted_module(existing, module)

        for module in discovered_modules.values():
            module.functions = self._deduplicate_result(module.functions)
            for extracted_class in module.classes.values():
                extracted_class.methods = self._deduplicate_result(extracted_class.methods)

        self._cache_modules = discovered_modules
        return self._cache_modules

    def _ensure_clang_ready(self) -> bool:
        """确保 clang 运行环境可用。"""
        try:
            if not clang.cindex.Config.loaded:
                packaged_libclang_path = _get_packaged_libclang_path()
                if packaged_libclang_path:
                    clang.cindex.Config.set_library_file(packaged_libclang_path)
        except Exception as ex:  # pragma: no cover
            logger.warning("Failed to configure packaged libclang: %s", ex)
        return True

    def _inject_python_include_directories(self, include_directories: list[str]) -> list[str]:
        """向 include 目录列表注入当前 Python 头文件目录。"""
        directories = list(include_directories)
        include_candidates = [
            sysconfig.get_path("include"),
            sysconfig.get_path("platinclude"),
        ]
        for include_dir in include_candidates:
            if not include_dir:
                continue
            if include_dir in directories:
                continue
            directories.append(include_dir)
        return directories

    def _normalize_include_literal(self, include_literal: str) -> str:
        normalized = include_literal.replace("\\", "/").strip()
        while "//" in normalized:
            normalized = normalized.replace("//", "/")
        if normalized.startswith("./"):
            normalized = normalized[2:]
        return normalized

    def _split_include_literal_parts(self, include_literal: str) -> tuple[str, ...]:
        normalized = self._normalize_include_literal(include_literal)
        return tuple(part for part in normalized.split("/") if part and part != ".")

    def _extract_missing_include_literals(self, diagnostics: list[Diagnostic]) -> list[str]:
        missing: set[str] = set()
        for diagnostic in diagnostics:
            if diagnostic.severity < clang.cindex.Diagnostic.Error:
                continue
            message = str(diagnostic.spelling)
            match = AUTO_INCLUDE_MISSING_HEADER_RE.search(message)
            if match is None:
                continue
            include_literal = self._normalize_include_literal(match.group(1))
            if not include_literal:
                continue
            missing.add(include_literal)
        return sorted(missing)

    def _match_include_to_include_dir(
            self,
            *,
            header_path: Path,
            include_parts: tuple[str, ...],
    ) -> Path | None:
        if not include_parts:
            return None
        header_parts = header_path.parts
        if len(include_parts) > len(header_parts):
            return None
        suffix_parts = header_parts[-len(include_parts):]
        if any(left.casefold() != right.casefold() for left, right in zip(suffix_parts, include_parts)):
            return None
        include_root_depth = len(include_parts) - 1
        if include_root_depth >= len(header_path.parents):
            return None
        return header_path.parents[include_root_depth]

    def _build_include_candidate_rank(
            self,
            *,
            source_dir: Path,
            header_dir: Path,
            include_dir: Path,
    ) -> tuple[int, int, str, str]:
        source_parts = source_dir.resolve().parts
        header_parts = header_dir.resolve().parts
        common_prefix = 0
        for left, right in zip(source_parts, header_parts):
            if left.casefold() != right.casefold():
                break
            common_prefix += 1
        distance = (len(source_parts) - common_prefix) + (len(header_parts) - common_prefix)
        include_dir_posix = include_dir.as_posix()
        return distance, -common_prefix, include_dir_posix.casefold(), include_dir_posix

    def _resolve_missing_include_dir(self, *, include_literal: str, source_file: Path) -> Path | None:
        include_literal = self._normalize_include_literal(include_literal)
        include_parts = self._split_include_literal_parts(include_literal)
        if not include_parts:
            return None

        if not self.source_root.exists():
            return None

        best_rank: tuple[int, int, str, str] | None = None
        best_include_dir: Path | None = None
        for header_path in self.source_root.rglob(include_literal):
            if not header_path.is_file():
                continue
            include_dir = self._match_include_to_include_dir(
                header_path=header_path,
                include_parts=include_parts,
            )
            if include_dir is None:
                continue
            rank = self._build_include_candidate_rank(
                source_dir=source_file.parent,
                header_dir=header_path.parent,
                include_dir=include_dir,
            )
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_include_dir = include_dir
        return best_include_dir

    def _append_include_args(self, include_args: Iterable[str]) -> list[str]:
        added: list[str] = []
        for include_dir in include_args:
            if include_dir in self._clang_include_directory:
                continue
            if include_dir in added:
                continue
            self._clang_include_directory.append(include_dir)
            added.append(include_dir)
        return added

    def _discover_missing_include_args(
            self,
            *,
            file_path: Path,
            diagnostics: list[Diagnostic],
    ) -> list[str]:
        resolved_pairs: list[tuple[str, str]] = []
        missing_literals = self._extract_missing_include_literals(diagnostics)
        for include_literal in missing_literals:
            include_dir = self._resolve_missing_include_dir(
                include_literal=include_literal,
                source_file=file_path,
            )
            if include_dir is None:
                continue
            resolved_pairs.append((include_literal, str(include_dir)))

        added = self._append_include_args(include_dir for _, include_dir in resolved_pairs)
        if not added:
            return added

        for include_literal, include_dir in resolved_pairs:
            if include_dir not in added:
                continue
            logger.info(
                "Auto-added clang include path for missing header %s in %s: %s",
                include_literal,
                file_path,
                include_dir,
            )
        return added

    def _find_candidate_files(self) -> list[Path]:
        """查找可能包含 CPython 扩展定义的 C/C++ 源文件。"""
        candidate_markers = (
            "PyModuleDef",
            "PyMethodDef",
            "PyTypeObject",
            "PyInit_",
            "PyModule_AddObject",
            "PyModule_AddObjectRef",
            "PyModule_AddType",
        )
        result: list[Path] = []
        for path in self.source_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in NATIVE_SOURCE_SUFFIXES:
                text = path.read_text(encoding="utf-8", errors="ignore")
                if any(marker in text for marker in candidate_markers):
                    result.append(path)
        result.sort()
        return result

    def _build_std_value_for_file(self, file_path: Path) -> str:
        """按后缀为源码文件选择 C 或 C++ 标准值。"""
        suffix = file_path.suffix.lower()
        if suffix in CPP_SOURCE_SUFFIXES:
            return self._normalize_std_value(self._clang_cpp_std, default_std="c++17")
        return self._normalize_std_value(self._clang_c_std, default_std="c11")

    def _build_clang_parse_args(self, file_path: Path) -> list[str]:
        parse_args = []
        std_value = self._build_std_value_for_file(file_path)
        parse_args.extend(["--std", std_value])
        for include_value in self._clang_include:
            parse_args.extend(["--include", include_value])
        for include_dir in self._clang_include_directory:
            parse_args.extend(["--include-directory", include_dir])
        return parse_args

    def _parse_translation_unit(self, index: Index, file_path: Path) -> TranslationUnit | None:
        """解析单个源码文件为 clang translation unit。"""
        translation_unit: TranslationUnit | None = None
        diagnostics: list[Diagnostic] = []
        retry_limit = 10
        for _ in range(retry_limit):
            parse_args = self._build_clang_parse_args(file_path)
            translation_unit = index.parse(str(file_path), args=parse_args)
            diagnostics = list(translation_unit.diagnostics)
            added = self._discover_missing_include_args(
                file_path=file_path,
                diagnostics=diagnostics,
            )
            if not added:
                break

        if translation_unit is None:
            return None

        if self._has_error_diagnostics(diagnostics):
            logger.warning(
                _format_diagnostics_message(
                    file_path=file_path,
                    parse_args=parse_args,
                    diagnostics=diagnostics,
                )
            )
        return translation_unit

    def _has_error_diagnostics(self, diagnostics: list[Diagnostic]) -> bool:
        """判断 diagnostics 中是否包含 Error/Fatal 级别。"""
        for diagnostic in diagnostics:
            if diagnostic.severity >= clang.cindex.Diagnostic.Error:
                return True
        return False

    def _normalize_std_value(self, std_value: str, *, default_std: str) -> str:
        """将标准配置统一为纯标准值（如 `c11`、`c++17`）。"""
        normalized = std_value.strip()
        if not normalized:
            normalized = default_std
        if normalized.startswith("--std="):
            normalized = normalized.partition("=")[2]
        elif normalized.startswith("-std="):
            normalized = normalized.partition("=")[2]
        return normalized

    def _discover_translation_unit(
            self,
            cursor: Cursor,
    ) -> list[ExtractedModule]:
        """从单个 translation unit 的 `PyInit_*` 根节点提取模块。"""
        discovered: list[ExtractedModule] = []
        method_table_cache: dict[tuple[str | None, int, int, str, str], _DiscoveredMethodTable] = {}
        for node in self._walk(cursor):
            if node.kind != CursorKind.FUNCTION_DECL:
                continue
            if not node.is_definition():
                continue
            if not str(node.spelling).startswith("PyInit_"):
                continue
            extracted = self._extract_module_from_init(
                node,
                method_table_cache=method_table_cache,
            )
            if extracted is not None:
                discovered.append(extracted)
        return discovered

    def _extract_module_from_init(
            self,
            func_cursor: Cursor,
            *,
            method_table_cache: dict[tuple[str | None, int, int, str, str], _DiscoveredMethodTable],
    ) -> ExtractedModule | None:
        module_def: _DiscoveredModuleDef | None = None
        registered_types: list[_RegisteredTypeRef] = []

        for node in self._walk(func_cursor):
            if node.kind != CursorKind.CALL_EXPR:
                continue

            call_name = self._get_call_target_name(node)
            if call_name is None:
                continue

            arg_nodes = self._extract_call_argument_nodes(node)
            if not arg_nodes:
                continue

            if call_name.startswith("PyModule_Create"):
                referenced_var = self._resolve_referenced_var_decl(arg_nodes[0])
                candidate_module_def = self._extract_module_def(referenced_var)
                if candidate_module_def is None:
                    continue
                if module_def is None:
                    module_def = candidate_module_def
                elif not self._same_cursor(module_def.cursor, candidate_module_def.cursor):
                    logger.warning(
                        "Multiple module defs referenced from %s; keeping %s and ignoring %s",
                        func_cursor.spelling,
                        module_def.name,
                        candidate_module_def.name,
                    )
                continue

            registered = self._extract_registered_type_ref(call_name, arg_nodes)
            if registered is not None:
                registered_types.append(registered)

        if module_def is None:
            return None

        module_name = (
            module_def.module_name
            or self._module_name_from_init_func(func_cursor.spelling)
            or module_def.name
        )
        module = ExtractedModule(name=module_name)
        module.module_def_name = module_def.name
        module.init_func_name = func_cursor.spelling
        module.lookup_names.update(self._build_module_lookup_names(module_name, func_cursor.spelling))

        for source_file in {
            self._get_cursor_source_file(func_cursor),
            module_def.source_file,
        }:
            if source_file is not None:
                module.source_files.add(source_file)

        module_method_table = self._extract_method_table(
            module_def.methods_table_cursor,
            method_table_cache=method_table_cache,
        )
        if module_method_table is not None:
            if module_method_table.source_file is not None:
                module.source_files.add(module_method_table.source_file)
            self._merge_discovered_functions(module.functions, module_method_table.functions)

        for registered in registered_types:
            type_def = registered.type_def
            class_name = (
                registered.exported_name
                or self._extract_type_leaf_name(type_def.tp_name)
                or type_def.name
            )
            extracted_class = module.classes.get(class_name)
            if extracted_class is None:
                extracted_class = ExtractedClass(name=class_name)
                module.classes[class_name] = extracted_class

            extracted_class.c_type_name = extracted_class.c_type_name or type_def.name
            extracted_class.tp_name = extracted_class.tp_name or type_def.tp_name
            extracted_class.source_file = extracted_class.source_file or type_def.source_file

            if type_def.source_file is not None:
                module.source_files.add(type_def.source_file)

            type_method_table = self._extract_method_table(
                type_def.methods_table_cursor,
                method_table_cache=method_table_cache,
            )
            if type_method_table is None:
                continue
            if type_method_table.source_file is not None:
                module.source_files.add(type_method_table.source_file)
            self._merge_discovered_functions(
                extracted_class.methods,
                self._specialize_functions_for_class(
                    type_method_table.functions,
                    class_name=class_name,
                ),
            )

        module.functions = self._deduplicate_result(module.functions)
        for extracted_class in module.classes.values():
            extracted_class.methods = self._deduplicate_result(extracted_class.methods)
        return module

    def _merge_extracted_module(self, target: ExtractedModule, incoming: ExtractedModule) -> None:
        target.lookup_names.update(incoming.lookup_names)
        target.source_files.update(incoming.source_files)
        target.module_def_name = target.module_def_name or incoming.module_def_name
        target.init_func_name = target.init_func_name or incoming.init_func_name
        for functions in incoming.functions.values():
            self._merge_discovered_functions(target.functions, functions)
        for class_name, incoming_class in incoming.classes.items():
            target_class = target.classes.get(class_name)
            if target_class is None:
                target.classes[class_name] = incoming_class
                continue
            self._merge_extracted_class(target_class, incoming_class)

    def _merge_extracted_class(self, target: ExtractedClass, incoming: ExtractedClass) -> None:
        target.c_type_name = target.c_type_name or incoming.c_type_name
        target.tp_name = target.tp_name or incoming.tp_name
        target.source_file = target.source_file or incoming.source_file
        for functions in incoming.methods.values():
            self._merge_discovered_functions(target.methods, functions)

    def _cursor_identity(self, cursor: Cursor) -> tuple[str | None, int, int, str, str]:
        location = cursor.location
        line = location.line if location is not None else 0
        column = location.column if location is not None else 0
        return (
            self._get_cursor_source_file(cursor),
            line,
            column,
            str(cursor.kind.name),
            str(cursor.spelling),
        )

    def _same_cursor(self, left: Cursor, right: Cursor) -> bool:
        return self._cursor_identity(left) == self._cursor_identity(right)

    def _is_type_definition(self, cursor: Cursor, accepted_spellings: set[str]) -> bool:
        if not cursor.is_definition():
            return False
        type_obj = cursor.type
        spellings: set[str] = set()
        if type_obj is not None:
            spellings.add(str(type_obj.spelling))
            canonical = type_obj.get_canonical()
            if canonical is not None:
                spellings.add(str(canonical.spelling))
        return bool(spellings & accepted_spellings)

    def _is_PyModuleDef_definition(self, cursor: Cursor) -> bool:
        return self._is_type_definition(
            cursor,
            {"PyModuleDef", "struct PyModuleDef"},
        )

    def _is_PyTypeObject_definition(self, cursor: Cursor) -> bool:
        return self._is_type_definition(
            cursor,
            {"PyTypeObject", "struct PyTypeObject", "struct _typeobject"},
        )

    def _extract_method_table(
            self,
            cursor: Cursor | None,
            *,
            method_table_cache: dict[tuple[str | None, int, int, str, str], _DiscoveredMethodTable] | None = None,
    ) -> _DiscoveredMethodTable | None:
        if cursor is None or cursor.kind != CursorKind.VAR_DECL:
            return None
        if not _is_PyMethodDef_array_definition(cursor):
            return None

        cache_key = self._cursor_identity(cursor)
        if method_table_cache is not None:
            cached = method_table_cache.get(cache_key)
            if cached is not None:
                return cached

        grouped: dict[str, list[ExtractedFunction]] = {}
        init_expr_node = self._array_VAR_DECL_to_INIT_LIST_EXPR(cursor)
        self._process_PyMethodDef_array_INIT_LIST_EXPR(
            cursor,
            init_expr_node,
            grouped,
        )
        deduped = self._deduplicate_result(grouped)
        functions = [item for items in deduped.values() for item in items]
        discovered = _DiscoveredMethodTable(
            cursor=cursor,
            name=cursor.spelling,
            source_file=self._get_cursor_source_file(cursor),
            functions=functions,
        )
        if method_table_cache is not None:
            method_table_cache[cache_key] = discovered
        return discovered

    def _extract_module_def(self, cursor: Cursor | None) -> _DiscoveredModuleDef | None:
        if cursor is None or cursor.kind != CursorKind.VAR_DECL:
            return None
        if not self._is_PyModuleDef_definition(cursor):
            return None

        field_values = self._extract_struct_initializer_fields(cursor)
        return _DiscoveredModuleDef(
            cursor=cursor,
            name=cursor.spelling,
            module_name=self._extract_string_literal_from_cursor(field_values.get("m_name")),
            methods_table_cursor=self._resolve_method_table_cursor(field_values.get("m_methods")),
            source_file=self._get_cursor_source_file(cursor),
        )

    def _extract_type_def(self, cursor: Cursor | None) -> _DiscoveredTypeDef | None:
        if cursor is None or cursor.kind != CursorKind.VAR_DECL:
            return None
        if not self._is_PyTypeObject_definition(cursor):
            return None

        field_values = self._extract_struct_initializer_fields(cursor)
        return _DiscoveredTypeDef(
            cursor=cursor,
            name=cursor.spelling,
            tp_name=self._extract_string_literal_from_cursor(field_values.get("tp_name")),
            methods_table_cursor=self._resolve_method_table_cursor(field_values.get("tp_methods")),
            source_file=self._get_cursor_source_file(cursor),
        )

    def _extract_registered_type_ref(
            self,
            call_name: str,
            arg_nodes: list[Cursor],
    ) -> _RegisteredTypeRef | None:
        if call_name in {"PyModule_AddObject", "PyModule_AddObjectRef"} and len(arg_nodes) >= 3:
            type_def = self._extract_type_def(self._resolve_referenced_var_decl(arg_nodes[2]))
            if type_def is None:
                return None
            return _RegisteredTypeRef(
                exported_name=self._extract_string_literal_from_cursor(arg_nodes[1]),
                type_def=type_def,
            )

        if call_name == "PyModule_AddType" and len(arg_nodes) >= 2:
            type_def = self._extract_type_def(self._resolve_referenced_var_decl(arg_nodes[1]))
            if type_def is None:
                return None
            return _RegisteredTypeRef(
                exported_name=self._extract_type_leaf_name(type_def.tp_name),
                type_def=type_def,
            )

        return None

    def _get_call_target_name(self, node: Cursor) -> str | None:
        referenced = node.referenced
        if referenced is not None and referenced.spelling:
            return str(referenced.spelling)
        return self._get_call_name(node)

    def _extract_call_argument_nodes(self, node: Cursor) -> list[Cursor]:
        children = list(node.get_children())
        if len(children) <= 1:
            return []
        return children[1:]

    def _resolve_method_table_cursor(self, cursor: Cursor | None) -> Cursor | None:
        referenced = self._resolve_referenced_var_decl(cursor)
        if referenced is None or not _is_PyMethodDef_array_definition(referenced):
            return None
        return referenced

    def _resolve_referenced_var_decl(self, cursor: Cursor | None) -> Cursor | None:
        return self._resolve_referenced_cursor(cursor, {CursorKind.VAR_DECL})

    def _resolve_referenced_cursor(
            self,
            cursor: Cursor | None,
            accepted_kinds: set[CursorKind],
    ) -> Cursor | None:
        if cursor is None:
            return None
        for node in self._walk(_unwrap_transparent(cursor)):
            referenced = node.referenced
            if referenced is None:
                continue
            if referenced.kind not in accepted_kinds:
                continue
            return referenced
        return None

    def _extract_struct_initializer_fields(self, cursor: Cursor) -> dict[str, Cursor]:
        init_list_expr = self._var_decl_to_init_list_expr(cursor)
        if init_list_expr is None:
            return {}

        field_names = self._get_record_field_names(cursor)
        if not field_names:
            return {}

        field_name_to_index = {field_name: index for index, field_name in enumerate(field_names)}
        values: dict[str, Cursor] = {}
        positional_index = 0

        for entry in init_list_expr.get_children():
            designated_name, value_cursor = self._extract_initializer_entry(entry)
            if value_cursor is None:
                continue

            if designated_name is None:
                if positional_index >= len(field_names):
                    continue
                field_name = field_names[positional_index]
                positional_index += 1
            else:
                field_name = designated_name
                designated_index = field_name_to_index.get(field_name)
                if designated_index is not None:
                    positional_index = designated_index + 1

            values[field_name] = value_cursor

        return values

    def _extract_initializer_entry(self, entry: Cursor) -> tuple[str | None, Cursor | None]:
        children = list(entry.get_children())
        if len(children) >= 2 and children[0].kind == CursorKind.MEMBER_REF:
            member_ref = children[0]
            field_name = str(member_ref.spelling) or None
            referenced = member_ref.referenced
            if referenced is not None and referenced.spelling:
                field_name = str(referenced.spelling)
            return field_name, _unwrap_transparent(children[-1])
        return None, _unwrap_transparent(entry)

    def _get_record_field_names(self, cursor: Cursor) -> list[str]:
        type_obj = cursor.type
        candidate_types = [type_obj]
        canonical = type_obj.get_canonical()
        if canonical is not None and canonical.spelling != type_obj.spelling:
            candidate_types.append(canonical)

        for candidate in candidate_types:
            record_decl = self._resolve_record_decl(candidate.get_declaration())
            if record_decl is None:
                continue
            field_names = [
                str(child.spelling)
                for child in record_decl.get_children()
                if child.kind == CursorKind.FIELD_DECL and child.spelling
            ]
            if field_names:
                return field_names
        return []

    def _resolve_record_decl(self, cursor: Cursor | None) -> Cursor | None:
        if cursor is None:
            return None

        if cursor.kind in record_cursor_kinds:
            if cursor.is_definition():
                return cursor
            definition = cursor.get_definition()
            if definition is not None:
                return definition
            return None

        if cursor.kind != CursorKind.TYPEDEF_DECL:
            return None

        for child in cursor.get_children():
            if child.kind not in record_cursor_kinds:
                continue
            if child.is_definition():
                return child
            definition = child.get_definition()
            if definition is not None:
                return definition

        underlying = cursor.underlying_typedef_type
        if underlying is None:
            return None
        underlying_decl = underlying.get_declaration()
        if underlying_decl == cursor:
            return None
        return self._resolve_record_decl(underlying_decl)

    def _var_decl_to_init_list_expr(self, cursor: Cursor) -> Cursor | None:
        for child in cursor.get_children():
            target = _unwrap_transparent(child)
            if target.kind == CursorKind.INIT_LIST_EXPR:
                return target
        return None

    def _collect_pymethod_defs(
            self,
            cursor: Cursor,
            output: dict[str, list[ExtractedFunction]],
    ) -> None:
        """在 AST 中定位 `PyMethodDef` 表并提取条目。"""
        for child in cursor.get_children():
            if child.kind == CursorKind.VAR_DECL:
                if _is_PyMethodDef_array_definition(child):
                    init_expr_node = self._array_VAR_DECL_to_INIT_LIST_EXPR(child)
                    self._process_PyMethodDef_array_INIT_LIST_EXPR(
                        child,
                        init_expr_node,
                        output,
                    )

    def _array_VAR_DECL_to_INIT_LIST_EXPR(self, cursor: Cursor) -> Cursor:
        assert cursor.kind == CursorKind.VAR_DECL
        init_list_expr = self._var_decl_to_init_list_expr(cursor)
        assert init_list_expr is not None
        assert init_list_expr.kind == CursorKind.INIT_LIST_EXPR
        return init_list_expr

    def _process_PyMethodDef_array_INIT_LIST_EXPR(
            self,
            var_decl_node: Cursor,
            init_list_expr_node: Cursor,
            output: dict[str, list[ExtractedFunction]],
    ) -> None:
        """处理单个方法表的 `INIT_LIST_EXPR` 并写入输出。"""
        assert init_list_expr_node.kind == CursorKind.INIT_LIST_EXPR

        table_name = var_decl_node.spelling
        location = var_decl_node.location
        source_file = str(location.file)
        for element in init_list_expr_node.get_children():
            if _is_PyMethodDef_array_sentinel(element):
                break
            extracted = self._extract_PyMethodDef_INIT_LIST_EXPR(
                init_list_expr=element,
                method_table=table_name,
                source_file=source_file,
            )
            if extracted is None:
                continue
            output.setdefault(extracted.py_name, []).append(extracted)

    def _extract_PyMethodDef_INIT_LIST_EXPR(
            self,
            init_list_expr: Cursor,
            method_table: str,
            source_file: str | None,
    ) -> ExtractedFunction | None:
        """
        从 `PyMethodDef` 的单个初始化项提取函数元数据和签名。

        若关键字段（Python 名、C 函数名）缺失则返回 `None`，
        保持提取过程对异常样本的容错性。
        """
        # {"name", func, METH_VARARGS, NULL}
        # INIT_LIST_EXPR
        #   UNEXPOSED_EXPR
        #     UNEXPOSED_EXPR
        #       STRING_LITERAL
        #   UNEXPOSED_EXPR
        #     DECL_REF_EXPR
        #   INTEGER_LITERAL
        assert init_list_expr.kind == CursorKind.INIT_LIST_EXPR

        fields = list(init_list_expr.get_children())
        assert len(fields) >= 3

        ml_name_cursor = _unwrap_transparent(fields[0])
        assert ml_name_cursor.kind == CursorKind.STRING_LITERAL
        ml_name = self._strip_string_literal_quotes(ml_name_cursor.spelling)

        ml_meth_cursor = _unwrap_transparent(fields[1])
        assert ml_meth_cursor.kind == CursorKind.DECL_REF_EXPR
        ml_meth = ml_meth_cursor.spelling

        ml_flags = self._extract_PyMethodDef_ml_flags(fields[2])

        function_cursor = ml_meth_cursor.referenced
        if function_cursor is None:
            logger.warning(f"cant find function cursor, location: {ml_meth_cursor.location}")
            return None

        signatures: list[ExtractedSignature] = []
        return_type_name: str | None = None
        signatures = self._extract_signatures_from_function(function_cursor, ml_flags)
        return_type_name = self._infer_return_type_from_function(function_cursor)
        if not signatures:
            # 解析不到 PyArg_* 调用时，回退到 C 形参声明推断。
            fallback = self._signature_from_param_decls(function_cursor)
            if fallback.arguments:
                signatures = [fallback]

        if not signatures:
            signatures = [ExtractedSignature(arguments=[], return_type_name=return_type_name)]

        signatures = [self._merge_signature_return_type(sig, return_type_name) for sig in signatures]
        signatures = [self._apply_method_flags(sig, ml_flags) for sig in signatures]
        signatures = self._deduplicate_signatures(signatures)
        return ExtractedFunction(
            py_name=ml_name,
            c_name=ml_meth,
            method_flags=ml_flags,
            signatures=signatures,
            source_file=source_file,
            method_table=method_table,
        )


    def _extract_PyMethodDef_ml_flags(self, field_cursor: Cursor) -> list[str]:
        """从 `ml_flags` 字段 AST 子树中提取 `METH_*` 列表。"""
        flags: list[str] = []
        for node in self._walk(field_cursor):
            for token in node.get_tokens():
                spelling = str(token.spelling)
                if token.kind == TokenKind.IDENTIFIER and spelling.startswith("METH_"):
                    flags.append(spelling)
                    continue
                if token.kind == TokenKind.LITERAL:
                    flags.extend(self._decode_meth_literal_flags(spelling))
        return self._unique_keep_order(flags)


    def _extract_signatures_from_function(self, func_cursor: Cursor, meth_flags: list[str]) -> list[ExtractedSignature]:
        """从函数体中提取候选签名，并在末尾做去重。"""
        signatures: list[ExtractedSignature] = []
        for token_list in self._collect_pyarg_token_lists(func_cursor):
            args = self._set_token_params(func_cursor, meth_flags, token_list)
            if args is not None:
                signatures.append(ExtractedSignature(arguments=args))

        decl_stmt = CursorKind.DECL_STMT
        for node in self._walk(func_cursor):
            if node.kind == decl_stmt:
                signatures.extend(self._extract_parser_signatures(node))
        return self._deduplicate_signatures(signatures)

    def _merge_signature_return_type(
            self,
            signature: ExtractedSignature,
            return_type_name: str | None,
    ) -> ExtractedSignature:
        """在签名未显式设置返回值时填充函数级推断结果。"""
        if return_type_name is None or signature.return_type_name is not None:
            return signature
        return ExtractedSignature(
            arguments=list(signature.arguments),
            return_type_name=return_type_name,
        )

    def _infer_return_type_from_function(self, func_cursor: Cursor) -> str | None:
        """
        从函数体中推断 Python 返回类型。

        规则：
        - 优先收集 `return` 语句中的显式工厂函数与宏。
        - 若出现多个已识别返回类型，回退为 `object`。
        """
        inferred_types: set[str] = set()

        all_tokens = set(self._collect_identifier_literal_tokens(func_cursor))
        for macro_name, type_name in RETURN_MACRO_TYPE_MAP.items():
            if macro_name in all_tokens:
                inferred_types.add(type_name)

        for node in self._walk(func_cursor):
            if node.kind != CursorKind.RETURN_STMT:
                continue
            inferred = self._infer_return_type_from_return_stmt(return_stmt=node, func_cursor=func_cursor)
            if inferred is not None:
                inferred_types.add(inferred)

        if not inferred_types:
            return None
        if len(inferred_types) == 1:
            return sorted(inferred_types)[0]
        return "object"

    def _infer_return_type_from_return_stmt(self, return_stmt: Cursor, func_cursor: Cursor) -> str | None:
        """从单条 `return` 语句中提取可识别的返回类型。"""
        tokens = self._collect_identifier_literal_tokens(return_stmt)
        if not tokens:
            return None
        token_set = set(tokens)

        for macro_name, type_name in RETURN_MACRO_TYPE_MAP.items():
            if macro_name in token_set:
                return type_name
        for token_name, type_name in RETURN_TOKEN_TYPE_MAP.items():
            if token_name in token_set:
                return type_name

        call_name = self._find_first_call_name(return_stmt)
        if call_name is None:
            return None
        return self._infer_return_type_from_call(
            call_name=call_name,
            return_stmt=return_stmt,
            func_cursor=func_cursor,
        )

    def _infer_return_type_from_call(
            self,
            *,
            call_name: str,
            return_stmt: Cursor,
            func_cursor: Cursor,
    ) -> str | None:
        """根据返回调用名推断 Python 返回类型。"""
        if call_name == "Py_BuildValue":
            return self._infer_return_type_from_py_buildvalue(return_stmt=return_stmt, func_cursor=func_cursor)

        if call_name == "Py_NewRef":
            token_set = set(self._collect_identifier_literal_tokens(return_stmt))
            if "Py_None" in token_set:
                return "None"
            if "Py_True" in token_set or "Py_False" in token_set:
                return "bool"
            return "object"

        for prefix, type_name in RETURN_CALL_PREFIX_TYPE_MAP:
            if call_name.startswith(prefix):
                return type_name

        if call_name.startswith("PyObject_Call"):
            return "object"
        return None

    def _infer_return_type_from_py_buildvalue(self, return_stmt: Cursor, func_cursor: Cursor) -> str:
        """从 `Py_BuildValue` 的格式串推断返回类型。"""
        tokens = self._collect_identifier_literal_tokens(return_stmt)
        if "Py_BuildValue" not in tokens:
            return "object"

        call_idx = tokens.index("Py_BuildValue")
        for token in tokens[call_idx + 1:]:
            format_text = self._resolve_buildvalue_format_token(func_cursor=func_cursor, token=token)
            if format_text is None:
                continue
            return self._infer_return_type_from_buildvalue_format(format_text)
        return "object"

    def _resolve_buildvalue_format_token(self, *, func_cursor: Cursor, token: str) -> str | None:
        """解析 `Py_BuildValue` 的格式串 token。"""
        if '"' in token:
            return self._strip_string_literal_quotes(token)
        if self._looks_like_identifier(token):
            return self._find_format_string(func_cursor=func_cursor, format_var_name=token)
        return None

    def _infer_return_type_from_buildvalue_format(self, format_text: str) -> str:
        """根据 `Py_BuildValue` 格式串估算返回类型。"""
        markers = self._explode_buildvalue_format_string(format_text)
        if not markers:
            return "None"

        if "{" in markers:
            return "dict"
        if "[" in markers:
            return "list"
        if "(" in markers:
            return "tuple"

        value_markers = [m for m in markers if m not in {"(", ")", "[", "]", "{", "}"}]
        if not value_markers:
            return "None"
        if len(value_markers) > 1:
            return "tuple"
        return PY_BUILDVALUE_SINGLE_MARKER_TYPE_MAP.get(value_markers[0], "object")

    def _explode_buildvalue_format_string(self, format_text: str) -> list[str]:
        """
        拆分 `Py_BuildValue` 格式串为返回值 marker。

        与 `PyArg_*` 的解析不同，这里不会把 `s#`/`z#` 等拆成两个参数位。
        """
        result: list[str] = []
        idx = 0
        while idx < len(format_text):
            current = format_text[idx]
            if current in {" ", ","}:
                idx += 1
                continue
            if current in {"(", ")", "[", "]", "{", "}"}:
                result.append(current)
                idx += 1
                continue
            if current in {":", ";"}:
                break
            if idx + 1 < len(format_text) and format_text[idx + 1] in {"!", "#", "&", "*"}:
                result.append(current + format_text[idx + 1])
                idx += 2
                continue
            if current not in {"!", "#", "&"}:
                result.append(current)
            idx += 1
        return result

    def _find_first_call_name(self, node: Cursor) -> str | None:
        """在子树中查找首个 `CALL_EXPR` 的函数名。"""
        for child in self._walk(node):
            if child.kind == CursorKind.CALL_EXPR and child.spelling:
                return str(child.spelling)
        return None

    def _collect_identifier_literal_tokens(self, node: Cursor) -> list[str]:
        """收集 cursor 子树中的 `IDENTIFIER` / `LITERAL` token。"""
        return [
            str(token.spelling)
            for token in node.get_tokens()
            if token.kind in {TokenKind.IDENTIFIER, TokenKind.LITERAL}
        ]

    def _collect_pyarg_token_lists(self, node: Cursor) -> list[list[str]]:
        """
        递归收集参数解析调用（`PyArg_*`）的 token 序列。

        `IF_STMT` 与 `UNEXPOSED_EXPR` 在不同编译单元下结构可能不同，
        因此这里采用保守递归策略统一处理。
        """
        result: list[list[str]] = []

        for child in node.get_children():
            token_list: list[str] | None = None
            if child.kind == CursorKind.CALL_EXPR:
                token_list = self._collect_call_tokens(child)
            elif child.kind == CursorKind.IF_STMT:
                first_child = next(child.get_children(), None)
                if first_child is not None and first_child.kind == CursorKind.UNEXPOSED_EXPR:
                    token_list = self._collect_call_tokens(first_child)
                else:
                    token_list = self._collect_call_tokens(child)

            if token_list and self._is_parameter_parser_call(token_list[0]):
                result.append(token_list)
                continue

            result.extend(self._collect_pyarg_token_lists(child))
        return result

    def _extract_parser_signatures(self, node: Cursor) -> list[ExtractedSignature]:
        """从声明语句中的 `\"func(type name, ...)\"` 文本签名提取参数。"""
        signatures: list[ExtractedSignature] = []
        literals: list[str] = []
        for token in node.get_tokens():
            if token.kind != TokenKind.LITERAL:
                continue
            text = self._strip_string_literal_quotes(str(token.spelling))
            if "(" in text and ")" in text:
                literals.append(text)
        for text in literals:
            args_part = text[text.find("(") + 1: text.rfind(")")]
            args = self._parse_parser_args(args_part)
            if args:
                signatures.append(ExtractedSignature(arguments=args))
        return signatures

    def _parse_parser_args(self, args_text: str) -> list[ExtractedArgument]:
        """
        解析 parser 风格的参数文本。

        支持默认值、`*`/`$`、`*args`/`**kwargs`，并在信息不足时生成占位参数名。
        """
        parts = self._split_top_level(args_text, ",")
        result: list[ExtractedArgument] = []
        kw_only = False
        for raw_part in parts:
            part = raw_part.strip()
            if not part:
                continue
            if part in {"*", "$"}:
                kw_only = True
                continue

            default: str | None = None
            if "=" in part:
                before_default, default = part.split("=", 1)
            else:
                before_default = part
            before_default = before_default.strip()

            type_name = "object"
            if " " in before_default:
                type_candidate, name_candidate = before_default.rsplit(" ", 1)
                if self._looks_like_identifier(name_candidate):
                    name = name_candidate
                    type_name = self._normalize_parser_type(type_candidate)
                else:
                    name = f"arg{len(result) + 1}"
                    type_name = self._normalize_parser_type(before_default)
            else:
                name = before_default if self._looks_like_identifier(before_default) else f"arg{len(result) + 1}"
                if name != before_default:
                    type_name = self._normalize_parser_type(before_default)

            kind = "keyword_only" if kw_only else "positional_or_keyword"
            if name.startswith("**"):
                kind = "var_keyword"
                name = name[2:]
            elif name.startswith("*"):
                kind = "var_positional"
                name = name[1:]
                kw_only = True

            result.append(
                ExtractedArgument(
                    name=name,
                    type_name=type_name,
                    default_value=default.strip() if default else None,
                    kind=kind,
                )
            )
        return result

    def _set_token_params(
            self,
            func_cursor: Cursor,
            meth_flags: list[str],
            token_list: list[str],
    ) -> list[ExtractedArgument] | None:
        """
        基于调用 token 与方法标志推断参数名、类型和默认值。

        返回 `None` 表示无法可靠解析该调用；返回空列表表示明确无参数。
        """
        if not token_list:
            return None

        call_name = token_list[0]
        if not self._is_parameter_parser_call(call_name):
            return None

        format_idx, offset = self._resolve_format_index(call_name=call_name, meth_flags=meth_flags)
        if format_idx < 0:
            return None
        if format_idx >= len(token_list):
            return None

        format_markers: list[str] | None = None
        format_token_index = format_idx
        for idx in range(format_idx, len(token_list)):
            parsed = self._parse_format_token(func_cursor, token_list[idx])
            if parsed is None:
                continue
            format_markers = parsed
            format_token_index = idx
            break

        if format_markers is None:
            return None

        required, optional = self._split_required_optional(format_markers)
        param_cursor = format_token_index + offset
        # 某些调用会把 kwlist 放在格式串后，实际参数名要从其后开始。
        if param_cursor < len(token_list) and token_list[param_cursor] == "kwlist":
            param_cursor += 1

        result: list[ExtractedArgument] = []
        kw_only = False
        for marker, is_optional in ([(m, False) for m in required] + [(m, True) for m in optional]):
            if marker in {"*", "$"}:
                kw_only = True
                continue
            if marker == ":":
                break
            if param_cursor >= len(token_list):
                break

            name = token_list[param_cursor].strip("\"")
            param_cursor += 1
            if not self._looks_like_identifier(name):
                continue

            type_name = FORMAT_TYPE_MAP.get(marker, "object")
            default_value = self._get_init_value(name=name, func_cursor=func_cursor) if is_optional else None
            kind = "keyword_only" if kw_only else "positional_or_keyword"
            result.append(
                ExtractedArgument(
                    name=name,
                    type_name=type_name,
                    default_value=default_value,
                    kind=kind,
                )
            )

        return result

    def _is_parameter_parser_call(self, call_name: str) -> bool:
        """判断调用名是否属于可提取参数签名的 `PyArg_*` 解析 API。"""
        if call_name.startswith("PyArg_Parse"):
            return True
        return call_name in {"PyArg_UnpackTuple"}

    def _decode_meth_literal_flags(self, literal: str) -> list[str]:
        """把数字字面量（含位掩码组合）解析为 `METH_*` 列表。"""
        text = literal.strip()
        if not text:
            return []
        text = re.sub(r"[uUlL]+$", "", text)
        try:
            mask = int(text, 0)
        except ValueError:
            return []
        if mask <= 0:
            return []

        result: list[str] = []
        for raw_bit, flag_name in METH_TYPE_LITERAL_MAP.items():
            try:
                bit = int(raw_bit, 0)
            except ValueError:
                continue
            if bit and (mask & bit) == bit:
                result.append(flag_name)
        return self._unique_keep_order(result)

    def _resolve_format_index(self, call_name: str, meth_flags: list[str]) -> tuple[int, int]:
        """
        按 `METH_*` 约定推导格式串索引与参数偏移量。

        不同 `PyArg_*` API 的实参布局不同，这里统一归一化为
        `(format_idx, offset)` 供后续解析复用。
        """
        has_keywords = "METH_KEYWORDS" in meth_flags
        has_varargs = "METH_VARARGS" in meth_flags or "METH_FASTCALL" in meth_flags
        if has_keywords and has_varargs:
            if call_name == "PyArg_ParseTupleAndKeywords":
                return 3, 2
            if call_name == "PyArg_ParseTuple":
                return 2, 1
            if call_name == "PyArg_NoKeywords":
                return -1, 0
            return 3, 2
        if "METH_VARARGS" in meth_flags or "METH_O" in meth_flags or "METH_FASTCALL" in meth_flags:
            return 2, 1
        if has_keywords:
            return 3, 2
        if "METH_NOARGS" in meth_flags:
            return -1, 0
        return 2, 1

    def _parse_format_token(self, func_cursor: Cursor, token: str) -> list[str] | None:
        """将格式串 token 解析为 marker 列表，支持字面量与变量两种来源。"""
        if token == "F_INT_PYFMT":
            return ["F_INT_PYFMT"]
        if '"' in token:
            text = self._strip_string_literal_quotes(token)
            return self._explode_format_string(text)
        if self._looks_like_identifier(token):
            text = self._find_format_string(func_cursor=func_cursor, format_var_name=token)
            if text:
                return self._explode_format_string(text)
        return None

    def _explode_format_string(self, format_text: str) -> list[str]:
        """
        拆分 CPython 格式串为逐参数 marker。

        对 `s#` / `z#` / `y#` / `O!` 这类“一个标记对应多个 C 参数”的情况，
        会展开成 `xxx1`/`xxx2` 以便后续一一对齐参数名。
        """
        result: list[str] = []
        idx = 0
        while idx < len(format_text):
            current = format_text[idx]
            if current in {" ", ",", "(", ")", "[", "]"}:
                idx += 1
                continue
            if current in {"|", "*", "$", ":"}:
                result.append(current)
                idx += 1
                if current == ":":
                    break
                continue
            if idx + 1 < len(format_text) and format_text[idx + 1] in {"!", "#", "&", "*"}:
                duo = current + format_text[idx + 1]
                if duo in {"s#", "z#", "y#", "O!"}:
                    # 这几类格式符消费两个 C 参数，需要拆成两段占位 marker。
                    result.extend([f"{duo}1", f"{duo}2"])
                    idx += 2
                    continue
                if duo in {"y*", "z*", "s*", "O&"}:
                    result.append(duo)
                    idx += 2
                    continue
            if current not in {"!", "#", "&"}:
                result.append(current)
            idx += 1
        return result

    def _split_required_optional(self, markers: list[str]) -> tuple[list[str], list[str]]:
        """按 `|` 分隔必填/可选参数，并在 `:` 处截断函数名后缀。"""
        required: list[str] = []
        optional: list[str] = []
        target = required
        for marker in markers:
            if marker == ":":
                break
            if marker == "|":
                target = optional
                continue
            target.append(marker)
        return required, optional

    def _collect_call_tokens(self, call_node: Cursor) -> list[str]:
        """
        从调用表达式提取与参数解析相关的 token。

        该步骤会过滤大量无关转换器/宏标识，降低误判概率。
        """
        result: list[str] = []
        started = False
        for token in call_node.get_tokens():
            if token.kind not in {TokenKind.IDENTIFIER, TokenKind.LITERAL}:
                continue
            spelling = str(token.spelling)
            if not started:
                if spelling.startswith("PyArg_") or spelling == "Py_BuildValue":
                    result.append(spelling)
                    started = True
                continue

            if spelling in {"NULL", "return"}:
                break
            if spelling in UNRELATED_TOKENS or spelling in {"if"}:
                # 跳过已知噪声 token，避免污染参数名/类型推断。
                continue
            result.append(spelling)
        return result

    def _signature_from_param_decls(self, func_cursor: Cursor) -> ExtractedSignature:
        """回退方案：直接从 C 形参声明推断签名。"""
        args: list[ExtractedArgument] = []
        for node in func_cursor.get_children():
            if node.kind != CursorKind.PARM_DECL or not node.spelling:
                continue
            args.append(
                ExtractedArgument(
                    name=node.spelling,
                    type_name=self._normalize_c_type(node.type.spelling),
                    default_value=None,
                )
            )
        return ExtractedSignature(arguments=args)

    def _apply_method_flags(self, signature: ExtractedSignature, meth_flags: list[str]) -> ExtractedSignature:
        """
        根据 `METH_*` 规则修正首参语义。

        该步骤负责统一 `self`/`cls` 行为，避免来源差异导致的方法签名不一致。
        """
        args = list(signature.arguments)
        return_type_name = signature.return_type_name
        if "METH_STATIC" in meth_flags:
            while args and args[0].name in {"self", "cls"}:
                args.pop(0)
            if "METH_NOARGS" in meth_flags:
                return ExtractedSignature(arguments=[], return_type_name=return_type_name)
            return ExtractedSignature(arguments=args, return_type_name=return_type_name)
        if "METH_CLASS" in meth_flags:
            if not args or args[0].name not in {"cls", "self"}:
                args.insert(0, ExtractedArgument(name="cls", type_name="type"))
            else:
                args[0].name = "cls"
                args[0].type_name = "type"
            return ExtractedSignature(arguments=args, return_type_name=return_type_name)
        if not args or args[0].name not in {"self", "cls"}:
            args.insert(0, ExtractedArgument(name="self", type_name="object"))
        else:
            args[0].name = "self"
            args[0].type_name = "object"
        return ExtractedSignature(arguments=args, return_type_name=return_type_name)

    def _specialize_functions_for_class(
            self,
            functions: list[ExtractedFunction],
            *,
            class_name: str,
    ) -> list[ExtractedFunction]:
        """将类方法候选的 `self` 标注专化为具体类名，避免统一降级成 `object`。"""
        return [
            ExtractedFunction(
                py_name=function.py_name,
                c_name=function.c_name,
                method_flags=list(function.method_flags),
                signatures=[
                    self._specialize_signature_for_class(
                        signature,
                        meth_flags=function.method_flags,
                        class_name=class_name,
                    )
                    for signature in function.signatures
                ],
                source_file=function.source_file,
                method_table=function.method_table,
            )
            for function in functions
        ]

    def _specialize_signature_for_class(
            self,
            signature: ExtractedSignature,
            *,
            meth_flags: list[str],
            class_name: str,
    ) -> ExtractedSignature:
        args = [
            ExtractedArgument(
                name=arg.name,
                type_name=arg.type_name,
                default_value=arg.default_value,
                kind=arg.kind,
            )
            for arg in signature.arguments
        ]
        if "METH_STATIC" not in meth_flags and args and args[0].name == "self":
            args[0].type_name = class_name
        return ExtractedSignature(
            arguments=args,
            return_type_name=signature.return_type_name,
        )

    def _get_cursor_source_file(self, cursor: Cursor) -> str | None:
        location = cursor.location
        if location is None:
            return None
        file_obj = location.file
        if file_obj is None:
            return None
        return str(file_obj)

    def _get_token_spellings(self, node: Cursor) -> list[str]:
        return [str(token.spelling) for token in node.get_tokens()]

    def _extract_var_decl_initializer_entries(self, cursor: Cursor) -> list[list[str]] | None:
        tokens = self._get_token_spellings(cursor)
        if "{" not in tokens:
            return None
        start = tokens.index("{")
        end = self._find_matching_token(tokens, start, "{", "}")
        if end is None:
            return None
        inner = tokens[start + 1:end]
        return self._split_top_level_tokens(inner, ",")

    def _find_matching_token(
            self,
            tokens: list[str],
            start: int,
            opening: str,
            closing: str,
    ) -> int | None:
        depth = 0
        for index in range(start, len(tokens)):
            token = tokens[index]
            if token == opening:
                depth += 1
                continue
            if token != closing:
                continue
            depth -= 1
            if depth == 0:
                return index
        return None

    def _split_top_level_tokens(self, tokens: list[str], delimiter: str) -> list[list[str]]:
        groups: list[list[str]] = []
        current: list[str] = []
        brace_depth = 0
        paren_depth = 0
        bracket_depth = 0
        for token in tokens:
            if token == delimiter and brace_depth == 0 and paren_depth == 0 and bracket_depth == 0:
                groups.append(current)
                current = []
                continue

            current.append(token)
            if token == "{":
                brace_depth += 1
            elif token == "}":
                brace_depth = max(0, brace_depth - 1)
            elif token == "(":
                paren_depth += 1
            elif token == ")":
                paren_depth = max(0, paren_depth - 1)
            elif token == "[":
                bracket_depth += 1
            elif token == "]":
                bracket_depth = max(0, bracket_depth - 1)

        groups.append(current)
        return [group for group in groups if group]

    def _extract_designated_field_name(self, tokens: list[str]) -> str | None:
        if len(tokens) >= 3 and tokens[0] == "." and tokens[2] == "=":
            return tokens[1]
        if len(tokens) >= 2 and tokens[0].startswith(".") and tokens[1] == "=":
            return tokens[0][1:]
        return None

    def _extract_designated_initializer_map(self, entries: list[list[str]]) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for entry in entries:
            field_name = self._extract_designated_field_name(entry)
            if field_name is None:
                continue
            if entry[0] == ".":
                result[field_name] = entry[3:]
            else:
                result[field_name] = entry[2:]
        return result

    def _extract_string_literal_from_cursor(self, cursor: Cursor | None) -> str | None:
        if cursor is None:
            return None
        for node in self._walk(_unwrap_transparent(cursor)):
            if node.kind != CursorKind.STRING_LITERAL:
                continue
            return self._strip_string_literal_quotes(str(node.spelling))
        return None

    def _extract_string_literal_from_tokens(self, tokens: list[str]) -> str | None:
        for token in tokens:
            if "\"" not in token:
                continue
            return self._strip_string_literal_quotes(token)
        return None

    def _extract_reference_name(
            self,
            tokens: list[str],
            *,
            require_address_of: bool = False,
    ) -> str | None:
        null_identifiers = {"NULL", "nullptr", "__null"}
        if require_address_of:
            for index in range(len(tokens) - 1, -1, -1):
                if tokens[index] != "&":
                    continue
                for token in tokens[index + 1:]:
                    if self._looks_like_identifier(token) and token not in null_identifiers:
                        return token
                return None
            return None

        if "&" in tokens:
            resolved = self._extract_reference_name(tokens, require_address_of=True)
            if resolved is not None:
                return resolved

        identifiers = [
            token
            for token in tokens
            if self._looks_like_identifier(token) and token not in null_identifiers
        ]
        if not identifiers:
            return None
        if len(identifiers) == 1:
            return identifiers[0]
        return identifiers[-1]

    def _extract_call_argument_groups(self, node: Cursor) -> list[list[str]]:
        tokens = self._get_token_spellings(node)
        if "(" not in tokens:
            return []
        start = tokens.index("(")
        end = self._find_matching_token(tokens, start, "(", ")")
        if end is None:
            return []
        inner = tokens[start + 1:end]
        return self._split_top_level_tokens(inner, ",")

    def _get_call_name(self, node: Cursor) -> str | None:
        if node.spelling:
            return str(node.spelling)
        tokens = self._get_token_spellings(node)
        for token in tokens:
            if self._looks_like_identifier(token):
                return token
        return None

    def _build_module_lookup_names(self, module_name: str, init_func_name: str | None) -> set[str]:
        lookup_names = {module_name}
        leaf_name = module_name.rsplit(".", 1)[-1]
        lookup_names.add(leaf_name)
        init_leaf = self._module_name_from_init_func(init_func_name)
        if init_leaf is not None:
            lookup_names.add(init_leaf)
        return lookup_names

    def _module_name_from_init_func(self, init_func_name: str | None) -> str | None:
        if not init_func_name:
            return None
        if not init_func_name.startswith("PyInit_"):
            return None
        return init_func_name[len("PyInit_"):]

    def _extract_type_leaf_name(self, tp_name: str | None) -> str | None:
        if not tp_name:
            return None
        return tp_name.rsplit(".", 1)[-1]

    def _merge_discovered_functions(
            self,
            target: dict[str, list[ExtractedFunction]],
            functions: list[ExtractedFunction],
    ) -> None:
        for function in functions:
            target.setdefault(function.py_name, []).append(function)

    def _get_init_value(self, name: str, func_cursor: Cursor) -> str | None:
        """从局部变量定义中提取默认值字面表达式。"""
        for node in self._walk(func_cursor):
            if node.kind != CursorKind.VAR_DECL or node.spelling != name:
                continue
            tokens = [str(token.spelling) for token in node.get_tokens()]
            if "=" not in tokens:
                continue
            eq_idx = tokens.index("=")
            value = "".join(tokens[eq_idx + 1:]).strip()
            if value:
                return value
        return None

    def _find_format_string(self, func_cursor: Cursor, format_var_name: str) -> str | None:
        """回溯查找格式串变量对应的字符串字面量。"""
        for node in self._walk(func_cursor):
            if node.kind != CursorKind.VAR_DECL or node.spelling != format_var_name:
                continue
            for child in self._walk(node):
                if child.kind == CursorKind.STRING_LITERAL:
                    return self._strip_string_literal_quotes(str(child.spelling))
        return None

    def _walk(self, node: Cursor) -> Iterable[Cursor]:
        """生成器，深度优先遍历 cursor 子树。"""
        yield node
        for child in node.get_children():
            yield from self._walk(child)

    def _split_top_level(self, text: str, delim: str) -> list[str]:
        """
        仅在顶层分隔文本，忽略括号/引号内部的分隔符。

        该工具用于安全拆分形如 `a, dict[str, int], "x,y"` 的参数串。
        """
        closing = {"(": ")", "[": "]", "{": "}", "<": ">"}
        stack: list[str] = []
        parts: list[str] = []
        start = 0
        idx = 0
        while idx < len(text):
            char = text[idx]
            if char in "\"'":
                end = self._find_string_end(text, idx)
                if end is None:
                    break
                idx = end + 1
                continue
            if char in closing:
                stack.append(closing[char])
            elif stack and char == stack[-1]:
                stack.pop()
            elif not stack and char == delim:
                parts.append(text[start:idx])
                start = idx + 1
            idx += 1
        parts.append(text[start:])
        return parts

    def _find_string_end(self, text: str, start: int) -> int | None:
        """在包含转义字符的场景下查找字符串结束位置。"""
        quote = text[start]
        idx = start + 1
        while idx < len(text):
            if text[idx] == "\\":
                idx += 2
                continue
            if text[idx] == quote:
                return idx
            idx += 1
        return None

    def _normalize_parser_type(self, raw_type: str) -> str:
        """将 parser 文本中的类型描述归一化为 Python 基础类型名。"""
        normalized = raw_type.replace("const", "").replace("&", "").replace("*", "").strip()
        normalized = normalized.replace("std::", "")
        lower = normalized.lower()
        if any(token in lower for token in ("str", "string", "unicode")):
            return "str"
        if any(token in lower for token in ("bool",)):
            return "bool"
        if any(token in lower for token in ("double", "float")):
            return "float"
        if any(token in lower for token in ("int", "long", "size_t", "ssize")):
            return "int"
        return "object"

    def _normalize_c_type(self, raw_type: str) -> str:
        """将 C 类型拼写映射到简化 Python 类型。"""
        lower = raw_type.lower()
        if "char" in lower and "*" in lower:
            return "str"
        if "bool" in lower:
            return "bool"
        if "float" in lower or "double" in lower:
            return "float"
        if any(token in lower for token in ("int", "long", "short", "size_t", "ssize")):
            return "int"
        return "object"

    def _strip_string_literal_quotes(self, literal: str) -> str:
        """移除 C/C++ 字符串字面量前缀与外围引号。"""
        string_match = _CPP_STRING_LITERAL_RE.match(literal)
        if string_match is not None:
            return string_match.group(1)
        return literal.strip('"')

    def _looks_like_identifier(self, value: str) -> bool:
        """判断文本是否符合标识符命名规则。"""
        return bool(re.match(r"^[_A-Za-z]\w*$", value))

    def _unique_keep_order(self, values: list[str]) -> list[str]:
        """在保持原顺序的前提下去重。"""
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result

    def _signature_key(
            self,
            signature: ExtractedSignature,
    ) -> SignatureKey:
        """构建可哈希签名键，用于稳定去重。"""
        return (
            signature.return_type_name,
            tuple(
                (arg.name, arg.type_name, arg.default_value, arg.kind)
                for arg in signature.arguments
            ),
        )

    def _deduplicate_signatures(self, signatures: list[ExtractedSignature]) -> list[ExtractedSignature]:
        """按参数四元组键去重签名列表。"""
        seen: set[SignatureKey] = set()
        result: list[ExtractedSignature] = []
        for signature in signatures:
            key = self._signature_key(signature)
            if key in seen:
                continue
            seen.add(key)
            result.append(signature)
        return result

    def _deduplicate_result(self, raw: dict[str, list[ExtractedFunction]]) -> dict[str, list[ExtractedFunction]]:
        """
        对同名 Python 函数的候选提取结果做稳定去重。

        去重键包含 C 名、方法表、来源文件和完整签名，避免跨表误合并。
        """
        result: dict[str, list[ExtractedFunction]] = {}
        for py_name, funcs in raw.items():
            seen: set[FunctionDedupKey] = set()
            deduped: list[ExtractedFunction] = []
            for item in funcs:
                signature_key = tuple(self._signature_key(sig) for sig in item.signatures)
                # 同一 Python 名下，只有来源与签名都一致才视为重复。
                key = (item.c_name, item.method_table, item.source_file, signature_key)
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(item)
            result[py_name] = deduped
        return result
