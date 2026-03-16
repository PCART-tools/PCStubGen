from __future__ import annotations

import logging
import posixpath
import re
import sysconfig
from collections.abc import Iterable
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
    ExtractedFunction,
    ExtractedModule,
    ExtractedSignature,
)

logger = logging.getLogger(__name__)


AUTO_INCLUDE_MISSING_HEADER_RE = re.compile(r"'([^']+)' file not found")


SignatureArgumentKey: TypeAlias = tuple[str, str | None, str | None, str]
SignatureKey: TypeAlias = tuple[str | None, tuple[SignatureArgumentKey, ...]]

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

_CPP_STRING_LITERAL_RE = re.compile(r'^(?:u8|u|U|L)?"(.*)"$', re.DOTALL)

def _walk_cursor(node: Cursor) -> Iterable[Cursor]:
    """生成器，深度优先遍历 cursor 子树。"""
    yield node
    for child in node.get_children():
        yield from _walk_cursor(child)

def _unwrap_transparent(cursor: Cursor) -> Cursor:
    """剥离透明包装节点，定位到更有语义价值的底层表达式。"""
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
        """检查 token 流里是否直接出现 `NULL` 标识。"""
        return any(str(token.spelling) == "NULL" for token in node.get_tokens())

    def _is_semantic_zero(node: Cursor) -> bool:
        """统一识别 `0` / `NULL` / `nullptr` 这类“语义上为空”的写法。"""
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
    """格式化 translation unit 的诊断日志块，便于统一输出和排查。"""
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


def check(condition: bool, message: str="check failed!") -> None:
    """在核心前置条件不满足时抛出显式异常。"""
    if not condition:
        raise RuntimeError(message)


def _VAR_DECL_to_INIT_LIST_EXPR(cursor: Cursor) -> Cursor | None:
    """从变量声明直接找出其初始化列表节点。"""
    assert cursor.kind == CursorKind.VAR_DECL
    for child in cursor.get_children():
        if child.kind == CursorKind.INIT_LIST_EXPR:
            return child
    return None


def _strip_string_literal_quotes(literal: str) -> str:
    """移除 C/C++ 字符串字面量前缀与外围引号。"""
    string_match = _CPP_STRING_LITERAL_RE.match(literal)
    if string_match is not None:
        return string_match.group(1)
    return literal.strip('"')


class CSignatureExtractor:
    """
    基于 libclang 的 C 签名提取引擎。

    该引擎从 `PyModuleDef` 变量定义出发，读取 `m_name` / `m_methods`
    还原模块级 `PyMethodDef`，再结合 `PyArg_*` 调用和格式串规则推断
    Python 侧参数信息。
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
        """初始化提取器、clang 参数和模块级缓存。"""
        self._source_root = source_root
        self._clang_include = list(clang_include)
        self._clang_include_directory = list(clang_include_directory)
        self._clang_c_std = clang_c_std
        self._clang_cpp_std = clang_cpp_std
        self._cache_modules: dict[str, ExtractedModule] | None = None

    def extract_modules(self) -> dict[str, ExtractedModule]:
        """执行模块级签名提取主流程。"""
        if self._cache_modules is not None:
            return self._cache_modules

        check(self._source_root.exists())
        self._ensure_clang_ready()

        self._clang_include_directory = self._inject_python_include_directories(self._clang_include_directory)

        source_files = self._find_candidate_files()
        if not source_files:
            self._cache_modules = {}
            return self._cache_modules

        index = Index.create()

        translation_units: list[TranslationUnit] = []
        for file_path in source_files:
            tu = self._parse_translation_unit(index, file_path)
            translation_units.append(tu)

        discovered_modules: dict[str, ExtractedModule] = {}
        for tu in translation_units:
            try:
                modules = self._process_translation_unit(tu.cursor)
            except AssertionError as ex:
                logger.exception("AssertionError", exc_info=ex)
                continue
            for module in modules:
                existing = discovered_modules.get(module.name)
                if existing is None:
                    discovered_modules[module.name] = module
                    continue
                self._merge_extracted_module(existing, module)

        self._cache_modules = discovered_modules
        return self._cache_modules

    def _ensure_clang_ready(self) -> None:
        """确保 clang 运行环境可用。"""
        if not clang.cindex.Config.loaded:
            packaged_libclang_path = _get_packaged_libclang_path()
            if packaged_libclang_path:
                clang.cindex.Config.set_library_file(packaged_libclang_path)

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
        """规范化报错里的头文件字面量，便于后续路径匹配。"""
        normalized = include_literal.replace("\\", "/").strip()
        if not normalized:
            return ""
        normalized = posixpath.normpath(normalized)
        if normalized == ".":
            return ""
        return normalized

    def _split_include_literal_parts(self, include_literal: str) -> tuple[str, ...]:
        """将 include 字面量拆成稳定的路径片段元组。"""
        normalized = self._normalize_include_literal(include_literal)
        return tuple(part for part in normalized.split("/") if part and part != ".")

    def _extract_missing_include_literals(self, diagnostics: list[Diagnostic]) -> list[str]:
        """从 clang 错误诊断中提取缺失头文件名。"""
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

    def _resolve_missing_include_dir(self, *, include_literal: str) -> Path | None:
        """在源码树内搜索缺失头文件，找到首个匹配的 include 目录后立即返回。"""
        include_literal = self._normalize_include_literal(include_literal)
        include_parts = self._split_include_literal_parts(include_literal)
        if not include_parts:
            return None

        if not self._source_root.exists():
            return None

        for header_path in self._source_root.rglob(include_literal):
            if not header_path.is_file():
                continue
            include_root_depth = len(include_parts) - 1
            if include_root_depth >= len(header_path.parents):
                continue
            return header_path.parents[include_root_depth]
        return None

    def _append_include_args(self, include_args: Iterable[str]) -> list[str]:
        """将新发现的 include 目录追加到 clang 参数中，并返回实际新增项。"""
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
        """基于缺失头文件诊断自动补全 clang include 目录。"""
        resolved_pairs: list[tuple[str, str]] = []
        missing_literals = self._extract_missing_include_literals(diagnostics)
        for include_literal in missing_literals:
            include_dir = self._resolve_missing_include_dir(include_literal=include_literal)
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
        """查找包含 `PyModuleDef` 定义线索的 C/C++ 源文件。"""
        result: list[Path] = []
        for path in self._source_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in NATIVE_SOURCE_SUFFIXES:
                text = path.read_text(encoding="utf-8", errors="ignore")
                if "PyModuleDef" in text:
                    result.append(path)
        result.sort()
        return result

    def _get_std_value_for_file(self, file_path: Path) -> str:
        """按后缀为源码文件选择 C 或 C++ 标准值。"""
        suffix = file_path.suffix.lower()
        if suffix in CPP_SOURCE_SUFFIXES:
            return self._clang_cpp_std
        return self._clang_c_std

    def _build_clang_parse_args(self, file_path: Path) -> list[str]:
        """为指定源码文件构建 clang 解析参数列表。"""
        parse_args = []
        std_value = self._get_std_value_for_file(file_path)
        parse_args.extend(["--std", std_value])
        for include_value in self._clang_include:
            parse_args.extend(["--include", include_value])
        for include_dir in self._clang_include_directory:
            parse_args.extend(["--include-directory", include_dir])
        return parse_args

    def _parse_translation_unit(self, index: Index, file_path: Path) -> TranslationUnit:
        """解析单个源码文件为 clang translation unit。"""
        translation_unit: TranslationUnit | None = None
        diagnostics: list[Diagnostic] = []
        for _ in range(10):
            parse_args = self._build_clang_parse_args(file_path)
            translation_unit = index.parse(str(file_path), args=parse_args)
            diagnostics = translation_unit.diagnostics
            added = self._discover_missing_include_args(
                file_path=file_path,
                diagnostics=diagnostics,
            )
            if not added:
                break

        if self._has_error_diagnostics(diagnostics):
            logger.warning(
                _format_diagnostics_message(
                    file_path=file_path,
                    parse_args=parse_args,
                    diagnostics=diagnostics,
                )
            )
        check(translation_unit is not None, f"Failed to parse translation unit: {file_path}")
        return translation_unit

    def _has_error_diagnostics(self, diagnostics: list[Diagnostic]) -> bool:
        """判断 diagnostics 中是否包含 Error/Fatal 级别。"""
        for diagnostic in diagnostics:
            if diagnostic.severity >= clang.cindex.Diagnostic.Error:
                return True
        return False


    def _process_translation_unit(
            self,
            cursor: Cursor,
    ) -> list[ExtractedModule]:
        """从单个 translation unit 的 `PyModuleDef` 变量定义提取模块。"""
        discovered: list[ExtractedModule] = []
        # PyModuleDef可能定义为PyInit_函数内的局部变量
        for node in _walk_cursor(cursor):
            if node.kind != CursorKind.VAR_DECL:
                continue
            if not node.is_definition():
                continue
            if node.type.spelling not in {"PyModuleDef", "struct PyModuleDef"}:
                continue
            extracted = self._extract_module_from_PyModuleDef(node)
            if extracted is not None:
                discovered.append(extracted)
        return discovered

    def _extract_module_from_PyModuleDef(
            self,
            module_def_cursor: Cursor,
    ) -> ExtractedModule | None:
        """
        从单个 `PyModuleDef` 变量中提取模块定义与模块方法。

        模块名只认 `m_name`，方法只认 `m_methods`；若缺少 `m_name`，
        直接忽略该定义，避免把局部模板或未导出的占位结构误当成模块。
        """
        init_list_expr = _VAR_DECL_to_INIT_LIST_EXPR(module_def_cursor)
        assert init_list_expr is not None

        PyModuleDef_field_names = (
            "m_base",
            "m_name",
            "m_doc",
            "m_size",
            "m_methods",
            "m_slots",
            "m_traverse",
            "m_clear",
            "m_free",
        )
        field_values = self._resolve_INIT_LIST_EXPR(init_list_expr, PyModuleDef_field_names)

        # 模块名
        m_name_cursor = field_values.get("m_name")
        if m_name_cursor is None:
            return None
        assert m_name_cursor.kind == CursorKind.STRING_LITERAL

        m_name = m_name_cursor.spelling.strip('"')

        module = ExtractedModule(name=m_name)
        module.lookup_names.update(self._build_module_lookup_names(m_name))

        # 方法表
        m_methods_cursor = field_values.get("m_methods")
        if m_methods_cursor is None:
            return module
        if m_methods_cursor.kind == CursorKind.DECL_REF_EXPR:
            method_list_cursor = m_methods_cursor.referenced
            assert _is_PyMethodDef_array_definition(method_list_cursor)
            self._merge_discovered_functions(
                module.functions,
                self._extract_method_table(method_list_cursor, module_name=m_name).values(),
                module_name=m_name,
            )
        else:
            assert (
                m_methods_cursor.kind in cpp_nullptr_literal_cursor_kinds
                or _is_integer_literal_value(m_methods_cursor, 0)
                or any(str(token.spelling) == "NULL" for token in m_methods_cursor.get_tokens())
            )

        return module

    def _merge_extracted_module(self, target: ExtractedModule, incoming: ExtractedModule) -> None:
        """合并两个同名模块的提取结果，保留已有信息并追加新发现。"""
        target.lookup_names.update(incoming.lookup_names)
        self._merge_discovered_functions(
            target.functions,
            incoming.functions.values(),
            module_name=target.name,
        )

    def _extract_method_table(
            self,
            cursor: Cursor,
            *,
            module_name: str,
    ) -> dict[str, ExtractedFunction]:
        """解析 `PyMethodDef[]` 变量。"""
        assert cursor.kind == CursorKind.VAR_DECL
        assert _is_PyMethodDef_array_definition(cursor)

        grouped: dict[str, ExtractedFunction] = {}
        # 方法表本质上是数组初始化列表，逐项还原即可。
        init_expr_node = self._array_VAR_DECL_to_INIT_LIST_EXPR(cursor)
        self._process_PyMethodDef_array_INIT_LIST_EXPR(
            cursor,
            init_expr_node,
            grouped,
            module_name=module_name,
        )
        return grouped

    def _resolve_INIT_LIST_EXPR(
            self,
            cursor: Cursor,
            field_names: tuple[str, ...] | list[str],
    ) -> dict[str, Cursor]:
        """解析顶层初始化列表，支持位置初始化与 designated initializer 混用。"""
        assert cursor.kind == CursorKind.INIT_LIST_EXPR

        field_name_to_index = {
            field_name: index
            for index, field_name in enumerate(field_names)
        }
        field_values: dict[str, Cursor] = {}
        positional_index = 0

        for entry in cursor.get_children():
            entry_children = list(entry.get_children())

            if len(entry_children) >= 2 and entry_children[0].kind == CursorKind.MEMBER_REF:
                # 指定初始化
                field_name = entry_children[0].spelling
                value_cursor = _unwrap_transparent(entry_children[1])
                designated_index = field_name_to_index.get(field_name)
                if designated_index is None:
                    continue
                positional_index = designated_index + 1
            else:
                # 位置初始化
                if positional_index >= len(field_names):
                    continue
                field_name = field_names[positional_index]
                value_cursor = _unwrap_transparent(entry)
                positional_index += 1

            field_values[field_name] = value_cursor

        return field_values

    def _array_VAR_DECL_to_INIT_LIST_EXPR(self, cursor: Cursor) -> Cursor:
        """断言变量声明是数组初始化，并返回对应 `INIT_LIST_EXPR`。"""
        assert cursor.kind == CursorKind.VAR_DECL
        init_list_expr = _VAR_DECL_to_INIT_LIST_EXPR(cursor)
        assert init_list_expr is not None
        assert init_list_expr.kind == CursorKind.INIT_LIST_EXPR
        return init_list_expr

    def _process_PyMethodDef_array_INIT_LIST_EXPR(
            self,
            var_decl_node: Cursor,
            init_list_expr_node: Cursor,
            output: dict[str, ExtractedFunction],
            *,
            module_name: str,
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
            self._add_discovered_function(output, extracted, module_name=module_name)

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
        ml_name = _strip_string_literal_quotes(ml_name_cursor.spelling)

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
        for node in _walk_cursor(field_cursor):
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
        for node in _walk_cursor(func_cursor):
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

        for node in _walk_cursor(func_cursor):
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
        """
        根据返回调用名推断 Python 返回类型。

        优先处理 `Py_BuildValue`、`Py_NewRef` 这类需要额外上下文的调用，
        其余再按前缀表或兜底规则映射。
        """
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
            return _strip_string_literal_quotes(token)
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
        for child in _walk_cursor(node):
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
            text = _strip_string_literal_quotes(str(token.spelling))
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
                # `*` / `$` 之后的参数都视为 keyword-only。
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
            text = _strip_string_literal_quotes(token)
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

    def _get_cursor_source_file(self, cursor: Cursor) -> str | None:
        """读取 cursor 所在源文件路径；无位置信息时返回 `None`。"""
        location = cursor.location
        if location is None:
            return None
        file_obj = location.file
        if file_obj is None:
            return None
        return str(file_obj)

    def _build_module_lookup_names(self, module_name: str) -> set[str]:
        """构建模块别名集合，仅保留完整名与叶子名。"""
        lookup_names = {module_name}
        leaf_name = module_name.rsplit(".", 1)[-1]
        lookup_names.add(leaf_name)
        return lookup_names

    def _merge_discovered_functions(
            self,
            target: dict[str, ExtractedFunction],
            functions: Iterable[ExtractedFunction],
            *,
            module_name: str,
    ) -> None:
        """把函数按 Python 名聚合到目标映射中，重复名时保留旧值。"""
        for function in functions:
            self._add_discovered_function(target, function, module_name=module_name)

    def _add_discovered_function(
            self,
            target: dict[str, ExtractedFunction],
            function: ExtractedFunction,
            *,
            module_name: str,
    ) -> None:
        """向模块函数映射写入函数；若 Python 名重复则告警并保留旧值。"""
        existing = target.get(function.py_name)
        if existing is None:
            target[function.py_name] = function
            return
        logger.warning(
            "Discarded duplicate extracted function in module %s for Python name %s: "
            "kept c_name=%s%s, dropped c_name=%s%s",
            module_name,
            function.py_name,
            existing.c_name,
            self._format_function_origin(existing),
            function.c_name,
            self._format_function_origin(function),
        )

    def _format_function_origin(self, function: ExtractedFunction) -> str:
        """格式化函数来源上下文，便于重复名告警定位。"""
        details: list[str] = []
        if function.source_file is not None:
            details.append(f"source_file={function.source_file}")
        if function.method_table is not None:
            details.append(f"method_table={function.method_table}")
        if not details:
            return ""
        return f" ({', '.join(details)})"

    def _get_init_value(self, name: str, func_cursor: Cursor) -> str | None:
        """从局部变量定义中提取默认值字面表达式。"""
        for node in _walk_cursor(func_cursor):
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
        for node in _walk_cursor(func_cursor):
            if node.kind != CursorKind.VAR_DECL or node.spelling != format_var_name:
                continue
            for child in _walk_cursor(node):
                if child.kind == CursorKind.STRING_LITERAL:
                    return _strip_string_literal_quotes(str(child.spelling))
        return None

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
        """构建签名的可哈希键，供模块/类方法去重复用。"""
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

