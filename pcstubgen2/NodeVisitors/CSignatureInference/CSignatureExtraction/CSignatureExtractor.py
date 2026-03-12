from __future__ import annotations

import logging
import os
import re
import sysconfig
from pathlib import Path
from typing import Iterable, TypeAlias

import clang
import clang.cindex
from clang.cindex import (
    Cursor,
    CursorKind,
    Diagnostic,
    Index,
    Token,
    TokenKind,
    TranslationUnit,
    Type,
    TypeKind
)

from .Constants import (
    CPP_SOURCE_SUFFIXES,
    FORMAT_TYPE_MAP,
    METH_TYPE_LITERAL_MAP,
    NATIVE_SOURCE_SUFFIXES,
    POINTER_CAST_IDENTIFIER_SKIP,
    PY_BUILDVALUE_SINGLE_MARKER_TYPE_MAP,
    RETURN_CALL_PREFIX_TYPE_MAP,
    RETURN_MACRO_TYPE_MAP,
    RETURN_TOKEN_TYPE_MAP,
    UNRELATED_TOKENS,
)
from .Models import ExtractedArgument, ExtractedFunction, ExtractedSignature

logger = logging.getLogger(__name__)

DEFAULT_CLANG_C_STD = "c11"
DEFAULT_CLANG_CPP_STD = "c++17"

SignatureArgumentKey: TypeAlias = tuple[str, str | None, str | None, str]
SignatureKey: TypeAlias = tuple[str | None, tuple[SignatureArgumentKey, ...]]
FunctionDedupKey: TypeAlias = tuple[str, str | None, str | None, tuple[SignatureKey, ...]]


def _is_PyMethodDef_array_sentinel(element: Cursor) -> bool:
    """判断 `PyMethodDef` 条目是否为数组结束哨兵。只要ml_name语义上为0就判定为哨兵"""
    transparent_kinds = {
        CursorKind.UNEXPOSED_EXPR,  # clang 额外包裹层，常见于宏展开或隐式转换外壳
        CursorKind.PAREN_EXPR,  # (expr)
        CursorKind.CSTYLE_CAST_EXPR,  # (T)expr
        CursorKind.CXX_STATIC_CAST_EXPR,  # static_cast<T>(expr)
        CursorKind.CXX_REINTERPRET_CAST_EXPR,  # reinterpret_cast<T>(expr)
        CursorKind.CXX_CONST_CAST_EXPR,  # const_cast<T>(expr)
        CursorKind.CXX_FUNCTIONAL_CAST_EXPR,  # T(expr)
    }

    cpp_nullptr_literal_kinds = {
        CursorKind.CXX_NULL_PTR_LITERAL_EXPR,  # nullptr
        CursorKind.GNU_NULL_EXPR,  # GNU 扩展 __null
    }

    def _unwrap_transparent(node: Cursor) -> Cursor:
        current = node
        while current.kind in transparent_kinds:
            children = list(current.get_children())
            if len(children) != 1:
                break
            current = children[0]
        return current

    def _is_zero_integer_literal(node: Cursor) -> bool:
        """
        是0整数字面量，NULL如果展开为((void*)0)也会走这里
        """
        if node.kind != CursorKind.INTEGER_LITERAL:
            return False
        for token in node.get_tokens():
            if token.kind != TokenKind.LITERAL:
                continue
            text = str(token.spelling).replace("'", "").rstrip("uUlLzZ")
            try:
                return int(text, 0) == 0
            except ValueError:
                return False
        return False

    def _is_null_token(node: Cursor) -> bool:
        return any(str(token.spelling) == "NULL" for token in node.get_tokens())

    def _is_semantic_zero(node: Cursor) -> bool:
        target = _unwrap_transparent(node)
        if target.kind in cpp_nullptr_literal_kinds:
            return True
        if _is_zero_integer_literal(target):
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


def _format_parse_exception_message(
        *,
    file_path: Path,
    parse_args: list[str],
    error: Exception,
) -> str:
    """格式化 translation unit 解析异常日志。"""
    return "\n".join(
        [
            f"Failed to parse translation unit",
            f"  file_path: {file_path}",
            f"  suffix: {file_path.suffix.lower() or '<none>'}",
            f"  parse_args: {parse_args!r}",
            f"  exception_type: {type(error).__name__}",
            f"  exception: {error}",
        ]
    )


def _get_packaged_libclang_path() -> str | None:
    """从 `clang` 包的 `native` 目录探测可用的 `libclang` 动态库。"""
    native_dir = Path(clang.__file__).resolve().parent / "native"
    for filename in ("libclang.dll", "libclang.so", "libclang.dylib"):
        candidate = native_dir / filename
        if candidate.exists():
            return str(candidate)
    return None


def _is_array_kind(kind: TypeKind) -> bool:
    """判断 clang 类型是否为 C/C++ 数组类型。"""
    return (
        kind == TypeKind.CONSTANTARRAY  # 固定长度数组，如 `int values[8]`
        or kind == TypeKind.INCOMPLETEARRAY  # 不完整数组，如 `extern int values[]`
        or kind == TypeKind.VARIABLEARRAY  # 变长数组，如 `int values[n]`
        or kind == TypeKind.DEPENDENTSIZEDARRAY  # 依赖表达式推导长度的数组，多见于模板/泛型上下文
    )


def _is_PyMethodDef_array_definition(cursor: Cursor) -> bool:
    """判断节点是否为 `PyMethodDef[]`。"""
    if _is_array_kind(cursor.type.kind) and cursor.is_definition():
        elem_type = cursor.type.get_array_element_type()
        if elem_type.get_canonical().spelling == "struct PyMethodDef":
            return True
    return False


def _is_initializer_list_PyMethodDef(cursor: Cursor) -> bool:
    """判断变量是否是 `initializer_list<PyMethodDef>` 风格定义。"""
    try:
        type_obj = cursor.type
        candidate_types = [type_obj]
        canonical_type = type_obj.get_canonical()
        if canonical_type is not None:
            candidate_types.append(canonical_type)
        for candidate_type in candidate_types:
            spelling = candidate_type.spelling or ""
            if "initializer_list" in spelling and "PyMethodDef" in spelling:
                return True
    except Exception:
        return False
    return False


class CSignatureExtractor:
    """
    基于 libclang 的 C 签名提取引擎。

    该引擎会扫描源码中的 `PyMethodDef` 表，定位对应 C 函数，
    再结合 `PyArg_*` 调用和格式串规则推断 Python 侧参数信息。
    """

    def __init__(
        self,
        source_root: Path,
        *,
        clang_parse_args: Iterable[str] = (),
        clang_c_std: str | None = None,
        clang_cpp_std: str | None = None,
    ) -> None:
        """初始化提取器并准备惰性缓存。"""
        self.source_root = source_root
        self._clang_parse_args = list(clang_parse_args)
        self._clang_c_std = clang_c_std
        self._clang_cpp_std = clang_cpp_std
        self._cache_result: dict[str, list[ExtractedFunction]] | None = None

    def extract(self) -> dict[str, list[ExtractedFunction]]:
        """
        执行签名提取主流程。

        返回值按 Python 函数名聚合候选提取结果；失败时降级为 `{}`，
        并缓存结果避免重复解析同一源码树。
        """
        if self._cache_result is not None:
            return self._cache_result

        if not self.source_root.exists():
            logger.warning("c_source_root does not exist: %s", self.source_root)
            self._cache_result = {}
            return self._cache_result

        if not self._ensure_clang_ready():
            self._cache_result = {}
            return self._cache_result

        self._prepare_extract_parse_args()

        source_files = self._find_candidate_files()
        if not source_files:
            self._cache_result = {}
            return self._cache_result

        index = Index.create()

        translation_units: list[TranslationUnit] = []
        for file_path in source_files:
            tu = self._parse_translation_unit(index=index, file_path=file_path)
            if tu is None:
                continue
            translation_units.append(tu)

        # 第一阶段先收集函数定义，供方法表条目回查 C 函数体。
        function_defs: dict[str, list[Cursor]] = {}
        for tu in translation_units:
            self._collect_function_definitions(tu.cursor, function_defs)

        # 第二阶段处理 PyMethodDef，拼装提取结果。
        result: dict[str, list[ExtractedFunction]] = {}
        for tu in translation_units:
            self._collect_pymethod_defs(tu.cursor, function_defs, result)

        self._cache_result = self._deduplicate_result(result)
        return self._cache_result

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

    def _prepare_extract_parse_args(self) -> None:
        """在提取阶段补齐解析所需的额外参数。"""
        self._clang_parse_args = self._inject_python_include_args(self._clang_parse_args)

    def _inject_python_include_args(self, parse_args: list[str]) -> list[str]:
        """向 clang 参数注入当前 Python 头文件目录。"""
        args = list(parse_args)
        include_candidates = [
            sysconfig.get_path("include"),
            sysconfig.get_path("platinclude"),
        ]
        for include_dir in include_candidates:
            if not include_dir:
                continue
            include_arg = f"-I{include_dir}"
            if include_arg not in args:
                args.append(include_arg)
        return args

    def _find_candidate_files(self) -> list[Path]:
        """查找可能包含 `PyMethodDef` 的 C/C++ 源文件。"""
        result: list[Path] = []
        for path in self.source_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in NATIVE_SOURCE_SUFFIXES:
                text = path.read_text(encoding="utf-8", errors="ignore")
                if "PyMethodDef" in text:
                    result.append(path)
        result.sort()
        return result

    def _parse_translation_unit(self, index: Index, file_path: Path) -> TranslationUnit | None:
        """解析单个源码文件为 clang translation unit。"""
        parse_args = self._build_parse_args(file_path)
        try:
            translation_unit = index.parse(str(file_path), args=parse_args)
        except Exception as ex:  # pragma: no cover
            logger.warning(_format_parse_exception_message(file_path=file_path, parse_args=parse_args, error=ex))
            return None
        diagnostics: list[Diagnostic] = list(translation_unit.diagnostics)
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

    def _build_parse_args(self, file_path: Path) -> list[str]:
        """为单个源码文件拼装 clang 参数。"""
        parse_args = list(self._clang_parse_args)
        std_arg = self._build_std_arg_for_file(file_path)
        if std_arg is not None:
            parse_args.insert(0, std_arg)
        return parse_args

    def _build_std_arg_for_file(self, file_path: Path) -> str | None:
        """按后缀为源码文件选择 C 或 C++ 标准参数。"""
        suffix = file_path.suffix.lower()
        if suffix in CPP_SOURCE_SUFFIXES:
            return self._normalize_std_arg(self._clang_cpp_std or DEFAULT_CLANG_CPP_STD)
        return self._normalize_std_arg(self._clang_c_std or DEFAULT_CLANG_C_STD)

    def _normalize_std_arg(self, std_value: str | None) -> str | None:
        """将标准配置统一为 `-std=` 参数。"""
        if std_value is None:
            return None
        normalized = std_value.strip()
        if not normalized:
            return None
        if normalized.startswith("-std="):
            return normalized
        return f"-std={normalized}"

    def _collect_function_definitions(self, cursor: Cursor, output: dict[str, list[Cursor]]) -> None:
        """遍历 AST，按函数名收集 `FUNCTION_DECL` 节点。"""
        for node in self._walk(cursor):
            if node.kind == CursorKind.FUNCTION_DECL and node.spelling:
                output.setdefault(node.spelling, []).append(node)

    def _collect_pymethod_defs(
        self,
        cursor: Cursor,
        function_defs: dict[str, list[Cursor]],
        output: dict[str, list[ExtractedFunction]],
    ) -> None:
        """在 AST 中定位 `PyMethodDef` 表并提取条目。"""
        for node in self._walk(cursor):
            if node.kind == CursorKind.VAR_DECL:
                if _is_PyMethodDef_array_definition(node):
                    init_expr_node = self._array_VAR_DECL_to_INIT_LIST_EXPR(node)
                    self._process_PyMethodDef_array_INIT_LIST_EXPR(
                        node,
                        init_expr_node,
                        function_defs,
                        output,
                    )

    @staticmethod
    def _array_VAR_DECL_to_INIT_LIST_EXPR(cursor: Cursor) -> Cursor:
        # VAR_DECL
        #   TYPE_REF
        #   INIT_LIST_EXPR
        it = iter(cursor.get_children())
        _ = next(it)
        init_list_expr = next(it)
        return init_list_expr

    def _process_PyMethodDef_array_INIT_LIST_EXPR(
        self,
        var_decl_node: Cursor,
        init_expr_node: Cursor,
        function_defs: dict[str, list[Cursor]],
        output: dict[str, list[ExtractedFunction]],
    ) -> None:
        """处理单个方法表的 `INIT_LIST_EXPR` 并写入输出。"""
        table_name = var_decl_node.spelling
        location = var_decl_node.location
        source_file = str(location.file)
        for element in init_expr_node.get_children():
            if _is_PyMethodDef_array_sentinel(element):
                break
            extracted = self._extract_PyMethodDef_struct_fields(
                struct_init=element,
                method_table=table_name,
                source_file=source_file,
                function_defs=function_defs,
            )
            if extracted is None:
                continue
            output.setdefault(extracted.py_name, []).append(extracted)

    def _extract_PyMethodDef_struct_fields(
        self,
        struct_init: Cursor,
        method_table: str,
        source_file: str | None,
        function_defs: dict[str, list[Cursor]],
    ) -> ExtractedFunction | None:
        """
        从 `PyMethodDef` 的单个初始化项提取函数元数据和签名。

        若关键字段（Python 名、C 函数名）缺失则返回 `None`，
        保持提取过程对异常样本的容错性。
        """
        tokens = list(struct_init.get_tokens())
        if not tokens:
            return None

        ml_name: str | None = None
        ml_flags: list[str] = []
        for token in tokens:
            spelling = str(token.spelling)
            if token.kind == TokenKind.LITERAL:
                # 第一个字符串字面量通常是 Python 暴露名。
                if ml_name is None and '"' in spelling:
                    lit = self._strip_literal_quotes(spelling)
                    if lit and lit != "NULL":
                        ml_name = lit
                ml_flags.extend(self._decode_meth_literal_flags(spelling))
                continue
            if token.kind == TokenKind.IDENTIFIER and spelling.startswith("METH_"):
                ml_flags.append(spelling)

        if not ml_name:
            return None
        ml_flags = self._unique_keep_order(ml_flags)

        c_name = self._find_c_function_name(tokens)
        if not c_name:
            return None

        function_cursor = self._select_function_cursor(
            function_defs.get(c_name, []),
            preferred_file=source_file,
        )
        signatures: list[ExtractedSignature] = []
        return_type_name: str | None = None
        if function_cursor is not None:
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
            c_name=c_name,
            method_flags=ml_flags,
            signatures=signatures,
            source_file=source_file,
            method_table=method_table,
        )

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
        for token in tokens[call_idx + 1 :]:
            format_text = self._resolve_buildvalue_format_token(func_cursor=func_cursor, token=token)
            if format_text is None:
                continue
            return self._infer_return_type_from_buildvalue_format(format_text)
        return "object"

    def _resolve_buildvalue_format_token(self, *, func_cursor: Cursor, token: str) -> str | None:
        """解析 `Py_BuildValue` 的格式串 token。"""
        if '"' in token:
            return self._strip_literal_quotes(token)
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
            text = self._strip_literal_quotes(str(token.spelling))
            if "(" in text and ")" in text:
                literals.append(text)
        for text in literals:
            args_part = text[text.find("(") + 1 : text.rfind(")")]
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
            text = self._strip_literal_quotes(token)
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

    def _get_init_value(self, name: str, func_cursor: Cursor) -> str | None:
        """从局部变量定义中提取默认值字面表达式。"""
        for node in self._walk(func_cursor):
            if node.kind != CursorKind.VAR_DECL or node.spelling != name:
                continue
            tokens = [str(token.spelling) for token in node.get_tokens()]
            if "=" not in tokens:
                continue
            eq_idx = tokens.index("=")
            value = "".join(tokens[eq_idx + 1 :]).strip()
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
                    return self._strip_literal_quotes(str(child.spelling))
        return None

    def _find_c_function_name(self, tokens: list[Token]) -> str | None:
        """从方法表初始化 token 中提取 C 函数符号名。"""
        literal_found = False
        for token in tokens:
            spelling = str(token.spelling)
            if token.kind == TokenKind.LITERAL and not literal_found:
                literal_found = True
                continue
            if not literal_found:
                continue
            if token.kind != TokenKind.IDENTIFIER:
                continue
            if spelling.startswith("METH_"):
                continue
            if spelling in POINTER_CAST_IDENTIFIER_SKIP:
                # 过滤函数指针 cast 与空指针标识，避免误识别为函数名。
                continue
            return spelling
        return None

    def _select_function_cursor(
        self,
        candidates: list[Cursor],
        *,
        preferred_file: str | None = None,
    ) -> Cursor | None:
        """优先返回同源文件中的函数定义；若无定义则退化到首个声明。"""
        if not candidates:
            return None

        selected = list(candidates)
        preferred_key = self._normalize_file_key(preferred_file)
        if preferred_key is not None:
            same_file_candidates: list[Cursor] = []
            for candidate in candidates:
                candidate_file = candidate.location.file
                if self._normalize_file_key(candidate_file) == preferred_key:
                    same_file_candidates.append(candidate)
            if same_file_candidates:
                selected = same_file_candidates

        for candidate in selected:
            try:
                if candidate.is_definition():
                    return candidate
            except Exception:
                continue
        return selected[0]

    def _normalize_file_key(self, value: object | None) -> str | None:
        """将 clang 文件位置标准化为可比较键值。"""
        if value is None:
            return None
        raw = str(value).strip()
        if not raw:
            return None
        return os.path.normcase(os.path.normpath(raw))

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

    def _strip_literal_quotes(self, literal: str) -> str:
        """移除 C/C++ 字符串字面量前缀与外围引号。"""
        stripped = literal
        while stripped and stripped[0] in {"u", "U", "L", "R"} and '"' in stripped:
            stripped = stripped[1:]
        if stripped.startswith('"') and stripped.endswith('"') and len(stripped) >= 2:
            return stripped[1:-1]
        return stripped.strip('"')

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


