from __future__ import annotations

import logging
import re
import sysconfig
from pathlib import Path
from typing import Any, Iterable

from .Constants import (
    C_SOURCE_SUFFIXES,
    FORMAT_TYPE_MAP,
    METH_TYPE_LITERAL_MAP,
    POINTER_CAST_IDENTIFIER_SKIP,
    PY_BUILDVALUE_SINGLE_MARKER_TYPE_MAP,
    RETURN_CALL_PREFIX_TYPE_MAP,
    RETURN_MACRO_TYPE_MAP,
    RETURN_TOKEN_TYPE_MAP,
    UNRELATED_TOKENS,
)
from .Models import ExtractedArgument, ExtractedFunction, ExtractedSignature

logger = logging.getLogger("pybind11_stubgen")


class CSignatureExtractionEngine:
    """
    基于 libclang 的 C 签名提取引擎。

    该引擎会扫描源码中的 `PyMethodDef` 表，定位对应 C 函数，
    再结合 `PyArg_*` 调用和格式串规则推断 Python 侧参数信息。
    """

    def __init__(
        self,
        source_root: str | Path,
        *,
        clang_library_path: str | None = None,
        clang_parse_args: Iterable[str] | None = None,
    ) -> None:
        """初始化提取器并准备惰性缓存。"""
        self.source_root = Path(source_root)
        self._clang_library_path = clang_library_path
        self._clang_parse_args = list(clang_parse_args) if clang_parse_args is not None else None
        self._clang: Any | None = None
        self._result_cache: dict[str, list[ExtractedFunction]] | None = None

    def extract(self) -> dict[str, list[ExtractedFunction]]:
        """
        执行签名提取主流程。

        返回值按 Python 函数名聚合候选提取结果；失败时降级为 `{}`，
        并缓存结果避免重复解析同一源码树。
        """
        if self._result_cache is not None:
            return self._result_cache

        if not self.source_root.exists():
            logger.warning("c_source_root does not exist: %s", self.source_root)
            self._result_cache = {}
            return self._result_cache

        if not self._ensure_clang_ready():
            self._result_cache = {}
            return self._result_cache

        source_files = self._find_candidate_files()
        if not source_files:
            self._result_cache = {}
            return self._result_cache

        index = self._clang.Index.create()
        translation_units: list[Any] = []
        function_defs: dict[str, list[Any]] = {}

        for file_path in source_files:
            tu = self._parse_translation_unit(index=index, file_path=file_path)
            if tu is None:
                continue
            translation_units.append(tu)
            # 第一阶段先收集函数定义，供方法表条目回查 C 函数体。
            self._collect_function_definitions(tu.cursor, function_defs)

        result: dict[str, list[ExtractedFunction]] = {}
        for tu in translation_units:
            # 第二阶段处理 PyMethodDef，拼装提取结果。
            self._collect_pymethod_defs(tu.cursor, function_defs, result)

        self._result_cache = self._deduplicate_result(result)
        return self._result_cache

    def _ensure_clang_ready(self) -> bool:
        """确保 clang 运行环境可用，并补齐解析配置。"""
        if self._clang is None:
            try:
                import clang.cindex as clang_cindex
            except Exception as ex:  # pragma: no cover
                logger.warning("clang.cindex is unavailable, skip C extraction: %s", ex)
                return False
            self._clang = clang_cindex

        if self._clang_parse_args is None:
            self._clang_parse_args = ["-std=c11"]
        self._clang_parse_args = self._inject_python_include_args(self._clang_parse_args)

        if self._clang_library_path:
            try:
                loaded = bool(getattr(self._clang.Config, "loaded", False))
                if not loaded:
                    self._clang.Config.set_library_file(self._clang_library_path)
            except Exception as ex:  # pragma: no cover
                logger.warning("Failed to configure libclang '%s': %s", self._clang_library_path, ex)
        return True

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
            if not path.is_file():
                continue
            if path.suffix.lower() not in C_SOURCE_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if "PyMethodDef" in text:
                result.append(path)
        result.sort()
        return result

    def _parse_translation_unit(self, index: Any, file_path: Path) -> Any | None:
        """解析单个源码文件为 clang translation unit。"""
        try:
            return index.parse(str(file_path), args=self._clang_parse_args or [])
        except Exception as ex:  # pragma: no cover
            logger.warning("Failed to parse '%s': %s", file_path, ex)
            return None

    def _collect_function_definitions(self, cursor: Any, output: dict[str, list[Any]]) -> None:
        """遍历 AST，按函数名收集 `FUNCTION_DECL` 节点。"""
        func_kind = self._clang.CursorKind.FUNCTION_DECL
        for node in self._walk(cursor):
            if node.kind != func_kind or not node.spelling:
                continue
            output.setdefault(node.spelling, []).append(node)

    def _collect_pymethod_defs(
        self,
        cursor: Any,
        function_defs: dict[str, list[Any]],
        output: dict[str, list[ExtractedFunction]],
    ) -> None:
        """在 AST 中定位 `PyMethodDef` 表并提取条目。"""
        var_decl = self._clang.CursorKind.VAR_DECL
        for node in self._walk(cursor):
            if node.kind != var_decl:
                continue
            if self._is_pymethod_array(node):
                self._process_array(node, node, function_defs, output)
                continue
            if self._is_initializer_list(node):
                init_node = self._find_initializer_list_node(node)
                if init_node is not None:
                    self._process_array(init_node, node, function_defs, output)

    def _process_array(
        self,
        array_node: Any,
        owner_node: Any,
        function_defs: dict[str, list[Any]],
        output: dict[str, list[ExtractedFunction]],
    ) -> None:
        """处理单个方法表数组节点并写入输出。"""
        table_name = owner_node.spelling if owner_node.spelling else "<anonymous>"
        source_file = str(owner_node.location.file) if owner_node.location.file else None
        for element in self._iter_array_elements(array_node):
            extracted = self._extract_struct_fields(
                struct_init=element,
                method_table=table_name,
                source_file=source_file,
                function_defs=function_defs,
            )
            if extracted is None:
                continue
            output.setdefault(extracted.py_name, []).append(extracted)

    def _iter_array_elements(self, array_node: Any) -> Iterable[Any]:
        """迭代方法表数组元素，遇到终止哨兵即停止。"""
        init_kind = self._clang.CursorKind.INIT_LIST_EXPR
        init_nodes = [array_node] if array_node.kind == init_kind else [
            child for child in array_node.get_children() if child.kind == init_kind
        ]
        for init_node in init_nodes:
            for element in init_node.get_children():
                if self._is_end_array_element(element):
                    break
                yield element

    def _is_end_array_element(self, element: Any) -> bool:
        """判断当前数组元素是否为 `{..., nullptr}` 终止项。"""
        null_kind = self._clang.CursorKind.CXX_NULL_PTR_LITERAL_EXPR
        return any(child.kind == null_kind for child in element.get_children())

    def _extract_struct_fields(
        self,
        struct_init: Any,
        method_table: str,
        source_file: str | None,
        function_defs: dict[str, list[Any]],
    ) -> ExtractedFunction | None:
        """
        从 `PyMethodDef` 的单个初始化项提取函数元数据和签名。

        若关键字段（Python 名、C 函数名）缺失则返回 `None`，
        保持提取过程对异常样本的容错性。
        """
        token_kind = self._clang.TokenKind
        tokens = list(struct_init.get_tokens())
        if not tokens:
            return None

        py_name: str | None = None
        meth_flags: list[str] = []
        for token in tokens:
            spelling = str(token.spelling)
            if token.kind == token_kind.LITERAL and py_name is None:
                # 第一个字符串字面量通常是 Python 暴露名。
                lit = self._strip_literal_quotes(spelling)
                if lit and lit != "NULL":
                    py_name = lit
                continue
            if token.kind == token_kind.IDENTIFIER and spelling.startswith("METH_"):
                meth_flags.append(spelling)
                continue
            if token.kind == token_kind.LITERAL and spelling in METH_TYPE_LITERAL_MAP:
                meth_flags.append(METH_TYPE_LITERAL_MAP[spelling])

        if not py_name:
            return None
        meth_flags = self._unique_keep_order(meth_flags)

        c_name = self._find_c_function_name(tokens)
        if not c_name:
            return None

        function_cursor = self._select_function_cursor(function_defs.get(c_name, []))
        signatures: list[ExtractedSignature] = []
        return_type_name: str | None = None
        if function_cursor is not None:
            signatures = self._extract_signatures_from_function(function_cursor, meth_flags)
            return_type_name = self._infer_return_type_from_function(function_cursor)
            if not signatures:
                # 解析不到 PyArg_* 调用时，回退到 C 形参声明推断。
                fallback = self._signature_from_param_decls(function_cursor)
                if fallback.arguments:
                    signatures = [fallback]

        if not signatures:
            signatures = [ExtractedSignature(arguments=[], return_type_name=return_type_name)]

        signatures = [self._merge_signature_return_type(sig, return_type_name) for sig in signatures]
        signatures = [self._apply_method_flags(sig, meth_flags) for sig in signatures]
        signatures = self._deduplicate_signatures(signatures)
        return ExtractedFunction(
            py_name=py_name,
            c_name=c_name,
            method_flags=meth_flags,
            signatures=signatures,
            source_file=source_file,
            method_table=method_table,
        )

    def _extract_signatures_from_function(self, func_cursor: Any, meth_flags: list[str]) -> list[ExtractedSignature]:
        """从函数体中提取候选签名，并在末尾做去重。"""
        signatures: list[ExtractedSignature] = []
        for token_list in self._collect_pyarg_token_lists(func_cursor):
            args = self._set_token_params(func_cursor, meth_flags, token_list)
            if args is not None:
                signatures.append(ExtractedSignature(arguments=args))

        decl_stmt = self._clang.CursorKind.DECL_STMT
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

    def _infer_return_type_from_function(self, func_cursor: Any) -> str | None:
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

        return_stmt_kind = self._clang.CursorKind.RETURN_STMT
        for node in self._walk(func_cursor):
            if node.kind != return_stmt_kind:
                continue
            inferred = self._infer_return_type_from_return_stmt(return_stmt=node, func_cursor=func_cursor)
            if inferred is not None:
                inferred_types.add(inferred)

        if not inferred_types:
            return None
        if len(inferred_types) == 1:
            return sorted(inferred_types)[0]
        return "object"

    def _infer_return_type_from_return_stmt(self, return_stmt: Any, func_cursor: Any) -> str | None:
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
        return_stmt: Any,
        func_cursor: Any,
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

    def _infer_return_type_from_py_buildvalue(self, return_stmt: Any, func_cursor: Any) -> str:
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

    def _resolve_buildvalue_format_token(self, *, func_cursor: Any, token: str) -> str | None:
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

    def _find_first_call_name(self, node: Any) -> str | None:
        """在子树中查找首个 `CALL_EXPR` 的函数名。"""
        call_kind = self._clang.CursorKind.CALL_EXPR
        for child in self._walk(node):
            if child.kind == call_kind and child.spelling:
                return str(child.spelling)
        return None

    def _collect_identifier_literal_tokens(self, node: Any) -> list[str]:
        """收集 cursor 子树中的 `IDENTIFIER` / `LITERAL` token。"""
        token_kind = self._clang.TokenKind
        return [
            str(token.spelling)
            for token in node.get_tokens()
            if token.kind in {token_kind.IDENTIFIER, token_kind.LITERAL}
        ]

    def _collect_pyarg_token_lists(self, node: Any) -> list[list[str]]:
        """
        递归收集 `PyArg_*` / `Py_BuildValue` 调用的 token 序列。

        `IF_STMT` 与 `UNEXPOSED_EXPR` 在不同编译单元下结构可能不同，
        因此这里采用保守递归策略统一处理。
        """
        result: list[list[str]] = []
        call_kind = self._clang.CursorKind.CALL_EXPR
        if_kind = self._clang.CursorKind.IF_STMT
        unexposed_kind = self._clang.CursorKind.UNEXPOSED_EXPR

        for child in node.get_children():
            token_list: list[str] | None = None
            if child.kind == call_kind:
                token_list = self._collect_call_tokens(child)
            elif child.kind == if_kind:
                first_child = next(child.get_children(), None)
                if first_child is not None and first_child.kind == unexposed_kind:
                    token_list = self._collect_call_tokens(first_child)
                else:
                    token_list = self._collect_call_tokens(child)

            if token_list and token_list[0].startswith("PyArg_"):
                result.append(token_list)
                continue
            if token_list and token_list[0] == "Py_BuildValue":
                result.append(token_list)
                continue

            result.extend(self._collect_pyarg_token_lists(child))
        return result

    def _extract_parser_signatures(self, node: Any) -> list[ExtractedSignature]:
        """从声明语句中的 `\"func(type name, ...)\"` 文本签名提取参数。"""
        token_kind = self._clang.TokenKind
        signatures: list[ExtractedSignature] = []
        literals: list[str] = []
        for token in node.get_tokens():
            if token.kind != token_kind.LITERAL:
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
        func_cursor: Any,
        meth_flags: list[str],
        token_list: list[str],
    ) -> list[ExtractedArgument] | None:
        """
        基于调用 token 与方法标志推断参数名、类型和默认值。

        返回 `None` 表示无法可靠解析该调用；返回空列表表示明确无参数。
        """
        if not token_list:
            return None

        format_idx, offset = self._resolve_format_index(call_name=token_list[0], meth_flags=meth_flags)
        if format_idx < 0:
            return []
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
            return []

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

    def _parse_format_token(self, func_cursor: Any, token: str) -> list[str] | None:
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

    def _collect_call_tokens(self, call_node: Any) -> list[str]:
        """
        从调用表达式提取与参数解析相关的 token。

        该步骤会过滤大量无关转换器/宏标识，降低误判概率。
        """
        token_kind = self._clang.TokenKind
        result: list[str] = []
        started = False
        for token in call_node.get_tokens():
            if token.kind not in {token_kind.IDENTIFIER, token_kind.LITERAL}:
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

    def _signature_from_param_decls(self, func_cursor: Any) -> ExtractedSignature:
        """回退方案：直接从 C 形参声明推断签名。"""
        parm_decl = self._clang.CursorKind.PARM_DECL
        args: list[ExtractedArgument] = []
        for node in func_cursor.get_children():
            if node.kind != parm_decl or not node.spelling:
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

    def _get_init_value(self, name: str, func_cursor: Any) -> str | None:
        """从局部变量定义中提取默认值字面表达式。"""
        var_decl = self._clang.CursorKind.VAR_DECL
        for node in self._walk(func_cursor):
            if node.kind != var_decl or node.spelling != name:
                continue
            tokens = [str(token.spelling) for token in node.get_tokens()]
            if "=" not in tokens:
                continue
            eq_idx = tokens.index("=")
            value = "".join(tokens[eq_idx + 1 :]).strip()
            if value:
                return value
        return None

    def _find_format_string(self, func_cursor: Any, format_var_name: str) -> str | None:
        """回溯查找格式串变量对应的字符串字面量。"""
        var_decl = self._clang.CursorKind.VAR_DECL
        for node in self._walk(func_cursor):
            if node.kind != var_decl or node.spelling != format_var_name:
                continue
            for child in self._walk(node):
                if child.kind == self._clang.CursorKind.STRING_LITERAL:
                    return self._strip_literal_quotes(str(child.spelling))
        return None

    def _find_c_function_name(self, tokens: list[Any]) -> str | None:
        """从方法表初始化 token 中提取 C 函数符号名。"""
        token_kind = self._clang.TokenKind
        literal_found = False
        for token in tokens:
            spelling = str(token.spelling)
            if token.kind == token_kind.LITERAL and not literal_found:
                literal_found = True
                continue
            if not literal_found:
                continue
            if token.kind != token_kind.IDENTIFIER:
                continue
            if spelling.startswith("METH_"):
                continue
            if spelling in POINTER_CAST_IDENTIFIER_SKIP:
                # 过滤函数指针 cast 与空指针标识，避免误识别为函数名。
                continue
            return spelling
        return None

    def _select_function_cursor(self, candidates: list[Any]) -> Any | None:
        """优先返回函数定义节点；若无定义则退化到首个声明。"""
        if not candidates:
            return None
        for candidate in candidates:
            try:
                if candidate.is_definition():
                    return candidate
            except Exception:
                continue
        return candidates[0]

    def _is_pymethod_array(self, node: Any) -> bool:
        """判断变量是否为 `PyMethodDef[]`。"""
        try:
            elem_type = node.type.get_array_element_type()
            decl = elem_type.get_declaration()
            return decl.spelling == "PyMethodDef"
        except Exception:
            return False

    def _is_initializer_list(self, node: Any) -> bool:
        """判断变量是否是 `initializer_list<PyMethodDef>` 风格定义。"""
        template_ref = self._clang.CursorKind.TEMPLATE_REF
        has_init = False
        has_pmd = False
        for child in node.get_children():
            if child.kind == template_ref and child.spelling == "initializer_list":
                has_init = True
            if child.spelling == "PyMethodDef":
                has_pmd = True
        return has_init and has_pmd

    def _find_initializer_list_node(self, node: Any) -> Any | None:
        """递归定位包含 `INIT_LIST_EXPR` 的实际初始化节点。"""
        for child in node.get_children():
            grand_children = list(child.get_children())
            if grand_children and grand_children[0].kind == self._clang.CursorKind.INIT_LIST_EXPR:
                return child
            nested = self._find_initializer_list_node(child)
            if nested is not None:
                return nested
        return None

    def _walk(self, node: Any) -> Iterable[Any]:
        """深度优先遍历 cursor 子树。"""
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
    ) -> tuple[str | None, tuple[tuple[str, str | None, str | None, str], ...]]:
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
        seen: set[tuple[str | None, tuple[tuple[str, str | None, str | None, str], ...]]] = set()
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
            seen: set[tuple[str, str | None, str | None, tuple[Any, ...]]] = set()
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

