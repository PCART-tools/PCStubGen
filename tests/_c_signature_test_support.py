from __future__ import annotations

import sysconfig
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import clang.cindex
import pytest

from pcstubgen.c_signature import extract_c_signature_modules
from pcstubgen.c_signature.constants import (
    METH_KEYWORDS,
    METH_VARARGS,
)
from pcstubgen.c_signature import (
    c_signature_extraction as c_signature_extraction_module,
)
from pcstubgen.c_signature import cursor_utils as cursor_utils_module
from pcstubgen.c_signature import signature_inference as signature_rules_module
from pcstubgen.c_signature import module_table as module_table_module
from pcstubgen.c_signature import clang_parser as translation_unit_module
from pcstubgen.c_signature.types import (
    AnyType,
    ListType,
    RawType,
    Type,
    TupleType,
    UnionType,
)
from pcstubgen.c_signature.module_table import (
    extract_method_table as _extract_method_table,
    extract_pymethoddef_init_list_expr as _extract_PyMethodDef_INIT_LIST_EXPR,
    resolve_init_list_expr as _resolve_INIT_LIST_EXPR,
)
from pcstubgen.c_signature.models import (
    ExtractedArgument,
    ExtractedFunction,
    ExtractedModule,
    ExtractedSignature,
)
from pcstubgen.c_signature.resolver import (
    CSignatureResolver,
)
from pcstubgen.docstring_signature import (
    DocstringSignatureParser,
    resolve_docstring_signatures,
)
from pcstubgen.inspect_signature import (
    resolve_inspect_signatures,
)
from pcstubgen.ir import (
    IRArgument,
    IRArgumentKind,
    IRClass,
    IRFunction,
    IRMethod,
    IRModule,
    IRModuleType,
    IRSignature,
    QualifiedName,
)
from pcstubgen.stub_generation_options import StubGenerationOptions
from pcstubgen.supplementer import (
    SignatureSupplementSummary,
    SignatureSupplementer,
    supplement_signatures,
)


def _signature(
    *,
    args: list[IRArgument] | None = None,
    return_type: Type | None = None,
    doc: str | None = None,
) -> IRSignature:
    """构造测试用 IR 签名。"""
    return IRSignature(
        args=list(args or ()),
        return_type=return_type,
        doc=doc,
    )


def _raw(text: str, *, imports: list[str] | None = None) -> RawType:
    return RawType(text, imports=imports)


def _arg(
    name: str,
    type_text: str | None = None,
    *,
    imports: list[str] | None = None,
    default_value: str | None = None,
    has_default: bool = False,
    kind: IRArgumentKind = IRArgumentKind.POSITIONAL_OR_KEYWORD,
) -> ExtractedArgument:
    return ExtractedArgument(
        name=name,
        type=None if type_text is None else _raw(type_text, imports=imports),
        default_value=default_value,
        has_default=has_default,
        kind=kind,
    )


def _unknown_function(
    name: str,
    *,
    doc: str | None = None,
    runtime_function: object | None = None,
) -> IRFunction:
    """构造签名未知的测试函数。"""
    return IRFunction(name=name, doc=doc, runtime_function=runtime_function)


def _module_fixture(
    *,
    name: str = "pkg.mod",
    functions: dict[str, ExtractedFunction] | None = None,
) -> dict[str, ExtractedModule]:
    return {
        name: ExtractedModule(
            name=name,
            functions=functions or {},
        )
    }


def _make_extraction_config(
    *,
    source_root: Path,
    include: list[str] = (),
    include_directory: list[Path] = (),
    c_std: str = "c11",
    cpp_std: str = "c++17",
) -> dict[str, object]:
    return {
        "source_root": source_root,
        "include": list(include),
        "include_directory": translation_unit_module.inject_python_include_directories(
            list(include_directory)
        ),
        "c_std": c_std,
        "cpp_std": cpp_std,
    }


class CSignatureExtractor:
    def __init__(
        self,
        source_root: Path,
        *,
        include: list[str] = (),
        include_directory: list[Path] = (),
        c_std: str = "c11",
        cpp_std: str = "c++17",
    ) -> None:
        self._source_root = source_root
        self._include = list(include)
        self._include_directory = translation_unit_module.inject_python_include_directories(
            list(include_directory)
        )
        self._c_std = c_std
        self._cpp_std = cpp_std

    def extract_modules(self) -> dict[str, ExtractedModule]:
        return extract_c_signature_modules(
            self._source_root,
            include=self._include,
            include_directory=self._include_directory,
            c_std=self._c_std,
            cpp_std=self._cpp_std,
        )


class _FakeExtractor:
    def __init__(
        self,
        modules: dict[str, ExtractedModule] | None = None,
    ) -> None:
        self.modules = modules or {}
        self.called = 0

    def extract_modules(self) -> dict[str, ExtractedModule]:
        self.called += 1
        return self.modules


def _patch_c_signature_extractor(
    monkeypatch: pytest.MonkeyPatch,
    modules: dict[str, ExtractedModule] | None = None,
) -> _FakeExtractor:
    extractor = _FakeExtractor(modules=modules)

    def _patched_extract_c_signature_modules(
        source_root: Path,
        *,
        include: list[str] = (),
        include_directory: list[Path] = (),
        c_std: str = "c11",
        cpp_std: str = "c++17",
    ) -> dict[str, ExtractedModule]:
        _ = (source_root, include, include_directory, c_std, cpp_std)
        return extractor.extract_modules()

    import pcstubgen.c_signature.resolver as resolver_module

    monkeypatch.setattr(resolver_module, "extract_c_signature_modules", _patched_extract_c_signature_modules)
    return extractor


def _patch_raising_c_signature_extractor(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    def _patched_extract_c_signature_modules(
        source_root: Path,
        *,
        include: list[str] = (),
        include_directory: list[Path] = (),
        c_std: str = "c11",
        cpp_std: str = "c++17",
    ) -> dict[str, ExtractedModule]:
        _ = (source_root, include, include_directory, c_std, cpp_std)
        raise error

    import pcstubgen.c_signature.resolver as resolver_module

    monkeypatch.setattr(resolver_module, "extract_c_signature_modules", _patched_extract_c_signature_modules)


def _get_packaged_libclang_path() -> str | None:
    import clang

    native_dir = Path(clang.__file__).resolve().parent / "native"
    for filename in ("libclang.dll", "libclang.so", "libclang.dylib"):
        candidate = native_dir / filename
        if candidate.exists():
            return str(candidate)
    return None


class _FakeDiagnosticType:
    Ignored = 0
    Note = 1
    Warning = 2
    Error = 3
    Fatal = 4


class _FakeClangWithDiagnostics:
    Diagnostic = _FakeDiagnosticType


class _FakeDiagnosticFile:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeDiagnosticLocation:
    def __init__(self, *, file_name: str | None, line: int, column: int) -> None:
        self.file = _FakeDiagnosticFile(file_name) if file_name is not None else None
        self.line = line
        self.column = column


class _FakeDiagnostic:
    def __init__(
        self,
        *,
        severity: int,
        message: str,
        file_name: str | None,
        line: int,
        column: int,
    ) -> None:
        self.severity = severity
        self.spelling = message
        self.location = _FakeDiagnosticLocation(file_name=file_name, line=line, column=column)


class _FakeTranslationUnit:
    def __init__(self, diagnostics: list[_FakeDiagnostic]) -> None:
        self.diagnostics = diagnostics


class _FakeIndex:
    def __init__(self, translation_unit: _FakeTranslationUnit) -> None:
        self.translation_unit = translation_unit

    def parse(self, filename: str, args: list[str]) -> _FakeTranslationUnit:
        return self.translation_unit


class _SequentialIndex:
    def __init__(self, translation_units: list[_FakeTranslationUnit]) -> None:
        self._translation_units = translation_units
        self._index = 0
        self.calls: list[tuple[str, list[str]]] = []

    def parse(self, filename: str, args: list[str]) -> _FakeTranslationUnit:
        self.calls.append((filename, list(args)))
        if not self._translation_units:
            raise AssertionError("translation_units must not be empty")
        if self._index < len(self._translation_units):
            current = self._translation_units[self._index]
            self._index += 1
            return current
        return self._translation_units[-1]


def _has_include_directory_arg(args: list[str], include_dir: str | Path) -> bool:
    include_dir_str = str(include_dir)
    for index, token in enumerate(args):
        if token != "--include-directory":
            continue
        if index + 1 >= len(args):
            continue
        if args[index + 1] == include_dir_str:
            return True
    return False


def _has_std_arg(args: list[str], std_value: str) -> bool:
    for index, token in enumerate(args):
        if token != "--std":
            continue
        if index + 1 >= len(args):
            continue
        if args[index + 1] == std_value:
            return True
    return False
class _FakeToken:
    def __init__(self, kind: object, spelling: str) -> None:
        self.kind = kind
        self.spelling = spelling


class _FakeCursorLocation:
    def __init__(self, file: str | None = None, offset: int = 0) -> None:
        self.file = _FakeCursorFile(file) if file is not None else None
        self.offset = offset


class _FakeCursorFile:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSourceRange:
    def __init__(self, start: _FakeCursorLocation, end: _FakeCursorLocation) -> None:
        self.start = start
        self.end = end


class _FakeNode:
    def __init__(
        self,
        *,
        kind: object,
        tokens: list[_FakeToken] | None = None,
        children: list[object] | None = None,
        spelling: str = "",
        location: object | None = None,
        extent: object | None = None,
        referenced: object | None = None,
    ) -> None:
        self.kind = kind
        self._tokens = tokens or []
        self._children = children or []
        self.spelling = spelling
        self.location = location if location is not None else _FakeCursorLocation()
        self.extent = extent
        self.referenced = referenced
        self.type = None

    def get_tokens(self) -> list[_FakeToken]:
        return self._tokens

    def get_children(self) -> Iterable[object]:
        return iter(self._children)

    def is_definition(self) -> bool:
        return False


def _fake_function_cursor(name: str = "fake_function") -> clang.cindex.Cursor:
    """构造可复用的假函数游标。"""
    return cast(
        clang.cindex.Cursor,
        _FakeNode(kind=clang.cindex.CursorKind.FUNCTION_DECL, spelling=name),
    )


def _int_literal(value: str = "0") -> _FakeNode:
    return _FakeNode(
        kind=clang.cindex.CursorKind.INTEGER_LITERAL,
        tokens=[_FakeToken(clang.cindex.TokenKind.LITERAL, value)],
    )


def _null_ptr_literal() -> _FakeNode:
    return _FakeNode(kind=clang.cindex.CursorKind.CXX_NULL_PTR_LITERAL_EXPR)


def _gnu_null_literal() -> _FakeNode:
    return _FakeNode(kind=clang.cindex.CursorKind.GNU_NULL_EXPR)


def _identifier_node(name: str) -> _FakeNode:
    return _FakeNode(
        kind=clang.cindex.CursorKind.DECL_REF_EXPR,
        spelling=name,
        tokens=[_FakeToken(clang.cindex.TokenKind.IDENTIFIER, name)],
    )


def _wrap(kind: object, child: _FakeNode) -> _FakeNode:
    return _FakeNode(kind=kind, children=[child])


def _init_list(*children: _FakeNode) -> _FakeNode:
    return _FakeNode(kind=clang.cindex.CursorKind.INIT_LIST_EXPR, children=list(children))


def _designated_initializer(field_name: str, value: _FakeNode) -> _FakeNode:
    referenced = _FakeNode(kind=clang.cindex.CursorKind.FIELD_DECL, spelling=field_name)
    return _FakeNode(
        kind=clang.cindex.CursorKind.UNEXPOSED_EXPR,
        children=[
            _token_identifier_node(
                field_name,
                kind=clang.cindex.CursorKind.MEMBER_REF,
                referenced=referenced,
            ),
            value,
        ],
    )


def _string_literal(value: str) -> _FakeNode:
    return _FakeNode(
        kind=clang.cindex.CursorKind.STRING_LITERAL,
        tokens=[_FakeToken(clang.cindex.TokenKind.LITERAL, f'"{value}"')],
        spelling=f'"{value}"',
    )


def _float_literal(value: str) -> _FakeNode:
    return _FakeNode(
        kind=clang.cindex.CursorKind.FLOATING_LITERAL,
        tokens=[_FakeToken(clang.cindex.TokenKind.LITERAL, value)],
    )


def _token_identifier_node(
    name: str,
    *,
    kind: object = clang.cindex.CursorKind.DECL_REF_EXPR,
    referenced: object | None = None,
) -> _FakeNode:
    return _FakeNode(
        kind=kind,
        spelling=name,
        tokens=[_FakeToken(clang.cindex.TokenKind.IDENTIFIER, name)],
        referenced=referenced,
    )


def _var_decl(name: str, initializer: _FakeNode | None = None) -> _FakeNode:
    children = [initializer] if initializer is not None else []
    return _FakeNode(
        kind=clang.cindex.CursorKind.VAR_DECL,
        spelling=name,
        children=children,
    )


def _address_of(name: str, *, referenced: object | None = None) -> _FakeNode:
    return _FakeNode(
        kind=clang.cindex.CursorKind.UNARY_OPERATOR,
        children=[_token_identifier_node(name, referenced=referenced)],
    )


def _extent_for_source_snippet(source_path: Path, snippet: str) -> _FakeSourceRange:
    source_bytes = source_path.read_bytes()
    snippet_bytes = snippet.encode("utf-8")
    start_offset = source_bytes.index(snippet_bytes)
    end_offset = start_offset + len(snippet_bytes)
    return _FakeSourceRange(
        _FakeCursorLocation(str(source_path), start_offset),
        _FakeCursorLocation(str(source_path), end_offset),
    )


def _signed_numeric_literal(sign: str, child: _FakeNode) -> _FakeNode:
    return _FakeNode(
        kind=clang.cindex.CursorKind.UNARY_OPERATOR,
        tokens=[_FakeToken(clang.cindex.TokenKind.PUNCTUATION, sign)],
        children=[child],
    )


def _call_expr(name: str, *args: _FakeNode) -> _FakeNode:
    return _FakeNode(
        kind=clang.cindex.CursorKind.CALL_EXPR,
        spelling=name,
        children=[
            _FakeNode(
                kind=clang.cindex.CursorKind.UNEXPOSED_EXPR,
                spelling=name,
                children=[_token_identifier_node(name)],
            ),
            *args,
        ],
    )


def _conditional_expr(condition: _FakeNode, when_true: _FakeNode, when_false: _FakeNode) -> _FakeNode:
    return _FakeNode(
        kind=clang.cindex.CursorKind.CONDITIONAL_OPERATOR,
        children=[condition, when_true, when_false],
    )


def _return_stmt(expr: _FakeNode | None = None) -> _FakeNode:
    """构造 return 语句节点。"""
    children = [] if expr is None else [expr]
    return _FakeNode(kind=clang.cindex.CursorKind.RETURN_STMT, children=children)


def _macro_expr(name: str) -> _FakeNode:
    return _FakeNode(
        kind=clang.cindex.CursorKind.UNEXPOSED_EXPR,
        spelling=name,
        children=[_FakeNode(kind=clang.cindex.CursorKind.DECL_REF_EXPR, spelling=name)],
    )


def _fake_function_cursor_with_children(
    *children: _FakeNode,
    name: str = "fake_function",
) -> clang.cindex.Cursor:
    """构造带子节点的假函数游标。"""
    return cast(
        clang.cindex.Cursor,
        _FakeNode(
            kind=clang.cindex.CursorKind.FUNCTION_DECL,
            spelling=name,
            children=list(children),
        ),
    )


def _ml_name_field(name: str) -> _FakeNode:
    return _wrap(clang.cindex.CursorKind.UNEXPOSED_EXPR, _wrap(clang.cindex.CursorKind.UNEXPOSED_EXPR, _string_literal(name)))


def _ml_meth_field(
    name: str,
    *,
    referenced_kind: object = clang.cindex.CursorKind.FUNCTION_DECL,
) -> _FakeNode:
    referenced = _FakeNode(kind=referenced_kind, spelling=name)
    return _FakeNode(
        kind=clang.cindex.CursorKind.UNEXPOSED_EXPR,
        spelling=name,
        children=[_token_identifier_node(name, referenced=referenced)],
    )


def _ml_meth_cast_field(name: str) -> _FakeNode:
    referenced = _FakeNode(kind=clang.cindex.CursorKind.FUNCTION_DECL, spelling=name)
    return _wrap(
        clang.cindex.CursorKind.UNEXPOSED_EXPR,
        _wrap(
            clang.cindex.CursorKind.PAREN_EXPR,
            _FakeNode(
                kind=clang.cindex.CursorKind.CSTYLE_CAST_EXPR,
                children=[_token_identifier_node(name, referenced=referenced)],
            ),
        ),
    )


def _ml_flags_identifier_field(*flags: str) -> _FakeNode:
    return _FakeNode(
        kind=clang.cindex.CursorKind.BINARY_OPERATOR,
        children=[_token_identifier_node(flag) for flag in flags],
    )


def _kwlist_decl(name: str, *keywords: str) -> _FakeNode:
    return _var_decl(
        name,
        _init_list(
            *[_string_literal(keyword) for keyword in keywords],
            _FakeNode(kind=clang.cindex.CursorKind.GNU_NULL_EXPR),
        ),
    )


def _type_object_decl(name: str, tp_name: str) -> _FakeNode:
    return _var_decl(
        name,
        _init_list(_designated_initializer("tp_name", _string_literal(tp_name))),
    )


def _patch_fake_eval_int(monkeypatch: pytest.MonkeyPatch) -> None:
    original_eval_int = cursor_utils_module.clang_eval.eval_int
    method_flag_values = {
        "METH_VARARGS": METH_VARARGS,
        "METH_KEYWORDS": METH_KEYWORDS,
    }

    def _eval_int(cursor: object) -> int | None:
        if not isinstance(cursor, _FakeNode):
            return original_eval_int(cursor)
        if cursor.kind == clang.cindex.CursorKind.INTEGER_LITERAL:
            for token in cursor.get_tokens():
                if token.kind != clang.cindex.TokenKind.LITERAL:
                    continue
                text = str(token.spelling).strip()
                if not text:
                    continue
                try:
                    return int(text, 0)
                except ValueError:
                    continue
            return None
        if cursor.kind == clang.cindex.CursorKind.DECL_REF_EXPR:
            return method_flag_values.get(cursor.spelling)
        if cursor.kind == clang.cindex.CursorKind.BINARY_OPERATOR:
            value = 0
            for child in cursor.get_children():
                child_value = _eval_int(child)
                if child_value is None:
                    return None
                value |= child_value
            return value
        return None

    monkeypatch.setattr(cursor_utils_module.clang_eval, "eval_int", _eval_int)


__all__ = [name for name in globals() if not name.startswith("__")]
