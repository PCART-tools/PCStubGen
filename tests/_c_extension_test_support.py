from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import clang.cindex
from clang.cindex import LinkageKind
import pytest

from pcstubgen.signature_completion.c_extension import (
    source as c_extension_source_module,
)
from pcstubgen.signature_completion.c_extension.clang import ast_utils as ast_utils_module
from pcstubgen.type_models import RawType, Type
from pcstubgen.signature_completion.c_extension.source import (
    CInferenceResult,
    CExtensionSource,
)
from pcstubgen.models import (
    Argument,
    ArgumentKind,
    Function,
    Module,
    Signature,
)


def _signature(
    *,
    args: list[Argument] | None = None,
    return_type: Type | None = None,
) -> Signature:
    """构造测试用签名。"""
    return Signature(
        args=list(args or ()),
        return_type=return_type,
    )


def _arg(
    name: str,
    type_text: str | Type | None = None,
    *,
    imports: tuple[str, ...] = (),
    default_value: str | None = None,
    kind: ArgumentKind = ArgumentKind.POSITIONAL_OR_KEYWORD,
) -> Argument:
    return Argument(
        name=name,
        type=(
            None
            if type_text is None
            else type_text
            if isinstance(type_text, Type)
            else RawType(type_text, imports=imports)
        ),
        default_value=default_value,
        kind=kind,
    )


def _unknown_function(
    name: str,
    *,
    doc: str | None = None,
) -> Function:
    """构造签名未知的测试函数。"""
    return Function(name=name, runtime_handle=object(), doc=doc)


@dataclass
class ResolvedFunctionFixture:
    signatures: list[Signature]
    function_cursor: clang.cindex.Cursor | None = None


def _patch_c_signature_extractor(
    monkeypatch: pytest.MonkeyPatch,
    functions: dict[str, ResolvedFunctionFixture] | None = None,
) -> None:
    extracted_functions = functions or {}

    def _patched_infer_function_signatures(
        self: CExtensionSource,
        module_node: Module,
        function_node: Function,
    ) -> CInferenceResult:
        _ = self
        extracted = extracted_functions.get(function_node.name)
        if extracted is None:
            raise RuntimeError(f"未找到函数 {module_node.full_name}.{function_node.name}")
        if not extracted.signatures:
            raise RuntimeError(f"C函数 {module_node.full_name}.{function_node.name} 没有可用签名")

        comment = ""
        if extracted.function_cursor is not None and extracted.function_cursor.extent is not None:
            location_text = str(extracted.function_cursor.location)
            source_text = ast_utils_module.get_cursor_text(extracted.function_cursor)
            comment = f"{location_text}\n{source_text}"

        return CInferenceResult(signatures=extracted.signatures, comment=comment)

    monkeypatch.setattr(
        c_extension_source_module.CExtensionSource,
        "infer_function_signatures",
        _patched_infer_function_signatures,
    )


class _FakeToken:
    def __init__(self, kind: object, spelling: str) -> None:
        self.kind = kind
        self.spelling = spelling


class _FakeCursorLocation:
    def __init__(
        self,
        file: str | None = None,
        offset: int = 0,
        line: int = 0,
        column: int = 0,
    ) -> None:
        self.file = _FakeCursorFile(file) if file is not None else None
        self.offset = offset
        self.line = line
        self.column = column


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
        canonical: object | None = None,
        usr: str = "",
        definition: object | None = None,
        is_definition: bool = False,
        linkage: object = LinkageKind.EXTERNAL,
    ) -> None:
        self.kind = kind
        self._tokens = tokens or []
        self._children = children or []
        self.spelling = spelling
        self.location = location if location is not None else _FakeCursorLocation()
        self.extent = extent
        self.referenced = referenced
        self.canonical = self if canonical is None else canonical
        self._usr = usr
        self._definition = definition
        self._is_definition = is_definition
        self.linkage = linkage
        self.type = None

    def get_tokens(self) -> list[_FakeToken]:
        return self._tokens

    def get_children(self) -> Iterable[object]:
        return iter(self._children)

    def is_definition(self) -> bool:
        return self._is_definition

    def get_usr(self) -> str:
        return self._usr

    def get_definition(self) -> object | None:
        if self._definition is not None:
            return self._definition
        if self.referenced is not None and getattr(self.referenced, "is_definition", None):
            if self.referenced.is_definition():
                return self.referenced
        return None


def _int_literal(value: str = "0") -> _FakeNode:
    return _FakeNode(
        kind=clang.cindex.CursorKind.INTEGER_LITERAL,
        tokens=[_FakeToken(clang.cindex.TokenKind.LITERAL, value)],
    )


def _null_ptr_literal() -> _FakeNode:
    return _FakeNode(kind=clang.cindex.CursorKind.CXX_NULL_PTR_LITERAL_EXPR)


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
    canonical: object | None = None,
    usr: str = "",
) -> _FakeNode:
    return _FakeNode(
        kind=kind,
        spelling=name,
        tokens=[_FakeToken(clang.cindex.TokenKind.IDENTIFIER, name)],
        referenced=referenced,
        canonical=canonical,
        usr=usr,
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


def _call_expr(name: str, *args: _FakeNode) -> _FakeNode:
    return _FakeNode(
        kind=clang.cindex.CursorKind.CALL_EXPR,
        spelling=name,
        children=[
            _FakeNode(
                kind=clang.cindex.CursorKind.UNEXPOSED_EXPR,
                spelling=name,
                extent=name,
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


ExtractedArgument = Argument
ExtractedSignature = Signature


__all__ = [name for name in globals() if not name.startswith("__")]
