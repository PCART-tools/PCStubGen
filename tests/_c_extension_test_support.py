from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import cast

import clang.cindex
from clang.cindex import LinkageKind, StorageClass
import pytest

from pcstubgen.signature_completion.c_extension import (
    inferencer as signature_rules_module,
)
from pcstubgen.signature_completion.c_extension.libclang.libclang_wrap import (
    CX_BINARY_OPERATOR_ASSIGN,
)
from pcstubgen.type_models import RawType, Type
from pcstubgen.models import (
    Argument,
    ArgumentKind,
    Signature,
)


def _signature(
    *,
    args: list[Argument] | None = None,
    return_type: Type | None = None,
    comment: str | None = None,
) -> Signature:
    """构造测试用签名。"""
    return Signature(
        args=list(args or ()),
        return_type=return_type,
        comment=comment,
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


class _FakeCanonicalType:
    def __init__(self, kind: object | None, spelling: str = "") -> None:
        self.kind = kind
        self.spelling = spelling

    def get_canonical(self) -> "_FakeCanonicalType":
        return self


def _location_text(text: str) -> object:
    class _Location:
        def __str__(self) -> str:
            return text

    return _Location()


def patch_inference_clang_helpers(
    monkeypatch: pytest.MonkeyPatch,
    target_module=signature_rules_module,
) -> None:
    """让 fake cursor 支持推断测试依赖的 libclang helper。"""
    real_cursor_get_text = target_module.get_cursor_source_text

    def fake_cursor_get_text(cursor: object) -> str:
        extent = getattr(cursor, "extent", None)
        if isinstance(extent, str):
            return extent
        if hasattr(extent, "start") and hasattr(extent, "end"):
            start = extent.start
            end = extent.end
            if start.file is not None:
                source_bytes = Path(start.file.name).read_bytes()
                return source_bytes[start.offset:end.offset].decode(
                    "utf-8",
                    errors="ignore",
                )
        return real_cursor_get_text(cast(clang.cindex.Cursor, cursor))

    monkeypatch.setattr(
        target_module,
        "get_cursor_source_text",
        fake_cursor_get_text,
    )
    monkeypatch.setattr(
        target_module.ast_utils,
        "get_cursor_source_text",
        fake_cursor_get_text,
    )

    real_get_call_expr_source_name = target_module.get_first_token_str

    def fake_get_call_expr_source_name(cursor: object) -> str:
        if isinstance(cursor, _FakeNode):
            tokens = list(cursor.get_tokens())
            if not tokens:
                raise RuntimeError(f"调用表达式起点缺少 token, cursor: {cursor.location}")
            return tokens[0].spelling
        return real_get_call_expr_source_name(cast(clang.cindex.Cursor, cursor))

    monkeypatch.setattr(
        target_module,
        "get_first_token_str",
        fake_get_call_expr_source_name,
    )
    monkeypatch.setattr(
        target_module.ast_utils,
        "get_first_token_str",
        fake_get_call_expr_source_name,
    )

    real_unwrap_single_unary_op = target_module.ast_utils.unwrap_single_unary_op

    def fake_unwrap_single_unary_op(cursor: object) -> object:
        if not isinstance(cursor, _FakeNode):
            return real_unwrap_single_unary_op(cast(clang.cindex.Cursor, cursor))

        cursor = target_module.unwrap_transparent(cursor)
        tokens = list(cursor.get_tokens())
        if (
            cursor.kind == clang.cindex.CursorKind.UNARY_OPERATOR
            and tokens
            and tokens[0].spelling == "&"
        ):
            children = list(cursor.get_children())
            return target_module.unwrap_transparent(cast(_FakeNode, children[0]))
        return cursor

    monkeypatch.setattr(
        target_module.ast_utils,
        "unwrap_single_unary_op",
        fake_unwrap_single_unary_op,
    )
    monkeypatch.setattr(
        target_module.ast_utils,
        "evaluate_cursor",
        lambda cursor: target_module.evaluate_cursor(cursor),
    )

    def fake_cursor_binary_operator_kind(cursor: object) -> int:
        operator_kind = cast(_FakeNode, cursor).binary_operator_kind
        assert operator_kind is not None
        return operator_kind

    monkeypatch.setattr(
        target_module,
        "get_cursor_binary_operator_kind",
        fake_cursor_binary_operator_kind,
    )

    def _find_function_cursor(cursor: object) -> clang.cindex.Cursor:
        current = cursor
        while current is not None:
            if getattr(current, "kind", None) == clang.cindex.CursorKind.FUNCTION_DECL:
                return cast(clang.cindex.Cursor, current)
            current = getattr(current, "semantic_parent", None)
        return _fake_function_cursor_with_children(cast(_FakeNode, cursor))

    def infer_expr_type(
        cursor: clang.cindex.Cursor,
        *,
        flags: int = 0,
        owner_class: type | None = None,
    ) -> Type:
        inferencer = target_module.Inferencer(_find_function_cursor(cursor), flags, owner_class)
        return inferencer._infer_expr_type(cursor)

    def infer_return_type(
        cursor: clang.cindex.Cursor,
        *,
        flags: int = 0,
        owner_class: type | None = None,
    ) -> Type:
        inferencer = target_module.Inferencer(cursor, flags, owner_class)
        return inferencer._infer_return_type()

    def infer_arguments_list(
        cursor: clang.cindex.Cursor,
        *,
        flags: int = 0,
        owner_class: type | None = None,
    ) -> list[list[Argument]]:
        inferencer = target_module.Inferencer(cursor, flags, owner_class)
        return inferencer._infer_arguments_list()

    def infer_signature(
        cursor: clang.cindex.Cursor,
        *,
        flags: int = 0,
        owner_class: type | None = None,
    ) -> list[Signature]:
        inferencer = target_module.Inferencer(cursor, flags, owner_class)
        return inferencer.run()

    def infer_type_object_type_for_pyarg(cursor: clang.cindex.Cursor) -> Type:
        inferencer = target_module.Inferencer(_find_function_cursor(cursor), 0, None)
        return inferencer._infer_type_object_type_for_pyarg(cursor)

    def infer_converter_type_for_pyarg(cursor: clang.cindex.Cursor) -> Type:
        inferencer = target_module.Inferencer(_find_function_cursor(cursor), 0, None)
        return inferencer._infer_converter_type_for_pyarg(cursor)

    def infer_default_value_for_pyarg(
        cursor: clang.cindex.Cursor,
        expected_type: Type,
    ) -> str:
        inferencer = target_module.Inferencer(_find_function_cursor(cursor), 0, None)
        return inferencer._infer_default_value_for_pyarg(cursor, expected_type)

    monkeypatch.setattr(target_module, "infer_expr_type", infer_expr_type, raising=False)
    monkeypatch.setattr(target_module, "infer_return_type", infer_return_type, raising=False)
    monkeypatch.setattr(
        target_module,
        "infer_arguments_list",
        infer_arguments_list,
        raising=False,
    )
    monkeypatch.setattr(target_module, "infer_signature", infer_signature, raising=False)
    monkeypatch.setattr(
        target_module,
        "_infer_type_object_type_for_pyarg",
        infer_type_object_type_for_pyarg,
        raising=False,
    )
    monkeypatch.setattr(
        target_module,
        "_infer_converter_type_for_pyarg",
        infer_converter_type_for_pyarg,
        raising=False,
    )
    monkeypatch.setattr(
        target_module,
        "_infer_default_value_for_pyarg",
        infer_default_value_for_pyarg,
        raising=False,
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
        semantic_parent: object | None = None,
        lexical_parent: object | None = None,
        storage_class: object = StorageClass.NONE,
        binary_operator_kind: int | None = None,
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
        self.semantic_parent = semantic_parent
        self.lexical_parent = lexical_parent
        self.storage_class = storage_class
        self.binary_operator_kind = binary_operator_kind
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


def _python_singleton_default_expr(struct_name: str) -> _FakeNode:
    return _wrap(
        clang.cindex.CursorKind.UNARY_OPERATOR,
        _token_identifier_node(struct_name),
    )


def _unary_default_expr(child: _FakeNode) -> _FakeNode:
    return _wrap(clang.cindex.CursorKind.UNARY_OPERATOR, child)


def _cxx_bool_literal() -> _FakeNode:
    return _FakeNode(kind=clang.cindex.CursorKind.CXX_BOOL_LITERAL_EXPR)


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


def _var_decl(
    name: str,
    initializer: _FakeNode | None = None,
    *,
    storage_class: object = StorageClass.NONE,
) -> _FakeNode:
    children = [initializer] if initializer is not None else []
    return _FakeNode(
        kind=clang.cindex.CursorKind.VAR_DECL,
        spelling=name,
        children=children,
        storage_class=storage_class,
    )


def _param_decl(name: str) -> _FakeNode:
    return _FakeNode(
        kind=clang.cindex.CursorKind.PARM_DECL,
        spelling=name,
    )


def _assignment(
    name: str,
    value: _FakeNode,
    *,
    referenced: object | None = None,
) -> _FakeNode:
    """构造 `name = value` 形式的赋值节点。"""
    return _FakeNode(
        kind=clang.cindex.CursorKind.BINARY_OPERATOR,
        tokens=[
            _FakeToken(clang.cindex.TokenKind.IDENTIFIER, name),
            _FakeToken(clang.cindex.TokenKind.PUNCTUATION, "="),
        ],
        children=[
            _token_identifier_node(name, referenced=referenced),
            value,
        ],
        binary_operator_kind=CX_BINARY_OPERATOR_ASSIGN,
    )


def _expr_assignment(target: _FakeNode, value: _FakeNode) -> _FakeNode:
    """构造 `target = value` 形式的赋值节点。"""
    return _FakeNode(
        kind=clang.cindex.CursorKind.BINARY_OPERATOR,
        children=[
            target,
            value,
        ],
        binary_operator_kind=CX_BINARY_OPERATOR_ASSIGN,
    )


def _address_of(name: str, *, referenced: object | None = None) -> _FakeNode:
    return _FakeNode(
        kind=clang.cindex.CursorKind.UNARY_OPERATOR,
        tokens=[_FakeToken(clang.cindex.TokenKind.PUNCTUATION, "&")],
        children=[_token_identifier_node(name, referenced=referenced)],
    )


def _address_of_expr(expr: _FakeNode) -> _FakeNode:
    return _FakeNode(
        kind=clang.cindex.CursorKind.UNARY_OPERATOR,
        tokens=[_FakeToken(clang.cindex.TokenKind.PUNCTUATION, "&")],
        children=[expr],
    )


def _c_style_cast_expr(expr: _FakeNode) -> _FakeNode:
    return _FakeNode(
        kind=clang.cindex.CursorKind.CSTYLE_CAST_EXPR,
        children=[expr],
    )


def _array_subscript(
    name: str,
    index: _FakeNode,
    *,
    referenced: object | None = None,
) -> _FakeNode:
    return _FakeNode(
        kind=clang.cindex.CursorKind.ARRAY_SUBSCRIPT_EXPR,
        children=[
            _token_identifier_node(name, referenced=referenced),
            index,
        ],
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
        tokens=[_FakeToken(clang.cindex.TokenKind.IDENTIFIER, name)],
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
    function_cursor = _FakeNode(
        kind=clang.cindex.CursorKind.FUNCTION_DECL,
        tokens=[_FakeToken(clang.cindex.TokenKind.IDENTIFIER, name)],
        spelling=name,
        children=list(children),
    )
    for child in children:
        _attach_fake_parent(child, function_cursor)
    return cast(
        clang.cindex.Cursor,
        function_cursor,
    )


def _attach_fake_parent(node: _FakeNode, parent: _FakeNode) -> None:
    """递归补齐 fake AST 的语义与词法父节点。"""
    node.semantic_parent = parent
    node.lexical_parent = parent
    for child in node.get_children():
        if isinstance(child, _FakeNode):
            _attach_fake_parent(child, node)
