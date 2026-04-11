from __future__ import annotations

import re
from enum import Enum, auto

from ..types import RawType, Type
from ..ir_modules import IRArgument, IRArgumentKind, IRFunction, IRModule, IRSignature


class _ArgsParseState(Enum):
    POSITIONAL = auto()
    POSITIONAL_OR_KEYWORD = auto()
    KEYWORD_ONLY = auto()
    FINISHED = auto()


def parse_docstring_signatures(
    _irmodule: IRModule,
    irfunction: IRFunction,
) -> list[IRSignature]:
    """从函数 docstring 中解析签名，失败时抛出 RuntimeError。"""
    func_name = irfunction.name
    doc = irfunction.doc
    if not doc:
        raise RuntimeError("docstring为空或缺失，无法解析签名。")

    doc_lines = doc.splitlines()
    top_signature_regex = re.compile(
        rf"^{re.escape(func_name)}\((?P<args>.*)\)\s*(->\s*(?P<returns>.+))?$"
    )
    match = top_signature_regex.match(doc_lines[0])
    if match is None:
        raise RuntimeError("docstring首行不是目标函数签名声明。")

    if len(doc_lines) < 2 or doc_lines[1].strip() != "Overloaded function.":
        try:
            args = parse_args_str(match.group("args"))
        except ValueError as ex:
            raise RuntimeError(f"docstring签名参数解析失败: {ex}") from ex
        returns = parse_annotation_str((match.group("returns") or "").strip('"'))
        return [
            IRSignature(
                args=args,
                return_type=returns,
            )
        ]

    overload_signature_regex = re.compile(
        rf"^(\s*(?P<overload_number>\d+).\s*)"
        rf"{re.escape(func_name)}\((?P<args>.*)\)\s*->\s*(?P<returns>.+)$"
    )

    overloads: list[IRSignature] = []
    expected_overload_number = 1

    for line in doc_lines[2:]:
        if not line.strip():
            continue

        match = overload_signature_regex.match(line)
        if match is None:
            raise RuntimeError(
                f"重载签名第{expected_overload_number}项格式非法: {line}"
            )

        overload_number = int(match.group("overload_number"))
        if overload_number != expected_overload_number:
            raise RuntimeError(
                f"重载签名序号不连续，期望 {expected_overload_number}，实际 {overload_number}。"
            )

        try:
            args = parse_args_str(match.group("args"))
        except ValueError as ex:
            raise RuntimeError(
                f"重载签名第{expected_overload_number}项参数解析失败: {ex}"
            ) from ex
        overloads.append(
            IRSignature(
                args=args,
                return_type=parse_annotation_str(match.group("returns")),
            )
        )
        expected_overload_number += 1

    if not overloads:
        raise RuntimeError("Overloaded function. 之后未找到有效重载签名。")

    return overloads


def parse_args_str(args_str: str) -> list[IRArgument]:
    split_args = _split_args_str(args_str)

    result: list[IRArgument] = []
    state = _ArgsParseState.POSITIONAL

    for arg_decl, annotation, default_str in split_args:
        if state is _ArgsParseState.FINISHED:
            raise ValueError("可变关键字参数之后不允许再出现其他参数。")

        if arg_decl == "/":
            if (
                state is not _ArgsParseState.POSITIONAL
                or annotation is not None
                or default_str is not None
            ):
                raise ValueError("位置参数分隔符 '/' 位置非法。")

            if not any(
                arg.kind is IRArgumentKind.POSITIONAL_OR_KEYWORD for arg in result
            ):
                raise ValueError("位置参数分隔符 '/' 前必须至少有一个普通参数。")

            for arg in result:
                if arg.kind is IRArgumentKind.POSITIONAL_OR_KEYWORD:
                    arg.kind = IRArgumentKind.POSITIONAL_ONLY
            state = _ArgsParseState.POSITIONAL_OR_KEYWORD
            continue

        if arg_decl == "*":
            if (
                state not in (_ArgsParseState.POSITIONAL, _ArgsParseState.POSITIONAL_OR_KEYWORD)
                or annotation is not None
                or default_str is not None
            ):
                raise ValueError("关键字专用分隔符 '*' 位置非法。")

            state = _ArgsParseState.KEYWORD_ONLY
            continue

        if arg_decl.startswith("**"):
            if default_str is not None:
                raise ValueError("可变关键字参数不允许默认值。")

            name = arg_decl[2:].strip()
            if not name:
                raise ValueError("可变关键字参数名不能为空。")

            kind = IRArgumentKind.VAR_KEYWORD
            state = _ArgsParseState.FINISHED
        elif arg_decl.startswith("*"):
            if state not in (_ArgsParseState.POSITIONAL, _ArgsParseState.POSITIONAL_OR_KEYWORD):
                raise ValueError("可变位置参数必须出现在普通参数之后。")
            if default_str is not None:
                raise ValueError("可变位置参数不允许默认值。")

            name = arg_decl[1:].strip()
            if not name:
                raise ValueError("可变位置参数名不能为空。")

            kind = IRArgumentKind.VAR_POSITIONAL
            state = _ArgsParseState.KEYWORD_ONLY
        else:
            name = arg_decl.strip()
            if not name:
                raise ValueError("参数名不能为空。")

            if state is _ArgsParseState.KEYWORD_ONLY:
                kind = IRArgumentKind.KEYWORD_ONLY
            else:
                kind = IRArgumentKind.POSITIONAL_OR_KEYWORD

        result.append(
            IRArgument(
                name=name,
                default_value=default_str,
                has_default=default_str is not None,
                type=annotation,
                kind=kind,
            )
        )

    return result


def parse_annotation_str(annotation_str: str) -> Type | None:
    text = annotation_str.strip()
    if not text:
        return None
    return RawType(text)


def _split_args_str(
    args_str: str,
) -> list[tuple[str, Type | None, str | None]]:
    if not args_str.strip():
        return []

    arg_blocks = _split_top_level(args_str, ",")

    result: list[tuple[str, Type | None, str | None]] = []
    for arg_block in arg_blocks:
        if not arg_block.strip():
            raise ValueError("参数列表中存在空参数块。")

        name_and_default = _split_top_level(arg_block, "=")
        if len(name_and_default) > 2:
            raise ValueError("参数默认值声明中包含多个 '='。")

        name_and_type = name_and_default[0]
        default = name_and_default[1].strip() if len(name_and_default) == 2 else None

        name_type_parts = _split_top_level(name_and_type, ":")
        if len(name_type_parts) > 2:
            raise ValueError("参数注解声明中包含多个 ':'。")

        name = name_type_parts[0].strip()
        type_ = (
            parse_annotation_str(name_type_parts[1])
            if len(name_type_parts) == 2
            else None
        )
        result.append((name, type_, default))

    return result


def _split_top_level(text: str, delim: str) -> list[str]:
    if len(delim) != 1:
        raise ValueError("delim must be a single character")

    left_to_right = {"(": ")", "{": "}", "[": "]"}
    rights = left_to_right.values()
    stack: list[str] = []
    parts: list[str] = []
    start = 0
    index = 0

    while index < len(text):
        ch = text[index]
        if ch in "\"'":
            str_end = _find_str_end(text, index)
            index = str_end + 1
            continue

        if ch in left_to_right:
            stack.append(left_to_right[ch])
        elif ch in rights:
            if not stack or ch != stack[-1]:
                raise ValueError("括号不匹配。")
            stack.pop()
        elif not stack and ch == delim:
            parts.append(text[start:index])
            start = index + 1
        index += 1

    if stack:
        raise ValueError("存在未闭合的括号。")

    parts.append(text[start:])
    return parts


def _find_str_end(text: str, start: int) -> int:
    quote = text[start]
    index = start + 1
    while index < len(text):
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == quote:
            return index
        index += 1
    raise ValueError("字符串字面量未闭合。")
