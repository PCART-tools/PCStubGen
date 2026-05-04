from __future__ import annotations

from enum import Enum, auto

from ...models import Argument, ArgumentKind, Signature
from ...type_models import RawType, Type


class _ArgsParseState(Enum):
    POSITIONAL = auto()
    POSITIONAL_OR_KEYWORD = auto()
    KEYWORD_ONLY = auto()
    FINISHED = auto()


def parse_pybind11_signature(signature_text: str) -> Signature:
    """解析一条 pybind11 `function_record::signature`。"""
    text = signature_text.strip()
    if not text.startswith("("):
        raise RuntimeError("pybind11 单签名必须以 '(' 开始。")

    closing_index = _find_closing_paren(text)
    remainder = text[closing_index + 1 :].strip()
    if not remainder.startswith("->"):
        raise RuntimeError("pybind11 单签名缺少返回值箭头。")

    return_text = remainder[2:].strip()
    if not return_text:
        raise RuntimeError("pybind11 单签名缺少返回值类型。")

    args = parse_args_str(text[1:closing_index])
    return Signature(
        args=args,
        return_type=RawType(return_text),
    )


def parse_args_str(args_str: str) -> list[Argument]:
    """解析 pybind11 单签名中的参数列表。"""
    split_args = _split_args_str(args_str)

    result: list[Argument] = []
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
                arg.kind is ArgumentKind.POSITIONAL_OR_KEYWORD
                for arg in result
            ):
                raise ValueError("位置参数分隔符 '/' 前必须至少有一个普通参数。")

            for arg in result:
                if arg.kind is ArgumentKind.POSITIONAL_OR_KEYWORD:
                    arg.kind = ArgumentKind.POSITIONAL_ONLY
            state = _ArgsParseState.POSITIONAL_OR_KEYWORD
            continue

        if arg_decl == "*":
            if (
                state
                not in (
                    _ArgsParseState.POSITIONAL,
                    _ArgsParseState.POSITIONAL_OR_KEYWORD,
                )
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

            kind = ArgumentKind.VAR_KEYWORD
            state = _ArgsParseState.FINISHED
        elif arg_decl.startswith("*"):
            if state not in (
                _ArgsParseState.POSITIONAL,
                _ArgsParseState.POSITIONAL_OR_KEYWORD,
            ):
                raise ValueError("可变位置参数必须出现在普通参数之后。")
            if default_str is not None:
                raise ValueError("可变位置参数不允许默认值。")

            name = arg_decl[1:].strip()
            if not name:
                raise ValueError("可变位置参数名不能为空。")

            kind = ArgumentKind.VAR_POSITIONAL
            state = _ArgsParseState.KEYWORD_ONLY
        else:
            name = arg_decl.strip()
            if not name:
                raise ValueError("参数名不能为空。")

            if annotation is None:
                raise ValueError(f"普通参数缺少类型注解: {name}")

            if state is _ArgsParseState.KEYWORD_ONLY:
                kind = ArgumentKind.KEYWORD_ONLY
            else:
                kind = ArgumentKind.POSITIONAL_OR_KEYWORD

        result.append(
            Argument(
                name=name,
                default_value=default_str,
                type=annotation,
                kind=kind,
            )
        )

    return result


def _split_args_str(args_str: str) -> list[tuple[str, Type | None, str | None]]:
    """拆分参数列表文本为名称、注解和默认值。"""
    if not args_str.strip():
        return []

    arg_blocks = _split_top_level(args_str, ",")

    result: list[tuple[str, Type | None, str | None]] = []
    for arg_block in arg_blocks:
        if not arg_block.strip():
            raise ValueError("参数列表中存在空参数块。")

        if arg_block.strip() in {"/", "*"}:
            result.append((arg_block.strip(), None, None))
            continue

        default_index = _find_top_level_char(arg_block, "=")
        extra_default_index = (
            _find_top_level_char(arg_block[default_index + 1 :], "=")
            if default_index != -1
            else -1
        )
        if extra_default_index != -1:
            raise ValueError("参数默认值声明中包含多个 '='。")

        name_and_type = arg_block if default_index == -1 else arg_block[:default_index]
        default = arg_block[default_index + 1 :].strip() if default_index != -1 else None

        annotation_index = _find_top_level_char(name_and_type, ":")
        if annotation_index == -1:
            name = name_and_type.strip()
            annotation = None
        else:
            name = name_and_type[:annotation_index].strip()
            annotation_text = name_and_type[annotation_index + 1 :].strip()
            if not annotation_text:
                raise ValueError(f"参数类型注解为空: {name or name_and_type.strip()}")
            annotation = RawType(annotation_text)

        result.append((name, annotation, default))

    return result


def _find_closing_paren(text: str) -> int:
    """查找起始 '(' 对应的闭合位置。"""
    stack = [")"]
    index = 1

    while index < len(text):
        ch = text[index]
        if ch in "\"'":
            index = _find_str_end(text, index) + 1
            continue

        if ch == "(":
            stack.append(")")
        elif ch == ")":
            if not stack:
                raise RuntimeError("pybind11 单签名括号不匹配。")
            stack.pop()
            if not stack:
                return index
        elif ch == "[":
            stack.append("]")
        elif ch == "]":
            _pop_expected(stack, "]")
        elif ch == "{":
            stack.append("}")
        elif ch == "}":
            _pop_expected(stack, "}")
        elif ch == "<":
            stack.append(">")
        elif ch == ">":
            _pop_expected(stack, ">")
        index += 1

    raise RuntimeError("pybind11 单签名缺少闭合 ')'.")


def _split_top_level(text: str, delim: str) -> list[str]:
    if len(delim) != 1:
        raise ValueError("delim must be a single character")

    parts: list[str] = []
    start = 0
    index = 0
    stack: list[str] = []

    while index < len(text):
        ch = text[index]
        if ch in "\"'":
            index = _find_str_end(text, index) + 1
            continue

        if ch == "(":
            stack.append(")")
        elif ch == ")":
            _pop_expected(stack, ")")
        elif ch == "[":
            stack.append("]")
        elif ch == "]":
            _pop_expected(stack, "]")
        elif ch == "{":
            stack.append("}")
        elif ch == "}":
            _pop_expected(stack, "}")
        elif ch == "<":
            stack.append(">")
        elif ch == ">":
            _pop_expected(stack, ">")
        elif not stack and ch == delim:
            parts.append(text[start:index])
            start = index + 1
        index += 1

    if stack:
        raise ValueError("存在未闭合的括号。")

    parts.append(text[start:])
    return parts


def _find_top_level_char(text: str, target: str) -> int:
    """查找最外层字符位置，找不到返回 -1。"""
    index = 0
    stack: list[str] = []

    while index < len(text):
        ch = text[index]
        if ch in "\"'":
            index = _find_str_end(text, index) + 1
            continue

        if ch == "(":
            stack.append(")")
        elif ch == ")":
            _pop_expected(stack, ")")
        elif ch == "[":
            stack.append("]")
        elif ch == "]":
            _pop_expected(stack, "]")
        elif ch == "{":
            stack.append("}")
        elif ch == "}":
            _pop_expected(stack, "}")
        elif ch == "<":
            stack.append(">")
        elif ch == ">":
            _pop_expected(stack, ">")
        elif not stack and ch == target:
            return index
        index += 1

    if stack:
        raise ValueError("存在未闭合的括号。")
    return -1


def _pop_expected(stack: list[str], expected: str) -> None:
    if not stack or stack[-1] != expected:
        raise ValueError("括号不匹配。")
    stack.pop()


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
