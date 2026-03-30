from __future__ import annotations

"""`PyArg_ParseTuple` 格式串到参数信息的解析器。"""

from dataclasses import dataclass
from typing import Callable

from clang.cindex import Cursor

from .models import ExtractedArgument
from .py_arg_parse_format_units import _FORMAT_UNIT_SPECS
from .types import RawType, Type


class PyArgParseTupleTypeParserError(ValueError):
    """表示 `PyArg_ParseTuple` 格式串无法被当前解析器接受。"""


class _ParsedValue:
    """表示一个已完成格式串消费、但尚未包装成参数的值节点。"""

    # 仅保留真正接收 ParseTuple 输出值的 decl-ref 槽位。
    c_args: tuple[Cursor, ...]

    def build_type(self) -> Type:
        raise NotImplementedError

    def render_default_value(
        self,
        resolve_default_value_func: Callable[[Cursor], str | None] | None,
    ) -> str | None:
        raise NotImplementedError


@dataclass(frozen=True)
class _ScalarParsedValue(_ParsedValue):
    """标量格式单元的解析结果。"""

    type: Type
    c_args: tuple[Cursor, ...]
    default_value_cursor: Cursor

    def build_type(self) -> Type:
        return self.type

    def render_default_value(
        self,
        resolve_default_value_func: Callable[[Cursor], str | None] | None,
    ) -> str | None:
        if resolve_default_value_func is None:
            return None
        return resolve_default_value_func(self.default_value_cursor)


@dataclass(frozen=True)
class _TupleParsedValue(_ParsedValue):
    """tuple 格式单元的解析结果。"""

    items: tuple[_ParsedValue, ...]
    c_args: tuple[Cursor, ...]

    def build_type(self) -> Type:
        item_types = tuple(item.build_type() for item in self.items)
        if not item_types:
            raise PyArgParseTupleTypeParserError("不支持空 tuple format '()'。")
        item_rendered = [item_type.render() for item_type in item_types]
        imports: set[str] = set()
        for item_type in item_types:
            imports.update(item_type.collect_imports())
        if len(item_rendered) == 1:
            return RawType(f"tuple[{item_rendered[0]},]", imports=sorted(imports))
        return RawType(f"tuple[{', '.join(item_rendered)}]", imports=sorted(imports))

    def render_default_value(
        self,
        resolve_default_value_func: Callable[[Cursor], str | None] | None,
    ) -> str | None:
        if not self.items:
            raise PyArgParseTupleTypeParserError("不支持空 tuple format '()'。")

        rendered_items: list[str] = []
        for item in self.items:
            default_value = item.render_default_value(resolve_default_value_func)
            if default_value is None:
                return None
            rendered_items.append(default_value)

        if len(rendered_items) == 1:
            return f"({rendered_items[0]},)"
        return f"({', '.join(rendered_items)})"


class PyArgParseTupleTypeParser:
    """将 `PyArg_ParseTuple` 的格式串解析为提取参数列表。"""

    def __init__(
        self,
        fmt: str,
        args: list[Cursor],
        resolve_name_func: Callable[[list[Cursor]], str | None],
        resolve_object_type_func: Callable[[Cursor], Type | str | None] | None = None,
        resolve_default_value_func: Callable[[Cursor], str | None] | None = None,
    ) -> None:
        """初始化格式串解析器。"""
        self._format = fmt
        self._args = args
        self._resolve_name_func = resolve_name_func
        self._resolve_object_type_func = resolve_object_type_func
        self._resolve_default_value_func = resolve_default_value_func
        self._char_index = 0
        self._arg_index = 0

    def parse(self) -> list[ExtractedArgument]:
        """解析格式串并返回参数列表。"""
        self._char_index = 0
        self._arg_index = 0

        arguments: list[ExtractedArgument] = []
        in_optional_section = False

        while True:
            self._skip_separators()
            current = self._peek_char()

            if current is None or current in ":;":
                break

            if current == "|":
                if in_optional_section:
                    raise PyArgParseTupleTypeParserError("format string 中发现重复的 '|'。")
                self._advance_char()
                in_optional_section = True
                continue

            arguments.append(self._parse_argument(has_default=in_optional_section))

        self._validate_counts()
        return arguments

    def _parse_argument(self, *, has_default: bool) -> ExtractedArgument:
        """解析一个顶层参数单元并包装为 `ExtractedArgument`。"""
        value = self._parse_value()
        name = self._resolve_name(value.c_args)
        default_value: str | None = None
        if has_default:
            default_value = value.render_default_value(self._resolve_default_value_func)

        return ExtractedArgument(
            name=name,
            type=value.build_type(),
            default_value=default_value,
            has_default=has_default,
        )

    def _parse_value(self) -> _ParsedValue:
        """解析单个格式值，可递归进入 tuple 单元。"""
        current = self._peek_char_required()
        if current == "(":
            return self._parse_tuple_value()

        return self._parse_scalar_value()

    def _parse_tuple_value(self) -> _TupleParsedValue:
        """
        解析 tuple 格式单元。

        tuple 内允许继续嵌套 tuple 或标量单元；默认值和类型渲染都会保留
        当前的结构而不是将叶子参数拍平。
        """
        self._consume_char("(")
        self._skip_separators()
        current = self._peek_char()
        if current is None:
            raise PyArgParseTupleTypeParserError(
                "在 format string 结束前应找到 ')'。"
            )
        if current == ")":
            raise PyArgParseTupleTypeParserError("不支持空 tuple format '()'。")

        items: list[_ParsedValue] = []
        while True:
            items.append(self._parse_value())
            self._skip_separators()
            current = self._peek_char()

            if current is None:
                raise PyArgParseTupleTypeParserError(
                    "在 format string 结束前应找到 ')'。"
                )

            if current == ")":
                self._advance_char()
                break

        c_args: list[Cursor] = []
        for item in items:
            c_args.extend(item.c_args)
        return _TupleParsedValue(tuple(items), tuple(c_args))

    def _parse_scalar_value(self) -> _ScalarParsedValue:
        """按最长匹配规则消费一个标量格式单元。"""
        current = self._peek_char_required()
        for spec in _FORMAT_UNIT_SPECS:
            if self._format.startswith(spec.unit, self._char_index):
                self._char_index += len(spec.unit)
                raw_c_args = self._advance_c_args_required(spec.c_arg_count)
                value_type = spec.type
                if spec.object_type_arg_offset is not None:
                    value_type = self._resolve_object_type(
                        raw_c_args[spec.object_type_arg_offset]
                    )
                decl_ref_cursor = raw_c_args[spec.decl_ref_offset]
                return _ScalarParsedValue(
                    type=value_type,
                    c_args=(decl_ref_cursor,),
                    default_value_cursor=decl_ref_cursor,
                )

        raise PyArgParseTupleTypeParserError(
            f"索引 {self._char_index} 处的 format unit {current!r} 不受支持。"
        )

    def _validate_counts(self) -> None:
        """在解析结束后统一校验 C 参数槽位数量。"""
        if self._arg_index != len(self._args):
            raise PyArgParseTupleTypeParserError(
                f"期望 {self._arg_index} 个 C arguments，实际找到 {len(self._args)} 个。"
            )

    def _resolve_name(self, c_args: tuple[Cursor, ...]) -> str:
        """解析顶层 Python 参数名。"""
        name = self._resolve_name_func(list(c_args))
        if name is None:
            raise PyArgParseTupleTypeParserError("无法解析 argument name。")
        return name

    def _resolve_object_type(self, cursor: Cursor) -> Type:
        """解析对象单元的 Python 类型，未知时回退为 `object`。"""
        if self._resolve_object_type_func is not None:
            resolved_type = self._resolve_object_type_func(cursor)
            if resolved_type is not None:
                if isinstance(resolved_type, str):
                    return RawType(resolved_type)
                return resolved_type
        return RawType("object")

    def _peek_char(self) -> str | None:
        """查看当前位置字符而不推进游标。"""
        if self._char_index >= len(self._format):
            return None
        return self._format[self._char_index]

    def _peek_char_required(self) -> str:
        """查看当前位置字符；若已结束则抛错。"""
        current = self._peek_char()
        if current is None:
            raise PyArgParseTupleTypeParserError("已到达 format string 末尾。")
        return current

    def _advance_char(self) -> str | None:
        """推进一个格式串字符。"""
        current = self._peek_char()
        if current is None:
            return None
        self._char_index += 1
        return current

    def _consume_char(self, expected: str) -> None:
        """消费一个预期字符并在不匹配时抛错。"""
        current = self._advance_char()
        if current != expected:
            found = "end of format string" if current is None else repr(current)
            raise PyArgParseTupleTypeParserError(
                f"期望在索引 {self._char_index - 1} 处找到 {expected!r}，实际为 {found}。"
            )

    def _skip_separators(self) -> None:
        """跳过格式串中的空白与逗号分隔符。"""
        while True:
            current = self._peek_char()
            if current is None or current not in " \t,":
                return
            self._char_index += 1

    def _advance_c_args_required(self, count: int) -> tuple[Cursor, ...]:
        """消费指定数量的 C 参数槽位。"""
        end_index = self._arg_index + count
        if end_index > len(self._args):
            raise PyArgParseTupleTypeParserError(
                f"期望从索引 {self._arg_index} 开始取得 {count} 个 C arguments，但已没有剩余参数。"
            )

        values = tuple(self._args[self._arg_index:end_index])
        self._arg_index = end_index
        return values
