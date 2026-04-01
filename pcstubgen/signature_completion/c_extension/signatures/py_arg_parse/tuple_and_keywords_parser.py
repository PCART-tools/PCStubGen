from __future__ import annotations

"""`PyArg_ParseTupleAndKeywords` 格式串到参数信息的解析器。"""

from enum import Enum
from typing import Callable

from clang.cindex import Cursor

from .....ir import IRArgumentKind
from ...models import CArgument
from .format_units import _FORMAT_UNIT_SPECS, _FormatUnitSpec
from .....types import RawType, Type


class PyArgParseTupleAndKeywordsTypeParserError(ValueError):
    """表示 `PyArg_ParseTupleAndKeywords` 格式串无法被当前解析器接受。"""


class _ArgumentSection(Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    KEYWORD_ONLY = "keyword_only"


class PyArgParseTupleAndKeywordsTypeParser:
    """将 `PyArg_ParseTupleAndKeywords` 的格式串解析为提取参数列表。"""

    def __init__(
        self,
        fmt: str,
        kwlist: list[str],
        args: list[Cursor],
        resolve_object_type_func: Callable[[Cursor], Type | str | None] | None = None,
        resolve_default_value_func: Callable[[Cursor], str | None] | None = None,
    ) -> None:
        """初始化格式串解析器。"""
        self._format = fmt
        self._kwlist = kwlist
        self._args = args
        self._resolve_object_type_func = resolve_object_type_func
        self._resolve_default_value_func = resolve_default_value_func
        self._char_index = 0
        self._arg_index = 0
        self._python_arg_index = 0

    def parse(self) -> list[CArgument]:
        """解析格式串并返回参数列表。"""
        self._char_index = 0
        self._arg_index = 0
        self._python_arg_index = 0

        arguments: list[CArgument] = []
        section = _ArgumentSection.REQUIRED

        while True:
            self._skip_separators()
            current = self._peek_char()

            if current is None or current in ":;":
                break

            if current == "|":
                if section is _ArgumentSection.OPTIONAL:
                    raise PyArgParseTupleAndKeywordsTypeParserError(
                        "format string 中发现重复的 '|'。"
                    )
                if section is _ArgumentSection.KEYWORD_ONLY:
                    raise PyArgParseTupleAndKeywordsTypeParserError(
                        "format string 中在 '$' 之后出现了 '|'。"
                    )
                self._advance_char()
                section = _ArgumentSection.OPTIONAL
                continue

            if current == "$":
                if section is _ArgumentSection.REQUIRED:
                    raise PyArgParseTupleAndKeywordsTypeParserError(
                        "format string 中在 '|' 之前出现了 '$'。"
                    )
                if section is _ArgumentSection.KEYWORD_ONLY:
                    raise PyArgParseTupleAndKeywordsTypeParserError(
                        "format string 中发现重复的 '$'。"
                    )
                self._advance_char()
                section = _ArgumentSection.KEYWORD_ONLY
                continue

            arguments.append(self._parse_argument(section))

        self._validate_counts()
        return arguments

    def _parse_argument(self, section: _ArgumentSection) -> CArgument:
        """解析一个格式单元并产出单个 Python 参数。"""
        name = self._advance_keyword_name_required()
        spec = self._advance_format_unit_required()
        c_args = self._advance_c_args_required(spec.c_arg_count)

        arg_type = spec.type
        if spec.object_type_arg_offset is not None:
            arg_type = self._resolve_object_type(c_args[spec.object_type_arg_offset])

        has_default = section is not _ArgumentSection.REQUIRED

        default_value: str | None = None
        if has_default:
            default_value = self._resolve_default_value(c_args[spec.decl_ref_offset])

        kind = IRArgumentKind.POSITIONAL_OR_KEYWORD
        if section is _ArgumentSection.KEYWORD_ONLY:
            kind = IRArgumentKind.KEYWORD_ONLY

        return CArgument(
            name=name,
            type=arg_type,
            default_value=default_value,
            has_default=has_default,
            kind=kind,
        )

    def _validate_counts(self) -> None:
        """在解析结束后统一校验 Python 参数和 C 槽位计数。"""
        if self._python_arg_index != len(self._kwlist):
            raise PyArgParseTupleAndKeywordsTypeParserError(
                f"期望 {self._python_arg_index} 个 keyword names，实际找到 {len(self._kwlist)} 个。"
            )
        if self._arg_index != len(self._args):
            raise PyArgParseTupleAndKeywordsTypeParserError(
                f"期望 {self._arg_index} 个 C arguments，实际找到 {len(self._args)} 个。"
            )

    def _advance_format_unit_required(self) -> _FormatUnitSpec:
        """按最长匹配规则消费一个格式单元。"""
        current = self._peek_char_required()
        if current in "()[]{}":
            raise PyArgParseTupleAndKeywordsTypeParserError(
                f"索引 {self._char_index} 处的 format structure {current!r} 不受支持。"
            )

        for spec in _FORMAT_UNIT_SPECS:
            if self._format.startswith(spec.unit, self._char_index):
                self._char_index += len(spec.unit)
                return spec

        raise PyArgParseTupleAndKeywordsTypeParserError(
            f"索引 {self._char_index} 处的 format unit {current!r} 不受支持。"
        )

    def _peek_char(self) -> str | None:
        """查看当前位置字符而不推进游标。"""
        if self._char_index >= len(self._format):
            return None
        return self._format[self._char_index]

    def _peek_char_required(self) -> str:
        """查看当前位置字符；若已结束则抛错。"""
        current = self._peek_char()
        if current is None:
            raise PyArgParseTupleAndKeywordsTypeParserError("已到达 format string 末尾。")
        return current

    def _advance_char(self) -> str | None:
        """推进一个格式串字符。"""
        current = self._peek_char()
        if current is None:
            return None
        self._char_index += 1
        return current

    def _skip_separators(self) -> None:
        """跳过格式串中的空白与逗号分隔符。"""
        while True:
            current = self._peek_char()
            if current is None or current not in " \t,":
                return
            self._char_index += 1

    def _advance_keyword_name_required(self) -> str:
        """消费一个 Python 关键字参数名。"""
        if self._python_arg_index >= len(self._kwlist):
            raise PyArgParseTupleAndKeywordsTypeParserError(
                f"期望在索引 {self._python_arg_index} 处取得 keyword name，但已没有剩余项。"
            )

        name = self._kwlist[self._python_arg_index]
        self._python_arg_index += 1
        return name

    def _advance_c_args_required(self, count: int) -> list[Cursor]:
        """消费指定数量的 C 参数槽位。"""
        end_index = self._arg_index + count
        if end_index > len(self._args):
            raise PyArgParseTupleAndKeywordsTypeParserError(
                f"期望从索引 {self._arg_index} 开始取得 {count} 个 C arguments，但已没有剩余参数。"
            )

        values = self._args[self._arg_index:end_index]
        self._arg_index = end_index
        return values

    def _resolve_object_type(self, cursor: Cursor) -> Type:
        """解析对象单元的 Python 类型，未知时回退为 `object`。"""
        if self._resolve_object_type_func is not None:
            resolved_type = self._resolve_object_type_func(cursor)
            if resolved_type is not None:
                if isinstance(resolved_type, str):
                    return RawType(resolved_type)
                return resolved_type
        return RawType("object")

    def _resolve_default_value(self, cursor: Cursor) -> str | None:
        """解析可选参数的默认值文本，无法解析时保留为未知。"""
        if self._resolve_default_value_func is None:
            return None
        return self._resolve_default_value_func(cursor)
