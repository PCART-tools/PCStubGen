from __future__ import annotations

"""`PyArg_ParseTupleAndKeywords` 格式串到参数信息的解析器。"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from clang.cindex import Cursor

from ....ir import IRArgumentKind
from .cursor_utils import looks_like_identifier
from .models import ExtractedArgument


class PyArgParseTupleAndKeywordsTypeParserError(ValueError):
    """表示 `PyArg_ParseTupleAndKeywords` 格式串无法被当前解析器接受。"""


class _ArgumentSection(Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    KEYWORD_ONLY = "keyword_only"


@dataclass(frozen=True)
class _FormatUnitSpec:
    """描述单个格式单元如何映射到 Python 参数。"""

    # format unit
    unit: str

    # 对应Python类型
    type_name: str

    # 读取c参数数量
    c_arg_count: int

    # 默认值参数在c参数中的位置
    default_arg_offset: int

    # 处理O! O&要从c参数中resolve类型，在c参数中的位置
    object_type_arg_offset: int | None = None


_FORMAT_UNIT_SPECS: tuple[_FormatUnitSpec, ...] = (
    _FormatUnitSpec("es#", "str", 3, 1),
    _FormatUnitSpec("et#", "str | bytes | bytearray", 3, 1),
    _FormatUnitSpec("s*", "str | collections.abc.Buffer", 1, 0),
    _FormatUnitSpec("s#", "str | collections.abc.Buffer", 2, 0),
    _FormatUnitSpec("z*", "str | collections.abc.Buffer | None", 1, 0),
    _FormatUnitSpec("z#", "str | collections.abc.Buffer | None", 2, 0),
    _FormatUnitSpec("y*", "collections.abc.Buffer", 1, 0),
    _FormatUnitSpec("y#", "collections.abc.Buffer", 2, 0),
    _FormatUnitSpec("es", "str", 2, 1),
    _FormatUnitSpec("et", "str | bytes | bytearray", 2, 1),
    _FormatUnitSpec("w*", "collections.abc.Buffer", 1, 0),
    _FormatUnitSpec("O!", "object", 2, 1, object_type_arg_offset=0),
    _FormatUnitSpec("O&", "object", 2, 1, object_type_arg_offset=0),
    _FormatUnitSpec("s", "str", 1, 0),
    _FormatUnitSpec("z", "str | None", 1, 0),
    _FormatUnitSpec("y", "collections.abc.Buffer", 1, 0),
    _FormatUnitSpec("S", "bytes", 1, 0),
    _FormatUnitSpec("Y", "bytearray", 1, 0),
    _FormatUnitSpec("U", "str", 1, 0),
    _FormatUnitSpec("b", "int", 1, 0),
    _FormatUnitSpec("B", "int", 1, 0),
    _FormatUnitSpec("h", "int", 1, 0),
    _FormatUnitSpec("H", "int", 1, 0),
    _FormatUnitSpec("i", "int", 1, 0),
    _FormatUnitSpec("I", "int", 1, 0),
    _FormatUnitSpec("l", "int", 1, 0),
    _FormatUnitSpec("k", "int", 1, 0),
    _FormatUnitSpec("L", "int", 1, 0),
    _FormatUnitSpec("K", "int", 1, 0),
    _FormatUnitSpec("n", "int", 1, 0),
    _FormatUnitSpec("c", "bytes | bytearray", 1, 0),
    _FormatUnitSpec("C", "str", 1, 0),
    _FormatUnitSpec("f", "float", 1, 0),
    _FormatUnitSpec("d", "float", 1, 0),
    _FormatUnitSpec("D", "complex", 1, 0),
    _FormatUnitSpec("O", "object", 1, 0),
    _FormatUnitSpec("p", "object", 1, 0),
)


class PyArgParseTupleAndKeywordsTypeParser:
    """将 `PyArg_ParseTupleAndKeywords` 的格式串解析为提取参数列表。"""

    def __init__(
        self,
        fmt: str,
        kwlist: list[str],
        args: list[Cursor],
        resolve_object_type_func: Callable[[Cursor], str | None] | None = None,
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

    def parse(self) -> list[ExtractedArgument]:
        """解析格式串并返回参数列表。"""
        self._char_index = 0
        self._arg_index = 0
        self._python_arg_index = 0
        self._validate_kwlist()

        arguments: list[ExtractedArgument] = []
        section = _ArgumentSection.REQUIRED

        while True:
            self._skip_separators()
            current = self._peek_char()

            if current is None or current in ":;":
                break

            if current == "|":
                if section is _ArgumentSection.OPTIONAL:
                    raise PyArgParseTupleAndKeywordsTypeParserError(
                        "Found duplicate '|' in format string."
                    )
                if section is _ArgumentSection.KEYWORD_ONLY:
                    raise PyArgParseTupleAndKeywordsTypeParserError(
                        "Found '|' after '$' in format string."
                    )
                self._advance_char()
                section = _ArgumentSection.OPTIONAL
                continue

            if current == "$":
                if section is _ArgumentSection.REQUIRED:
                    raise PyArgParseTupleAndKeywordsTypeParserError(
                        "Found '$' before '|'."
                    )
                if section is _ArgumentSection.KEYWORD_ONLY:
                    raise PyArgParseTupleAndKeywordsTypeParserError(
                        "Found duplicate '$' in format string."
                    )
                self._advance_char()
                section = _ArgumentSection.KEYWORD_ONLY
                continue

            arguments.append(self._parse_argument(section))

        self._validate_counts()
        return arguments

    def _parse_argument(self, section: _ArgumentSection) -> ExtractedArgument:
        """解析一个格式单元并产出单个 Python 参数。"""
        name = self._advance_keyword_name_required()
        spec = self._advance_format_unit_required()
        c_args = self._advance_c_args_required(spec.c_arg_count)

        type_name = spec.type_name
        if spec.object_type_arg_offset is not None:
            type_name = self._resolve_object_type(c_args[spec.object_type_arg_offset])

        has_default = section is not _ArgumentSection.REQUIRED

        default_value: str | None = None
        if has_default:
            default_value = self._resolve_default_value(c_args[spec.default_arg_offset])

        kind = IRArgumentKind.POSITIONAL_OR_KEYWORD
        if section is _ArgumentSection.KEYWORD_ONLY:
            kind = IRArgumentKind.KEYWORD_ONLY

        return ExtractedArgument(
            name=name,
            type_name=type_name,
            default_value=default_value,
            has_default=has_default,
            kind=kind,
        )

    def _validate_kwlist(self) -> None:
        """校验关键字名列表是否满足唯一且合法的约束。"""
        seen: set[str] = set()
        for keyword_name in self._kwlist:
            if not keyword_name or not looks_like_identifier(keyword_name):
                raise PyArgParseTupleAndKeywordsTypeParserError(
                    f"Invalid keyword name {keyword_name!r}."
                )
            if keyword_name in seen:
                raise PyArgParseTupleAndKeywordsTypeParserError(
                    f"Duplicate keyword name {keyword_name!r}."
                )
            seen.add(keyword_name)

    def _validate_counts(self) -> None:
        """在解析结束后统一校验 Python 参数和 C 槽位计数。"""
        if self._python_arg_index != len(self._kwlist):
            raise PyArgParseTupleAndKeywordsTypeParserError(
                f"Expected {self._python_arg_index} keyword names, found {len(self._kwlist)}."
            )
        if self._arg_index != len(self._args):
            raise PyArgParseTupleAndKeywordsTypeParserError(
                f"Expected {self._arg_index} C arguments, found {len(self._args)}."
            )

    def _advance_format_unit_required(self) -> _FormatUnitSpec:
        """按最长匹配规则消费一个格式单元。"""
        current = self._peek_char_required()
        if current in "()[]{}":
            raise PyArgParseTupleAndKeywordsTypeParserError(
                f"Unsupported format structure {current!r} at index {self._char_index}."
            )

        for spec in _FORMAT_UNIT_SPECS:
            if self._format.startswith(spec.unit, self._char_index):
                self._char_index += len(spec.unit)
                return spec

        raise PyArgParseTupleAndKeywordsTypeParserError(
            f"Unsupported format unit {current!r} at index {self._char_index}."
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
            raise PyArgParseTupleAndKeywordsTypeParserError("Found end of format string.")
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
                f"Expected keyword name at index {self._python_arg_index}, but none remained."
            )

        name = self._kwlist[self._python_arg_index]
        self._python_arg_index += 1
        return name

    def _advance_c_args_required(self, count: int) -> list[Cursor]:
        """消费指定数量的 C 参数槽位。"""
        end_index = self._arg_index + count
        if end_index > len(self._args):
            raise PyArgParseTupleAndKeywordsTypeParserError(
                f"Expected {count} C arguments starting at index {self._arg_index}, but none remained."
            )

        values = self._args[self._arg_index:end_index]
        self._arg_index = end_index
        return values

    def _resolve_object_type(self, cursor: Cursor) -> str:
        """解析对象单元的 Python 类型，未知时回退为 `object`。"""
        if self._resolve_object_type_func is not None:
            resolved_type = self._resolve_object_type_func(cursor)
            if resolved_type is not None:
                return resolved_type
        return "object"

    def _resolve_default_value(self, cursor: Cursor) -> str | None:
        """解析可选参数的默认值文本，无法解析时保留为未知。"""
        if self._resolve_default_value_func is None:
            return None
        return self._resolve_default_value_func(cursor)
