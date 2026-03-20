from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .errors import (
    InvalidExpressionError,
    InvalidIdentifierError,
    NameResolutionError,
    ParserError,
)

if TYPE_CHECKING:
    from .ir import QualifiedName

logger = logging.getLogger(__name__)


@dataclass
class ErrorCollector:
    """在存根生成期间收集并上报错误。"""

    errors: list[ParserError] = field(default_factory=list)
    _reported: set[str] = field(default_factory=set)
    _current_path: QualifiedName | None = field(default=None)

    # 错误过滤选项
    ignore_invalid_expressions: re.Pattern | None = field(default=None)
    ignore_all_errors: bool = field(default=False)

    def set_current_path(self, path: QualifiedName) -> None:
        """设置错误上报时使用的当前路径上下文。"""
        self._current_path = path

    def report_error(self, error: ParserError) -> None:
        """上报错误，并避免重复记录。"""
        if self.ignore_all_errors:
            return

        if isinstance(error, NameResolutionError):
            if len(error.name) > 0 and error.name[0] in {
                "module",
                "pybind11_builtins",
                "PyCapsule",
            }:
                return

        if isinstance(error, InvalidIdentifierError):
            name = error.name
            if (
                name.startswith("ItemsView[")
                and name.endswith("]")
                or name.startswith("KeysView[")
                and name.endswith("]")
                or name.startswith("ValuesView[")
                and name.endswith("]")
            ):
                return

        if isinstance(error, InvalidExpressionError):
            if (
                self.ignore_invalid_expressions
                and self.ignore_invalid_expressions.match(error.expression)
            ):
                return

        if self._current_path:
            error_str = f"In {self._current_path} : {error}"
        else:
            error_str = str(error)

        if error_str not in self._reported:
            self.errors.append(error)
            self._reported.add(error_str)
            logger.error(error_str)

    def has_errors(self) -> bool:
        """检查是否存在已上报错误。"""
        return len(self.errors) > 0

    def clear(self) -> None:
        """清空所有已收集错误。"""
        self.errors.clear()
        self._reported.clear()
