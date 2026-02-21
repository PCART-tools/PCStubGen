from __future__ import annotations

import re

from ...IR import IRVariable, IRFunction, IRValue
from ..NodeVisitor import NodeVisitor

class OverridePrintSafeValuesVisitor(NodeVisitor):
    def __init__(self, pattern: re.Pattern | None):
        self.pattern = pattern

    def visit_variable(self, node: IRVariable) -> None:
        if self.pattern is None:
            super().visit_variable(node)
            return
        if node.value:
            self._check_value(node.value)
        super().visit_variable(node)

    def visit_function(self, node: IRFunction) -> None:
        if self.pattern is None:
            return
        for arg in node.args:
            if arg.default and isinstance(arg.default, IRValue):
                self._check_value(arg.default)
        super().visit_function(node)

    def _check_value(self, value: IRValue) -> None:
        if not value.is_print_safe and self.pattern.match(value.repr):
            value.is_print_safe = True
