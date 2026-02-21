from __future__ import annotations

import logging
import re

from ...IR import IRVariable, IRFunction, InvalidExpression, IRValue
from ..NodeVisitor import NodeVisitor

logger = logging.getLogger("pcstubgen2")

class RewritePybind11EnumValueReprVisitor(NodeVisitor):
    """
    重写 pybind11 枚举值表示。
    
    将类似 `<Enum.Value: 1>` 的模式转换为 `Enum.Value`。
    """
    
    _pybind11_enum_pattern = re.compile(r"<(?P<enum>\w+(\.\w+)+): (?P<value>-?\d+)>")
    
    def __init__(self, enum_class_locations: dict[re.Pattern, str] | None = None):
        self.enum_class_locations = enum_class_locations or {}
        self._unknown_enum_classes: set[str] = set()
    
    def visit_variable(self, node: IRVariable) -> None:
        if node.value:
            node.value = self._rewrite_value(node.value)
        super().visit_variable(node)
    
    def visit_function(self, node: IRFunction) -> None:
        for arg in node.args:
            if arg.default and isinstance(arg.default, IRValue):
                arg.default = self._rewrite_value(arg.default)
            elif arg.default and isinstance(arg.default, InvalidExpression):
                arg.default = self._rewrite_invalid_expression(arg.default)
        super().visit_function(node)
    
    def _rewrite_value(self, value: IRValue) -> IRValue:
        rewritten = self._resolve_enum_repr(value.repr)
        if rewritten is not None:
            return rewritten
        return value

    def _rewrite_invalid_expression(
        self, invalid: InvalidExpression
    ) -> IRValue | InvalidExpression:
        rewritten = self._resolve_enum_repr(invalid.text)
        if rewritten is not None:
            return rewritten
        return invalid

    def _resolve_enum_repr(self, raw_text: str) -> IRValue | None:
        match = self._pybind11_enum_pattern.match(raw_text.strip())
        if not match:
            return None

        enum_qual_name = match.group("enum")
        class_path, entry = enum_qual_name.rsplit(".", maxsplit=1)
        for pattern, prefix in self.enum_class_locations.items():
            if pattern.match(class_path):
                return IRValue(repr=f"{prefix}.{class_path}.{entry}", is_print_safe=True)

        self._unknown_enum_classes.add(class_path)
        return None

    def finalize(self) -> None:
        if self._unknown_enum_classes:
            logger.warning(
                "Enum-like str representations were found with no "
                "matching mapping to the enum class location.\n"
                "Use `--enum-class-locations` to specify "
                "full path to the following enum(s):\n"
                + "\n".join(f" - {c}" for c in self._unknown_enum_classes)
            )
