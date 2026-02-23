from __future__ import annotations

import typing

from ...IR import IRModule
from ..NodeVisitor import NodeVisitor


class FilterTypingModuleAttributesVisitor(NodeVisitor):
    __typing_sentinel = object()
    __typing_never_filter = {"__all__"}

    def visit_module(self, node: IRModule) -> None:
        node.variables = [
            variable
            for variable in node.variables
            if not self._is_typing_module_attr(variable.name, variable.runtime_value)
        ]

        super().visit_module(node)

    def _is_typing_module_attr(self, name: str, member: object | None) -> bool:
        if str(name) in self.__typing_never_filter:
            return False
        return getattr(typing, str(name), self.__typing_sentinel) is member
