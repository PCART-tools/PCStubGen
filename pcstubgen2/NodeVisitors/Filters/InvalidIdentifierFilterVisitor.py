from __future__ import annotations

import re

from ...IR import IRClass, IRModule
from ..NodeVisitor import NodeVisitor


class InvalidIdentifierFilterVisitor(NodeVisitor):
    def __init__(self, ignore_regex: str | None = None):
        self.regex = re.compile(ignore_regex) if ignore_regex else None

    def visit_module(self, node: IRModule) -> None:
        node.variables = [v for v in node.variables if self._is_valid(v.name)]

        node.classes = [c for c in node.classes if self._is_valid(c.name)]
        node.functions = [f for f in node.functions if self._is_valid(f.name)]

        super().visit_module(node)

    def visit_class(self, node: IRClass) -> None:
        node.methods = [m for m in node.methods if self._is_valid(m.function.name)]
        node.properties = [p for p in node.properties if self._is_valid(p.name)]
        node.fields = [f for f in node.fields if self._is_valid(f.variable.name)]
        super().visit_class(node)

    def _is_valid(self, name: str) -> bool:
        return name.isidentifier()
