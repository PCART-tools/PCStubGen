from __future__ import annotations

from typing import Sequence

from .ir import IRModule
from .node_visitors.node_visitor import NodeVisitor


class Pipeline:
    def __init__(self, visitors: Sequence[NodeVisitor]):
        self.visitors = visitors

    def run(self, module: IRModule) -> IRModule:
        for visitor in self.visitors:
            visitor.visit_module(module)
        return module
