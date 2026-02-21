from __future__ import annotations

from typing import Sequence

from .IR import IRModule
from .NodeVisitors.NodeVisitor import NodeVisitor


class Pipeline:
    def __init__(self, visitors: Sequence[NodeVisitor]):
        self.visitors = visitors

    def run(self, module: IRModule) -> IRModule:
        for visitor in self.visitors:
            visitor.visit_module(module)
        for visitor in self.visitors:
            visitor.finalize()
        return module
