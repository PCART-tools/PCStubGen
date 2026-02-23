from __future__ import annotations

from ...IR import IRModule
from ..NodeVisitor import NodeVisitor


class FilterPybind11ViewClassesVisitor(NodeVisitor):
    '''
    过滤掉pybind11视图类
    '''
    __view_classes = {
        "ItemsView",
        "KeysView",
        "ValuesView",
    }

    def visit_module(self, node: IRModule) -> None:
        node.classes = [
            class_ for class_ in node.classes if class_.name not in self.__view_classes
        ]

        super().visit_module(node)
