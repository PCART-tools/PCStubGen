from __future__ import annotations

from typing import Any

from ...IR import (
    IRModule, IRFunction, IRVariable, IRImport, QualifiedName, ResolvedType, IRValue
)
from ..NodeVisitor import NodeVisitor

class FixMissingFixedSizeImportVisitor(NodeVisitor):
    def __init__(self):
        self._current_module: IRModule | None = None
        
    def visit_module(self, node: IRModule) -> None:
        self._current_module = node
        super().visit_module(node)
        self._current_module = None
        
    def visit_function(self, node: IRFunction) -> None:
        if node.return_annotation:
             self._check_annotation(node.return_annotation)
        for arg in node.args:
             if arg.annotation:
                 self._check_annotation(arg.annotation)
        super().visit_function(node)

    def visit_variable(self, node: IRVariable) -> None:
        if node.annotation:
             self._check_annotation(node.annotation)
        super().visit_variable(node)

    def _check_annotation(self, annotation: Any) -> None:
        if isinstance(annotation, ResolvedType):
            if annotation.parameters:
                for param in annotation.parameters:
                    self._check_annotation(param)
        elif isinstance(annotation, IRValue):
            if "FixedSize" in annotation.repr or "DynamicSize" in annotation.repr:
                if self._current_module:
                    self._current_module.imports.add(
                        IRImport(
                            name="FixedSize",
                            origin=QualifiedName.from_str(
                                "pybind11_stubgen.typing_ext.FixedSize"
                            ),
                        )
                    )
                    self._current_module.imports.add(
                        IRImport(
                            name="DynamicSize",
                            origin=QualifiedName.from_str(
                                "pybind11_stubgen.typing_ext.DynamicSize"
                            ),
                        )
                    )
