from __future__ import annotations

from typing import Any

from ...IR import (
    IRModule, IRClass, IRFunction, ResolvedType, QualifiedName, IRValue
)
from ..NodeVisitor import NodeVisitor

class FixCurrentModulePrefixInTypeNamesVisitor(NodeVisitor):
    """从类型名称中剥离当前模块前缀。"""
    
    def __init__(self):
        self._current_module_name: QualifiedName = QualifiedName()
    
    def visit_module(self, node: IRModule) -> None:
        old_name = self._current_module_name
        self._current_module_name = node.full_name
        super().visit_module(node)
        self._current_module_name = old_name
    
    def visit_class(self, node: IRClass, module: IRModule) -> None:
        # 修复基类
        new_bases = []
        for base in node.bases:
             new_bases.append(self._strip_current_module(base))
        node.bases = new_bases
        super().visit_class(node, module)

    def visit_function(self, node: IRFunction) -> None:
        if node.return_annotation:
            node.return_annotation = self._fix_type(node.return_annotation)
        for arg in node.args:
            if arg.annotation:
                arg.annotation = self._fix_type(arg.annotation)
            if isinstance(arg.default, IRValue):
                self._strip_value_repr(arg.default)
        super().visit_function(node)
    
    def _fix_type(self, annotation: Any) -> Any:
        if not isinstance(annotation, ResolvedType):
            return annotation
        
        annotation.name = self._strip_current_module(annotation.name)
        
        # 递归修复参数
        if annotation.parameters:
            annotation.parameters = [
                self._fix_type(p) for p in annotation.parameters
            ]
        
        return annotation
    
    def _strip_current_module(self, name: QualifiedName) -> QualifiedName:
        if len(self._current_module_name) == 0:
            return name
        prefix_len = len(self._current_module_name)
        if name[:prefix_len] == self._current_module_name:
            return QualifiedName(name[prefix_len:])
        return name

    def _strip_value_repr(self, value: IRValue) -> None:
        if not value.is_print_safe:
            return
        if "." not in value.repr:
            return
        name = QualifiedName.from_str(value.repr)
        if any(not part.isidentifier() for part in name):
            return
        stripped = self._strip_current_module(name)
        if stripped != name:
            value.repr = str(stripped)
