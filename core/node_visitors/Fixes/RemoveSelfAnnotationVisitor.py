from __future__ import annotations

from ...ir import (
    IRModule, IRClass, IRMethod, IRFunction, ResolvedType, QualifiedName
)
from ..NodeVisitor import NodeVisitor

class RemoveSelfAnnotationVisitor(NodeVisitor):
    """移除 'self' 参数的类型注释。"""
    
    __any_names = {
        QualifiedName.from_str("Any"),
        QualifiedName.from_str("typing.Any"),
    }
    
    def __init__(self):
        self._current_class_name: QualifiedName | None = None
        self._class_name_stack: list[QualifiedName] = []
    
    def visit_module(self, node: IRModule) -> None:
        # 访问新模块时重置类栈
        self._class_name_stack = [node.full_name]
        super().visit_module(node)
        self._class_name_stack = []
    
    def visit_class(self, node: IRClass, module: IRModule) -> None:
        # 通过附加到当前路径来构建完整的类名
        if self._class_name_stack:
            parent_path = self._class_name_stack[-1]
            self._current_class_name = parent_path.concat(node.name)
        else:
            self._current_class_name = QualifiedName((node.name,))
        
        self._class_name_stack.append(self._current_class_name)
        super().visit_class(node, module)
        self._class_name_stack.pop()
        
        if self._class_name_stack:
            self._current_class_name = self._class_name_stack[-1]
        else:
            self._current_class_name = None
    
    def visit_method(self, node: IRMethod) -> None:
        self._remove_self_arg_annotation(node.function)
        super().visit_method(node)
    
    def _remove_self_arg_annotation(self, func: IRFunction) -> None:
        if len(func.args) == 0:
            return
        first_arg = func.args[0]
        if (
            first_arg.name == "self"
            and isinstance(first_arg.annotation, ResolvedType)
            and not first_arg.annotation.parameters
        ):
            annotation_name = first_arg.annotation.name
            
            # 检查各种模式：
            # 1. Any 或 typing.Any
            # 2. 完全限定的类名
            # 3. 部分匹配（完整类名的后缀）
            should_remove = False
            
            if annotation_name in self.__any_names:
                should_remove = True
            elif self._current_class_name is not None:
                if annotation_name == self._current_class_name:
                    should_remove = True
                elif len(annotation_name) <= len(self._current_class_name):
                    # 检查注释是否为类名的后缀
                    suffix = self._current_class_name[-len(annotation_name):]
                    if annotation_name == suffix:
                        should_remove = True
            
            if should_remove:
                first_arg.annotation = None
