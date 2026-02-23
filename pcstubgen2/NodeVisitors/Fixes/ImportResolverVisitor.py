from __future__ import annotations

import ast
import builtins
import importlib
import re
from typing import Any

from ...ErrorCollector import ErrorCollector
from ...IR import (
    IRModule, IRClass, IRFunction, IRVariable, IRImport, QualifiedName, IRValue
)
from ..NodeVisitor import NodeVisitor
from ...Errors import NameResolutionError

class ImportResolverVisitor(NodeVisitor):
    _qualified_name_regex = re.compile(r"([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+)")

    def __init__(self, error_collector: ErrorCollector):
        self.error_collector = error_collector
        self._extra_imports: set[IRImport] = set()
        self._current_module: IRModule | None = None
        self._current_class: IRClass | None = None
    
    def visit_module(self, node: IRModule) -> None:
        old_module = self._current_module
        old_imports = self._extra_imports
        self._current_module = node
        self._extra_imports = set()
        
        for alias in node.aliases:
            self._add_import(alias.origin)

        for import_ in node.imports:
            self._extra_imports.add(import_)

        # 递归
        super().visit_module(node)
        
        # 添加收集的导入
        node.imports.update(self._extra_imports)
        self._current_module = old_module
        self._extra_imports = old_imports

    def visit_class(self, node: IRClass) -> None:
        old_class = self._current_class
        self._current_class = node
        
        # 检查导入的基类
        for base in node.bases:
            self._add_import(base)
            
        super().visit_class(node)
        self._current_class = old_class

    def visit_function(self, node: IRFunction) -> None:
        if node.return_annotation:
             self._check_annotation(node.return_annotation)
        for arg in node.args:
             if arg.annotation:
                 self._check_annotation(arg.annotation)
             if arg.default:
                 self._check_value(arg.default)
        
        for decorator in node.decorators:
            for match in self._qualified_name_regex.finditer(decorator):
                name_str = match.group(1)
                self._add_import(QualifiedName.from_str(name_str))

        super().visit_function(node)

    def visit_variable(self, node: IRVariable) -> None:
        if node.annotation:
            self._check_annotation(node.annotation)
        if node.value:
            self._check_value(node.value)
        super().visit_variable(node)

    def _check_annotation(self, annotation: Any) -> None:
        from ...IR import ResolvedType
        if isinstance(annotation, ResolvedType):
            self._add_import(annotation.name)
            if annotation.parameters:
                for param in annotation.parameters:
                    self._check_annotation(param)

    def _check_value(self, value: Any) -> None:
        if isinstance(value, IRValue) and value.is_print_safe:
            try:
                expr = ast.parse(value.repr, mode="eval")
            except SyntaxError:
                return
            for name in self._extract_qualified_names(expr):
                self._add_import(name)
        return

    def _extract_qualified_names(self, node: ast.AST) -> list[QualifiedName]:
        result: list[QualifiedName] = []
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(node):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent

        for child in ast.walk(node):
            if not isinstance(child, ast.Attribute):
                continue
            if isinstance(parents.get(child), ast.Attribute):
                continue
            parts: list[str] = []
            current: ast.AST | None = child
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            else:
                continue
            full_name = ".".join(reversed(parts))
            result.append(QualifiedName.from_str(full_name))
        return result

    def _add_import(self, name: QualifiedName) -> None:
        if len(name) == 0:
            return
        
        if len(name) == 1 and len(name[0]) == 0:
            return

        # 跳过无效标识符（例如被解析出的泛型 "list[str]"）
        if any(not part.isidentifier() for part in name):
            return

        root = name[0]
        if not root: 
            return
        if root == "typing" and len(name) == 1:
            return
        # 检查 builtins
        if hasattr(builtins, root):
            return

        # 检查是否在当前类中定义
        if self._current_class is not None:
            if hasattr(self._current_class, root):
                return

        # 检查是否在当前模块中定义
        if self._current_module:
            # 检查类
            if any(c.name == root for c in self._current_module.classes):
                return
            # 检查函数
            if any(f.name == root for f in self._current_module.functions):
                return
            # 检查 __all__
            if (
                self._current_module.all is not None
                and self._current_module.all.name == root
            ):
                return
            # 检查变量
            if any(v.name == root for v in self._current_module.variables):
                return
            # 检查类型变量
            if any(tv.name == root for tv in self._current_module.type_vars):
                return
            # 检查别名
            if any(a.name == root for a in self._current_module.aliases):
                return
            # 检查现有的导入
            if any(
                (imp.name == root if imp.name else imp.origin.name == root)
                for imp in self._current_module.imports
            ):
                return
        
        # 尝试查找父模块
        module_name = self._get_parent_module(name)
        if module_name is None:
            # 检查它是否是顶级模块
            if self._is_module(name):
                self._extra_imports.add(IRImport(name=None, origin=name))
                return

            self.error_collector.report_error(NameResolutionError(name))
            return
        
        self._extra_imports.add(IRImport(name=None, origin=module_name))

    def _get_parent_module(self, name: QualifiedName) -> QualifiedName | None:
        """查找限定名称的父模块。"""
        parent = name.parent
        while len(parent) != 0:
            if self._is_module(parent):
                if self._is_accessible(name, from_module=parent):
                    return parent
                return None
            parent = parent.parent
        return None

    def _is_module(self, name: QualifiedName) -> bool:
        """检查名称是否引用可导入的模块。"""
        try:
            return importlib.import_module(str(name)) is not None
        except (ModuleNotFoundError, ImportError):
            return False

    def _is_accessible(self, name: QualifiedName, from_module: QualifiedName) -> bool:
        """检查名称是否可以从模块访问。"""
        try:
            parent = importlib.import_module(str(from_module))
        except (ModuleNotFoundError, ImportError):
            return False
        relative_path = name[len(from_module):]
        for part in relative_path:
            if not hasattr(parent, part):
                return False
            parent = getattr(parent, part)
        return True
