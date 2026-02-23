from __future__ import annotations

from ...IR import (
    IRClass, IRArgument, ResolvedType, QualifiedName
)
from ..NodeVisitor import NodeVisitor

class FixPybind11EnumStrDocVisitor(NodeVisitor):
    """修复 pybind11 枚举类中的 __str__ 方法签名。"""
    
    def visit_class(self, node: IRClass) -> None:
        # 检查这是否是一个枚举类（具有 __members__ 属性）
        is_enum = any(
            f.variable.name == "__members__" for f in node.fields
        )
        
        if is_enum:
            for method in node.methods:
                if (
                    method.function.name == "__str__"
                    and method.function.doc == "name(self: handle) -> str\n"
                ):
                    # 修复签名
                    method.function.args = [
                        IRArgument(name="self")
                    ]
                    method.function.return_annotation = ResolvedType(
                        name=QualifiedName.from_str("str")
                    )
                    # 修复修饰符（由于泛型签名曾是 static）
                    method.modifier = None
                    method.function.doc = None
        
        super().visit_class(node)
