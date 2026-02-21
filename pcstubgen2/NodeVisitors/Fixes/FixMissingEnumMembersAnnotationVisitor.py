from __future__ import annotations

from ...IR import (
    IRClass, ResolvedType, QualifiedName
)
from ..NodeVisitor import NodeVisitor

class FixMissingEnumMembersAnnotationVisitor(NodeVisitor):
    """为枚举类中的 __members__ 字段添加正确的类型注释。"""
    
    __class_var_dict = ResolvedType(
        name=QualifiedName.from_str("typing.ClassVar"),
        parameters=[ResolvedType(name=QualifiedName.from_str("dict"))],
    )
    
    def visit_class(self, node: IRClass) -> None:
        # 具体 dict 类型在 inspection 阶段已基于真实值推断；
        # 这里保持不覆盖已有注解，避免引入猜测。
        super().visit_class(node)
    
    def _guess_dict_type_from_class(self, node: IRClass) -> ResolvedType | None:
        """根据枚举类字段猜测 dict 类型。"""
        # 对于枚举类，__members__ 通常是 Dict[str, EnumClass]
        # 我们返回 Dict[str, ClassName]
        return ResolvedType(
            name=QualifiedName.from_str("dict"),
            parameters=[
                ResolvedType(name=QualifiedName.from_str("str")),
                ResolvedType(name=QualifiedName((node.name,))),
            ],
        )
