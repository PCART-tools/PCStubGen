from __future__ import annotations

from ...IR import IRModule, IRVariable, IRValue, ResolvedType, QualifiedName
from ..NodeVisitor import NodeVisitor

class FixMissingAllVisitor(NodeVisitor):
    """如果不存在，生成 __all__。"""
    
    def visit_module(self, node: IRModule) -> None:
        # 不要覆盖现有的 __all__
        if node.all is not None:
            super().visit_module(node)
            return
        
        # 收集候选名称（排除 __future__ 导入）
        candidate_names = [
            *(class_.name for class_ in node.classes),
            *(variable.name for variable in node.variables),
            *(func.name for func in node.functions),
            *(alias.name for alias in node.aliases),
            *(
                import_.name
                for import_ in node.imports
                if import_.name is not None
                and (len(import_.origin) == 0 or import_.origin[0] != "__future__")
            ),
            *(sub_module.Name for sub_module in node.sub_modules),
        ]

        # 保留公共名称并稳定排序
        all_names: list[str] = sorted(
            {name for name in candidate_names if not name.startswith("_")}
        )
        
        # 创建 __all__ 属性
        all_value = "list()" if not all_names else repr(all_names)
        node.all = (
            IRVariable(
                name="__all__",
                value=IRValue(repr=all_value, is_print_safe=True),
                annotation=ResolvedType(
                    name=QualifiedName.from_str("list"),
                    parameters=[ResolvedType(name=QualifiedName.from_str("str"))],
                ),
            )
        )
        
        super().visit_module(node)
