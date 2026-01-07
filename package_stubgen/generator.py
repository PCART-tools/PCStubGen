from __future__ import annotations

from typing import TYPE_CHECKING
from .utils import quote_docstring, is_private_name, is_not_in_all

if TYPE_CHECKING:
    from .models import ModuleStubData, ClassStubData, VariableInfo, PropertyInfo
    from .stubdoc import FunctionSig, ArgSig

class StubGenerator:
    """将 ModuleStubData 树形结构转换为 .pyi 字符串。"""

    def __init__(
        self, include_private: bool = False, include_docstrings: bool = False
    ) -> None:
        self.include_private = include_private
        self.include_docstrings = include_docstrings
        self._indent = ""

    def generate_module(self, module: ModuleStubData) -> str:
        lines: list[str] = []
        
        # 导入
        if module.imports:
            lines.extend(module.imports)
            lines.append("")

        # __all__
        if module._all_ is not None:
            lines.append(f"__all__ = {module._all_!r}")
            lines.append("")

        # 模块文档字符串
        if self.include_docstrings and module.docstring:
            lines.append(quote_docstring(module.docstring))
            lines.append("")

        # 变量
        variables = [
            v for v in module.variables
            if not is_private_name(v.name, module._all_, self.include_private)
            and not is_not_in_all(v.name, module._all_, is_top_level=True)
        ]
        for var in variables:
            lines.append(f"{var.name}: {var.type}")
        
        if variables and (module.classes or module.functions):
            lines.append("")

        # 类
        classes = [
            c for c in module.classes
            if not is_private_name(c.name, module._all_, self.include_private)
            and not is_not_in_all(c.name, module._all_, is_top_level=True)
        ]
        for cls in classes:
            lines.append(self.generate_class(cls, module_all=module._all_))
            lines.append("")

        # 函数
        functions = [
            f for f in module.functions
            if not is_private_name(f.name, module._all_, self.include_private)
            and not is_not_in_all(f.name, module._all_, is_top_level=True)
        ]
        for func in functions:
            lines.append(self.generate_function(func))

        return "\n".join(lines).strip() + "\n"

    def generate_class(self, cls: ClassStubData, module_all: list[str] | None = None) -> str:
        bases_str = f"({', '.join(cls.bases)})" if cls.bases else ""
        header = f"{self._indent}class {cls.name}{bases_str}:"
        
        old_indent = self._indent
        self._indent += "    "
        
        body: list[str] = []
        
        # 文档字符串
        if self.include_docstrings and cls.docstring:
            body.append(f"{self._indent}{quote_docstring(cls.docstring)}")

        # 嵌套类
        # 类成员不使用 module_all 过滤，且 is_top_level=False
        nested_classes = [
            c for c in cls.classes
            if not is_private_name(c.name, include_private=self.include_private)
        ]
        for nested_cls in nested_classes:
            body.append(self.generate_class(nested_cls))

        # 类变量
        variables = [
            v for v in cls.variables
            if not is_private_name(v.name, include_private=self.include_private)
        ]
        for var in variables:
            body.append(f"{self._indent}{var.name}: {var.type} = ...")

        # 属性
        properties = [
            p for p in cls.properties
            if not is_private_name(p.name, include_private=self.include_private)
        ]
        for prop in properties:
            body.append(self.generate_property(prop))

        # 方法
        methods = [
            m for m in cls.methods
            if not is_private_name(m.name, include_private=self.include_private)
        ]
        for method in methods:
            body.append(self.generate_function(method))

        self._indent = old_indent
        
        if not body:
            return f"{header} ..."
        
        return f"{header}\n" + "\n".join(body)

    def generate_function(self, func: FunctionSig) -> str:
        # 注意：FunctionSig.format_sig 已经处理了大部分逻辑，
        # 但我们需要确保它符合我们的缩进和文档字符串设置。
        # 这里的实现参考了原来的 format_sig，但更灵活。
        
        # 暂时直接调用 format_sig，因为它已经支持 indent 和 include_docstrings
        return func.format_sig(
            indent=self._indent,
            include_docstrings=self.include_docstrings
        )

    def generate_property(self, prop: PropertyInfo) -> str:
        if prop.readonly:
            lines = [
                f"{self._indent}@property",
                f"{self._indent}def {prop.name}(self) -> {prop.type}: ..."
            ]
            return "\n".join(lines)
        else:
            return f"{self._indent}{prop.name}: {prop.type}"
