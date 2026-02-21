from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .IRAlias import IRAlias
    from .IRVariable import IRVariable
    from .IRClass import IRClass
    from .IRFunction import IRFunction
    from .IRImport import IRImport
    from .QualifiedName import QualifiedName
    from .IRTypeVar import IRTypeVar


@dataclass
class IRModule:
    full_name: QualifiedName

    # 文档
    doc: str | None = field(default=None)

    # 导入
    imports: set[IRImport] = field(default_factory=set)
    
    # __all__
    all: IRVariable | None = field(default=None)

    # 类
    classes: list[IRClass] = field(default_factory=list)

    # 函数
    functions: list[IRFunction] = field(default_factory=list)

    # 子模块
    sub_modules: list[IRModule] = field(default_factory=list)

    # 变量
    variables: list[IRVariable] = field(default_factory=list)

    # 别名
    aliases: list[IRAlias] = field(default_factory=list)

    # 类型变量
    type_vars: list[IRTypeVar] = field(default_factory=list)

    # 是否是包
    is_package: bool = field(default=False)

    @property
    def Name(self) -> str:
        return self.full_name.name
