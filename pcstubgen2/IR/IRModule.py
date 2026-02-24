from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .IRClass import IRClass
    from .IRFunction import IRFunction
    from .QualifiedName import QualifiedName

IRModuleType = Literal["python", "builtin", "c"]


@dataclass
class IRModule:
    full_name: QualifiedName

    # 文档
    doc: str | None = field(default=None)

    # 模块实现类型
    module_type: IRModuleType = field(default="python")

    # 类
    classes: list[IRClass] = field(default_factory=list)

    # 函数
    functions: list[IRFunction] = field(default_factory=list)

    # 子模块
    sub_modules: list[IRModule] = field(default_factory=list)

    # 是否是包
    is_package: bool = field(default=False)

    @property
    def Name(self) -> str:
        return self.full_name.name
