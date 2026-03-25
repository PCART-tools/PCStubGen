from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ir_class import IRClass
    from .ir_function import IRFunction
    from .qualified_name import QualifiedName

class IRModuleType(Enum):
    UNKNOWN = "unknown"
    PYTHON = "python"
    BUILTIN = "builtin"
    EXTENSION = "extension"


@dataclass
class IRModule:
    full_name: QualifiedName

    # 文档
    doc: str | None = field(default=None)

    # 模块实现类型
    module_type: IRModuleType = field(default=IRModuleType.UNKNOWN)

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
