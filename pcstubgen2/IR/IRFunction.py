from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .IRArgument import IRArgumentKind

if TYPE_CHECKING:
    from .IRAnnotation import IRAnnotation
    from .IRArgument import IRArgument


@dataclass
class IRFunction:
    name: str
    args: list[IRArgument] = field(default_factory=list)
    returns: IRAnnotation | None = field(default=None)
    doc: str | None = field(default=None)
    decorators: list[str] = field(default_factory=list)

    def is_generic_signature(self) -> bool:
        """检查函数是否具有泛型 (*args, **kwargs) 签名。"""
        if len(self.args) != 2:
            return False
        return (
            self.args[0].kind is IRArgumentKind.VAR_POSITIONAL
            and self.args[0].name == "args"
            and self.args[1].kind is IRArgumentKind.VAR_KEYWORD
            and self.args[1].name == "kwargs"
        )

    def __str__(self) -> str:
        return (
            f"{self.name}({', '.join(str(arg) for arg in self.args)}) -> {self.returns}"
        )
