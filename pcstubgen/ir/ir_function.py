from __future__ import annotations

from dataclasses import dataclass, field

from .ir_signature import IRSignature


@dataclass
class IRFunction:
    """IR 中的函数节点。"""

    name: str
    signatures: list[IRSignature] = field(default_factory=list)
    doc: str | None = field(default=None)
    c_inferred_source_comment: str | None = field(default=None)

    def __str__(self) -> str:
        """返回函数节点的调试字符串。"""
        if not self.signatures:
            return f"{self.name}(...)"

        rendered = []
        for signature in self.signatures:
            args = ", ".join(str(arg) for arg in signature.args)
            return_type = signature.return_type.render() if signature.return_type is not None else None
            rendered.append(f"({args}) -> {return_type}")
        return f"{self.name}{' | '.join(rendered)}"
