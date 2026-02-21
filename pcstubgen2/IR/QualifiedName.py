from __future__ import annotations

from typing import Tuple


class QualifiedName(Tuple[str, ...]):
    """完全限定名称"""

    @classmethod
    def from_str(cls, name: str) -> QualifiedName:
        return QualifiedName(name.split("."))

    def __str__(self) -> str:
        return ".".join(self)

    def concat(self, name: str) -> QualifiedName:
        return QualifiedName((*self, name))

    @property
    def parent(self) -> QualifiedName:
        return QualifiedName(self[:-1])

    @property
    def name(self) -> str:
        return self[-1]
