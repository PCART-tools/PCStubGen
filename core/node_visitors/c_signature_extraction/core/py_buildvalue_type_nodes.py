from abc import ABC, abstractmethod
from dataclasses import dataclass

class TypeNode(ABC):
    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class NamedTypeNode(TypeNode):
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class UnionTypeNode(TypeNode):
    members: tuple[TypeNode, ...]

    def __str__(self) -> str:
        if not self.members:
            return "Any"
        if len(self.members) == 1:
            return str(self.members[0])
        return " | ".join(str(member) for member in self.members)


@dataclass(frozen=True)
class TupleTypeNode(TypeNode):
    items: tuple[TypeNode, ...]

    def __str__(self) -> str:
        if not self.items:
            return "tuple[()]"
        if len(self.items) == 1:
            return f"tuple[{self.items[0]},]"
        return f"tuple[{', '.join(str(item) for item in self.items)}]"


@dataclass(frozen=True)
class ListTypeNode(TypeNode):
    element: UnionTypeNode

    def __str__(self) -> str:
        return f"list[{self.element}]"


@dataclass(frozen=True)
class DictTypeNode(TypeNode):
    key: UnionTypeNode
    value: UnionTypeNode

    def __str__(self) -> str:
        return f"dict[{self.key}, {self.value}]"