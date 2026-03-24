from __future__ import annotations

"""Py_BuildValue 类型树节点及其规范化、渲染逻辑。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


class TypeNode(ABC):
    @abstractmethod
    def canonicalize(self) -> TypeNode:
        raise NotImplementedError

    @abstractmethod
    def render(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class NamedTypeNode(TypeNode):
    name: str

    def canonicalize(self) -> TypeNode:
        return self

    def render(self) -> str:
        return self.name


@dataclass(frozen=True)
class AnyTypeNode(TypeNode):
    def canonicalize(self) -> TypeNode:
        return self

    def render(self) -> str:
        return "Any"


def _union_sort_key(member: TypeNode) -> str:
    """返回 union 成员的稳定排序键。"""
    return member.render()


@dataclass(frozen=True)
class UnionTypeNode(TypeNode):
    """表示联合类型；空成员列表表示 `Never`。"""

    members: tuple[TypeNode, ...]

    def is_empty(self) -> bool:
        """返回当前 union 是否不含任何成员。"""
        return len(self.members) == 0

    def canonicalize(self) -> TypeNode:
        """规范化联合类型并折叠 `Any` / `Never` 语义。"""
        if self.is_empty():
            return self

        member_set: set[TypeNode] = set()
        for member in self.members:
            canonical_member = member.canonicalize()
            if isinstance(canonical_member, AnyTypeNode):
                # Any | int = Any
                return canonical_member

            if isinstance(canonical_member, UnionTypeNode):
                # 这里不需要递归flatten，canonicalize已保证flat
                for m in canonical_member.members:
                    member_set.add(m)
                continue

            member_set.add(canonical_member)

        unique_members = list(member_set)
        unique_members.sort(key=_union_sort_key)

        if len(unique_members) == 1:
            return unique_members[0]
        return UnionTypeNode(tuple(unique_members))

    def render(self) -> str:
        """按当前成员顺序渲染联合类型。"""
        if self.is_empty():
            return "Never"
        if len(self.members) == 1:
            return self.members[0].render()
        return " | ".join(member.render() for member in self.members)


@dataclass(frozen=True)
class TupleTypeNode(TypeNode):
    items: tuple[TypeNode, ...]

    def canonicalize(self) -> TypeNode:
        return TupleTypeNode(tuple(item.canonicalize() for item in self.items))

    def render(self) -> str:
        if not self.items:
            return "tuple[()]"
        if len(self.items) == 1:
            return f"tuple[{self.items[0].render()},]"
        return f"tuple[{', '.join(item.render() for item in self.items)}]"


@dataclass(frozen=True)
class ListTypeNode(TypeNode):
    element: TypeNode

    def canonicalize(self) -> TypeNode:
        return ListTypeNode(self.element.canonicalize())

    def render(self) -> str:
        if isinstance(self.element, UnionTypeNode) and self.element.is_empty():
            return "list[Any]"
        return f"list[{self.element.render()}]"


@dataclass(frozen=True)
class DictTypeNode(TypeNode):
    key: TypeNode
    value: TypeNode

    def canonicalize(self) -> TypeNode:
        return DictTypeNode(self.key.canonicalize(), self.value.canonicalize())

    def render(self) -> str:
        key_rendered = "Any" if isinstance(self.key, UnionTypeNode) and self.key.is_empty() else self.key.render()
        value_rendered = (
            "Any"
            if isinstance(self.value, UnionTypeNode) and self.value.is_empty()
            else self.value.render()
        )
        return f"dict[{key_rendered}, {value_rendered}]"
