from __future__ import annotations

"""`Py_BuildValue` 类型树及其规范化、渲染逻辑。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


class Type(ABC):
    @abstractmethod
    def canonicalize(self) -> Type:
        raise NotImplementedError

    @abstractmethod
    def render(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def collect_imports(self) -> set[str]:
        raise NotImplementedError


@dataclass(frozen=True)
class RawType(Type):
    text: str
    imports: tuple[str, ...] = ()

    def canonicalize(self) -> Type:
        return self

    def render(self) -> str:
        return self.text

    def collect_imports(self) -> set[str]:
        return set(self.imports)


@dataclass(frozen=True)
class AnyType(Type):
    def canonicalize(self) -> Type:
        return self

    def render(self) -> str:
        return "typing.Any"

    def collect_imports(self) -> set[str]:
        return {"typing"}


def _union_sort_key(member: Type) -> str:
    """返回 union 成员的稳定排序键。"""
    return member.render()


@dataclass(frozen=True)
class UnionType(Type):
    """表示联合类型；空成员列表表示 `Never`。"""

    members: tuple[Type, ...]

    def is_empty(self) -> bool:
        """返回当前 union 是否不含任何成员。"""
        return len(self.members) == 0

    def canonicalize(self) -> Type:
        """规范化联合类型并折叠 `Any` / `Never` 语义。"""
        if self.is_empty():
            return self

        member_set: set[Type] = set()
        for member in self.members:
            canonical_member = member.canonicalize()
            if isinstance(canonical_member, AnyType):
                # Any | int = Any
                return canonical_member

            if isinstance(canonical_member, UnionType):
                # 这里不需要递归flatten，canonicalize已保证flat
                for m in canonical_member.members:
                    member_set.add(m)
                continue

            member_set.add(canonical_member)

        unique_members = list(member_set)
        if len(unique_members) == 1:
            return unique_members[0]

        unique_members.sort(key=_union_sort_key)
        return UnionType(tuple(unique_members))

    def render(self) -> str:
        """按当前成员顺序渲染联合类型。"""
        if self.is_empty():
            return "Any"
        if len(self.members) == 1:
            return self.members[0].render()
        return " | ".join(member.render() for member in self.members)

    def collect_imports(self) -> set[str]:
        imports: set[str] = set()
        for member in self.members:
            imports.update(member.collect_imports())
        return imports


@dataclass(frozen=True)
class TupleType(Type):
    items: tuple[Type, ...]

    def canonicalize(self) -> Type:
        return TupleType(tuple(item.canonicalize() for item in self.items))

    def render(self) -> str:
        if not self.items:
            return "tuple[()]"
        if len(self.items) == 1:
            return f"tuple[{self.items[0].render()},]"
        return f"tuple[{', '.join(item.render() for item in self.items)}]"

    def collect_imports(self) -> set[str]:
        imports: set[str] = set()
        for item in self.items:
            imports.update(item.collect_imports())
        return imports


@dataclass(frozen=True)
class ListType(Type):
    element: Type

    def canonicalize(self) -> Type:
        return ListType(self.element.canonicalize())

    def render(self) -> str:
        if isinstance(self.element, UnionType) and self.element.is_empty():
            return "list[typing.Any]"
        return f"list[{self.element.render()}]"

    def collect_imports(self) -> set[str]:
        imports = set(self.element.collect_imports())
        if isinstance(self.element, UnionType) and self.element.is_empty():
            imports.add("typing")
        return imports


@dataclass(frozen=True)
class DictType(Type):
    key: Type
    value: Type

    def canonicalize(self) -> Type:
        return DictType(self.key.canonicalize(), self.value.canonicalize())

    def render(self) -> str:
        key_rendered = (
            "typing.Any"
            if isinstance(self.key, UnionType) and self.key.is_empty()
            else self.key.render()
        )
        value_rendered = (
            "typing.Any"
            if isinstance(self.value, UnionType) and self.value.is_empty()
            else self.value.render()
        )
        return f"dict[{key_rendered}, {value_rendered}]"

    def collect_imports(self) -> set[str]:
        imports = set(self.key.collect_imports()) | set(self.value.collect_imports())
        if isinstance(self.key, UnionType) and self.key.is_empty():
            imports.add("typing")
        if isinstance(self.value, UnionType) and self.value.is_empty():
            imports.add("typing")
        return imports
