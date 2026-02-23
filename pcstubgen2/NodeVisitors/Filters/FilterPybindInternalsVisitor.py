from __future__ import annotations

from ...IR import IRClass, IRModule
from ..NodeVisitor import NodeVisitor


class FilterPybindInternalsVisitor(NodeVisitor):
    __attribute_blacklist = {"__entries"}
    __class_blacklist = {"pybind11_type"}

    def visit_module(self, node: IRModule) -> None:
        node.variables = [
            variable
            for variable in node.variables
            if variable.name not in self.__attribute_blacklist
        ]

        super().visit_module(node)

    def visit_class(self, node: IRClass) -> None:
        node.classes = [
            class_
            for class_ in node.classes
            if not self._is_blacklisted_class_member_name(class_.name)
        ]

        node.aliases = [
            alias
            for alias in node.aliases
            if not self._is_blacklisted_class_member_name(alias.name)
        ]

        node.methods = [
            method
            for method in node.methods
            if not self._is_blacklisted_class_member_name(method.function.name)
        ]

        node.fields = [
            field
            for field in node.fields
            if field.variable.name not in self.__attribute_blacklist
            and not self._is_blacklisted_class_member_name(field.variable.name)
        ]

        node.properties = [
            prop
            for prop in node.properties
            if not self._is_blacklisted_class_member_name(prop.name)
        ]

        super().visit_class(node)

    def _is_blacklisted_class_member_name(self, name: str) -> bool:
        if name in self.__class_blacklist:
            return True
        if name.startswith("__pybind11_module"):
            return True
        if name.startswith("_pybind11_conduit_v1_"):
            return True
        return False
