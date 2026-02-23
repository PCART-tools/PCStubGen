from __future__ import annotations

from ...IR import IRClass, IRModule
from ..NodeVisitor import NodeVisitor


class FilterClassMembersVisitor(NodeVisitor):
    '''
    过滤掉类成员中的黑名单成员
    '''
    __attribute_blacklist = {
        "__annotations__",
        "__builtins__",
        "__cached__",
        "__file__",
        "__firstlineno__",
        "__loader__",
        "__name__",
        "__package__",
        "__path__",
        "__spec__",
        "__static_attributes__",
    }

    __class_member_blacklist = {
        "__annotations__",
        "__class__",
        "__dict__",
        "__module__",
        "__qualname__",
        "__weakref__",
    }

    __method_blacklist = {
        "__dir__",
        "__sizeof__",
    }

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
            if class_.name not in self.__class_member_blacklist
        ]

        node.aliases = [
            alias
            for alias in node.aliases
            if alias.name not in self.__class_member_blacklist
        ]

        node.methods = [
            method
            for method in node.methods
            if method.function.name not in self.__class_member_blacklist
            and method.function.name not in self.__method_blacklist
        ]

        node.fields = [
            field
            for field in node.fields
            if field.variable.name not in self.__class_member_blacklist
            and field.variable.name not in self.__attribute_blacklist
        ]

        node.properties = [
            prop
            for prop in node.properties
            if prop.name not in self.__class_member_blacklist
        ]

        super().visit_class(node)
