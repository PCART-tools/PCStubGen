from __future__ import annotations

from ...IR import IRClass, IRModule
from ..NodeVisitor import NodeVisitor


class StandardFilterVisitor(NodeVisitor):
    __attribute_blacklist = {
        "__annotations__",
        "__builtins__",
        "__cached__",
        "__file__",
        "__firstlineno__",  # 在 Python 3.13 中属于运行时字段，非公共 API
        "__loader__",
        "__name__",
        "__package__",
        "__path__",
        "__spec__",
        "__static_attributes__",
        "__entries",  # 内部字段（pybind）
    }

    __class_member_blacklist = {
        "__annotations__",
        "__class__",
        "__dict__",
        "__firstlineno__",  # 在 Python 3.13 中属于运行时字段，非公共 API
        "__module__",
        "__qualname__",
        "__static_attributes__",
        "__weakref__",
    }

    __method_blacklist = {
        "__dir__",
        "__sizeof__",
    }

    __class_blacklist = {
        "pybind11_type",
    }

    def visit_module(self, node: IRModule) -> None:
        # 过滤变量
        node.variables = [
            v for v in node.variables if v.name not in self.__attribute_blacklist
        ]

        # 过滤别名
        node.aliases = [
            alias for alias in node.aliases if alias.name not in self.__attribute_blacklist
        ]

        # 黑名单包含像 __builtins__ 这样的名称，如果显式导入，则不应从导入中过滤

        # 过滤类
        node.classes = [c for c in node.classes if c.name not in self.__class_blacklist]

        # 过滤 pybind11 内部类 (KeysView, ItemsView, ValuesView)
        view_classes = {"ItemsView", "KeysView", "ValuesView"}
        node.classes = [c for c in node.classes if c.name not in view_classes]

        # 递归
        super().visit_module(node)

    def visit_class(self, node: IRClass) -> None:
        # 过滤嵌套类
        node.classes = [
            c
            for c in node.classes
            if c.name not in self.__class_member_blacklist
            and c.name not in self.__class_blacklist
            and not c.name.startswith("__pybind11_module")
            and not c.name.startswith("_pybind11_conduit_v1_")
        ]

        node.aliases = [
            a
            for a in node.aliases
            if a.name not in self.__class_member_blacklist
            and not a.name.startswith("__pybind11_module")
            and not a.name.startswith("_pybind11_conduit_v1_")
        ]

        # 过滤方法
        node.methods = [
            m
            for m in node.methods
            if m.function.name not in self.__method_blacklist
            and not m.function.name.startswith("__pybind11_module")
            and not m.function.name.startswith("_pybind11_conduit_v1_")
        ]

        # 过滤属性/字段/属性访问器
        node.fields = [
            f
            for f in node.fields
            if f.variable.name not in self.__class_member_blacklist
            and not f.variable.name.startswith("__pybind11_module")
            and not f.variable.name.startswith("_pybind11_conduit_v1_")
        ]
        node.properties = [
            p
            for p in node.properties
            if p.name not in self.__class_member_blacklist
            and not p.name.startswith("__pybind11_module")
            and not p.name.startswith("_pybind11_conduit_v1_")
        ]

        # 递归
        super().visit_class(node)
