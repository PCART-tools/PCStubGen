from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from ...ErrorCollector import ErrorCollector
from ...Errors import InvalidIdentifierError
from ...IR import IRClass, IRModule, QualifiedName
from ..NodeVisitor import NodeVisitor

T = TypeVar("T")


class FilterInvalidIdentifierVisitor(NodeVisitor):
    def __init__(self, error_collector: ErrorCollector):
        self.error_collector = error_collector
        self._path_stack: list[QualifiedName] = []

    def visit_module(self, node: IRModule) -> None:
        parent_path = node.full_name
        self.error_collector.set_current_path(parent_path)

        node.variables = self._filter_invalid_identifiers(
            node.variables, lambda variable: variable.name, parent_path
        )
        node.classes = self._filter_invalid_identifiers(
            node.classes, lambda class_: class_.name, parent_path
        )
        node.functions = self._filter_invalid_identifiers(
            node.functions, lambda function: function.name, parent_path
        )
        node.aliases = self._filter_invalid_identifiers(
            node.aliases, lambda alias: alias.name, parent_path
        )
        node.sub_modules = self._filter_invalid_identifiers(
            node.sub_modules, lambda sub_module: sub_module.Name, parent_path
        )
        node.type_vars = self._filter_invalid_identifiers(
            node.type_vars, lambda type_var: type_var.name, parent_path
        )
        node.imports = set(
            self._filter_invalid_identifiers(
                list(node.imports),
                lambda import_: import_.name or import_.origin.name,
                parent_path,
            )
        )

        if node.all is not None and not node.all.name.isidentifier():
            self._report_invalid_identifier(node.all.name, parent_path)
            node.all = None

        self._path_stack.append(parent_path)
        try:
            super().visit_module(node)
        finally:
            self._path_stack.pop()

    def visit_class(self, node: IRClass) -> None:
        if self._path_stack:
            class_path = self._path_stack[-1].concat(node.name)
        else:
            class_path = QualifiedName((node.name,))
        self.error_collector.set_current_path(class_path)

        node.classes = self._filter_invalid_identifiers(
            node.classes, lambda class_: class_.name, class_path
        )
        node.aliases = self._filter_invalid_identifiers(
            node.aliases, lambda alias: alias.name, class_path
        )
        node.methods = self._filter_invalid_identifiers(
            node.methods, lambda method: method.function.name, class_path
        )
        node.properties = self._filter_invalid_identifiers(
            node.properties, lambda prop: prop.name, class_path
        )
        node.fields = self._filter_invalid_identifiers(
            node.fields, lambda field: field.variable.name, class_path
        )

        self._path_stack.append(class_path)
        try:
            super().visit_class(node)
        finally:
            self._path_stack.pop()

    def _filter_invalid_identifiers(
        self,
        nodes: list[T],
        get_name: Callable[[T], str],
        parent_path: QualifiedName,
    ) -> list[T]:
        kept_nodes: list[T] = []
        for node in nodes:
            name = get_name(node)
            if name.isidentifier():
                kept_nodes.append(node)
            else:
                self._report_invalid_identifier(name, parent_path)
        return kept_nodes

    def _report_invalid_identifier(self, name: str, path: QualifiedName) -> None:
        self.error_collector.set_current_path(path)
        self.error_collector.report_error(InvalidIdentifierError(name=name, path=path))
