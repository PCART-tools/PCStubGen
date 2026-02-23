from __future__ import annotations

import typing

from pcstubgen2.IR import IRAlias, IRClass, IRField, IRFunction, IRMethod, IRModule, IRProperty, IRVariable, QualifiedName
from pcstubgen2.NodeVisitors.Filters import (
    FilterClassMembersVisitor,
    FilterPybind11ViewClassesVisitor,
    FilterPybindInternalsVisitor,
    FilterTypingModuleAttributesVisitor,
)


def test_filter_typing_module_attributes_visitor_filters_by_name_and_identity() -> None:
    visitor = FilterTypingModuleAttributesVisitor()
    ir_module = IRModule(full_name=QualifiedName.from_str("pkg.mod"))

    any_not_typing = object()
    ir_module.variables = [
        IRVariable(name="Any", value=None, runtime_value=typing.Any),
        IRVariable(name="Any", value=None, runtime_value=any_not_typing),
        IRVariable(name="keep_value", value=None, runtime_value=typing.Any),
    ]

    visitor.visit_module(ir_module)

    assert len(ir_module.variables) == 2
    assert [variable.name for variable in ir_module.variables] == ["Any", "keep_value"]
    assert ir_module.variables[0].runtime_value is any_not_typing


def test_filter_class_members_visitor_filters_expected_blacklists() -> None:
    visitor = FilterClassMembersVisitor()
    ir_module = IRModule(full_name=QualifiedName.from_str("pkg.mod"))
    ir_module.variables = [
        IRVariable(name="__builtins__", value=None),
        IRVariable(name="visible_var", value=None),
    ]

    ir_class = IRClass(name="VisibleClass")
    ir_class.classes = [IRClass(name="__dict__"), IRClass(name="NestedVisible")]
    ir_class.aliases = [
        IRAlias(name="__module__", origin=QualifiedName.from_str("typing.Any")),
        IRAlias(name="AliasVisible", origin=QualifiedName.from_str("typing.Any")),
    ]
    ir_class.methods = [
        IRMethod(function=IRFunction(name="__dir__"), modifier=None),
        IRMethod(function=IRFunction(name="__class__"), modifier=None),
        IRMethod(function=IRFunction(name="method_visible"), modifier=None),
    ]
    ir_class.fields = [
        IRField(variable=IRVariable(name="__annotations__", value=None), modifier=None),
        IRField(variable=IRVariable(name="__firstlineno__", value=None), modifier=None),
        IRField(variable=IRVariable(name="field_visible", value=None), modifier=None),
    ]
    ir_class.properties = [
        IRProperty(name="__weakref__", modifier=None),
        IRProperty(name="prop_visible", modifier=None),
    ]
    ir_module.classes = [ir_class]

    visitor.visit_module(ir_module)

    assert [variable.name for variable in ir_module.variables] == ["visible_var"]
    assert [class_.name for class_ in ir_class.classes] == ["NestedVisible"]
    assert [alias.name for alias in ir_class.aliases] == ["AliasVisible"]
    assert [method.function.name for method in ir_class.methods] == ["method_visible"]
    assert [field.variable.name for field in ir_class.fields] == ["field_visible"]
    assert [prop.name for prop in ir_class.properties] == ["prop_visible"]


def test_filter_pybind_internals_visitor_filters_internal_members() -> None:
    visitor = FilterPybindInternalsVisitor()
    ir_module = IRModule(full_name=QualifiedName.from_str("pkg.mod"))
    ir_module.variables = [
        IRVariable(name="__entries", value=None),
        IRVariable(name="visible_var", value=None),
    ]
    ir_module.classes = [IRClass(name="pybind11_type"), IRClass(name="TopLevelVisible")]

    ir_class = IRClass(name="VisibleClass")
    ir_class.classes = [
        IRClass(name="pybind11_type"),
        IRClass(name="__pybind11_module_local"),
        IRClass(name="_pybind11_conduit_v1_abc"),
        IRClass(name="NestedVisible"),
    ]
    ir_class.aliases = [
        IRAlias(name="__pybind11_module_local", origin=QualifiedName.from_str("typing.Any")),
        IRAlias(name="_pybind11_conduit_v1_alias", origin=QualifiedName.from_str("typing.Any")),
        IRAlias(name="AliasVisible", origin=QualifiedName.from_str("typing.Any")),
    ]
    ir_class.methods = [
        IRMethod(function=IRFunction(name="__pybind11_module_method"), modifier=None),
        IRMethod(function=IRFunction(name="_pybind11_conduit_v1_method"), modifier=None),
        IRMethod(function=IRFunction(name="method_visible"), modifier=None),
    ]
    ir_class.fields = [
        IRField(variable=IRVariable(name="__entries", value=None), modifier=None),
        IRField(variable=IRVariable(name="__pybind11_module_field", value=None), modifier=None),
        IRField(variable=IRVariable(name="_pybind11_conduit_v1_field", value=None), modifier=None),
        IRField(variable=IRVariable(name="field_visible", value=None), modifier=None),
    ]
    ir_class.properties = [
        IRProperty(name="__pybind11_module_prop", modifier=None),
        IRProperty(name="_pybind11_conduit_v1_prop", modifier=None),
        IRProperty(name="prop_visible", modifier=None),
    ]
    ir_module.classes.append(ir_class)

    visitor.visit_module(ir_module)

    assert [variable.name for variable in ir_module.variables] == ["visible_var"]
    assert [class_.name for class_ in ir_module.classes] == ["pybind11_type", "TopLevelVisible", "VisibleClass"]
    assert [class_.name for class_ in ir_class.classes] == ["NestedVisible"]
    assert [alias.name for alias in ir_class.aliases] == ["AliasVisible"]
    assert [method.function.name for method in ir_class.methods] == ["method_visible"]
    assert [field.variable.name for field in ir_class.fields] == ["field_visible"]
    assert [prop.name for prop in ir_class.properties] == ["prop_visible"]


def test_filter_pybind11_view_classes_visitor_filters_only_module_level_view_classes() -> None:
    visitor = FilterPybind11ViewClassesVisitor()
    ir_module = IRModule(full_name=QualifiedName.from_str("pkg.mod"))

    keep_class = IRClass(name="KeepClass")
    keep_class.classes = [IRClass(name="ItemsView")]
    ir_module.classes = [
        IRClass(name="ItemsView"),
        IRClass(name="KeysView"),
        IRClass(name="ValuesView"),
        keep_class,
    ]

    visitor.visit_module(ir_module)

    assert [class_.name for class_ in ir_module.classes] == ["KeepClass"]
    assert [class_.name for class_ in keep_class.classes] == ["ItemsView"]
