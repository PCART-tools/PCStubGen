from __future__ import annotations

import re

from pcstubgen2.ErrorCollector import ErrorCollector
from pcstubgen2.Errors import InvalidIdentifierError
from pcstubgen2.IR import (
    IRAlias,
    IRClass,
    IRField,
    IRFunction,
    IRImport,
    IRMethod,
    IRModule,
    IRProperty,
    IRTypeVar,
    IRVariable,
    QualifiedName,
)
from pcstubgen2.NodeVisitors.Filters import FilterInvalidIdentifierVisitor


def test_filter_invalid_identifier_visitor_filters_and_reports_errors() -> None:
    collector = ErrorCollector()
    visitor = FilterInvalidIdentifierVisitor(error_collector=collector)

    ir_module = IRModule(full_name=QualifiedName.from_str("pkg.mod"))
    ir_module.variables = [IRVariable(name="ok_var", value=None), IRVariable(name="bad-var", value=None)]
    ir_module.functions = [IRFunction(name="ok_func"), IRFunction(name="bad-func")]
    ir_module.aliases = [
        IRAlias(name="ok_alias", origin=QualifiedName.from_str("typing.Any")),
        IRAlias(name="bad-alias", origin=QualifiedName.from_str("typing.Any")),
    ]
    ir_module.type_vars = [IRTypeVar(name="T"), IRTypeVar(name="bad-type")]
    ir_module.imports = {
        IRImport(name="ok_import", origin=QualifiedName.from_str("typing")),
        IRImport(name="bad-import", origin=QualifiedName.from_str("typing")),
    }
    ir_module.sub_modules = [
        IRModule(full_name=QualifiedName.from_str("pkg.mod.good_sub")),
        IRModule(full_name=QualifiedName.from_str("pkg.mod.bad-sub")),
    ]

    ir_class = IRClass(name="GoodClass")
    ir_class.methods = [
        IRMethod(function=IRFunction(name="ok_method"), modifier=None),
        IRMethod(function=IRFunction(name="bad-method"), modifier=None),
    ]
    ir_class.properties = [
        IRProperty(name="ok_prop", modifier=None),
        IRProperty(name="bad-prop", modifier=None),
    ]
    ir_class.fields = [
        IRField(variable=IRVariable(name="ok_field", value=None), modifier=None),
        IRField(variable=IRVariable(name="bad-field", value=None), modifier=None),
    ]
    ir_class.aliases = [
        IRAlias(name="ok_inner_alias", origin=QualifiedName.from_str("typing.Any")),
        IRAlias(name="bad-inner-alias", origin=QualifiedName.from_str("typing.Any")),
    ]
    ir_class.classes = [IRClass(name="NestedGood"), IRClass(name="Nested-Bad")]

    ir_module.classes = [ir_class, IRClass(name="bad-class")]

    visitor.visit_module(ir_module)

    assert [v.name for v in ir_module.variables] == ["ok_var"]
    assert [f.name for f in ir_module.functions] == ["ok_func"]
    assert [a.name for a in ir_module.aliases] == ["ok_alias"]
    assert [t.name for t in ir_module.type_vars] == ["T"]
    assert sorted(imp.name for imp in ir_module.imports if imp.name is not None) == ["ok_import"]
    assert [m.Name for m in ir_module.sub_modules] == ["good_sub"]
    assert [c.name for c in ir_module.classes] == ["GoodClass"]

    assert [m.function.name for m in ir_class.methods] == ["ok_method"]
    assert [p.name for p in ir_class.properties] == ["ok_prop"]
    assert [f.variable.name for f in ir_class.fields] == ["ok_field"]
    assert [a.name for a in ir_class.aliases] == ["ok_inner_alias"]
    assert [c.name for c in ir_class.classes] == ["NestedGood"]

    invalid_errors = [error for error in collector.errors if isinstance(error, InvalidIdentifierError)]
    invalid_names = {error.name for error in invalid_errors}
    assert {
        "bad-var",
        "bad-func",
        "bad-alias",
        "bad-type",
        "bad-import",
        "bad-sub",
        "bad-class",
        "bad-method",
        "bad-prop",
        "bad-field",
        "bad-inner-alias",
        "Nested-Bad",
    }.issubset(invalid_names)


def test_filter_invalid_identifier_visitor_respects_ignore_regex() -> None:
    collector = ErrorCollector()
    collector.ignore_invalid_identifiers = re.compile(r"^skip-")
    visitor = FilterInvalidIdentifierVisitor(error_collector=collector)

    ir_module = IRModule(full_name=QualifiedName.from_str("pkg.mod"))
    ir_module.variables = [
        IRVariable(name="skip-bad", value=None),
        IRVariable(name="keep-bad", value=None),
    ]

    visitor.visit_module(ir_module)

    assert ir_module.variables == []
    invalid_errors = [error for error in collector.errors if isinstance(error, InvalidIdentifierError)]
    invalid_names = {error.name for error in invalid_errors}
    assert "skip-bad" not in invalid_names
    assert "keep-bad" in invalid_names
