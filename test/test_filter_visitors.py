from __future__ import annotations

from pcstubgen2.ErrorCollector import ErrorCollector
from pcstubgen2.IR import (
    IRArgument,
    IRArgumentKind,
    IRClass,
    IRFunction,
    IRMethod,
    IRModule,
    QualifiedName,
    ResolvedType,
)
from pcstubgen2.NodeVisitors.DocStringSignatureParserVisitor import (
    DocStringSignatureParserVisitor,
)
from pcstubgen2.NodeVisitors.NodeVisitor import NodeVisitor
from pcstubgen2.NodeVisitors.Fixes import (
    FixBuiltinTypesVisitor,
    FixCurrentModulePrefixInTypeNamesVisitor,
    InferMethodModifierVisitor,
    FixPEP585CollectionNamesVisitor,
    FixRedundantMethodsFromBuiltinObjectVisitor,
    FixTypingTypeNamesVisitor,
    RemoveSelfAnnotationVisitor,
)


def _generic_signature() -> list[IRArgument]:
    return [
        IRArgument(name="args", kind=IRArgumentKind.VAR_POSITIONAL),
        IRArgument(name="kwargs", kind=IRArgumentKind.VAR_KEYWORD),
    ]


def test_docstring_parser_parses_generic_function_signature() -> None:
    visitor = DocStringSignatureParserVisitor(error_collector=ErrorCollector())
    ir_module = IRModule(full_name=QualifiedName.from_str("pkg.mod"))
    ir_module.functions = [
        IRFunction(
            name="foo",
            args=_generic_signature(),
            doc="foo(x: int, y: str) -> str\n\nparsed from docstring",
        )
    ]

    visitor.visit_module(ir_module)

    parsed = ir_module.functions[0]
    assert [arg.name for arg in parsed.args] == ["x", "y"]
    assert str(parsed.return_annotation) == "str"
    assert parsed.doc == "parsed from docstring"


def test_infer_method_modifier_visitor_reinfers_after_docstring_parse() -> None:
    parser_visitor = DocStringSignatureParserVisitor(error_collector=ErrorCollector())
    infer_modifier_visitor = InferMethodModifierVisitor()
    ir_class = IRClass(name="C")
    ir_class.methods = [
        IRMethod(
            function=IRFunction(
                name="build",
                args=_generic_signature(),
                doc="build(cls: C, count: int) -> C",
            ),
            decorator="staticmethod",
        )
    ]

    parser_visitor.visit_class(ir_class)

    parsed = ir_class.methods[0]
    assert [arg.name for arg in parsed.function.args] == ["cls", "count"]
    assert parsed.decorator == "staticmethod"

    infer_modifier_visitor.visit_class(ir_class)

    assert parsed.decorator == "classmethod"


def test_infer_method_modifier_visitor_covers_all_first_arg_cases() -> None:
    visitor = InferMethodModifierVisitor()
    ir_class = IRClass(
        name="C",
        methods=[
            IRMethod(
                function=IRFunction(name="instance_method", args=[IRArgument(name="self")]),
                decorator="staticmethod",
            ),
            IRMethod(
                function=IRFunction(name="class_method", args=[IRArgument(name="cls")]),
                decorator=None,
            ),
            IRMethod(
                function=IRFunction(name="static_no_args", args=[]),
                decorator=None,
            ),
            IRMethod(
                function=IRFunction(name="static_other_first", args=[IRArgument(name="value")]),
                decorator="classmethod",
            ),
        ],
    )

    visitor.visit_class(ir_class)

    decorators = {method.function.name: method.decorator for method in ir_class.methods}
    assert decorators == {
        "instance_method": None,
        "class_method": "classmethod",
        "static_no_args": "staticmethod",
        "static_other_first": "staticmethod",
    }


def test_type_fix_visitors_update_annotations_and_bases() -> None:
    method = IRMethod(
        function=IRFunction(
            name="m",
            args=[
                IRArgument(
                    name="value",
                    annotation=ResolvedType(name=QualifiedName.from_str("sequence")),
                )
            ],
            return_annotation=ResolvedType(name=QualifiedName.from_str("builtins.NoneType")),
        ),
        decorator=None,
    )
    ir_class = IRClass(
        name="C",
        bases=[QualifiedName.from_str("typing.List")],
        methods=[method],
    )
    ir_module = IRModule(full_name=QualifiedName.from_str("pkg.mod"), classes=[ir_class])

    FixTypingTypeNamesVisitor().visit_module(ir_module)
    FixPEP585CollectionNamesVisitor().visit_module(ir_module)
    FixBuiltinTypesVisitor().visit_module(ir_module)

    assert str(method.function.args[0].annotation) == "typing.Sequence"
    assert str(method.function.return_annotation) == "None"
    assert [str(base) for base in ir_class.bases] == ["list"]


def test_remove_self_annotation_visitor_strips_class_self_type() -> None:
    method = IRMethod(
        function=IRFunction(
            name="m",
            args=[
                IRArgument(
                    name="self",
                    annotation=ResolvedType(name=QualifiedName.from_str("pkg.mod.C")),
                )
            ],
        ),
        decorator=None,
    )
    ir_class = IRClass(name="C", methods=[method])
    ir_module = IRModule(full_name=QualifiedName.from_str("pkg.mod"), classes=[ir_class])

    RemoveSelfAnnotationVisitor().visit_module(ir_module)

    assert method.function.args[0].annotation is None


def test_fix_current_module_prefix_visitor_strips_local_prefix() -> None:
    method = IRMethod(
        function=IRFunction(
            name="m",
            args=[
                IRArgument(
                    name="x",
                    annotation=ResolvedType(name=QualifiedName.from_str("pkg.mod.LocalType")),
                )
            ],
            return_annotation=ResolvedType(name=QualifiedName.from_str("pkg.mod.ResultType")),
        ),
        decorator=None,
    )
    ir_module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        classes=[IRClass(name="C", methods=[method])],
    )

    FixCurrentModulePrefixInTypeNamesVisitor().visit_module(ir_module)

    assert str(method.function.args[0].annotation) == "LocalType"
    assert str(method.function.return_annotation) == "ResultType"


def test_fix_redundant_object_init_visitor_removes_only_builtin_init() -> None:
    ir_class = IRClass(
        name="C",
        methods=[
            IRMethod(
                function=IRFunction(name="__init__", doc=object.__init__.__doc__),
                decorator=None,
            ),
            IRMethod(function=IRFunction(name="run"), decorator=None),
        ],
    )

    FixRedundantMethodsFromBuiltinObjectVisitor().visit_class(ir_class)

    assert [m.function.name for m in ir_class.methods] == ["run"]


def test_node_visitor_none_return_removes_classes_functions_and_methods() -> None:
    class DropByNameVisitor(NodeVisitor):
        def visit_class(self, node: IRClass) -> IRClass | None:
            if node.name.startswith("Drop"):
                return None
            return super().visit_class(node)

        def visit_function(self, node: IRFunction) -> IRFunction | None:
            if node.name.startswith("drop"):
                return None
            return super().visit_function(node)

        def visit_method(self, node: IRMethod) -> IRMethod | None:
            if node.function.name.startswith("drop"):
                return None
            return super().visit_method(node)

    keep_class = IRClass(
        name="KeepClass",
        classes=[IRClass(name="DropNested"), IRClass(name="KeepNested")],
        methods=[
            IRMethod(function=IRFunction(name="keep_method"), decorator=None),
            IRMethod(function=IRFunction(name="drop_method"), decorator=None),
        ],
    )
    ir_module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        classes=[IRClass(name="DropClass"), keep_class],
        functions=[IRFunction(name="drop_func"), IRFunction(name="keep_func")],
    )

    DropByNameVisitor().visit_module(ir_module)

    assert [cls.name for cls in ir_module.classes] == ["KeepClass"]
    assert [func.name for func in ir_module.functions] == ["keep_func"]
    assert [cls.name for cls in keep_class.classes] == ["KeepNested"]
    assert [method.function.name for method in keep_class.methods] == ["keep_method"]


def test_node_visitor_keeps_nodes_when_returning_node() -> None:
    class RenameVisitedFunctionsVisitor(NodeVisitor):
        def visit_function(self, node: IRFunction) -> IRFunction | None:
            node.name = f"visited_{node.name}"
            return node

    method = IRMethod(function=IRFunction(name="m"), decorator=None)
    ir_class = IRClass(name="C", methods=[method])
    ir_module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        classes=[ir_class],
        functions=[IRFunction(name="f")],
    )

    RenameVisitedFunctionsVisitor().visit_module(ir_module)

    assert [func.name for func in ir_module.functions] == ["visited_f"]
    assert [m.function.name for m in ir_class.methods] == ["visited_m"]
