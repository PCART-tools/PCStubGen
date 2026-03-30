from __future__ import annotations

from pcstubgen.ir import IRClass, IRFunction, IRMethod, IRModule, QualifiedName
from pcstubgen.visitors.node_visitor import NodeVisitor
from pcstubgen.visitor_runner import run_visitors


def test_visitor_runner_inplace_mutation_removes_classes_functions_and_methods() -> None:
    class DropByNameVisitor(NodeVisitor):
        def visit_module(self, node: IRModule) -> None:
            node.classes = [cls for cls in node.classes if not cls.name.startswith("Drop")]
            node.functions = [
                func for func in node.functions if not func.name.startswith("drop")
            ]

        def visit_class(self, node: IRClass, module: IRModule) -> None:
            node.classes = [cls for cls in node.classes if not cls.name.startswith("Drop")]
            node.methods = [
                method
                for method in node.methods
                if not method.function.name.startswith("drop")
            ]

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

    run_visitors(ir_module, [DropByNameVisitor()])

    assert [cls.name for cls in ir_module.classes] == ["KeepClass"]
    assert [func.name for func in ir_module.functions] == ["keep_func"]
    assert [cls.name for cls in keep_class.classes] == ["KeepNested"]
    assert [method.function.name for method in keep_class.methods] == ["keep_method"]


def test_visitor_runner_visits_functions_in_module_and_methods() -> None:
    class RenameVisitedFunctionsVisitor(NodeVisitor):
        def visit_function(self, node: IRFunction, module: IRModule) -> None:
            assert module.full_name == QualifiedName.from_str("pkg.mod")
            node.name = f"visited_{node.name}"

    method = IRMethod(function=IRFunction(name="m"), decorator=None)
    ir_class = IRClass(name="C", methods=[method])
    ir_module = IRModule(
        full_name=QualifiedName.from_str("pkg.mod"),
        classes=[ir_class],
        functions=[IRFunction(name="f")],
    )

    run_visitors(ir_module, [RenameVisitedFunctionsVisitor()])

    assert [func.name for func in ir_module.functions] == ["visited_f"]
    assert [m.function.name for m in ir_class.methods] == ["visited_m"]
