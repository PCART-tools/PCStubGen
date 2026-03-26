from __future__ import annotations

import types

from core.ir import QualifiedName
from core.module_builder import build_module


def test_module_builder_keeps_only_tree_functions_and_methods() -> None:
    module = types.ModuleType("pkg")

    def root_function(*args, **kwargs):
        return args, kwargs

    root_function.__module__ = "pkg"
    root_function.__doc__ = "root_function(x: int) -> int"

    class RootClass:
        class_attr = 123

        def method(*args, **kwargs):
            return args, kwargs

        @property
        def prop(self):
            return 1

    RootClass.__module__ = "pkg"
    RootClass.method.__doc__ = "method(self, y: int) -> int"

    sub = types.ModuleType("pkg.sub")

    def sub_function(*args, **kwargs):
        return args, kwargs

    sub_function.__module__ = "pkg.sub"
    sub.sub_function = sub_function

    module.root_function = root_function
    module.alias_function = root_function
    module.RootClass = RootClass
    module.sub = sub
    module.VALUE = 10

    ir_module = build_module(QualifiedName.from_str("pkg"), module)

    assert [func.name for func in ir_module.functions] == ["root_function"]
    assert [cls.name for cls in ir_module.classes] == ["RootClass"]
    assert [sub_mod.Name for sub_mod in ir_module.sub_modules] == ["sub"]

    root_cls = ir_module.classes[0]
    assert [method.function.name for method in root_cls.methods] == ["method"]
    assert root_cls.classes == []

    # 精简后的 IR 不再携带变量/属性/字段等节点
    assert not hasattr(ir_module, "variables")
    assert not hasattr(root_cls, "properties")
    assert not hasattr(root_cls, "fields")
