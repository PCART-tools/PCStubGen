from __future__ import annotations

import re

from ...IR import IRVariable, IRFunction, IRValue
from ..NodeVisitor import NodeVisitor

class FixValueReprRandomAddressVisitor(NodeVisitor):
    """
    从值表示中移除随机内存地址。
    
    将类似 `<foo.bar.Baz object at 0x7fdfdf8b5f20>` 的模式
    转换为 `<foo.bar.Baz object>`。
    """
    
    _pattern = re.compile(
        r"<(?P<name>[\w.]+) object "
        r"(?P<capsule>\w+\s)*at "
        r"(?P<address>0x[a-fA-F0-9]+)>"
    )
    
    def visit_variable(self, node: IRVariable) -> None:
        if node.value:
            node.value = self._fix_value(node.value)
        super().visit_variable(node)
    
    def visit_function(self, node: IRFunction) -> None:
        for arg in node.args:
            if arg.default and isinstance(arg.default, IRValue):
                arg.default = self._fix_value(arg.default)
        super().visit_function(node)
    
    def _fix_value(self, value: IRValue) -> IRValue:
        new_repr = self._pattern.sub(r"<\g<name> object>", value.repr)
        if new_repr != value.repr:
            return IRValue(repr=new_repr, is_print_safe=value.is_print_safe)
        return value
