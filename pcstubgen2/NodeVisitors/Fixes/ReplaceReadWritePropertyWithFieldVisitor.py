from __future__ import annotations

from ...IR import (
    IRClass, IRField, IRVariable, IRProperty
)
from ..NodeVisitor import NodeVisitor

class ReplaceReadWritePropertyWithFieldVisitor(NodeVisitor):
    def visit_class(self, node: IRClass) -> None:
        new_properties = []
        for prop in node.properties:
            field = self._try_convert_to_field(prop)
            if field:
                node.fields.append(field)
            else:
                new_properties.append(prop)
        node.properties = new_properties
        super().visit_class(node)

    def _try_convert_to_field(self, prop: IRProperty) -> IRField | None:
        if (
            prop.doc is None
            and prop.getter is not None
            and prop.setter is not None
            and len(prop.getter.args) == 1
            and len(prop.setter.args) == 2
            and prop.getter.doc is None
            and prop.setter.doc is None
            and prop.getter.returns == prop.setter.args[1].annotation
        ):
             return IRField(
                variable=IRVariable(
                    name=prop.name, annotation=prop.getter.returns, value=None
                ),
                modifier=None,
             )
        return None
