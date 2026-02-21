from __future__ import annotations

from typing import Any

from ...IR import (
    IRModule, IRFunction, IRVariable, ResolvedType, IRValue,
    QualifiedName, IRImport, IRTypeVar, InvalidExpression
)
from ..NodeVisitor import NodeVisitor
from .FixNumpyArrayDimAnnotationVisitor import FixNumpyArrayDimAnnotationVisitor

class FixNumpyArrayDimTypeVarVisitor(NodeVisitor):
    __array_names = {QualifiedName.from_str("numpy.ndarray")}
    numpy_primitive_types = FixNumpyArrayDimAnnotationVisitor.numpy_primitive_types
    __DIM_VARS = set()

    def __init__(self):
        self.__DIM_VARS = set()

    def visit_module(self, node: IRModule) -> None:
        self.__DIM_VARS.clear()
        super().visit_module(node)
        
        if self.__DIM_VARS:
            node.imports.add(
                IRImport(name=None, origin=QualifiedName.from_str("typing"))
            )
            for name in self.__DIM_VARS:
                 node.type_vars.append(
                    IRTypeVar(
                        name=name,
                        bound=ResolvedType(name=QualifiedName.from_str("int")),
                    )
                )

    def visit_function(self, node: IRFunction) -> None:
        if node.returns:
            node.returns = self._fix_type(node.returns)
        for arg in node.args:
            if arg.annotation:
                arg.annotation = self._fix_type(arg.annotation)
        super().visit_function(node)
    
    def visit_variable(self, node: IRVariable) -> None:
        if node.annotation:
            node.annotation = self._fix_type(node.annotation)
        super().visit_variable(node)

    def _fix_type(self, annotation: Any) -> Any:
        if not isinstance(annotation, ResolvedType):
            return annotation
            
        if len(annotation.name) == 1 and len(annotation.name[0]) == 1:
             # 在 NumPy 文档中，常用单字母维度占位（m/n/k 等），这里转为 TypeVar
             name_str = annotation.name[0]
             new_name = name_str.upper()
             annotation.name = QualifiedName.from_str(new_name)
             self.__DIM_VARS.add(new_name)
             
        if annotation.parameters:
            annotation.parameters = [self._fix_type(p) for p in annotation.parameters]

        if annotation.name not in self.__array_names:
            return annotation
            
        if annotation.parameters is None or len(annotation.parameters) == 0:
             return ResolvedType(
                 name=annotation.name,
                 parameters=[
                     ResolvedType(name=QualifiedName.from_str("typing.Any")),
                     ResolvedType(
                         name=QualifiedName.from_str("numpy.dtype"),
                         parameters=[ResolvedType(name=QualifiedName.from_str("typing.Any"))]
                     )
                 ]
             )
             
        scalar_with_dims = annotation.parameters[0]
        
        if (
            not isinstance(scalar_with_dims, ResolvedType)
            or scalar_with_dims.name not in self.numpy_primitive_types
        ):
            return annotation
            
        name = scalar_with_dims.name
        if str(name) == "bool":
            name = QualifiedName.from_str("numpy.bool_")
            
        dtype = ResolvedType(
            name=QualifiedName.from_str("numpy.dtype"),
            parameters=[ResolvedType(name=name)]
        )
        
        shape = ResolvedType(name=QualifiedName.from_str("typing.Any"))
        
        if scalar_with_dims.parameters:
            dims = self.__to_dims(scalar_with_dims.parameters)
            if dims is not None:
                shape = ResolvedType(name=QualifiedName.from_str("tuple"))
                shape.parameters = []
                for dim in dims:
                    if isinstance(dim, int):
                         literal = ResolvedType(name=QualifiedName.from_str("typing.Literal"))
                         literal.parameters = [IRValue(repr=str(dim), is_print_safe=True)]
                         shape.parameters.append(literal)
                    else:
                         shape.parameters.append(ResolvedType(name=QualifiedName.from_str(dim)))
                         
        return ResolvedType(
            name=annotation.name,
            parameters=[shape, dtype]
        )

    def __to_dims(self, dimensions: list[ResolvedType | IRValue | InvalidExpression]) -> list[int | str] | None:
         result = []
         for dim_param in dimensions:
             if isinstance(dim_param, IRValue):
                 try:
                     result.append(int(dim_param.repr))
                 except ValueError:
                     return None
             elif isinstance(dim_param, ResolvedType):
                 result.append(str(dim_param.name))
             else:
                 return None
         return result
