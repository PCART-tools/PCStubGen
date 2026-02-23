from __future__ import annotations

from typing import Any

from ...IR import (
    IRFunction, IRVariable, ResolvedType, IRValue, QualifiedName, InvalidExpression
)
from ..NodeVisitor import NodeVisitor

class FixNumpyArrayDimAnnotationVisitor(NodeVisitor):
    __array_names = {
        QualifiedName.from_str("numpy.ndarray"),
        *(
            QualifiedName.from_str(f"scipy.sparse.{storage}_{arr}")
            for storage in ["bsr", "coo", "csr", "csc", "dia", "dok", "lil"]
            for arr in ["array", "matrix"]
        ),
    }
    __annotated_name = QualifiedName.from_str("typing.Annotated")
    numpy_primitive_types = {
        QualifiedName.from_str(name) for name in (
            "bool",
            *map(
                lambda name: f"numpy.{name}",
                (
                    "uint8", "int8", "uint16", "int16", "uint32", "int32",
                    "uint64", "int64", "float16", "float32", "float64",
                    "complex32", "complex64", "longcomplex",
                ),
            ),
        )
    }
    __DIM_VARS = ["n", "m"]

    def visit_function(self, node: IRFunction) -> None:
        if node.return_annotation:
            node.return_annotation = self._fix_type(node.return_annotation)
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

        # 首先递归修复参数
        if annotation.parameters:
            annotation.parameters = [self._fix_type(p) for p in annotation.parameters]
            
        # 检查它是否匹配 ARRAY_T[PRIMITIVE_TYPE[*DIMS], *FLAGS]
        if (
            annotation.name not in self.__array_names
            or annotation.parameters is None
            or len(annotation.parameters) == 0
        ):
            return annotation

        scalar_with_dims = annotation.parameters[0]
        flags = annotation.parameters[1:]

        if (
            not isinstance(scalar_with_dims, ResolvedType)
            or scalar_with_dims.name not in self.numpy_primitive_types
        ):
            return annotation
            
        # 构造 Annotated[ARRAY_T, PRIMITIVE_TYPE, FixedSize/DynamicSize[*DIMS], *FLAGS]
        
        array_type = ResolvedType(name=annotation.name) 
        
        new_params = [
            array_type,
            ResolvedType(scalar_with_dims.name)
        ]
        
        if scalar_with_dims.parameters:
            dims = self.__to_dims(scalar_with_dims.parameters)
            if dims is not None and len(dims) > 0:
                 size_val = self.__wrap_with_size_helper(dims)
                 new_params.append(size_val)
                 
        new_params.extend(flags)
        
        return ResolvedType(
            name=self.__annotated_name,
            parameters=new_params
        )

    def __to_dims(self, dimensions: list[ResolvedType | IRValue | InvalidExpression]) -> list[int | str] | None:
        result = []
        for dim_param in dimensions:
            if isinstance(dim_param, IRValue):
                try:
                    dim = int(dim_param.repr)
                except ValueError:
                    return None
            elif isinstance(dim_param, ResolvedType):
                dim = str(dim_param.name) # 将 QualifiedName 转为 str
                if dim not in self.__DIM_VARS:
                    return None
            else:
                return None
            result.append(dim)
        return result

    def __wrap_with_size_helper(self, dims: list[int | str]) -> IRValue:
        if all(isinstance(d, int) for d in dims):
            helper_name = "FixedSize"
            args_repr = ", ".join(str(d) for d in dims)
        else:
            helper_name = "DynamicSize"
            args_repr = ", ".join(repr(d) for d in dims)

        return IRValue(
            repr=f"pybind11_stubgen.typing_ext.{helper_name}({args_repr})",
            is_print_safe=True,
        )
