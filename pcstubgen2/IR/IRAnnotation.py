from __future__ import annotations

from typing import Union

from .InvalidExpression import InvalidExpression
from .ResolvedType import ResolvedType
from .IRValue import IRValue

# 类型注解
IRAnnotation = Union[ResolvedType, IRValue, InvalidExpression]

__all__ = ["IRAnnotation"]
