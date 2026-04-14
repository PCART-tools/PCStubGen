from __future__ import annotations

from ....type_models import RawType, Type, UnionType

PY_ARG_PARSE_TYPE_OBJECT_NAME_TO_TYPE: dict[str, Type] = {
    "PyArray_Type": RawType("numpy.ndarray", imports=("numpy",)),
}


OBJECT_NAME_TO_TYPE: dict[str, Type] = {}


FUNCTION_NAME_TO_TYPE: dict[str, Type] = {
    "PyArray_Return": UnionType(
        (
            RawType("numpy.ndarray", imports=("numpy",)),
            RawType("numpy.generic", imports=("numpy",)),
        )
    ),
}
