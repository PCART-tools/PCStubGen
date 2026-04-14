from __future__ import annotations

from ....type_models import RawType, Type, UnionType

_NDARRAY_TYPE = RawType("numpy.ndarray", imports=("numpy",))
_NDARRAY_OR_NONE_TYPE = UnionType((_NDARRAY_TYPE, RawType("None")))

PY_ARG_PARSE_TYPE_OBJECT_NAME_TO_TYPE: dict[str, Type] = {
    "PyArray_Type": _NDARRAY_TYPE,
}

PY_ARG_PARSE_CONVERTER_NAME_TO_TYPE: dict[str, Type] = {
    "NI_ObjectToInputArray": _NDARRAY_TYPE,
    "NI_ObjectToOutputArray": _NDARRAY_TYPE,
    "NI_ObjectToInputOutputArray": _NDARRAY_TYPE,
    "NI_ObjectToOptionalInputArray": _NDARRAY_OR_NONE_TYPE,
    "NI_ObjectToOptionalOutputArray": _NDARRAY_OR_NONE_TYPE,
    "PyArray_IntpConverter": RawType("tuple[int, ...]"),
}

OBJECT_NAME_TO_TYPE: dict[str, Type] = {}


FUNCTION_NAME_TO_TYPE: dict[str, Type] = {
    "PyArray_ContiguousFromObject": _NDARRAY_TYPE,
    "PyArray_Arange": _NDARRAY_TYPE,
    "PyArray_SimpleNew": _NDARRAY_TYPE,
    "PyArray_FROMANY": _NDARRAY_TYPE,
    "PyArray_Return": UnionType(
        (
            RawType("numpy.ndarray", imports=("numpy",)),
            RawType("numpy.generic", imports=("numpy",)),
        )
    ),
}
