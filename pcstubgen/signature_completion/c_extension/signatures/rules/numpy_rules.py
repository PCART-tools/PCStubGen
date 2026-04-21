from __future__ import annotations

from .....type_models import RawType, Type, UnionType

_NDARRAY_TYPE = RawType("numpy.ndarray", imports=("numpy",))
_NDARRAY_OR_NONE_TYPE = UnionType((_NDARRAY_TYPE, RawType("None")))
_DTYPE_TYPE = RawType("numpy.dtype", imports=("numpy",))
_BUSDAYCALENDAR_TYPE = RawType("numpy.busdaycalendar", imports=("numpy",))
_UFUNC_TYPE = RawType("numpy.ufunc", imports=("numpy",))
_INT_OR_NONE_TYPE = UnionType((RawType("int"), RawType("None")))
_BOOL_OR_NONE_TYPE = UnionType((RawType("bool"), RawType("None")))
_ORDER_TYPE = RawType(
    'typing.Literal["K", "A", "C", "F"]',
    imports=("typing",),
)
_ORDER_OR_NONE_TYPE = UnionType((_ORDER_TYPE, RawType("None")))
_BYTEORDER_TYPE = RawType(
    'typing.Literal["S", "<", "L", "little", ">", "B", "big", "=", "N", "native", "|", "I"]',
    imports=("typing",),
)
_CASTING_TYPE = RawType(
    'typing.Literal["no", "equiv", "safe", "same_kind", "unsafe"]',
    imports=("typing",),
)
_SEARCHSIDE_TYPE = RawType(
    'typing.Literal["left", "right"]',
    imports=("typing",),
)
_SELECTKIND_TYPE = RawType(
    'typing.Literal["introselect"]',
    imports=("typing",),
)
_SORTKIND_TYPE = RawType(
    'typing.Literal["Q", "quick", "quicksort", "M", "merge", "mergesort", "H", "heap", "heapsort", "S", "stable", "stablesort"]',
    imports=("typing",),
)
_CLIPMODE_STRING_TYPE = RawType(
    'typing.Literal["clip", "wrap", "raise"]',
    imports=("typing",),
)
_CLIPMODE_TYPE = UnionType((_CLIPMODE_STRING_TYPE, RawType("int")))

PY_ARG_PARSE_TYPE_OBJECT_NAME_TO_TYPE: dict[str, Type] = {
    "PyArray_Type": _NDARRAY_TYPE,
    "PyArrayDescr_Type": _DTYPE_TYPE,
    "NpyBusDayCalendar_Type": _BUSDAYCALENDAR_TYPE,
    "PyUFunc_Type": _UFUNC_TYPE,
}

PY_ARG_PARSE_CONVERTER_NAME_TO_TYPE: dict[str, Type] = {
    "NI_ObjectToInputArray": _NDARRAY_TYPE,
    "NI_ObjectToOutputArray": _NDARRAY_TYPE,
    "NI_ObjectToInputOutputArray": _NDARRAY_TYPE,
    "NI_ObjectToOptionalInputArray": _NDARRAY_OR_NONE_TYPE,
    "NI_ObjectToOptionalOutputArray": _NDARRAY_OR_NONE_TYPE,
    "PyArray_IntpConverter": RawType("tuple[int, ...]"),
    "PyArray_OutputConverter": _NDARRAY_OR_NONE_TYPE,
    "PyArray_AxisConverter": _INT_OR_NONE_TYPE,
    "PyArray_BoolConverter": RawType("bool"),
    "PyArray_OptionalBoolConverter": _BOOL_OR_NONE_TYPE,
    "PyArray_OrderConverter": _ORDER_OR_NONE_TYPE,
    "PyArray_ByteorderConverter": _BYTEORDER_TYPE,
    "PyArray_CastingConverter": _CASTING_TYPE,
    "PyArray_SearchsideConverter": _SEARCHSIDE_TYPE,
    "PyArray_SelectkindConverter": _SELECTKIND_TYPE,
    "PyArray_SortkindConverter": _SORTKIND_TYPE,
    "PyArray_ClipmodeConverter": _CLIPMODE_TYPE,
}

OBJECT_NAME_TO_TYPE: dict[str, Type] = {}


CALL_NAME_TO_TYPE: dict[str, Type] = {
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
