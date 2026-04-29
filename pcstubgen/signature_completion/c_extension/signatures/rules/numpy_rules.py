from __future__ import annotations

from collections.abc import Callable

from clang.cindex import Cursor
from loguru import logger

from .....models import Argument, ArgumentKind
from .....type_models import RawType, Type, UnionType
from ...libclang.ast_utils import get_string_literal, is_nullptr_or_zero, unwrap_transparent

_NDARRAY_TYPE = RawType("numpy.ndarray", imports=("numpy",))
_NDARRAY_OR_NONE_TYPE = UnionType((_NDARRAY_TYPE, RawType.none_))
_DTYPE_TYPE = RawType("numpy.dtype", imports=("numpy",))
_DTYPE_META_TYPE = RawType("type[numpy.dtype]", imports=("numpy",))
_DTYPE_LIKE_TYPE = RawType("numpy.typing.DTypeLike", imports=("numpy.typing",))
_DTYPE_LIKE_OR_NONE_TYPE = UnionType((_DTYPE_LIKE_TYPE, RawType.none_))
_ARRAY_LIKE_TYPE = RawType("numpy.typing.ArrayLike", imports=("numpy.typing",))
_ARRAY_LIKE_OR_NONE_TYPE = UnionType((_ARRAY_LIKE_TYPE, RawType.none_))
_BUSDAYCALENDAR_TYPE = RawType("numpy.busdaycalendar", imports=("numpy",))
_UFUNC_TYPE = RawType("numpy.ufunc", imports=("numpy",))
_LIST_TYPE = RawType("list")
_INT_OR_NONE_TYPE = UnionType((RawType.int_, RawType.none_))
_BOOL_OR_NONE_TYPE = UnionType((RawType.bool_, RawType.none_))
_INTP_OR_SHAPE_TYPE = UnionType((RawType.int_, RawType("tuple[int, ...]")))
_INTP_OR_SHAPE_OR_NONE_TYPE = UnionType((RawType.int_, RawType("tuple[int, ...]"), RawType.none_))
_SEQUENCE_STR_TYPE = RawType("collections.abc.Sequence[str]", imports=("collections.abc",))
_ORDER_TYPE = RawType(
    'typing.Literal["K", "A", "C", "F"]',
    imports=("typing",),
)
_ORDER_OR_NONE_TYPE = UnionType((_ORDER_TYPE, RawType.none_))
_BYTEORDER_TYPE = RawType(
    'typing.Literal["S", "<", "L", "little", ">", "B", "big", "=", "N", "native", "|", "I"]',
    imports=("typing",),
)
_CASTING_TYPE = RawType(
    'typing.Literal["no", "equiv", "safe", "same_kind", "unsafe"]',
    imports=("typing",),
)
_CASTING_WITH_SAME_VALUE_TYPE = RawType(
    'typing.Literal["no", "equiv", "safe", "same_kind", "same_value", "unsafe"]',
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
_CLIPMODE_TYPE = UnionType((_CLIPMODE_STRING_TYPE, RawType.int_))
_CORRELATEMODE_TYPE = RawType(
    'typing.Literal["valid", "same", "full"]',
    imports=("typing",),
)
_COPY_MODE_TYPE = UnionType(
    (
        RawType.bool_,
        RawType('typing.Literal[False, True, 2]', imports=("typing",)),
        RawType.none_,
    )
)
_DEVICE_TYPE = RawType('typing.Literal["cpu"] | None', imports=("typing",))
_ERRMODE_TYPE = RawType(
    'typing.Literal["ignore", "warn", "raise", "call", "print", "log"] | None',
    imports=("typing",),
)
_TRIMMODE_TYPE = RawType(
    'typing.Literal["k", ".", "0", "-"]',
    imports=("typing",),
)
_PYSCALAR_MODE_TYPE = RawType(
    'typing.Literal["convert", "preserve", "convert_if_no_array"]',
    imports=("typing",),
)
_BUSDAY_ROLL_TYPE = RawType(
    'typing.Literal["raise", "nat", "forward", "following", "backward", "preceding", "modifiedfollowing", "modifiedpreceding"]',
    imports=("typing",),
)
_DLPACK_DEVICE_TYPE = RawType("tuple[int, int] | None")
_STR_OR_NONE_TYPE = UnionType((RawType.str_, RawType.none_))

PY_ARG_PARSE_TYPE_OBJECT_NAME_TO_TYPE: dict[str, Type] = {
    "PyArray_Type": _NDARRAY_TYPE,
    "PyArrayDescr_Type": _DTYPE_TYPE,
    "PyArrayDTypeMeta_Type": _DTYPE_META_TYPE,
    "NpyBusDayCalendar_Type": _BUSDAYCALENDAR_TYPE,
    "PyUFunc_Type": _UFUNC_TYPE,
}

PY_ARG_PARSE_CONVERTER_NAME_TO_TYPE: dict[str, Type] = {
    "NI_ObjectToInputArray": _NDARRAY_TYPE,
    "NI_ObjectToOutputArray": _NDARRAY_TYPE,
    "NI_ObjectToInputOutputArray": _NDARRAY_TYPE,
    "NI_ObjectToOptionalInputArray": _NDARRAY_OR_NONE_TYPE,
    "NI_ObjectToOptionalOutputArray": _NDARRAY_OR_NONE_TYPE,
    "PyArray_IntpConverter": _INTP_OR_SHAPE_TYPE,
    "PyArray_IntpFromPyIntConverter": RawType.int_,
    "PyArray_OptionalIntpConverter": _INTP_OR_SHAPE_OR_NONE_TYPE,
    "PyArray_OutputConverter": _NDARRAY_OR_NONE_TYPE,
    "PyArray_Converter": _ARRAY_LIKE_TYPE,
    "PyArray_AxisConverter": _INT_OR_NONE_TYPE,
    "PyArray_BoolConverter": RawType.bool_,
    "PyArray_OptionalBoolConverter": _BOOL_OR_NONE_TYPE,
    "PyArray_OrderConverter": _ORDER_OR_NONE_TYPE,
    "PyArray_ByteorderConverter": _BYTEORDER_TYPE,
    "PyArray_CastingConverter": _CASTING_TYPE,
    "PyArray_CastingConverterSameValue": _CASTING_WITH_SAME_VALUE_TYPE,
    "PyArray_SearchsideConverter": _SEARCHSIDE_TYPE,
    "PyArray_SelectkindConverter": _SELECTKIND_TYPE,
    "PyArray_SortkindConverter": _SORTKIND_TYPE,
    "PyArray_ClipmodeConverter": _CLIPMODE_TYPE,
    "PyArray_CorrelatemodeConverter": _CORRELATEMODE_TYPE,
    "PyArray_CopyConverter": _COPY_MODE_TYPE,
    "PyArray_AsTypeCopyConverter": RawType.bool_,
    "PyArray_DeviceConverterOptional": _DEVICE_TYPE,
    "PyArray_PythonPyIntFromInt": RawType.int_,
    "PyArray_DTypeOrDescrConverterRequired": _DTYPE_LIKE_TYPE,
    "PyArray_DTypeOrDescrConverterOptional": _DTYPE_LIKE_OR_NONE_TYPE,
    "PyArray_DescrConverter": _DTYPE_LIKE_TYPE,
    "PyArray_DescrConverter2": _DTYPE_LIKE_OR_NONE_TYPE,
    "PyArray_WeekMaskConverter": _ARRAY_LIKE_TYPE,
    "PyArray_HolidaysConverter": _ARRAY_LIKE_OR_NONE_TYPE,
    "PyArray_BusDayRollConverter": _BUSDAY_ROLL_TYPE,
    "device_converter": _DLPACK_DEVICE_TYPE,
    "parse_control_character": _STR_OR_NONE_TYPE,
    "NpyIter_GlobalFlagsConverter": _SEQUENCE_STR_TYPE,
    "errmodeconverter": _ERRMODE_TYPE,
    "trimmode_converter": _TRIMMODE_TYPE,
    "pyscalar_mode_conv": _PYSCALAR_MODE_TYPE,
}

OBJECT_NAME_TO_TYPE: dict[str, Type] = {}

OBJECT_USE_FUNCTION_NAME_TO_TYPE: dict[str, Type] = {
    "PyArray_Check": _NDARRAY_TYPE,
    "PyArray_CheckExact": _NDARRAY_TYPE,
}


CALL_NAME_TO_TYPE: dict[str, Type] = {
    "PyArray_ContiguousFromObject": _NDARRAY_TYPE,
    "PyArray_Arange": _NDARRAY_TYPE,
    "PyArray_SimpleNew": _NDARRAY_TYPE,
    "PyArray_FROMANY": _NDARRAY_TYPE,
    "PyArray_Empty_int": _NDARRAY_TYPE,
    "PyArray_FromBuffer": _NDARRAY_TYPE,
    "PyArray_FromString": _NDARRAY_TYPE,
    "PyArray_FromIter": _NDARRAY_TYPE,
    "PyArray_Where": _NDARRAY_TYPE,
    "PyArray_NewCopy": _NDARRAY_TYPE,
    "PyArray_View": _NDARRAY_TYPE,
    "PyArray_NewFromDescr": _NDARRAY_TYPE,
    "PyArray_NewFromDescrAndBase": _NDARRAY_TYPE,
    "PyArray_Ravel": _NDARRAY_TYPE,
    "PyArray_Flatten": _NDARRAY_TYPE,
    "PyArray_SwapAxes": _NDARRAY_TYPE,
    "PyArray_DescrNewByteorder": _DTYPE_TYPE,
    "PyArray_DescrFromType": _DTYPE_TYPE,
    "PyArray_ToList": _LIST_TYPE,
    "PyArray_ToString": RawType.bytes_,
    "PyArray_Dumps": RawType.bytes_,
    "pylong_from_int128": RawType.int_,
    "PyArray_Return": _NDARRAY_TYPE,
}


def infer_npy_parse_arguments(
    call_expr: Cursor,
    *,
    infer_name_func: Callable[[list[Cursor]], str],
    infer_converter_type_func: Callable[[Cursor], Type],
    infer_refined_object_type_func: Callable[[Cursor], Type],
    infer_default_value_func: Callable[[Cursor, Type], str],
) -> list[Argument]:
    """
    将 `npy_parse_arguments` 调用解析成参数列表。

    NumPy 源码里通常写成：
    `npy_parse_arguments(funcname, args, len_args, kwnames, ...)`

    但 libclang 在宏展开后看到的调用形态实际是：
    `_npy_parse_arguments(funcname, &__argparse_cache, args, len_args, kwnames, ...)`

    因此这里解析时，固定参数区按
    `funcname, &__argparse_cache, args, len_args, kwnames`
    处理，其后才是 `name, converter, &slot` 三元组序列。
    """
    args = list(call_expr.get_children())[1:]
    _validate_npy_parse_arguments_shape(args)

    arguments: list[Argument] = []
    triplets = args[5:]
    arg_index = 0
    while arg_index < len(triplets):
        name_cursor = unwrap_transparent(triplets[arg_index])
        converter_cursor = unwrap_transparent(triplets[arg_index + 1])
        slot_cursor = triplets[arg_index + 2]
        arg_index += 3

        if (
            is_nullptr_or_zero(name_cursor)
            and is_nullptr_or_zero(converter_cursor)
            and is_nullptr_or_zero(unwrap_transparent(slot_cursor))
        ):
            if arg_index != len(triplets):
                raise RuntimeError("npy_parse_arguments sentinel 后仍有多余参数。")
            break

        if is_nullptr_or_zero(name_cursor):
            raise RuntimeError("npy_parse_arguments 参数名槽位不能为 NULL。")
        if is_nullptr_or_zero(unwrap_transparent(slot_cursor)):
            raise RuntimeError("npy_parse_arguments 输出槽位不能为 NULL。")

        argument_name_spec = get_string_literal(name_cursor)
        argument_name, kind, is_optional = _parse_npy_argument_name(argument_name_spec)
        if argument_name == "":
            argument_name = infer_name_func([slot_cursor])

        argument_type = RawType.object_
        if not is_nullptr_or_zero(converter_cursor):
            try:
                argument_type = infer_converter_type_func(converter_cursor)
            except Exception as ex:
                logger.warning(
                    "npy_parse_arguments converter 类型推断失败，回退为 object, reason: {!r}",
                    ex,
                )
        if argument_type == RawType.object_:
            argument_type = infer_refined_object_type_func(slot_cursor)

        default_value = None
        if is_optional:
            try:
                default_value = infer_default_value_func(slot_cursor, argument_type)
            except Exception as ex:
                logger.warning(
                    "npy_parse_arguments 默认值推断失败，回退为 '...', reason: {!r}",
                    ex,
                )
                default_value = "..."

        arguments.append(
            Argument(
                name=argument_name,
                type=argument_type,
                default_value=default_value,
                kind=kind,
            )
        )

    return arguments


def _validate_npy_parse_arguments_shape(args: list[Cursor]) -> None:
    """校验宏展开后的 `npy_parse_arguments` 调用形态。"""
    if len(args) < 7:
        raise RuntimeError("npy_parse_arguments 参数数量不足。")

    triplet_count = len(args) - 5
    if triplet_count < 3 or triplet_count % 3 != 0:
        raise RuntimeError("npy_parse_arguments 三元组序列长度不正确。")


def _parse_npy_argument_name(argument_name_spec: str) -> tuple[str, ArgumentKind, bool]:
    """解析 NumPy 参数名前缀语义。"""
    if argument_name_spec.startswith("$"):
        return argument_name_spec[1:], ArgumentKind.KEYWORD_ONLY, True
    if argument_name_spec.startswith("|"):
        argument_name = argument_name_spec[1:]
        kind = ArgumentKind.POSITIONAL_ONLY
        if argument_name != "":
            kind = ArgumentKind.POSITIONAL_OR_KEYWORD
        return argument_name, kind, True
    if argument_name_spec == "":
        return "", ArgumentKind.POSITIONAL_ONLY, False
    return argument_name_spec, ArgumentKind.POSITIONAL_OR_KEYWORD, False
