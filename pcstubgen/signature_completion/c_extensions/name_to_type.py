from ...type_system.types import RawType, Type, UnionType


def _raw(text: str, *, imports: tuple[str, ...] = ()) -> RawType:
    return RawType(text, imports=imports)


# 参数类型推导
PY_TYPE_OBJECT_NAME_TO_TYPE: dict[str, Type] = {
    "PyList_Type": _raw("list"),
    "PyTuple_Type": _raw("tuple"),
    "PyDict_Type": _raw("dict"),
    "PyUnicode_Type": _raw("str"),
    "PyLong_Type": _raw("int"),
    "PyFloat_Type": _raw("float"),
    "PyBool_Type": _raw("bool"),
    "PyBytes_Type": _raw("bytes"),
    "PyByteArray_Type": _raw("bytearray"),
    "PySet_Type": _raw("set"),
    "PyFrozenSet_Type": _raw("frozenset"),
    "PyType_Type": _raw("type"),
    "PyBaseObject_Type": _raw("object"),

    # numpy
    "PyArray_Type": _raw("numpy.ndarray", imports=("numpy",)),
}

# 参数类型推导，O& converter
PARSE_CONVERTER_NAME_TO_TYPE: dict[str, str] = {
    # scipy
    "NI_ObjectToInputArray": "",
    "NI_ObjectToOptionalInputArray": "",
    "NI_ObjectToOutputArray": "",
    "NI_ObjectToOptionalOutputArray": "",
    "NI_ObjectToIoArray": ""
}

# 参数默认值推导
DEFAULT_IDENTIFIER_TO_VALUE: dict[str, str] = {
    "Py_None": "None",
    "Py_True": "True",
    "Py_False": "False",
}

# 返回值类型推导
FUNCTION_NAME_TO_TYPE: dict[str, Type] = {
    # bool
    "PyBool_FromLong": _raw("bool"),

    # int
    "PyLong_FromLong": _raw("int"),
    "PyLong_FromUnsignedLong": _raw("int"),
    "PyLong_FromSsize_t": _raw("int"),
    "PyLong_FromSize_t": _raw("int"),
    "PyLong_FromLongLong": _raw("int"),
    "PyLong_FromUnsignedLongLong": _raw("int"),
    "PyLong_FromDouble": _raw("int"),
    "PyLong_FromString": _raw("int"),
    "PyLong_FromUnicodeObject": _raw("int"),
    "PyLong_FromVoidPtr": _raw("int"),

    # float
    "PyFloat_FromString": _raw("float"),
    "PyFloat_FromDouble": _raw("float"),

    # complex
    "PyComplex_FromCComplex": _raw("complex"),
    "PyComplex_FromDoubles": _raw("complex"),

    # str
    "PyUnicode_New": _raw("str"),
    "PyUnicode_FromKindAndData": _raw("str"),
    "PyUnicode_FromString": _raw("str"),
    "PyUnicode_FromStringAndSize": _raw("str"),
    "PyUnicode_FromFormat": _raw("str"),
    "PyUnicode_FromFormatV": _raw("str"),
    "PyUnicode_FromObject": _raw("str"),
    "PyUnicode_FromEncodedObject": _raw("str"),
    "PyUnicode_FromWideChar": _raw("str"),
    "PyUnicode_Decode": _raw("str"),
    "PyUnicode_DecodeUTF8": _raw("str"),
    "PyUnicode_DecodeUTF8Stateful": _raw("str"),
    "PyUnicode_DecodeUTF32": _raw("str"),
    "PyUnicode_DecodeUTF32Stateful": _raw("str"),
    "PyUnicode_DecodeUTF16": _raw("str"),
    "PyUnicode_DecodeUTF16Stateful": _raw("str"),
    "PyUnicode_DecodeUTF7": _raw("str"),
    "PyUnicode_DecodeUTF7Stateful": _raw("str"),
    "PyUnicode_DecodeUnicodeEscape": _raw("str"),
    "PyUnicode_DecodeRawUnicodeEscape": _raw("str"),
    "PyUnicode_DecodeLatin1": _raw("str"),
    "PyUnicode_DecodeASCII": _raw("str"),
    "PyUnicode_DecodeCharmap": _raw("str"),
    "PyUnicode_DecodeLocaleAndSize": _raw("str"),
    "PyUnicode_DecodeLocale": _raw("str"),
    "PyUnicode_DecodeFSDefaultAndSize": _raw("str"),
    "PyUnicode_DecodeFSDefault": _raw("str"),
    "PyUnicode_Translate": _raw("str"),
    "PyUnicode_DecodeMBCS": _raw("str"),
    "PyUnicode_DecodeMBCSStateful": _raw("str"),
    "PyUnicode_DecodeCodePageStateful": _raw("str"),
    "PyUnicode_Substring": _raw("str"),
    "PyUnicode_Concat": _raw("str"),
    "PyUnicode_Join": _raw("str"),
    "PyUnicode_Replace": _raw("str"),
    "PyUnicode_Format": _raw("str"),
    "PyUnicode_InternFromString": _raw("str"),

    # bytes
    "PyBytes_FromString": _raw("bytes"),
    "PyBytes_FromStringAndSize": _raw("bytes"),
    "PyBytes_FromFormat": _raw("bytes"),
    "PyBytes_FromFormatV": _raw("bytes"),
    "PyBytes_FromObject": _raw("bytes"),
    "PyUnicode_AsEncodedString": _raw("bytes"),
    "PyUnicode_AsUTF8String": _raw("bytes"),
    "PyUnicode_AsUTF32String": _raw("bytes"),
    "PyUnicode_AsUTF16String": _raw("bytes"),
    "PyUnicode_AsUnicodeEscapeString": _raw("bytes"),
    "PyUnicode_AsRawUnicodeEscapeString": _raw("bytes"),
    "PyUnicode_AsLatin1String": _raw("bytes"),
    "PyUnicode_AsASCIIString": _raw("bytes"),
    "PyUnicode_AsCharmapString": _raw("bytes"),
    "PyUnicode_EncodeLocale": _raw("bytes"),
    "PyUnicode_EncodeFSDefault": _raw("bytes"),
    "PyUnicode_AsMBCSString": _raw("bytes"),
    "PyUnicode_EncodeCodePage": _raw("bytes"),

    # bytearray
    "PyByteArray_FromObject": _raw("bytearray"),
    "PyByteArray_FromStringAndSize": _raw("bytearray"),
    "PyByteArray_Concat": _raw("bytearray"),

    # slice
    "PySlice_New": _raw("slice"),

    # memoryview
    "PyMemoryView_FromObject": _raw("memoryview"),
    "PyMemoryView_FromMemory": _raw("memoryview"),
    "PyMemoryView_FromBuffer": _raw("memoryview"),
    "PyMemoryView_GetContiguous": _raw("memoryview"),

    # tuple
    "PyTuple_New": _raw("tuple"),
    "PyTuple_Pack": _raw("tuple"),
    "PyTuple_GetSlice": _raw("tuple"),
    "PyList_AsTuple": _raw("tuple"),
    "PyUnicode_Partition": _raw("tuple"),
    "PyUnicode_RPartition": _raw("tuple"),

    # list
    "PyList_New": _raw("list"),
    "PyList_GetSlice": _raw("list"),
    "PyUnicode_Split": _raw("list"),
    "PyUnicode_RSplit": _raw("list"),
    "PyUnicode_Splitlines": _raw("list"),
    "PyDict_Items": _raw("list"),
    "PyDict_Keys": _raw("list"),
    "PyDict_Values": _raw("list"),

    # dict
    "PyDict_New": _raw("dict"),
    "PyDict_Copy": _raw("dict"),

    # set
    "PySet_New": _raw("set"),

    # frozenset
    "PyFrozenSet_New": _raw("frozenset"),

    # numpy
    "PyArray_Return": UnionType(
        (
            _raw("numpy.ndarray", imports=("numpy",)),
            _raw("numpy.generic", imports=("numpy",)),
        )
    ),
}

BUILD_CONVERTER_NAME_TO_TYPE: dict[str, str] = {

}

OBJECT_NAME_TO_TYPE: dict[str, Type] = {
    "_Py_NoneStruct": _raw("None"),
    "_Py_TrueStruct": _raw("bool"),
    "_Py_FalseStruct": _raw("bool"),
}
