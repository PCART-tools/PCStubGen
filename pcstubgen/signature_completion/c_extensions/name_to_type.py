from ...type_system.types import RawType, Type, UnionType


# 参数类型推导
PY_TYPE_OBJECT_NAME_TO_TYPE: dict[str, Type] = {
    "PyList_Type": RawType("list"),
    "PyTuple_Type": RawType("tuple"),
    "PyDict_Type": RawType("dict"),
    "PyUnicode_Type": RawType("str"),
    "PyLong_Type": RawType("int"),
    "PyFloat_Type": RawType("float"),
    "PyBool_Type": RawType("bool"),
    "PyBytes_Type": RawType("bytes"),
    "PyByteArray_Type": RawType("bytearray"),
    "PySet_Type": RawType("set"),
    "PyFrozenSet_Type": RawType("frozenset"),
    "PyType_Type": RawType("type"),
    "PyBaseObject_Type": RawType("object"),

    # numpy
    "PyArray_Type": RawType("numpy.ndarray", imports=("numpy",)),
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
    "PyBool_FromLong": RawType("bool"),

    # int
    "PyLong_FromLong": RawType("int"),
    "PyLong_FromUnsignedLong": RawType("int"),
    "PyLong_FromSsize_t": RawType("int"),
    "PyLong_FromSize_t": RawType("int"),
    "PyLong_FromLongLong": RawType("int"),
    "PyLong_FromUnsignedLongLong": RawType("int"),
    "PyLong_FromDouble": RawType("int"),
    "PyLong_FromString": RawType("int"),
    "PyLong_FromUnicodeObject": RawType("int"),
    "PyLong_FromVoidPtr": RawType("int"),

    # float
    "PyFloat_FromString": RawType("float"),
    "PyFloat_FromDouble": RawType("float"),

    # complex
    "PyComplex_FromCComplex": RawType("complex"),
    "PyComplex_FromDoubles": RawType("complex"),

    # str
    "PyUnicode_New": RawType("str"),
    "PyUnicode_FromKindAndData": RawType("str"),
    "PyUnicode_FromString": RawType("str"),
    "PyUnicode_FromStringAndSize": RawType("str"),
    "PyUnicode_FromFormat": RawType("str"),
    "PyUnicode_FromFormatV": RawType("str"),
    "PyUnicode_FromObject": RawType("str"),
    "PyUnicode_FromEncodedObject": RawType("str"),
    "PyUnicode_FromWideChar": RawType("str"),
    "PyUnicode_Decode": RawType("str"),
    "PyUnicode_DecodeUTF8": RawType("str"),
    "PyUnicode_DecodeUTF8Stateful": RawType("str"),
    "PyUnicode_DecodeUTF32": RawType("str"),
    "PyUnicode_DecodeUTF32Stateful": RawType("str"),
    "PyUnicode_DecodeUTF16": RawType("str"),
    "PyUnicode_DecodeUTF16Stateful": RawType("str"),
    "PyUnicode_DecodeUTF7": RawType("str"),
    "PyUnicode_DecodeUTF7Stateful": RawType("str"),
    "PyUnicode_DecodeUnicodeEscape": RawType("str"),
    "PyUnicode_DecodeRawUnicodeEscape": RawType("str"),
    "PyUnicode_DecodeLatin1": RawType("str"),
    "PyUnicode_DecodeASCII": RawType("str"),
    "PyUnicode_DecodeCharmap": RawType("str"),
    "PyUnicode_DecodeLocaleAndSize": RawType("str"),
    "PyUnicode_DecodeLocale": RawType("str"),
    "PyUnicode_DecodeFSDefaultAndSize": RawType("str"),
    "PyUnicode_DecodeFSDefault": RawType("str"),
    "PyUnicode_Translate": RawType("str"),
    "PyUnicode_DecodeMBCS": RawType("str"),
    "PyUnicode_DecodeMBCSStateful": RawType("str"),
    "PyUnicode_DecodeCodePageStateful": RawType("str"),
    "PyUnicode_Substring": RawType("str"),
    "PyUnicode_Concat": RawType("str"),
    "PyUnicode_Join": RawType("str"),
    "PyUnicode_Replace": RawType("str"),
    "PyUnicode_Format": RawType("str"),
    "PyUnicode_InternFromString": RawType("str"),

    # bytes
    "PyBytes_FromString": RawType("bytes"),
    "PyBytes_FromStringAndSize": RawType("bytes"),
    "PyBytes_FromFormat": RawType("bytes"),
    "PyBytes_FromFormatV": RawType("bytes"),
    "PyBytes_FromObject": RawType("bytes"),
    "PyUnicode_AsEncodedString": RawType("bytes"),
    "PyUnicode_AsUTF8String": RawType("bytes"),
    "PyUnicode_AsUTF32String": RawType("bytes"),
    "PyUnicode_AsUTF16String": RawType("bytes"),
    "PyUnicode_AsUnicodeEscapeString": RawType("bytes"),
    "PyUnicode_AsRawUnicodeEscapeString": RawType("bytes"),
    "PyUnicode_AsLatin1String": RawType("bytes"),
    "PyUnicode_AsASCIIString": RawType("bytes"),
    "PyUnicode_AsCharmapString": RawType("bytes"),
    "PyUnicode_EncodeLocale": RawType("bytes"),
    "PyUnicode_EncodeFSDefault": RawType("bytes"),
    "PyUnicode_AsMBCSString": RawType("bytes"),
    "PyUnicode_EncodeCodePage": RawType("bytes"),

    # bytearray
    "PyByteArray_FromObject": RawType("bytearray"),
    "PyByteArray_FromStringAndSize": RawType("bytearray"),
    "PyByteArray_Concat": RawType("bytearray"),

    # slice
    "PySlice_New": RawType("slice"),

    # memoryview
    "PyMemoryView_FromObject": RawType("memoryview"),
    "PyMemoryView_FromMemory": RawType("memoryview"),
    "PyMemoryView_FromBuffer": RawType("memoryview"),
    "PyMemoryView_GetContiguous": RawType("memoryview"),

    # tuple
    "PyTuple_New": RawType("tuple"),
    "PyTuple_Pack": RawType("tuple"),
    "PyTuple_GetSlice": RawType("tuple"),
    "PyList_AsTuple": RawType("tuple"),
    "PyUnicode_Partition": RawType("tuple"),
    "PyUnicode_RPartition": RawType("tuple"),

    # list
    "PyList_New": RawType("list"),
    "PyList_GetSlice": RawType("list"),
    "PyUnicode_Split": RawType("list"),
    "PyUnicode_RSplit": RawType("list"),
    "PyUnicode_Splitlines": RawType("list"),
    "PyDict_Items": RawType("list"),
    "PyDict_Keys": RawType("list"),
    "PyDict_Values": RawType("list"),

    # dict
    "PyDict_New": RawType("dict"),
    "PyDict_Copy": RawType("dict"),

    # set
    "PySet_New": RawType("set"),

    # frozenset
    "PyFrozenSet_New": RawType("frozenset"),

    # numpy
    "PyArray_Return": UnionType(
        (
            RawType("numpy.ndarray", imports=("numpy",)),
            RawType("numpy.generic", imports=("numpy",)),
        )
    ),
}

BUILD_CONVERTER_NAME_TO_TYPE: dict[str, str] = {

}

OBJECT_NAME_TO_TYPE: dict[str, Type] = {
    "_Py_NoneStruct": RawType("None"),
    "_Py_TrueStruct": RawType("bool"),
    "_Py_FalseStruct": RawType("bool"),
}
