from .types import NamedType, Type, UnionType


# 参数类型推导
PY_TYPE_OBJECT_NAME_TO_TYPE: dict[str, str] = {
    "PyList_Type": "list",
    "PyTuple_Type": "tuple",
    "PyDict_Type": "dict",
    "PyUnicode_Type": "str",
    "PyLong_Type": "int",
    "PyFloat_Type": "float",
    "PyBool_Type": "bool",
    "PyBytes_Type": "bytes",
    "PyByteArray_Type": "bytearray",
    "PySet_Type": "set",
    "PyFrozenSet_Type": "frozenset",
    "PyType_Type": "type",
    "PyBaseObject_Type": "object",

    # numpy
    "PyArray_Type": "numpy.ndarray"
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
    "PyBool_FromLong": NamedType("bool"),

    # int
    "PyLong_FromLong": NamedType("int"),
    "PyLong_FromUnsignedLong": NamedType("int"),
    "PyLong_FromSsize_t": NamedType("int"),
    "PyLong_FromSize_t": NamedType("int"),
    "PyLong_FromLongLong": NamedType("int"),
    "PyLong_FromUnsignedLongLong": NamedType("int"),
    "PyLong_FromDouble": NamedType("int"),
    "PyLong_FromString": NamedType("int"),
    "PyLong_FromUnicodeObject": NamedType("int"),
    "PyLong_FromVoidPtr": NamedType("int"),

    # float
    "PyFloat_FromString": NamedType("float"),
    "PyFloat_FromDouble": NamedType("float"),

    # complex
    "PyComplex_FromCComplex": NamedType("complex"),
    "PyComplex_FromDoubles": NamedType("complex"),

    # str
    "PyUnicode_New": NamedType("str"),
    "PyUnicode_FromKindAndData": NamedType("str"),
    "PyUnicode_FromString": NamedType("str"),
    "PyUnicode_FromStringAndSize": NamedType("str"),
    "PyUnicode_FromFormat": NamedType("str"),
    "PyUnicode_FromFormatV": NamedType("str"),
    "PyUnicode_FromObject": NamedType("str"),
    "PyUnicode_FromEncodedObject": NamedType("str"),
    "PyUnicode_FromWideChar": NamedType("str"),
    "PyUnicode_Decode": NamedType("str"),
    "PyUnicode_DecodeUTF8": NamedType("str"),
    "PyUnicode_DecodeUTF8Stateful": NamedType("str"),
    "PyUnicode_DecodeUTF32": NamedType("str"),
    "PyUnicode_DecodeUTF32Stateful": NamedType("str"),
    "PyUnicode_DecodeUTF16": NamedType("str"),
    "PyUnicode_DecodeUTF16Stateful": NamedType("str"),
    "PyUnicode_DecodeUTF7": NamedType("str"),
    "PyUnicode_DecodeUTF7Stateful": NamedType("str"),
    "PyUnicode_DecodeUnicodeEscape": NamedType("str"),
    "PyUnicode_DecodeRawUnicodeEscape": NamedType("str"),
    "PyUnicode_DecodeLatin1": NamedType("str"),
    "PyUnicode_DecodeASCII": NamedType("str"),
    "PyUnicode_DecodeCharmap": NamedType("str"),
    "PyUnicode_DecodeLocaleAndSize": NamedType("str"),
    "PyUnicode_DecodeLocale": NamedType("str"),
    "PyUnicode_DecodeFSDefaultAndSize": NamedType("str"),
    "PyUnicode_DecodeFSDefault": NamedType("str"),
    "PyUnicode_Translate": NamedType("str"),
    "PyUnicode_DecodeMBCS": NamedType("str"),
    "PyUnicode_DecodeMBCSStateful": NamedType("str"),
    "PyUnicode_DecodeCodePageStateful": NamedType("str"),
    "PyUnicode_Substring": NamedType("str"),
    "PyUnicode_Concat": NamedType("str"),
    "PyUnicode_Join": NamedType("str"),
    "PyUnicode_Replace": NamedType("str"),
    "PyUnicode_Format": NamedType("str"),
    "PyUnicode_InternFromString": NamedType("str"),

    # bytes
    "PyBytes_FromString": NamedType("bytes"),
    "PyBytes_FromStringAndSize": NamedType("bytes"),
    "PyBytes_FromFormat": NamedType("bytes"),
    "PyBytes_FromFormatV": NamedType("bytes"),
    "PyBytes_FromObject": NamedType("bytes"),
    "PyUnicode_AsEncodedString": NamedType("bytes"),
    "PyUnicode_AsUTF8String": NamedType("bytes"),
    "PyUnicode_AsUTF32String": NamedType("bytes"),
    "PyUnicode_AsUTF16String": NamedType("bytes"),
    "PyUnicode_AsUnicodeEscapeString": NamedType("bytes"),
    "PyUnicode_AsRawUnicodeEscapeString": NamedType("bytes"),
    "PyUnicode_AsLatin1String": NamedType("bytes"),
    "PyUnicode_AsASCIIString": NamedType("bytes"),
    "PyUnicode_AsCharmapString": NamedType("bytes"),
    "PyUnicode_EncodeLocale": NamedType("bytes"),
    "PyUnicode_EncodeFSDefault": NamedType("bytes"),
    "PyUnicode_AsMBCSString": NamedType("bytes"),
    "PyUnicode_EncodeCodePage": NamedType("bytes"),

    # bytearray
    "PyByteArray_FromObject": NamedType("bytearray"),
    "PyByteArray_FromStringAndSize": NamedType("bytearray"),
    "PyByteArray_Concat": NamedType("bytearray"),

    # slice
    "PySlice_New": NamedType("slice"),

    # memoryview
    "PyMemoryView_FromObject": NamedType("memoryview"),
    "PyMemoryView_FromMemory": NamedType("memoryview"),
    "PyMemoryView_FromBuffer": NamedType("memoryview"),
    "PyMemoryView_GetContiguous": NamedType("memoryview"),

    # tuple
    "PyTuple_New": NamedType("tuple"),
    "PyTuple_Pack": NamedType("tuple"),
    "PyTuple_GetSlice": NamedType("tuple"),
    "PyList_AsTuple": NamedType("tuple"),
    "PyUnicode_Partition": NamedType("tuple"),
    "PyUnicode_RPartition": NamedType("tuple"),

    # list
    "PyList_New": NamedType("list"),
    "PyList_GetSlice": NamedType("list"),
    "PyUnicode_Split": NamedType("list"),
    "PyUnicode_RSplit": NamedType("list"),
    "PyUnicode_Splitlines": NamedType("list"),
    "PyDict_Items": NamedType("list"),
    "PyDict_Keys": NamedType("list"),
    "PyDict_Values": NamedType("list"),

    # dict
    "PyDict_New": NamedType("dict"),
    "PyDict_Copy": NamedType("dict"),

    # set
    "PySet_New": NamedType("set"),

    # frozenset
    "PyFrozenSet_New": NamedType("frozenset"),

    # numpy
    "PyArray_Return": UnionType((NamedType("numpy.ndarray"), NamedType("numpy.generic")))
}

BUILD_CONVERTER_NAME_TO_TYPE: dict[str, str] = {

}

OBJECT_NAME_TO_TYPE: dict[str, str] = {
    "_Py_NoneStruct": "None",
    "_Py_TrueStruct": "bool",
    "_Py_FalseStruct": "bool",
}
