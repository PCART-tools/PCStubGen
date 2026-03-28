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
FUNCTION_NAME_TO_TYPE: dict[str, str] = {
    # bool
    "PyBool_FromLong": "bool",

    # int
    "PyLong_FromLong": "int",
    "PyLong_FromUnsignedLong": "int",
    "PyLong_FromSsize_t": "int",
    "PyLong_FromSize_t": "int",
    "PyLong_FromLongLong": "int",
    "PyLong_FromUnsignedLongLong": "int",
    "PyLong_FromDouble": "int",
    "PyLong_FromString": "int",
    "PyLong_FromUnicodeObject": "int",
    "PyLong_FromVoidPtr": "int",

    # float
    "PyFloat_FromString": "float",
    "PyFloat_FromDouble": "float",

    # complex
    "PyComplex_FromCComplex": "complex",
    "PyComplex_FromDoubles": "complex",

    # str
    "PyUnicode_New": "str",
    "PyUnicode_FromKindAndData": "str",
    "PyUnicode_FromString": "str",
    "PyUnicode_FromStringAndSize": "str",
    "PyUnicode_FromFormat": "str",
    "PyUnicode_FromFormatV": "str",
    "PyUnicode_FromObject": "str",
    "PyUnicode_FromEncodedObject": "str",
    "PyUnicode_FromWideChar": "str",
    "PyUnicode_Decode": "str",
    "PyUnicode_DecodeUTF8": "str",
    "PyUnicode_DecodeUTF8Stateful": "str",
    "PyUnicode_DecodeUTF32": "str",
    "PyUnicode_DecodeUTF32Stateful": "str",
    "PyUnicode_DecodeUTF16": "str",
    "PyUnicode_DecodeUTF16Stateful": "str",
    "PyUnicode_DecodeUTF7": "str",
    "PyUnicode_DecodeUTF7Stateful": "str",
    "PyUnicode_DecodeUnicodeEscape": "str",
    "PyUnicode_DecodeRawUnicodeEscape": "str",
    "PyUnicode_DecodeLatin1": "str",
    "PyUnicode_DecodeASCII": "str",
    "PyUnicode_DecodeCharmap": "str",
    "PyUnicode_DecodeLocaleAndSize": "str",
    "PyUnicode_DecodeLocale": "str",
    "PyUnicode_DecodeFSDefaultAndSize": "str",
    "PyUnicode_DecodeFSDefault": "str",
    "PyUnicode_Translate": "str",
    "PyUnicode_DecodeMBCS": "str",
    "PyUnicode_DecodeMBCSStateful": "str",
    "PyUnicode_DecodeCodePageStateful": "str",
    "PyUnicode_Substring": "str",
    "PyUnicode_Concat": "str",
    "PyUnicode_Join": "str",
    "PyUnicode_Replace": "str",
    "PyUnicode_Format": "str",
    "PyUnicode_InternFromString": "str",

    # bytes
    "PyBytes_FromString": "bytes",
    "PyBytes_FromStringAndSize": "bytes",
    "PyBytes_FromFormat": "bytes",
    "PyBytes_FromFormatV": "bytes",
    "PyBytes_FromObject": "bytes",
    "PyUnicode_AsEncodedString": "bytes",
    "PyUnicode_AsUTF8String": "bytes",
    "PyUnicode_AsUTF32String": "bytes",
    "PyUnicode_AsUTF16String": "bytes",
    "PyUnicode_AsUnicodeEscapeString": "bytes",
    "PyUnicode_AsRawUnicodeEscapeString": "bytes",
    "PyUnicode_AsLatin1String": "bytes",
    "PyUnicode_AsASCIIString": "bytes",
    "PyUnicode_AsCharmapString": "bytes",
    "PyUnicode_EncodeLocale": "bytes",
    "PyUnicode_EncodeFSDefault": "bytes",
    "PyUnicode_AsMBCSString": "bytes",
    "PyUnicode_EncodeCodePage": "bytes",

    # bytearray
    "PyByteArray_FromObject": "bytearray",
    "PyByteArray_FromStringAndSize": "bytearray",
    "PyByteArray_Concat": "bytearray",

    # slice
    "PySlice_New": "slice",

    # memoryview
    "PyMemoryView_FromObject": "memoryview",
    "PyMemoryView_FromMemory": "memoryview",
    "PyMemoryView_FromBuffer": "memoryview",
    "PyMemoryView_GetContiguous": "memoryview",

    # tuple
    "PyTuple_New": "tuple",
    "PyTuple_Pack": "tuple",
    "PyTuple_GetSlice": "tuple",
    "PyList_AsTuple": "tuple",
    "PyUnicode_Partition": "tuple",
    "PyUnicode_RPartition": "tuple",

    # list
    "PyList_New": "list",
    "PyList_GetSlice": "list",
    "PyUnicode_Split": "list",
    "PyUnicode_RSplit": "list",
    "PyUnicode_Splitlines": "list",
    "PyDict_Items": "list",
    "PyDict_Keys": "list",
    "PyDict_Values": "list",

    # dict
    "PyDict_New": "dict",
    "PyDict_Copy": "dict",

    # set
    "PySet_New": "set",

    # frozenset
    "PyFrozenSet_New": "frozenset",
}

BUILD_CONVERTER_NAME_TO_TYPE: dict[str, str] = {

}

OBJECT_NAME_TO_TYPE: dict[str, str] = {
    "Py_None": "None",
    "Py_True": "bool",
    "Py_False": "bool",
}

RETURN_MACRO_TO_TYPE: dict[str, str] = {
    "Py_RETURN_NONE": "None",
    "Py_RETURN_TRUE": "bool",
    "Py_RETURN_FALSE": "bool",
    "Py_RETURN_NAN": "float",
    "Py_RETURN_INF": "float",
}