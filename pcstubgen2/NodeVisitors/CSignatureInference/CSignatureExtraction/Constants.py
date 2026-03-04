from __future__ import annotations

# 说明：
# - 本文件集中维护 C 签名提取所依赖的映射与过滤规则。
# - 常量尽量按“用途分组”维护，方便和解析逻辑（ExtractionEngine）对照更新。

# PyMethodDef.flags 中常见数字字面量到 METH_* 的映射。
METH_TYPE_LITERAL_MAP: dict[str, str] = {
    "1": "METH_VARARGS",
    "2": "METH_KEYWORDS",
    "4": "METH_NOARGS",
    "8": "METH_O",
    "16": "METH_CLASS",
    "32": "METH_STATIC",
    "64": "METH_COEXIST",
    "128": "METH_FASTCALL",
    "512": "METH_METHOD",
}

# PyArg_* 格式串 marker 到 Python 类型的近似映射。
# 注意：带序号的 key（如 y#1 / y#2）表示“同一格式符展开出的第 1/2 个 C 参数”。
FORMAT_TYPE_MAP: dict[str, str] = {
    "self": "object",
    "cls": "type",
    "y": "bytes",
    "*": "object",
    "$": "object",
    "i": "int",
    "I": "int",
    "e": "str",
    "t": "str",
    "s": "str",
    "z": "str",
    "u": "str",
    "U": "str",
    "d": "float",
    "D": "complex",
    "f": "float",
    "b": "int",
    "h": "int",
    "l": "int",
    "B": "int",
    "H": "int",
    "L": "int",
    "c": "int",
    "C": "str",
    "k": "int",
    "K": "int",
    "n": "int",
    "N": "object",
    "O": "object",
    "S": "object",
    "p": "bool",
    "F_INT_PYFMT": "int",
    # y# -> (const char* buffer, Py_ssize_t length)
    "y#1": "bytes",
    "y#2": "int",
    "y*": "bytes",
    # z# / s# 同样会拆成“内容 + 长度”两段参数。
    "z#1": "str",
    "z#2": "int",
    "s#1": "str",
    "s#2": "int",
    # O! -> (PyTypeObject*, PyObject*)
    "O!1": "object",
    "O!2": "object",
    "O&": "object",
}

# 返回值推断：常见 `Py_RETURN_*` 宏到 Python 类型。
RETURN_MACRO_TYPE_MAP: dict[str, str] = {
    "Py_RETURN_NONE": "None",
    "Py_RETURN_TRUE": "bool",
    "Py_RETURN_FALSE": "bool",
}

# 返回值推断：在 return 语句中直接出现的常见标识到 Python 类型。
RETURN_TOKEN_TYPE_MAP: dict[str, str] = {
    "Py_None": "None",
    "Py_True": "bool",
    "Py_False": "bool",
}

# 返回值推断：常见工厂函数前缀到 Python 类型。
RETURN_CALL_PREFIX_TYPE_MAP: tuple[tuple[str, str], ...] = (
    ("PyBool_", "bool"),
    ("PyLong_", "int"),
    ("PyInt_", "int"),
    ("PyFloat_", "float"),
    ("PyUnicode_", "str"),
    ("PyString_", "str"),
    ("PyBytes_", "bytes"),
    ("PyByteArray_", "bytearray"),
    ("PyTuple_", "tuple"),
    ("PyList_", "list"),
    ("PyDict_", "dict"),
    ("PySet_", "set"),
    ("PyFrozenSet_", "frozenset"),
    ("PyComplex_", "complex"),
    ("PyMemoryView_", "memoryview"),
)

# `Py_BuildValue` 单值格式符到 Python 类型的近似映射。
PY_BUILDVALUE_SINGLE_MARKER_TYPE_MAP: dict[str, str] = {
    "i": "int",
    "I": "int",
    "h": "int",
    "H": "int",
    "l": "int",
    "L": "int",
    "k": "int",
    "K": "int",
    "n": "int",
    "b": "int",
    "B": "int",
    "c": "int",
    "p": "bool",
    "f": "float",
    "d": "float",
    "D": "complex",
    "s": "str",
    "s#": "str",
    "z": "str",
    "z#": "str",
    "u": "str",
    "U": "str",
    "y": "bytes",
    "y#": "bytes",
    "y*": "bytes",
    "O": "object",
    "N": "object",
    "S": "object",
    "O!": "object",
    "O&": "object",
}

# 调用 token 提取阶段需要忽略的噪声标识（转换器、宏、数字字面量等）。
UNRELATED_TOKENS: set[str] = {
    "NI_ObjectToInputArray",
    "NI_ObjectToOutputArray",
    "PyArray_IntpConverter",
    "NI_ObjectToOptionalInputArray",
    "PyArray_Type",
    "NI_ObjectToOptionalOutputArray",
    "NI_ObjectToInputOutputArray",
    "NPY_BEGIN_ALLOW_THREADS",
    "PyArray_DATA",
    "PyArray_DIMS",
    "NPY_END_ALLOW_THREADS",
    "DEFINE_WRAP_CDIST",
    "CmsProfile_Type",
    "PyArray_AxisConverter",
    "PyArray_DescrConverter2",
    "PyArray_OutputConverter",
    "PyArrayDescr_TypeFull",
    "PyArray_DescrConverter",
    "PyArray_CopyConverter",
    "PyArray_ByteorderConverter",
    "PyArray_BoolConverter",
    "PyArray_ClipmodeConverter",
    "PyArray_OrderConverter",
    "ConvertDeviceName",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
}

# 在方法表 token 中查找 C 函数名时需要跳过的 cast/空指针标识。
POINTER_CAST_IDENTIFIER_SKIP: set[str] = {
    "PyCFunction",
    "PyCFunctionWithKeywords",
    "PyCMethod",
    "PyObject",
    "nullptr",
    "NULL",
}

# 参与扫描的 C/C++ 源码后缀。
C_SOURCE_SUFFIXES: tuple[str, ...] = (
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
)

