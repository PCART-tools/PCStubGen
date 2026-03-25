from __future__ import annotations

# 说明：
# - 本文件集中维护 C 签名提取所依赖的映射与过滤规则。
# - 常量尽量按“用途分组”维护，方便和解析逻辑（CSignatureExtractor）对照更新。

# PyMethodDef.ml_flags 常量值，和 CPython 侧 bitmask 语义保持一致。
METH_VARARGS = 1
METH_KEYWORDS = 2
METH_NOARGS = 4
METH_O = 8
METH_CLASS = 16
METH_STATIC = 32
METH_COEXIST = 64
METH_FASTCALL = 128
METH_METHOD = 512

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

# 参与扫描的 C 源码后缀。
C_SOURCE_SUFFIXES: set[str] = {
    ".c",
}

# 参与扫描并按 C++ 标准解析的源码后缀。
CPP_SOURCE_SUFFIXES: set[str] = {
    ".cc",
    ".cpp",
    ".cxx",
    ".c++",
    ".cp"
}

# 提取阶段需要扫描的全部原生源码后缀。
NATIVE_SOURCE_SUFFIXES: set[str] = C_SOURCE_SUFFIXES | CPP_SOURCE_SUFFIXES

HEADER_SOURCE_SUFFIXES = {
    ".h",
    ".hpp",
    ".hh",
    ".hxx",
    ".h++",
}
