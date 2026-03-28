from __future__ import annotations

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
