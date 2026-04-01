from __future__ import annotations

C_SOURCE_SUFFIXES: set[str] = {
    ".c",
}

CPP_SOURCE_SUFFIXES: set[str] = {
    ".cc",
    ".cpp",
    ".cxx",
    ".c++",
    ".cp",
}

NATIVE_SOURCE_SUFFIXES: set[str] = C_SOURCE_SUFFIXES | CPP_SOURCE_SUFFIXES

HEADER_SOURCE_SUFFIXES: set[str] = {
    ".h",
    ".hpp",
    ".hh",
    ".hxx",
    ".h++",
}
