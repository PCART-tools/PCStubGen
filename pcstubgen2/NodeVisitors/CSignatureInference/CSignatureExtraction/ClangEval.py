import clang.cindex
from ctypes import c_void_p, c_int, c_longlong, c_ulonglong, c_uint


def _init_eval_api() -> None:
    lib = clang.cindex.conf.lib

    # CXEvalResult 本质上按 opaque pointer 处理
    lib.clang_Cursor_Evaluate.argtypes = [clang.cindex.Cursor]
    lib.clang_Cursor_Evaluate.restype = c_void_p

    lib.clang_EvalResult_getKind.argtypes = [c_void_p]
    lib.clang_EvalResult_getKind.restype = c_int

    lib.clang_EvalResult_getAsLongLong.argtypes = [c_void_p]
    lib.clang_EvalResult_getAsLongLong.restype = c_longlong

    lib.clang_EvalResult_getAsUnsigned.argtypes = [c_void_p]
    lib.clang_EvalResult_getAsUnsigned.restype = c_ulonglong

    lib.clang_EvalResult_isUnsignedInt.argtypes = [c_void_p]
    lib.clang_EvalResult_isUnsignedInt.restype = c_uint

    lib.clang_EvalResult_dispose.argtypes = [c_void_p]
    lib.clang_EvalResult_dispose.restype = None


_init_eval_api()


def eval_int(cursor: clang.cindex.Cursor) -> int | None:
    lib = clang.cindex.conf.lib
    ev = lib.clang_Cursor_Evaluate(cursor)
    if not ev:
        return None

    CXEval_Int = 1
    try:
        if lib.clang_EvalResult_getKind(ev) != CXEval_Int:
            return None

        if lib.clang_EvalResult_isUnsignedInt(ev):
            return int(lib.clang_EvalResult_getAsUnsigned(ev))
        else:
            return int(lib.clang_EvalResult_getAsLongLong(ev))
    finally:
        lib.clang_EvalResult_dispose(ev)