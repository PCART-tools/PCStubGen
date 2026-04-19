from __future__ import annotations

import ctypes
from pathlib import Path


class _DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


_dladdr = ctypes.CDLL(None).dladdr
_dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
_dladdr.restype = ctypes.c_int


def get_binary_and_ra(address: int) -> tuple[Path, int]:
    """用 dladdr 将运行时地址拆解为共享库路径和库内相对地址。"""
    dl_info = _DlInfo()
    if _dladdr(ctypes.c_void_p(address), ctypes.byref(dl_info)) != 1:
        raise RuntimeError(f"无法定位函数地址所属共享库: 0x{address:x}")
    if dl_info.dli_fname is None or dl_info.dli_fbase is None:
        raise RuntimeError(f"共享库位置信息不完整: 0x{address:x}")

    binary_path = Path(dl_info.dli_fname.decode("utf-8", errors="replace")).resolve()
    base_address = int(dl_info.dli_fbase)
    return binary_path, address - base_address
