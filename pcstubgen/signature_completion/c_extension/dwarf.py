from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

from elftools.elf.elffile import ELFFile


@dataclass(frozen=True)
class ResolvedAddressLocation:
    binary_path: Path
    relative_address: int
    symbol_name: str | None = None
    linkage_name: str | None = None
    source_path: Path | None = None
    source_line: int | None = None
    declaration_path: Path | None = None
    declaration_line: int | None = None
    compilation_directory: Path | None = None


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


def resolve_address_location(address: int) -> ResolvedAddressLocation:
    """将函数地址映射到 ELF / DWARF 中的共享库与源码位置。"""
    dl_info = _DlInfo()
    if _dladdr(ctypes.c_void_p(address), ctypes.byref(dl_info)) != 1:
        raise RuntimeError(f"无法定位函数地址所属共享库: 0x{address:x}")
    if dl_info.dli_fname is None or dl_info.dli_fbase is None:
        raise RuntimeError(f"共享库位置信息不完整: 0x{address:x}")

    binary_path = Path(dl_info.dli_fname.decode("utf-8", errors="replace")).resolve()
    base_address = int(dl_info.dli_fbase)
    relative_address = address - base_address

    with binary_path.open("rb") as binary_file:
        elf = ELFFile(binary_file)
        symbol_name = _find_symbol_name(elf, relative_address)
        if not elf.has_dwarf_info():
            return ResolvedAddressLocation(
                binary_path=binary_path,
                relative_address=relative_address,
                symbol_name=symbol_name,
            )
        dwarf = elf.get_dwarf_info()
        subprogram = _find_subprogram(dwarf, relative_address)
        line_location = _find_line_location(dwarf, relative_address)

    compilation_directory = None
    linkage_name = None
    declaration_path = None
    declaration_line = None
    if subprogram is not None:
        cu, die = subprogram
        top_die = cu.get_top_DIE()
        compilation_directory = _decode_path_attr(top_die.attributes.get("DW_AT_comp_dir"))
        linkage_name = _decode_bytes_attr(die.attributes.get("DW_AT_linkage_name"))
        declaration_line = _decode_int_attr(die.attributes.get("DW_AT_decl_line"))
        line_program = dwarf.line_program_for_CU(cu)
        declaration_path = _resolve_file_path(
            line_program=line_program,
            compilation_directory=compilation_directory,
            file_index=_decode_int_attr(die.attributes.get("DW_AT_decl_file")),
        )
        if symbol_name is None:
            symbol_name = _decode_bytes_attr(die.attributes.get("DW_AT_name"))

    source_path = None
    source_line = None
    if line_location is not None:
        line_program, state = line_location
        source_path = _resolve_file_path(
            line_program=line_program,
            compilation_directory=compilation_directory,
            file_index=int(state.file),
        )
        source_line = int(state.line)

    return ResolvedAddressLocation(
        binary_path=binary_path,
        relative_address=relative_address,
        symbol_name=symbol_name,
        linkage_name=linkage_name,
        source_path=source_path,
        source_line=source_line,
        declaration_path=declaration_path,
        declaration_line=declaration_line,
        compilation_directory=compilation_directory,
    )


def _find_symbol_name(elf: ELFFile, relative_address: int) -> str | None:
    for section_name in (".symtab", ".dynsym"):
        section = elf.get_section_by_name(section_name)
        if section is None:
            continue
        for symbol in section.iter_symbols():
            if symbol["st_info"]["type"] != "STT_FUNC":
                continue
            start = int(symbol["st_value"])
            size = int(symbol["st_size"])
            end = start + size
            if size == 0:
                if start == relative_address:
                    return symbol.name or None
                continue
            if start <= relative_address < end:
                return symbol.name or None
    return None


def _find_subprogram(dwarf_info: object, relative_address: int) -> tuple[object, object] | None:
    for cu in dwarf_info.iter_CUs():
        for die in cu.iter_DIEs():
            if die.tag != "DW_TAG_subprogram":
                continue
            low_pc_attr = die.attributes.get("DW_AT_low_pc")
            high_pc_attr = die.attributes.get("DW_AT_high_pc")
            if low_pc_attr is None or high_pc_attr is None:
                continue

            low_pc = int(low_pc_attr.value)
            if high_pc_attr.form == "DW_FORM_addr":
                high_pc = int(high_pc_attr.value)
            else:
                high_pc = low_pc + int(high_pc_attr.value)

            if low_pc <= relative_address < high_pc:
                return cu, die
    return None


def _find_line_location(dwarf_info: object, relative_address: int) -> tuple[object, object] | None:
    for cu in dwarf_info.iter_CUs():
        line_program = dwarf_info.line_program_for_CU(cu)
        if line_program is None:
            continue
        previous_state = None
        for entry in line_program.get_entries():
            state = entry.state
            if state is None:
                continue
            if previous_state is not None and previous_state.address <= relative_address < state.address:
                return line_program, previous_state
            previous_state = state
    return None


def _resolve_file_path(
    *,
    line_program: object | None,
    compilation_directory: Path | None,
    file_index: int | None,
) -> Path | None:
    if line_program is None or file_index is None or file_index <= 0:
        return None

    file_entries = line_program["file_entry"]
    if file_index > len(file_entries):
        return None
    file_entry = file_entries[file_index - 1]
    file_name = Path(file_entry.name.decode("utf-8", errors="replace"))
    if file_name.is_absolute():
        return file_name.resolve()

    directory = Path()
    dir_index = int(file_entry.dir_index)
    if dir_index > 0:
        directory = Path(
            line_program["include_directory"][dir_index - 1].decode("utf-8", errors="replace")
        )
    if not directory.is_absolute() and compilation_directory is not None:
        directory = (compilation_directory / directory).resolve()

    candidate = directory / file_name
    return candidate.resolve()


def _decode_path_attr(attribute: object | None) -> Path | None:
    if attribute is None:
        return None
    return Path(attribute.value.decode("utf-8", errors="replace")).resolve()


def _decode_bytes_attr(attribute: object | None) -> str | None:
    if attribute is None:
        return None
    value = attribute.value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _decode_int_attr(attribute: object | None) -> int | None:
    if attribute is None:
        return None
    return int(attribute.value)
