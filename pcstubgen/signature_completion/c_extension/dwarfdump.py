from __future__ import annotations

from bisect import bisect_left, bisect_right, insort
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from elftools.common.utils import struct_parse
from elftools.dwarf.compileunit import CompileUnit
from elftools.dwarf.descriptions import describe_form_class
from elftools.dwarf.die import DIE
from elftools.dwarf.dwarfinfo import DWARFInfo
from elftools.dwarf.ranges import BaseAddressEntry, RangeEntry
from elftools.elf.elffile import ELFFile


@dataclass(frozen=True)
class LookupResult:
    compilation_unit_path: Path
    function_name: str
    linkage_name: str | None = None


@dataclass(frozen=True)
class _CompileUnitRangeMapEntry:
    start: int
    end: int
    cu_offset: int


@dataclass(frozen=True)
class _RangeEndpoint:
    address: int
    cu_offset: int
    is_range_start: bool


def lookup(binary_path: Path, relative_address: int) -> LookupResult:
    """按共享库内函数入口相对地址查询 DWARF 编译单元路径与函数身份。"""
    with binary_path.open("rb") as binary_file:
        elf = ELFFile(binary_file)
        if not elf.has_dwarf_info():
            raise RuntimeError(f"共享库缺少DWARF调试信息: {binary_path}")

        dwarf = elf.get_dwarf_info()
        compilation_units = tuple(dwarf.iter_CUs())
        if not compilation_units:
            raise RuntimeError(f"共享库缺少DWARF调试信息: {binary_path}")

        compilation_unit = _find_compilation_unit(
            dwarf=dwarf,
            compilation_units=compilation_units,
            relative_address=relative_address,
        )
        if compilation_unit is None:
            raise RuntimeError(f"DWARF 中未找到地址所属编译单元: 0x{relative_address:x}")

        compilation_unit_path = _resolve_compilation_unit_path(compilation_unit)
        if compilation_unit_path is None:
            raise RuntimeError(f"DWARF 编译单元缺少源码路径: 0x{relative_address:x}")

        matched_die = _find_function_die(compilation_unit, relative_address)
        if matched_die is None:
            raise RuntimeError(f"DWARF 中未找到入口地址对应的函数: 0x{relative_address:x}")

        function_name, linkage_name = _resolve_subprogram_identity(matched_die)
        if function_name is None:
            raise RuntimeError(f"DWARF 函数缺少名称: 0x{relative_address:x}")

        return LookupResult(
            compilation_unit_path=compilation_unit_path,
            function_name=function_name,
            linkage_name=linkage_name,
        )


def _find_compilation_unit(
    *,
    dwarf: DWARFInfo,
    compilation_units: tuple[CompileUnit, ...],
    relative_address: int,
) -> CompileUnit | None:
    """按 LLVM 生成式 aranges 语义将地址映射到编译单元。"""
    compilation_units_by_offset = {
        compilation_unit.cu_offset: compilation_unit for compilation_unit in compilation_units
    }
    range_map = _build_compilation_unit_range_map(
        dwarf=dwarf,
        compilation_units=compilation_units,
        compilation_units_by_offset=compilation_units_by_offset,
    )
    cu_offset = _find_compile_unit_offset(range_map, relative_address)
    if cu_offset is None:
        return None
    return compilation_units_by_offset.get(cu_offset)


def _build_compilation_unit_range_map(
    *,
    dwarf: DWARFInfo,
    compilation_units: tuple[CompileUnit, ...],
    compilation_units_by_offset: dict[int, CompileUnit] | None = None,
    explicit_ranges: Iterable[_CompileUnitRangeMapEntry] | None = None,
    explicitly_covered_cu_offsets: set[int] | None = None,
) -> list[_CompileUnitRangeMapEntry]:
    """构建与 LLVM `getDebugAranges()` 对齐的 CU 地址区间索引。"""
    if compilation_units_by_offset is None:
        compilation_units_by_offset = {
            compilation_unit.cu_offset: compilation_unit for compilation_unit in compilation_units
        }
    assert compilation_units_by_offset is not None

    known_cu_offsets = set(compilation_units_by_offset)
    if explicit_ranges is None or explicitly_covered_cu_offsets is None:
        explicit_range_entries, covered_cu_offsets = _read_explicit_compilation_unit_ranges(
            dwarf=dwarf,
            known_cu_offsets=known_cu_offsets,
        )
    else:
        explicit_range_entries = [
            entry for entry in explicit_ranges if entry.cu_offset in known_cu_offsets
        ]
        covered_cu_offsets = {
            cu_offset for cu_offset in explicitly_covered_cu_offsets if cu_offset in known_cu_offsets
        }

    all_ranges = list(explicit_range_entries)
    for compilation_unit in compilation_units:
        if compilation_unit.cu_offset in covered_cu_offsets:
            continue

        top_die = compilation_unit.get_top_DIE()
        for range_start, range_end in _iter_die_ranges(top_die):
            all_ranges.append(
                _CompileUnitRangeMapEntry(
                    start=range_start,
                    end=range_end,
                    cu_offset=compilation_unit.cu_offset,
                )
            )

    return _construct_compilation_unit_range_map(all_ranges)


def _read_explicit_compilation_unit_ranges(
    *,
    dwarf: DWARFInfo,
    known_cu_offsets: set[int],
) -> tuple[list[_CompileUnitRangeMapEntry], set[int]]:
    """读取显式 `.debug_aranges`，并记录 LLVM `ParsedCUOffsets` 等价集合。"""
    explicitly_covered_cu_offsets = _read_debug_aranges_cu_offsets(dwarf) & known_cu_offsets
    aranges = dwarf.get_aranges()
    if aranges is None:
        return [], explicitly_covered_cu_offsets

    explicit_ranges: list[_CompileUnitRangeMapEntry] = []
    for entry in aranges.entries:
        cu_offset = int(entry.info_offset)
        if cu_offset not in known_cu_offsets:
            continue
        range_start = int(entry.begin_addr)
        range_end = range_start + int(entry.length)
        if range_start >= range_end:
            continue
        explicit_ranges.append(
            _CompileUnitRangeMapEntry(
                start=range_start,
                end=range_end,
                cu_offset=cu_offset,
            )
        )

    return explicit_ranges, explicitly_covered_cu_offsets


def _read_debug_aranges_cu_offsets(dwarf: DWARFInfo) -> set[int]:
    """读取 `.debug_aranges` 的 set 级 CU 覆盖信息。"""
    debug_aranges_sec = dwarf.debug_aranges_sec
    if debug_aranges_sec is None:
        return set()

    if dwarf.get_aranges() is None:
        return set()

    stream = debug_aranges_sec.stream
    size = int(debug_aranges_sec.size)
    section_offset = 0
    covered_cu_offsets: set[int] = set()

    while section_offset < size:
        header = struct_parse(dwarf.structs.Dwarf_aranges_header, stream, section_offset)
        covered_cu_offsets.add(int(header["debug_info_offset"]))

        if int(header["segment_size"]) != 0:
            raise NotImplementedError("Segmentation not implemented")

        section_offset += int(header.unit_length) + dwarf.structs.initial_length_field_size()

    return covered_cu_offsets


def _construct_compilation_unit_range_map(
    range_entries: Iterable[_CompileUnitRangeMapEntry],
) -> list[_CompileUnitRangeMapEntry]:
    """按 LLVM `appendRange()/construct()` 语义归并 CU 地址区间。"""
    endpoints: list[_RangeEndpoint] = []
    for entry in range_entries:
        if entry.start >= entry.end:
            continue
        endpoints.append(
            _RangeEndpoint(address=entry.start, cu_offset=entry.cu_offset, is_range_start=True)
        )
        endpoints.append(
            _RangeEndpoint(address=entry.end, cu_offset=entry.cu_offset, is_range_start=False)
        )

    endpoints.sort(key=lambda range_endpoint: range_endpoint.address)
    if not endpoints:
        return []

    range_map: list[_CompileUnitRangeMapEntry] = []
    active_cu_offsets: list[int] = []
    previous_address: int | None = None

    for endpoint in endpoints:
        if previous_address is not None and previous_address < endpoint.address and active_cu_offsets:
            cu_offset = active_cu_offsets[0]
            if (
                range_map
                and range_map[-1].end == previous_address
                and range_map[-1].cu_offset == cu_offset
            ):
                range_map[-1] = _CompileUnitRangeMapEntry(
                    start=range_map[-1].start,
                    end=endpoint.address,
                    cu_offset=cu_offset,
                )
            else:
                range_map.append(
                    _CompileUnitRangeMapEntry(
                        start=previous_address,
                        end=endpoint.address,
                        cu_offset=cu_offset,
                    )
                )

        if endpoint.is_range_start:
            insort(active_cu_offsets, endpoint.cu_offset)
        else:
            cu_index = bisect_left(active_cu_offsets, endpoint.cu_offset)
            if cu_index >= len(active_cu_offsets) or active_cu_offsets[cu_index] != endpoint.cu_offset:
                raise RuntimeError(
                    f"DWARF .debug_aranges 区间端点不匹配: cu_offset=0x{endpoint.cu_offset:x}"
                )
            active_cu_offsets.pop(cu_index)

        previous_address = endpoint.address

    return range_map


def _find_compile_unit_offset(
    range_map: list[_CompileUnitRangeMapEntry],
    relative_address: int,
) -> int | None:
    """在 CU 地址区间索引中查找覆盖目标地址的编译单元偏移。"""
    if not range_map:
        return None

    starts = [entry.start for entry in range_map]
    matched_index = bisect_right(starts, relative_address) - 1
    if matched_index < 0:
        return None

    matched = range_map[matched_index]
    if relative_address >= matched.end:
        return None
    return matched.cu_offset


def _resolve_compilation_unit_path(compilation_unit: CompileUnit) -> Path | None:
    """从编译单元 DIE 解析源码绝对路径。"""
    top_die = compilation_unit.get_top_DIE()
    name = _read_string_attr(top_die, "DW_AT_name")
    if name is None:
        return None

    compilation_unit_path = Path(name)
    if not compilation_unit_path.is_absolute():
        comp_dir = _read_string_attr(top_die, "DW_AT_comp_dir")
        if comp_dir is not None:
            compilation_unit_path = Path(comp_dir) / compilation_unit_path
    return compilation_unit_path.resolve()


def _find_function_die(
    compilation_unit: CompileUnit,
    relative_address: int,
) -> DIE | None:
    """在单个编译单元内按入口地址查找函数定义 DIE。"""
    for die in _iter_dies_preorder(compilation_unit.get_top_DIE()):
        if die.tag != "DW_TAG_subprogram":
            continue
        if _get_die_entry_pc(die) == relative_address:
            return die
    return None


def _iter_dies_preorder(root: DIE) -> Iterable[DIE]:
    """前序遍历 DIE 子树。"""
    yield root
    for child in root.iter_children():
        yield from _iter_dies_preorder(child)


def _get_die_entry_pc(die: DIE) -> int | None:
    """读取函数入口地址；仅接受显式 `DW_AT_low_pc`。"""
    low_pc_attr = die.attributes.get("DW_AT_low_pc")
    if low_pc_attr is None:
        return None
    return int(low_pc_attr.value)


def _resolve_subprogram_identity(subprogram: DIE) -> tuple[str | None, str | None]:
    """沿 specification / abstract_origin 链补全函数名与 linkage name。"""
    function_name: str | None = None
    linkage_name: str | None = None
    visited_offsets: set[int] = set()
    current: DIE | None = subprogram

    while current is not None and current.offset not in visited_offsets:
        visited_offsets.add(current.offset)
        if function_name is None:
            function_name = _read_string_attr(current, "DW_AT_name")
        if linkage_name is None:
            linkage_name = (
                _read_string_attr(current, "DW_AT_linkage_name")
                or _read_string_attr(current, "DW_AT_MIPS_linkage_name")
            )
        current = _follow_specification_chain(current)

    return function_name, linkage_name


def _follow_specification_chain(die: DIE) -> DIE | None:
    """返回 specification / abstract_origin 指向的下一个 DIE。"""
    for attr_name in ("DW_AT_specification", "DW_AT_abstract_origin"):
        if attr_name in die.attributes:
            return die.get_DIE_from_attribute(attr_name)
    return None


def _iter_die_ranges(die: DIE) -> Iterable[tuple[int, int]]:
    """迭代 DIE 覆盖的所有半开地址区间。"""
    low_pc_attr = die.attributes.get("DW_AT_low_pc")
    high_pc_attr = die.attributes.get("DW_AT_high_pc")
    if low_pc_attr is not None and high_pc_attr is not None:
        low_pc = int(low_pc_attr.value)
        high_pc = _resolve_high_pc(low_pc, high_pc_attr)
        yield low_pc, high_pc
        return

    ranges_attr = die.attributes.get("DW_AT_ranges")
    if ranges_attr is None:
        return

    base_address = int(low_pc_attr.value) if low_pc_attr is not None else 0
    range_lists = die.dwarfinfo.range_lists()
    for entry in range_lists.get_range_list_at_offset(ranges_attr.value, cu=die.cu):
        if isinstance(entry, BaseAddressEntry):
            base_address = int(entry.base_address)
            continue
        if not isinstance(entry, RangeEntry):
            continue

        range_start = int(entry.begin_offset)
        range_end = int(entry.end_offset)
        if not entry.is_absolute:
            range_start += base_address
            range_end += base_address
        if range_start < range_end:
            yield range_start, range_end


def _resolve_high_pc(low_pc: int, high_pc_attr: object) -> int:
    """解析 DWARF high_pc 的 address / offset 两种编码。"""
    high_pc_form = getattr(high_pc_attr, "form", None)
    high_pc_value = int(getattr(high_pc_attr, "value"))
    if high_pc_form is not None and describe_form_class(high_pc_form) == "address":
        return high_pc_value
    return low_pc + high_pc_value


def _read_string_attr(die: DIE, attr_name: str) -> str | None:
    """读取并规范化字符串属性值。"""
    attr = die.attributes.get(attr_name)
    if attr is None:
        return None
    value = attr.value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    return str(value)
