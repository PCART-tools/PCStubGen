from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess

from elftools.dwarf.die import DIE
from elftools.elf.elffile import ELFFile
import pytest

from pcstubgen.signature_completion.c_extension import dwarfdump


def _require_program(name: str) -> str:
    program = shutil.which(name)
    if program is None:
        pytest.skip(f"缺少外部程序: {name}")
    return program


def _build_shared_library(
    tmp_path: Path,
    *,
    compiler: str,
    source_name: str,
    source_text: str,
    debug: bool = True,
    optimization: str = "-O0",
    use_relative_source_path: bool = False,
    extra_compile_args: tuple[str, ...] = (),
) -> Path:
    """编译单源共享库样例。"""
    return _build_shared_library_from_sources(
        tmp_path,
        compiler=compiler,
        sources={source_name: source_text},
        debug=debug,
        optimization=optimization,
        use_relative_source_path=use_relative_source_path,
        extra_compile_args=extra_compile_args,
    )


def _build_shared_library_from_sources(
    tmp_path: Path,
    *,
    compiler: str,
    sources: dict[str, str],
    debug: bool = True,
    optimization: str = "-O0",
    use_relative_source_path: bool = False,
    extra_compile_args: tuple[str, ...] = (),
) -> Path:
    """编译多源共享库样例。"""
    source_paths: list[Path] = []
    for source_name, source_text in sources.items():
        source_path = tmp_path / source_name
        source_path.write_text(source_text, encoding="utf-8", newline="\n")
        source_paths.append(source_path)

    binary_path = tmp_path / "sample.so"
    compile_args = [compiler, "-shared", "-fPIC", optimization]
    if debug:
        compile_args.append("-g")
    compile_args.extend(extra_compile_args)
    compile_args.extend([
        *[
            source_path.name if use_relative_source_path else str(source_path)
            for source_path in source_paths
        ],
        "-o",
        str(binary_path),
    ])

    subprocess.run(
        compile_args,
        cwd=tmp_path if use_relative_source_path else None,
        check=True,
        capture_output=True,
        text=True,
    )
    return binary_path


def _copy_without_debug_aranges(binary_path: Path) -> Path:
    """复制二进制并移除 `.debug_aranges`。"""
    objcopy = _require_program("objcopy")
    output_path = binary_path.with_name(f"{binary_path.stem}.no_aranges{binary_path.suffix}")
    subprocess.run(
        [
            objcopy,
            "--remove-section",
            ".debug_aranges",
            str(binary_path),
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return output_path


def _find_symbol_value(binary_path: Path, symbol_name: str) -> int:
    """读取 ELF 符号表中的相对地址。"""
    with binary_path.open("rb") as binary_file:
        elf = ELFFile(binary_file)
        for section_name in (".symtab", ".dynsym"):
            section = elf.get_section_by_name(section_name)
            if section is None:
                continue
            for symbol in section.iter_symbols():
                if symbol.name == symbol_name:
                    return int(symbol["st_value"])
    raise AssertionError(f"未找到符号: {symbol_name}")


def _build_discontinuous_range_library(
    tmp_path: Path,
    *,
    compiler: str,
    extra_compile_args: tuple[str, ...] = (),
) -> Path:
    """构造顶层 CU 使用不连续地址范围的共享库。"""
    return _build_shared_library(
        tmp_path,
        compiler=compiler,
        source_name="sample.c",
        source_text=(
            "__attribute__((section(\".text.hot.foo\"))) int hot1(void) { return 1; }\n"
            "__attribute__((section(\".text.unlikely.foo\"))) int cold1(void) { return 2; }\n"
            "int wrapper(int x) { return x ? hot1() : cold1(); }\n"
        ),
        extra_compile_args=extra_compile_args,
    )


def _read_top_die_attr_form(binary_path: Path, attr_name: str) -> str:
    """读取首个编译单元顶层 DIE 的属性编码形式。"""
    with binary_path.open("rb") as binary_file:
        dwarf = ELFFile(binary_file).get_dwarf_info()
        top_die = next(dwarf.iter_CUs()).get_top_DIE()
        attr = top_die.attributes.get(attr_name)
        if attr is None:
            raise AssertionError(f"顶层 DIE 缺少属性: {attr_name}")
        return str(attr.form)


def _run_llvm_dwarfdump_lookup(binary_path: Path, relative_address: int) -> dwarfdump.LookupResult:
    """运行 `llvm-dwarfdump --lookup` 并提取可比较的关键信息。"""
    llvm_dwarfdump = _require_program("llvm-dwarfdump")
    completed = subprocess.run(
        [llvm_dwarfdump, f"--lookup=0x{relative_address:x}", str(binary_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    output = completed.stdout

    cu_path_match = re.search(
        r"DW_TAG_compile_unit.*?DW_AT_name\s*\(\"([^\"]+)\"\)",
        output,
        re.DOTALL,
    )
    subprogram_match = re.search(
        r"DW_TAG_subprogram(?P<body>.*?)(?:\n0x[0-9a-f]+:|\nLine info:|\Z)",
        output,
        re.DOTALL,
    )
    if cu_path_match is None or subprogram_match is None:
        raise AssertionError(f"无法解析 llvm-dwarfdump 输出:\n{output}")

    subprogram_body = subprogram_match.group("body")
    function_name_match = re.search(r"DW_AT_name\s*\(\"([^\"]+)\"\)", subprogram_body)
    linkage_name_match = re.search(
        r"DW_AT_(?:linkage_name|MIPS_linkage_name)\s*\(\"([^\"]+)\"\)",
        subprogram_body,
    )
    if function_name_match is None:
        raise AssertionError(f"llvm-dwarfdump 输出缺少函数名:\n{output}")

    return dwarfdump.LookupResult(
        compilation_unit_path=Path(cu_path_match.group(1)).resolve(),
        function_name=function_name_match.group(1),
        linkage_name=None if linkage_name_match is None else linkage_name_match.group(1),
    )


def test_lookup_reads_compilation_unit_path_from_relative_dwarf_name(
    tmp_path: Path,
) -> None:
    compiler = _require_program("cc")
    binary_path = _build_shared_library(
        tmp_path,
        compiler=compiler,
        source_name="sample.c",
        source_text=(
            "static int helper(int value) { return value + 1; }\n"
            "int foo_impl(int value) { return helper(value); }\n"
        ),
        use_relative_source_path=True,
    )
    relative_address = _find_symbol_value(binary_path, "foo_impl")

    result = dwarfdump.lookup(binary_path, relative_address)

    assert result.compilation_unit_path == (tmp_path / "sample.c").resolve()
    assert result.function_name == "foo_impl"
    assert result.linkage_name is None


def test_lookup_follows_specification_for_cpp_overload(
    tmp_path: Path,
) -> None:
    compiler = _require_program("g++")
    binary_path = _build_shared_library(
        tmp_path,
        compiler=compiler,
        source_name="sample.cpp",
        source_text=(
            "namespace ns {\n"
            "int foo(int value) { return value; }\n"
            "double foo(double value) { return value; }\n"
            "}\n"
        ),
        use_relative_source_path=True,
    )
    linkage_name = "_ZN2ns3fooEi"
    relative_address = _find_symbol_value(binary_path, linkage_name)

    result = dwarfdump.lookup(binary_path, relative_address)

    assert result.compilation_unit_path == (tmp_path / "sample.cpp").resolve()
    assert result.function_name == "foo"
    assert result.linkage_name == linkage_name


def test_find_function_die_matches_subprogram_entry_address_with_inline_call(
    tmp_path: Path,
) -> None:
    compiler = _require_program("cc")
    binary_path = _build_shared_library(
        tmp_path,
        compiler=compiler,
        source_name="sample.c",
        source_text=(
            "static inline __attribute__((always_inline)) int add1(int x) { return x + 1; }\n"
            "int foo_impl(int x) { return add1(x); }\n"
        ),
        optimization="-O2",
    )
    relative_address = _find_symbol_value(binary_path, "foo_impl")

    with binary_path.open("rb") as binary_file:
        compilation_unit = next(ELFFile(binary_file).get_dwarf_info().iter_CUs())

        matched_die = dwarfdump._find_function_die(compilation_unit, relative_address)

    assert matched_die is not None
    assert matched_die.tag == "DW_TAG_subprogram"
    assert dwarfdump._resolve_subprogram_identity(matched_die) == ("foo_impl", None)


def test_lookup_matches_outer_subprogram_when_function_contains_inline_call(
    tmp_path: Path,
) -> None:
    compiler = _require_program("cc")
    binary_path = _build_shared_library(
        tmp_path,
        compiler=compiler,
        source_name="sample.c",
        source_text=(
            "static inline __attribute__((always_inline)) int add1(int x) { return x + 1; }\n"
            "int foo_impl(int x) { return add1(x); }\n"
        ),
        optimization="-O2",
    )
    relative_address = _find_symbol_value(binary_path, "foo_impl")

    result = dwarfdump.lookup(binary_path, relative_address)

    assert result.compilation_unit_path == (tmp_path / "sample.c").resolve()
    assert result.function_name == "foo_impl"


def test_lookup_rejects_binary_without_dwarf(
    tmp_path: Path,
) -> None:
    compiler = _require_program("cc")
    binary_path = _build_shared_library(
        tmp_path,
        compiler=compiler,
        source_name="sample.c",
        source_text="int foo_impl(void) { return 1; }\n",
        debug=False,
    )
    relative_address = _find_symbol_value(binary_path, "foo_impl")

    with pytest.raises(RuntimeError, match="缺少DWARF调试信息"):
        dwarfdump.lookup(binary_path, relative_address)


def test_lookup_uses_generated_aranges_when_debug_aranges_missing(
    tmp_path: Path,
) -> None:
    compiler = _require_program("cc")
    binary_path = _build_shared_library(
        tmp_path,
        compiler=compiler,
        source_name="sample.c",
        source_text="int foo_impl(void) { return 1; }\n",
    )
    no_aranges_binary_path = _copy_without_debug_aranges(binary_path)
    relative_address = _find_symbol_value(no_aranges_binary_path, "foo_impl")

    result = dwarfdump.lookup(no_aranges_binary_path, relative_address)

    assert result.compilation_unit_path == (tmp_path / "sample.c").resolve()
    assert result.function_name == "foo_impl"
    assert result.linkage_name is None


def test_lookup_uses_generated_aranges_for_dwarf4_ranges(
    tmp_path: Path,
) -> None:
    compiler = _require_program("cc")
    binary_path = _build_discontinuous_range_library(
        tmp_path,
        compiler=compiler,
        extra_compile_args=("-gdwarf-4",),
    )
    no_aranges_binary_path = _copy_without_debug_aranges(binary_path)
    relative_address = _find_symbol_value(no_aranges_binary_path, "wrapper")

    assert _read_top_die_attr_form(no_aranges_binary_path, "DW_AT_ranges") == "DW_FORM_sec_offset"

    result = dwarfdump.lookup(no_aranges_binary_path, relative_address)

    assert result.compilation_unit_path == (tmp_path / "sample.c").resolve()
    assert result.function_name == "wrapper"
    assert result.linkage_name is None


def test_lookup_uses_generated_aranges_for_dwarf5_rnglists(
    tmp_path: Path,
) -> None:
    compiler = _require_program("clang")
    binary_path = _build_discontinuous_range_library(
        tmp_path,
        compiler=compiler,
        extra_compile_args=("-gdwarf-5",),
    )
    no_aranges_binary_path = _copy_without_debug_aranges(binary_path)
    relative_address = _find_symbol_value(no_aranges_binary_path, "wrapper")

    assert _read_top_die_attr_form(no_aranges_binary_path, "DW_AT_ranges") == "DW_FORM_rnglistx"

    result = dwarfdump.lookup(no_aranges_binary_path, relative_address)

    assert result.compilation_unit_path == (tmp_path / "sample.c").resolve()
    assert result.function_name == "wrapper"
    assert result.linkage_name is None


def test_lookup_uses_generated_aranges_for_multi_cu_binary(
    tmp_path: Path,
) -> None:
    compiler = _require_program("cc")
    binary_path = _build_shared_library_from_sources(
        tmp_path,
        compiler=compiler,
        sources={
            "first.c": "int first_impl(void) { return 1; }\n",
            "second.c": "int second_impl(void) { return 2; }\n",
        },
        extra_compile_args=("-gdwarf-4",),
    )
    no_aranges_binary_path = _copy_without_debug_aranges(binary_path)

    first_result = dwarfdump.lookup(
        no_aranges_binary_path,
        _find_symbol_value(no_aranges_binary_path, "first_impl"),
    )
    second_result = dwarfdump.lookup(
        no_aranges_binary_path,
        _find_symbol_value(no_aranges_binary_path, "second_impl"),
    )

    assert first_result.compilation_unit_path == (tmp_path / "first.c").resolve()
    assert first_result.function_name == "first_impl"
    assert second_result.compilation_unit_path == (tmp_path / "second.c").resolve()
    assert second_result.function_name == "second_impl"


def test_build_compilation_unit_range_map_only_generates_for_uncovered_cus(
    tmp_path: Path,
) -> None:
    compiler = _require_program("cc")
    binary_path = _build_shared_library_from_sources(
        tmp_path,
        compiler=compiler,
        sources={
            "first.c": "int first_impl(void) { return 1; }\n",
            "second.c": "int second_impl(void) { return 2; }\n",
        },
        extra_compile_args=("-gdwarf-4",),
    )

    with binary_path.open("rb") as binary_file:
        dwarf = ELFFile(binary_file).get_dwarf_info()
        compilation_units = tuple(dwarf.iter_CUs())

        first_cu, second_cu = compilation_units
        second_ranges = [
            dwarfdump._CompileUnitRangeMapEntry(start=start, end=end, cu_offset=second_cu.cu_offset)
            for start, end in dwarfdump._iter_die_ranges(second_cu.get_top_DIE())
        ]

        first_symbol = _find_symbol_value(binary_path, "first_impl")
        range_map = dwarfdump._build_compilation_unit_range_map(
            dwarf=dwarf,
            compilation_units=compilation_units,
            explicit_ranges=[
                dwarfdump._CompileUnitRangeMapEntry(
                    start=first_symbol,
                    end=first_symbol + 1,
                    cu_offset=first_cu.cu_offset,
                )
            ],
            explicitly_covered_cu_offsets={first_cu.cu_offset},
        )

    first_entries = [entry for entry in range_map if entry.cu_offset == first_cu.cu_offset]
    second_entries = [entry for entry in range_map if entry.cu_offset == second_cu.cu_offset]

    assert first_entries == [
        dwarfdump._CompileUnitRangeMapEntry(
            start=first_symbol,
            end=first_symbol + 1,
            cu_offset=first_cu.cu_offset,
        )
    ]
    assert second_entries == second_ranges


def test_lookup_rejects_address_outside_debug_aranges(
    tmp_path: Path,
) -> None:
    compiler = _require_program("cc")
    binary_path = _build_shared_library(
        tmp_path,
        compiler=compiler,
        source_name="sample.c",
        source_text="int foo_impl(void) { return 1; }\n",
    )
    relative_address = _find_symbol_value(binary_path, "foo_impl") + 0x1000

    with pytest.raises(RuntimeError, match="未找到地址所属编译单元"):
        dwarfdump.lookup(binary_path, relative_address)


def test_lookup_rejects_non_entry_address_inside_function(
    tmp_path: Path,
) -> None:
    compiler = _require_program("cc")
    binary_path = _build_shared_library(
        tmp_path,
        compiler=compiler,
        source_name="sample.c",
        source_text="int foo_impl(int value) { return value + 1; }\n",
    )
    relative_address = _find_symbol_value(binary_path, "foo_impl") + 1

    with pytest.raises(RuntimeError, match="未找到入口地址对应的函数"):
        dwarfdump.lookup(binary_path, relative_address)


def test_lookup_rejects_binary_without_top_die_address_ranges(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    compiler = _require_program("cc")
    binary_path = _build_shared_library(
        tmp_path,
        compiler=compiler,
        source_name="sample.c",
        source_text="int foo_impl(void) { return 1; }\n",
    )
    no_aranges_binary_path = _copy_without_debug_aranges(binary_path)
    relative_address = _find_symbol_value(no_aranges_binary_path, "foo_impl")

    original_iter_die_ranges = dwarfdump._iter_die_ranges

    def fake_iter_die_ranges(die: DIE) -> object:
        if getattr(die, "tag", None) == "DW_TAG_compile_unit":
            return iter(())
        return original_iter_die_ranges(die)

    monkeypatch.setattr(dwarfdump, "_iter_die_ranges", fake_iter_die_ranges)

    with pytest.raises(RuntimeError, match="未找到地址所属编译单元"):
        dwarfdump.lookup(no_aranges_binary_path, relative_address)


def test_lookup_rejects_missing_function_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    compiler = _require_program("cc")
    binary_path = _build_shared_library(
        tmp_path,
        compiler=compiler,
        source_name="sample.c",
        source_text="int foo_impl(void) { return 1; }\n",
    )
    relative_address = _find_symbol_value(binary_path, "foo_impl")

    monkeypatch.setattr(dwarfdump, "_resolve_subprogram_identity", lambda die: (None, None))

    with pytest.raises(RuntimeError, match="函数缺少名称"):
        dwarfdump.lookup(binary_path, relative_address)


def test_lookup_matches_llvm_dwarfdump_for_low_pc_high_pc(
    tmp_path: Path,
) -> None:
    compiler = _require_program("cc")
    _require_program("llvm-dwarfdump")
    binary_path = _build_shared_library(
        tmp_path,
        compiler=compiler,
        source_name="sample.c",
        source_text="int foo_impl(void) { return 1; }\n",
    )
    no_aranges_binary_path = _copy_without_debug_aranges(binary_path)
    relative_address = _find_symbol_value(no_aranges_binary_path, "foo_impl")

    result = dwarfdump.lookup(no_aranges_binary_path, relative_address)
    llvm_result = _run_llvm_dwarfdump_lookup(no_aranges_binary_path, relative_address)

    assert result == llvm_result


def test_lookup_matches_llvm_dwarfdump_for_rnglists(
    tmp_path: Path,
) -> None:
    compiler = _require_program("clang")
    _require_program("llvm-dwarfdump")
    binary_path = _build_discontinuous_range_library(
        tmp_path,
        compiler=compiler,
        extra_compile_args=("-gdwarf-5",),
    )
    no_aranges_binary_path = _copy_without_debug_aranges(binary_path)
    relative_address = _find_symbol_value(no_aranges_binary_path, "wrapper")

    result = dwarfdump.lookup(no_aranges_binary_path, relative_address)
    llvm_result = _run_llvm_dwarfdump_lookup(no_aranges_binary_path, relative_address)

    assert result == llvm_result
