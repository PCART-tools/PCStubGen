from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace

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
    readelf = _require_program("readelf")
    completed = subprocess.run(
        [readelf, "-Ws", "--wide", str(binary_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in completed.stdout.splitlines():
        match = re.match(
            r"\s*\d+:\s*([0-9a-fA-F]+)\s+\d+\s+\w+\s+\w+\s+\w+\s+\w+\s+(.+)",
            line,
        )
        if match is None:
            continue
        if match.group(2).strip() == symbol_name:
            return int(match.group(1), 16)
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


def _run_llvm_dwarfdump_lookup(
    binary_path: Path,
    relative_address: int,
) -> dwarfdump.LookupResult | None:
    """运行 `llvm-dwarfdump --lookup` 并提取可比较的关键信息。"""
    llvm_dwarfdump = _require_program("llvm-dwarfdump")
    completed = subprocess.run(
        [llvm_dwarfdump, f"--lookup=0x{relative_address:x}", str(binary_path)],
        check=False,
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
        return None

    subprogram_body = subprogram_match.group("body")
    function_name_match = re.search(r"DW_AT_name\s*\(\"([^\"]+)\"\)", subprogram_body)
    linkage_name_match = re.search(
        r"DW_AT_(?:linkage_name|MIPS_linkage_name)\s*\(\"([^\"]+)\"\)",
        subprogram_body,
    )
    if function_name_match is None:
        return None

    return dwarfdump.LookupResult(
        compilation_unit_path=Path(cu_path_match.group(1)).resolve(),
        function_name=function_name_match.group(1),
        linkage_name=None if linkage_name_match is None else linkage_name_match.group(1),
    )


def test_lookup_wraps_raw_result_and_normalizes_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "dir" / ".." / "sample.c"
    monkeypatch.setattr(
        dwarfdump,
        "_dwarfdump",
        SimpleNamespace(
            lookup=lambda binary_path, relative_address: (
                str(raw_path),
                "foo_impl",
                "_Z8foo_implv",
            )
        ),
    )

    result = dwarfdump.lookup(tmp_path / "sample.so", 0x1234)

    assert result == dwarfdump.LookupResult(
        compilation_unit_path=(tmp_path / "sample.c").resolve(),
        function_name="foo_impl",
        linkage_name="_Z8foo_implv",
    )

def test_lookup_propagates_runtime_error_from_native_extension(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        dwarfdump,
        "_dwarfdump",
        SimpleNamespace(
            lookup=lambda binary_path, relative_address: (_ for _ in ()).throw(
                RuntimeError("DWARF 中未找到地址所属编译单元: 0x1234")
            )
        ),
    )

    with pytest.raises(RuntimeError, match="未找到地址所属编译单元"):
        dwarfdump.lookup(tmp_path / "sample.so", 0x1234)


@pytest.mark.integration
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


@pytest.mark.integration
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


@pytest.mark.integration
def test_lookup_matches_outer_subprogram_for_inner_address(
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

    result = dwarfdump.lookup(binary_path, relative_address)

    assert result.compilation_unit_path == (tmp_path / "sample.c").resolve()
    assert result.function_name == "foo_impl"


@pytest.mark.integration
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


@pytest.mark.integration
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


@pytest.mark.integration
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


@pytest.mark.integration
@pytest.mark.slow
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

    assert llvm_result is not None
    assert result == llvm_result


@pytest.mark.integration
@pytest.mark.slow
def test_lookup_matches_llvm_dwarfdump_for_rnglists(
    tmp_path: Path,
) -> None:
    compiler = _require_program("libclang")
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

    assert llvm_result is not None
    assert result == llvm_result


@pytest.mark.integration
@pytest.mark.slow
def test_lookup_raises_when_llvm_dwarfdump_finds_no_match(
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
    relative_address = _find_symbol_value(binary_path, "foo_impl") + 0x1000

    assert _run_llvm_dwarfdump_lookup(binary_path, relative_address) is None

    with pytest.raises(RuntimeError, match="未找到地址所属编译单元"):
        dwarfdump.lookup(binary_path, relative_address)
