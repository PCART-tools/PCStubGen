import sys
import difflib
import shutil
from pathlib import Path


TARGET_MODULE = sys.argv[1] if len(sys.argv) > 1 else "numpy.random"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Add project root to path
sys.path.insert(0, str(PROJECT_ROOT))

# Add reference implementation to path
ref_path = PROJECT_ROOT / "reference" / "pybind11-stubgen"
sys.path.insert(0, str(ref_path))

# Ensure Chinese output renders correctly on Windows consoles.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Import new implementation (write_stubs API)
from pcstubgen2 import write_stubs
from pcstubgen2.StubGenerationOptions import StubGenerationOptions
from pcstubgen2.Writer import Writer as NewWriter

# Import old implementation parts
# We need to hack a bit because the old one is designed as CLI
try:
    from pybind11_stubgen import (
        stub_parser_from_args,
        Printer,
        run,
        Writer as OldWriter,
        CLIArgs,
    )
except ImportError as e:
    print(f"Failed to import reference pybind11_stubgen: {e}")
    sys.exit(1)


class MockArgs(CLIArgs):
    def __init__(self):
        self.output_dir = "."
        self.root_suffix = None
        self.ignore_invalid_expressions = None
        self.ignore_invalid_identifiers = None
        self.ignore_unresolved_names = None
        self.ignore_all_errors = False
        self.enum_class_locations = []
        self.numpy_array_wrap_with_annotated = False
        self.numpy_array_use_type_var = False
        self.numpy_array_remove_parameters = False
        self.print_invalid_expressions_as_is = False
        self.print_safe_value_reprs = None
        self.exit_code = False
        self.dry_run = False
        self.stub_extension = "pyi"
        self.module_name = "dummy"  # Overwritten later


def prepare_output_dir(base_dir: Path) -> Path:
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def generate_old(module_name, output_dir: Path):
    args = MockArgs()
    args.module_name = module_name

    parser = stub_parser_from_args(args)
    printer = Printer(invalid_expr_as_ellipses=not args.print_invalid_expressions_as_is)

    writer = OldWriter(stub_ext=args.stub_extension)

    # run() expects out_dir as Path
    run(
        parser,
        printer,
        args.module_name,
        output_dir,
        sub_dir=None,
        dry_run=False,
        writer=writer,
    )


def generate_new(module_name, options, output_dir: Path):
    writer = NewWriter(stub_extension=options.stub_extension)
    write_stubs(module_name, output_dir, options=options, writer=writer)


def normalize_import_block(code: str) -> str:
    lines = code.splitlines()
    if not lines:
        return code

    start = 0
    if lines[0].startswith('"""'):
        # find end of module docstring
        start = 1
        while start < len(lines):
            if lines[start].startswith('"""'):
                start += 1
                break
            start += 1

    # include blank lines after docstring
    while start < len(lines) and lines[start].strip() == "":
        start += 1

    # collect consecutive import lines
    import_start = start
    while start < len(lines) and (
        lines[start].startswith("import ") or lines[start].startswith("from ")
    ):
        start += 1
    import_end = start

    if import_end == import_start:
        return code

    import_lines = sorted(lines[import_start:import_end])
    normalized = [
        *lines[:import_start],
        *import_lines,
        *lines[import_end:],
    ]
    return "\n".join(normalized)


def read_stub_files(base_dir: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in base_dir.rglob("*.pyi"):
        rel_path = path.relative_to(base_dir).as_posix()
        files[rel_path] = path.read_text(encoding="utf-8")
    return files


def compare_outputs(
    old_dir: Path,
    new_dir: Path,
    diff_file: Path,
    normalize_imports: bool,
    title: str,
) -> None:
    old_files = read_stub_files(old_dir)
    new_files = read_stub_files(new_dir)

    old_keys = set(old_files.keys())
    new_keys = set(new_files.keys())

    only_old = sorted(old_keys - new_keys)
    only_new = sorted(new_keys - old_keys)
    common = sorted(old_keys & new_keys)

    diff_lines: list[str] = []

    def emit(line: str = "") -> None:
        print(line)
        diff_lines.append(line)

    emit(f"# {title}")
    emit()
    emit("## 概览")
    emit(f"- 旧版文件数: {len(old_keys)}")
    emit(f"- 新版文件数: {len(new_keys)}")
    emit(f"- 共同文件数: {len(common)}")
    emit(f"- 仅旧版存在: {len(only_old)}")
    emit(f"- 仅新版存在: {len(only_new)}")
    emit()
    if only_old:
        emit("## 仅旧版存在")
        for item in only_old:
            emit(f"- {item}")
        emit()
    if only_new:
        emit("## 仅新版存在")
        for item in only_new:
            emit(f"- {item}")
        emit()

    for rel_path in common:
        old_raw = old_files[rel_path]
        new_raw = new_files[rel_path]
        old_norm = normalize_import_block(old_raw) if normalize_imports else old_raw
        new_norm = normalize_import_block(new_raw) if normalize_imports else new_raw

        if old_raw == new_raw:
            continue

        emit(f"## 差异: {rel_path}")
        emit("```diff")
        diff = list(
            difflib.unified_diff(
                old_norm.splitlines(),
                new_norm.splitlines(),
                fromfile=f"Old:{rel_path}",
                tofile=f"New:{rel_path}",
                n=3,
            )
        )
        for line in diff[:300]:
            emit(line)
        if len(diff) > 200:
            emit("... 已截断")
        emit("```")
        emit()

    diff_file.write_text("\n".join(diff_lines) + "\n", encoding="utf-8")


def main():
    print(f"正在生成存根：{TARGET_MODULE}...")

    # Setup Options to match MockArgs
    args = MockArgs()
    options = StubGenerationOptions(
        root_suffix=args.root_suffix,
        ignore_invalid_expressions=args.ignore_invalid_expressions,
        ignore_invalid_identifiers=args.ignore_invalid_identifiers,
        ignore_unresolved_names=args.ignore_unresolved_names,
        ignore_all_errors=args.ignore_all_errors,
        enum_class_locations=args.enum_class_locations,
        numpy_array_wrap_with_annotated=args.numpy_array_wrap_with_annotated,
        numpy_array_use_type_var=args.numpy_array_use_type_var,
        numpy_array_remove_parameters=args.numpy_array_remove_parameters,
        print_invalid_expressions_as_is=args.print_invalid_expressions_as_is,
        print_safe_value_reprs=args.print_safe_value_reprs,
        stub_extension=args.stub_extension,
    )

    output_root = SCRIPT_DIR / "output" / "test_compare_stubs"
    old_dir = prepare_output_dir(output_root / "old")
    new_dir = prepare_output_dir(output_root / "new")

    # NEW
    print("运行新版生成器...")
    try:
        generate_new(TARGET_MODULE, options, new_dir)
        print(f"新版生成完成，输出目录：{new_dir}")
    except Exception as e:
        print(f"新版生成失败：{e}")
        import traceback

        traceback.print_exc()

    # OLD
    print("运行旧版生成器...")
    try:
        generate_old(TARGET_MODULE, old_dir)
        print(f"旧版生成完成，输出目录：{old_dir}")
    except Exception as e:
        print(f"旧版生成失败：{e}")
        import traceback

        traceback.print_exc()

    diff_raw_path = output_root / "diff_raw.md"
    diff_sorted_path = output_root / "diff_sorted.md"
    compare_outputs(
        old_dir,
        new_dir,
        diff_raw_path,
        normalize_imports=False,
        title="存根对比报告（原始）",
    )
    compare_outputs(
        old_dir,
        new_dir,
        diff_sorted_path,
        normalize_imports=True,
        title="存根对比报告（导入排序）",
    )
    print(f"原始 Diff 已写入：{diff_raw_path}")
    print(f"排序 Diff 已写入：{diff_sorted_path}")


if __name__ == "__main__":
    main()
