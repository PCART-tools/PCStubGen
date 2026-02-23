"""
独立测试：输入模块名，输出模块与子模块是否为 C 实现。

示例:
    python test/test_module_c_impl.py
    python test/test_module_c_impl.py numpy.random
    python test/test_module_c_impl.py math
"""

from __future__ import annotations

from contextlib import suppress
import csv
import inspect
import importlib
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
ROOT_DIR = SCRIPT_PATH.parents[1]

DEFAULT_MODULE = "numpy.random"
OUTPUT_DIR = SCRIPT_DIR / "output" / SCRIPT_PATH.stem


def configure_output_encoding() -> None:
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


def is_c_implemented(module_name: str) -> bool:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        module = importlib.import_module(module_name)
        spec = getattr(module, "__spec__", None)
    loader = getattr(spec, "loader", None) if spec is not None else None
    return (
        isinstance(loader, importlib.machinery.ExtensionFileLoader)
        or loader is importlib.machinery.BuiltinImporter
    )


def _iter_submodule_names(module: types.ModuleType) -> list[str]:
    prefix = module.__name__ + "."
    result: list[str] = []
    for _, member in inspect.getmembers(module):
        if not inspect.ismodule(member):
            continue
        member_name = getattr(member, "__name__", "")
        if member_name.startswith(prefix):
            result.append(member_name)
    return result


def collect_module_names(module_name: str) -> list[str]:
    root_module = importlib.import_module(module_name)
    pending_modules: list[types.ModuleType] = [root_module]
    visited_names: set[str] = set()
    all_names: set[str] = set()

    while pending_modules:
        current = pending_modules.pop()
        current_name = current.__name__
        if current_name in visited_names:
            continue
        visited_names.add(current_name)
        all_names.add(current_name)

        for sub_name in _iter_submodule_names(current):
            all_names.add(sub_name)
            if sub_name in visited_names:
                continue
            with suppress(Exception):
                member_module = importlib.import_module(sub_name)
                if inspect.ismodule(member_module):
                    pending_modules.append(member_module)

    return sorted(all_names)


def output_file_for(module_name: str) -> Path:
    return OUTPUT_DIR / f"{module_name}.csv"


def write_report(module_names: list[str], report_path: Path) -> int:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    c_module_count = 0
    rows: list[tuple[str, bool]] = []

    for name in module_names:
        try:
            is_c = is_c_implemented(name)
        except Exception as exc:
            print(f"警告: 无法判断模块 {name!r}: {exc}")
            is_c = False

        if is_c:
            c_module_count += 1
        rows.append((name, is_c))

    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["module_name", "is_c_implemented"])
        writer.writerows(rows)
    return c_module_count


def run_single_module(module_name: str) -> int:
    print(f"开始检查模块: {module_name}")

    try:
        module_names = collect_module_names(module_name)
    except Exception as exc:
        print(f"检查失败: {exc}")
        return 1

    report_path = output_file_for(module_name)
    c_count = write_report(module_names, report_path)

    try:
        report_display = report_path.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        report_display = report_path.as_posix()

    print("检查完成")
    print(f"- 模块总数: {len(module_names)}")
    print(f"- C实现数: {c_count}")
    print(f"- 输出文件: {report_display}")
    return 0


def main() -> int:
    configure_output_encoding()
    module_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODULE
    return run_single_module(module_name)


if __name__ == "__main__":
    raise SystemExit(main())
