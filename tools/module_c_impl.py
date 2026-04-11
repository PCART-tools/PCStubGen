"""
工具：输入模块名，输出模块与子模块是否为 C 实现。

示例:
    python -m tools.module_c_impl
    python -m tools.module_c_impl numpy.random
    python tools/module_c_impl.py math
"""

from __future__ import annotations

import csv
import importlib
import importlib.machinery
import importlib.util
import inspect
from pathlib import Path
import types

import typer


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
ROOT_DIR = SCRIPT_DIR.parent

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / f"{SCRIPT_PATH.stem}_output"
EXIT_ERROR = 1

app = typer.Typer(add_completion=False)


def is_c_implemented(module_name: str) -> bool:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        module = importlib.import_module(module_name)
        spec = module.__spec__
    loader = spec.loader if spec is not None else None
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
        member_name = member.__name__
        if member_name.startswith(prefix):
            result.append(member_name)
    return result


def collect_module_names(module_name: str) -> list[str]:
    """
    从根模块开始遍历已暴露的子模块属性，并返回去重后的模块名列表。
    """
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
            try:
                member_module = importlib.import_module(sub_name)
                if inspect.ismodule(member_module):
                    pending_modules.append(member_module)
            except Exception as exc:
                print(f"警告: 无法导入子模块 {sub_name!r}: {exc}")

    return sorted(all_names)


def output_file_for(module_name: str, *, output: Path) -> Path:
    return output / f"{module_name}.csv"


def write_report(module_names: list[str], report_path: Path) -> int:
    """
    将模块检查结果写入 CSV，并返回其中 C 实现模块的数量。
    """
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

    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["module_name", "is_c_implemented"])
        rows.sort(key=lambda item: not item[1])
        writer.writerows(rows)
    return c_module_count


def run_single_module(module_name: str, *, output: Path) -> int:
    """
    运行单个模块检查流程，并将结果写入指定输出目录。
    """
    print(f"开始检查模块: {module_name}")

    try:
        module_names = collect_module_names(module_name)
    except Exception as exc:
        print(f"检查失败: {exc}")
        return EXIT_ERROR

    report_path = output_file_for(module_name, output=output)
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


@app.command(help="检查模块及其子模块是否为 C 实现，并导出 CSV 报告。")
def command(
    module_name: str = typer.Argument(
        ...,
        metavar="MODULE_NAME",
        help="待检查的模块名。",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="CSV 输出目录，默认写入 tool/module_c_impl_output。",
    ),
) -> None:
    effective_output = DEFAULT_OUTPUT_DIR if output is None else output
    exit_code = run_single_module(module_name, output=effective_output)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    app()
