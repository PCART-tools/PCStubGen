"""
独立测试 pcstubgen2（单模块）。

示例:
    python test/<当前脚本>.py
    python test/<当前脚本>.py numpy.random
    python test/<当前脚本>.py math
"""

import shutil
import sys
import traceback
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
ROOT_DIR = SCRIPT_PATH.parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pcstubgen2 import write_stubs
from pcstubgen2.StubGenerationOptions import StubGenerationOptions

DEFAULT_MODULE = "numpy"
OUTPUT_DIR = SCRIPT_DIR / "output" / SCRIPT_PATH.stem


DEFAULT_C_SOURCE_ROOT = Path(r"C:/Things/third_package_source/numpy_numpy/numpy")
DEFAULT_CLANG_PARSE_ARGS = ["-std=c11"]

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


def prepare_output_dir(output_dir: Path) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def collect_stub_files(output_dir: Path) -> list[Path]:
    return sorted(output_dir.rglob("*.pyi"))


def run_single_module(module_name: str) -> int:
    print(f"开始测试 pcstubgen2，目标模块: {module_name}")
    output_dir = prepare_output_dir(OUTPUT_DIR)
    print(f"输出目录: {output_dir}")

    try:
        write_stubs(
            module_name,
            output_dir,
            options=StubGenerationOptions(include_docstrings=False,
            include_module_type_comment=True,
            c_source_root=DEFAULT_C_SOURCE_ROOT,
            clang_parse_args=DEFAULT_CLANG_PARSE_ARGS,
            ),
        )
    except Exception as exc:
        print(f"生成失败: {exc}")
        traceback.print_exc()
        return 1

    stub_files = collect_stub_files(output_dir)
    if not stub_files:
        print("生成失败: 未找到任何 .pyi 文件")
        return 2

    sample = stub_files[0]
    try:
        sample_display = sample.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        sample_display = sample.as_posix()

    print("生成成功")
    print(f"- 模块名: {module_name}")
    print(f"- 生成文件数: {len(stub_files)}")
    print(f"- 示例文件: {sample_display}")
    return 0


def main() -> int:
    configure_output_encoding()
    module_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODULE
    return run_single_module(module_name)


if __name__ == "__main__":
    raise SystemExit(main())
