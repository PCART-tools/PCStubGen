from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pcstubgen2 import write_stubs
from pcstubgen2.StubGenerationOptions import StubGenerationOptions

DEFAULT_MODULE = "numpy"
DEFAULT_SOURCE_ROOT = Path(r"C:/Things/third_package_source/numpy_numpy/numpy")
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "output" / SCRIPT_PATH.stem

EXIT_OK = 0
EXIT_GENERATION_FAILED = 1
EXIT_REPORT_FAILED = 2


@dataclass(frozen=True)
class GenerationResult:
    c_inference_enabled: bool
    output_dir: Path
    success: bool
    stub_count: int
    generic_signature_count: int
    elapsed_seconds: float
    error: str | None = None


def c_inference_label(enabled: bool) -> str:
    return "enabled" if enabled else "disabled"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="手工分析用：生成 numpy 存根，并对比 C 签名推导开关前后的结果。"
    )
    parser.add_argument(
        "module_name",
        nargs="?",
        default=DEFAULT_MODULE,
        help=f"目标包名，默认: {DEFAULT_MODULE}",
    )
    parser.add_argument(
        "--source-root",
        default=str(DEFAULT_SOURCE_ROOT),
        help=f"C 源码根目录，默认: {DEFAULT_SOURCE_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=f"输出根目录，默认: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--clang-include",
        action="append",
        default=[],
        help="追加 libclang include 路径，可重复传入（不要带 -I 前缀）。",
    )
    parser.add_argument(
        "--clang-c-std",
        default="c11",
        help="C 源文件使用的标准版本，例如 c11。",
    )
    parser.add_argument(
        "--clang-cpp-std",
        default="c++17",
        help="C++ 源文件使用的标准版本，例如 c++17。",
    )
    return parser.parse_args()


def prepare_output_dir(output_dir: Path) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def collect_stub_files(output_dir: Path, stub_extension: str = "pyi") -> list[Path]:
    return sorted(output_dir.rglob(f"*.{stub_extension}"))


def count_generic_signatures(stub_files: list[Path]) -> int:
    pattern = re.compile(r"\*args,\s*\*\*kwargs")
    total = 0
    for file_path in stub_files:
        text = file_path.read_text(encoding="utf-8")
        total += len(pattern.findall(text))
    return total


def top_generic_hotspots(output_dir: Path, stub_files: list[Path], limit: int = 15) -> list[tuple[str, int]]:
    pattern = re.compile(r"\*args,\s*\*\*kwargs")
    hotspots: list[tuple[str, int]] = []
    for file_path in stub_files:
        text = file_path.read_text(encoding="utf-8")
        count = len(pattern.findall(text))
        if count <= 0:
            continue
        relative_path = file_path.relative_to(output_dir).as_posix()
        hotspots.append((relative_path, count))
    hotspots.sort(key=lambda item: (-item[1], item[0]))
    return hotspots[:limit]


def run_single_generation(
    module_name: str,
    output_dir: Path,
    c_inference_enabled: bool,
    source_root: Path | None,
    clang_include: list[str],
    clang_c_std: str,
    clang_cpp_std: str,
) -> GenerationResult:
    prepare_output_dir(output_dir)
    started = time.perf_counter()
    try:
        options = StubGenerationOptions(
            enable_c_signature_inference=c_inference_enabled,
            source_root=source_root if c_inference_enabled else None,
            clang_include=clang_include,
            clang_c_std=clang_c_std,
            clang_cpp_std=clang_cpp_std,
            include_docstrings=False,
            include_module_type_comment=True,
        )
        write_stubs(module_name, output_dir, options=options)
        stub_files = collect_stub_files(output_dir, stub_extension=options.stub_extension)
        if not stub_files:
            elapsed = time.perf_counter() - started
            return GenerationResult(
                c_inference_enabled=c_inference_enabled,
                output_dir=output_dir,
                success=False,
                stub_count=0,
                generic_signature_count=0,
                elapsed_seconds=elapsed,
                error="未生成任何 .pyi 文件。",
            )
        generic_count = count_generic_signatures(stub_files)
        elapsed = time.perf_counter() - started
        return GenerationResult(
            c_inference_enabled=c_inference_enabled,
            output_dir=output_dir,
            success=True,
            stub_count=len(stub_files),
            generic_signature_count=generic_count,
            elapsed_seconds=elapsed,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return GenerationResult(
            c_inference_enabled=c_inference_enabled,
            output_dir=output_dir,
            success=False,
            stub_count=0,
            generic_signature_count=0,
            elapsed_seconds=elapsed,
            error=str(exc),
        )


def format_hotspots(title: str, hotspots: list[tuple[str, int]]) -> list[str]:
    lines = [f"## {title}", ""]
    if not hotspots:
        lines.append("- 未发现 `*args, **kwargs`。")
        lines.append("")
        return lines
    lines.append("| 文件 | `*args, **kwargs` 次数 |")
    lines.append("|---|---:|")
    for relative_path, count in hotspots:
        lines.append(f"| `{relative_path}` | {count} |")
    lines.append("")
    return lines


def write_report(
    report_path: Path,
    module_name: str,
    source_root: Path,
    disabled_result: GenerationResult,
    enabled_result: GenerationResult,
) -> None:
    disabled_stub_files = collect_stub_files(disabled_result.output_dir)
    enabled_stub_files = collect_stub_files(enabled_result.output_dir)
    disabled_hotspots = top_generic_hotspots(disabled_result.output_dir, disabled_stub_files)
    enabled_hotspots = top_generic_hotspots(enabled_result.output_dir, enabled_stub_files)

    delta = disabled_result.generic_signature_count - enabled_result.generic_signature_count
    trend = "减少" if delta > 0 else "增加或不变"

    lines: list[str] = []
    lines.append("# Numpy 存根生成手工分析报告")
    lines.append("")
    lines.append("## 运行参数")
    lines.append("")
    lines.append(f"- 目标模块: `{module_name}`")
    lines.append(f"- C 源码目录: `{source_root.as_posix()}`")
    lines.append(f"- 关闭 C 推导输出目录: `{disabled_result.output_dir.as_posix()}`")
    lines.append(f"- 启用 C 推导输出目录: `{enabled_result.output_dir.as_posix()}`")
    lines.append("")
    lines.append("## 结果汇总")
    lines.append("")
    lines.append("| c_inference | 成功 | `.pyi` 文件数 | `*args, **kwargs` 次数 | 耗时(秒) | 错误 |")
    lines.append("|---|---|---:|---:|---:|---|")
    lines.append(
        f"| `{c_inference_label(disabled_result.c_inference_enabled)}` | "
        f"{'是' if disabled_result.success else '否'} | "
        f"{disabled_result.stub_count} | {disabled_result.generic_signature_count} | "
        f"{disabled_result.elapsed_seconds:.2f} | {disabled_result.error or '-'} |"
    )
    lines.append(
        f"| `{c_inference_label(enabled_result.c_inference_enabled)}` | "
        f"{'是' if enabled_result.success else '否'} | "
        f"{enabled_result.stub_count} | {enabled_result.generic_signature_count} | "
        f"{enabled_result.elapsed_seconds:.2f} | {enabled_result.error or '-'} |"
    )
    lines.append("")
    lines.append("## 观察建议")
    lines.append("")
    lines.append(
        f"- 相比 `{c_inference_label(disabled_result.c_inference_enabled)}`，"
        f"`{c_inference_label(enabled_result.c_inference_enabled)}` 的泛型签名计数变化: `{delta}`（{trend}）。"
    )
    lines.append("- 优先人工检查热点文件中函数签名是否从泛型变为具体参数。")
    if not source_root.exists():
        lines.append("- 注意：当前 `source_root` 不存在，C 签名推导可能被跳过。")
    lines.append("")

    lines.extend(
        format_hotspots(
            f"{c_inference_label(disabled_result.c_inference_enabled)} 热点（Top 15）",
            disabled_hotspots,
        )
    )
    lines.extend(
        format_hotspots(
            f"{c_inference_label(enabled_result.c_inference_enabled)} 热点（Top 15）",
            enabled_hotspots,
        )
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    configure_output_encoding()
    args = parse_args()

    module_name = args.module_name
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    clang_include: list[str] = list(args.clang_include)
    clang_c_std: str = args.clang_c_std
    clang_cpp_std: str = args.clang_cpp_std

    disabled_output_dir = output_root / "disabled"
    enabled_output_dir = output_root / "enabled"
    report_path = output_root / "report.md"

    print(f"开始生成存根，目标模块: {module_name}")
    print(f"输出根目录: {output_root}")
    print(f"C 源码目录: {source_root}")
    if not source_root.exists():
        print("警告: C 源码目录不存在，C 签名推导可能不会生效。")
    if clang_include:
        print(f"clang include 路径: {clang_include}")
    if clang_c_std:
        print(f"clang C 标准: {clang_c_std}")
    if clang_cpp_std:
        print(f"clang C++ 标准: {clang_cpp_std}")

    print("\n[1/2] 生成关闭 C 推导版本（enable_c_signature_inference=False）...")
    disabled_result = run_single_generation(
        module_name=module_name,
        output_dir=disabled_output_dir,
        c_inference_enabled=False,
        source_root=source_root,
        clang_include=clang_include,
        clang_c_std=clang_c_std,
        clang_cpp_std=clang_cpp_std,
    )
    print(
        f"完成: success={disabled_result.success}, stubs={disabled_result.stub_count}, "
        f"generic={disabled_result.generic_signature_count}, elapsed={disabled_result.elapsed_seconds:.2f}s"
    )
    print(f"输出目录: {disabled_result.output_dir}")
    if disabled_result.error:
        print(f"错误: {disabled_result.error}")

    print("\n[2/2] 生成启用 C 推导版本（enable_c_signature_inference=True）...")
    enabled_result = run_single_generation(
        module_name=module_name,
        output_dir=enabled_output_dir,
        c_inference_enabled=True,
        source_root=source_root,
        clang_include=clang_include,
        clang_c_std=clang_c_std,
        clang_cpp_std=clang_cpp_std,
    )
    print(
        f"完成: success={enabled_result.success}, stubs={enabled_result.stub_count}, "
        f"generic={enabled_result.generic_signature_count}, elapsed={enabled_result.elapsed_seconds:.2f}s"
    )
    print(f"输出目录: {enabled_result.output_dir}")
    if enabled_result.error:
        print(f"错误: {enabled_result.error}")

    try:
        write_report(
            report_path=report_path,
            module_name=module_name,
            source_root=source_root,
            disabled_result=disabled_result,
            enabled_result=enabled_result,
        )
        print(f"\n报告已生成: {report_path}")
    except Exception as exc:
        print(f"\n报告生成失败: {exc}")
        return EXIT_REPORT_FAILED

    print("\n建议下一步: 打开 report.md，并在两套输出目录中按热点文件做人工比对。")

    if not disabled_result.success or not enabled_result.success:
        print(f"退出码: {EXIT_GENERATION_FAILED}（至少一轮生成失败）")
        return EXIT_GENERATION_FAILED

    print(f"退出码: {EXIT_OK}（两轮生成成功）")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())

