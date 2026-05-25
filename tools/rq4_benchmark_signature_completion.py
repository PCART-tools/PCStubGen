"""
工具：复用 ModuleCollector，逐函数执行一次签名补全，并收集 `complete()` 单次耗时。

示例:
    uv run python tools/rq4_benchmark_signature_completion.py ujson --compilation-database ./build/compile_commands.json --output ./benchmark_output
"""

from __future__ import annotations

import csv
import dataclasses
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from pcstubgen.module_collector import ModuleCollector
from pcstubgen.signature_completion import SignatureCompleter
from pcstubgen.signature_completion.completion_models import (
    SignatureCompletionContext,
    SignatureCompletionResult,
    UnsupportedSignatureCompletion,
)


@dataclass(frozen=True)
class BenchmarkMeasurement:
    """保存单次 `complete()` 调用的耗时和结果摘要。"""

    function_id: str
    elapsed_ns: int
    elapsed_ms: float
    provider: str | None
    mapping_status: str
    parameter_inference_status: str
    return_inference_status: str
    signature_count: int
    failure_reason: str | None


class BenchmarkSignatureCompleter(SignatureCompleter):
    """在真实签名补全外包一层单次耗时采集。"""

    def __init__(self, compilation_database: Path) -> None:
        """创建真实 completer，并保存计时结果容器。"""
        super().__init__(compilation_database)
        self.measurements: list[BenchmarkMeasurement] = []

    def complete(self, context: SignatureCompletionContext) -> SignatureCompletionResult:
        """执行真实补全，并记录单次耗时。"""
        function_id = _build_function_id(context)
        started_ns = time.perf_counter_ns()
        try:
            result = super().complete(context)
        except UnsupportedSignatureCompletion:
            raise

        elapsed_ns = time.perf_counter_ns() - started_ns
        self.measurements.append(_build_measurement(function_id, elapsed_ns, result))
        return result


def _build_function_id(context: SignatureCompletionContext) -> str:
    """构造稳定的函数级 benchmark 名称。"""
    if context.owner_class is None:
        return f"{context.module_name}:{context.func_name}"
    owner_name = context.owner_class.__qualname__
    return f"{context.module_name}:{owner_name}.{context.func_name}"


def _build_measurement(
    function_id: str,
    elapsed_ns: int,
    result: SignatureCompletionResult,
) -> BenchmarkMeasurement:
    """把补全结果转换成可序列化的耗时记录。"""
    return BenchmarkMeasurement(
        function_id=function_id,
        elapsed_ns=elapsed_ns,
        elapsed_ms=elapsed_ns / 1_000_000,
        provider=result.provider,
        mapping_status=result.mapping_status,
        parameter_inference_status=result.parameter_inference_status,
        return_inference_status=result.return_inference_status,
        signature_count=len(result.signatures),
        failure_reason=result.failure_reason,
    )


def _build_summary(measurements: list[BenchmarkMeasurement]) -> dict[str, int | float]:
    """汇总本次单次补全耗时统计。"""
    elapsed_values = [measurement.elapsed_ns for measurement in measurements]
    return {
        "functions": len(measurements),
        "total_elapsed_ms": sum(elapsed_values) / 1_000_000,
        "avg_elapsed_ms": (sum(elapsed_values) / len(elapsed_values) / 1_000_000)
        if elapsed_values
        else 0.0,
        "min_elapsed_ms": (min(elapsed_values) / 1_000_000) if elapsed_values else 0.0,
        "max_elapsed_ms": (max(elapsed_values) / 1_000_000) if elapsed_values else 0.0,
        "p50_elapsed_ms": _percentile_elapsed_ms(elapsed_values, 0.50),
        "p90_elapsed_ms": _percentile_elapsed_ms(elapsed_values, 0.90),
        "p95_elapsed_ms": _percentile_elapsed_ms(elapsed_values, 0.95),
    }


def _percentile_elapsed_ms(elapsed_values: list[int], ratio: float) -> float:
    """按最近秩近似计算耗时分位数。"""
    if not elapsed_values:
        return 0.0

    sorted_values = sorted(elapsed_values)
    index = int((len(sorted_values) - 1) * ratio)
    return sorted_values[index] / 1_000_000


def _write_output(
    output_dir: Path,
    module_name: str,
    compilation_database: Path,
    measurements: list[BenchmarkMeasurement],
) -> tuple[Path, Path]:
    """把基础信息写入 JSON，把逐函数耗时明细写入 CSV。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_output_path, csv_output_path = _build_output_paths(output_dir, module_name)
    payload = {
        "module_name": module_name,
        "compilation_database": str(compilation_database),
        "summary": _build_summary(measurements),
        "timing_csv": str(csv_output_path),
    }
    json_output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_timing_csv(csv_output_path, measurements)
    return json_output_path, csv_output_path


def _build_output_file_stem(module_name: str, now: datetime | None = None) -> str:
    """构造 `{模块名}_benchmark_{时间}` 文件名前缀。"""
    if now is None:
        now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    return f"{module_name}_benchmark_{timestamp}"


def _build_output_paths(
    output_dir: Path,
    module_name: str,
    now: datetime | None = None,
) -> tuple[Path, Path]:
    """根据输出目录和模块名构造同一时间戳的 JSON/CSV 输出路径。"""
    stem = _build_output_file_stem(module_name, now)
    return output_dir / f"{stem}.json", output_dir / f"{stem}.csv"


def _write_timing_csv(
    csv_output_path: Path,
    measurements: list[BenchmarkMeasurement],
) -> None:
    """把逐函数耗时明细写入单个 CSV。"""
    rows = _build_csv_rows(measurements)
    fieldnames = [field.name for field in dataclasses.fields(BenchmarkMeasurement)]
    with csv_output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dataclasses.asdict(row))


def _build_csv_rows(
    measurements: list[BenchmarkMeasurement],
) -> list[BenchmarkMeasurement]:
    """构造按耗时倒序排列的 CSV 行。"""
    return sorted(
        measurements,
        key=lambda current: current.elapsed_ns,
        reverse=True,
    )


def _print_measurements(measurements: list[BenchmarkMeasurement]) -> None:
    """按耗时倒序打印单函数结果，便于直接查看慢函数。"""
    for measurement in sorted(
        measurements,
        key=lambda current: current.elapsed_ns,
        reverse=True,
    ):
        print(
            f"{measurement.elapsed_ms:9.3f} ms | "
            f"{measurement.provider or '-':11s} | "
            f"{measurement.function_id}"
        )


def _print_summary(summary: dict[str, int | float]) -> None:
    """打印本次单次补全耗时摘要。"""
    print()
    print("汇总:")
    print(f"  函数数: {summary['functions']}")
    print(f"  总耗时: {summary['total_elapsed_ms']:.3f} ms")
    print(f"  平均耗时: {summary['avg_elapsed_ms']:.3f} ms")
    print(f"  最小耗时: {summary['min_elapsed_ms']:.3f} ms")
    print(f"  P50 耗时: {summary['p50_elapsed_ms']:.3f} ms")
    print(f"  P90 耗时: {summary['p90_elapsed_ms']:.3f} ms")
    print(f"  P95 耗时: {summary['p95_elapsed_ms']:.3f} ms")
    print(f"  最大耗时: {summary['max_elapsed_ms']:.3f} ms")


def main(
    module_name: Annotated[str, typer.Argument(help="模块名。")],
    compilation_database: Annotated[
        Path,
        typer.Option("--compilation-database", help="compile_commands.json 文件路径。"),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option("--output", help="可选的输出目录。"),
    ] = None,
) -> None:
    """脚本入口。"""
    completer = BenchmarkSignatureCompleter(compilation_database)
    collector = ModuleCollector(completer)
    collector.run(module_name)

    summary = _build_summary(completer.measurements)

    print(f"收集并计时函数数量: {len(completer.measurements)}")
    _print_measurements(completer.measurements)
    _print_summary(summary)

    if output_dir is not None:
        json_output_path, csv_output_path = _write_output(
            output_dir,
            module_name,
            compilation_database,
            completer.measurements,
        )
        print()
        print(f"JSON 已写入: {json_output_path}")
        print(f"CSV 已写入: {csv_output_path}")


if __name__ == "__main__":
    typer.run(main)
