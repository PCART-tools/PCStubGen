"""
工具：合并多个 benchmark CSV，并输出总体耗时统计与累计计数图。

示例:
    uv run python tools/summarize_benchmark_csvs.py out/benchmark/*.csv
    uv run python tools/summarize_benchmark_csvs.py out/benchmark/*.csv --output out/benchmark_summary
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated

_MPL_CONFIG_DIR = Path("/tmp/matplotlib")
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib
import typer

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

_SOURCE_HAN_SANS_FONT = Path("/mnt/c/Windows/Fonts/SourceHanSansHWSC-Regular.otf")


def _configure_matplotlib_font() -> None:
    """把 Matplotlib 默认中文字体固定为思源黑体。"""
    font_manager.fontManager.addfont(str(_SOURCE_HAN_SANS_FONT))
    font_name = font_manager.FontProperties(fname=str(_SOURCE_HAN_SANS_FONT)).get_name()
    plt.rcParams["font.family"] = [font_name]
    plt.rcParams["axes.unicode_minus"] = False


_configure_matplotlib_font()

app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class BenchmarkRow:
    """保存单条 benchmark CSV 记录。"""

    source_csv: Path
    function_id: str
    elapsed_ns: int
    elapsed_ms: float
    provider: str | None
    mapping_status: str
    parameter_inference_status: str
    return_inference_status: str
    signature_count: int
    failure_reason: str | None


@dataclass(frozen=True)
class BenchmarkSummary:
    """保存合并后的总体耗时统计。"""

    csv_files: int
    functions: int
    total_elapsed_ms: float
    avg_elapsed_ms: float
    min_elapsed_ms: float
    p50_elapsed_ms: float
    p90_elapsed_ms: float
    p95_elapsed_ms: float
    max_elapsed_ms: float


def _load_rows(csv_paths: list[Path]) -> list[BenchmarkRow]:
    """读取并合并多个 benchmark CSV。"""
    rows: list[BenchmarkRow] = []
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            _validate_columns(csv_path, reader.fieldnames)
            for raw_row in reader:
                rows.append(_build_row(csv_path, raw_row))
    return rows


def _validate_columns(csv_path: Path, fieldnames: list[str] | None) -> None:
    """校验输入 CSV 是否包含 benchmark 必需列。"""
    if fieldnames is None:
        raise RuntimeError(f"CSV 文件 {csv_path} 缺少表头。")
    required_columns = {
        "function_id",
        "elapsed_ns",
        "elapsed_ms",
        "provider",
        "mapping_status",
        "parameter_inference_status",
        "return_inference_status",
        "signature_count",
        "failure_reason",
    }
    missing_columns = sorted(required_columns - set(fieldnames))
    if missing_columns:
        raise RuntimeError(
            f"CSV 文件 {csv_path} 缺少列: {', '.join(missing_columns)}"
        )


def _build_row(csv_path: Path, raw_row: dict[str, str]) -> BenchmarkRow:
    """把原始 CSV 行转换成强类型记录。"""
    provider = raw_row["provider"] or None
    failure_reason = raw_row["failure_reason"] or None
    return BenchmarkRow(
        source_csv=csv_path,
        function_id=raw_row["function_id"],
        elapsed_ns=int(raw_row["elapsed_ns"]),
        elapsed_ms=float(raw_row["elapsed_ms"]),
        provider=provider,
        mapping_status=raw_row["mapping_status"],
        parameter_inference_status=raw_row["parameter_inference_status"],
        return_inference_status=raw_row["return_inference_status"],
        signature_count=int(raw_row["signature_count"]),
        failure_reason=failure_reason,
    )


def _build_summary(rows: list[BenchmarkRow], csv_files: int) -> BenchmarkSummary:
    """汇总所有输入 CSV 的总体耗时统计。"""
    elapsed_values = sorted(row.elapsed_ms for row in rows)
    if not elapsed_values:
        raise RuntimeError("没有可汇总的 benchmark 行。")

    return BenchmarkSummary(
        csv_files=csv_files,
        functions=len(rows),
        total_elapsed_ms=sum(elapsed_values),
        avg_elapsed_ms=sum(elapsed_values) / len(elapsed_values),
        min_elapsed_ms=elapsed_values[0],
        p50_elapsed_ms=_percentile(elapsed_values, 0.50),
        p90_elapsed_ms=_percentile(elapsed_values, 0.90),
        p95_elapsed_ms=_percentile(elapsed_values, 0.95),
        max_elapsed_ms=elapsed_values[-1],
    )


def _percentile(sorted_values: list[float], ratio: float) -> float:
    """按最近秩近似计算分位数。"""
    index = int((len(sorted_values) - 1) * ratio)
    return sorted_values[index]


def _build_output_stem(now: datetime | None = None) -> str:
    """构造输出文件名前缀。"""
    if now is None:
        now = datetime.now()
    return f"benchmark_summary_{now.strftime('%Y%m%d_%H%M%S')}"


def _build_output_paths(
    output_dir: Path,
    now: datetime | None = None,
) -> tuple[Path, Path]:
    """构造汇总 JSON 与累计计数图输出路径。"""
    stem = _build_output_stem(now)
    return output_dir / f"{stem}.json", output_dir / f"{stem}.png"


def _write_summary_json(output_path: Path, summary: BenchmarkSummary, csv_paths: list[Path]) -> None:
    """把总体统计写入 JSON。"""
    payload = {
        "summary": {
            "csv_files": summary.csv_files,
            "functions": summary.functions,
            "total_elapsed_ms": summary.total_elapsed_ms,
            "avg_elapsed_ms": summary.avg_elapsed_ms,
            "min_elapsed_ms": summary.min_elapsed_ms,
            "p50_elapsed_ms": summary.p50_elapsed_ms,
            "p90_elapsed_ms": summary.p90_elapsed_ms,
            "p95_elapsed_ms": summary.p95_elapsed_ms,
            "max_elapsed_ms": summary.max_elapsed_ms,
        },
        "input_csvs": [str(csv_path) for csv_path in csv_paths],
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _plot_cumulative_counts(output_path: Path, rows: list[BenchmarkRow], summary: BenchmarkSummary) -> None:
    """绘制总体累计计数图，并标注 P90。"""
    elapsed_values = sorted(row.elapsed_ms for row in rows)
    cumulative_counts = list(range(1, len(elapsed_values) + 1))
    p90_index = int((len(elapsed_values) - 1) * 0.90)
    p90_elapsed_ms = elapsed_values[p90_index]
    p90_count = cumulative_counts[p90_index]

    figure, axes = plt.subplots(figsize=(10, 6))
    axes.step(
        elapsed_values,
        cumulative_counts,
        where="post",
        linewidth=2.0,
        color="#1f77b4",
        label="累积数量",
    )
    axes.scatter([p90_elapsed_ms], [p90_count], color="#d62728", s=40, zorder=3)
    axes.annotate(
        f"P90 = {p90_elapsed_ms:.3f} 毫秒\ncount = {p90_count}",
        xy=(p90_elapsed_ms, p90_count),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        color="#d62728",
    )
    axes.set_xlabel("消耗的时间/毫秒")
    axes.set_ylabel("C扩展API数量/个")
    axes.grid(True, alpha=0.3)
    axes.legend()
    axes.axvline(summary.p90_elapsed_ms, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _print_summary(summary: BenchmarkSummary) -> None:
    """打印总体统计结果。"""
    print("汇总:")
    print(f"  CSV 文件数: {summary.csv_files}")
    print(f"  函数数: {summary.functions}")
    print(f"  P90 耗时: {summary.p90_elapsed_ms:.3f} ms")


@app.command(help="合并多个 benchmark CSV，并输出总体统计与累计计数图。")
def main(
    csv_paths: Annotated[
        list[Path],
        typer.Argument(
            ...,
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="由 tools/rq4_benchmark_signature_completion.py 生成的 benchmark CSV 文件。",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output", help="汇总 JSON 和累计计数图的输出目录。"),
    ] = Path("out/benchmark_summary"),
) -> None:
    """Typer 命令入口。"""
    rows = _load_rows(csv_paths)
    summary = _build_summary(rows, len(csv_paths))

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path, plot_path = _build_output_paths(output_dir)
    _write_summary_json(summary_json_path, summary, csv_paths)
    _plot_cumulative_counts(plot_path, rows, summary)

    _print_summary(summary)
    print()
    print(f"汇总 JSON 已写入: {summary_json_path}")
    print(f"累计计数图已写入: {plot_path}")


if __name__ == "__main__":
    app()
