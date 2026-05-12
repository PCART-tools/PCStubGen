"""
工具：汇总新版函数层 TOML 与 RQ3 CSV 评估结果。

示例:
    uv run python tools/summarize_rq_results.py out/pcstubgen/ujson.toml
    uv run python tools/summarize_rq_results.py out/pcstubgen/ujson.toml --evaluation-csv out/pcstubgen/ujson_eval_20260510_120000.csv
"""

from __future__ import annotations

import csv
import json
import tomllib
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(add_completion=False)

VALID_RQ3_CODES = {
    "rule_unqualified",
    "unqualified",
    "qualified",
}
VALID_RQ3_STATUSES = {
    "ok",
    "llm_error",
}


def _load_toml_functions(path: Path) -> list[dict[str, Any]]:
    """读取新版函数层 TOML。"""
    with path.open("rb") as file:
        payload = tomllib.load(file)
    functions = payload.get("functions")
    if not isinstance(functions, list):
        raise RuntimeError(f"TOML 文件 {path} 缺少新版 functions 列表。")
    return functions


def _load_evaluation_rows(path: Path | None) -> list[dict[str, Any]]:
    """读取 RQ3 CSV 评估结果。"""
    if path is None:
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(dict(row))
    return rows


def _percentage(numerator: int, denominator: int) -> float | None:
    """计算百分比，分母为 0 时返回 None。"""
    if denominator == 0:
        return None
    return numerator / denominator * 100


def _summarize_functions(functions: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总 RQ1/RQ2 函数级指标。"""
    total = len(functions)
    mapping_success = sum(1 for function in functions if function.get("mapping_status") == "success")
    mapped_functions = [
        function
        for function in functions
        if function.get("mapping_status") == "success"
    ]
    parameter_success = sum(
        1
        for function in mapped_functions
        if function.get("parameter_inference_status") == "success"
    )
    return_success = sum(
        1
        for function in mapped_functions
        if function.get("return_inference_status") == "success"
    )
    both_success = sum(
        1
        for function in mapped_functions
        if (
            function.get("parameter_inference_status") == "success"
            and function.get("return_inference_status") == "success"
        )
    )
    provider_counts: dict[str, int] = {}
    for function in functions:
        provider = function.get("provider")
        provider_text = provider if isinstance(provider, str) and provider else "<unknown>"
        provider_counts[provider_text] = provider_counts.get(provider_text, 0) + 1

    return {
        "total_functions": total,
        "provider_counts": provider_counts,
        "rq1_mapping_success": mapping_success,
        "rq1_mapping_success_rate": _percentage(mapping_success, total),
        "rq2_inference_denominator": mapping_success,
        "rq2_parameter_inference_success": parameter_success,
        "rq2_parameter_inference_success_rate": _percentage(parameter_success, mapping_success),
        "rq2_return_inference_success": return_success,
        "rq2_return_inference_success_rate": _percentage(return_success, mapping_success),
        "rq2_both_inference_success": both_success,
        "rq2_both_inference_success_rate": _percentage(both_success, mapping_success),
    }


def _count_signatures(functions: list[dict[str, Any]]) -> int:
    """统计 TOML 中的签名条数。"""
    total = 0
    for function in functions:
        signatures = function.get("signatures")
        if isinstance(signatures, list):
            total += len(signatures)
    return total


def _summarize_evaluations(
    functions: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """汇总 RQ3 签名级指标。"""
    _validate_evaluation_rows(evaluation_rows)
    total_signatures = _count_signatures(functions)
    dimension_fields = {
        "parameter_structure": "parameter_structure_code",
        "parameter_type": "parameter_type_code",
        "return_type": "return_type_code",
    }
    result: dict[str, Any] = {
        "total_signatures": total_signatures,
        "evaluated_rows": len(evaluation_rows),
    }
    for dimension, field_name in dimension_fields.items():
        rule_unqualified = _count_dimension_code(evaluation_rows, field_name, "rule_unqualified")
        qualified = _count_dimension_code(evaluation_rows, field_name, "qualified")
        unqualified = _count_dimension_code(evaluation_rows, field_name, "unqualified")
        result[f"rq3_{dimension}_rule_unqualified_count"] = rule_unqualified
        result[f"rq3_{dimension}_qualified_count"] = qualified
        result[f"rq3_{dimension}_unqualified_count"] = unqualified
    return result


def _validate_evaluation_rows(evaluation_rows: list[dict[str, Any]]) -> None:
    """校验 RQ3 评估结果是否可用于汇总。"""
    for row in evaluation_rows:
        status = row.get("status")
        if status not in VALID_RQ3_STATUSES:
            raise RuntimeError(
                "RQ3 评估 CSV 包含非法 status。"
                f" function_id={row.get('function_id', '')},"
                f" signature_index={row.get('signature_index', '')},"
                f" status={status!r}"
            )
        if status == "llm_error":
            raise RuntimeError(
                "RQ3 评估 CSV 包含 llm_error 记录，无法汇总。"
                f" function_id={row.get('function_id', '')},"
                f" signature_index={row.get('signature_index', '')}"
            )
        for field_name in (
            "parameter_structure_code",
            "parameter_type_code",
            "return_type_code",
        ):
            code = row.get(field_name)
            if code not in VALID_RQ3_CODES:
                raise RuntimeError(
                    "RQ3 评估 CSV 包含非法维度判定码。"
                    f" function_id={row.get('function_id', '')},"
                    f" signature_index={row.get('signature_index', '')},"
                    f" field={field_name},"
                    f" code={code!r}"
                )


def _count_dimension_code(
    evaluation_rows: list[dict[str, Any]],
    field_name: str,
    code: str,
) -> int:
    """统计某个评估维度上的指定判定码数量。"""
    return sum(
        1
        for row in evaluation_rows
        if row.get(field_name) == code
    )


def summarize(
    generated_toml: Path,
    *,
    evaluation_csv: Path | None,
) -> dict[str, Any]:
    """汇总单个库的 RQ 指标。"""
    functions = _load_toml_functions(generated_toml)
    evaluation_rows = _load_evaluation_rows(evaluation_csv)
    return {
        "library": generated_toml.stem,
        "generated_toml": str(generated_toml),
        "evaluation_csv": None if evaluation_csv is None else str(evaluation_csv),
        "rq1_rq2": _summarize_functions(functions),
        "rq3": _summarize_evaluations(functions, evaluation_rows),
    }


@app.command(help="汇总新版 TOML 与 CSV 评估结果中的 RQ 指标。")
def command(
    generated_toml: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="由 `pcstubgen gen --toml` 生成的新版函数层 TOML 文件。",
    ),
    evaluation_csv: Path | None = typer.Option(
        None,
        "--evaluation-csv",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="由 tools/rq3_eval_inferred_signatures.py 生成的 CSV 评估结果。",
    ),
) -> None:
    """Typer 命令入口。"""
    summary = summarize(generated_toml, evaluation_csv=evaluation_csv)
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
