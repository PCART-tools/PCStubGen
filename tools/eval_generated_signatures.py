"""
工具：批量评估已生成 TOML 中的签名是否合格。

示例:
    uv run python tools/eval_generated_signatures.py out/pcstubgen/psycopg2.toml out/pcstubgen/psycopg2_eval.csv
    uv run python tools/eval_generated_signatures.py out/pcstubgen/psycopg2.toml out/pcstubgen/psycopg2_eval.csv --manual-stub-root ./stubs
"""

from __future__ import annotations

import ast
import asyncio
import csv
import json
import os
import re
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import typer
from openai import AsyncOpenAI
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-pro"
REASONING_EFFORT = "max"
DEFAULT_CONCURRENCY = 16
DEFAULT_MAX_ATTEMPTS = 3
EXIT_ERROR = 1

VALID_REASON_CODES = {
    "parameter_mismatch",
    "return_mismatch",
    "overload_mismatch",
}

CSV_HEADERS = [
    "module_name",
    "class_name",
    "function_name",
    "generated_signature",
    "status",
    "llm_verdict",
    "reason_code",
    "reason",
    "reference_kind",
    "reference_path",
    "reference_line",
]

app = typer.Typer(add_completion=False)

ClassPath = tuple[str, ...]


@dataclass(frozen=True)
class GeneratedSignatureEntry:
    """单条生成签名记录。"""

    module_name: str
    class_name: str | None
    function_name: str
    generated_signature: str
    comment: str | None


@dataclass(frozen=True)
class ManualStubSignature:
    """人工 stub 中提取出的单条签名。"""

    text: str
    line: int


@dataclass(frozen=True)
class ManualStubReference:
    """人工 stub 中定位到的一组参考签名。"""

    path: Path
    signatures: list[ManualStubSignature]


@dataclass
class ParsedStubModule:
    """单个 `.pyi` 模块解析后的索引。"""

    path: Path
    functions: dict[ClassPath, dict[str, list[ManualStubSignature]]] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedEvaluation:
    """已经解析好参考材料、可直接进入评估的任务。"""

    entry: GeneratedSignatureEntry
    reference_kind: str
    reference_payload: str
    reference_path: str
    reference_line: str


@dataclass(frozen=True)
class EvaluationRow:
    """CSV 输出的一行结果。"""

    module_name: str
    class_name: str
    function_name: str
    generated_signature: str
    status: str
    llm_verdict: str
    reason_code: str
    reason: str
    reference_kind: str
    reference_path: str
    reference_line: str

    def to_csv_row(self) -> dict[str, str]:
        """将结果行转换为 CSV writer 可接受的字典。"""
        return {
            "module_name": self.module_name,
            "class_name": self.class_name,
            "function_name": self.function_name,
            "generated_signature": self.generated_signature,
            "status": self.status,
            "llm_verdict": self.llm_verdict,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "reference_kind": self.reference_kind,
            "reference_path": self.reference_path,
            "reference_line": self.reference_line,
        }


class ManualStubRepository:
    """按模块定位并解析人工维护的 `.pyi` 参考。"""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._cache: dict[str, ParsedStubModule | None] = {}

    def get_reference(self, entry: GeneratedSignatureEntry) -> ManualStubReference | None:
        """返回某条生成签名对应的人工 stub 参考。"""
        parsed_module = self._load_module(entry.module_name)
        if parsed_module is None:
            return None

        class_path = tuple(entry.class_name.split(".")) if entry.class_name else ()
        function_map = parsed_module.functions.get(class_path)
        if function_map is None:
            return None

        signatures = function_map.get(entry.function_name)
        if not signatures:
            return None

        return ManualStubReference(path=parsed_module.path, signatures=signatures)

    def _load_module(self, module_name: str) -> ParsedStubModule | None:
        """加载并缓存单个模块的 `.pyi` 解析结果。"""
        if module_name not in self._cache:
            stub_path = _locate_stub_path(self._root, module_name)
            if stub_path is None:
                self._cache[module_name] = None
            else:
                self._cache[module_name] = _parse_stub_module(stub_path)
        return self._cache[module_name]


def _locate_stub_path(root: Path, module_name: str) -> Path | None:
    """根据模块名定位唯一允许的 `.pyi` 路径。"""
    module_parts = module_name.split(".")
    module_file = root.joinpath(*module_parts).with_suffix(".pyi")
    package_file = root.joinpath(*module_parts, "__init__.pyi")

    file_exists = module_file.exists()
    package_exists = package_file.exists()

    if file_exists and package_exists:
        raise RuntimeError(
            f"模块 {module_name!r} 在人工 stub 根目录中同时匹配 {module_file} 与 {package_file}"
        )
    if file_exists:
        return module_file
    if package_exists:
        return package_file
    return None


def _parse_stub_module(path: Path) -> ParsedStubModule:
    """解析单个 `.pyi` 文件并建立函数索引。"""
    source = path.read_text(encoding="utf-8")
    module_node = ast.parse(source, filename=str(path))
    parsed = ParsedStubModule(path=path)
    _index_stub_body(parsed.functions, module_node.body, class_path=())
    return parsed


def _index_stub_body(
    index: dict[ClassPath, dict[str, list[ManualStubSignature]]],
    body: Iterable[ast.stmt],
    *,
    class_path: ClassPath,
) -> None:
    """递归收集模块或类体中的函数定义。"""
    for node in body:
        if isinstance(node, ast.ClassDef):
            _index_stub_body(index, node.body, class_path=(*class_path, node.name))
            continue

        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        signature = ManualStubSignature(
            text=_render_stub_signature(node),
            line=node.lineno,
        )
        function_map = index.setdefault(class_path, {})
        function_map.setdefault(node.name, []).append(signature)


def _render_stub_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """将 `.pyi` 中的函数定义还原为签名文本。"""
    decorator_lines = [f"@{ast.unparse(decorator)}" for decorator in node.decorator_list]
    def_keyword = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    signature_line = f"{def_keyword} {node.name}({ast.unparse(node.args)})"
    if node.returns is not None:
        signature_line += f" -> {ast.unparse(node.returns)}"
    signature_line += ": ..."
    return "\n".join([*decorator_lines, signature_line])


def _load_generated_entries(path: Path) -> list[GeneratedSignatureEntry]:
    """读取并校验生成 TOML 中的签名记录。"""
    with path.open("rb") as file:
        payload = tomllib.load(file)

    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        raise RuntimeError(f"TOML 文件 {path} 缺少 entries 列表。")

    result: list[GeneratedSignatureEntry] = []
    for index, raw_entry in enumerate(raw_entries, start=1):
        if not isinstance(raw_entry, dict):
            raise RuntimeError(f"TOML 第 {index} 条记录不是对象。")

        missing_keys = [
            key
            for key in ("module_name", "function_name", "signature")
            if key not in raw_entry
        ]
        if missing_keys:
            missing = ", ".join(missing_keys)
            raise RuntimeError(f"TOML 第 {index} 条记录缺少关键字段: {missing}")

        module_name = raw_entry["module_name"]
        class_name = raw_entry.get("class_name")
        function_name = raw_entry["function_name"]
        generated_signature = raw_entry["signature"]
        comment = raw_entry.get("comment")

        if not isinstance(module_name, str) or not module_name.strip():
            raise RuntimeError(f"TOML 第 {index} 条记录的 module_name 非法。")
        if class_name is not None and (not isinstance(class_name, str) or not class_name.strip()):
            raise RuntimeError(f"TOML 第 {index} 条记录的 class_name 非法。")
        if not isinstance(function_name, str) or not function_name.strip():
            raise RuntimeError(f"TOML 第 {index} 条记录的 function_name 非法。")
        if not isinstance(generated_signature, str) or not generated_signature.strip():
            raise RuntimeError(f"TOML 第 {index} 条记录的 signature 非法。")
        if comment is not None and not isinstance(comment, str):
            raise RuntimeError(f"TOML 第 {index} 条记录的 comment 非法。")

        result.append(
            GeneratedSignatureEntry(
                module_name=module_name,
                class_name=class_name,
                function_name=function_name,
                generated_signature=generated_signature,
                comment=_normalize_optional_text(comment),
            )
        )

    return result


def _normalize_optional_text(value: str | None) -> str | None:
    """将空白文本归一化为 None。"""
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized


def _prepare_evaluations(
    entries: list[GeneratedSignatureEntry],
    manual_stub_root: Path | None,
) -> tuple[list[PreparedEvaluation], list[EvaluationRow]]:
    """根据优先级为每条记录准备参考材料。"""
    repository = None if manual_stub_root is None else ManualStubRepository(manual_stub_root)
    pending: list[PreparedEvaluation] = []
    immediate_rows: list[EvaluationRow] = []

    for entry in entries:
        manual_reference = None if repository is None else repository.get_reference(entry)
        if manual_reference is not None:
            pending.append(
                PreparedEvaluation(
                    entry=entry,
                    reference_kind="manual_stub",
                    reference_payload=_render_manual_stub_payload(manual_reference),
                    reference_path=str(manual_reference.path),
                    reference_line=str(manual_reference.signatures[0].line),
                )
            )
            continue

        if entry.comment is not None:
            comment_path, comment_line = _extract_comment_location(entry.comment)
            pending.append(
                PreparedEvaluation(
                    entry=entry,
                    reference_kind="comment",
                    reference_payload=entry.comment,
                    reference_path=comment_path,
                    reference_line=comment_line,
                )
            )
            continue

        immediate_rows.append(
            _build_error_row(
                entry=entry,
                reference_kind="comment",
                reference_path="",
                reference_line="",
                reason="缺少人工 stub 参考，且 comment 证据为空。",
            )
        )

    return pending, immediate_rows


def _render_manual_stub_payload(reference: ManualStubReference) -> str:
    """将人工 stub 参考渲染为紧凑 JSON 文本，便于直接放入提示词。"""
    payload = [
        {
            "signature": signature.text,
            "line": signature.line,
        }
        for signature in reference.signatures
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _extract_comment_location(comment: str) -> tuple[str, str]:
    """从 comment 首部尝试提取文件路径与行号。"""
    first_line = comment.splitlines()[0].strip() if comment.splitlines() else ""
    if not first_line:
        return "", ""

    source_location_match = re.match(
        r"^<SourceLocation file '([^']+)', line (\d+), column \d+>$",
        first_line,
    )
    if source_location_match is not None:
        return source_location_match.group(1), source_location_match.group(2)

    simple_location_match = re.match(r"^(.+?):(\d+):\d+$", first_line)
    if simple_location_match is not None:
        return simple_location_match.group(1), simple_location_match.group(2)

    return "", ""


def _build_error_row(
    *,
    entry: GeneratedSignatureEntry,
    reference_kind: str,
    reference_path: str,
    reference_line: str,
    reason: str,
) -> EvaluationRow:
    """构造一条流程级错误结果。"""
    return EvaluationRow(
        module_name=entry.module_name,
        class_name=entry.class_name or "",
        function_name=entry.function_name,
        generated_signature=entry.generated_signature,
        status="error",
        llm_verdict="",
        reason_code="",
        reason=reason,
        reference_kind=reference_kind,
        reference_path=reference_path,
        reference_line=reference_line,
    )


def _build_messages(prepared: PreparedEvaluation) -> list[dict[str, str]]:
    """为单条签名评估组装 DeepSeek 对话消息。"""
    entry = prepared.entry
    reference_instructions = (
        "参考材料是人工维护 stub 提取出的可信参考签名列表。只要生成签名与其中任意一条语义一致，就应该判定为 pass。"
        if prepared.reference_kind == "manual_stub"
        else "参考材料是原始证据 comment，请直接根据原始证据判断生成签名是否成立。"
    )

    system_prompt = (
        "你是 Python 签名评估器。"
        "请判断一条生成签名是否与参考材料语义一致。"
        "输出必须是严格 JSON 对象，不要输出 Markdown、代码块或额外文字。"
        'JSON 结构固定为 {"verdict": "...", "reason_code": "...", "reason": "..."}。'
        'verdict 只允许是 "pass" 或 "fail"。'
        '如果 verdict 是 "pass"，reason_code 和 reason 必须都是空字符串。'
        '如果 verdict 是 "fail"，reason_code 只允许是 "parameter_mismatch"、"return_mismatch"、"overload_mismatch" 之一，'
        "reason 必须是一句简短中文说明。"
        "按语义一致判断，不要求字面完全一致。"
    )

    user_prompt = (
        f"模块: {entry.module_name}\n"
        f"类: {entry.class_name or '<module>'}\n"
        f"函数: {entry.function_name}\n"
        f"生成签名:\n{entry.generated_signature}\n\n"
        f"参考类型: {prepared.reference_kind}\n"
        f"参考说明: {reference_instructions}\n"
        f"参考材料:\n{prepared.reference_payload}\n"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def _write_pending_rows(
    output_path: Path,
    prepared_entries: list[PreparedEvaluation],
    *,
    concurrency: int,
) -> None:
    """并发评估待处理签名，并在任务完成后立刻追加写入 CSV。"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("环境变量 OPENAI_API_KEY 未设置。")

    semaphore = asyncio.Semaphore(concurrency)
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        max_retries=2,
    )
    try:
        tasks = [
            asyncio.create_task(_evaluate_one(client, semaphore, prepared))
            for prepared in prepared_entries
        ]
        with output_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_HEADERS)
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("{task.completed:.0f}/{task.total:.0f}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=False,
            ) as progress:
                task_id = progress.add_task("评估签名", total=len(prepared_entries))
                for task in asyncio.as_completed(tasks):
                    row = await task
                    writer.writerow(row.to_csv_row())
                    file.flush()
                    progress.advance(task_id, 1)
    finally:
        await client.close()


def _write_header_and_rows(output_path: Path, rows: Iterable[EvaluationRow]) -> None:
    """写入 CSV 表头，并先落盘所有无需 LLM 的结果。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_HEADERS)
        writer.writeheader()
        file.flush()
        for row in rows:
            writer.writerow(row.to_csv_row())
            file.flush()


async def _evaluate_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    prepared: PreparedEvaluation,
) -> EvaluationRow:
    """执行单条签名的 LLM 评估。"""
    async with semaphore:
        try:
            response = await _request_completion(client, prepared)
            parsed = _parse_llm_response(response)
            return _build_llm_row(prepared, parsed)
        except Exception as ex:
            return _build_error_row(
                entry=prepared.entry,
                reference_kind=prepared.reference_kind,
                reference_path=prepared.reference_path,
                reference_line=prepared.reference_line,
                reason=_format_exception_message(ex),
            )


async def _request_completion(
    client: AsyncOpenAI,
    prepared: PreparedEvaluation,
) -> str:
    """带简单重试地请求 DeepSeek 评估结果。"""
    last_error: Exception | None = None
    delay_seconds = 1.0
    messages = _build_messages(prepared)

    for attempt in range(1, DEFAULT_MAX_ATTEMPTS + 1):
        try:
            completion = await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                reasoning_effort=REASONING_EFFORT,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "enabled"}},
            )
            content = completion.choices[0].message.content
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("模型返回的 content 为空。")
            return content
        except Exception as ex:
            last_error = ex
            if attempt >= DEFAULT_MAX_ATTEMPTS:
                break
            await asyncio.sleep(delay_seconds)
            delay_seconds *= 2

    assert last_error is not None
    raise last_error


def _parse_llm_response(response_text: str) -> dict[str, str]:
    """解析并校验模型返回的严格 JSON。"""
    normalized_text = response_text.strip()
    if normalized_text.startswith("```"):
        normalized_text = re.sub(r"^```(?:json)?\s*", "", normalized_text)
        normalized_text = re.sub(r"\s*```$", "", normalized_text)
        normalized_text = normalized_text.strip()

    payload = json.loads(normalized_text)
    if not isinstance(payload, dict):
        raise RuntimeError("模型返回的 JSON 顶层不是对象。")

    verdict = payload.get("verdict")
    reason_code = payload.get("reason_code", "")
    reason = payload.get("reason", "")

    if verdict not in {"pass", "fail"}:
        raise RuntimeError(f"模型返回了非法 verdict: {verdict!r}")
    if not isinstance(reason_code, str):
        raise RuntimeError("模型返回的 reason_code 不是字符串。")
    if not isinstance(reason, str):
        raise RuntimeError("模型返回的 reason 不是字符串。")

    if verdict == "pass":
        if reason_code or reason:
            raise RuntimeError("verdict=pass 时 reason_code 和 reason 必须为空字符串。")
        return {
            "verdict": "pass",
            "reason_code": "",
            "reason": "",
        }

    if reason_code not in VALID_REASON_CODES:
        raise RuntimeError(f"模型返回了非法 reason_code: {reason_code!r}")
    if not reason.strip():
        raise RuntimeError("verdict=fail 时 reason 不能为空。")

    return {
        "verdict": "fail",
        "reason_code": reason_code,
        "reason": reason.strip(),
    }


def _build_llm_row(
    prepared: PreparedEvaluation,
    parsed_response: dict[str, str],
) -> EvaluationRow:
    """将 LLM 结果映射为最终 CSV 行。"""
    verdict = parsed_response["verdict"]
    return EvaluationRow(
        module_name=prepared.entry.module_name,
        class_name=prepared.entry.class_name or "",
        function_name=prepared.entry.function_name,
        generated_signature=prepared.entry.generated_signature,
        status=verdict,
        llm_verdict=verdict,
        reason_code=parsed_response["reason_code"],
        reason=parsed_response["reason"],
        reference_kind=prepared.reference_kind,
        reference_path=prepared.reference_path,
        reference_line=prepared.reference_line,
    )


def _format_exception_message(ex: Exception) -> str:
    """将异常格式化为适合写入 CSV 的短文本。"""
    message = str(ex).strip()
    if message:
        return f"{type(ex).__name__}: {message}"
    return type(ex).__name__


def run(
    generated_toml: Path,
    output_csv: Path,
    *,
    manual_stub_root: Path | None,
    concurrency: int,
) -> int:
    """执行完整的签名评估批处理流程。"""
    if output_csv.exists():
        raise RuntimeError(f"输出 CSV 已存在: {output_csv}")
    if manual_stub_root is not None and not manual_stub_root.is_dir():
        raise RuntimeError(f"人工 stub 根目录不存在或不是目录: {manual_stub_root}")

    entries = _load_generated_entries(generated_toml)
    typer.echo(f"读取 TOML: {generated_toml}")
    typer.echo(f"- 生成签名条数: {len(entries)}")
    pending, immediate_rows = _prepare_evaluations(entries, manual_stub_root)
    typer.echo(f"- 直接记为 error 的条数: {len(immediate_rows)}")
    typer.echo(f"- 进入 LLM 评估的条数: {len(pending)}")
    if manual_stub_root is not None:
        typer.echo(f"- 人工 stub 根目录: {manual_stub_root}")
    typer.echo(f"- 并发数: {concurrency}")
    typer.echo(f"- 输出 CSV: {output_csv}")
    if pending and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("环境变量 OPENAI_API_KEY 未设置。")
    _write_header_and_rows(output_csv, immediate_rows)

    if pending:
        typer.echo("开始调用模型评估...")
        asyncio.run(
            _write_pending_rows(
                output_csv,
                pending,
                concurrency=concurrency,
            )
        )
    else:
        typer.echo("没有需要进入 LLM 的记录。")

    typer.echo("评估完成。")

    return 0


@app.command(help="批量评估生成 TOML 中的函数签名，并输出 CSV 报告。")
def command(
    generated_toml: Path = typer.Argument(
        ...,
        metavar="GENERATED_TOML",
        help="由 `pcstubgen gen --toml` 生成的 TOML 文件。",
    ),
    output_csv: Path = typer.Argument(
        ...,
        metavar="OUTPUT_CSV",
        help="评估结果 CSV 输出路径。",
    ),
    manual_stub_root: Path | None = typer.Option(
        None,
        "--manual-stub-root",
        help="可选的人工 stub 根目录。",
    ),
    concurrency: int = typer.Option(
        DEFAULT_CONCURRENCY,
        "--concurrency",
        min=1,
        help="并发请求数。",
    ),
) -> None:
    exit_code = run(
        generated_toml=generated_toml,
        output_csv=output_csv,
        manual_stub_root=manual_stub_root,
        concurrency=concurrency,
    )
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    app()
