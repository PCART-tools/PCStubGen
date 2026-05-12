"""
工具：批量评估新版函数层 TOML 中的推断签名是否合格。

示例:
    uv run python tools/rq3_eval_inferred_signatures.py out/pcstubgen/psycopg2.toml
    uv run python tools/rq3_eval_inferred_signatures.py out/pcstubgen/psycopg2.toml --manual-stub-root ./stubs
"""

from __future__ import annotations

import ast
import asyncio
import csv
import json
import os
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TextIO

import json_repair
import typer
from openai import AsyncOpenAI
from openai.types import ReasoningEffort
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.shared_params.response_format_json_object import ResponseFormatJSONObject
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
REASONING_EFFORT: ReasoningEffort = "high"
RESPONSE_FORMAT_JSON_OBJECT: ResponseFormatJSONObject = {"type": "json_object"}
EXTRA_BODY = {"thinking": {"type": "enabled"}}
DEFAULT_CONCURRENCY = 32
DEFAULT_MAX_ATTEMPTS = 4
EXIT_ERROR = 1

VALID_LLM_RESULT_CODES = {
    "qualified",
    "unqualified",
    "uncertain",
}
STATUS_OK = "ok"
STATUS_MISSING_REFERENCE = "missing_reference"
STATUS_LLM_ERROR = "llm_error"

EVALUATION_CSV_FIELDS = [
    "function_id",
    "module_name",
    "class_name",
    "function_name",
    "signature_index",
    "inferred_signature",
    "parameter_structure_code",
    "parameter_structure_reason",
    "parameter_type_code",
    "parameter_type_reason",
    "return_type_code",
    "return_type_reason",
    "status",
    "status_reason",
    "reasoning_content",
    "reference_kind",
    "reference",
]

app = typer.Typer(add_completion=False)

ClassPath = tuple[str, ...]


@dataclass(frozen=True)
class GeneratedFunctionEntry:
    """新版 TOML 中的函数级记录。"""

    function_id: str
    module_name: str
    class_name: str | None
    function_name: str
    provider: str
    parameter_inference_status: str
    return_inference_status: str
    failure_reason: str | None
    evidence: str | None


@dataclass(frozen=True)
class InferredSignatureEntry:
    """新版 TOML 中的单条推断签名记录。"""

    function: GeneratedFunctionEntry
    signature_index: int
    inferred_signature: str
    raw_signature: str | None

    @property
    def function_id(self) -> str:
        """返回所属函数标识。"""
        return self.function.function_id

    @property
    def module_name(self) -> str:
        """返回所属模块名。"""
        return self.function.module_name

    @property
    def class_name(self) -> str | None:
        """返回所属类名。"""
        return self.function.class_name

    @property
    def function_name(self) -> str:
        """返回函数名。"""
        return self.function.function_name


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

    entry: InferredSignatureEntry
    reference_kind: str
    reference: str


@dataclass(frozen=True)
class LLMCompletionResult:
    """一次 LLM 评估请求返回的正文与推理内容。"""

    content: str
    reasoning_content: str


@dataclass(frozen=True)
class EvaluationRow:
    """CSV 输出的一行评估结果。"""

    function_id: str
    module_name: str
    class_name: str
    function_name: str
    signature_index: int
    inferred_signature: str
    parameter_structure_code: str
    parameter_structure_reason: str
    parameter_type_code: str
    parameter_type_reason: str
    return_type_code: str
    return_type_reason: str
    status: str
    status_reason: str
    reasoning_content: str
    reference_kind: str
    reference: str

    def to_csv_dict(self) -> dict[str, str | int]:
        """将结果行转换为 CSV 可写入字典。"""
        return {
            "function_id": self.function_id,
            "module_name": self.module_name,
            "class_name": self.class_name,
            "function_name": self.function_name,
            "signature_index": self.signature_index,
            "inferred_signature": self.inferred_signature,
            "parameter_structure_code": self.parameter_structure_code,
            "parameter_structure_reason": self.parameter_structure_reason,
            "parameter_type_code": self.parameter_type_code,
            "parameter_type_reason": self.parameter_type_reason,
            "return_type_code": self.return_type_code,
            "return_type_reason": self.return_type_reason,
            "status": self.status,
            "status_reason": self.status_reason,
            "reasoning_content": self.reasoning_content,
            "reference_kind": self.reference_kind,
            "reference": self.reference,
        }


class ManualStubRepository:
    """按模块定位并解析人工维护的 `.pyi` 参考。"""

    def __init__(self, root: Path) -> None:
        """保存人工 stub 根目录。"""
        self._root = root
        self._cache: dict[str, ParsedStubModule | None] = {}

    def get_reference(self, entry: InferredSignatureEntry) -> ManualStubReference | None:
        """返回某条推断签名对应的人工 stub 参考。"""
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


def _load_inferred_signatures(path: Path) -> list[InferredSignatureEntry]:
    """读取并校验新版函数层 TOML 中的推断签名记录。"""
    with path.open("rb") as file:
        payload = tomllib.load(file)

    raw_functions = payload.get("functions")
    if not isinstance(raw_functions, list):
        raise RuntimeError(f"TOML 文件 {path} 缺少新版 functions 列表。")

    result: list[InferredSignatureEntry] = []
    for function_index, raw_function in enumerate(raw_functions, start=1):
        function = _parse_generated_function(function_index, raw_function)
        raw_signature_entries = raw_function.get("signatures")
        if not isinstance(raw_signature_entries, list):
            raise RuntimeError(f"TOML 第 {function_index} 个函数缺少 signatures 列表。")

        for signature_index, raw_signature_entry in enumerate(raw_signature_entries):
            if not isinstance(raw_signature_entry, dict):
                raise RuntimeError(
                    f"TOML 第 {function_index} 个函数第 {signature_index} 条签名不是对象。"
                )
            signature_text = _require_str(raw_signature_entry, "signature", f"第 {function_index} 个函数签名")
            raw_signature_text = _optional_str(
                raw_signature_entry.get("raw_signature"),
                f"第 {function_index} 个函数第 {signature_index} 条签名 raw_signature",
            )
            result.append(
                InferredSignatureEntry(
                    function=function,
                    signature_index=signature_index,
                    inferred_signature=signature_text,
                    raw_signature=raw_signature_text,
                )
            )

    return result


def _parse_generated_function(index: int, raw_function: object) -> GeneratedFunctionEntry:
    """解析单条函数层 TOML 记录。"""
    if not isinstance(raw_function, dict):
        raise RuntimeError(f"TOML 第 {index} 个函数不是对象。")

    module_name = _require_str(raw_function, "module_name", f"第 {index} 个函数")
    function_name = _require_str(raw_function, "function_name", f"第 {index} 个函数")
    class_name = raw_function.get("class_name")
    if class_name is not None and not isinstance(class_name, str):
        raise RuntimeError(f"TOML 第 {index} 个函数的 class_name 非法。")

    function_id = _require_str(raw_function, "function_id", f"第 {index} 个函数")

    provider = _require_str(raw_function, "provider", f"第 {index} 个函数")
    parameter_inference_status = _require_str(raw_function, "parameter_inference_status", f"第 {index} 个函数")
    return_inference_status = _require_str(raw_function, "return_inference_status", f"第 {index} 个函数")
    failure_reason = _optional_str(raw_function.get("failure_reason"), f"第 {index} 个函数 failure_reason")
    evidence = _render_function_evidence(raw_function)

    return GeneratedFunctionEntry(
        function_id=function_id,
        module_name=module_name,
        class_name=class_name,
        function_name=function_name,
        provider=provider,
        parameter_inference_status=parameter_inference_status,
        return_inference_status=return_inference_status,
        failure_reason=failure_reason,
        evidence=evidence,
    )


def _require_str(payload: dict[object, object], key: str, context: str) -> str:
    """读取必需字符串字段。"""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"TOML {context} 的 {key} 非法。")
    return value


def _optional_str(value: object, context: str) -> str | None:
    """读取可选字符串字段。"""
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeError(f"TOML {context} 非法。")
    normalized = value.strip()
    if not normalized:
        return None
    return normalized


def _render_function_evidence(raw_function: dict[object, object]) -> str | None:
    """把函数层来源证据渲染为紧凑 JSON 文本。"""
    evidence_payload: dict[str, object] = {}
    for key in ("provider", "source_location", "source_text"):
        value = raw_function.get(key)
        if isinstance(value, str) and value.strip():
            evidence_payload[key] = value

    if not evidence_payload:
        return None
    return json.dumps(evidence_payload, ensure_ascii=False, indent=2)


def _prepare_evaluations(
    entries: list[InferredSignatureEntry],
    manual_stub_root: Path | None,
) -> tuple[list[PreparedEvaluation], list[EvaluationRow]]:
    """根据优先级为每条记录准备参考材料。"""
    repository = None if manual_stub_root is None else ManualStubRepository(manual_stub_root)
    pending: list[PreparedEvaluation] = []
    immediate_rows: list[EvaluationRow] = []

    for entry in entries:
        if _has_double_inference_failure(entry):
            immediate_rows.append(_build_double_inference_failure_row(entry))
            continue

        manual_reference = None if repository is None else repository.get_reference(entry)
        if manual_reference is not None:
            pending.append(
                PreparedEvaluation(
                    entry=entry,
                    reference_kind="manual_stub",
                    reference=_render_manual_stub_reference(manual_reference),
                )
            )
            continue

        signature_evidence = _render_signature_evidence(entry)
        if signature_evidence is not None:
            pending.append(
                PreparedEvaluation(
                    entry=entry,
                    reference_kind="signature_evidence",
                    reference=signature_evidence,
                )
            )
            continue

        immediate_rows.append(
            _build_missing_reference_row(
                entry=entry,
            )
        )

    return pending, immediate_rows


def _render_signature_evidence(entry: InferredSignatureEntry) -> str | None:
    """合并函数层来源证据和签名层原始签名证据。"""
    evidence_payload: dict[str, object] = {}
    if entry.function.evidence is not None:
        evidence_payload["function_evidence"] = entry.function.evidence
    if entry.raw_signature is not None:
        evidence_payload["raw_signature"] = entry.raw_signature
    if not evidence_payload:
        return None
    return json.dumps(evidence_payload, ensure_ascii=False, indent=2)


def _build_missing_reference_row(entry: InferredSignatureEntry) -> EvaluationRow:
    """为缺少裁判参考的签名构造结果。"""
    row = _override_failed_dimensions(
        entry,
        {
            "parameter_structure_code": "uncertain",
            "parameter_structure_reason": "缺少可用于裁判的参考材料。",
            "parameter_type_code": "uncertain",
            "parameter_type_reason": "缺少可用于裁判的参考材料。",
            "return_type_code": "uncertain",
            "return_type_reason": "缺少可用于裁判的参考材料。",
        },
    )
    return _build_status_row(
        entry=entry,
        status=STATUS_MISSING_REFERENCE,
        status_reason="缺少人工 stub 参考，且函数层来源证据为空。",
        reasoning_content="",
        reference_kind="",
        reference="",
        parameter_structure_code=row["parameter_structure_code"],
        parameter_structure_reason=row["parameter_structure_reason"],
        parameter_type_code=row["parameter_type_code"],
        parameter_type_reason=row["parameter_type_reason"],
        return_type_code=row["return_type_code"],
        return_type_reason=row["return_type_reason"],
    )


def _has_double_inference_failure(entry: InferredSignatureEntry) -> bool:
    """判断签名是否因参数和返回值推断均失败而无需 LLM 裁判。"""
    return (
        entry.function.parameter_inference_status != "success"
        and entry.function.return_inference_status != "success"
    )


def _build_double_inference_failure_row(entry: InferredSignatureEntry) -> EvaluationRow:
    """为参数和返回值均推断失败的签名构造本地评估结果。"""
    return _build_status_row(
        entry=entry,
        status=STATUS_OK,
        status_reason=entry.function.failure_reason or "参数和返回值推断阶段均失败。",
        reasoning_content="",
        reference_kind="inference_status",
        reference=entry.function.evidence or "",
        parameter_structure_code="unqualified",
        parameter_structure_reason="参数推断阶段失败，参数结构计为不合格。",
        parameter_type_code="unqualified",
        parameter_type_reason="参数推断阶段失败，参数类型计为不合格。",
        return_type_code="unqualified",
        return_type_reason="返回值推断阶段失败，返回类型计为不合格。",
    )


def _render_manual_stub_reference(reference: ManualStubReference) -> str:
    """将人工 stub 参考渲染为紧凑 JSON 文本，便于直接放入提示词。"""
    payload = [
        {
            "signature": signature.text,
            "line": signature.line,
        }
        for signature in reference.signatures
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _build_status_row(
    *,
    entry: InferredSignatureEntry,
    status: str,
    status_reason: str,
    reasoning_content: str,
    reference_kind: str,
    reference: str,
    parameter_structure_code: str,
    parameter_structure_reason: str,
    parameter_type_code: str,
    parameter_type_reason: str,
    return_type_code: str,
    return_type_reason: str,
) -> EvaluationRow:
    """构造一条带状态信息的结果。"""
    return EvaluationRow(
        function_id=entry.function_id,
        module_name=entry.module_name,
        class_name=entry.class_name or "",
        function_name=entry.function_name,
        signature_index=entry.signature_index,
        inferred_signature=entry.inferred_signature,
        parameter_structure_code=parameter_structure_code,
        parameter_structure_reason=parameter_structure_reason,
        parameter_type_code=parameter_type_code,
        parameter_type_reason=parameter_type_reason,
        return_type_code=return_type_code,
        return_type_reason=return_type_reason,
        status=status,
        status_reason=status_reason,
        reasoning_content=reasoning_content,
        reference_kind=reference_kind,
        reference=reference,
    )


def _build_messages(prepared: PreparedEvaluation) -> list[ChatCompletionMessageParam]:
    """为单条签名评估组装 DeepSeek 对话消息。"""
    entry = prepared.entry
    if prepared.reference_kind == "manual_stub":
        reference_instructions = (
            "参考材料是人工维护 stub 提取出的可信参考签名列表，属于高优先级事实依据。"
            "如果它与你已知的 API 语义冲突，以人工 stub 为准。"
        )
    else:
        reference_instructions = (
            "参考材料是函数层签名推断来源，可能是 c_extension 源代码或 pybind11 的原始 overload 签名链。"
            "你应先使用这些参考材料判断；如果它们不足以确定函数真实语义，再结合你已知的 API 知识补充判断。"
        )

    system_prompt = (
        "你是 Python Stub 函数签名评估专家。"
        "请分别判断一条从扩展API签名推断来源推断出的 Python 层函数签名的参数结构、参数类型与返回类型，是否符合函数在Python层的真实语义。"
        "输出必须是严格 JSON 对象，不要输出 Markdown、代码块或额外文字。"
        'JSON 结构固定为 {"parameter_structure_code": "...", "parameter_structure_reason": "...", '
        '"parameter_type_code": "...", "parameter_type_reason": "...", '
        '"return_type_code": "...", "return_type_reason": "..."}。'
        '三个 *_code 只允许是 "qualified"、"unqualified" 或 "uncertain"。'
        "三个 *_reason 都必须是中文说明，分别解释对应维度的结论。"
        "当参考材料足以确定真实语义时，应优先基于参考材料判断。"
        "当参考材料不足以确定真实语义时，可以结合你已知的该模块、类、函数的 API 语义补充判断。"
        "如果结合参考材料与已有知识后，仍无法较有把握地确定真实语义，就输出 uncertain，不要强行判 qualified 或 unqualified。"
        "如果某一维度输出 uncertain，对应的 *_reason 中既要说明你为什么无法确定，也要说明你当前更倾向的真实语义是什么。"
        "要评估的推断签名为单条，参考材料可能包含多条重载语义。"
        "若可判定为 qualified，则三个维度都必须能对应到同一条真实语义或同一条参考签名。"
        "按语义一致判断，不要求字面完全一致。"
        "理解 PyArg_ParseTuple 或 PyArg_ParseTupleAndKeywords 的格式串时，参考以下 Python 官方文档原文："
        "(items) (tuple) [matching-items] "
        "对象必须是 Python 序列，它的长度是 items 中格式单元的数量。"
        "C 参数必须对应 items 中每一个独立的格式单元。"
        "序列中的格式单元可能有嵌套。"
    )

    user_prompt = (
        f"函数ID: {entry.function_id}\n"
        f"模块: {entry.module_name}\n"
        f"类: {entry.class_name or '<module>'}\n"
        f"函数: {entry.function_name}\n"
        f"签名序号: {entry.signature_index}\n"
        f"推断签名:\n{entry.inferred_signature}\n\n"
        f"参考类型: {prepared.reference_kind}\n"
        f"参考说明: {reference_instructions}\n"
        f"参考材料:\n{prepared.reference}\n"
    )

    system_message: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": system_prompt,
    }
    user_message: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": user_prompt,
    }
    return [
        system_message,
        user_message,
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
        with output_path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=EVALUATION_CSV_FIELDS)
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
                    _write_csv_row(writer, file, row)
                    progress.advance(task_id, 1)
    finally:
        await client.close()


def _write_initial_rows(output_path: Path, rows: Iterable[EvaluationRow]) -> None:
    """写入所有无需 LLM 的初始结果。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=EVALUATION_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            _write_csv_row(writer, file, row)


def _write_csv_row(writer: csv.DictWriter[str], file: TextIO, row: EvaluationRow) -> None:
    """向已打开文件写入一条 CSV 记录。"""
    writer.writerow(row.to_csv_dict())
    file.flush()


async def _evaluate_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    prepared: PreparedEvaluation,
) -> EvaluationRow:
    """执行单条签名的 LLM 评估。"""
    async with semaphore:
        delay_seconds = 1.0
        messages = _build_messages(prepared)
        last_exception: Exception | None = None

        for attempt in range(1, DEFAULT_MAX_ATTEMPTS + 1):
            try:
                response = await _request_completion(client, messages)
                parsed = _parse_llm_response(response.content)
                return _build_llm_row(prepared, parsed, response.reasoning_content)
            except Exception as ex:
                last_exception = ex
                typer.echo(
                    f"\n评估尝试失败, "
                    f"attempt={attempt}/{DEFAULT_MAX_ATTEMPTS}, "
                    f"function_id={prepared.entry.function_id}, "
                    f"signature_index={prepared.entry.signature_index}, "
                    f"ex={ex!r}"
                )
                if attempt >= DEFAULT_MAX_ATTEMPTS:
                    break
                await asyncio.sleep(delay_seconds)
                delay_seconds *= 2

        if last_exception is None:
            raise RuntimeError("单条评估未产生结果，且没有捕获到异常。")

        return _build_status_row(
            entry=prepared.entry,
            status=STATUS_LLM_ERROR,
            status_reason=f"{last_exception!r}",
            reasoning_content="",
            reference_kind=prepared.reference_kind,
            reference=prepared.reference,
            parameter_structure_code="uncertain",
            parameter_structure_reason="LLM 评估失败，无法判定参数结构。",
            parameter_type_code="uncertain",
            parameter_type_reason="LLM 评估失败，无法判定参数类型。",
            return_type_code="uncertain",
            return_type_reason="LLM 评估失败，无法判定返回类型。",
        )


async def _request_completion(
    client: AsyncOpenAI,
    messages: list[ChatCompletionMessageParam],
) -> LLMCompletionResult:
    """请求一次 DeepSeek 评估结果。"""
    completion = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        reasoning_effort=REASONING_EFFORT,
        response_format=RESPONSE_FORMAT_JSON_OBJECT,
        stream=False,
        extra_body=EXTRA_BODY,
    )
    message = completion.choices[0].message
    content = message.content
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("模型返回的 content 为空。")
    return LLMCompletionResult(
        content=content,
        reasoning_content=message.reasoning_content or "",
    )


def _parse_llm_response(response_text: str) -> dict[str, str]:
    """解析并校验模型返回的 JSON 响应。"""
    payload = json_repair.loads(response_text)
    if not isinstance(payload, dict):
        raise RuntimeError("模型返回的 JSON 顶层不是对象。")

    normalized_response = {}
    for field_name in (
        "parameter_structure_code",
        "parameter_structure_reason",
        "parameter_type_code",
        "parameter_type_reason",
        "return_type_code",
        "return_type_reason",
    ):
        field_value = payload.get(field_name)
        if not isinstance(field_value, str):
            raise RuntimeError(f"模型返回的 {field_name} 不是字符串。")
        normalized_response[field_name] = field_value.strip()

    for field_name in (
        "parameter_structure_code",
        "parameter_type_code",
        "return_type_code",
    ):
        if normalized_response[field_name] not in VALID_LLM_RESULT_CODES:
            raise RuntimeError(f"模型返回了非法 {field_name}: {payload.get(field_name)!r}")

    for field_name in (
        "parameter_structure_reason",
        "parameter_type_reason",
        "return_type_reason",
    ):
        if not normalized_response[field_name]:
            raise RuntimeError(f"模型返回的 {field_name} 不能为空。")

    return normalized_response


def _build_llm_row(
    prepared: PreparedEvaluation,
    parsed_response: dict[str, str],
    reasoning_content: str,
) -> EvaluationRow:
    """将 LLM 结果映射为最终 CSV 行。"""
    row = _override_failed_dimensions(prepared.entry, parsed_response)
    return EvaluationRow(
        function_id=prepared.entry.function_id,
        module_name=prepared.entry.module_name,
        class_name=prepared.entry.class_name or "",
        function_name=prepared.entry.function_name,
        signature_index=prepared.entry.signature_index,
        inferred_signature=prepared.entry.inferred_signature,
        parameter_structure_code=row["parameter_structure_code"],
        parameter_structure_reason=row["parameter_structure_reason"],
        parameter_type_code=row["parameter_type_code"],
        parameter_type_reason=row["parameter_type_reason"],
        return_type_code=row["return_type_code"],
        return_type_reason=row["return_type_reason"],
        status=STATUS_OK,
        status_reason="",
        reasoning_content=reasoning_content,
        reference_kind=prepared.reference_kind,
        reference=prepared.reference,
    )


def _override_failed_dimensions(
    entry: InferredSignatureEntry,
    parsed_response: dict[str, str],
) -> dict[str, str]:
    """按函数层推断状态覆盖已知失败的评估维度。"""
    row = dict(parsed_response)
    if entry.function.parameter_inference_status != "success":
        row["parameter_structure_code"] = "unqualified"
        row["parameter_structure_reason"] = (
            "参数推断阶段失败，参数结构计为不合格。"
        )
        row["parameter_type_code"] = "unqualified"
        row["parameter_type_reason"] = (
            "参数推断阶段失败，参数类型计为不合格。"
        )
    if entry.function.return_inference_status != "success":
        row["return_type_code"] = "unqualified"
        row["return_type_reason"] = (
            "返回值推断阶段失败，返回类型计为不合格。"
        )
    return row


def run(
    generated_toml: Path,
    *,
    manual_stub_root: Path | None,
    concurrency: int,
) -> int:
    """执行完整的签名评估批处理流程。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = generated_toml.parent / f"{generated_toml.stem}_eval_{timestamp}.csv"
    if output_csv.exists():
        raise RuntimeError(f"输出 CSV 已存在: {output_csv}")

    entries = _load_inferred_signatures(generated_toml)
    typer.echo(f"读取 TOML: {generated_toml}")
    typer.echo(f"- 推断签名条数: {len(entries)}")
    pending, immediate_rows = _prepare_evaluations(entries, manual_stub_root)
    typer.echo(f"- 直接产出本地结果的条数: {len(immediate_rows)}")
    typer.echo(f"- 进入 LLM 评估的条数: {len(pending)}")
    if manual_stub_root is not None:
        typer.echo(f"- 人工 stub 根目录: {manual_stub_root}")
    typer.echo(f"- 并发数: {concurrency}")
    typer.echo(f"- 输出 CSV: {output_csv}")
    if pending and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("环境变量 OPENAI_API_KEY 未设置。")
    _write_initial_rows(output_csv, immediate_rows)

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


@app.command(help="批量评估新版函数层 TOML 中的推断签名，并在源文件目录输出 CSV 报告。")
def command(
    generated_toml: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="由 `pcstubgen gen --toml` 生成的新版函数层 TOML 文件。",
    ),
    manual_stub_root: Path | None = typer.Option(
        None,
        "--manual-stub-root",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="人工 stub 根目录。",
    ),
    concurrency: int = typer.Option(
        DEFAULT_CONCURRENCY,
        "--concurrency",
        min=1,
        help="并发请求数。",
    ),
) -> None:
    """Typer 命令入口。"""
    exit_code = run(
        generated_toml=generated_toml,
        manual_stub_root=manual_stub_root,
        concurrency=concurrency,
    )
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    app()
