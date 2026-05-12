"""
工具：使用 pyperf 对单函数签名补全进行逐函数基准测试。

示例:
    uv run python tools/rq4_benchmark_signature_completion.py ujson --compilation-database ./build/compile_commands.json --output ujson_pyperf.json
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import pkgutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyperf

from pcstubgen.models import QualifiedName
from pcstubgen.signature_completion import SignatureCompleter
from pcstubgen.signature_completion.completion_models import (
    SignatureCompletionContext,
    UnsupportedSignatureCompletion,
)


@dataclass(frozen=True)
class BenchmarkTarget:
    """单个待基准测试的 callable。"""

    function_id: str
    context: SignatureCompletionContext


class CallableContextCollector:
    """收集 `match()` 能识别且不明确 unsupported 的 callable 上下文。"""

    def __init__(self, signature_completer: SignatureCompleter) -> None:
        """保存用于识别 callable 的签名补全器。"""
        self._signature_completer = signature_completer

    def run(self, module_name: str) -> list[BenchmarkTarget]:
        """导入目标模块并递归收集 benchmark target。"""
        module = importlib.import_module(module_name)
        return self._collect_module(QualifiedName.from_str(module_name), module)

    def _collect_module(
        self,
        path: QualifiedName,
        module: types.ModuleType,
    ) -> list[BenchmarkTarget]:
        """收集单个模块及其直接子模块。"""
        targets: list[BenchmarkTarget] = []
        for name, member in inspect.getmembers(module):
            member_path = path.concat(name)
            if self._is_imported_member(member_path, member, module):
                continue

            if self._signature_completer.match(member):
                context = SignatureCompletionContext(
                    module_name=member_path.parent,
                    func_name=member_path.name,
                    member=member,
                )
                self._append_if_supported(targets, context)
            elif inspect.isclass(member):
                targets.extend(self._collect_class(member_path, member))

        if self._is_package(module):
            for submodule_name in self._iter_submodule_names(module):
                try:
                    sub_module = importlib.import_module(submodule_name)
                except (KeyboardInterrupt, GeneratorExit):
                    raise
                except BaseException:
                    continue
                targets.extend(
                    self._collect_module(
                        QualifiedName.from_str(submodule_name),
                        sub_module,
                    )
                )

        return targets

    def _collect_class(
        self,
        path: QualifiedName,
        class_: type,
    ) -> list[BenchmarkTarget]:
        """收集类方法和嵌套类中的 benchmark target。"""
        targets: list[BenchmarkTarget] = []
        for name, member in class_.__dict__.items():
            member_path = path.concat(name)
            if self._signature_completer.match(member, class_):
                context = SignatureCompletionContext(
                    module_name=member_path.parent,
                    func_name=member_path.name,
                    member=member,
                    owner_class=class_,
                )
                self._append_if_supported(targets, context)
            elif inspect.isclass(member) and member.__qualname__.startswith(
                class_.__qualname__ + "."
            ):
                targets.extend(self._collect_class(member_path, member))
        return targets

    def _append_if_supported(
        self,
        targets: list[BenchmarkTarget],
        context: SignatureCompletionContext,
    ) -> None:
        """排除明确抛 UnsupportedSignatureCompletion 的对象。"""
        try:
            self._signature_completer.complete(context)
        except UnsupportedSignatureCompletion:
            return
        targets.append(
            BenchmarkTarget(
                function_id=_build_function_id(context),
                context=context,
            )
        )

    @staticmethod
    def _is_package(module: types.ModuleType) -> bool:
        """判断模块是否为包或命名空间包。"""
        spec = module.__spec__
        return spec is not None and spec.submodule_search_locations is not None

    @staticmethod
    def _iter_submodule_names(module: types.ModuleType) -> list[str]:
        """按 import 拓扑枚举包的直接子模块全名。"""
        spec = module.__spec__
        if spec is None or spec.submodule_search_locations is None:
            return []
        return sorted(
            module_info.name
            for module_info in pkgutil.iter_modules(
                spec.submodule_search_locations,
                prefix=f"{module.__name__}.",
            )
        )

    @staticmethod
    def _get_module_name(obj: Any) -> str | None:
        """读取对象的 `__module__`。"""
        module_name = getattr(obj, "__module__", None)
        if isinstance(module_name, str):
            return module_name
        return None

    def _is_imported_member(
        self,
        path: QualifiedName,
        member: Any,
        module: types.ModuleType,
    ) -> bool:
        """判断成员是否来自外部模块导入。"""
        if path.name == "annotations":
            return True
        if inspect.isclass(member) or inspect.isroutine(member):
            return self._get_module_name(member) != module.__name__
        return False


def _build_parser() -> argparse.ArgumentParser:
    """构造同时支持实验参数与 pyperf 参数的解析器。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("module_name", help="模块名。")
    parser.add_argument(
        "--compilation-database",
        required=True,
        type=Path,
        help="compile_commands.json 文件路径。",
    )
    parser.add_argument(
        "--max-functions",
        type=int,
        default=None,
        help="仅用于小规模手动验证的函数数量上限；正式实验不传。",
    )
    return parser


def _build_program_args(args: argparse.Namespace) -> tuple[str, ...]:
    """构造 pyperf worker 重新执行当前脚本所需的业务参数。"""
    program_args = [
        sys.argv[0],
        args.module_name,
        "--compilation-database",
        str(args.compilation_database),
    ]
    if args.max_functions is not None:
        program_args.extend(["--max-functions", str(args.max_functions)])
    return tuple(program_args)


def _build_function_id(context: SignatureCompletionContext) -> str:
    """构造稳定的函数级 benchmark 名称。"""
    if context.owner_class is None:
        return f"{context.module_name}:{context.func_name}"
    owner_name = context.owner_class.__qualname__
    return f"{context.module_name}:{owner_name}.{context.func_name}"


def _complete_context(
    completer: SignatureCompleter,
    context: SignatureCompletionContext,
) -> None:
    """执行一次单函数签名补全。"""
    completer.complete(context)


def _collect_targets(
    module_name: str,
    compilation_database: Path,
    max_functions: int | None,
) -> list[BenchmarkTarget]:
    """收集 pyperf benchmark target。"""
    prefilter_completer = SignatureCompleter(compilation_database)
    collector = CallableContextCollector(prefilter_completer)
    targets = collector.run(module_name)
    if max_functions is not None:
        targets = targets[:max_functions]
    return targets


def main() -> None:
    """脚本入口。"""
    parser = _build_parser()
    runner = pyperf.Runner(
        program_args=_build_program_args(parser.parse_known_args()[0]),
        _argparser=parser,
    )
    args = runner.parse_args()
    targets = _collect_targets(
        args.module_name,
        args.compilation_database,
        args.max_functions,
    )
    if not args.worker:
        print(f"收集 benchmark target 数量: {len(targets)}")

    benchmark_completer = SignatureCompleter(args.compilation_database)
    for target in targets:
        runner.bench_func(
            target.function_id,
            _complete_context,
            benchmark_completer,
            target.context,
        )


if __name__ == "__main__":
    main()
