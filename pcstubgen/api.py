from __future__ import annotations

from pathlib import Path

from .module_collector import ModuleCollector
from .signature_completion import SignatureCompleter
from .stub_output import StubRenderer, StubWriter, TomlWriter

__all__ = ["gen_stubs"]


def gen_stubs(
    module_name: str,
    output: Path,
    compilation_database: Path,
    *,
    include_docstrings: bool = False,
    writer: StubWriter | TomlWriter | None = None,
) -> None:
    """
    为模块生成 stub 并写入输出目录。
    """
    _writer = writer if writer is not None else StubWriter()

    module_node = ModuleCollector().run(module_name)

    SignatureCompleter(compilation_database).run(module_node)

    output.mkdir(parents=True, exist_ok=True)
    renderer = StubRenderer(include_docstrings=include_docstrings)
    _writer.write(module_node, renderer, to=output)
