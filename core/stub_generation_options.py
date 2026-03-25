from __future__ import annotations

import dataclasses
from pathlib import Path


@dataclasses.dataclass
class StubGenerationOptions:
    # 签名补全管道控制
    enable_docstring_signature_parser: bool = True

    source_root: Path | None = None
    clang_c_std: str = "c11"
    clang_cpp_std: str = "c++17"
    clang_include: list[str] = dataclasses.field(default_factory=list)
    clang_include_directory: list[str] = dataclasses.field(default_factory=list)

    # 输出选项
    include_docstrings: bool = True
    include_module_type_comment: bool = False
