from __future__ import annotations

import dataclasses
from pathlib import Path


@dataclasses.dataclass
class StubGenerationOptions:
    # 签名补全管道控制
    enable_docstring_signature_parser: bool = True

    source_root: Path | None = None
    c_std: str = "c11"
    cpp_std: str = "c++17"
    include: list[str] = dataclasses.field(default_factory=list)
    include_directory: list[Path] = dataclasses.field(default_factory=list)

    # 输出选项
    include_docstrings: bool = True
    include_module_type_comment: bool = False
