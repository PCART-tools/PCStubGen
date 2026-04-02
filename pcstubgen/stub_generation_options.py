from __future__ import annotations

import dataclasses
from pathlib import Path


@dataclasses.dataclass
class StubGenerationOptions:
    source: Path | None = None
    c_std: str = "c11"
    cpp_std: str = "c++17"
    include: list[str] = dataclasses.field(default_factory=list)
    include_directory: list[Path] = dataclasses.field(default_factory=list)

    # 输出选项
    include_docstrings: bool = False
    include_module_type_comment: bool = False
    include_c_inferred_source_comment: bool = False
