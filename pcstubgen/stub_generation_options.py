from __future__ import annotations

import dataclasses
from pathlib import Path


@dataclasses.dataclass
class StubGenerationOptions:
    compilation_database: Path | None = None

    # 输出选项
    include_docstrings: bool = False
    include_c_inferred_source_comment: bool = False
