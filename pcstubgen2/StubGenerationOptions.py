from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Literal


@dataclasses.dataclass
class StubGenerationOptions:
    # 错误过滤
    ignore_invalid_expressions: re.Pattern | None = None
    ignore_all_errors: bool = False

    # 由 (正则, 前缀) 组成的列表，用于 docstring 中 pybind11 枚举值重写
    enum_class_locations: list[tuple[re.Pattern, str]] = dataclasses.field(default_factory=list)

    # 签名补全管道控制
    enable_docstring_signature_parser: bool = True
    signature_inference_scope: Literal["off", "c_modules", "all_modules"] = "c_modules"
    c_source_root: str | Path | None = None
    clang_parse_args: list[str] | None = None

    # 输出选项
    print_invalid_expressions_as_is: bool = False
    include_docstrings: bool = True
    include_module_type_comment: bool = False
    stub_extension: str = "pyi"
