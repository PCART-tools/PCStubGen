from __future__ import annotations

import dataclasses
import re


@dataclasses.dataclass
class StubGenerationOptions:
    # 错误过滤
    ignore_invalid_expressions: re.Pattern | None = None
    ignore_all_errors: bool = False

    # 由 (正则, 前缀) 组成的列表，用于 docstring 中 pybind11 枚举值重写
    enum_class_locations: list[tuple[re.Pattern, str]] = dataclasses.field(default_factory=list)

    # 输出选项
    print_invalid_expressions_as_is: bool = False
    include_docstrings: bool = True
    include_module_type_comment: bool = False
    stub_extension: str = "pyi"
