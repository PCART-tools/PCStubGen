from __future__ import annotations

import re
from typing import Final

# 将 Typing 构造替换为其内置等效项。
TYPING_BUILTIN_REPLACEMENTS: Final = {
    "typing.Text": "builtins.str",
    "typing.Tuple": "builtins.tuple",
    "typing.List": "builtins.list",
    "typing.Dict": "builtins.dict",
    "typing.Set": "builtins.set",
    "typing.FrozenSet": "builtins.frozenset",
    "typing.Type": "builtins.type",
}

IGNORED_DUNDERS: Final = {
    "__all__", "__author__", "__about__", "__copyright__", "__email__",
    "__license__", "__summary__", "__title__", "__uri__", "__str__",
    "__repr__", "__getstate__", "__setstate__", "__slots__", "__builtins__",
    "__cached__", "__file__", "__name__", "__package__", "__path__",
    "__spec__", "__loader__",
}

def is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")

def quote_docstring(docstr: str) -> str:
    """返回正确封装在单引号或双引号形式中的文档字符串。"""
    # 使用 repr 来获取关于正确引号的提示，并正确转义所有内容。
    # 创建多行字符串以获得更美观的输出。
    docstr_repr = "\n".join(re.split(r"(?<=[^\\])\\n", repr(docstr)))

    if docstr_repr.startswith("'"):
        # 在安全的情况下强制使用双引号。
        # 即当双引号不在字符串中，或者当它不以单引号结尾时。
        if '"' not in docstr_repr[1:-1] and docstr_repr[-2] != "'":
            return f'"""{docstr_repr[1:-1]}"""'
        return f"''{docstr_repr}''"
    else:
        return f'""{docstr_repr}""'

def get_parent_name(fullname: str) -> str:
    """返回父级名称，如果没有则返回空字符串。"""
    if "." not in fullname:
        return ""
    return fullname.rpartition(".")[0]

def get_short_name(fullname: str) -> str:
    """返回短名称（最后一个组件）。"""
    return fullname.rpartition(".")[-1]

def method_name_sort_key(name: str) -> tuple[int, str]:
    if name in ("__new__", "__init__"):
        return 0, name
    if name.startswith("__") and name.endswith("__"):
        return 2, name
    return 1, name

def is_private_name(
    name: str, _all_: list[str] | None = None, include_private: bool = False
) -> bool:
    """判断一个名称是否为私有。"""
    if "__mypy-" in name:
        return True
    if include_private:
        return False
    if name == "_":
        return False
    if not name.startswith("_"):
        return False
    if _all_ is not None and name in _all_:
        return False
    if name.startswith("__") and name.endswith("__"):
        return name in IGNORED_DUNDERS
    return True

def is_not_in_all(
    name: str, _all_: list[str] | None = None, is_top_level: bool = True
) -> bool:
    """判断一个名称是否不在 __all__ 中（仅针对顶级成员）。"""
    if not is_top_level:
        return False
    if _all_ is None:
        return False
    return name not in _all_
