from __future__ import annotations

import types
from typing import Any


def get_doc(obj: Any) -> str | None:
    doc = getattr(obj, "__doc__", None)
    if isinstance(doc, str) and doc and not doc.isspace():
        return doc
    return None


def get_module_name(obj: Any) -> str | None:
    module_name = getattr(obj, "__module__", None)
    if isinstance(module_name, str):
        return module_name
    return None


def get_module_qualname(obj: Any) -> tuple[str | None, str | None]:
    module_name = get_module_name(obj)
    qual_name = getattr(obj, "__qualname__", None)
    if not isinstance(qual_name, str):
        qual_name = None
    return module_name, qual_name


def is_package(module: types.ModuleType) -> bool:
    return hasattr(module, "__path__")


def get_generic_alias_type() -> type | None:
    generic_alias = getattr(types, "GenericAlias", None)
    if isinstance(generic_alias, type):
        return generic_alias
    return None
