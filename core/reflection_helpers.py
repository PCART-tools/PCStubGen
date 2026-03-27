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