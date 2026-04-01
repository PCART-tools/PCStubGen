from __future__ import annotations

from ....types import RawType, Type


OBJECT_NAME_TO_TYPE: dict[str, Type] = {
    "_Py_NoneStruct": RawType("None"),
    "_Py_TrueStruct": RawType("bool"),
    "_Py_FalseStruct": RawType("bool"),
}
