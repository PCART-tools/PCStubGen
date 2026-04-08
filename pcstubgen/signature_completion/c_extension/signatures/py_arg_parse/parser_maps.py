from __future__ import annotations

from .....types import RawType, Type


PY_TYPE_OBJECT_NAME_TO_TYPE: dict[str, Type] = {
    "PyList_Type": RawType("list"),
    "PyTuple_Type": RawType("tuple"),
    "PyDict_Type": RawType("dict"),
    "PyUnicode_Type": RawType("str"),
    "PyLong_Type": RawType("int"),
    "PyFloat_Type": RawType("float"),
    "PyBool_Type": RawType("bool"),
    "PyBytes_Type": RawType("bytes"),
    "PyByteArray_Type": RawType("bytearray"),
    "PySet_Type": RawType("set"),
    "PyFrozenSet_Type": RawType("frozenset"),
    "PyType_Type": RawType("type"),
    "PyBaseObject_Type": RawType("object"),
    "PyArray_Type": RawType("numpy.ndarray", imports=("numpy",)),
}
DEFAULT_IDENTIFIER_TO_VALUE: dict[str, str] = {
    "Py_None": "None",
    "Py_True": "True",
    "Py_False": "False",
}
