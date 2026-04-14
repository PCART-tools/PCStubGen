from __future__ import annotations

from . import cpython_rules, numpy_rules


PY_ARG_PARSE_TYPE_OBJECT_NAME_TO_TYPE = {
    **cpython_rules.PY_ARG_PARSE_TYPE_OBJECT_NAME_TO_TYPE,
    **numpy_rules.PY_ARG_PARSE_TYPE_OBJECT_NAME_TO_TYPE,
}

OBJECT_NAME_TO_TYPE = {
    **cpython_rules.OBJECT_NAME_TO_TYPE,
    **numpy_rules.OBJECT_NAME_TO_TYPE,
}

FUNCTION_NAME_TO_TYPE = {
    **cpython_rules.FUNCTION_NAME_TO_TYPE,
    **numpy_rules.FUNCTION_NAME_TO_TYPE,
}
