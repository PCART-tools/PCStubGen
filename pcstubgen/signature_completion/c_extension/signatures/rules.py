from __future__ import annotations

from . import cpython_rules, numpy_rules, pillow_rules


PY_ARG_PARSE_TYPE_OBJECT_NAME_TO_TYPE = {
    **cpython_rules.PY_ARG_PARSE_TYPE_OBJECT_NAME_TO_TYPE,
    **numpy_rules.PY_ARG_PARSE_TYPE_OBJECT_NAME_TO_TYPE,
    **pillow_rules.PY_ARG_PARSE_TYPE_OBJECT_NAME_TO_TYPE,
}

PY_ARG_PARSE_CONVERTER_NAME_TO_TYPE = {
    **cpython_rules.PY_ARG_PARSE_CONVERTER_NAME_TO_TYPE,
    **numpy_rules.PY_ARG_PARSE_CONVERTER_NAME_TO_TYPE,
    **pillow_rules.PY_ARG_PARSE_CONVERTER_NAME_TO_TYPE,
}

OBJECT_NAME_TO_TYPE = {
    **cpython_rules.OBJECT_NAME_TO_TYPE,
    **numpy_rules.OBJECT_NAME_TO_TYPE,
    **pillow_rules.OBJECT_NAME_TO_TYPE,
}

FUNCTION_NAME_TO_TYPE = {
    **cpython_rules.FUNCTION_NAME_TO_TYPE,
    **numpy_rules.FUNCTION_NAME_TO_TYPE,
    **pillow_rules.FUNCTION_NAME_TO_TYPE,
}
