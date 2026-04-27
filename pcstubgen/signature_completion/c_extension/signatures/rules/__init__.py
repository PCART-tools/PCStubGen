from __future__ import annotations

from collections.abc import Callable

from .....type_models import Type
from . import cpython_rules, numpy_rules, pillow_rules, psycopg2_rules, pytorch_rules
from .lookup import get_name_to_type


PY_ARG_PARSE_TYPE_OBJECT_NAME_TO_TYPE = {
    **cpython_rules.PY_ARG_PARSE_TYPE_OBJECT_NAME_TO_TYPE,
    **numpy_rules.PY_ARG_PARSE_TYPE_OBJECT_NAME_TO_TYPE,
    **pillow_rules.PY_ARG_PARSE_TYPE_OBJECT_NAME_TO_TYPE,
    **psycopg2_rules.PY_ARG_PARSE_TYPE_OBJECT_NAME_TO_TYPE,
}

PY_ARG_PARSE_CONVERTER_NAME_TO_TYPE = {
    **cpython_rules.PY_ARG_PARSE_CONVERTER_NAME_TO_TYPE,
    **numpy_rules.PY_ARG_PARSE_CONVERTER_NAME_TO_TYPE,
    **pillow_rules.PY_ARG_PARSE_CONVERTER_NAME_TO_TYPE,
    **psycopg2_rules.PY_ARG_PARSE_CONVERTER_NAME_TO_TYPE,
}

OBJECT_NAME_TO_TYPE = {
    **cpython_rules.OBJECT_NAME_TO_TYPE,
    **numpy_rules.OBJECT_NAME_TO_TYPE,
    **pillow_rules.OBJECT_NAME_TO_TYPE,
    **psycopg2_rules.OBJECT_NAME_TO_TYPE,
}

OBJECT_USE_FUNCTION_NAME_TO_TYPE = {
    **cpython_rules.OBJECT_USE_FUNCTION_NAME_TO_TYPE,
    **numpy_rules.OBJECT_USE_FUNCTION_NAME_TO_TYPE,
}

CALL_NAME_TO_TYPE: dict[str, Type | Callable[[Cursor], Type]] = {
    **cpython_rules.CALL_NAME_TO_TYPE,
    **numpy_rules.CALL_NAME_TO_TYPE,
    **pillow_rules.CALL_NAME_TO_TYPE,
    **psycopg2_rules.CALL_NAME_TO_TYPE,
    **pytorch_rules.CALL_NAME_TO_TYPE,
}
