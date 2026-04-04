from __future__ import annotations

from io import StringIO

from loguru import logger

from tests._c_extension_test_support import *


def _pyinit_lines(
    init_name: str,
    *,
    moduledef_name: str = "moduledef",
    create_func_name: str = "PyModule_Create",
) -> list[str]:
    return [
        f"PyObject* {create_func_name}(PyModuleDef* def);",
        f"PyObject* PyInit_{init_name}(void) {{",
        f"    return {create_func_name}(&{moduledef_name});",
        "}",
    ]


def _basic_c_extension_definitions(*extra_lines: str) -> str:
    return "\n".join(
        [
            "typedef struct _object PyObject;",
            "typedef struct PyMethodDef {",
            "    const char* ml_name;",
            "    void* ml_meth;",
            "    int ml_flags;",
            "    const char* ml_doc;",
            "} PyMethodDef;",
            "typedef struct PyModuleDef {",
            "    int m_base;",
            "    const char* m_name;",
            "    const char* m_doc;",
            "    int m_size;",
            "    PyMethodDef* m_methods;",
            "    void* m_slots;",
            "    void* m_traverse;",
            "    void* m_clear;",
            "    void* m_free;",
            "} PyModuleDef;",
            "#define PyModuleDef_HEAD_INIT 0",
            "#define METH_VARARGS 1",
            "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
            *extra_lines,
            "",
        ]
    )


def test_c_signature_extraction_engine_extract_modules_isolates_same_named_functions_per_module(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    first_source = tmp_path / "first.c"
    second_source = tmp_path / "second.c"
    for source, module_name, c_name in [
        (first_source, "first", "first_foo_impl"),
        (second_source, "second", "second_foo_impl"),
    ]:
        source.write_text(
            "\n".join(
                [
                    "typedef struct _object PyObject;",
                    "typedef struct PyMethodDef {",
                    "    const char* ml_name;",
                    "    void* ml_meth;",
                    "    int ml_flags;",
                    "    const char* ml_doc;",
                    "} PyMethodDef;",
                    "typedef struct PyModuleDef {",
                    "    int m_base;",
                    "    const char* m_name;",
                    "    const char* m_doc;",
                    "    int m_size;",
                    "    PyMethodDef* m_methods;",
                    "    void* m_slots;",
                    "    void* m_traverse;",
                    "    void* m_clear;",
                    "    void* m_free;",
                    "} PyModuleDef;",
                    "#define PyModuleDef_HEAD_INIT 0",
                    "#define METH_VARARGS 1",
                    "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                    f"static PyObject* {c_name}(PyObject* self, PyObject* args) {{",
                    "    int value = 0;",
                    "    if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                    "        return (PyObject*)0;",
                    "    }",
                    "    return (PyObject*)0;",
                    "}",
                    "static PyMethodDef Methods[] = {",
                    f"    {{\"foo\", {c_name}, METH_VARARGS, \"doc\"}},",
                    "    {0, 0, 0, 0}",
                    "};",
                    "static PyModuleDef moduledef = {",
                    "    PyModuleDef_HEAD_INIT,",
                    f"    \"{module_name}\",",
                    "    0,",
                    "    -1,",
                    "    Methods,",
                    "    0, 0, 0, 0",
                    "};",
                    *_pyinit_lines(module_name),
                ]
            ),
            encoding="utf-8",
        )

    engine = CSignatureExtractor(
        source=tmp_path,
        c_std="c11",
    )
    extracted = engine.extract_modules()

    assert set(extracted) == {"first", "second"}
    assert set(extracted["first"].functions) == {"foo"}
    assert set(extracted["second"].functions) == {"foo"}
    assert extracted["first"].functions["foo"].ml_flags == METH_VARARGS
    assert extracted["second"].functions["foo"].ml_flags == METH_VARARGS


def test_c_signature_extraction_engine_extract_modules_populates_inferred_return_only_signature(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "return_only_module.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct PyMethodDef {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "typedef struct PyModuleDef {",
                "    int m_base;",
                "    const char* m_name;",
                "    const char* m_doc;",
                "    int m_size;",
                "    PyMethodDef* m_methods;",
                "    void* m_slots;",
                "    void* m_traverse;",
                "    void* m_clear;",
                "    void* m_free;",
                "} PyModuleDef;",
                "#define PyModuleDef_HEAD_INIT 0",
                "#define METH_VARARGS 1",
                "PyObject* PyLong_FromLong(long value);",
                "static PyObject* foo_impl(PyObject* self, PyObject* args) {",
                "    return PyLong_FromLong(1);",
                "}",
                "static PyMethodDef Methods[] = {",
                "    {\"foo\", foo_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyModuleDef moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                "    \"return_only\",",
                "    0,",
                "    -1,",
                "    Methods,",
                "    0, 0, 0, 0",
                "};",
                *_pyinit_lines("return_only"),
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source=tmp_path,
        c_std="c11",
    )
    extracted = engine.extract_modules()

    signatures = extracted["return_only"].functions["foo"].signatures
    assert len(signatures) == 1
    assert signatures[0].arguments == []
    assert signatures[0].return_type is not None
    assert signatures[0].return_type.render() == "int"


def test_c_signature_extraction_engine_extract_modules_infers_parse_tuple_arguments(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "parse_tuple_args.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct _typeobject PyTypeObject;",
                "typedef struct PyMethodDef {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "typedef struct PyModuleDef {",
                "    int m_base;",
                "    const char* m_name;",
                "    const char* m_doc;",
                "    int m_size;",
                "    PyMethodDef* m_methods;",
                "    void* m_slots;",
                "    void* m_traverse;",
                "    void* m_clear;",
                "    void* m_free;",
                "} PyModuleDef;",
                "#define PyModuleDef_HEAD_INIT 0",
                "#define METH_VARARGS 1",
                "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                "PyTypeObject PyList_Type;",
                "static PyObject* foo_impl(PyObject* self, PyObject* args) {",
                "    int count = 0;",
                "    PyObject* items = (PyObject*)0;",
                "    if (!PyArg_ParseTuple(args, \"iO!\", &count, &PyList_Type, &items)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyMethodDef Methods[] = {",
                "    {\"foo\", foo_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyModuleDef moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                "    \"parse_tuple_args\",",
                "    0,",
                "    -1,",
                "    Methods,",
                "    0, 0, 0, 0",
                "};",
                *_pyinit_lines("parse_tuple_args"),
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source=tmp_path,
        c_std="c11",
    )
    extracted = engine.extract_modules()

    signatures = extracted["parse_tuple_args"].functions["foo"].signatures
    assert signatures == [
        ExtractedSignature(
            arguments=[
                _arg("count", "int"),
                _arg("items", "list"),
            ]
        )
    ]


def test_c_signature_extraction_engine_extract_modules_reads_object_type_from_extent_source_text(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "parse_tuple_extent_text.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct _typeobject PyTypeObject;",
                "typedef struct PyMethodDef {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "typedef struct PyModuleDef {",
                "    int m_base;",
                "    const char* m_name;",
                "    const char* m_doc;",
                "    int m_size;",
                "    PyMethodDef* m_methods;",
                "    void* m_slots;",
                "    void* m_traverse;",
                "    void* m_clear;",
                "    void* m_free;",
                "} PyModuleDef;",
                "#define PyModuleDef_HEAD_INIT 0",
                "#define METH_VARARGS 1",
                "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                "PyTypeObject PyArray_Type;",
                "static PyObject* foo_impl(PyObject* self, PyObject* args) {",
                "    /* 中文注释，验证 extent offset 按字节切片 */",
                "    PyObject* array = (PyObject*)0;",
                "    if (!PyArg_ParseTuple(args, \"O!\", &PyArray_Type, &array)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyMethodDef Methods[] = {",
                "    {\"foo\", foo_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyModuleDef moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                "    \"parse_tuple_extent_text\",",
                "    0,",
                "    -1,",
                "    Methods,",
                "    0, 0, 0, 0",
                "};",
                *_pyinit_lines("parse_tuple_extent_text"),
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source=tmp_path,
        c_std="c11",
    )
    extracted = engine.extract_modules()

    signatures = extracted["parse_tuple_extent_text"].functions["foo"].signatures
    assert signatures == [
        ExtractedSignature(
            arguments=[
                _arg("array", "numpy.ndarray", imports=("numpy",)),
            ]
        )
    ]


def test_c_signature_extraction_engine_extract_modules_emits_multiple_signatures_for_multiple_pyarg_calls(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "multiple_pyarg_signatures.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct PyMethodDef {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "typedef struct PyModuleDef {",
                "    int m_base;",
                "    const char* m_name;",
                "    const char* m_doc;",
                "    int m_size;",
                "    PyMethodDef* m_methods;",
                "    void* m_slots;",
                "    void* m_traverse;",
                "    void* m_clear;",
                "    void* m_free;",
                "} PyModuleDef;",
                "#define PyModuleDef_HEAD_INIT 0",
                "#define METH_VARARGS 1",
                "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                "static PyObject* foo_impl(PyObject* self, PyObject* args) {",
                "    int value = 0;",
                "    const char* label = 0;",
                "    if (0) {",
                "        if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                "            return (PyObject*)0;",
                "        }",
                "    }",
                "    if (!PyArg_ParseTuple(args, \"s\", &label)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyMethodDef Methods[] = {",
                "    {\"foo\", foo_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyModuleDef moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                "    \"multiple_pyarg_signatures\",",
                "    0,",
                "    -1,",
                "    Methods,",
                "    0, 0, 0, 0",
                "};",
                *_pyinit_lines("multiple_pyarg_signatures"),
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source=tmp_path,
        c_std="c11",
    )
    extracted = engine.extract_modules()

    signatures = extracted["multiple_pyarg_signatures"].functions["foo"].signatures
    assert signatures == [
        ExtractedSignature(arguments=[_arg("value", "int")]),
        ExtractedSignature(arguments=[_arg("label", "str")]),
    ]


def test_c_signature_extraction_engine_extract_modules_handles_multiple_moduledefs_in_one_file(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "multi_init_module.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct PyMethodDef {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "typedef struct PyModuleDef {",
                "    int m_base;",
                "    const char* m_name;",
                "    const char* m_doc;",
                "    int m_size;",
                "    PyMethodDef* m_methods;",
                "    void* m_slots;",
                "    void* m_traverse;",
                "    void* m_clear;",
                "    void* m_free;",
                "} PyModuleDef;",
                "#define PyModuleDef_HEAD_INIT 0",
                "#define METH_VARARGS 1",
                "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                "static PyObject* first_foo_impl(PyObject* self, PyObject* args) {",
                "    int value = 0;",
                "    if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyObject* second_foo_impl(PyObject* self, PyObject* args) {",
                "    int value = 0;",
                "    if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyMethodDef FirstMethods[] = {",
                "    {\"foo\", first_foo_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyMethodDef SecondMethods[] = {",
                "    {\"foo\", second_foo_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyModuleDef first_moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                "    \"first\",",
                "    0,",
                "    -1,",
                "    FirstMethods,",
                "    0, 0, 0, 0",
                "};",
                "static PyModuleDef second_moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                "    \"second\",",
                "    0,",
                "    -1,",
                "    SecondMethods,",
                "    0, 0, 0, 0",
                "};",
                *_pyinit_lines("first", moduledef_name="first_moduledef"),
                *_pyinit_lines("second", moduledef_name="second_moduledef"),
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source=tmp_path,
        c_std="c11",
    )
    extracted = engine.extract_modules()

    assert set(extracted) == {"first", "second"}
    assert set(extracted["first"].functions) == {"foo"}
    assert set(extracted["second"].functions) == {"foo"}
    assert extracted["first"].functions["foo"].ml_flags == METH_VARARGS
    assert extracted["second"].functions["foo"].ml_flags == METH_VARARGS


def test_c_signature_extraction_engine_discards_duplicate_modules_across_files(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    first_source = tmp_path / "a_first.c"
    second_source = tmp_path / "b_second.c"
    for source, py_name, c_name in [
        (first_source, "foo", "first_foo_impl"),
        (second_source, "bar", "second_bar_impl"),
    ]:
        source.write_text(
            "\n".join(
                [
                    "typedef struct _object PyObject;",
                    "typedef struct PyMethodDef {",
                    "    const char* ml_name;",
                    "    void* ml_meth;",
                    "    int ml_flags;",
                    "    const char* ml_doc;",
                    "} PyMethodDef;",
                    "typedef struct PyModuleDef {",
                    "    int m_base;",
                    "    const char* m_name;",
                    "    const char* m_doc;",
                    "    int m_size;",
                    "    PyMethodDef* m_methods;",
                    "    void* m_slots;",
                    "    void* m_traverse;",
                    "    void* m_clear;",
                    "    void* m_free;",
                    "} PyModuleDef;",
                    "#define PyModuleDef_HEAD_INIT 0",
                    "#define METH_VARARGS 1",
                    "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                    f"static PyObject* {c_name}(PyObject* self, PyObject* args) {{",
                    "    int value = 0;",
                    "    if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                    "        return (PyObject*)0;",
                    "    }",
                    "    return (PyObject*)0;",
                    "}",
                    "static PyMethodDef Methods[] = {",
                    f"    {{\"{py_name}\", {c_name}, METH_VARARGS, \"doc\"}},",
                    "    {0, 0, 0, 0}",
                    "};",
                    "static PyModuleDef moduledef = {",
                    "    PyModuleDef_HEAD_INIT,",
                    "    \"dup.shared\",",
                    "    0,",
                    "    -1,",
                    "    Methods,",
                    "    0, 0, 0, 0",
                    "};",
                    *_pyinit_lines("shared"),
                ]
            ),
            encoding="utf-8",
        )

    engine = CSignatureExtractor(
        source=tmp_path,
        c_std="c11",
    )
    extracted = engine.extract_modules()

    module = extracted["shared"]
    assert set(module.functions) == {"foo"}
    assert module.functions["foo"].ml_flags == METH_VARARGS


def test_c_signature_extraction_engine_extract_modules_ignores_unreachable_moduledefs_in_one_file(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "duplicate_modules.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct PyMethodDef {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "typedef struct PyModuleDef {",
                "    int m_base;",
                "    const char* m_name;",
                "    const char* m_doc;",
                "    int m_size;",
                "    PyMethodDef* m_methods;",
                "    void* m_slots;",
                "    void* m_traverse;",
                "    void* m_clear;",
                "    void* m_free;",
                "} PyModuleDef;",
                "#define PyModuleDef_HEAD_INIT 0",
                "#define METH_VARARGS 1",
                "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                "static PyObject* first_foo_impl(PyObject* self, PyObject* args) {",
                "    int value = 0;",
                "    if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyObject* second_bar_impl(PyObject* self, PyObject* args) {",
                "    int value = 0;",
                "    if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyMethodDef FirstMethods[] = {",
                "    {\"foo\", first_foo_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyMethodDef SecondMethods[] = {",
                "    {\"bar\", second_bar_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyModuleDef first_moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                "    \"dup.same_file\",",
                "    0,",
                "    -1,",
                "    FirstMethods,",
                "    0, 0, 0, 0",
                "};",
                "static PyModuleDef second_moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                "    \"dup.same_file\",",
                "    0,",
                "    -1,",
                "    SecondMethods,",
                "    0, 0, 0, 0",
                "};",
                *_pyinit_lines("mod", moduledef_name="first_moduledef"),
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source=tmp_path,
        c_std="c11",
    )
    extracted = engine.extract_modules()

    module = extracted["mod"]
    assert set(module.functions) == {"foo"}
    assert module.functions["foo"].ml_flags == METH_VARARGS


def test_c_signature_extraction_engine_warns_and_keeps_first_duplicate_in_same_method_table(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "duplicate_methods.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct PyMethodDef {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "typedef struct PyModuleDef {",
                "    int m_base;",
                "    const char* m_name;",
                "    const char* m_doc;",
                "    int m_size;",
                "    PyMethodDef* m_methods;",
                "    void* m_slots;",
                "    void* m_traverse;",
                "    void* m_clear;",
                "    void* m_free;",
                "} PyModuleDef;",
                "#define PyModuleDef_HEAD_INIT 0",
                "#define METH_VARARGS 1",
                "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                "static PyObject* first_foo_impl(PyObject* self, PyObject* args) {",
                "    int value = 0;",
                "    if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyObject* second_foo_impl(PyObject* self, PyObject* args) {",
                "    int value = 0;",
                "    if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyMethodDef Methods[] = {",
                "    {\"foo\", first_foo_impl, METH_VARARGS, \"doc\"},",
                "    {\"foo\", second_foo_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyModuleDef moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                "    \"dup.mod\",",
                "    0,",
                "    -1,",
                "    Methods,",
                "    0, 0, 0, 0",
                "};",
                *_pyinit_lines("dup"),
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source=tmp_path,
        c_std="c11",
    )
    extracted = engine.extract_modules()

    module = extracted["dup"]
    assert module.functions["foo"].ml_flags == METH_VARARGS


def test_c_signature_extraction_engine_warns_and_discards_duplicate_module_across_files(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    first_source = tmp_path / "a_first.c"
    second_source = tmp_path / "b_second.c"
    for source, c_name in [
        (first_source, "first_foo_impl"),
        (second_source, "second_foo_impl"),
    ]:
        source.write_text(
            "\n".join(
                [
                    "typedef struct _object PyObject;",
                    "typedef struct PyMethodDef {",
                    "    const char* ml_name;",
                    "    void* ml_meth;",
                    "    int ml_flags;",
                    "    const char* ml_doc;",
                    "} PyMethodDef;",
                    "typedef struct PyModuleDef {",
                    "    int m_base;",
                    "    const char* m_name;",
                    "    const char* m_doc;",
                    "    int m_size;",
                    "    PyMethodDef* m_methods;",
                    "    void* m_slots;",
                    "    void* m_traverse;",
                    "    void* m_clear;",
                    "    void* m_free;",
                    "} PyModuleDef;",
                    "#define PyModuleDef_HEAD_INIT 0",
                    "#define METH_VARARGS 1",
                    "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                    f"static PyObject* {c_name}(PyObject* self, PyObject* args) {{",
                    "    int value = 0;",
                    "    if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                    "        return (PyObject*)0;",
                    "    }",
                    "    return (PyObject*)0;",
                    "}",
                    "static PyMethodDef Methods[] = {",
                    f"    {{\"foo\", {c_name}, METH_VARARGS, \"doc\"}},",
                    "    {0, 0, 0, 0}",
                    "};",
                    "static PyModuleDef moduledef = {",
                    "    PyModuleDef_HEAD_INIT,",
                    "    \"dup.shared\",",
                    "    0,",
                    "    -1,",
                    "    Methods,",
                    "    0, 0, 0, 0",
                    "};",
                    *_pyinit_lines("shared"),
                ]
            ),
            encoding="utf-8",
        )

    engine = CSignatureExtractor(
        source=tmp_path,
        c_std="c11",
    )
    extracted = engine.extract_modules()

    module = extracted["shared"]
    assert module.functions["foo"].ml_flags == METH_VARARGS


def test_c_signature_extraction_engine_warns_and_keeps_first_module_create_candidate_in_pyinit(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "multiple_candidates.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct PyMethodDef {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "typedef struct PyModuleDef {",
                "    int m_base;",
                "    const char* m_name;",
                "    const char* m_doc;",
                "    int m_size;",
                "    PyMethodDef* m_methods;",
                "    void* m_slots;",
                "    void* m_traverse;",
                "    void* m_clear;",
                "    void* m_free;",
                "} PyModuleDef;",
                "#define PyModuleDef_HEAD_INIT 0",
                "#define METH_VARARGS 1",
                "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                "PyObject* PyModule_Create(PyModuleDef* def);",
                "static PyObject* first_foo_impl(PyObject* self, PyObject* args) {",
                "    int value = 0;",
                "    if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyObject* second_bar_impl(PyObject* self, PyObject* args) {",
                "    int value = 0;",
                "    if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyMethodDef FirstMethods[] = {",
                "    {\"foo\", first_foo_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyMethodDef SecondMethods[] = {",
                "    {\"bar\", second_bar_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyModuleDef first_moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                "    \"dup.same_file.first\",",
                "    0,",
                "    -1,",
                "    FirstMethods,",
                "    0, 0, 0, 0",
                "};",
                "static PyModuleDef second_moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                "    \"dup.same_file.second\",",
                "    0,",
                "    -1,",
                "    SecondMethods,",
                "    0, 0, 0, 0",
                "};",
                "PyObject* PyInit_mod(void) {",
                "    PyModule_Create(&first_moduledef);",
                "    return PyModule_Create(&second_moduledef);",
                "}",
            ]
        ),
        encoding="utf-8",
    )

    log_output = StringIO()
    sink_id = logger.add(log_output, format="{message}")
    try:
        extracted = CSignatureExtractor(source=tmp_path, c_std="c11").extract_modules()
    finally:
        logger.remove(sink_id)

    assert "PyInit 中存在多个模块创建候选, 保留首个" in log_output.getvalue()
    assert set(extracted["mod"].functions) == {"foo"}


def test_c_signature_extraction_engine_extract_modules_ignores_registered_types_from_pymodule_addobject(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "module_with_type.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct PyMethodDef {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "typedef struct PyModuleDef {",
                "    int m_base;",
                "    const char* m_name;",
                "    const char* m_doc;",
                "    int m_size;",
                "    PyMethodDef* m_methods;",
                "    void* m_slots;",
                "    void* m_traverse;",
                "    void* m_clear;",
                "    void* m_free;",
                "} PyModuleDef;",
                "typedef struct PyTypeObject {",
                "    const char* tp_name;",
                "    PyMethodDef* tp_methods;",
                "} PyTypeObject;",
                "#define PyModuleDef_HEAD_INIT 0",
                "#define METH_VARARGS 1",
                "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                "PyObject* PyModule_Create(PyModuleDef* def);",
                "int PyModule_AddObject(PyObject* module, const char* name, PyObject* value);",
                "static PyObject* module_foo(PyObject* self, PyObject* args) {",
                "    int count = 0;",
                "    if (!PyArg_ParseTuple(args, \"i\", &count)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyObject* point_foo(PyObject* self, PyObject* args) {",
                "    const char* label = 0;",
                "    if (!PyArg_ParseTuple(args, \"s\", &label)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyMethodDef ModuleMethods[] = {",
                "    {\"foo\", module_foo, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyMethodDef PointMethods[] = {",
                "    {\"foo\", point_foo, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyTypeObject PointType = {",
                "    .tp_name = \"pkg.mod.Point\",",
                "    .tp_methods = PointMethods,",
                "};",
                "static PyModuleDef moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                "    \"pkg.mod\",",
                "    0,",
                "    -1,",
                "    ModuleMethods,",
                "    0, 0, 0, 0",
                "};",
                "PyObject* PyInit_mod(void) {",
                "    PyObject* m = PyModule_Create(&moduledef);",
                "    PyModule_AddObject(m, \"Point\", (PyObject*)&PointType);",
                "    return m;",
                "}",
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source=tmp_path,
        c_std="c11",
    )
    extracted = engine.extract_modules()

    module = extracted["mod"]
    assert module.functions["foo"].ml_name == "foo"
    assert module.functions["foo"].ml_flags == METH_VARARGS


def test_c_signature_extraction_engine_extract_modules_supports_pymodule_addobjectref(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "module_with_addobjectref_type.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct PyMethodDef {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "typedef struct PyModuleDef {",
                "    int m_base;",
                "    const char* m_name;",
                "    const char* m_doc;",
                "    int m_size;",
                "    PyMethodDef* m_methods;",
                "    void* m_slots;",
                "    void* m_traverse;",
                "    void* m_clear;",
                "    void* m_free;",
                "} PyModuleDef;",
                "typedef struct PyTypeObject {",
                "    const char* tp_name;",
                "    PyMethodDef* tp_methods;",
                "} PyTypeObject;",
                "#define PyModuleDef_HEAD_INIT 0",
                "#define METH_VARARGS 1",
                "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                "PyObject* PyModule_Create(PyModuleDef* def);",
                "int PyModule_AddObjectRef(PyObject* module, const char* name, PyObject* value);",
                "static PyObject* point_foo(PyObject* self, PyObject* args) {",
                "    const char* label = 0;",
                "    if (!PyArg_ParseTuple(args, \"s\", &label)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyMethodDef ModuleMethods[] = {",
                "    {0, 0, 0, 0}",
                "};",
                "static PyMethodDef PointMethods[] = {",
                "    {\"foo\", point_foo, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyTypeObject PointType = {",
                "    .tp_name = \"pkg.mod.Point\",",
                "    .tp_methods = PointMethods,",
                "};",
                "static PyModuleDef moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                "    \"pkg.mod\",",
                "    0,",
                "    -1,",
                "    ModuleMethods,",
                "    0, 0, 0, 0",
                "};",
                "PyObject* PyInit_mod(void) {",
                "    PyObject* m = PyModule_Create(&moduledef);",
                "    PyModule_AddObjectRef(m, \"Point\", (PyObject*)&PointType);",
                "    return m;",
                "}",
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source=tmp_path,
        c_std="c11",
    )
    extracted = engine.extract_modules()


def test_c_signature_extraction_engine_extract_modules_supports_pymodule_addtype(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "module_with_addtype.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct PyMethodDef {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "typedef struct PyModuleDef {",
                "    int m_base;",
                "    const char* m_name;",
                "    const char* m_doc;",
                "    int m_size;",
                "    PyMethodDef* m_methods;",
                "    void* m_slots;",
                "    void* m_traverse;",
                "    void* m_clear;",
                "    void* m_free;",
                "} PyModuleDef;",
                "typedef struct PyTypeObject {",
                "    const char* tp_name;",
                "    PyMethodDef* tp_methods;",
                "} PyTypeObject;",
                "#define PyModuleDef_HEAD_INIT 0",
                "#define METH_VARARGS 1",
                "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                "PyObject* PyModule_Create(PyModuleDef* def);",
                "int PyModule_AddType(PyObject* module, PyTypeObject* type);",
                "static PyObject* point_foo(PyObject* self, PyObject* args) {",
                "    const char* label = 0;",
                "    if (!PyArg_ParseTuple(args, \"s\", &label)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyMethodDef ModuleMethods[] = {",
                "    {0, 0, 0, 0}",
                "};",
                "static PyMethodDef PointMethods[] = {",
                "    {\"foo\", point_foo, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyTypeObject PointType = {",
                "    .tp_name = \"pkg.mod.Point\",",
                "    .tp_methods = PointMethods,",
                "};",
                "static PyModuleDef moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                "    \"pkg.mod\",",
                "    0,",
                "    -1,",
                "    ModuleMethods,",
                "    0, 0, 0, 0",
                "};",
                "PyObject* PyInit_mod(void) {",
                "    PyObject* m = PyModule_Create(&moduledef);",
                "    PyModule_AddType(m, &PointType);",
                "    return m;",
                "}",
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source=tmp_path,
        c_std="c11",
    )
    extracted = engine.extract_modules()


def test_c_signature_extraction_engine_extract_modules_supports_pymoduledef_init(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "module_def_init.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct PyMethodDef {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "typedef struct PyModuleDef {",
                "    int m_base;",
                "    const char* m_name;",
                "    const char* m_doc;",
                "    int m_size;",
                "    PyMethodDef* m_methods;",
                "    void* m_slots;",
                "    void* m_traverse;",
                "    void* m_clear;",
                "    void* m_free;",
                "} PyModuleDef;",
                "#define PyModuleDef_HEAD_INIT 0",
                "#define METH_VARARGS 1",
                "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                "PyObject* PyModuleDef_Init(PyModuleDef* def);",
                "static PyObject* foo_impl(PyObject* self, PyObject* args) {",
                "    int value = 0;",
                "    if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyMethodDef Methods[] = {",
                "    {\"foo\", foo_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyModuleDef moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                "    \"pkg.mod\",",
                "    0,",
                "    -1,",
                "    Methods,",
                "    0, 0, 0, 0",
                "};",
                "PyObject* PyInit_mod(void) {",
                "    return PyModuleDef_Init(&moduledef);",
                "}",
            ]
        ),
        encoding="utf-8",
    )

    extracted = CSignatureExtractor(source=tmp_path, c_std="c11").extract_modules()

    assert "mod" in extracted
    assert extracted["mod"].functions["foo"].ml_flags == METH_VARARGS


def test_c_signature_extraction_engine_extract_modules_supports_designated_moduledef_initializer(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "designated_module.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct PyMethodDef {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "typedef struct PyModuleDef {",
                "    int m_base;",
                "    const char* m_name;",
                "    const char* m_doc;",
                "    int m_size;",
                "    PyMethodDef* m_methods;",
                "    void* m_slots;",
                "    void* m_traverse;",
                "    void* m_clear;",
                "    void* m_free;",
                "} PyModuleDef;",
                "#define METH_VARARGS 1",
                "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                "static PyObject* foo_impl(PyObject* self, PyObject* args) {",
                "    int value = 0;",
                "    if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyMethodDef Methods[] = {",
                "    {\"foo\", foo_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyModuleDef moduledef = {",
                "    .m_name = \"designated.mod\",",
                "    .m_doc = 0,",
                "    .m_size = -1,",
                "    .m_methods = Methods,",
                "};",
                *_pyinit_lines("mod"),
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source=tmp_path,
        c_std="c11",
    )
    extracted = engine.extract_modules()

    assert "mod" in extracted
    assert extracted["mod"].functions["foo"].ml_name == "foo"
    assert extracted["mod"].functions["foo"].ml_flags == METH_VARARGS


def test_c_signature_extraction_engine_extract_modules_supports_mixed_moduledef_initializer_styles(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "mixed_designated_module.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct PyMethodDef {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "typedef struct PyModuleDef {",
                "    int m_base;",
                "    const char* m_name;",
                "    const char* m_doc;",
                "    int m_size;",
                "    PyMethodDef* m_methods;",
                "    void* m_slots;",
                "    void* m_traverse;",
                "    void* m_clear;",
                "    void* m_free;",
                "} PyModuleDef;",
                "#define PyModuleDef_HEAD_INIT 0",
                "#define METH_VARARGS 1",
                "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                "static PyObject* foo_impl(PyObject* self, PyObject* args) {",
                "    int value = 0;",
                "    if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyMethodDef Methods[] = {",
                "    {\"foo\", foo_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyModuleDef moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                "    .m_name = \"mixed.mod\",",
                "    0,",
                "    -1,",
                "    Methods,",
                "    0, 0, 0, 0",
                "};",
                *_pyinit_lines("mod"),
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source=tmp_path,
        c_std="c11",
    )
    extracted = engine.extract_modules()

    assert "mod" in extracted
    assert extracted["mod"].functions["foo"].ml_name == "foo"
    assert extracted["mod"].functions["foo"].ml_flags == METH_VARARGS


def test_c_signature_extraction_engine_extract_modules_ignores_moduledefs_without_pyinit(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "unreachable_module.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct PyMethodDef {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "typedef struct PyModuleDef {",
                "    int m_base;",
                "    const char* m_name;",
                "    const char* m_doc;",
                "    int m_size;",
                "    PyMethodDef* m_methods;",
                "    void* m_slots;",
                "    void* m_traverse;",
                "    void* m_clear;",
                "    void* m_free;",
                "} PyModuleDef;",
                "#define PyModuleDef_HEAD_INIT 0",
                "#define METH_VARARGS 1",
                "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                "static PyObject* foo_impl(PyObject* self, PyObject* args) {",
                "    int value = 0;",
                "    if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyMethodDef Methods[] = {",
                "    {\"foo\", foo_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyModuleDef moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                "    \"orphan.mod\",",
                "    0,",
                "    -1,",
                "    Methods,",
                "    0, 0, 0, 0",
                "};",
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source=tmp_path,
        c_std="c11",
    )
    extracted = engine.extract_modules()

    assert extracted == {}


def test_c_signature_extraction_engine_extract_modules_keeps_named_modules_without_methods(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "module_without_methods.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct PyMethodDef {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "typedef struct PyModuleDef {",
                "    int m_base;",
                "    const char* m_name;",
                "    const char* m_doc;",
                "    int m_size;",
                "    PyMethodDef* m_methods;",
                "    void* m_slots;",
                "    void* m_traverse;",
                "    void* m_clear;",
                "    void* m_free;",
                "} PyModuleDef;",
                "#define PyModuleDef_HEAD_INIT 0",
                "static PyModuleDef moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                "    \"empty.mod\",",
                "    0,",
                "    -1,",
                "    0,",
                "    0, 0, 0, 0",
                "};",
                *_pyinit_lines("mod"),
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source=tmp_path,
        c_std="c11",
    )
    extracted = engine.extract_modules()

    assert "mod" in extracted
    assert extracted["mod"].functions == {}


def test_c_signature_extraction_engine_extract_modules_ignores_moduledefs_without_m_name(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    source = tmp_path / "nameless_module.c"
    source.write_text(
        "\n".join(
            [
                "typedef struct _object PyObject;",
                "typedef struct PyMethodDef {",
                "    const char* ml_name;",
                "    void* ml_meth;",
                "    int ml_flags;",
                "    const char* ml_doc;",
                "} PyMethodDef;",
                "typedef struct PyModuleDef {",
                "    int m_base;",
                "    const char* m_name;",
                "    const char* m_doc;",
                "    int m_size;",
                "    PyMethodDef* m_methods;",
                "    void* m_slots;",
                "    void* m_traverse;",
                "    void* m_clear;",
                "    void* m_free;",
                "} PyModuleDef;",
                "#define PyModuleDef_HEAD_INIT 0",
                "#define METH_VARARGS 1",
                "int PyArg_ParseTuple(PyObject* args, const char* fmt, ...);",
                "static PyObject* foo_impl(PyObject* self, PyObject* args) {",
                "    int value = 0;",
                "    if (!PyArg_ParseTuple(args, \"i\", &value)) {",
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "static PyMethodDef Methods[] = {",
                "    {\"foo\", foo_impl, METH_VARARGS, \"doc\"},",
                "    {0, 0, 0, 0}",
                "};",
                "static PyModuleDef moduledef = {",
                "    .m_doc = 0,",
                "    .m_size = -1,",
                "    .m_methods = Methods,",
                "};",
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source=tmp_path,
        c_std="c11",
    )
    extracted = engine.extract_modules()

    assert extracted == {}


def test_c_signature_extraction_engine_extract_modules_resolves_ml_meth_definition_across_translation_units(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    shared_header = tmp_path / "shared.h"
    module_source = tmp_path / "module.c"
    impl_source = tmp_path / "impl.c"

    shared_header.write_text(
        _basic_c_extension_definitions(
            "PyObject* foo_impl(PyObject* self, PyObject* args);",
        ),
        encoding="utf-8",
    )
    module_source.write_text(
        "\n".join(
            [
                '#include "shared.h"',
                "static PyMethodDef Methods[] = {",
                '    {"foo", foo_impl, METH_VARARGS, "doc"},',
                "    {0, 0, 0, 0}",
                "};",
                "static PyModuleDef moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                '    "cross.func",',
                "    0,",
                "    -1,",
                "    Methods,",
                "    0, 0, 0, 0",
                "};",
                *_pyinit_lines("mod"),
                "",
            ]
        ),
        encoding="utf-8",
    )
    impl_source.write_text(
        "\n".join(
            [
                '#include "shared.h"',
                "PyObject* foo_impl(PyObject* self, PyObject* args) {",
                "    int value = 0;",
                '    if (!PyArg_ParseTuple(args, "i", &value)) {',
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    extracted = CSignatureExtractor(source=tmp_path, c_std="c11").extract_modules()

    signatures = extracted["mod"].functions["foo"].signatures
    assert signatures == [ExtractedSignature(arguments=[_arg("value", "int")])]


def test_c_signature_extraction_engine_extract_modules_resolves_method_table_definition_across_translation_units(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    shared_header = tmp_path / "shared.h"
    module_source = tmp_path / "module.c"
    impl_source = tmp_path / "impl.c"

    shared_header.write_text(
        _basic_c_extension_definitions(
            "PyObject* foo_impl(PyObject* self, PyObject* args);",
            "extern PyMethodDef Methods[];",
        ),
        encoding="utf-8",
    )
    module_source.write_text(
        "\n".join(
            [
                '#include "shared.h"',
                "static PyModuleDef moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                '    "cross.methods",',
                "    0,",
                "    -1,",
                "    Methods,",
                "    0, 0, 0, 0",
                "};",
                *_pyinit_lines("mod"),
                "",
            ]
        ),
        encoding="utf-8",
    )
    impl_source.write_text(
        "\n".join(
            [
                '#include "shared.h"',
                "PyObject* foo_impl(PyObject* self, PyObject* args) {",
                "    int value = 0;",
                '    if (!PyArg_ParseTuple(args, "i", &value)) {',
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "PyMethodDef Methods[] = {",
                '    {"foo", foo_impl, METH_VARARGS, "doc"},',
                "    {0, 0, 0, 0}",
                "};",
                "",
            ]
        ),
        encoding="utf-8",
    )

    extracted = CSignatureExtractor(source=tmp_path, c_std="c11").extract_modules()

    signatures = extracted["mod"].functions["foo"].signatures
    assert signatures == [ExtractedSignature(arguments=[_arg("value", "int")])]


def test_c_signature_extraction_engine_usr_index_deduplicates_same_location_header_definitions_without_warning(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    shared_header = tmp_path / "shared.h"
    module_source = tmp_path / "module.c"
    helper_source = tmp_path / "helper.c"

    shared_header.write_text(
        _basic_c_extension_definitions(
            "static inline PyObject* inline_impl(PyObject* self, PyObject* args) {",
            "    int value = 0;",
            '    if (!PyArg_ParseTuple(args, "i", &value)) {',
            "        return (PyObject*)0;",
            "    }",
            "    return (PyObject*)0;",
            "}",
        ),
        encoding="utf-8",
    )
    module_source.write_text(
        "\n".join(
            [
                '#include "shared.h"',
                "static PyMethodDef Methods[] = {",
                '    {"foo", inline_impl, METH_VARARGS, "doc"},',
                "    {0, 0, 0, 0}",
                "};",
                "static PyModuleDef moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                '    "inline.mod",',
                "    0,",
                "    -1,",
                "    Methods,",
                "    0, 0, 0, 0",
                "};",
                *_pyinit_lines("inline"),
                "",
            ]
        ),
        encoding="utf-8",
    )
    helper_source.write_text('#include "shared.h"\n', encoding="utf-8")

    log_output = StringIO()
    sink_id = logger.add(log_output, format="{message}")
    try:
        extracted = CSignatureExtractor(source=tmp_path, c_std="c11").extract_modules()
    finally:
        logger.remove(sink_id)

    assert "USR 定义冲突" not in log_output.getvalue()
    assert extracted["inline"].functions["foo"].signatures == [
        ExtractedSignature(arguments=[_arg("value", "int")])
    ]


def test_c_signature_extraction_engine_usr_index_warns_for_conflicting_definitions_and_keeps_first(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    shared_header = tmp_path / "shared.h"
    module_source = tmp_path / "module.c"
    first_impl_source = tmp_path / "a_first_impl.c"
    second_impl_source = tmp_path / "b_second_impl.c"

    shared_header.write_text(
        _basic_c_extension_definitions(
            "PyObject* foo_impl(PyObject* self, PyObject* args);",
        ),
        encoding="utf-8",
    )
    module_source.write_text(
        "\n".join(
            [
                '#include "shared.h"',
                "static PyMethodDef Methods[] = {",
                '    {"foo", foo_impl, METH_VARARGS, "doc"},',
                "    {0, 0, 0, 0}",
                "};",
                "static PyModuleDef moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                '    "conflict.mod",',
                "    0,",
                "    -1,",
                "    Methods,",
                "    0, 0, 0, 0",
                "};",
                *_pyinit_lines("conflict"),
                "",
            ]
        ),
        encoding="utf-8",
    )
    first_impl_source.write_text(
        "\n".join(
            [
                '#include "shared.h"',
                "PyObject* foo_impl(PyObject* self, PyObject* args) {",
                "    int value = 0;",
                '    if (!PyArg_ParseTuple(args, "i", &value)) {',
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    second_impl_source.write_text(
        "\n".join(
            [
                '#include "shared.h"',
                "PyObject* foo_impl(PyObject* self, PyObject* args) {",
                "    const char* label = 0;",
                '    if (!PyArg_ParseTuple(args, "s", &label)) {',
                "        return (PyObject*)0;",
                "    }",
                "    return (PyObject*)0;",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    log_output = StringIO()
    sink_id = logger.add(log_output, format="{message}")
    try:
        extracted = CSignatureExtractor(source=tmp_path, c_std="c11").extract_modules()
    finally:
        logger.remove(sink_id)

    assert "USR 定义冲突, 保留首个定义" in log_output.getvalue()
    assert extracted["conflict"].functions["foo"].signatures == [
        ExtractedSignature(arguments=[_arg("value", "int")])
    ]


def test_c_signature_extraction_engine_warns_and_skips_when_cross_tu_definition_is_missing(
    tmp_path: Path,
) -> None:
    pytest.importorskip("clang.cindex")
    if _get_packaged_libclang_path() is None:
        pytest.skip("Packaged libclang library is not available")

    shared_header = tmp_path / "shared.h"
    module_source = tmp_path / "module.c"

    shared_header.write_text(
        _basic_c_extension_definitions(
            "PyObject* foo_impl(PyObject* self, PyObject* args);",
        ),
        encoding="utf-8",
    )
    module_source.write_text(
        "\n".join(
            [
                '#include "shared.h"',
                "static PyMethodDef Methods[] = {",
                '    {"foo", foo_impl, METH_VARARGS, "doc"},',
                "    {0, 0, 0, 0}",
                "};",
                "static PyModuleDef moduledef = {",
                "    PyModuleDef_HEAD_INIT,",
                '    "missing.mod",',
                "    0,",
                "    -1,",
                "    Methods,",
                "    0, 0, 0, 0",
                "};",
                *_pyinit_lines("missing"),
                "",
            ]
        ),
        encoding="utf-8",
    )

    log_output = StringIO()
    sink_id = logger.add(log_output, format="{message}")
    try:
        extracted = CSignatureExtractor(source=tmp_path, c_std="c11").extract_modules()
    finally:
        logger.remove(sink_id)

    assert "找不到 function definition, ml_name: foo" in log_output.getvalue()
    assert extracted["missing"].functions == {}


def test_c_signature_engine_extract_modules_accepts_empty_compilation_database(
    tmp_path: Path,
) -> None:
    compilation_database = _write_compilation_database(tmp_path, files=[])
    engine = CSignatureExtractor(compilation_database=compilation_database)

    assert engine.extract_modules() == {}


def test_c_signature_extraction_engine_logs_parse_success_after_tu_collection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sample_command = translation_unit_module.CompilationCommand(
        file_path=tmp_path / "sample.c",
        working_directory=tmp_path,
        parse_args=["-Iinclude"],
    )
    recorded_parse_calls: list[tuple[object, object, list[str] | None]] = []

    monkeypatch.setattr(
        c_signature_extraction_module.clang_parser,
        "list_compilation_commands",
        lambda compilation_database: [sample_command],
    )
    monkeypatch.setattr(
        c_signature_extraction_module.Index,
        "create",
        staticmethod(lambda: object()),
    )
    monkeypatch.setattr(
        c_signature_extraction_module.clang_parser,
        "build_effective_parse_args",
        lambda compilation_command: ["-Iinclude", "-resource-dir", "/opt/clang/resource"],
    )

    def _parse(index: object, compilation_command: object, **kwargs: object) -> SimpleNamespace:
        recorded_parse_calls.append(
            (index, compilation_command, kwargs.get("effective_parse_args"))
        )
        return SimpleNamespace(
            cursor=_FakeNode(kind=clang.cindex.CursorKind.TRANSLATION_UNIT),
            diagnostics=[],
        )

    monkeypatch.setattr(c_signature_extraction_module.clang_parser, "parse", _parse)
    monkeypatch.setattr(
        c_signature_extraction_module.module_table,
        "collect_modules_from_translation_unit",
        lambda cursor, definition_index: [],
    )

    log_output = StringIO()
    sink_id = logger.add(log_output, format="{message}")
    try:
        extracted = c_signature_extraction_module.extract_c_signature_modules(
            tmp_path / "compile_commands.json"
        )
    finally:
        logger.remove(sink_id)

    assert extracted == {}
    assert "阶段进度 [1/4] 开始Parse, 文件数: 1" in log_output.getvalue()
    assert f"Parse进度 [1/1], 文件: {sample_command.file_path}" in log_output.getvalue()
    assert len(recorded_parse_calls) == 1
    assert recorded_parse_calls[0][1] is sample_command
    assert recorded_parse_calls[0][2] == [
        "-Iinclude",
        "-resource-dir",
        "/opt/clang/resource",
    ]
    assert "Parse成功" in log_output.getvalue()
    assert "阶段进度 [2/4] 开始构建索引, TU数: 1" in log_output.getvalue()
    assert "阶段进度 [3/4] 开始收集模块, TU数: 1" in log_output.getvalue()
    assert "阶段进度 [4/4] 开始推断签名, 模块数: 0, 函数数: 0" in log_output.getvalue()
    assert f"文件路径: {sample_command.file_path}" in log_output.getvalue()
    assert f"工作目录: {sample_command.working_directory}" in log_output.getvalue()
    assert "解析参数: -Iinclude -resource-dir /opt/clang/resource" in log_output.getvalue()


def test_c_signature_extraction_engine_logs_parse_diagnostics_after_tu_collection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "sample.c"
    sample_command = translation_unit_module.CompilationCommand(
        file_path=source_path,
        working_directory=tmp_path,
        parse_args=["-Iinclude"],
    )
    translation_unit = _FakeTranslationUnit(
        diagnostics=[
            _FakeDiagnostic(
                severity=_FakeDiagnosticType.Warning,
                message="warning detail",
                file_name=str(source_path),
                line=3,
                column=1,
            ),
            _FakeDiagnostic(
                severity=_FakeDiagnosticType.Error,
                message="error detail",
                file_name=str(source_path),
                line=7,
                column=9,
            ),
        ]
    )

    monkeypatch.setattr(
        c_signature_extraction_module.clang_parser,
        "list_compilation_commands",
        lambda compilation_database: [sample_command],
    )
    monkeypatch.setattr(
        c_signature_extraction_module.Index,
        "create",
        staticmethod(lambda: object()),
    )
    monkeypatch.setattr(
        c_signature_extraction_module.clang_parser,
        "build_effective_parse_args",
        lambda compilation_command: ["-Iinclude"],
    )
    monkeypatch.setattr(
        c_signature_extraction_module.clang_parser,
        "parse",
        lambda *args, **kwargs: translation_unit,
    )
    monkeypatch.setattr(
        c_signature_extraction_module.module_table,
        "collect_modules_from_translation_unit",
        lambda cursor, definition_index: [],
    )

    log_output = StringIO()
    sink_id = logger.add(log_output, format="{message}")
    try:
        extracted = c_signature_extraction_module.extract_c_signature_modules(
            tmp_path / "compile_commands.json"
        )
    finally:
        logger.remove(sink_id)

    assert extracted == {}
    assert "阶段进度 [1/4] 开始Parse, 文件数: 1" in log_output.getvalue()
    assert f"Parse进度 [1/1], 文件: {sample_command.file_path}" in log_output.getvalue()
    assert "Parse诊断" in log_output.getvalue()
    assert "阶段进度 [2/4] 开始构建索引, TU数: 1" in log_output.getvalue()
    assert "阶段进度 [3/4] 开始收集模块, TU数: 1" in log_output.getvalue()
    assert "阶段进度 [4/4] 开始推断签名, 模块数: 0, 函数数: 0" in log_output.getvalue()
    assert f"文件路径: {sample_command.file_path}" in log_output.getvalue()
    assert f"工作目录: {sample_command.working_directory}" in log_output.getvalue()
    assert "解析参数: -Iinclude" in log_output.getvalue()
    assert "- [WARNING]" in log_output.getvalue()
    assert "warning detail" in log_output.getvalue()
    assert "- [ERROR]" in log_output.getvalue()
    assert "error detail" in log_output.getvalue()


def test_c_signature_extraction_engine_skips_failed_parse_and_continues_next_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first_command = translation_unit_module.CompilationCommand(
        file_path=tmp_path / "first.c",
        working_directory=tmp_path,
        parse_args=["-DFIRST"],
    )
    second_command = translation_unit_module.CompilationCommand(
        file_path=tmp_path / "second.c",
        working_directory=tmp_path,
        parse_args=["-DSECOND"],
    )
    parse_calls: list[object] = []
    second_tu = _FakeTranslationUnit(diagnostics=[])

    monkeypatch.setattr(
        c_signature_extraction_module.clang_parser,
        "list_compilation_commands",
        lambda compilation_database: [first_command, second_command],
    )
    monkeypatch.setattr(
        c_signature_extraction_module.Index,
        "create",
        staticmethod(lambda: object()),
    )

    def _build_effective_parse_args(
        compilation_command: translation_unit_module.CompilationCommand,
    ) -> list[str]:
        return list(compilation_command.parse_args)

    def _parse(index: object, compilation_command: object, **kwargs: object) -> _FakeTranslationUnit:
        _ = (index, kwargs)
        parse_calls.append(compilation_command)
        if compilation_command is first_command:
            raise clang.cindex.TranslationUnitLoadError("broken parse")
        return second_tu

    monkeypatch.setattr(
        c_signature_extraction_module.clang_parser,
        "build_effective_parse_args",
        _build_effective_parse_args,
    )
    monkeypatch.setattr(c_signature_extraction_module.clang_parser, "parse", _parse)
    monkeypatch.setattr(
        c_signature_extraction_module.module_table,
        "collect_modules_from_translation_unit",
        lambda cursor, definition_index: [],
    )

    log_output = StringIO()
    sink_id = logger.add(log_output, format="{message}")
    try:
        extracted = c_signature_extraction_module.extract_c_signature_modules(
            tmp_path / "compile_commands.json"
        )
    finally:
        logger.remove(sink_id)

    assert extracted == {}
    assert parse_calls == [first_command, second_command]
    assert "阶段进度 [1/4] 开始Parse, 文件数: 2" in log_output.getvalue()
    assert f"Parse进度 [1/2], 文件: {first_command.file_path}" in log_output.getvalue()
    assert f"Parse进度 [2/2], 文件: {second_command.file_path}" in log_output.getvalue()
    assert "Parse失败, 跳过文件" in log_output.getvalue()
    assert f"文件路径: {first_command.file_path}" in log_output.getvalue()
    assert "解析参数: -DFIRST" in log_output.getvalue()
    assert "Parse成功" in log_output.getvalue()
    assert f"文件路径: {second_command.file_path}" in log_output.getvalue()
    assert "解析参数: -DSECOND" in log_output.getvalue()
    assert "阶段进度 [2/4] 开始构建索引, TU数: 1" in log_output.getvalue()
    assert "阶段进度 [3/4] 开始收集模块, TU数: 1" in log_output.getvalue()
    assert "阶段进度 [4/4] 开始推断签名, 模块数: 0, 函数数: 0" in log_output.getvalue()


def test_c_signature_extraction_engine_logs_exception_and_continues_next_function(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bad_cursor = object()
    good_cursor = object()
    bad_function = ExtractedFunction(ml_name="bad", function_cursor=bad_cursor)
    good_function = ExtractedFunction(ml_name="good", function_cursor=good_cursor)
    logged_messages: list[str] = []
    log_output = StringIO()

    sample_command = translation_unit_module.CompilationCommand(
        file_path=tmp_path / "sample.c",
        working_directory=tmp_path,
        parse_args=[],
    )
    monkeypatch.setattr(
        c_signature_extraction_module.clang_parser,
        "list_compilation_commands",
        lambda compilation_database: [sample_command],
    )
    monkeypatch.setattr(
        c_signature_extraction_module.clang_parser,
        "build_effective_parse_args",
        lambda compilation_command: [],
    )
    monkeypatch.setattr(
        c_signature_extraction_module.Index,
        "create",
        staticmethod(lambda: object()),
    )
    monkeypatch.setattr(
        c_signature_extraction_module.clang_parser,
        "parse",
        lambda *args, **kwargs: SimpleNamespace(
            cursor=_FakeNode(kind=clang.cindex.CursorKind.TRANSLATION_UNIT),
            diagnostics=[],
        ),
    )
    monkeypatch.setattr(
        c_signature_extraction_module.module_table,
        "collect_modules_from_translation_unit",
        lambda cursor, definition_index: [
            ExtractedModule(
                name="pkg.mod",
                functions={
                    "bad": bad_function,
                    "good": good_function,
                },
            )
        ],
    )

    def _infer_signature(c_function: object) -> list[ExtractedSignature]:
        if getattr(c_function, "function_cursor", None) is bad_cursor:
            raise RuntimeError("broken inference")
        return [ExtractedSignature(return_type=RawType("int"))]

    def _exception(message: str, *args: object) -> None:
        logged_messages.append(message.format(*args))

    monkeypatch.setattr(
        c_signature_extraction_module.signature_inference,
        "infer_signature",
        _infer_signature,
    )
    monkeypatch.setattr(c_signature_extraction_module.logger, "exception", _exception)

    sink_id = logger.add(log_output, format="{message}")
    try:
        extracted = c_signature_extraction_module.extract_c_signature_modules(
            tmp_path / "compile_commands.json"
        )
    finally:
        logger.remove(sink_id)

    assert extracted["pkg.mod"].functions["bad"].signatures == []
    assert extracted["pkg.mod"].functions["good"].signatures == [
        ExtractedSignature(return_type=RawType("int"))
    ]
    assert "阶段进度 [4/4] 开始推断签名, 模块数: 1, 函数数: 2" in log_output.getvalue()
    assert "签名推断进度 [1/2], module_name: pkg.mod, func_name: bad" in log_output.getvalue()
    assert "签名推断进度 [2/2], module_name: pkg.mod, func_name: good" in log_output.getvalue()
    assert len(logged_messages) == 1
    assert "pkg.mod" in logged_messages[0]
    assert "bad" in logged_messages[0]
