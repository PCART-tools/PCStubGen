from __future__ import annotations

from io import StringIO

from loguru import logger

from tests._c_extension_test_support import *


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
                ]
            ),
            encoding="utf-8",
        )

    engine = CSignatureExtractor(
        source=tmp_path,
        c_std="c11",
    )
    extracted = engine.extract_modules()

    module = extracted["dup.shared"]
    assert set(module.functions) == {"foo"}
    assert module.functions["foo"].ml_flags == METH_VARARGS


def test_c_signature_extraction_engine_discards_duplicate_modules_in_one_file(
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
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source=tmp_path,
        c_std="c11",
    )
    extracted = engine.extract_modules()

    module = extracted["dup.same_file"]
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
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source=tmp_path,
        c_std="c11",
    )
    extracted = engine.extract_modules()

    module = extracted["dup.mod"]
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
                ]
            ),
            encoding="utf-8",
        )

    engine = CSignatureExtractor(
        source=tmp_path,
        c_std="c11",
    )
    extracted = engine.extract_modules()

    module = extracted["dup.shared"]
    assert module.functions["foo"].ml_flags == METH_VARARGS


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

    module = extracted["pkg.mod"]
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
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source=tmp_path,
        c_std="c11",
    )
    extracted = engine.extract_modules()

    assert "designated.mod" in extracted
    assert extracted["designated.mod"].functions["foo"].ml_name == "foo"
    assert extracted["designated.mod"].functions["foo"].ml_flags == METH_VARARGS


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
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source=tmp_path,
        c_std="c11",
    )
    extracted = engine.extract_modules()

    assert "mixed.mod" in extracted
    assert extracted["mixed.mod"].functions["foo"].ml_name == "foo"
    assert extracted["mixed.mod"].functions["foo"].ml_flags == METH_VARARGS


def test_c_signature_extraction_engine_extract_modules_accepts_moduledefs_without_pyinit(
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

    assert extracted["orphan.mod"].functions["foo"].ml_name == "foo"
    assert extracted["orphan.mod"].functions["foo"].ml_flags == METH_VARARGS


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
            ]
        ),
        encoding="utf-8",
    )

    engine = CSignatureExtractor(
        source=tmp_path,
        c_std="c11",
    )
    extracted = engine.extract_modules()

    assert "empty.mod" in extracted
    assert extracted["empty.mod"].functions == {}


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

    signatures = extracted["cross.func"].functions["foo"].signatures
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

    signatures = extracted["cross.methods"].functions["foo"].signatures
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
    assert extracted["inline.mod"].functions["foo"].signatures == [
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
    assert extracted["conflict.mod"].functions["foo"].signatures == [
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
    assert extracted["missing.mod"].functions == {}


def test_c_signature_engine_extract_modules_keeps_external_include_options_and_injects_python_include_dirs(
    tmp_path: Path,
) -> None:
    engine = CSignatureExtractor(
        source=tmp_path,
        include=["Python.h"],
        include_directory=[Path("C:/MyInclude")],
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(translation_unit_module, "list_files", lambda source: [])

    try:
        assert engine.extract_modules() == {}
        expected_include_dirs = [Path("C:/MyInclude")]
        for include_dir in [sysconfig.get_path("include"), sysconfig.get_path("platinclude")]:
            if not include_dir:
                continue
            include_path = Path(include_dir)
            if include_path in expected_include_dirs:
                continue
            expected_include_dirs.append(include_path)

        assert engine._include == ["Python.h"]
        assert engine._include_directory == expected_include_dirs
    finally:
        monkeypatch.undo()


def test_c_signature_extraction_engine_logs_exception_and_continues_next_function(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bad_cursor = object()
    good_cursor = object()
    bad_function = ExtractedFunction(ml_name="bad", function_cursor=bad_cursor)
    good_function = ExtractedFunction(ml_name="good", function_cursor=good_cursor)
    logged_messages: list[str] = []

    monkeypatch.setattr(
        c_signature_extraction_module.clang_parser,
        "inject_python_include_directories",
        lambda include_directory: list(include_directory),
    )
    monkeypatch.setattr(
        c_signature_extraction_module.clang_parser,
        "list_files",
        lambda source: [source / "sample.c"],
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
            cursor=_FakeNode(kind=clang.cindex.CursorKind.TRANSLATION_UNIT)
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

    def _infer_signature(function_cursor: object) -> list[ExtractedSignature]:
        if function_cursor is bad_cursor:
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

    extracted = c_signature_extraction_module.extract_c_signature_modules(tmp_path)

    assert extracted["pkg.mod"].functions["bad"].signatures == []
    assert extracted["pkg.mod"].functions["good"].signatures == [
        ExtractedSignature(return_type=RawType("int"))
    ]
    assert len(logged_messages) == 1
    assert "pkg.mod" in logged_messages[0]
    assert "bad" in logged_messages[0]


def test_c_signature_extraction_engine_logs_exception_and_continues_next_function(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bad_cursor = object()
    good_cursor = object()
    bad_function = ExtractedFunction(ml_name="bad", function_cursor=bad_cursor)
    good_function = ExtractedFunction(ml_name="good", function_cursor=good_cursor)
    logged_messages: list[str] = []

    monkeypatch.setattr(
        c_signature_extraction_module.clang_parser,
        "inject_python_include_directories",
        lambda include_directory: list(include_directory),
    )
    monkeypatch.setattr(
        c_signature_extraction_module.clang_parser,
        "list_files",
        lambda source: [source / "sample.c"],
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
            cursor=_FakeNode(kind=clang.cindex.CursorKind.TRANSLATION_UNIT)
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

    def _infer_signature(function_cursor: object) -> list[ExtractedSignature]:
        if function_cursor is bad_cursor:
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

    extracted = c_signature_extraction_module.extract_c_signature_modules(tmp_path)

    assert extracted["pkg.mod"].functions["bad"].signatures == []
    assert extracted["pkg.mod"].functions["good"].signatures == [
        ExtractedSignature(return_type=RawType("int"))
    ]
    assert len(logged_messages) == 1
    assert "pkg.mod" in logged_messages[0]
    assert "bad" in logged_messages[0]


