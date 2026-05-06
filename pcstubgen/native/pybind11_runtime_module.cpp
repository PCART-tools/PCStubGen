#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

constexpr const char *pybind11_v2_capsule_name = "pybind11_function_record_capsule";
constexpr const char *pybind11_v3_tp_plain_prefix = "pybind11_detail_function_record_v1_";
constexpr std::size_t pybind11_v3_tp_plain_prefix_len
    = sizeof("pybind11_detail_function_record_v1_") - 1;
constexpr const char *pybind11_v3_tp_qualified_prefix
    = "pybind11_builtins.pybind11_detail_function_record_v1_";
constexpr std::size_t pybind11_v3_tp_qualified_prefix_len
    = sizeof("pybind11_builtins.pybind11_detail_function_record_v1_") - 1;
constexpr int max_overloads = 256;

struct function_call;

struct handle {
    PyObject *ptr;
};

enum class return_value_policy : int {};

struct argument_record {
    const char *name;
    const char *descr;
    handle value;
    bool convert : 1;
    bool none : 1;
};

struct function_record {
    char *name = nullptr;
    char *doc = nullptr;
    char *signature = nullptr;
    std::vector<argument_record> args;
    handle (*impl)(function_call &) = nullptr;
    void *data[3] = {};
    void (*free_data)(function_record *ptr) = nullptr;
    return_value_policy policy = static_cast<return_value_policy>(0);
    bool is_constructor : 1;
    bool is_new_style_constructor : 1;
    bool is_stateless : 1;
    bool is_operator : 1;
    bool is_method : 1;
    bool is_setter : 1;
    bool has_args : 1;
    bool has_kwargs : 1;
    bool prepend : 1;
    std::uint16_t nargs;
    std::uint16_t nargs_pos = 0;
    std::uint16_t nargs_pos_only = 0;
    PyMethodDef *def = nullptr;
    handle scope;
    handle sibling;
    function_record *next = nullptr;
};

struct function_record_PyObject {
    PyObject_HEAD
    function_record *cpp_func_rec;
};

function_record *read_capsule_function_record(PyObject *self) {
    if (!PyCapsule_CheckExact(self)) {
        return nullptr;
    }

    auto *record = reinterpret_cast<function_record *>(
        PyCapsule_GetPointer(self, pybind11_v2_capsule_name));
    if (record == nullptr) {
        PyErr_Clear();
    }
    return record;
}

function_record *read_v3_function_record(PyObject *self) {
    PyTypeObject *type = Py_TYPE(self);
    if (type == nullptr || type->tp_name == nullptr) {
        return nullptr;
    }
    const bool matches_plain = std::strncmp(
                                   type->tp_name,
                                   pybind11_v3_tp_plain_prefix,
                                   pybind11_v3_tp_plain_prefix_len)
                               == 0;
    const bool matches_qualified = std::strncmp(
                                       type->tp_name,
                                       pybind11_v3_tp_qualified_prefix,
                                       pybind11_v3_tp_qualified_prefix_len)
                                   == 0;
    if (!matches_plain && !matches_qualified) {
        return nullptr;
    }

    auto *record_object = reinterpret_cast<function_record_PyObject *>(self);
    return record_object->cpp_func_rec;
}

function_record *read_function_record(PyObject *self) {
    if (function_record *record = read_v3_function_record(self)) {
        return record;
    }

    if (function_record *record = read_capsule_function_record(self)) {
        return record;
    }

    return nullptr;
}

PyObject *is_pybind11_self(PyObject *, PyObject *self) {
    if (read_function_record(self) != nullptr) {
        Py_RETURN_TRUE;
    }

    Py_RETURN_FALSE;
}

PyObject *extract_signatures(PyObject *, PyObject *obj) {
    PyObject *func = obj;
    if (PyInstanceMethod_Check(func)) {
        func = PyInstanceMethod_GET_FUNCTION(func);
    }

    if (!PyCFunction_Check(func)) {
        PyErr_SetString(PyExc_RuntimeError, "目标对象不是 PyCFunction。");
        return nullptr;
    }

    PyObject *self = PyCFunction_GET_SELF(func);
    if (self == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "PyCFunction 缺少 self。");
        return nullptr;
    }

    function_record *record = read_function_record(self);
    if (record == nullptr) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "不支持的 pybind11 self 布局，无法读取 function_record。");
        return nullptr;
    }

    PyObject *result = PyList_New(0);
    if (result == nullptr) {
        return nullptr;
    }

    int overload_count = 0;
    for (function_record *it = record; it != nullptr && overload_count < max_overloads;
         it = it->next, overload_count++) {
        if (it->signature == nullptr) {
            Py_DECREF(result);
            PyErr_SetString(PyExc_RuntimeError, "pybind11 overload 签名指针为空。");
            return nullptr;
        }
        if (it->signature[0] == '\0') {
            Py_DECREF(result);
            PyErr_SetString(PyExc_RuntimeError, "pybind11 overload 签名文本为空。");
            return nullptr;
        }

        PyObject *signature = PyUnicode_FromString(it->signature);
        if (signature == nullptr) {
            Py_DECREF(result);
            return nullptr;
        }

        if (PyList_Append(result, signature) < 0) {
            Py_DECREF(signature);
            Py_DECREF(result);
            return nullptr;
        }

        Py_DECREF(signature);
    }

    assert(PyList_GET_SIZE(result) > 0);

    return result;
}

PyMethodDef methods[] = {
    {
        "is_pybind11_self",
        reinterpret_cast<PyCFunction>(is_pybind11_self),
        METH_O,
        "判断 self 对象是否为 pybind11 function_record 运行时承载对象。",
    },
    {
        "extract_signatures",
        reinterpret_cast<PyCFunction>(extract_signatures),
        METH_O,
        "提取 pybind11 overload chain 上的单条签名文本。",
    },
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_pybind11_runtime",
    "pybind11 runtime signature extractor.",
    -1,
    methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

} // namespace

PyMODINIT_FUNC PyInit__pybind11_runtime(void) {
    return PyModule_Create(&module);
}
