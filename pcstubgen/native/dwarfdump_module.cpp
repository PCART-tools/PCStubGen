#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace {

struct LookupResult {
    std::string compilation_unit_path;
    std::string function_name;
    std::optional<std::string> linkage_name;
};

std::string join_compilation_unit_path(llvm::DWARFUnit &unit,
                                       llvm::StringRef unit_name) {
    llvm::SmallString<256> full_path(unit_name);
    if (!llvm::sys::path::is_absolute(full_path)) {
        if (const char *compilation_dir = unit.getCompilationDir();
            compilation_dir != nullptr && compilation_dir[0] != '\0') {
            llvm::SmallString<256> prefixed_path(compilation_dir);
            llvm::sys::path::append(prefixed_path, unit_name);
            full_path = prefixed_path;
        }
    }
    return std::string(full_path.str());
}

llvm::Expected<LookupResult> lookup(llvm::StringRef binary_path,
                                    std::uint64_t relative_address) {
    auto object_file = llvm::object::ObjectFile::createObjectFile(binary_path);
    if (!object_file) {
        return object_file.takeError();
    }

    auto *object = object_file->getBinary();
    auto dwarf_context = llvm::DWARFContext::create(*object);
    const auto compile_units = dwarf_context->compile_units();
    if (compile_units.begin() == compile_units.end()) {
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "共享库缺少DWARF调试信息: %s",
            binary_path.str().c_str());
    }

    auto *compile_unit = dwarf_context->getCompileUnitForCodeAddress(relative_address);
    if (compile_unit == nullptr) {
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "DWARF 中未找到地址所属编译单元: 0x%llx",
            static_cast<unsigned long long>(relative_address));
    }

    const auto unit_die = compile_unit->getUnitDIE();
    if (!unit_die.isValid()) {
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "共享库缺少DWARF调试信息: %s",
            binary_path.str().c_str());
    }

    const char *unit_name = unit_die.getShortName();
    if (unit_name == nullptr || unit_name[0] == '\0') {
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "DWARF 编译单元缺少源码路径: 0x%llx",
            static_cast<unsigned long long>(relative_address));
    }

    const auto subprogram = compile_unit->getSubroutineForAddress(relative_address);
    if (!subprogram.isValid()) {
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "DWARF 中未找到地址对应的函数: 0x%llx",
            static_cast<unsigned long long>(relative_address));
    }

    const char *function_name = subprogram.getShortName();
    if (function_name == nullptr || function_name[0] == '\0') {
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "DWARF 函数缺少名称: 0x%llx",
            static_cast<unsigned long long>(relative_address));
    }

    std::optional<std::string> linkage_name_value;
    if (const char *linkage_name = subprogram.getLinkageName();
        linkage_name != nullptr && linkage_name[0] != '\0') {
        linkage_name_value = std::string(linkage_name);
    }

    return LookupResult{
        join_compilation_unit_path(*compile_unit, unit_name),
        std::string(function_name),
        std::move(linkage_name_value),
    };
}

PyObject *lookup_wrap(PyObject *, PyObject *args) {
    const char *binary_path = nullptr;
    unsigned long long relative_address = 0;
    if (!PyArg_ParseTuple(args, "sK:lookup", &binary_path, &relative_address)) {
        return nullptr;
    }

    auto result = lookup(binary_path, static_cast<std::uint64_t>(relative_address));
    if (!result) {
        const auto message = llvm::toString(result.takeError());
        PyErr_SetString(PyExc_RuntimeError, message.c_str());
        return nullptr;
    }

    PyObject *path_obj = PyUnicode_FromString(result->compilation_unit_path.c_str());
    if (!path_obj) {
        return nullptr;
    }

    PyObject *function_name_obj = PyUnicode_FromString(result->function_name.c_str());
    if (!function_name_obj) {
        Py_DECREF(path_obj);
        return nullptr;
    }

    PyObject *linkage_name_obj = nullptr;
    if (result->linkage_name.has_value()) {
        linkage_name_obj = PyUnicode_FromString(result->linkage_name->c_str());
        if (!linkage_name_obj) {
            Py_DECREF(path_obj);
            Py_DECREF(function_name_obj);
            return nullptr;
        }
    } else {
        Py_INCREF(Py_None);
        linkage_name_obj = Py_None;
    }

    PyObject *tuple = PyTuple_Pack(3, path_obj, function_name_obj, linkage_name_obj);
    Py_DECREF(path_obj);
    Py_DECREF(function_name_obj);
    Py_DECREF(linkage_name_obj);
    return tuple;
}

PyMethodDef methods[] = {
    {
        "lookup",
        lookup_wrap,
        METH_VARARGS,
        "按共享库路径和库内相对地址查询编译单元路径与函数身份。",
    },
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_dwarfdump",
    "基于 LLVM 的 dwarfdump 查询扩展。",
    -1,
    methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

} // namespace

PyMODINIT_FUNC PyInit__dwarfdump(void) {
    return PyModule_Create(&module);
}
