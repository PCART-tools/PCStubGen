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
#include <memory>
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
                                       llvm::StringRef unit_name);

class DwarfFile {
public:
    static llvm::Expected<std::unique_ptr<DwarfFile>> create(llvm::StringRef binary_path) {
        auto object_file = llvm::object::ObjectFile::createObjectFile(binary_path);
        if (!object_file) {
            return object_file.takeError();
        }

        auto dwarf_context = llvm::DWARFContext::create(*object_file->getBinary());
        return std::unique_ptr<DwarfFile>(
            new DwarfFile(binary_path.str(), std::move(*object_file), std::move(dwarf_context)));
    }

    llvm::Expected<LookupResult> lookup(std::uint64_t relative_address) {
        const auto compile_units = dwarf_context_->compile_units();
        if (compile_units.begin() == compile_units.end()) {
            return llvm::createStringError(
                llvm::inconvertibleErrorCode(),
                "共享库缺少DWARF调试信息: %s",
                binary_path_.c_str());
        }

        auto *compile_unit = dwarf_context_->getCompileUnitForCodeAddress(relative_address);
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
                binary_path_.c_str());
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

private:
    DwarfFile(
        std::string binary_path,
        llvm::object::OwningBinary<llvm::object::ObjectFile> object_file,
        std::unique_ptr<llvm::DWARFContext> dwarf_context)
        : binary_path_(std::move(binary_path)),
          object_file_(std::move(object_file)),
          dwarf_context_(std::move(dwarf_context)) {}

    std::string binary_path_;
    llvm::object::OwningBinary<llvm::object::ObjectFile> object_file_;
    std::unique_ptr<llvm::DWARFContext> dwarf_context_;
};

struct DwarfFileObject {
    PyObject_HEAD
    DwarfFile *file;
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

int dwarf_file_init(DwarfFileObject *self, PyObject *args, PyObject *) {
    const char *binary_path = nullptr;
    if (!PyArg_ParseTuple(args, "s:DWARFFile", &binary_path)) {
        return -1;
    }

    auto file = DwarfFile::create(binary_path);
    if (!file) {
        const auto message = llvm::toString(file.takeError());
        PyErr_SetString(PyExc_RuntimeError, message.c_str());
        return -1;
    }

    self->file = file->release();
    return 0;
}

void dwarf_file_dealloc(DwarfFileObject *self) {
    delete self->file;
    PyTypeObject *type = Py_TYPE(self);
    type->tp_free(reinterpret_cast<PyObject *>(self));
}

PyObject *dwarf_file_lookup(DwarfFileObject *self, PyObject *args) {
    unsigned long long relative_address = 0;
    if (!PyArg_ParseTuple(args, "K:lookup", &relative_address)) {
        return nullptr;
    }

    auto result = self->file->lookup(static_cast<std::uint64_t>(relative_address));
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

PyMethodDef dwarf_file_methods[] = {
    {
        "lookup",
        reinterpret_cast<PyCFunction>(dwarf_file_lookup),
        METH_VARARGS,
        "按共享库内相对地址查询编译单元路径与函数身份。",
    },
    {nullptr, nullptr, 0, nullptr},
};

PyTypeObject dwarf_file_type = {
    PyVarObject_HEAD_INIT(nullptr, 0)
};

PyMethodDef methods[] = {
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
    dwarf_file_type.tp_name = "_dwarfdump.DWARFFile";
    dwarf_file_type.tp_basicsize = sizeof(DwarfFileObject);
    dwarf_file_type.tp_dealloc = reinterpret_cast<destructor>(dwarf_file_dealloc);
    dwarf_file_type.tp_flags = Py_TPFLAGS_DEFAULT;
    dwarf_file_type.tp_doc = "持有单个共享库 DWARF 上下文的查询对象。";
    dwarf_file_type.tp_methods = dwarf_file_methods;
    dwarf_file_type.tp_init = reinterpret_cast<initproc>(dwarf_file_init);
    dwarf_file_type.tp_new = PyType_GenericNew;

    if (PyType_Ready(&dwarf_file_type) < 0) {
        return nullptr;
    }

    PyObject *module_obj = PyModule_Create(&module);
    if (module_obj == nullptr) {
        return nullptr;
    }

    if (PyModule_AddType(module_obj, &dwarf_file_type) < 0) {
        Py_DECREF(module_obj);
        return nullptr;
    }

    return module_obj;
}
