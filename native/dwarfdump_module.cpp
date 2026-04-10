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
#include <optional>
#include <string>

namespace {

struct LookupResult {
  std::string compilation_unit_path;
  std::string function_name;
  std::optional<std::string> linkage_name;
};

PyObject *set_runtime_error(const std::string &message) {
  PyErr_SetString(PyExc_RuntimeError, message.c_str());
  return nullptr;
}

std::string join_compilation_unit_path(llvm::DWARFUnit &unit,
                                       const char *unit_name) {
  llvm::SmallString<256> full_path(unit_name == nullptr ? "" : unit_name);
  if (!llvm::sys::path::is_absolute(full_path)) {
    if (const char *compilation_dir = unit.getCompilationDir();
        compilation_dir != nullptr && compilation_dir[0] != '\0') {
      llvm::SmallString<256> prefixed_path(compilation_dir);
      llvm::sys::path::append(prefixed_path, unit_name == nullptr ? "" : unit_name);
      full_path = prefixed_path;
    }
  }
  return std::string(full_path.str());
}

llvm::Expected<LookupResult> lookup_with_llvm(llvm::StringRef binary_path,
                                              uint64_t relative_address) {
  llvm::Expected<llvm::object::OwningBinary<llvm::object::ObjectFile>>
      object_file = llvm::object::ObjectFile::createObjectFile(binary_path);
  if (!object_file) {
    return object_file.takeError();
  }

  llvm::object::ObjectFile *object = object_file->getBinary();
  auto dwarf_context = llvm::DWARFContext::create(*object);
  auto compile_units = dwarf_context->compile_units();
  if (compile_units.begin() == compile_units.end()) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "共享库缺少DWARF调试信息: %s",
        binary_path.str().c_str());
  }

  llvm::DWARFCompileUnit *compile_unit =
      dwarf_context->getCompileUnitForCodeAddress(relative_address);
  if (compile_unit == nullptr) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "DWARF 中未找到地址所属编译单元: 0x%llx",
        static_cast<unsigned long long>(relative_address));
  }

  llvm::DWARFDie unit_die = compile_unit->getUnitDIE();
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

  llvm::DWARFDie subprogram = compile_unit->getSubroutineForAddress(relative_address);
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

  LookupResult result;
  result.compilation_unit_path = join_compilation_unit_path(*compile_unit, unit_name);
  result.function_name = function_name;
  result.linkage_name = std::nullopt;

  if (const char *linkage_name = subprogram.getLinkageName();
      linkage_name != nullptr && linkage_name[0] != '\0') {
    result.linkage_name = std::string(linkage_name);
  }

  return result;
}

PyObject *lookup_raw(PyObject *, PyObject *args) {
  const char *binary_path = nullptr;
  unsigned long long relative_address = 0;
  if (!PyArg_ParseTuple(args, "sK:lookup_raw", &binary_path, &relative_address)) {
    return nullptr;
  }

  llvm::Expected<LookupResult> result =
      lookup_with_llvm(binary_path, static_cast<uint64_t>(relative_address));
  if (!result) {
    return set_runtime_error(llvm::toString(result.takeError()));
  }

  PyObject *path_obj = PyUnicode_FromString(result->compilation_unit_path.c_str());
  if (path_obj == nullptr) {
    return nullptr;
  }
  PyObject *function_name_obj = PyUnicode_FromString(result->function_name.c_str());
  if (function_name_obj == nullptr) {
    Py_DECREF(path_obj);
    return nullptr;
  }

  PyObject *linkage_name_obj = Py_None;
  Py_INCREF(Py_None);
  if (result->linkage_name.has_value()) {
    Py_DECREF(linkage_name_obj);
    linkage_name_obj = PyUnicode_FromString(result->linkage_name->c_str());
    if (linkage_name_obj == nullptr) {
      Py_DECREF(path_obj);
      Py_DECREF(function_name_obj);
      return nullptr;
    }
  }

  PyObject *tuple =
      PyTuple_Pack(3, path_obj, function_name_obj, linkage_name_obj);
  Py_DECREF(path_obj);
  Py_DECREF(function_name_obj);
  Py_DECREF(linkage_name_obj);
  return tuple;
}

PyMethodDef module_methods[] = {
    {
        "lookup_raw",
        lookup_raw,
        METH_VARARGS,
        "按共享库路径和库内相对地址查询编译单元路径与函数身份。",
    },
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT,
    "_dwarfdump_llvm",
    "基于 LLVM 的 dwarfdump 查询扩展。",
    -1,
    module_methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

} // namespace

PyMODINIT_FUNC PyInit__dwarfdump_llvm(void) {
  return PyModule_Create(&module_definition);
}
