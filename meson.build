project(
  'pcstubgen',
  ['cpp', 'c'],
  version : '0.1.0',
  default_options : ['cpp_std=c++17'],
)

py = import('python').find_installation(pure : false)

llvm_dep = dependency(
  'llvm',
  version : '>=18',
  modules : ['object', 'debuginfodwarf', 'support'],
  required : true,
)

py.extension_module(
  '_dwarfdump',
  ['pcstubgen/native/dwarfdump_module.cpp'],
  dependencies : [llvm_dep],
  subdir : 'pcstubgen/signature_completion/c_extension',
  install : true,
)

install_subdir(
  'pcstubgen',
  install_dir : py.get_install_dir(pure : false),
  exclude_directories : ['__pycache__'],
)
