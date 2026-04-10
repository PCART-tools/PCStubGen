cmake_minimum_required(VERSION 3.30)

project(pcstubgen LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

find_program(LLVM_CONFIG_EXECUTABLE NAMES llvm-config REQUIRED)

function(run_llvm_config OUTPUT_VARIABLE)
    execute_process(
        COMMAND "${LLVM_CONFIG_EXECUTABLE}" ${ARGN}
        OUTPUT_VARIABLE LLVM_CONFIG_OUTPUT
        ERROR_VARIABLE LLVM_CONFIG_ERROR
        RESULT_VARIABLE LLVM_CONFIG_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT LLVM_CONFIG_RESULT EQUAL 0)
        message(
            FATAL_ERROR
            "调用 llvm-config 失败: ${ARGN}\n${LLVM_CONFIG_ERROR}"
        )
    endif()
    set(${OUTPUT_VARIABLE} "${LLVM_CONFIG_OUTPUT}" PARENT_SCOPE)
endfunction()

run_llvm_config(LLVM_INCLUDEDIR --includedir)
run_llvm_config(LLVM_LIBDIR --libdir)
run_llvm_config(LLVM_CXXFLAGS_RAW --cxxflags)
run_llvm_config(LLVM_LIBS_RAW --libs all)
run_llvm_config(LLVM_SYSTEM_LIBS_RAW --system-libs)

separate_arguments(LLVM_CXXFLAGS NATIVE_COMMAND "${LLVM_CXXFLAGS_RAW}")
separate_arguments(LLVM_LIBS NATIVE_COMMAND "${LLVM_LIBS_RAW}")
separate_arguments(LLVM_SYSTEM_LIBS NATIVE_COMMAND "${LLVM_SYSTEM_LIBS_RAW}")

Python_add_library(_dwarfdump_llvm MODULE WITH_SOABI native/dwarfdump_module.cpp)
target_compile_features(_dwarfdump_llvm PRIVATE cxx_std_17)
target_include_directories(_dwarfdump_llvm PRIVATE "${LLVM_INCLUDEDIR}")
target_compile_options(_dwarfdump_llvm PRIVATE ${LLVM_CXXFLAGS})
target_link_directories(_dwarfdump_llvm PRIVATE "${LLVM_LIBDIR}")
target_link_libraries(_dwarfdump_llvm PRIVATE ${LLVM_LIBS} ${LLVM_SYSTEM_LIBS})

install(
    TARGETS _dwarfdump_llvm
    LIBRARY DESTINATION pcstubgen/signature_completion/c_extension
)
