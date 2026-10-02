# PCStubGen

[![中文](https://img.shields.io/badge/lang-%E4%B8%AD%E6%96%87-green.svg)](README.zh.md)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE.txt)

**Stub generation for Python C extension APIs to support cross-version interface comparison.**

## What is PCStubGen?

PCStubGen is a tool for generating Python stubs for C extension APIs to support cross-version interface comparison.

PCStubGen currently supports:

- extensions based on the **Python/C API**, by analyzing C/C++ source ASTs;
- extensions based on **pybind11**, by parsing signature strings.

## Quick Start

We recommend using [uv](https://docs.astral.sh/uv/) for fast, reproducible environment setup.

```bash
# Clone the repository
git clone https://github.com/PCART-tools/PCStubGen.git
cd PCStubGen

# Install system-level dependencies
sudo apt install llvm bear

# Sync the Python environment
uv sync --no-build-isolation
```

## Usage

### Step 1: Build the Target Project

See the [system-level dependencies and notes](SYSTEM_LEVEL_DEPS_REF_AND_NOTES.md) for setup requirements of some target projects.

```bash
uv run pcstubgen build <target-project-directory>
```

After a successful build, the command outputs the paths to the wheel and `compile_commands.json`.

### Step 2: Install the Wheel

```bash
uv pip install <wheel-path>
```

### Step 3: Generate Stubs

```bash
uv run pcstubgen gen <target-python-package-name> --compilation-database <compile_commands.json-path>
```

## Compatibility

PCStubGen has been developed and tested on **Ubuntu 24.04.2 LTS** with **Python 3.12** and **LLVM 18**.

It should work on Linux and macOS. Windows support is currently limited because PCStubGen only supports DWARF symbols, and building target projects on Windows is more challenging.

## License

PCStubGen is licensed under the Apache License 2.0. See [LICENSE.txt](./LICENSE.txt) for details.
