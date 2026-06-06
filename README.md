# PCStubGen: Generate Python Stubs for C Extension APIs

[![中文](https://img.shields.io/badge/lang-%E4%B8%AD%E6%96%87-green.svg)](README.zh.md)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE.txt)

## What PCStubGen Can Do

- Generate stubs for extensions based on the Python/C API by analyzing C/C++ source ASTs
- Generate stubs for extensions based on pybind11 by parsing signature strings

## Installation

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

1. Build the target project

   See the [system-level dependencies and notes](SYSTEM_LEVEL_DEPS_REF_AND_NOTES.md) for some target projects.

   ```bash
   uv run pcstubgen build <target-project-directory>
   ```

   After a successful build, the command outputs the paths to the wheel and `compile_commands.json`.

2. Install the wheel into the current environment

   ```bash
   uv pip install <wheel-path>
   ```

3. Generate stubs

   ```bash
   uv run pcstubgen gen <target-project-python-package-name> --compilation-database <compile-commands-path>
   ```

## Compatibility

PCStubGen has been developed and tested on **Ubuntu 24.04.2 LTS** with **Python 3.12** and **LLVM 18**.

It should work on Linux and macOS.
Windows support is currently limited because PCStubGen only supports DWARF symbols, and building target projects on Windows is more challenging.

## License

PCStubGen is licensed under the Apache License 2.0. See [LICENSE.txt](./LICENSE.txt) for details.
