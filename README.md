# PCStubGen - Generate Python Stubs for C Extension APIs

[中文](README.zh.md)

- Supports extensions based on the Python/C API through libclang source AST analysis
- Supports extensions based on pybind11 through signature string parsing

## Installation

```bash
sudo apt install llvm bear
```

## Usage

1. Build the target project to produce `compile_commands.json` and a wheel package with debug symbols.

   - Use the built-in build command:

   ```bash
   uv run pcstubgen build <target-project-directory>
   ```

   - Or wrap a custom build command:

   ```bash
   cd <target-project-directory>
   uv run pcstubgen wrap -- <build-command>
   ```

2. Install the wheel package into the environment.

3. Generate stubs:

   ```bash
   uv run pcstubgen gen <python-module-name> --compilation-database ./build/compile_commands.json
   ```

## Development

```bash
sudo apt install llvm bear
git clone https://github.com/PCART-tools/PCStubGen
cd PCStubGen
uv sync --no-build-isolation
```

## Compatibility

PCStubGen has been developed and tested on **Ubuntu 24.04.2 LTS** with **Python 3.12** and **LLVM 18**.

It should work on Linux and macOS. Windows support is currently limited because PCStubGen only supports DWARF symbols, and building target projects on Windows is more challenging.

## Example Projects

### [SciPy](https://github.com/scipy/scipy)

Install build dependencies:

```bash
sudo apt install gfortran libopenblas-dev liblapack-dev pkg-config
```

### [Pillow](https://github.com/python-pillow/Pillow)

Install build dependencies:

```bash
sudo apt install libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev \
    libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk \
    libharfbuzz-dev libfribidi-dev libxcb1-dev
```

### [NumPy](https://github.com/numpy/numpy)

Install build dependencies:

```bash
sudo apt install gfortran libopenblas-dev liblapack-dev pkg-config
```

### [psycopg2](https://github.com/psycopg/psycopg2)

Install build dependencies:

```bash
sudo apt-get install libpq-dev
```

### [UltraJSON](https://github.com/ultrajson/ultrajson)

### [PyTorch](https://github.com/pytorch/pytorch)

Install build dependencies:

```bash
sudo apt install libomp-dev
```

## License

PCStubGen is licensed under the Apache License 2.0. See [LICENSE.txt](./LICENSE.txt) for details.
