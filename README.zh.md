# PCStubGen - 从 C 源码生成 Python Stub

[English](README.md)

PCStubGen 基于 libclang 解析 C 源码支持 Python C 扩展，基于 docstring 解析支持 pybind11 扩展。

## 安装

```bash
sudo apt install llvm bear
```

## 使用

1. 构建目标项目，得到 `compile_commands.json` 和带调试符号的 wheel 包。

   - 使用自带构建命令：

   ```bash
   uv run pcstubgen build <目标项目目录>
   ```

   - 或者，使用自定义构建命令：

   ```bash
   cd <目标项目目录>
   uv run pcstubgen wrap -- <构建命令>
   ```

2. 安装 wheel 包到环境。

3. 生成 stub：

   ```bash
   uv run pcstubgen gen <项目 Python 内名> --compilation-database ./build/compile_commands.json
   ```

## 开发

```bash
sudo apt install llvm bear
git clone https://github.com/PCART-tools/PCStubGen
cd PCStubGen
uv sync --no-build-isolation
```

## 兼容性

PCStubGen 在 **Ubuntu 24.04.2 LTS**、**Python 3.12** 和 **LLVM 18** 环境下开发并测试。

理论上可运行于 Linux 和 macOS。当前对 Windows 的支持较为有限，因为 PCStubGen 目前只支持 DWARF 符号，且在 Windows 上构建目标项目更具挑战。

## 样例项目

### [SciPy](https://github.com/scipy/scipy)

安装构建依赖：

```bash
sudo apt install gfortran libopenblas-dev liblapack-dev pkg-config
```

### [Pillow](https://github.com/python-pillow/Pillow)

安装构建依赖：

```bash
sudo apt install libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev \
    libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk \
    libharfbuzz-dev libfribidi-dev libxcb1-dev
```

### [NumPy](https://github.com/numpy/numpy)

安装构建依赖：

```bash
sudo apt install gfortran libopenblas-dev liblapack-dev pkg-config
```

### [psycopg2](https://github.com/psycopg/psycopg2)

安装构建依赖：

```bash
sudo apt-get install libpq-dev
```

### [UltraJSON](https://github.com/ultrajson/ultrajson)

### [PyTorch](https://github.com/pytorch/pytorch)

安装构建依赖：

```bash
sudo apt install libomp-dev
```
