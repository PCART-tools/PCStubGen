# PCStubGen

[English](README.md)

为 Python 模块生成 `.pyi` stub。
PCStubGen 基于 docstring 解析支持 pybind11 扩展，基于 libclang C 源码解析支持 Python C 扩展。

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

## 开发

```bash
sudo apt install llvm bear
git clone https://github.com/PCART-tools/PCStubGen
cd PCStubGen
uv sync --no-build-isolation
```
