# PCStubGen：为 C 扩展 API 生成 Python 存根

[![English](https://img.shields.io/badge/lang-English-green.svg)](README.md)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE.txt)

## PCStubGen 可以做什么

- 通过分析 C/C++ 源码 AST，为基于 **Python/C API** 的扩展 API 生成存根
- 通过解析签名字符串，为基于 **pybind11** 的扩展 API 生成存根

## 安装

推荐使用 [uv](https://docs.astral.sh/uv/) 进行快速、可复现的环境配置。

```bash
# 克隆仓库
git clone https://github.com/PCART-tools/PCStubGen.git
cd PCStubGen

# 安装系统级依赖
sudo apt install llvm bear

# 同步 Python 环境
uv sync --no-build-isolation
```

## 使用

1. 构建目标项目

   一些目标项目的[系统级依赖和注意事项](SYSTEM_LEVEL_DEPS_REF_AND_NOTES.md)。

   ```bash
   uv run pcstubgen build <目标项目目录>
   ```

   构建成功后，命令会输出 wheel 和 `compile_commands.json` 的路径。

2. 将 wheel 安装到当前环境

   ```bash
   uv pip install <wheel 路径>
   ```

3. 生成存根

   ```bash
   uv run pcstubgen gen <目标 Python 库名> --compilation-database <compile_commands.json 路径>
   ```

## 兼容性

PCStubGen 在 **Ubuntu 24.04.2 LTS**、**Python 3.12** 和 **LLVM 18** 环境下开发并测试。

理论上可在 Linux 和 macOS 上运行。
当前对 Windows 的支持较为有限，因为 PCStubGen 目前只支持 DWARF 符号，且在 Windows 上构建目标项目更具挑战。

## 许可证

PCStubGen 采用 Apache License 2.0 授权。详情请参见 [LICENSE.txt](./LICENSE.txt)。
