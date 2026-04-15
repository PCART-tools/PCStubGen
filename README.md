# PCStubGen
为 Python 模块生成 `.pyi` stub。
基于docstring解析支持 pybind11 扩展，基于libclang C源码解析支持Python C扩展。

## 使用
生成 stub:

```bash
uv run pcstubgen gen numpy --compilation-database ./build/compile_commands.json --output ./stubs
uv run pcstubgen gen pandas._libs.lib --compilation-database ./build/compile_commands.json
```

输出 TOML 格式的结构化函数记录:

```bash
uv run pcstubgen gen pandas._libs.lib --compilation-database ./build/compile_commands.json --output ./stubs --toml
```

以前缀包装器运行原始构建命令，并产出 `compile_commands.json`:

```bash
cd /path/to/python-project
uv run pcstubgen wrap -- python -m build --wheel
uv run pcstubgen wrap --output ./out/compile_commands.json -- uv build --wheel
```

构建 Python 项目，并为 stub 生成产出 `compile_commands.json`:

```bash
uv run pcstubgen build /path/to/python-project
```
