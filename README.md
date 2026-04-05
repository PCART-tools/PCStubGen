# PCStubGen
为 Python 模块生成 `.pyi` stub。
基于docstring解析支持 pybind11 扩展，基于libclang C源码解析支持Python C扩展。

## 使用
生成 stub:

```bash
uv run pcstubgen gen numpy --output ./stubs
uv run pcstubgen gen pandas._libs.lib --compilation-database ./build/compile_commands.json
```

构建 wheel 并生成 `compile_commands.json`:

```bash
uv run pcstubgen build /path/to/python-project
```
