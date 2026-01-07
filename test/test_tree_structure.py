import sys
import os

# Add current directory to sys.path to find stubgen_api
sys.path.insert(0, os.getcwd())

from package_stubgen import build_package_tree

def print_tree(data, indent=0):
    prefix = "  " * indent
    print(f"{prefix}Module: {data.name}")
    
    if data.imports:
        print(f"{prefix}  Imports: {len(data.imports)}")
        for imp in data.imports:
            print(f"{prefix}    - {imp}")
            
    if data.variables:
        print(f"{prefix}  Variables: {len(data.variables)}")
        for var in data.variables:
            print(f"{prefix}    - {var.name}: {var.type}")
            
    if data.functions:
        print(f"{prefix}  Functions: {len(data.functions)}")
        for func in data.functions:
            print(f"{prefix}    - {func.name}{func.format_sig()}")
            
    if data.classes:
        print(f"{prefix}  Classes: {len(data.classes)}")
        for cls in data.classes:
            bases_str = f"({', '.join(cls.bases)})" if cls.bases else ""
            print(f"{prefix}    - class {cls.name}{bases_str}")
            if cls.methods:
                for m in cls.methods:
                    print(f"{prefix}      * method: {m.name}{m.format_sig()}")
            if cls.variables:
                for v in cls.variables:
                    print(f"{prefix}      * var: {v.name}: {v.type}")

    if data.submodules:
        print(f"{prefix}  Submodules: {len(data.submodules)}")
        for sub in data.submodules:
            print_tree(sub, indent + 2)

def test_stubgen_api_tree():
    # 使用一个有子模块的包进行测试，例如 numpy (如果安装了) 或者创建一个模拟包
    # 这里我们先尝试 numpy._core
    package = "numpy"
    try:
        import numpy
        print(f"Generating stubs for package: {package}")
        root = build_package_tree(package)
        
        print("\n--- Module Tree Structure ---")
        print_tree(root)
    except ImportError:
        print("Numpy not found, testing with math (single module)")
        root = build_package_tree("math")
        print_tree(root)

if __name__ == "__main__":
    test_stubgen_api_tree()
