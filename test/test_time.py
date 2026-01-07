import sys
import os

# Add current directory to sys.path to find stubgen_api
sys.path.insert(0, os.getcwd())

from package_stubgen import build_package_tree

def test_stubgen_api():
    packages = ["math", "time"]
    print(f"Generating stubs for packages: {packages}")
    
    for package in packages:
        data = build_package_tree(package)
        print(f"\n--- Structured Data for {data.name} ---")
        print(f"Name: {data.name}")
        print(f"Imports: {len(data.imports)}")
        print(f"Variables: {len(data.variables)}")
        print(f"Functions: {len(data.functions)}")
        print(f"Classes: {len(data.classes)}")

if __name__ == "__main__":
    test_stubgen_api()
