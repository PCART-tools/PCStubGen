import sys
import os

# Add current directory to sys.path to find stubgen_api
sys.path.insert(0, os.getcwd())

from package_stubgen import build_package_tree

def test_stubgen_api():
    package = "numpy"
    print(f"Generating stubs for package: {package}")
    data = build_package_tree(package)

    if data:
        print(f"\n--- Structured Data for {data.name} ---")
        print(f"Name: {data.name}")
        print(f"Imports: {len(data.imports)}")
        print(f"Variables: {len(data.variables)}")
        print(f"Functions: {len(data.functions)}")
        print(f"Classes: {len(data.classes)}")

        if data.imports:
            print("\nImports:")
            for imp in data.imports:
                print(f"  {imp.strip()}")

        if data.variables:
            print("\nVariables:")
            for var in data.variables:
                print(f"  - {var.name}: {var.type}")

        if data.functions:
            print("\nFunctions:")
            for func in data.functions:
                # Show full function signature
                sig = func.format_sig(indent="").split('\n')[0]
                if sig.startswith("def "):
                    sig = sig[4:]
                if sig.endswith(":"):
                    sig = sig[:-1]
                print(f"  - {sig}")

        if data.classes:
            print("\nClasses:")
            for cls in data.classes:
                print(f"  - {cls.name} (Bases: {cls.bases})")

                if cls.variables:
                    print("    Variables:")
                    for var in cls.variables:
                        print(f"      - {var.name}: {var.type}")

                if cls.methods:
                    print("    Methods:")
                    for method in cls.methods:
                        sig = method.format_sig(indent="").split('\n')[0]
                        if sig.startswith("def "):
                            sig = sig[4:]
                        if sig.endswith(":"):
                            sig = sig[:-1]
                        print(f"      - {sig}")

                if cls.properties:
                    print("    Properties:")
                    for prop in cls.properties:
                        readonly = " (readonly)" if prop.readonly else ""
                        print(f"      - {prop.name}: {prop.type}{readonly}")

if __name__ == "__main__":
    test_stubgen_api()
