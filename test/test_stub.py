import sys
import os

# Add current directory to sys.path to find stubgen_api
sys.path.insert(0, os.getcwd())

from package_stubgen import build_package_tree, generate_stub

def test_stub():
    package = "math"
    data = build_package_tree(package)
    stub = generate_stub(data)
    # save to file
    with open("test_stub.pyi", "w") as f:
        f.write(stub)

if __name__ == "__main__":
    test_stub()
