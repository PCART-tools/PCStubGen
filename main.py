import sys
import os




# Add current directory to sys.path to find stubgen_api

sys.path.insert(0, os.getcwd())


from pcstubgen import build_package_tree, generate_stub

def main():
    package = "math"
    package = "torch._C"
    data = build_package_tree(package)
    stub = generate_stub(data, False)
    # save to file
    print("Saving stub to test_stub.pyi")
    # print(stub)
    with open("test_stub.pyi", "w") as f:
        f.write(stub)

if __name__ == "__main__":
    main()