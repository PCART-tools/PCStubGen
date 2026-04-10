from __future__ import annotations

from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_pyproject_uses_scikit_build_core_backend() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["build-system"]["build-backend"] == "scikit_build_core.build"
    assert "scikit-build-core>=0.11.0" in pyproject["build-system"]["requires"]


def test_cmakelists_uses_llvm_config_for_native_extension() -> None:
    cmakelists = (PROJECT_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "find_program(LLVM_CONFIG_EXECUTABLE NAMES llvm-config REQUIRED)" in cmakelists
    assert "Python_add_library(_dwarfdump_llvm MODULE WITH_SOABI" in cmakelists
    assert "pcstubgen/signature_completion/c_extension" in cmakelists
