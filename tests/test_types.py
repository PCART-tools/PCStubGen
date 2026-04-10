from __future__ import annotations

from pcstubgen.types import AnyType, DictType, RawType, UnionType


def test_collect_imports_returns_recursive_dependency_set() -> None:
    node = DictType(
        RawType("numpy.ndarray", imports=("numpy",)),
        UnionType((AnyType(), RawType("collections.abc.Buffer", imports=("collections.abc",)))),
    )

    assert node.collect_imports() == {"numpy", "collections.abc", "typing"}
