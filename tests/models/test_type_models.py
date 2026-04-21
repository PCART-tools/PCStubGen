from __future__ import annotations

from pcstubgen.type_models import AnyType, DictType, ListType, RawType, UnionType


def test_collect_imports_returns_recursive_dependency_set() -> None:
    node = DictType(
        RawType("numpy.ndarray", imports=("numpy",)),
        UnionType((AnyType(), RawType("collections.abc.Buffer", imports=("collections.abc",)))),
    )

    assert node.collect_imports() == {"numpy", "collections.abc", "typing"}


def test_union_canonicalize_short_circuits_any() -> None:
    canonical = UnionType(
        (
            RawType("str"),
            UnionType((RawType("int"), AnyType())),
        )
    ).canonicalize()

    assert canonical == AnyType()


def test_union_canonicalize_flattens_deduplicates_and_sorts_members() -> None:
    canonical = UnionType(
        (
            RawType("str"),
            UnionType((RawType("float"), RawType("int"))),
            RawType("bool"),
            UnionType(
                (
                    RawType("int"),
                    UnionType((RawType("bool"), UnionType(()))),
                )
            ),
        )
    ).canonicalize()

    assert canonical == UnionType(
        (
            RawType("bool"),
            RawType("float"),
            RawType("int"),
            RawType("str"),
        )
    )


def test_container_canonicalize_recurses_into_nested_union_members() -> None:
    canonical = ListType(
        UnionType(
            (
                RawType("str"),
                UnionType((RawType("int"), RawType("bool"))),
                RawType("bool"),
            )
        )
    ).canonicalize()

    assert canonical == ListType(
        UnionType((RawType("bool"), RawType("int"), RawType("str")))
    )


def test_container_canonicalize_folds_single_union_members() -> None:
    canonical = DictType(
        UnionType((RawType("str"),)),
        UnionType((RawType("int"),)),
    ).canonicalize()

    assert canonical == DictType(RawType("str"), RawType("int"))
