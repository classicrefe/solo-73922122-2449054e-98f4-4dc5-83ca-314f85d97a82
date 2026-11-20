"""Solver unit tests: intervals, uniqueness, nullability, match types, cycles."""
from __future__ import annotations

import pytest

from app.models import (
    Column,
    ForeignKey,
    IntRange,
    MatchType,
    TableSchema,
    TableSpec,
    UniqueConstraint,
)
from app.solver import Generator, Unsatisfiable
import app.solver as solver_mod


def col(name, pg_type="integer", nullable=False):
    return Column(
        name=name,
        pg_type=pg_type,
        ordinal=1,
        nullable=nullable,
        has_default=False,
        is_identity=False,
        is_generated=False,
    )


def spec(table, rows, ranges):
    return TableSpec(
        table=table,
        rows=rows,
        ranges={c: IntRange(low=lo, high=hi) for c, (lo, hi) in ranges.items()},
    )


def _check_unique(data, key_cols):
    seen = set()
    for row in data:
        key = tuple(row[c] for c in key_cols)
        if any(v is None for v in key):
            continue
        assert key not in seen, f"duplicate non-NULL key {key}"
        seen.add(key)


def test_pk_uniqueness_and_ranges():
    ts = TableSchema(
        schema_name="public",
        name="t",
        columns=[col("id"), col("v", nullable=True)],
        unique_constraints=[UniqueConstraint(name="pk", columns=["id"], is_primary=True)],
        foreign_keys=[],
    )
    specs = {"t": spec("t", 5, {"id": (1, 5), "v": (10, 20)})}
    data = Generator({"t": ts}, specs).solve()["t"]
    assert len(data) == 5
    assert sorted(r["id"] for r in data) == [1, 2, 3, 4, 5]
    assert all(10 <= r["v"] <= 20 for r in data)


def test_rows_exceed_range_cardinality_is_unsat():
    ts = TableSchema(
        schema_name="public",
        name="t",
        columns=[col("id")],
        unique_constraints=[UniqueConstraint(name="pk", columns=["id"], is_primary=True)],
        foreign_keys=[],
    )
    specs = {"t": spec("t", 5, {"id": (1, 2)})}
    with pytest.raises(Unsatisfiable) as exc:
        Generator({"t": ts}, specs).solve()
    kinds = {g.kind for g in exc.value.groups}
    assert "unique" in kinds and "range" in kinds


def test_nullable_column_can_be_null():
    ts = TableSchema(
        schema_name="public",
        name="t",
        columns=[col("id"), col("v", nullable=True)],
        unique_constraints=[UniqueConstraint(name="pk", columns=["id"], is_primary=True)],
        foreign_keys=[],
    )
    # NOT NULL id has only 2 values but 3 rows requested; v being nullable does
    # not help ids — this stays unsat, proving nullability is column-scoped.
    specs_bad = {"t": spec("t", 3, {"id": (1, 2), "v": (1, 1)})}
    with pytest.raises(Unsatisfiable):
        Generator({"t": ts}, specs_bad).solve()

    # A nullable unique column may repeat NULLs freely.
    ts2 = TableSchema(
        schema_name="public",
        name="t",
        columns=[col("id"), col("v", nullable=True)],
        unique_constraints=[
            UniqueConstraint(name="pk", columns=["id"], is_primary=True),
            UniqueConstraint(name="uk_v", columns=["v"], is_primary=False),
        ],
        foreign_keys=[],
    )
    specs = {"t": spec("t", 4, {"id": (1, 10), "v": (100, 100)})}
    data = Generator({"t": ts2}, specs).solve()["t"]
    # v has a single-point interval; NULL repetition must be used to satisfy UK.
    nulls = sum(1 for r in data if r["v"] is None)
    assert nulls >= 3
    _check_unique(data, ["v"])


def test_self_referencing_foreign_key():
    ts = TableSchema(
        schema_name="public",
        name="node",
        columns=[col("id"), col("parent", nullable=True)],
        unique_constraints=[
            UniqueConstraint(name="pk", columns=["id"], is_primary=True)
        ],
        foreign_keys=[
            ForeignKey(
                name="fk_parent",
                columns=["parent"],
                ref_table="node",
                ref_columns=["id"],
                match=MatchType.SIMPLE,
                on_update="NO ACTION",
                on_delete="SET NULL",
                not_deferrable=True,
                initially_deferred=False,
            )
        ],
    )
    specs = {"node": spec("node", 4, {"id": (1, 4), "parent": (1, 4)})}
    data = Generator({"node": ts}, specs).solve()["node"]
    ids = {r["id"] for r in data}
    for r in data:
        assert r["parent"] is None or r["parent"] in ids


def test_circular_two_tables_resolved():
    a = TableSchema(
        schema_name="public",
        name="a",
        columns=[col("id"), col("b_id", nullable=True)],
        unique_constraints=[UniqueConstraint(name="pa", columns=["id"], is_primary=True)],
        foreign_keys=[
            ForeignKey(
                name="fk_a_b", columns=["b_id"], ref_table="b", ref_columns=["id"],
                match=MatchType.SIMPLE, on_update="NO ACTION", on_delete="SET NULL",
                not_deferrable=True, initially_deferred=False,
            )
        ],
    )
    b = TableSchema(
        schema_name="public",
        name="b",
        columns=[col("id"), col("a_id")],
        unique_constraints=[UniqueConstraint(name="pb", columns=["id"], is_primary=True)],
        foreign_keys=[
            ForeignKey(
                name="fk_b_a", columns=["a_id"], ref_table="a", ref_columns=["id"],
                match=MatchType.SIMPLE, on_update="NO ACTION", on_delete="NO ACTION",
                not_deferrable=True, initially_deferred=False,
            )
        ],
    )
    specs = {
        "a": spec("a", 3, {"id": (1, 3), "b_id": (10, 12)}),
        "b": spec("b", 3, {"id": (10, 12), "a_id": (1, 3)}),
    }
    out = Generator({"a": a, "b": b}, specs).solve()
    a_ids = {r["id"] for r in out["a"]}
    b_ids = {r["id"] for r in out["b"]}
    for r in out["a"]:
        assert r["b_id"] is None or r["b_id"] in b_ids
    for r in out["b"]:
        assert r["a_id"] in a_ids


def _composite_table(match: MatchType):
    parent = TableSchema(
        schema_name="public",
        name="p",
        columns=[col("k1"), col("k2", nullable=True)],
        unique_constraints=[
            UniqueConstraint(name="pk_p", columns=["k1", "k2"], is_primary=True)
        ],
        foreign_keys=[],
    )
    child = TableSchema(
        schema_name="public",
        name="c",
        columns=[col("id"), col("f1", nullable=True), col("f2", nullable=True)],
        unique_constraints=[
            UniqueConstraint(name="pk_c", columns=["id"], is_primary=True)
        ],
        foreign_keys=[
            ForeignKey(
                name="fk_c_p",
                columns=["f1", "f2"],
                ref_table="p",
                ref_columns=["k1", "k2"],
                match=match,
                on_update="NO ACTION",
                on_delete="NO ACTION",
                not_deferrable=True,
                initially_deferred=False,
            )
        ],
    )
    return parent, child


def test_match_full_forbids_mixed_null_via_solver_ranges():
    # Force the child f1 to NULL via an impossible non-null situation is hard
    # directly; instead assert a feasible solve honours FULL all-or-nothing.
    parent, child = _composite_table(MatchType.FULL)
    specs = {
        "p": spec("p", 2, {"k1": (1, 2), "k2": (10, 11)}),
        "c": spec("c", 4, {"id": (100, 103), "f1": (1, 2), "f2": (10, 11)}),
    }
    out = Generator({"p": parent, "c": child}, specs).solve()
    pkeys = {(r["k1"], r["k2"]) for r in out["p"]}
    for r in out["c"]:
        if r["f1"] is None and r["f2"] is None:
            continue  # all-NULL allowed
        # MATCH FULL: never a mixed key
        assert (r["f1"] is None) == (r["f2"] is None)
        assert (r["f1"], r["f2"]) in pkeys


def test_match_simple_allows_mixed_null():
    parent, child = _composite_table(MatchType.SIMPLE)
    specs = {
        "p": spec("p", 2, {"k1": (1, 2), "k2": (10, 11)}),
        "c": spec("c", 6, {"id": (100, 105), "f1": (1, 2), "f2": (10, 11)}),
    }
    out = Generator({"p": parent, "c": child}, specs).solve()
    # solver prefers non-NULL; feasibility is the key assertion here.
    assert len(out["c"]) == 6


def test_match_simple_vs_full_mixed_null_distinction():
    """Deterministic SIMPLE-vs-FULL distinction on a forced mixed-NULL key.

    The second FK column ``f2`` is NOT NULL, so an all-NULL key is impossible:
    a row whose ``f1`` is NULL while ``f2`` keeps its value is necessarily a
    *mixed* key. MATCH SIMPLE exempts mixed keys from the lookup, so the
    instance is feasible; MATCH FULL forbids mixed keys outright, so the very
    same instance is unsatisfiable. This avoids relying on which of several
    equally-valid NULL placements z3 happens to return.
    """
    parent = TableSchema(
        schema_name="public",
        name="p",
        columns=[col("k1"), col("k2")],
        unique_constraints=[
            UniqueConstraint(name="pk_p", columns=["k1", "k2"], is_primary=True)
        ],
        foreign_keys=[],
    )

    def child_with(match: MatchType):
        return TableSchema(
            schema_name="public",
            name="c",
            # f1 nullable, f2 NOT NULL: a NULL f1 forces a mixed (never
            # all-NULL) composite key.
            columns=[col("id"), col("f1", nullable=True), col("f2")],
            unique_constraints=[
                UniqueConstraint(name="pk_c", columns=["id"], is_primary=True),
                # Two non-NULL identical (f1, f2) keys would collide, forcing the
                # second single-point row onto a NULL-bearing (mixed) key.
                UniqueConstraint(name="uk_fk", columns=["f1", "f2"], is_primary=False),
            ],
            foreign_keys=[
                ForeignKey(
                    name="fk_c_p",
                    columns=["f1", "f2"],
                    ref_table="p",
                    ref_columns=["k1", "k2"],
                    match=match,
                    on_update="NO ACTION",
                    on_delete="NO ACTION",
                    not_deferrable=True,
                    initially_deferred=False,
                )
            ],
        )

    specs = {
        "p": spec("p", 1, {"k1": (1, 1), "k2": (10, 10)}),
        # Two child rows: one matches the single parent key; the other cannot
        # reuse it (single-point intervals) and must place f1 = NULL.
        "c": spec("c", 2, {"id": (100, 101), "f1": (1, 1), "f2": (10, 10)}),
    }

    # MATCH SIMPLE: the mixed-NULL row is exempt, so a solution exists and it
    # necessarily contains exactly one mixed key (f1 NULL, f2 non-NULL).
    simple_out = Generator(
        {"p": parent, "c": child_with(MatchType.SIMPLE)}, specs
    ).solve()
    mixed = [
        r for r in simple_out["c"] if (r["f1"] is None) != (r["f2"] is None)
    ]
    assert len(mixed) == 1
    assert mixed[0]["f1"] is None and mixed[0]["f2"] == 10

    # MATCH FULL: mixed keys are forbidden and all-NULL is impossible
    # (f2 NOT NULL), so no valid placement for the second row -> unsat.
    with pytest.raises(Unsatisfiable):
        Generator(
            {"p": parent, "c": child_with(MatchType.FULL)}, specs
        ).solve()


def test_zero_rows_table():
    ts = TableSchema(
        schema_name="public",
        name="empty_t",
        columns=[col("id")],
        unique_constraints=[UniqueConstraint(name="pk", columns=["id"], is_primary=True)],
        foreign_keys=[],
    )
    specs = {"empty_t": spec("empty_t", 0, {"id": (1, 10)})}
    out = Generator({"empty_t": ts}, specs).solve()
    assert out["empty_t"] == []


def test_type_bounds_enforced():
    ts = TableSchema(
        schema_name="public",
        name="t",
        columns=[col("id", pg_type="smallint")],
        unique_constraints=[UniqueConstraint(name="pk", columns=["id"], is_primary=True)],
        foreign_keys=[],
    )
    with pytest.raises(ValueError):
        IntRange(low=-40000, high=1).validated(ts.column("id"))


# ---------------------------------------------------------------------------
# Row-count conflict diagnosis: the labelled unsat core alone never contains
# a rows group (the count is encoded structurally), so a dedicated
# counterfactual analysis must surface the counts that make the instance
# infeasible while excluding counts that play no part.
# ---------------------------------------------------------------------------

def _parent_child_schemas(nullable_fk=False):
    parent = TableSchema(
        schema_name="public",
        name="registration",
        columns=[col("reg_no"), col("batch_code", pg_type="smallint")],
        unique_constraints=[
            UniqueConstraint(name="registration_pk", columns=["reg_no"], is_primary=True)
        ],
        foreign_keys=[],
    )
    child = TableSchema(
        schema_name="public",
        name="inspection_report",
        columns=[
            col("report_id"),
            col("registration_no", nullable=nullable_fk),
            col("result_code", nullable=True),
        ],
        unique_constraints=[
            UniqueConstraint(
                name="inspection_report_pk", columns=["report_id"], is_primary=True
            )
        ],
        foreign_keys=[
            ForeignKey(
                name="inspection_report_reg_fk",
                columns=["registration_no"],
                ref_table="registration",
                ref_columns=["reg_no"],
                match=MatchType.SIMPLE,
                on_update="CASCADE",
                on_delete="RESTRICT",
                not_deferrable=True,
                initially_deferred=False,
            )
        ],
    )
    return parent, child


def test_zero_parent_rows_with_required_fk_names_both_row_counts():
    parent, child = _parent_child_schemas()
    schemas = {"registration": parent, "inspection_report": child}
    specs = {
        "registration": spec(
            "registration", 0, {"reg_no": (8000, 8999), "batch_code": (1, 99)}
        ),
        "inspection_report": spec(
            "inspection_report",
            1,
            {"report_id": (5000, 5999), "registration_no": (8000, 8999), "result_code": (1, 9)},
        ),
    }
    with pytest.raises(Unsatisfiable) as exc:
        Generator(schemas, specs).solve()
    unsat = exc.value
    row_tables = {g.refs[0] for g in unsat.row_groups}
    assert row_tables == {"registration", "inspection_report"}
    assert unsat.row_diagnosis == "complete"
    labels = " ".join(g.label for g in unsat.row_groups)
    assert "恰好 0 行" in labels and "恰好 1 行" in labels
    # Original constraints stay in the explanation; nothing is weakened.
    kinds = {g.kind for g in unsat.groups}
    assert "fk" in kinds and "range" in kinds
    fk_labels = [g.label for g in unsat.groups if g.kind == "fk"]
    assert any("inspection_report_reg_fk" in x for x in fk_labels)


def test_nullable_fk_with_zero_parent_rows_is_sat_via_null():
    # Same cardinalities, but the FK column is nullable: NULL exempts the row,
    # so a 1-row child against a 0-row parent must remain feasible and must not
    # be reported as a row-count conflict.
    parent, child = _parent_child_schemas(nullable_fk=True)
    schemas = {"registration": parent, "inspection_report": child}
    specs = {
        "registration": spec(
            "registration", 0, {"reg_no": (8000, 8999), "batch_code": (1, 99)}
        ),
        "inspection_report": spec(
            "inspection_report",
            1,
            {"report_id": (5000, 5999), "registration_no": (8000, 8999), "result_code": (1, 9)},
        ),
    }
    out = Generator(schemas, specs).solve()
    assert out["inspection_report"][0]["registration_no"] is None


def test_row_count_diagnosis_excludes_unrelated_table():
    # Three tables: p (parent), c (NOT NULL FK -> p), bystander (no relations
    # to either). c=1 vs p=0 is unsat; the bystander's requested row count must
    # never appear in the row-count conflict set, whatever count it requests.
    parent, child = _parent_child_schemas()
    bystander = TableSchema(
        schema_name="public",
        name="bystander",
        columns=[col("id")],
        unique_constraints=[UniqueConstraint(name="bystander_pk", columns=["id"], is_primary=True)],
        foreign_keys=[],
    )
    schemas = {
        "registration": parent,
        "inspection_report": child,
        "bystander": bystander,
    }
    specs = {
        "registration": spec(
            "registration", 0, {"reg_no": (8000, 8999), "batch_code": (1, 99)}
        ),
        "inspection_report": spec(
            "inspection_report",
            1,
            {"report_id": (5000, 5999), "registration_no": (8000, 8999), "result_code": (1, 9)},
        ),
        "bystander": spec("bystander", 7, {"id": (1, 20)}),
    }
    with pytest.raises(Unsatisfiable) as exc:
        Generator(schemas, specs).solve()
    row_tables = {g.refs[0] for g in exc.value.row_groups}
    assert row_tables == {"registration", "inspection_report"}
    assert "bystander" not in row_tables


def test_cardinality_range_conflict_names_the_row_count():
    # 5 distinct PK ids requested from a 2-value interval: relaxing the row
    # count alone restores feasibility (2 rows fit), so the requested row
    # count genuinely participates and must be named alongside range/UK.
    ts = TableSchema(
        schema_name="public",
        name="t",
        columns=[col("id")],
        unique_constraints=[UniqueConstraint(name="pk", columns=["id"], is_primary=True)],
        foreign_keys=[],
    )
    specs = {"t": spec("t", 5, {"id": (1, 2)})}
    with pytest.raises(Unsatisfiable) as exc:
        Generator({"t": ts}, specs).solve()
    unsat = exc.value
    assert unsat.row_diagnosis == "complete"
    assert [g.refs[0] for g in unsat.row_groups] == ["t"]
    assert {g.kind for g in unsat.groups} >= {"unique", "range"}


def test_range_disjoint_fk_keeps_child_count_and_prunes_unrelated():
    # The FK intervals are disjoint and both requested counts are positive.
    # Relaxing the child count alone makes the system satisfiable (it may
    # choose 0 child rows), so the child's requested count participates;
    # relaxing only the parent cannot help (disjoint intervals stay disjoint),
    # so the parent's count is pruned. An independent bystander table is
    # pruned as well regardless of its requested count.
    parent, child = _parent_child_schemas()
    bystander = TableSchema(
        schema_name="public",
        name="bystander",
        columns=[col("id")],
        unique_constraints=[
            UniqueConstraint(name="bystander_pk", columns=["id"], is_primary=True)
        ],
        foreign_keys=[],
    )
    schemas = {
        "registration": parent,
        "inspection_report": child,
        "bystander": bystander,
    }
    specs = {
        "registration": spec(
            "registration", 2, {"reg_no": (100, 199), "batch_code": (1, 9)}
        ),
        "inspection_report": spec(
            "inspection_report",
            2,
            {"report_id": (5000, 5999), "registration_no": (8000, 8999), "result_code": (1, 9)},
        ),
        "bystander": spec("bystander", 3, {"id": (1, 20)}),
    }
    with pytest.raises(Unsatisfiable) as exc:
        Generator(schemas, specs).solve()
    unsat = exc.value
    assert [g.refs[0] for g in unsat.row_groups] == ["inspection_report"]
    kinds = {g.kind for g in unsat.groups}
    assert "fk" in kinds and "range" in kinds


def test_relaxed_probe_treats_row_count_as_variable():
    # Direct probe check: relaxing the empty parent's count must make the
    # otherwise-unsat instance satisfiable (it may choose any k in [0, max]).
    parent, child = _parent_child_schemas()
    schemas = {"registration": parent, "inspection_report": child}
    specs = {
        "registration": spec(
            "registration", 0, {"reg_no": (8000, 8999), "batch_code": (1, 99)}
        ),
        "inspection_report": spec(
            "inspection_report",
            1,
            {"report_id": (5000, 5999), "registration_no": (8000, 8999), "result_code": (1, 9)},
        ),
    }
    gen = Generator(schemas, specs)
    gen._build_cells()
    labelled = gen._hard_constraints()
    import z3 as _z3

    assert gen._probe_with_relaxed_counts({"rows:registration"}) == _z3.sat
    # Relaxing only the child count cannot help: an empty referenced table
    # means even one non-NULL FK value has nowhere to point... but a relaxed
    # child may choose k=0 rows, which IS feasible.
    assert gen._probe_with_relaxed_counts({"rows:inspection_report"}) == _z3.sat
    # Relaxing nothing reproduces the proven infeasibility.
    assert gen._probe_with_relaxed_counts(set()) == _z3.unsat


def test_timeout_in_row_diagnosis_keeps_proven_unsat(monkeypatch):
    # Force every diagnostic probe to time out. The Unsatisfiable verdict and
    # its constraint core must survive; only the row-count localisation is
    # flagged as unfinished.
    parent, child = _parent_child_schemas()
    schemas = {"registration": parent, "inspection_report": child}
    specs = {
        "registration": spec(
            "registration", 0, {"reg_no": (8000, 8999), "batch_code": (1, 99)}
        ),
        "inspection_report": spec(
            "inspection_report",
            1,
            {"report_id": (5000, 5999), "registration_no": (8000, 8999), "result_code": (1, 9)},
        ),
    }
    import z3 as _z3

    monkeypatch.setattr(
        solver_mod.Generator,
        "_probe_with_relaxed_counts",
        lambda self, keys: _z3.unknown,
    )
    with pytest.raises(Unsatisfiable) as exc:
        Generator(schemas, specs).solve()
    unsat = exc.value
    assert unsat.row_diagnosis == "timed_out"
    # Undecided counts are retained conservatively (never silently dropped),
    # while the proven constraint core still carries the no-solution verdict.
    assert {g.refs[0] for g in unsat.row_groups} == {
        "registration",
        "inspection_report",
    }
    assert {g.kind for g in unsat.groups} >= {"fk", "range"}
