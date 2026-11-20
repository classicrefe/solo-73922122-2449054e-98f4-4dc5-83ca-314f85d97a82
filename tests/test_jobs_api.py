"""End-to-end tests: jobs, SQL rendering, real isolated-database loading."""
from __future__ import annotations

import json
import zipfile
from io import BytesIO

import psycopg
from fastapi.testclient import TestClient

from app.jobs import STORE, package_zip, run_job
from app.main import app
from app.models import GenerateRequest, JobStatus

client = TestClient(app)


def _load(name):
    raw = json.load(open(f"/workspace/examples/{name}"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _run(req_dict):
    req = GenerateRequest(**req_dict)
    jid = STORE.create()
    run_job(jid, req)
    return jid, STORE.get(jid)


def test_healthz():
    assert client.get("/healthz").json() == {"status": "ok"}


def test_catalog_listing_and_structure(source_dsn):
    r = client.get("/schemas/public/tables")
    assert r.status_code == 200
    assert "employee" in r.json()["base_tables"]
    r2 = client.get("/schemas/public/tables/project/structure")
    body = r2.json()
    assert body["supported"] is True
    fks = body["structure"]["foreign_keys"]
    assert any(f["match"] == "MATCH FULL" for f in fks)


def test_cyclic_four_table_job_succeeds_and_publishes():
    jid, job = _run(_load("request_cyclic_four_tables.json"))
    assert job.status == JobStatus.SUCCEEDED
    assert set(job.selected_tables) == {
        "dept",
        "employee",
        "project",
        "assignment",
    }
    pkg = STORE.package(jid)
    assert set(pkg) == {
        "01_schema.sql",
        "02_data.sql",
        "03_constraints.sql",
        "manifest.json",
        "README.md",
    }
    # Constraints remain NOT DEFERRABLE; nothing was disabled or made deferred.
    fk_ddl = [
        ln for ln in pkg["03_constraints.sql"].splitlines()
        if ln.startswith("ALTER TABLE")
    ]
    assert fk_ddl and all("NOT DEFERRABLE" in ln for ln in fk_ddl)
    assert all("INITIALLY DEFERRED" not in ln for ln in fk_ddl)
    # FKs are post-data ALTERs so the cycle loads without deferral tricks.
    assert all("ALTER TABLE" in ln for ln in fk_ddl)
    # Zip is downloadable and well formed.
    zf = zipfile.ZipFile(BytesIO(package_zip(pkg)))
    assert set(zf.namelist()) == set(pkg)


def test_download_blocked_until_success():
    jid, job = _run(_load("request_unsat_conflict.json"))
    assert job.status == JobStatus.FAILED
    assert STORE.package(jid) is None
    r = client.get(f"/jobs/{jid}/download")
    assert r.status_code == 409


def test_unsat_reports_conflicting_rows_ranges_or_constraints():
    jid, job = _run(_load("request_unsat_conflict.json"))
    conflict = job.conflict.model_dump()
    assert conflict["kind"] == "no_solution"
    joined = json.dumps(conflict, ensure_ascii=False)
    assert "employee_dept_fk" in joined
    assert "500" in joined and "100" in joined
    assert "fk" in conflict["details"]["kinds"]


def test_zero_parent_rows_report_names_both_row_counts_and_fk():
    # Worked example from the task: one inspection_report row with a NOT NULL
    # FK into a registration table generated with zero rows.
    jid, job = _run(_load("request_zero_parent_rows.json"))
    assert job.status == JobStatus.FAILED
    conflict = job.conflict.model_dump()
    assert conflict["kind"] == "no_solution"
    details = conflict["details"]
    named = {row["table"] for row in details["conflicting_row_counts"]}
    assert named == {"inspection_report", "registration"}
    labels = " ".join(row["label"] for row in details["conflicting_row_counts"])
    assert "inspection_report 恰好 1 行" in labels
    assert "registration 恰好 0 行" in labels
    joined = json.dumps(conflict, ensure_ascii=False)
    # The foreign key stays part of the explanation; ranges/nullability too.
    assert "inspection_report_reg_fk" in joined
    assert "NOT NULL" in joined
    assert set(details["kinds"]) >= {"rows", "fk", "range"}
    # No package is ever published for a failed job.
    assert STORE.package(jid) is None
    assert details["row_count_diagnosis"].startswith("complete")


def test_row_count_conflict_excludes_unselected_tables(source_dsn):
    # The same unsat pair inside the full six-table catalog: closure of the
    # two domain tables contains no other table, and even if other tables
    # existed, their counts must not appear in the report.
    jid, job = _run(_load("request_zero_parent_rows.json"))
    joined = json.dumps(job.conflict.model_dump(), ensure_ascii=False)
    for unrelated in ("dept", "employee", "project", "assignment"):
        assert unrelated not in joined


def test_fk_closure_expands_target_table():
    # Only project requested; the transitive FK closure pulls in dept and
    # employee (dept.mgr_emp -> employee), staying within the 4-table cap.
    jid, job = _run(_load("request_closure_single.json"))
    assert job.status == JobStatus.SUCCEEDED
    assert set(job.selected_tables) == {"project", "dept", "employee"}


def test_generated_package_loads_in_fresh_empty_database(
    fresh_verify_database,
):
    jid, job = _run(_load("request_cyclic_four_tables.json"))
    assert job.status == JobStatus.SUCCEEDED
    dsn, _ = fresh_verify_database
    pkg = STORE.package(jid)
    with psycopg.connect(dsn, autocommit=False) as conn:
        for f in ("01_schema.sql", "02_data.sql", "03_constraints.sql"):
            conn.cursor().execute(pkg[f])
        conn.commit()
        cur = conn.cursor()
        for t, expected in [
            ("dept", 3),
            ("employee", 5),
            ("project", 4),
            ("assignment", 6),
        ]:
            cur.execute(f"SELECT count(*) FROM public.{t}")
            assert cur.fetchone()[0] == expected


def test_zero_rows_request(source_dsn):
    req = {
        "schema_name": "public",
        "tables": [
            {
                "table": "project",
                "rows": 0,
                "ranges": {
                    "project_id": {"low": 1, "high": 9},
                    "dept_id": {"low": 1, "high": 9},
                    "dept_code": {"low": 1, "high": 9},
                    "budget": {"low": 1, "high": 9},
                },
            }
        ],
    }
    jid, job = _run(req)
    # closure pulls dept (default 5 rows); project itself stays empty
    assert job.status == JobStatus.SUCCEEDED
    pkg = STORE.package(jid)
    assert "no INSERTs" in pkg["02_data.sql"]


def test_validation_rejects_bad_interval(source_dsn):
    req = {
        "schema_name": "public",
        "tables": [
            {
                "table": "dept",
                "rows": 1,
                "ranges": {
                    "dept_id": {"low": 10, "high": 5},  # low > high
                    "dept_code": {"low": 1, "high": 2},
                    "parent_dept": {"low": 1, "high": 9},
                    "mgr_emp": {"low": 1, "high": 9},
                },
            }
        ],
    }
    # pydantic-level or service-level rejection
    r = client.post("/jobs", json=req)
    if r.status_code == 202:
        jid = r.json()["job_id"]
        # The job runs on the serial background worker; poll to a terminal state.
        import time

        for _ in range(100):
            job = STORE.get(jid)
            if job.status in (JobStatus.FAILED, JobStatus.SUCCEEDED):
                break
            time.sleep(0.1)
        assert job.status == JobStatus.FAILED
        assert job.conflict.kind == "request"
    else:
        assert r.status_code == 422


def test_more_than_max_tables_rejected():
    r = client.post(
        "/jobs",
        json={
            "schema_name": "public",
            "tables": [
                {"table": f"t{i}", "rows": 1, "ranges": {}} for i in range(5)
            ],
        },
    )
    assert r.status_code == 422
