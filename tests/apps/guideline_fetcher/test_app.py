import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from matchminer_ai.apps.guideline_fetcher.app import create_app
from matchminer_ai.apps.guideline_fetcher.services import Catalogs, InputError, display_considerations, parse_upload
from matchminer_ai.apps.guideline_fetcher.settings import Settings


def catalog(root, name="fictional", complete=True):
    path = root / "run" / name
    path.mkdir(parents=True)
    (path / "validation.json").write_text(json.dumps({"status": "passed", "paradigms": 2}))
    (path / "status.json").write_text(
        json.dumps({"status": "complete" if complete else "running", "updated_utc": "2026-09-01"})
    )
    (path / "sources.json").write_text(
        json.dumps({"disease": name, "title": name.title(), "version": "1.2026"})
    )
    (path / "paradigms.jsonl").write_text("{}")
    return path


class FakeInference:
    def __init__(self):
        self.calls = []

    def retrieve(self, summary, paths, top_n, candidate_k, progress):
        self.calls.append((summary, paths, top_n, candidate_k))
        progress("Scoring with TrialChecker")
        return {"spaces": [], "summary": summary, "report": "Synthetic report"}

    def summarize(self, text, notes, progress):
        self.calls.append((text, notes))
        return {"summary": "Synthetic cancer history", "exclusion_evidence": "Unknown"}


@pytest.fixture
def setup(tmp_path):
    catalog(tmp_path)
    inference = FakeInference()
    app = create_app(Settings(catalog_root=tmp_path, example=tmp_path / "missing"), inference)
    return app, inference


def wait(client, key):
    for _ in range(100):
        response = client.get(f"/api/jobs/{key}")
        if response.json()["status"] in {"complete", "failed"}:
            return response
        time.sleep(0.01)
    pytest.fail("Job did not finish")


def test_only_completed_catalogs_are_exposed(tmp_path):
    catalog(tmp_path, "ready")
    catalog(tmp_path, "partial", False)
    rows = Catalogs(tmp_path).list()
    assert [r["disease"] for r in rows] == ["ready"]


def test_network_host_access_and_origin_boundary(setup):
    with TestClient(setup[0], base_url="http://192.0.2.10:8034") as client:
        assert client.get("/api/config").status_code == 200
        assert (
            client.post(
                "/api/summarize",
                json={"text": "Synthetic"},
                headers={"Origin": "http://192.0.2.10:8034"},
            ).status_code
            == 202
        )
        assert (
            client.post(
                "/api/summarize",
                json={"text": "Synthetic"},
                headers={"Origin": "https://outside.invalid"},
            ).status_code
            == 403
        )


def test_operator_host_allowlist(tmp_path, monkeypatch):
    monkeypatch.setenv("GUIDELINE_ALLOWED_HOSTS", "localhost, 192.0.2.10")
    app = create_app(Settings(catalog_root=tmp_path), FakeInference())
    with TestClient(app, base_url="http://192.0.2.10:8034") as client:
        assert client.get("/api/config").status_code == 200
        assert client.get("/api/config", headers={"Host": "other.invalid"}).status_code == 400


def test_loopback_host_access_and_origin_boundary(tmp_path, monkeypatch):
    monkeypatch.delenv("GUIDELINE_ALLOWED_HOSTS", raising=False)
    catalog(tmp_path)
    app = create_app(Settings(phi=True, catalog_root=tmp_path), FakeInference())
    assert Settings(phi=True).allowed_hosts == ["127.0.0.1", "localhost"]
    assert Settings().allowed_hosts == ["*"]
    with TestClient(app, base_url="http://127.0.0.1:8044") as client:
        assert client.get("/api/config").status_code == 200
        # A DNS-rebinding page or another host name cannot reach the PHI app.
        for host in ("192.0.2.10:8044", "rebind.invalid:8044", "0.0.0.0:8044"):
            assert client.get("/api/config", headers={"Host": host}).status_code == 400
        assert (
            client.post(
                "/api/summarize",
                json={"text": "Synthetic"},
                headers={"Origin": "http://127.0.0.1:8044"},
            ).status_code
            == 202
        )
        assert (
            client.post(
                "/api/summarize",
                json={"text": "Synthetic"},
                headers={"Origin": "https://outside.invalid"},
            ).status_code
            == 403
        )


def served_apps(monkeypatch):
    calls = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: calls.append((app, kw)))
    monkeypatch.setattr("multiprocessing.set_start_method", lambda *a, **kw: None)
    return calls


def routes(app):
    return {route.path for route in app.routes}


def test_run_starts_the_non_phi_app_by_default(monkeypatch, capsys):
    from matchminer_ai.apps.guideline_fetcher import run
    calls = served_apps(monkeypatch)
    run.main([])
    run.main(["--host", "192.0.2.10", "--port", "9000"])
    assert [kw for _, kw in calls] == [
        {"host": "0.0.0.0", "port": 8034, "access_log": False},
        {"host": "192.0.2.10", "port": 9000, "access_log": False},
    ]
    assert "/login" not in routes(calls[0][0])
    assert "/api/population/databases" not in routes(calls[0][0])
    assert capsys.readouterr().out == ""


def test_run_binds_only_to_loopback(tmp_path, monkeypatch):
    from matchminer_ai.apps.guideline_fetcher import run
    calls = served_apps(monkeypatch)
    run.main(["--phi"])
    assert [kw for _, kw in calls] == [{"host": "127.0.0.1", "port": 8044, "access_log": False}]
    assert {"/login", "/logout", "/api/population/databases", "/api/population/cache"} <= routes(
        calls[0][0])
    for host in ("0.0.0.0", "::", "192.0.2.10", "gpu-server"):
        with pytest.raises(SystemExit):
            run.main(["--phi", "--host", host])
    # PHI mode never starts on the non-PHI app's cache.
    monkeypatch.setenv("GUIDELINE_CACHE", str(tmp_path / "data/no_phi/guideline_fetcher_cache"))
    with pytest.raises(SystemExit):
        run.main(["--phi"])
    assert len(calls) == 1
    from matchminer_ai.apps.common.security import is_loopback

    assert is_loopback("localhost") and is_loopback("::1") and is_loopback("127.0.0.2")


def test_default_search_reranks_five_candidates_and_returns_one_space(setup):
    with TestClient(setup[0]) as client:
        response = client.post("/api/retrieve", json={"summary": "Synthetic"})
        result = wait(client, response.json()["job_id"])
        assert result.json()["result"]["search"]["top_n"] == 1
        assert result.json()["result"]["search"]["candidate_k"] == 5
        assert setup[1].calls[0][2:] == (1, 5)


def test_category_display_clarifies_absence_without_rewriting_evidence():
    import pandas as pd

    original = pd.DataFrame(
        [
            {
                "diagnostic_workup": [{"category": "Not specified in supplied evidence"}],
                "treatment_options": [
                    {
                        "category": "Not specified in the supplied evidence.",
                        "evidence": [{"quote": "Not specified in the supplied evidence"}],
                    },
                    {"category": "Category 1"},
                ],
            }
        ]
    )
    displayed = display_considerations(original).iloc[0]
    assert (
        displayed.diagnostic_workup[0]["category"] == "NCCN recommendation category not specified"
    )
    assert (
        displayed.treatment_options[0]["category"] == "NCCN recommendation category not specified"
    )
    assert displayed.treatment_options[1]["category"] == "Category 1"
    assert (
        displayed.treatment_options[0]["evidence"]
        == original.iloc[0].treatment_options[0]["evidence"]
    )
    assert original.iloc[0].diagnostic_workup[0]["category"] == "Not specified in supplied evidence"


def test_retrieval_preserves_parameters_and_isolates_sessions(setup):
    app, inference = setup
    with TestClient(app) as a, TestClient(app) as b:
        info = a.get("/api/catalogs").json()
        assert "path" not in info["catalogs"][0]
        request = a.post(
            "/api/retrieve",
            json={
                "summary": "Fabricated history",
                "catalogs": [info["catalogs"][0]["id"]],
                "top_n": 3,
                "candidate_k": 12,
            },
        )
        assert request.status_code == 202
        key = request.json()["job_id"]
        result = wait(a, key)
        assert result.json()["result"]["summary"] == "Fabricated history"
        assert inference.calls[0][2:] == (3, 12)
        assert b.get(f"/api/jobs/{key}").status_code == 404
        assert b.get("/api/session").json()["job"] is None
        assert a.get("/api/session").json()["job"]["id"] == key
        assert result.headers["cache-control"] == "no-store"
        assert a.delete("/api/session").status_code == 204
        assert a.get(f"/api/jobs/{key}").status_code == 404


def test_rejects_invalid_search_and_unrecognized_catalog(setup):
    with TestClient(setup[0]) as client:
        assert client.post("/api/retrieve", json={"summary": " ", "top_n": 5}).status_code == 422
        assert (
            client.post(
                "/api/retrieve", json={"summary": "x", "top_n": 5, "candidate_k": 2}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/retrieve", json={"summary": "x", "catalogs": ["../../data"]}
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/api/summarize", json={"text": "x"}, headers={"Origin": "https://outside.invalid"}
            ).status_code
            == 403
        )


def test_upload_dates_and_single_patient_validation():
    data = b"patient_id,note_text,note_date\na,Later,2026-02-01\na,Earlier,2026-01-01\na,Undated,\n"
    result = parse_upload("notes.csv", data)
    assert [r["note_text"] for r in result["notes"]] == ["Earlier", "Later", "Undated"]
    assert result["notes"][-1]["note_date"] is None
    assert "1900" not in result["text"]
    with pytest.raises(InputError, match="exactly one"):
        parse_upload("notes.csv", b"patient_id,note_text\na,one\nb,two")
    with pytest.raises(InputError, match="invalid"):
        parse_upload("notes.csv", b"note_text,note_date\none,not a date")


def test_upload_and_summary_adapter(setup):
    with TestClient(setup[0]) as client:
        data = client.post(
            "/api/upload",
            files={"file": ("notes.csv", b"note_text,note_date\nSynthetic note,2026-01-01")},
        ).json()
        response = client.post(
            "/api/summarize", json={"text": data["text"], "notes": data["notes"]}
        )
        result = wait(client, response.json()["job_id"]).json()
        assert result["result"]["summary"] == "Synthetic cancer history"
        assert setup[1].calls[0][1][0]["note_date"].startswith("2026-01-01")
        assert (
            client.post("/api/upload", files={"file": ("bad.exe", b"invalid")}).status_code == 400
        )


def test_unreadable_pdf_does_not_silently_skip_pages():
    PdfWriter = pytest.importorskip("pypdf").PdfWriter
    import io

    buffer = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(buffer)
    with pytest.raises(InputError, match="page 1"):
        parse_upload("scan.pdf", buffer.getvalue())


def test_inference_errors_do_not_echo_patient_data(setup):
    def fail(*args):
        raise RuntimeError("CONFIDENTIAL CONTENT")

    setup[1].summarize = fail
    with TestClient(setup[0]) as client:
        key = client.post("/api/summarize", json={"text": "Synthetic"}).json()["job_id"]
        result = wait(client, key)
        assert result.json()["status"] == "failed"
        assert "CONFIDENTIAL CONTENT" not in result.text


def test_job_results_expire(setup, monkeypatch):
    with TestClient(setup[0]) as client:
        key = client.post("/api/summarize", json={"text": "Synthetic"}).json()["job_id"]
        wait(client, key)
        setup[0].state.jobs.items[key]["updated"] -= 3601
        assert client.get(f"/api/jobs/{key}").status_code == 404


def test_busy_jobs_survive_refresh_and_cannot_be_cleared(setup):
    import threading

    release = threading.Event()

    def slow(*args):
        release.wait(5)
        return {"summary": "Synthetic", "exclusion_evidence": ""}

    setup[1].summarize = slow
    with TestClient(setup[0]) as client:
        try:
            key = client.post("/api/summarize", json={"text": "Synthetic"}).json()["job_id"]
            assert client.get("/api/session").json()["job"]["id"] == key
            assert client.delete("/api/session").status_code == 409
            assert client.post("/api/summarize", json={"text": "Synthetic"}).status_code == 409
        finally:
            release.set()
        assert wait(client, key).json()["status"] == "complete"


def test_access_token_gates_pages_and_api(tmp_path):
    from .conftest import ACCESS_TOKEN

    catalog(tmp_path)
    app = create_app(Settings(phi=True, catalog_root=tmp_path), FakeInference())
    with TestClient(app, base_url="http://127.0.0.1:8044") as client:
        client.headers.pop("Authorization")
        page = client.get("/", follow_redirects=False)
        assert page.status_code == 303 and page.headers["location"] == "/login"
        assert page.headers["content-security-policy"].startswith("default-src 'self'")
        for path in ("/api/config", "/api/population", "/api/population/databases",
                     "/api/session", "/api/jobs/x"):
            assert client.get(path).status_code == 401, path
        assert client.post("/api/summarize", json={"text": "Synthetic"}).status_code == 401
        assert "gf_phi_session" not in client.cookies
        assert client.get("/static/style.css").status_code == 200
        assert "Access token" in client.get("/login").text
        assert "Bearer" not in client.get("/login").text and ACCESS_TOKEN not in client.get("/login").text

        wrong = client.post("/login", data={"token": ACCESS_TOKEN + "x"}, follow_redirects=False)
        assert wrong.status_code == 401 and "not accepted" in wrong.text
        assert "gf_phi_auth" not in client.cookies
        assert client.get("/login", params={"token": "short"}).status_code == 401

        signed = client.post("/login", data={"token": ACCESS_TOKEN}, follow_redirects=False)
        assert signed.status_code == 303 and signed.headers["location"] == "/"
        cookie = client.cookies["gf_phi_auth"]
        assert ACCESS_TOKEN not in cookie
        assert "httponly" in signed.headers["set-cookie"].lower()
        assert "samesite=strict" in signed.headers["set-cookie"].lower()
        assert client.get("/").status_code == 200
        assert client.get("/api/config").status_code == 200

        client.post("/logout", follow_redirects=False)
        assert client.get("/api/config").status_code == 401
        link = client.get("/login", params={"token": ACCESS_TOKEN}, follow_redirects=False)
        assert link.status_code == 303 and client.get("/api/config").status_code == 200

    # A new token signs out browsers holding a cookie from the previous one.
    other = create_app(Settings(phi=True, catalog_root=tmp_path), FakeInference(),
                       access_token="another-fabricated-access-token-abcdef0123456789")
    with TestClient(other, base_url="http://127.0.0.1:8044") as client:
        client.headers.pop("Authorization")
        client.cookies.set("gf_phi_auth", cookie)
        assert client.get("/api/config").status_code == 401
        client.headers["Authorization"] = f"Bearer {ACCESS_TOKEN}"
        assert client.get("/api/config").status_code == 401
        client.headers["Authorization"] = "Bearer another-fabricated-access-token-abcdef0123456789"
        assert client.get("/api/config").status_code == 200


def test_non_phi_app_has_no_sign_in_or_phi_controls(setup, tmp_path, monkeypatch):
    # The non-PHI app never loads an access token: none is available here.
    monkeypatch.delenv("GUIDELINE_ACCESS_TOKEN")
    monkeypatch.setenv("GUIDELINE_ACCESS_TOKEN_FILE", "/nonexistent/guideline-fetcher-token")
    app = create_app(Settings(catalog_root=tmp_path), FakeInference())
    with TestClient(setup[0]) as client, TestClient(app) as fresh:
        client.headers.pop("Authorization")
        page = client.get("/")
        assert page.status_code == 200 and '<html lang="en" data-mode="standard">' in page.text
        assert "Load synthetic population" in page.text
        for phi_only in ("pop-cache", "pop-database", "Sign out", "real patient records", "mode:"):
            assert phi_only not in page.text, phi_only
        assert client.get("/login").status_code == 404
        assert client.post("/logout").status_code == 404
        assert client.get("/api/config").status_code == 200
        assert "gf_session" in client.cookies and "gf_phi_session" not in client.cookies
        defaults = client.get("/api/population").json()["defaults"]
        assert set(defaults) == {"size", "seed", "extraction_records"}
        assert "phi" not in Path(defaults["extraction_records"]).parts
        for path in ("/api/population/databases", "/api/population/database",
                     "/api/population/cache"):
            assert client.get(path).status_code == 404, path
        rejected = client.post("/api/population/sample",
                               json={"size": 1, "seed": 1, "database": "/fabricated.duckdb"})
        assert rejected.status_code == 422
        assert fresh.get("/api/config").status_code == 200


def test_phi_page_has_sign_out_and_population_database_controls(tmp_path):
    catalog(tmp_path)
    app = create_app(Settings(phi=True, catalog_root=tmp_path), FakeInference())
    with TestClient(app, base_url="http://127.0.0.1:8044") as client:
        page = client.get("/")
        assert page.status_code == 200 and '<html lang="en" data-mode="phi">' in page.text
        for phi_only in ('action="/logout"', 'id="pop-cache"', 'id="pop-database"',
                         'id="pop-database-dialog"', "authorized for PHI"):
            assert phi_only in page.text, phi_only
        assert "Load synthetic population" not in page.text and "mode:" not in page.text
        assert page.text.count('id="pop-load"') == 1
        assert "gf_phi_session" in client.cookies and "gf_session" not in client.cookies


def test_access_token_file_is_private_and_reused(tmp_path, monkeypatch):
    import os
    import stat

    from matchminer_ai.apps.guideline_fetcher.settings import load_access_token

    monkeypatch.delenv("GUIDELINE_ACCESS_TOKEN")
    path = tmp_path / "config" / "access_token"
    monkeypatch.setenv("GUIDELINE_ACCESS_TOKEN_FILE", str(path))
    token = load_access_token()
    assert len(token) >= 32 and load_access_token() == token
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) & 0o077 == 0
    os.chmod(path, 0o644)
    with pytest.raises(PermissionError, match="chmod 600"):
        load_access_token()
    monkeypatch.setenv("GUIDELINE_ACCESS_TOKEN", "too-short")
    with pytest.raises(ValueError, match="at least 32"):
        load_access_token()
    monkeypatch.setenv("GUIDELINE_ACCESS_TOKEN", "x" * 40)
    assert load_access_token() == "x" * 40


def test_run_prints_token_only_to_an_interactive_terminal(monkeypatch, capsys):
    from matchminer_ai.apps.guideline_fetcher import run
    from .conftest import ACCESS_TOKEN

    served_apps(monkeypatch)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    run.main(["--phi"])
    logged = capsys.readouterr().out
    assert ACCESS_TOKEN not in logged and "GUIDELINE_ACCESS_TOKEN" in logged
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    run.main(["--phi"])
    assert f"http://127.0.0.1:8044/login?token={ACCESS_TOKEN}" in capsys.readouterr().out
    run.main([])
    assert capsys.readouterr().out == ""
