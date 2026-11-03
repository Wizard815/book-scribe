import pytest
from fastapi.testclient import TestClient

from main import create_app
from memory.store import FileMemory
from settings import Settings
from tests.conftest import FakeLLM

ADMIN = "a-long-enough-admin-secret-1"


def make(books, **kw):
    s = Settings(project_dir=books, admin_token=kw.pop("admin_token", ADMIN), default_book="demo", **kw)
    tts = FakeLLM(); tts.enabled = False
    return create_app(s, llm=FakeLLM(), stt=FakeLLM(), tts=tts, memory=FileMemory(books / "m.json"), check_drift=False)


def with_account(app, user="writer", pw="correct horse battery"):
    c = TestClient(app)
    assert c.put("/api/admin/account", headers={"x-admin-token": ADMIN}, json={"username": user, "password": pw}).status_code == 200
    return c


PROTECTED = [("GET", "/api/books"), ("GET", "/api/books/demo/chapters"), ("GET", "/api/books/demo/chapters/1"), ("GET", "/api/books/demo/voices"),
             ("GET", "/api/books/demo/graph"), ("POST", "/api/chat"), ("POST", "/api/tts"), ("POST", "/api/transcribe"), ("POST", "/api/voice"),
             ("PUT", "/api/books/demo/paragraphs/ch1_s1_p1"), ("POST", "/api/account/password")]


@pytest.mark.parametrize("method,path", PROTECTED)
def test_nothing_the_writer_uses_is_reachable_without_logging_in(books, method, path):
    with TestClient(make(books)) as c:
        r = c.request(method, path, json={"text": "x", "current": "x", "new_password": "x"})
        assert r.status_code == 401, (method, path, r.status_code)


def test_admin_routes_need_the_admin_not_the_writer(books):
    with TestClient(make(books)) as anon:
        for path in ("/api/admin/providers", "/api/admin/status", "/api/admin/account"):
            assert anon.get(path).status_code == 401


def test_health_is_public_but_reveals_nothing(books):
    with TestClient(make(books)) as c:
        assert c.get("/api/health").json() == {"ok": True}
        st = c.get("/api/auth/status").json()
        assert st == {"writer_account": False, "admin_enabled": True, "auth_required": True}


def test_writer_login_logout_and_cookie_flags(books):
    app = make(books)
    admin = with_account(app)
    with TestClient(app) as w:
        assert w.post("/api/auth/login", json={"username": "writer", "password": "wrong password"}).status_code == 401
        r = w.post("/api/auth/login", json={"username": "writer", "password": "correct horse battery"})
        assert r.status_code == 200 and r.json() == {"role": "writer", "username": "writer"}
        cookie = r.headers["set-cookie"].lower()
        assert "httponly" in cookie and "samesite=strict" in cookie and "path=/" in cookie and "secure" not in cookie
        assert w.get("/api/auth/me").json() == {"role": "writer", "username": "writer"}
        assert w.get("/api/books").status_code == 200
        assert w.post("/api/auth/logout").status_code == 200
        assert w.get("/api/books").status_code == 401 and w.get("/api/auth/me").status_code == 401


def test_cookie_is_marked_secure_behind_https(books):
    app = make(books)
    with_account(app)
    with TestClient(app) as w:
        r = w.post("/api/auth/login", json={"username": "writer", "password": "correct horse battery"}, headers={"x-forwarded-proto": "https"})
        assert "secure" in r.headers["set-cookie"].lower()


def test_wrong_username_and_wrong_password_give_the_same_answer(books):
    app = make(books)
    with_account(app)
    with TestClient(app) as w:
        a = w.post("/api/auth/login", json={"username": "writer", "password": "nope nope nope"})
        b = w.post("/api/auth/login", json={"username": "nobody", "password": "correct horse battery"})
        assert (a.status_code, a.json()) == (b.status_code, b.json()) == (401, {"detail": "Wrong username or password."})


def test_guessing_is_throttled(books):
    app = make(books)
    with_account(app)
    with TestClient(app) as w:
        codes = [w.post("/api/auth/login", json={"username": "writer", "password": f"guess number {i}"}).status_code for i in range(7)]
        assert codes[:5] == [401] * 5 and codes[5:] == [429, 429]
        locked = w.post("/api/auth/login", json={"username": "writer", "password": "correct horse battery"})
        assert locked.status_code == 429 and int(locked.headers["retry-after"]) > 0  # even the right password waits


def test_the_two_logins_are_separate_and_roles_do_not_cross(books):
    app = make(books)
    with_account(app)
    with TestClient(app) as w, TestClient(app) as a:
        w.post("/api/auth/login", json={"username": "writer", "password": "correct horse battery"})
        assert w.get("/api/admin/providers").status_code == 401  # a writer is not an admin
        assert a.post("/api/auth/login", json={"username": "admin", "password": ADMIN}).status_code == 401  # nor the admin a writer
        r = a.post("/api/auth/admin-login", json={"username": "admin", "password": ADMIN})
        assert r.status_code == 200 and a.get("/api/auth/me").json()["role"] == "admin"
        assert a.get("/api/admin/providers").status_code == 200
        assert a.get("/api/books").status_code == 401  # the admin cannot read the book
        assert w.post("/api/auth/admin-login", json={"username": "writer", "password": "correct horse battery"}).status_code == 401


def test_admin_login_needs_the_right_username_and_a_strong_token(books):
    with TestClient(make(books)) as c:
        assert c.post("/api/auth/admin-login", json={"username": "root", "password": ADMIN}).status_code == 401
        assert c.post("/api/auth/admin-login", json={"username": "admin", "password": "x"}).status_code == 401
    with TestClient(make(books, admin_token="change-me")) as c:
        assert c.post("/api/auth/admin-login", json={"username": "admin", "password": "change-me"}).status_code == 503
        assert c.get("/api/auth/status").json()["admin_enabled"] is False
    with TestClient(make(books, admin_user="boss")) as c:
        assert c.post("/api/auth/admin-login", json={"username": "boss", "password": ADMIN}).status_code == 200


def test_changing_the_password_needs_the_current_one_and_ends_other_sessions(books):
    app = make(books)
    with_account(app)
    with TestClient(app) as phone, TestClient(app) as laptop:
        for c in (phone, laptop):
            c.post("/api/auth/login", json={"username": "writer", "password": "correct horse battery"})
        assert phone.post("/api/account/password", json={"current": "wrong wrong", "new_password": "a brand new password"}).status_code == 401
        assert phone.post("/api/account/password", json={"current": "correct horse battery", "new_password": "short"}).status_code == 400
        r = phone.post("/api/account/password", json={"current": "correct horse battery", "new_password": "a brand new password", "new_username": "novelist"})
        assert r.status_code == 200 and r.json()["username"] == "novelist"
        assert phone.get("/api/books").status_code == 200  # this device stays logged in
        assert laptop.get("/api/books").status_code == 401  # every other session ended
        with TestClient(app) as fresh:
            assert fresh.post("/api/auth/login", json={"username": "writer", "password": "correct horse battery"}).status_code == 401
            assert fresh.post("/api/auth/login", json={"username": "novelist", "password": "a brand new password"}).status_code == 200


def test_admin_resetting_the_account_logs_the_writer_out(books):
    app = make(books)
    admin = with_account(app)
    with TestClient(app) as w:
        w.post("/api/auth/login", json={"username": "writer", "password": "correct horse battery"})
        assert w.get("/api/books").status_code == 200
        admin.put("/api/admin/account", headers={"x-admin-token": ADMIN}, json={"username": "writer", "password": "reset by the admin"})
        assert w.get("/api/books").status_code == 401
        admin.delete("/api/admin/account", headers={"x-admin-token": ADMIN})
        assert w.post("/api/auth/login", json={"username": "writer", "password": "reset by the admin"}).status_code == 401


def test_account_rules(books):
    app = make(books)
    c = TestClient(app)
    h = {"x-admin-token": ADMIN}
    assert c.put("/api/admin/account", headers=h, json={"username": "ab", "password": "long enough pw"}).status_code == 400
    assert c.put("/api/admin/account", headers=h, json={"username": "has space", "password": "long enough pw"}).status_code == 400
    assert c.put("/api/admin/account", headers=h, json={"username": "writer", "password": "short"}).status_code == 400
    assert c.get("/api/admin/account", headers=h).json()["exists"] is False


def test_passwords_are_stored_hashed_and_the_file_is_private(books):
    app = make(books)
    with_account(app, pw="plain secret phrase")
    f = books / ".state" / "account.json"
    assert "plain secret phrase" not in f.read_text() and "scrypt$" in f.read_text() and (f.stat().st_mode & 0o077) == 0
    assert ((books / ".state" / "session_secret").stat().st_mode & 0o077) == 0


def test_forged_and_expired_sessions_are_rejected(books):
    app = make(books)
    with_account(app)
    from auth import Sessions
    sess = Sessions(books / ".state")
    with TestClient(app) as c:
        c.cookies.set("bs_session", "eyJyIjoid3JpdGVyIiwidSI6IngiLCJ2IjoxLCJlIjo5OTk5OTk5OTk5fQ.deadbeef")
        assert c.get("/api/books").status_code == 401  # bad signature
        c.cookies.set("bs_session", sess.issue("writer", "writer", 1, -10))
        assert c.get("/api/books").status_code == 401  # expired
        c.cookies.set("bs_session", sess.issue("writer", "writer", 1, 60))
        assert c.get("/api/books").status_code == 200  # a genuine one for the current account version


def test_cross_site_posts_are_refused(books):
    app = make(books)
    with_account(app)
    with TestClient(app) as w:
        w.post("/api/auth/login", json={"username": "writer", "password": "correct horse battery"})
        assert w.post("/api/chat", json={"text": "hi"}, headers={"origin": "https://evil.example"}).status_code == 403
        assert w.post("/api/chat", json={"text": "hi"}, headers={"origin": "http://testserver"}).status_code != 403


def test_security_headers_and_no_caching_of_api_answers(books):
    with TestClient(make(books)) as c:
        r = c.get("/api/health")
        assert r.headers["x-content-type-options"] == "nosniff" and r.headers["x-frame-options"] == "DENY" and r.headers["cache-control"] == "no-store"


def test_account_can_be_seeded_from_the_environment_but_not_with_a_weak_password(books):
    with TestClient(make(books, writer_username="seeded", writer_password="a good long password")) as c:
        assert c.post("/api/auth/login", json={"username": "seeded", "password": "a good long password"}).status_code == 200
    import shutil; shutil.rmtree(books / ".state")
    with TestClient(make(books, writer_username="seeded", writer_password="short")) as c:
        assert c.get("/api/auth/status").json()["writer_account"] is False


def test_admin_header_token_still_works_for_scripts(books):
    with TestClient(make(books)) as c:
        assert c.get("/api/admin/providers", headers={"x-admin-token": ADMIN}).status_code == 200
        assert c.get("/api/admin/providers", headers={"x-admin-token": ADMIN + "x"}).status_code == 401


def test_admin_status_has_the_details_that_health_hides(books):
    with TestClient(make(books)) as c:
        s = c.get("/api/admin/status", headers={"x-admin-token": ADMIN}).json()
        assert {"memory", "graph", "kg"} <= set(s)
