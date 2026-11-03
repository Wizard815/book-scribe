"""One writer account per container, plus a separate admin login.

  writer   username + password, stored hashed (scrypt) in <state>/account.json; created by the admin (or seeded from
           WRITER_USERNAME / WRITER_PASSWORD on first start). Can use the book, chat and settings. Cannot touch admin.
  admin    ADMIN_USER (default "admin") + ADMIN_TOKEN as the password; manages the writer account, models, embeddings.

Sessions are signed cookies (HttpOnly, SameSite=Strict, Secure behind https). A writer session carries the account's
version, so changing the password or username logs every existing session out. Wrong passwords are rate limited.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path

COOKIE = "bs_session"
WRITER_TTL = 14 * 24 * 3600
ADMIN_TTL = 8 * 3600
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.@\-]{3,40}$")
MIN_PASSWORD = 8
WEAK_TOKENS = {"change-me", "changeme", "admin", "password", "token", "secret"}

_N, _R, _P = 2 ** 14, 8, 1


class AuthError(Exception):
    pass


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    h = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return "scrypt${}${}${}${}${}".format(_N, _R, _P, base64.b64encode(salt).decode(), base64.b64encode(h).decode())


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, h = stored.split("$")
        got = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p), dklen=32)
        return hmac.compare_digest(got, base64.b64decode(h))
    except Exception:
        return False


_DUMMY = hash_password("not-a-real-password")  # verified against when the username is wrong, so timing doesn't reveal it


def check_password_rules(password: str) -> None:
    if len(password) < MIN_PASSWORD:
        raise AuthError(f"Use a password of at least {MIN_PASSWORD} characters.")
    if len(password) > 200:
        raise AuthError("That password is too long.")


def check_username_rules(username: str) -> None:
    if not USERNAME_RE.match(username):
        raise AuthError("Use 3 to 40 letters, numbers, dots, dashes or underscores for the username.")


class AccountStore:
    """The single writer account."""

    def __init__(self, state_dir: Path):
        self.path = state_dir / "account.json"
        state_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _load(self) -> dict | None:
        try:
            return json.loads(self.path.read_text()) if self.path.exists() else None
        except (OSError, ValueError):
            return None

    def get(self) -> dict | None:
        a = self._load()
        return {"username": a["username"], "version": a["version"], "updated": a.get("updated", 0)} if a else None

    def version(self) -> int:
        a = self._load()
        return a["version"] if a else -1

    def set(self, username: str, password: str) -> dict:
        check_username_rules(username)
        check_password_rules(password)
        with self._lock:
            old = self._load()
            data = {"username": username, "hash": hash_password(password), "version": (old["version"] + 1) if old else 1, "updated": int(time.time())}
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data))
            tmp.chmod(0o600)
            tmp.replace(self.path)
        return {"username": username, "version": data["version"]}

    def remove(self) -> None:
        with self._lock:
            if self.path.exists():
                self.path.unlink()

    def verify(self, username: str, password: str) -> bool:
        a = self._load()
        ok_user = bool(a) and hmac.compare_digest(a["username"].encode(), username.encode())
        ok_pw = verify_password(password, a["hash"] if a else _DUMMY)  # always do the work
        return ok_user and ok_pw


class Sessions:
    def __init__(self, state_dir: Path):
        p = state_dir / "session_secret"
        if not p.exists():
            p.write_bytes(secrets.token_bytes(32))
            p.chmod(0o600)
        self.secret = p.read_bytes()

    def issue(self, role: str, username: str, version, ttl: int) -> str:
        payload = base64.urlsafe_b64encode(json.dumps({"r": role, "u": username, "v": version, "e": int(time.time()) + ttl}, separators=(",", ":")).encode()).decode().rstrip("=")
        sig = hmac.new(self.secret, payload.encode(), hashlib.sha256).hexdigest()
        return f"{payload}.{sig}"

    def parse(self, token: str | None) -> dict | None:
        if not token or "." not in token:
            return None
        payload, _, sig = token.partition(".")
        if not hmac.compare_digest(sig, hmac.new(self.secret, payload.encode(), hashlib.sha256).hexdigest()):
            return None
        try:
            d = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        except Exception:
            return None
        return d if d.get("e", 0) > time.time() else None


class LoginLimiter:
    """Slows password guessing: too many failures from one address, or too many overall, lock logins for a while."""

    def __init__(self, per_key: int = 5, overall: int = 40, window: int = 600, lockout: int = 300, now=time.monotonic):
        self.per_key, self.overall, self.window, self.lockout, self.now = per_key, overall, window, lockout, now
        self.fails: dict[str, list[float]] = {}
        self.locked: dict[str, float] = {}

    def _prune(self, key: str) -> list[float]:
        t = self.now()
        self.fails[key] = [x for x in self.fails.get(key, []) if t - x < self.window]
        return self.fails[key]

    def wait(self, key: str) -> int:
        """Seconds the caller must wait before trying again (0 = free to try)."""
        t = self.now()
        return max(0, int(max(self.locked.get(key, 0), self.locked.get("*", 0)) - t))

    def fail(self, key: str) -> None:
        t = self.now()
        self._prune(key).append(t)
        self._prune("*").append(t)
        if len(self.fails[key]) >= self.per_key:
            self.locked[key] = t + self.lockout
        if len(self.fails["*"]) >= self.overall:
            self.locked["*"] = t + self.lockout

    def ok(self, key: str) -> None:
        self.fails.pop(key, None)
        self.locked.pop(key, None)


def admin_password_ok(token: str) -> bool:
    return len(token) >= 16 and token.lower() not in WEAK_TOKENS
