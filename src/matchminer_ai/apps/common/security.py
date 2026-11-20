"""Web security the apps share.

- ``is_loopback``: an app that serves confidential content binds only to this machine.
- ``LOOPBACK_HOSTS``: Host headers a loopback app accepts (rejecting DNS-rebinding requests from web pages).
- ``content_security_policy`` and ``security_headers``: same-origin pages, no framing, no referrer, no sniffing.
- ``cross_origin_write``: why a state-changing request from another origin is refused (None when it is not).
- ``load_access_token`` and ``SignIn``: an access token that keeps out other accounts on the same machine (which
  can reach a loopback port), from an environment variable or a private token file created on first use, checked
  as a bearer token or a sign-in cookie derived from it (a new token signs every browser out).
"""

from __future__ import annotations

import hmac
import ipaddress
import os
import secrets
from pathlib import Path
from urllib.parse import urlsplit

LOOPBACK_HOSTS = ["127.0.0.1", "localhost"]
MIN_ACCESS_TOKEN_LENGTH = 32


def is_loopback(host: str) -> bool:
    """Whether ``host`` (a bind address) is this machine only."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def content_security_policy(*, inline_styles: bool = False, form_action: str = "'self'") -> str:
    styles = "'self' 'unsafe-inline'" if inline_styles else "'self'"
    return (f"default-src 'self'; script-src 'self'; style-src {styles}; img-src 'self' data:; connect-src 'self'; "
            f"frame-ancestors 'none'; base-uri 'self'; form-action {form_action}")


def security_headers(csp: str) -> dict[str, str]:
    """Headers every response gets (callers add their own Cache-Control)."""
    return {"X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "Content-Security-Policy": csp}


def cross_origin_write(method: str, headers) -> str | None:
    """Why a request is refused as a cross-origin write (None when it is a read or same-origin)."""
    if method in {"GET", "HEAD", "OPTIONS"}:
        return None
    origin = headers.get("origin")
    if origin and urlsplit(origin).netloc != headers.get("host"):
        return "Cross-origin request rejected"
    if headers.get("sec-fetch-site") == "cross-site":
        return "Cross-site request rejected"
    return None


def load_access_token(env_var: str, token_file: str | os.PathLike) -> str:
    """``$env_var``, else the private token file (created 0600 in a 0700 folder on first use).

    The loopback bind keeps other machines out; the token keeps out other accounts on this machine. Never pass
    the token on a command line (ps shows it).
    """
    token = os.getenv(env_var, "").strip()
    if not token:
        path = Path(token_file).expanduser()
        if not path.exists():
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w") as file:
                    file.write(secrets.token_urlsafe(32) + "\n")
            except FileExistsError:
                pass
        if path.stat().st_mode & 0o077:
            raise PermissionError(f"Restrict the access token file to your account: chmod 600 {path}")
        token = path.read_text().strip()
    if len(token) < MIN_ACCESS_TOKEN_LENGTH or any(ch.isspace() for ch in token):
        raise ValueError(f"The access token needs at least {MIN_ACCESS_TOKEN_LENGTH} characters and no spaces.")
    return token


class SignIn:
    """Checks a request's bearer token or sign-in cookie against an access token."""

    def __init__(self, access_token: str, purpose: bytes, cookie_name: str):
        self.access_token = access_token
        self.cookie_name = cookie_name
        # A cookie derived from the token: a new token signs every browser out.
        self.cookie_value = hmac.new(access_token.encode(), purpose, "sha256").hexdigest()

    def valid_token(self, value: str) -> bool:
        return hmac.compare_digest(value.strip().encode(), self.access_token.encode())

    def authorized(self, request) -> bool:
        cookie = request.cookies.get(self.cookie_name, "")
        scheme, _, bearer = request.headers.get("authorization", "").partition(" ")
        return hmac.compare_digest(cookie.encode(), self.cookie_value.encode()) or (
            scheme.lower() == "bearer" and self.valid_token(bearer))

    def set_cookie(self, request, response):
        response.set_cookie(self.cookie_name, self.cookie_value, httponly=True, samesite="strict",
                            secure=request.url.scheme == "https")
        return response
