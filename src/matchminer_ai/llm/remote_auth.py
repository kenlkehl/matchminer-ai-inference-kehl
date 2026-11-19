"""Authentication and provider helpers for remote LLM endpoints."""

from __future__ import annotations

import asyncio
import os
import threading
from functools import lru_cache
from typing import Any, Awaitable, Callable, Mapping


OPENAI_COMPATIBLE_PROVIDER = "openai"
GOOGLE_AGENT_PLATFORM_PROVIDER = "google_agent_platform"
#: Claude through a Databricks AI Gateway's Anthropic Messages route
#: (``https://<workspace>/ai-gateway/anthropic``), authenticated with a Databricks
#: profile. Only structured requests (``StructuredClient``) support it.
DATABRICKS_ANTHROPIC_PROVIDER = "databricks_anthropic"
GOOGLE_CLOUD_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

AsyncAPIKey = str | Callable[[], Awaitable[str]]


def remote_provider_name(remote_config: Mapping[str, Any]) -> str:
    """Return the normalized remote provider name."""

    raw_provider = str(
        remote_config.get("provider") or OPENAI_COMPATIBLE_PROVIDER
    ).strip()
    normalized = raw_provider.casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "openai": OPENAI_COMPATIBLE_PROVIDER,
        "openai_compatible": OPENAI_COMPATIBLE_PROVIDER,
        "google": GOOGLE_AGENT_PLATFORM_PROVIDER,
        "gemini": GOOGLE_AGENT_PLATFORM_PROVIDER,
        "vertex": GOOGLE_AGENT_PLATFORM_PROVIDER,
        "vertex_ai": GOOGLE_AGENT_PLATFORM_PROVIDER,
        "google_agent_platform": GOOGLE_AGENT_PLATFORM_PROVIDER,
        "databricks": DATABRICKS_ANTHROPIC_PROVIDER,
        "databricks_anthropic": DATABRICKS_ANTHROPIC_PROVIDER,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        allowed = ", ".join(
            (OPENAI_COMPATIBLE_PROVIDER, GOOGLE_AGENT_PLATFORM_PROVIDER,
             DATABRICKS_ANTHROPIC_PROVIDER)
        )
        raise ValueError(
            f"Unsupported remote provider {raw_provider!r}. Use one of: {allowed}."
        ) from exc


class GoogleADCAccessTokenProvider:
    """Refresh Google Application Default Credentials and return access tokens."""

    def __init__(self, project_id: str = "") -> None:
        try:
            import google.auth
            from google.auth.transport.requests import Request
        except ImportError as exc:
            raise RuntimeError(
                "Google Agent Platform authentication requires the google-auth "
                "package. Install matchminer-ai with its current dependencies."
            ) from exc

        self._request_type = Request
        self._credentials, self.detected_project_id = google.auth.default(
            scopes=[GOOGLE_CLOUD_SCOPE],
            quota_project_id=project_id or None,
        )
        self.project_id = project_id or str(self.detected_project_id or "")
        self._lock = threading.Lock()

    def token(self) -> str:
        """Return a valid bearer token, refreshing ADC when needed."""

        with self._lock:
            if not self._credentials.valid:
                self._credentials.refresh(self._request_type())
            token = str(self._credentials.token or "").strip()
            if not token:
                raise RuntimeError(
                    "Google Application Default Credentials returned no access token."
                )
            return token

    async def async_token(self) -> str:
        """Return a valid token without blocking the async request loop."""

        return await asyncio.to_thread(self.token)


@lru_cache(maxsize=8)
def _google_adc_token_provider(project_id: str) -> GoogleADCAccessTokenProvider:
    return GoogleADCAccessTokenProvider(project_id)


def clear_google_adc_credential_cache() -> None:
    """Forget cached ADC credential objects."""

    _google_adc_token_provider.cache_clear()


class DatabricksCredentialsError(RuntimeError):
    """No Databricks access token (missing package, profile or expired login)."""


class DatabricksTokenProvider:
    """Databricks access tokens from a ``databricks-sdk`` WorkspaceClient.

    With a profile (``~/.databrickscfg``, written by ``databricks auth login``), the
    SDK asks the Databricks CLI for tokens and refreshes them. Without one it uses
    the runtime identity of a Databricks notebook or job, or a service principal
    from the ``ARM_*`` environment variables. Tokens are never logged.
    """

    def __init__(self, profile: str = "") -> None:
        try:
            from databricks.sdk import WorkspaceClient
        except ImportError as exc:
            raise DatabricksCredentialsError(
                "Claude through Databricks needs the databricks-sdk and anthropic "
                "packages: pip install 'matchminer-ai[databricks]'."
            ) from exc
        self.profile = profile
        try:
            self._client = WorkspaceClient(profile=profile) if profile else WorkspaceClient()
            self.host = str(self._client.config.host or "")
        except Exception:  # The SDK raises ValueError and its own config errors.
            raise DatabricksCredentialsError(_login_hint(profile)) from None
        self._lock = threading.Lock()

    def token(self) -> str:
        try:
            with self._lock:
                header = self._client.config.authenticate().get("Authorization", "")
        except Exception:
            raise DatabricksCredentialsError(_login_hint(self.profile)) from None
        token = header.removeprefix("Bearer ").strip()
        if not header.startswith("Bearer ") or not token:
            raise DatabricksCredentialsError(_login_hint(self.profile))
        return token


def _login_hint(profile: str) -> str:
    if not profile:
        return "Could not get a Databricks access token from the runtime identity."
    return (f"Could not get a Databricks access token for profile {profile!r}. Log in with "
            f"`databricks auth login --host <workspace URL> --profile {profile}`.")


@lru_cache(maxsize=8)
def databricks_token_provider(profile: str = "") -> DatabricksTokenProvider:
    """One token provider per Databricks profile (failed setups are not cached)."""
    return DatabricksTokenProvider(profile)


def _openai_only(remote_config: Mapping[str, Any]) -> None:
    if remote_provider_name(remote_config) == DATABRICKS_ANTHROPIC_PROVIDER:
        raise ValueError(
            "Claude through Databricks serves structured requests only (such as the SQL "
            "workup agent); choose an OpenAI-compatible or Google provider here."
        )


def remote_api_key(remote_config: Mapping[str, Any]) -> AsyncAPIKey:
    """Return the API key value or refresh callback for an OpenAI client."""

    _openai_only(remote_config)
    if remote_provider_name(remote_config) == GOOGLE_AGENT_PLATFORM_PROVIDER:
        project_id = str(remote_config.get("google_project_id") or "").strip()
        return _google_adc_token_provider(project_id).async_token
    return str(os.environ.get("OPENAI_API_KEY", "not-needed")).strip() or "not-needed"


def remote_bearer_token(
    remote_config: Mapping[str, Any],
    *,
    explicit_api_key: str | None = None,
) -> str:
    """Return a current synchronous bearer token for endpoint diagnostics."""

    _openai_only(remote_config)
    if remote_provider_name(remote_config) == GOOGLE_AGENT_PLATFORM_PROVIDER:
        project_id = str(remote_config.get("google_project_id") or "").strip()
        return _google_adc_token_provider(project_id).token()
    return (
        str(explicit_api_key or "").strip()
        or str(os.environ.get("OPENAI_API_KEY", "not-needed")).strip()
        or "not-needed"
    )


def prepare_messages_for_provider(
    messages: list[dict[str, str]],
    remote_config: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Adapt structured messages to the selected provider's chat contract."""

    normalized_messages = [
        {
            "role": str(message.get("role") or "user"),
            "content": str(message.get("content") or ""),
        }
        for message in messages
    ]
    if remote_provider_name(remote_config) != GOOGLE_AGENT_PLATFORM_PROVIDER:
        return normalized_messages
    if str(remote_config.get("model_name", "")).removeprefix("google/").startswith("gemini-"):
        # Gemini supports system instructions; the flattening below is for MaaS.
        return normalized_messages

    system_parts: list[str] = []
    while normalized_messages and normalized_messages[0]["role"] == "system":
        system_parts.append(normalized_messages.pop(0)["content"])
    if not system_parts:
        return normalized_messages

    instruction_text = "\n\n".join(part for part in system_parts if part).strip()
    if normalized_messages and normalized_messages[0]["role"] == "user":
        user_message = dict(normalized_messages[0])
        user_text = user_message["content"]
        user_message["content"] = (
            "Instructions:\n"
            f"{instruction_text}\n\n"
            "Request:\n"
            f"{user_text}"
        )
        normalized_messages[0] = user_message
    else:
        normalized_messages.insert(
            0,
            {"role": "user", "content": f"Instructions:\n{instruction_text}"},
        )
    return normalized_messages


__all__ = [
    "AsyncAPIKey",
    "DATABRICKS_ANTHROPIC_PROVIDER",
    "DatabricksCredentialsError",
    "DatabricksTokenProvider",
    "GOOGLE_AGENT_PLATFORM_PROVIDER",
    "GOOGLE_CLOUD_SCOPE",
    "OPENAI_COMPATIBLE_PROVIDER",
    "GoogleADCAccessTokenProvider",
    "clear_google_adc_credential_cache",
    "databricks_token_provider",
    "prepare_messages_for_provider",
    "remote_api_key",
    "remote_bearer_token",
    "remote_provider_name",
]
