"""
Scalekit OAuth 2.0 / OIDC authentication layer for Trace.

Flow:
  1. Client calls GET /auth/login → receives authorization_url.
  2. Client redirects the user's browser to authorization_url.
  3. Scalekit authenticates the user (SSO, Google, GitHub, etc.).
  4. Scalekit redirects to GET /auth/callback?code=...&state=...
  5. Server exchanges code for tokens via authenticate_with_code().
  6. Client receives access_token and includes it as:
       Authorization: Bearer <access_token>
  7. Server validates each request token via validate_access_token_and_get_claims().

WHY lru_cache ON get_scalekit_client:
  ScalekitClient fetches its JWKS keys on first token validation (network call
  to env_url/.well-known/jwks.json). A singleton reuses the fetched keys
  across requests rather than re-fetching per call.

WHY asyncio.to_thread FOR SDK CALLS:
  The Scalekit Python SDK is synchronous (uses the requests library internally).
  All blocking calls are wrapped in asyncio.to_thread to keep FastAPI handlers
  non-blocking.

WHY OPTIONAL SCALEKIT CONFIG:
  In local development and CI, Scalekit credentials may not be available.
  Making the fields optional lets the service start and the pipeline run
  without auth. Auth endpoints return 503 when Scalekit is unconfigured.
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from typing import Any

from fastapi import HTTPException, status

_log = logging.getLogger(__name__)

from trace.config import get_settings

# Lazy imports — scalekit is optional. The SDK is only needed when Scalekit
# credentials are configured. Without them the auth endpoints return 503 and
# the newsletter pipeline runs unauthenticated, which is fine for local dev.
try:
    from scalekit import AuthorizationUrlOptions, CodeAuthenticationOptions, ScalekitClient
    from scalekit.client import ScalekitValidateTokenFailureException
    _SCALEKIT_AVAILABLE = True
except ModuleNotFoundError:
    _SCALEKIT_AVAILABLE = False
    AuthorizationUrlOptions = None  # type: ignore[assignment,misc]
    CodeAuthenticationOptions = None  # type: ignore[assignment,misc]
    ScalekitClient = None  # type: ignore[assignment,misc]
    ScalekitValidateTokenFailureException = Exception  # type: ignore[assignment,misc]


@lru_cache(maxsize=1)
def get_scalekit_client() -> "ScalekitClient | None":
    """
    Return a cached ScalekitClient, or None if Scalekit is not configured.

    Returns None rather than raising so that callers can degrade gracefully
    (auth endpoints return 503) instead of crashing the whole service.
    Catches ScalekitForbiddenException (host not in allowlist) and any other
    constructor errors — all treated as "not configured".
    """
    if not _SCALEKIT_AVAILABLE:
        return None
    settings = get_settings()
    if not (settings.scalekit_env_url and settings.scalekit_client_id and settings.scalekit_client_secret):
        return None
    try:
        return ScalekitClient(
            env_url=settings.scalekit_env_url,
            client_id=settings.scalekit_client_id,
            client_secret=settings.scalekit_client_secret,
        )
    except Exception as exc:
        _log.warning(
            "Scalekit client init failed (%s) — running without Scalekit integration. "
            "Tier A/B actions will use stub mode.",
            str(exc).split('\n')[0],
        )
        return None


def _require_client() -> ScalekitClient:
    """Return the Scalekit client or raise 503 if not configured."""
    client = get_scalekit_client()
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scalekit authentication is not configured on this server.",
        )
    return client


class UserClaims:
    """
    Typed wrapper around the JWT claims returned by Scalekit token validation.

    Scalekit issues standard OIDC JWTs:
      sub   → unique user ID
      email → user's email (from profile/email scope)
      name  → display name (from profile scope)
      oid   → organization identifier (Scalekit-specific)
    """

    __slots__ = ("_claims", "user_id", "email", "name", "organization_id")

    def __init__(self, claims: dict[str, Any]) -> None:
        self._claims = claims
        self.user_id: str = str(claims.get("sub", ""))
        self.email: str = str(claims.get("email", ""))
        self.name: str = str(claims.get("name", ""))
        self.organization_id: str = str(claims.get("oid", ""))

    @property
    def display_name(self) -> str:
        """Best human-readable identifier: name → email → user_id."""
        return self.name or self.email or self.user_id

    @property
    def raw(self) -> dict[str, Any]:
        return dict(self._claims)


def build_login_url(redirect_uri: str, state: str | None = None) -> str:
    """
    Build the Scalekit OAuth authorization URL.

    The client redirects the user's browser to this URL. Scalekit handles
    identity provider selection, SSO, MFA, etc., then redirects back to
    redirect_uri with ?code=...&state=...
    """
    client = _require_client()
    options = AuthorizationUrlOptions()
    options.scopes = ["openid", "profile", "email"]
    if state:
        options.state = state
    return client.get_authorization_url(redirect_uri=redirect_uri, options=options)


async def exchange_code(code: str, redirect_uri: str) -> dict[str, Any]:
    """
    Exchange an OAuth authorization code for tokens.

    Returns a dict with keys:
      user           → dict of user profile fields
      access_token   → JWT for authorizing API requests
      id_token       → OIDC identity token
      refresh_token  → optional refresh token
      expires_in     → seconds until access_token expiry
      organization_id, connection_id
    """
    client = _require_client()
    options = CodeAuthenticationOptions()
    return await asyncio.to_thread(
        client.authenticate_with_code, code, redirect_uri, options
    )


async def verify_token(token: str) -> UserClaims:
    """
    Validate a Scalekit access token and return its claims.

    Scalekit validates the JWT signature against its JWKS endpoint, checks
    expiry, and verifies issuer. The call is blocking (network on first use
    to fetch JWKS), wrapped in asyncio.to_thread.

    Raises:
        HTTPException 401: token invalid, expired, or malformed.
        HTTPException 503: Scalekit not configured.
    """
    client = _require_client()
    try:
        claims: dict[str, Any] = await asyncio.to_thread(
            client.validate_access_token_and_get_claims, token
        )
    except ScalekitValidateTokenFailureException as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token validation failed",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    return UserClaims(claims)


# ── Scalekit Agent Connect ─────────────────────────────────────────────────
#
# Three helpers that wrap the official client.connect.* namespace.
# All SDK calls are synchronous; wrapped in asyncio.to_thread for FastAPI.


async def connect_get_authorization_link(
    identifier: str,
    connection_name: str,
    state: str | None = None,
    redirect_url: str | None = None,
) -> str:
    """Return a Scalekit magic link the user clicks to connect a third-party account.

    After the user approves, Scalekit stores the OAuth token in its Vault and
    the agent can call execute_tool() against that connection on their behalf.
    Returns "" when Scalekit is unconfigured.
    """
    client = get_scalekit_client()
    if client is None:
        return ""

    def _call() -> Any:
        kwargs: dict[str, Any] = {
            "identifier": identifier,
            "connection_name": connection_name,
        }
        if state:
            kwargs["state"] = state
        if redirect_url:
            kwargs["user_verify_url"] = redirect_url
        return client.connect.get_authorization_link(**kwargs)  # type: ignore[union-attr]

    try:
        result = await asyncio.to_thread(_call)
        if hasattr(result, "link"):
            return str(result.link)
        if isinstance(result, dict):
            return str(result.get("link", ""))
        return str(result)
    except Exception as exc:
        _log.warning("Scalekit get_authorization_link failed: %s", exc)
        return ""


async def connect_ensure_account(
    identifier: str,
    connection_name: str,
) -> dict[str, Any]:
    """Create or fetch a connected account for (identifier, connection_name).

    Idempotent — safe to call before every magic-link generation.
    Returns {} when Scalekit is unconfigured or on error.
    """
    client = get_scalekit_client()
    if client is None:
        return {}

    def _call() -> Any:
        return client.connect.get_or_create_connected_account(  # type: ignore[union-attr]
            connection_name=connection_name,
            identifier=identifier,
        )

    try:
        result = await asyncio.to_thread(_call)
        if hasattr(result, "connected_account"):
            acc = result.connected_account
            return {
                "id": getattr(acc, "id", ""),
                "status": getattr(acc, "status", ""),
                "connection_name": connection_name,
                "identifier": identifier,
            }
        if isinstance(result, dict):
            return result
        return {"raw": str(result)}
    except Exception as exc:
        _log.warning("Scalekit connect_ensure_account failed: %s", exc)
        return {}


async def connect_execute_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    identifier: str,
    connection_name: str | None = None,
) -> dict[str, Any]:
    """Execute a tool against a connected account through Scalekit.

    connection_name: Scalekit connector slug (e.g. "slack-RLnbqcmP").
      Required to route calls to the correct connector when multiple
      connectors of the same type exist in a workspace.

    The Apify token lives in Scalekit's Vault — never in our env vars.
    Returns {} when Scalekit is unconfigured or on error.
    """
    client = get_scalekit_client()
    if client is None:
        return {}

    def _call() -> Any:
        kwargs: dict[str, Any] = {
            "tool_name": tool_name,
            "tool_input": tool_input,
            "identifier": identifier,
        }
        if connection_name:
            kwargs["connection_name"] = connection_name
        return client.connect.execute_tool(**kwargs)  # type: ignore[union-attr]

    try:
        result = await asyncio.to_thread(_call)
        if isinstance(result, dict):
            return result
        if hasattr(result, "__dict__"):
            return dict(result.__dict__)
        return {"raw": str(result)}
    except Exception as exc:
        err_str = str(exc)
        # RESOURCE_NOT_FOUND / NOT_FOUND = connector or account not yet configured.
        # Log at DEBUG to avoid spam before the user sets up the connector.
        if (
            "RESOURCE_NOT_FOUND" in err_str
            or "failed to get tool" in err_str
            or ("NOT_FOUND" in err_str and "connected account not found" in err_str)
        ):
            _log.debug(
                "Scalekit connect_execute_tool(%s): connector not configured yet (%s)",
                tool_name, err_str.split('\n')[0],
            )
        else:
            _log.warning("Scalekit connect_execute_tool(%s) failed: %s", tool_name, exc)
        return {}
