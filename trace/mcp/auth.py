"""
ScalekitMCPTokenVerifier — secures the Trace MCP server with Scalekit MCP Auth.

Implements the MCP SDK's TokenVerifier protocol against Scalekit's JWKS-backed
OAuth 2.1 authorization server. When SCALEKIT_MCP_RESOURCE_ID is set, the
Trace MCP server enforces bearer-token authentication on every tool call.

MCP clients (Claude Desktop, Cursor) discover the auth endpoint via
/.well-known/oauth-protected-resource and obtain tokens through Scalekit's
authorization server with Dynamic Client Registration.

WHY SCALEKIT MCP AUTH:
  - OAuth 2.1 with PKCE — spec-compliant way to secure remote MCP servers
  - Dynamic Client Registration (DCR) — Claude Desktop auto-registers, no
    manual client provisioning needed
  - JWT audience/scope enforcement — per-tool scope checks possible
  - Revocation from Scalekit dashboard — live demo flourish (revoke token →
    Claude's next call cleanly 401s)

GRACEFUL DEGRADATION:
  When SCALEKIT_MCP_RESOURCE_ID is not set, returns a permissive AccessToken
  for any bearer string (single-user / local-dev mode). Server still boots
  and works in CI without Scalekit credentials.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from mcp.server.auth.middleware.bearer_auth import AccessToken, TokenVerifier

from trace.config import get_settings

_log = logging.getLogger(__name__)

# In-process JWKS cache — refreshed at most once per hour.
_JWKS_CACHE: dict[str, Any] = {"keys": None, "fetched_at": 0.0}
_JWKS_TTL = 3600.0


class ScalekitMCPTokenVerifier(TokenVerifier):
    """Verify MCP bearer tokens issued by Scalekit's OAuth 2.1 server."""

    def __init__(self) -> None:
        self._settings = get_settings()

    @property
    def _enforced(self) -> bool:
        return bool(
            self._settings.scalekit_mcp_resource_id
            and self._settings.scalekit_env_url
        )

    async def _jwks(self) -> dict[str, Any] | None:
        if not self._settings.scalekit_env_url:
            return None
        now = time.time()
        if _JWKS_CACHE["keys"] is not None and now - _JWKS_CACHE["fetched_at"] < _JWKS_TTL:
            return _JWKS_CACHE["keys"]
        url = f"{self._settings.scalekit_env_url}/keys"
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(url)
                resp.raise_for_status()
            _JWKS_CACHE["keys"] = resp.json()
            _JWKS_CACHE["fetched_at"] = now
            return _JWKS_CACHE["keys"]
        except Exception as exc:
            _log.warning("Scalekit JWKS fetch failed: %s", exc)
            return None

    async def verify_token(self, token: str) -> AccessToken | None:
        # Dev / local mode — auth not enforced.
        if not self._enforced:
            return AccessToken(
                token=token,
                client_id="local-dev",
                scopes=["trace:read", "trace:briefing", "trace:track"],
            )

        # Production: validate JWT signature against Scalekit JWKS.
        try:
            import jwt as pyjwt  # PyJWT
        except ImportError:
            _log.error("PyJWT not installed — run: pip install pyjwt[crypto]")
            return None

        jwks = await self._jwks()
        if jwks is None:
            return None
        try:
            unverified = pyjwt.get_unverified_header(token)
            kid = unverified.get("kid")
            key = next((k for k in jwks.get("keys", []) if k.get("kid") == kid), None)
            if key is None:
                _log.warning("Token kid %s not in Scalekit JWKS", kid)
                return None
            public_key = pyjwt.algorithms.RSAAlgorithm.from_jwk(key)
            claims = pyjwt.decode(
                token,
                key=public_key,
                algorithms=[key.get("alg", "RS256")],
                audience=self._settings.public_base_url,
                issuer=self._settings.scalekit_env_url,
            )
        except pyjwt.ExpiredSignatureError:
            return None
        except pyjwt.InvalidTokenError as exc:
            _log.warning("Scalekit token invalid: %s", exc)
            return None

        scope_str = claims.get("scope", claims.get("scp", "")) or ""
        scopes = scope_str.split() if isinstance(scope_str, str) else list(scope_str)
        return AccessToken(
            token=token,
            client_id=str(claims.get("client_id", claims.get("azp", "unknown"))),
            scopes=scopes,
            expires_at=int(claims["exp"]) if claims.get("exp") else None,
            resource=self._settings.public_base_url,
        )
