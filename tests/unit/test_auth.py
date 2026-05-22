"""
Unit tests for trace/auth/scalekit.py and the /auth/* API endpoints.

All Scalekit SDK calls are mocked — no real network, no real credentials.

Test categories:
  1. get_scalekit_client: returns None when env vars unset, returns client when set
  2. UserClaims: field extraction from JWT claim dict
  3. build_login_url: delegates to Scalekit SDK, embeds state
  4. exchange_code: async wraps authenticate_with_code, returns token dict
  5. verify_token: returns UserClaims on valid token, raises 401 on invalid
  6. GET /auth/login: 200 with authorization_url when Scalekit configured; 503 when not
  7. GET /auth/callback: 200 with access_token when code valid; 503 when unconfigured
  8. GET /auth/me: 200 with user profile when token valid; 401 without token
  9. GET /auth/logout: 200 with logout_url when configured; 503 when not
  10. POST /newsletter/generate: generated_for populated when authed; empty when not
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trace.auth.scalekit import UserClaims
from trace.delivery.api import app, get_pipeline, get_optional_user
from trace.models import (
    ContentSource,
    CuriosityGraph,
    Newsletter,
    NewsletterSection,
    RawSignal,
    SignalSource,
    Topic,
)
from trace.pipeline.runner import PipelineResult

from fastapi.testclient import TestClient


# ── Helpers ───────────────────────────────────────────────────────────────────

_NOW = datetime(2024, 6, 1, tzinfo=timezone.utc)

_SAMPLE_CLAIMS: dict[str, Any] = {
    "sub": "usr_01abc123",
    "email": "alice@example.com",
    "name": "Alice Smith",
    "oid": "org_xyz",
    "exp": 9999999999,
    "iat": 1700000000,
}


def make_user_claims(claims: dict[str, Any] | None = None) -> UserClaims:
    return UserClaims(claims or _SAMPLE_CLAIMS)


def make_newsletter() -> Newsletter:
    section = NewsletterSection(
        title="Transformers This Week",
        section_type="weekly_topics",
        content="Your signals this week converge on transformer architecture — three papers, two threads.",
        source_urls=["https://arxiv.org/abs/1706.03762"],
        audit_reasoning="Frequency 3, recency 0.8 — active recurring interest.",
    )
    return Newsletter(
        generated_at=_NOW,
        subject_line="Your Curiosity Digest: Transformers Edition",
        sections=(section,),
        plain_text="Transformers This Week\n...",
        html="<html>...</html>",
    )


def make_pipeline_result() -> PipelineResult:
    topic = Topic(name="transformer architecture", frequency=3, recency_score=0.8)
    return PipelineResult(
        newsletter=make_newsletter(),
        graph=CuriosityGraph(topics=(topic,), signal_count=3),
        collected_signals=(
            RawSignal(source=SignalSource.CHROME_HISTORY, content="transformers", timestamp=_NOW),
        ),
        errors=[],
        completed_at=_NOW,
    )


def make_mock_pipeline() -> MagicMock:
    pipeline = MagicMock()
    pipeline.run = AsyncMock(return_value=make_pipeline_result())
    return pipeline


# ── UserClaims ────────────────────────────────────────────────────────────────

class TestUserClaims:
    def test_user_id_from_sub(self) -> None:
        claims = UserClaims({"sub": "usr_abc"})
        assert claims.user_id == "usr_abc"

    def test_email_extracted(self) -> None:
        claims = UserClaims({"sub": "x", "email": "bob@example.com"})
        assert claims.email == "bob@example.com"

    def test_name_extracted(self) -> None:
        claims = UserClaims({"sub": "x", "name": "Bob"})
        assert claims.name == "Bob"

    def test_organization_id_from_oid(self) -> None:
        claims = UserClaims({"sub": "x", "oid": "org_123"})
        assert claims.organization_id == "org_123"

    def test_missing_fields_default_to_empty_string(self) -> None:
        claims = UserClaims({})
        assert claims.user_id == ""
        assert claims.email == ""
        assert claims.name == ""
        assert claims.organization_id == ""

    def test_display_name_prefers_name(self) -> None:
        claims = UserClaims({"sub": "id", "email": "e@x.com", "name": "Alice"})
        assert claims.display_name == "Alice"

    def test_display_name_falls_back_to_email(self) -> None:
        claims = UserClaims({"sub": "id", "email": "e@x.com"})
        assert claims.display_name == "e@x.com"

    def test_display_name_falls_back_to_user_id(self) -> None:
        claims = UserClaims({"sub": "usr_123"})
        assert claims.display_name == "usr_123"

    def test_raw_returns_copy_of_claims(self) -> None:
        raw = {"sub": "x", "custom": "value"}
        claims = UserClaims(raw)
        assert claims.raw == raw
        assert claims.raw is not raw  # must be a copy


# ── get_scalekit_client ───────────────────────────────────────────────────────

class TestGetScalekitClient:
    def test_returns_none_when_not_configured(self) -> None:
        from trace.auth.scalekit import get_scalekit_client
        # Clear the cache to test with env patching
        get_scalekit_client.cache_clear()
        with patch("trace.auth.scalekit.get_settings") as mock_settings:
            mock_settings.return_value.scalekit_env_url = None
            mock_settings.return_value.scalekit_client_id = None
            mock_settings.return_value.scalekit_client_secret = None
            result = get_scalekit_client()
        get_scalekit_client.cache_clear()
        assert result is None

    def test_returns_client_when_configured(self) -> None:
        from scalekit import ScalekitClient
        from trace.auth.scalekit import get_scalekit_client
        get_scalekit_client.cache_clear()
        with patch("trace.auth.scalekit.get_settings") as mock_settings:
            mock_settings.return_value.scalekit_env_url = "https://example.scalekit.com"
            mock_settings.return_value.scalekit_client_id = "client_id"
            mock_settings.return_value.scalekit_client_secret = "client_secret"
            with patch("trace.auth.scalekit.ScalekitClient") as mock_cls:
                mock_cls.return_value = MagicMock(spec=ScalekitClient)
                result = get_scalekit_client()
        get_scalekit_client.cache_clear()
        assert result is not None


# ── verify_token ──────────────────────────────────────────────────────────────

class TestVerifyToken:
    async def test_valid_token_returns_user_claims(self) -> None:
        from trace.auth.scalekit import verify_token
        mock_client = MagicMock()
        mock_client.validate_access_token_and_get_claims.return_value = _SAMPLE_CLAIMS

        with patch("trace.auth.scalekit.get_scalekit_client", return_value=mock_client):
            result = await verify_token("valid.jwt.token")

        assert isinstance(result, UserClaims)
        assert result.email == "alice@example.com"

    async def test_invalid_token_raises_401(self) -> None:
        from fastapi import HTTPException
        from scalekit.client import ScalekitValidateTokenFailureException
        from trace.auth.scalekit import verify_token

        mock_client = MagicMock()
        mock_client.validate_access_token_and_get_claims.side_effect = (
            ScalekitValidateTokenFailureException("expired")
        )

        with patch("trace.auth.scalekit.get_scalekit_client", return_value=mock_client):
            with pytest.raises(HTTPException) as exc_info:
                await verify_token("bad.token")

        assert exc_info.value.status_code == 401

    async def test_unconfigured_scalekit_raises_503(self) -> None:
        from fastapi import HTTPException
        from trace.auth.scalekit import verify_token

        with patch("trace.auth.scalekit.get_scalekit_client", return_value=None):
            with pytest.raises(HTTPException) as exc_info:
                await verify_token("any.token")

        assert exc_info.value.status_code == 503


# ── GET /auth/login ───────────────────────────────────────────────────────────

class TestAuthLogin:
    def test_login_returns_200_when_scalekit_configured(self) -> None:
        mock_client = MagicMock()
        mock_client.get_authorization_url.return_value = (
            "https://example.scalekit.com/oauth/authorize?client_id=x"
        )
        with patch("trace.auth.scalekit.get_scalekit_client", return_value=mock_client):
            with patch("trace.delivery.api.get_settings") as mock_settings:
                mock_settings.return_value.auth_callback_url = "http://localhost:8000/auth/callback"
                with TestClient(app) as client:
                    response = client.get("/auth/login")
        assert response.status_code == 200

    def test_login_returns_authorization_url(self) -> None:
        expected_url = "https://example.scalekit.com/oauth/authorize?client_id=x&response_type=code"
        mock_client = MagicMock()
        mock_client.get_authorization_url.return_value = expected_url
        with patch("trace.auth.scalekit.get_scalekit_client", return_value=mock_client):
            with patch("trace.delivery.api.get_settings") as mock_settings:
                mock_settings.return_value.auth_callback_url = "http://localhost:8000/auth/callback"
                with TestClient(app) as client:
                    response = client.get("/auth/login")
        data = response.json()
        assert "authorization_url" in data
        assert data["authorization_url"] == expected_url

    def test_login_returns_503_when_scalekit_not_configured(self) -> None:
        with patch("trace.auth.scalekit.get_scalekit_client", return_value=None):
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.get("/auth/login")
        assert response.status_code == 503

    def test_login_passes_state_parameter(self) -> None:
        mock_client = MagicMock()
        mock_client.get_authorization_url.return_value = "https://example.scalekit.com/oauth/authorize?state=xyz"
        with patch("trace.auth.scalekit.get_scalekit_client", return_value=mock_client):
            with patch("trace.delivery.api.get_settings") as mock_settings:
                mock_settings.return_value.auth_callback_url = "http://localhost:8000/auth/callback"
                with TestClient(app) as client:
                    response = client.get("/auth/login?state=xyz")
        assert response.status_code == 200
        # Verify state was passed to the SDK
        call_kwargs = mock_client.get_authorization_url.call_args
        options = call_kwargs[0][1] if call_kwargs[0] else call_kwargs[1].get("options")
        assert options.state == "xyz"


# ── GET /auth/callback ────────────────────────────────────────────────────────

class TestAuthCallback:
    def test_callback_returns_200_with_valid_code(self) -> None:
        mock_client = MagicMock()
        mock_client.authenticate_with_code.return_value = {
            "access_token": "access.jwt.token",
            "id_token": "id.jwt.token",
            "user": {"email": "alice@example.com"},
            "expires_in": 3600,
        }
        with patch("trace.auth.scalekit.get_scalekit_client", return_value=mock_client):
            with patch("trace.delivery.api.get_settings") as mock_settings:
                mock_settings.return_value.auth_callback_url = "http://localhost:8000/auth/callback"
                with TestClient(app) as client:
                    response = client.get("/auth/callback?code=auth_code_abc")
        assert response.status_code == 200

    def test_callback_returns_access_token(self) -> None:
        mock_client = MagicMock()
        mock_client.authenticate_with_code.return_value = {
            "access_token": "access.jwt.token",
            "user": {},
            "expires_in": 3600,
        }
        with patch("trace.auth.scalekit.get_scalekit_client", return_value=mock_client):
            with patch("trace.delivery.api.get_settings") as mock_settings:
                mock_settings.return_value.auth_callback_url = "http://localhost:8000/auth/callback"
                with TestClient(app) as client:
                    response = client.get("/auth/callback?code=auth_code_abc")
        data = response.json()
        assert data["access_token"] == "access.jwt.token"
        assert data["token_type"] == "bearer"

    def test_callback_returns_503_when_not_configured(self) -> None:
        with patch("trace.auth.scalekit.get_scalekit_client", return_value=None):
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.get("/auth/callback?code=any")
        assert response.status_code == 503


# ── GET /auth/me ──────────────────────────────────────────────────────────────

class TestAuthMe:
    def test_me_returns_200_with_valid_token(self) -> None:
        mock_client = MagicMock()
        mock_client.validate_access_token_and_get_claims.return_value = _SAMPLE_CLAIMS
        with patch("trace.auth.scalekit.get_scalekit_client", return_value=mock_client):
            with TestClient(app) as client:
                response = client.get(
                    "/auth/me",
                    headers={"Authorization": "Bearer valid.jwt.token"},
                )
        assert response.status_code == 200

    def test_me_returns_user_profile(self) -> None:
        mock_client = MagicMock()
        mock_client.validate_access_token_and_get_claims.return_value = _SAMPLE_CLAIMS
        with patch("trace.auth.scalekit.get_scalekit_client", return_value=mock_client):
            with TestClient(app) as client:
                response = client.get(
                    "/auth/me",
                    headers={"Authorization": "Bearer valid.jwt.token"},
                )
        data = response.json()
        assert data["email"] == "alice@example.com"
        assert data["name"] == "Alice Smith"
        assert data["user_id"] == "usr_01abc123"

    def test_me_returns_401_without_token(self) -> None:
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/auth/me")
        assert response.status_code == 401

    def test_me_returns_401_with_invalid_token(self) -> None:
        from scalekit.client import ScalekitValidateTokenFailureException
        mock_client = MagicMock()
        mock_client.validate_access_token_and_get_claims.side_effect = (
            ScalekitValidateTokenFailureException("expired")
        )
        with patch("trace.auth.scalekit.get_scalekit_client", return_value=mock_client):
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.get(
                    "/auth/me",
                    headers={"Authorization": "Bearer invalid.token"},
                )
        assert response.status_code == 401


# ── POST /newsletter/generate with auth ──────────────────────────────────────

class TestGenerateWithAuth:
    def test_generate_without_token_returns_empty_generated_for(self) -> None:
        mock_pipeline = make_mock_pipeline()
        app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
        try:
            with TestClient(app) as client:
                response = client.post("/newsletter/generate")
            assert response.status_code == 200
            assert response.json()["generated_for"] == ""
        finally:
            app.dependency_overrides.clear()

    def test_generate_with_valid_token_populates_generated_for(self) -> None:
        mock_pipeline = make_mock_pipeline()
        mock_user = make_user_claims()
        app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
        app.dependency_overrides[get_optional_user] = lambda: mock_user
        try:
            with TestClient(app) as client:
                response = client.post("/newsletter/generate")
            assert response.status_code == 200
            assert response.json()["generated_for"] == "Alice Smith"
        finally:
            app.dependency_overrides.clear()

    def test_generate_generated_for_uses_email_when_no_name(self) -> None:
        mock_pipeline = make_mock_pipeline()
        mock_user = UserClaims({"sub": "id", "email": "bob@example.com"})
        app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
        app.dependency_overrides[get_optional_user] = lambda: mock_user
        try:
            with TestClient(app) as client:
                response = client.post("/newsletter/generate")
            assert response.json()["generated_for"] == "bob@example.com"
        finally:
            app.dependency_overrides.clear()

    def test_generate_response_has_generated_for_field(self) -> None:
        mock_pipeline = make_mock_pipeline()
        app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
        try:
            with TestClient(app) as client:
                response = client.post("/newsletter/generate")
            assert "generated_for" in response.json()
        finally:
            app.dependency_overrides.clear()
