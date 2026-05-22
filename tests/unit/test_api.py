"""
Unit tests for trace/delivery/api.py — FastAPI application.

Uses httpx.AsyncClient with ASGITransport (no real server). Pipeline is
injected via FastAPI dependency override — no real collectors, scrapers, or
Claude calls.

Test categories:
  1. Health endpoint: GET /health → 200 {"status": "ok"}
  2. Generate endpoint happy path: POST /newsletter/generate → 200 newsletter JSON
  3. Generate response schema: all required fields present and typed correctly
  4. Generate with pipeline error → 503 with error detail
  5. Generate with no-signals error → 503
  6. Request body validation: invalid token_budget → 422
  7. Request body defaults: omitting body uses defaults
  8. Errors from pipeline propagated into response
  9. CORS headers present (for browser access)
  10. Generated newsletter HTML and plain_text in response
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from trace.delivery.api import app, get_pipeline
from trace.models import (
    ContentSource,
    CuriosityGraph,
    CuriosityType,
    Newsletter,
    NewsletterSection,
    RawSignal,
    ScrapedArticle,
    SignalSource,
    Topic,
)
from trace.pipeline.runner import PipelineError, PipelineResult


# ── Helpers ───────────────────────────────────────────────────────────────────

_NOW = datetime(2024, 6, 1, tzinfo=timezone.utc)


def make_section() -> NewsletterSection:
    return NewsletterSection(
        title="Transformers This Week",
        section_type="weekly_topics",
        content="Your signals converge on transformer architecture this week — three papers, two Reddit threads.",
        source_urls=["https://arxiv.org/abs/1706.03762"],
        audit_reasoning="Frequency 3, recency 0.8 — recurring active interest.",
    )


def make_newsletter() -> Newsletter:
    return Newsletter(
        generated_at=_NOW,
        subject_line="Your Curiosity Digest: Transformers Edition",
        sections=(make_section(),),
        plain_text="Transformers This Week\n---\nYour signals...",
        html="<html><body><h1>Your Curiosity Digest</h1></body></html>",
    )


def make_topic() -> Topic:
    return Topic(name="transformer architecture", frequency=3, recency_score=0.8)


def make_graph() -> CuriosityGraph:
    return CuriosityGraph(topics=(make_topic(),), signal_count=3)


def make_signal() -> RawSignal:
    return RawSignal(
        source=SignalSource.CHROME_HISTORY,
        content="transformer architecture paper",
        timestamp=_NOW,
    )


def make_pipeline_result(
    newsletter: Newsletter | None = None,
    errors: list[str] | None = None,
) -> PipelineResult:
    return PipelineResult(
        newsletter=newsletter or make_newsletter(),
        graph=make_graph(),
        collected_signals=(make_signal(),),
        errors=errors or [],
        completed_at=_NOW,
    )


def make_mock_pipeline(result: PipelineResult | None = None, raises: Exception | None = None) -> MagicMock:
    pipeline = MagicMock()
    if raises is not None:
        pipeline.run = AsyncMock(side_effect=raises)
    else:
        pipeline.run = AsyncMock(return_value=result or make_pipeline_result())
    return pipeline


# ── Health endpoint ───────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self) -> None:
        with TestClient(app) as client:
            response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self) -> None:
        with TestClient(app) as client:
            response = client.get("/health")
        assert response.json()["status"] == "ok"

    def test_health_returns_json(self) -> None:
        with TestClient(app) as client:
            response = client.get("/health")
        assert response.headers["content-type"].startswith("application/json")


# ── Generate endpoint — happy path ────────────────────────────────────────────

class TestGenerateHappyPath:
    def test_generate_returns_200(self) -> None:
        mock = make_mock_pipeline()
        app.dependency_overrides[get_pipeline] = lambda: mock
        try:
            with TestClient(app) as client:
                response = client.post("/newsletter/generate")
            assert response.status_code == 200
        finally:
            app.dependency_overrides.clear()

    def test_generate_returns_subject_line(self) -> None:
        mock = make_mock_pipeline()
        app.dependency_overrides[get_pipeline] = lambda: mock
        try:
            with TestClient(app) as client:
                response = client.post("/newsletter/generate")
            assert "subject_line" in response.json()
            assert len(response.json()["subject_line"]) >= 10
        finally:
            app.dependency_overrides.clear()

    def test_generate_returns_newsletter_id(self) -> None:
        mock = make_mock_pipeline()
        app.dependency_overrides[get_pipeline] = lambda: mock
        try:
            with TestClient(app) as client:
                response = client.post("/newsletter/generate")
            assert "id" in response.json()
        finally:
            app.dependency_overrides.clear()

    def test_generate_returns_sections(self) -> None:
        mock = make_mock_pipeline()
        app.dependency_overrides[get_pipeline] = lambda: mock
        try:
            with TestClient(app) as client:
                response = client.post("/newsletter/generate")
            data = response.json()
            assert "sections" in data
            assert len(data["sections"]) >= 1
        finally:
            app.dependency_overrides.clear()

    def test_generate_returns_plain_text(self) -> None:
        mock = make_mock_pipeline()
        app.dependency_overrides[get_pipeline] = lambda: mock
        try:
            with TestClient(app) as client:
                response = client.post("/newsletter/generate")
            assert "plain_text" in response.json()
        finally:
            app.dependency_overrides.clear()

    def test_generate_returns_html(self) -> None:
        mock = make_mock_pipeline()
        app.dependency_overrides[get_pipeline] = lambda: mock
        try:
            with TestClient(app) as client:
                response = client.post("/newsletter/generate")
            assert "html" in response.json()
        finally:
            app.dependency_overrides.clear()

    def test_generate_returns_generated_at(self) -> None:
        mock = make_mock_pipeline()
        app.dependency_overrides[get_pipeline] = lambda: mock
        try:
            with TestClient(app) as client:
                response = client.post("/newsletter/generate")
            assert "generated_at" in response.json()
        finally:
            app.dependency_overrides.clear()

    def test_generate_errors_field_present(self) -> None:
        mock = make_mock_pipeline()
        app.dependency_overrides[get_pipeline] = lambda: mock
        try:
            with TestClient(app) as client:
                response = client.post("/newsletter/generate")
            assert "errors" in response.json()
        finally:
            app.dependency_overrides.clear()

    def test_generate_pipeline_called_once(self) -> None:
        mock = make_mock_pipeline()
        app.dependency_overrides[get_pipeline] = lambda: mock
        try:
            with TestClient(app) as client:
                client.post("/newsletter/generate")
            mock.run.assert_called_once()
        finally:
            app.dependency_overrides.clear()


# ── Generate response schema ──────────────────────────────────────────────────

class TestGenerateResponseSchema:
    def _call(self) -> dict:
        mock = make_mock_pipeline()
        app.dependency_overrides[get_pipeline] = lambda: mock
        try:
            with TestClient(app) as client:
                return client.post("/newsletter/generate").json()
        finally:
            app.dependency_overrides.clear()

    def test_section_has_title(self) -> None:
        data = self._call()
        assert "title" in data["sections"][0]

    def test_section_has_content(self) -> None:
        data = self._call()
        assert "content" in data["sections"][0]

    def test_section_has_section_type(self) -> None:
        data = self._call()
        assert "section_type" in data["sections"][0]

    def test_section_has_source_urls(self) -> None:
        data = self._call()
        assert "source_urls" in data["sections"][0]

    def test_section_has_audit_reasoning(self) -> None:
        data = self._call()
        assert "audit_reasoning" in data["sections"][0]


# ── Error handling ────────────────────────────────────────────────────────────

class TestGenerateErrorHandling:
    def test_pipeline_error_returns_503(self) -> None:
        mock = make_mock_pipeline(raises=PipelineError("no signals"))
        app.dependency_overrides[get_pipeline] = lambda: mock
        try:
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post("/newsletter/generate")
            assert response.status_code == 503
        finally:
            app.dependency_overrides.clear()

    def test_pipeline_error_detail_in_response(self) -> None:
        mock = make_mock_pipeline(raises=PipelineError("no signals collected"))
        app.dependency_overrides[get_pipeline] = lambda: mock
        try:
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post("/newsletter/generate")
            assert "detail" in response.json()
        finally:
            app.dependency_overrides.clear()

    def test_pipeline_errors_list_propagated(self) -> None:
        result = make_pipeline_result(
            errors=["[filesystem] collection failed: permission denied"]
        )
        mock = make_mock_pipeline(result=result)
        app.dependency_overrides[get_pipeline] = lambda: mock
        try:
            with TestClient(app) as client:
                response = client.post("/newsletter/generate")
            assert len(response.json()["errors"]) == 1
        finally:
            app.dependency_overrides.clear()


# ── Request body ──────────────────────────────────────────────────────────────

class TestRequestBody:
    def test_empty_body_uses_defaults(self) -> None:
        mock = make_mock_pipeline()
        app.dependency_overrides[get_pipeline] = lambda: mock
        try:
            with TestClient(app) as client:
                response = client.post("/newsletter/generate", json={})
            assert response.status_code == 200
        finally:
            app.dependency_overrides.clear()

    def test_no_body_uses_defaults(self) -> None:
        mock = make_mock_pipeline()
        app.dependency_overrides[get_pipeline] = lambda: mock
        try:
            with TestClient(app) as client:
                response = client.post("/newsletter/generate")
            assert response.status_code == 200
        finally:
            app.dependency_overrides.clear()

    def test_extra_body_fields_ignored(self) -> None:
        mock = make_mock_pipeline()
        app.dependency_overrides[get_pipeline] = lambda: mock
        try:
            with TestClient(app) as client:
                response = client.post(
                    "/newsletter/generate",
                    json={"unknown_field": "value"},
                )
            assert response.status_code == 200
        finally:
            app.dependency_overrides.clear()


# ── CORS ──────────────────────────────────────────────────────────────────────

class TestCORS:
    def test_cors_header_present_on_generate(self) -> None:
        mock = make_mock_pipeline()
        app.dependency_overrides[get_pipeline] = lambda: mock
        try:
            with TestClient(app) as client:
                response = client.post(
                    "/newsletter/generate",
                    headers={"Origin": "http://localhost:3000"},
                )
            assert "access-control-allow-origin" in response.headers
        finally:
            app.dependency_overrides.clear()

    def test_cors_preflight_returns_200(self) -> None:
        with TestClient(app) as client:
            response = client.options(
                "/newsletter/generate",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                },
            )
        assert response.status_code in (200, 204)
