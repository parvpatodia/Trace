"""
Unit tests for trace/composer/newsletter.py — NewsletterComposer.

All Anthropic API calls are intercepted with MagicMock — no real network.

Test categories:
  1. Constructor: missing client raises; model/token params accepted
  2. Empty context: no selected topics → raises or returns minimal Newsletter
  3. Happy path: valid AssemblyContext → Newsletter with correct fields
  4. Subject line: present, length 10–100 chars
  5. Sections: at least one section returned; section types from allowed set
  6. Source URLs: populated from articles in context
  7. Audit reasoning: non-trivial (len >= 10)
  8. Claude API usage: system prompt sent, user message contains topic names
  9. Prompt caching: system prompt has cache_control ephemeral
  10. TopicExtractionError propagates as NewsletterGenerationError on API failure
  11. Malformed Claude response → NewsletterGenerationError
  12. Newsletter immutability: frozen Pydantic model
  13. Plain text and HTML: non-empty when sections present
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from trace.composer.assembler import AssemblyContext
from trace.composer.newsletter import NewsletterComposer, NewsletterGenerationError
from trace.models import (
    ContentSource,
    CuriosityGraph,
    CuriosityType,
    Newsletter,
    ScrapedArticle,
    Topic,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

_NOW = datetime(2024, 6, 1, tzinfo=timezone.utc)


def make_topic(
    name: str = "transformer architecture",
    frequency: int = 3,
    recency_score: float = 0.8,
    debt_score: float = 0.0,
    curiosity_type: CuriosityType = CuriosityType.SHALLOW,
    topic_id: str | None = None,
) -> Topic:
    t = Topic(
        name=name,
        frequency=frequency,
        recency_score=recency_score,
        debt_score=debt_score,
        curiosity_type=curiosity_type,
    )
    if topic_id:
        t = t.model_copy(update={"id": topic_id})
    return t


def make_article(
    topic: Topic,
    title: str = "Attention Is All You Need",
    url: str = "https://arxiv.org/abs/1706.03762",
    summary: str = "A paper about transformer models and attention mechanisms.",
    relevance_score: float = 0.9,
) -> ScrapedArticle:
    return ScrapedArticle(
        topic_id=topic.id,
        topic_name=topic.name,
        source=ContentSource.ARXIV,
        title=title,
        url=url,
        summary=summary,
        relevance_score=relevance_score,
    )


def make_assembly_context(
    topics: list[Topic] | None = None,
    articles: list[ScrapedArticle] | None = None,
    debt_topics: list[Topic] | None = None,
) -> AssemblyContext:
    if topics is None:
        t = make_topic()
        topics = [t]
        articles = [make_article(t)]
    if articles is None:
        articles = []
    articles_by_id: dict[str, list[ScrapedArticle]] = {
        t.id: [a for a in articles if a.topic_id == t.id]
        for t in topics
    }
    return AssemblyContext(
        selected_topics=tuple(topics),
        articles_by_topic_id=articles_by_id,
        debt_topics=tuple(debt_topics or []),
        token_estimate=500,
        assembled_at=_NOW,
    )


def _make_claude_response(content: str) -> MagicMock:
    """Build a minimal Anthropic Message-shaped mock."""
    msg = MagicMock()
    msg.content = [MagicMock(text=content)]
    return msg


_VALID_JSON_RESPONSE = """{
  "subject_line": "Your Weekly AI Curiosity Digest",
  "sections": [
    {
      "title": "Transformers: What You're Really Chasing",
      "section_type": "weekly_topics",
      "content": "This week your signals converge on transformer architecture — the attention mechanism that sparked a revolution in sequence modelling. Three papers, two Reddit threads, and one saved HN post suggest you're moving past the basics toward implementation details.",
      "source_urls": ["https://arxiv.org/abs/1706.03762"],
      "audit_reasoning": "Signal frequency of 3 over the past week, with recency score 0.8, indicates a genuine recurring interest rather than a one-off search."
    }
  ]
}"""


def make_mock_client(response_text: str = _VALID_JSON_RESPONSE) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = _make_claude_response(response_text)
    return client


# ── Constructor ───────────────────────────────────────────────────────────────

class TestConstructor:
    def test_none_client_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="client"):
            NewsletterComposer(client=None)  # type: ignore[arg-type]

    def test_valid_client_accepted(self) -> None:
        composer = NewsletterComposer(client=make_mock_client())
        assert composer is not None

    def test_custom_model_accepted(self) -> None:
        composer = NewsletterComposer(
            client=make_mock_client(),
            model="claude-opus-4-7",
        )
        assert composer is not None

    def test_custom_max_tokens_accepted(self) -> None:
        composer = NewsletterComposer(
            client=make_mock_client(),
            max_tokens=2048,
        )
        assert composer is not None


# ── Empty context ─────────────────────────────────────────────────────────────

class TestEmptyContext:
    async def test_empty_context_raises_generation_error(self) -> None:
        ctx = AssemblyContext(
            selected_topics=(),
            articles_by_topic_id={},
            debt_topics=(),
            token_estimate=0,
            assembled_at=_NOW,
        )
        composer = NewsletterComposer(client=make_mock_client())
        with pytest.raises(NewsletterGenerationError):
            await composer.compose(ctx)


# ── Happy path ────────────────────────────────────────────────────────────────

class TestHappyPath:
    async def test_returns_newsletter_instance(self) -> None:
        ctx = make_assembly_context()
        composer = NewsletterComposer(client=make_mock_client())
        result = await composer.compose(ctx)
        assert isinstance(result, Newsletter)

    async def test_newsletter_has_subject_line(self) -> None:
        ctx = make_assembly_context()
        result = await NewsletterComposer(client=make_mock_client()).compose(ctx)
        assert len(result.subject_line) >= 10

    async def test_newsletter_subject_line_max_100_chars(self) -> None:
        ctx = make_assembly_context()
        result = await NewsletterComposer(client=make_mock_client()).compose(ctx)
        assert len(result.subject_line) <= 100

    async def test_newsletter_has_at_least_one_section(self) -> None:
        ctx = make_assembly_context()
        result = await NewsletterComposer(client=make_mock_client()).compose(ctx)
        assert len(result.sections) >= 1

    async def test_newsletter_generated_at_is_utc(self) -> None:
        ctx = make_assembly_context()
        result = await NewsletterComposer(client=make_mock_client()).compose(ctx)
        assert result.generated_at.tzinfo is not None

    async def test_newsletter_id_is_set(self) -> None:
        ctx = make_assembly_context()
        result = await NewsletterComposer(client=make_mock_client()).compose(ctx)
        assert result.id != ""


# ── Section structure ─────────────────────────────────────────────────────────

class TestSectionStructure:
    async def test_section_has_valid_type(self) -> None:
        ctx = make_assembly_context()
        result = await NewsletterComposer(client=make_mock_client()).compose(ctx)
        valid_types = {"weekly_topics", "curiosity_debt", "rabbit_hole"}
        for section in result.sections:
            assert section.section_type in valid_types

    async def test_section_content_at_least_50_chars(self) -> None:
        ctx = make_assembly_context()
        result = await NewsletterComposer(client=make_mock_client()).compose(ctx)
        for section in result.sections:
            assert len(section.content) >= 50

    async def test_section_audit_reasoning_at_least_10_chars(self) -> None:
        ctx = make_assembly_context()
        result = await NewsletterComposer(client=make_mock_client()).compose(ctx)
        for section in result.sections:
            assert len(section.audit_reasoning) >= 10

    async def test_section_title_present(self) -> None:
        ctx = make_assembly_context()
        result = await NewsletterComposer(client=make_mock_client()).compose(ctx)
        for section in result.sections:
            assert len(section.title) >= 1

    async def test_section_source_urls_populated(self) -> None:
        t = make_topic()
        a = make_article(t)
        ctx = make_assembly_context(topics=[t], articles=[a])
        result = await NewsletterComposer(client=make_mock_client()).compose(ctx)
        all_urls = [url for s in result.sections for url in s.source_urls]
        assert len(all_urls) >= 1


# ── Plain text and HTML ───────────────────────────────────────────────────────

class TestRendering:
    async def test_plain_text_non_empty_when_sections_present(self) -> None:
        ctx = make_assembly_context()
        result = await NewsletterComposer(client=make_mock_client()).compose(ctx)
        assert len(result.plain_text) > 0

    async def test_html_non_empty_when_sections_present(self) -> None:
        ctx = make_assembly_context()
        result = await NewsletterComposer(client=make_mock_client()).compose(ctx)
        assert len(result.html) > 0

    async def test_html_contains_subject_line(self) -> None:
        ctx = make_assembly_context()
        result = await NewsletterComposer(client=make_mock_client()).compose(ctx)
        assert result.subject_line in result.html or result.subject_line in result.plain_text


# ── Claude API usage ──────────────────────────────────────────────────────────

class TestClaudeAPIUsage:
    async def test_messages_create_called_once(self) -> None:
        client = make_mock_client()
        ctx = make_assembly_context()
        await NewsletterComposer(client=client).compose(ctx)
        client.messages.create.assert_called_once()

    async def test_system_prompt_sent(self) -> None:
        client = make_mock_client()
        ctx = make_assembly_context()
        await NewsletterComposer(client=client).compose(ctx)
        call_kwargs = client.messages.create.call_args
        # system can be positional or keyword
        kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        assert "system" in kwargs or len(call_kwargs.args) > 1

    async def test_user_message_contains_topic_name(self) -> None:
        client = make_mock_client()
        t = make_topic(name="diffusion models")
        ctx = make_assembly_context(topics=[t])
        await NewsletterComposer(client=client).compose(ctx)
        call_kwargs = client.messages.create.call_args
        messages = call_kwargs.kwargs.get("messages") or []
        user_text = " ".join(
            m["content"] for m in messages if m.get("role") == "user"
        )
        assert "diffusion models" in user_text

    async def test_model_parameter_forwarded(self) -> None:
        client = make_mock_client()
        ctx = make_assembly_context()
        await NewsletterComposer(client=client, model="claude-haiku-4-5-20251001").compose(ctx)
        call_kwargs = client.messages.create.call_args
        assert call_kwargs.kwargs.get("model") == "claude-haiku-4-5-20251001"


# ── Prompt caching ────────────────────────────────────────────────────────────

class TestPromptCaching:
    async def test_system_prompt_has_cache_control(self) -> None:
        client = make_mock_client()
        ctx = make_assembly_context()
        await NewsletterComposer(client=client).compose(ctx)
        call_kwargs = client.messages.create.call_args
        system = call_kwargs.kwargs.get("system")
        assert system is not None
        # system must be a list of blocks with cache_control
        assert isinstance(system, list)
        cache_controls = [
            b.get("cache_control") for b in system if isinstance(b, dict)
        ]
        assert any(cc is not None for cc in cache_controls)

    async def test_cache_control_type_is_ephemeral(self) -> None:
        client = make_mock_client()
        ctx = make_assembly_context()
        await NewsletterComposer(client=client).compose(ctx)
        call_kwargs = client.messages.create.call_args
        system = call_kwargs.kwargs.get("system", [])
        for block in system:
            if isinstance(block, dict) and block.get("cache_control"):
                assert block["cache_control"]["type"] == "ephemeral"


# ── Error handling ────────────────────────────────────────────────────────────

class TestErrorHandling:
    async def test_api_exception_raises_generation_error(self) -> None:
        import anthropic
        client = MagicMock()
        client.messages.create.side_effect = anthropic.APIError(
            message="rate limited", request=MagicMock(), body=None
        )
        ctx = make_assembly_context()
        with pytest.raises(NewsletterGenerationError, match="API"):
            await NewsletterComposer(client=client).compose(ctx)

    async def test_malformed_json_raises_generation_error(self) -> None:
        client = make_mock_client(response_text="not valid json at all {{")
        ctx = make_assembly_context()
        with pytest.raises(NewsletterGenerationError):
            await NewsletterComposer(client=client).compose(ctx)

    async def test_missing_subject_line_raises_generation_error(self) -> None:
        bad_json = """{
          "sections": [
            {
              "title": "Section without subject",
              "section_type": "weekly_topics",
              "content": "This is valid content that is long enough to pass the 50 char minimum here.",
              "source_urls": [],
              "audit_reasoning": "Valid reasoning that is long enough."
            }
          ]
        }"""
        client = make_mock_client(response_text=bad_json)
        ctx = make_assembly_context()
        with pytest.raises(NewsletterGenerationError):
            await NewsletterComposer(client=client).compose(ctx)

    async def test_empty_sections_raises_generation_error(self) -> None:
        bad_json = '{"subject_line": "Valid Subject Line Here", "sections": []}'
        client = make_mock_client(response_text=bad_json)
        ctx = make_assembly_context()
        with pytest.raises(NewsletterGenerationError):
            await NewsletterComposer(client=client).compose(ctx)


# ── Immutability ──────────────────────────────────────────────────────────────

class TestImmutability:
    async def test_newsletter_is_frozen(self) -> None:
        ctx = make_assembly_context()
        result = await NewsletterComposer(client=make_mock_client()).compose(ctx)
        with pytest.raises(Exception):
            result.subject_line = "mutated"  # type: ignore[misc]
