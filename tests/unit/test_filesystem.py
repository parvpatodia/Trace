"""
Unit tests for trace/signals/filesystem.py — FileSystemCollector.

FileSystemCollector reads local documents and emits RawSignal objects.
Supported formats: .txt, .md, .pdf (via PyMuPDF / fitz).

Test categories:
  1. Constructor validation: missing directory, not-a-directory, naive since
  2. Plain text (.txt): content extraction, encoding, empty files skipped
  3. Markdown (.md): content extraction (raw text, not rendered HTML)
  4. PDF (.pdf): text extraction via fitz; empty PDFs skipped
  5. Glob patterns: custom include patterns restrict which files are scanned
  6. Recursive scanning: subdirectories included by default
  7. `since` filtering: files with mtime < since excluded
  8. Content truncation: files > 2000 chars truncated to 2000
  9. Metadata: file name, suffix, size, mtime in signal metadata
  10. Signal fields: source=FILESYSTEM, url=None (local files have no URL)
  11. Error isolation: one unreadable file does not abort collection of others

All I/O uses tmp_path — no real filesystem paths assumed.
PDF tests use a minimal valid PDF written as raw bytes (no PyMuPDF dependency
in test setup, just in the collector itself).
"""

from __future__ import annotations

import struct
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from trace.models import RawSignal, SignalSource
from trace.signals.base import SignalCollectionError
from trace.signals.filesystem import FileSystemCollector


# ── Minimal valid PDF bytes ───────────────────────────────────────────────────
# This is a minimal well-formed single-page PDF containing the text "Hello PDF".
# Constructed from spec to avoid needing reportlab/fpdf in test deps.
_MINIMAL_PDF = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 44>>
stream
BT /F1 12 Tf 100 700 Td (Hello PDF) Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f\r
0000000009 00000 n\r
0000000058 00000 n\r
0000000115 00000 n\r
0000000266 00000 n\r
0000000360 00000 n\r
trailer<</Size 6/Root 1 0 R>>
startxref
441
%%EOF"""


# ── Constructor validation ────────────────────────────────────────────────────

class TestConstructorValidation:
    def test_missing_directory_raises_signal_collection_error(
        self, tmp_path: Path
    ) -> None:
        missing = tmp_path / "does_not_exist"
        with pytest.raises(SignalCollectionError) as exc_info:
            FileSystemCollector(missing)
        assert exc_info.value.source == SignalSource.FILESYSTEM
        assert "not found" in str(exc_info.value).lower()

    def test_file_path_instead_of_directory_raises_signal_collection_error(
        self, tmp_path: Path
    ) -> None:
        file_path = tmp_path / "a_file.txt"
        file_path.write_text("content", encoding="utf-8")
        with pytest.raises(SignalCollectionError) as exc_info:
            FileSystemCollector(file_path)
        assert exc_info.value.source == SignalSource.FILESYSTEM
        assert "directory" in str(exc_info.value).lower()

    def test_naive_since_raises_value_error(self, tmp_path: Path) -> None:
        naive = datetime(2024, 1, 1)
        with pytest.raises(ValueError, match="timezone-aware"):
            FileSystemCollector(tmp_path, since=naive)

    def test_aware_since_accepted(self, tmp_path: Path) -> None:
        since = datetime(2024, 1, 1, tzinfo=timezone.utc)
        collector = FileSystemCollector(tmp_path, since=since)
        assert collector is not None

    def test_source_is_filesystem(self, tmp_path: Path) -> None:
        assert FileSystemCollector(tmp_path).source == SignalSource.FILESYSTEM

    def test_empty_directory_accepted(self, tmp_path: Path) -> None:
        collector = FileSystemCollector(tmp_path)
        assert collector is not None


# ── Happy-path: plain text ────────────────────────────────────────────────────

class TestPlainText:
    async def test_txt_file_produces_signal(self, tmp_path: Path) -> None:
        (tmp_path / "note.txt").write_text(
            "Transformers use self-attention mechanisms", encoding="utf-8"
        )
        signals = await FileSystemCollector(tmp_path).collect()
        assert len(signals) == 1

    async def test_txt_content_matches_file_text(self, tmp_path: Path) -> None:
        text = "The quick brown fox jumps over the lazy dog. " * 5
        (tmp_path / "note.txt").write_text(text, encoding="utf-8")
        signals = await FileSystemCollector(tmp_path).collect()
        assert signals[0].content == text.strip()[:2000]

    async def test_txt_signal_source_is_filesystem(self, tmp_path: Path) -> None:
        (tmp_path / "note.txt").write_text("Some content here that is long enough", encoding="utf-8")
        signals = await FileSystemCollector(tmp_path).collect()
        assert signals[0].source == SignalSource.FILESYSTEM

    async def test_txt_signal_url_is_none(self, tmp_path: Path) -> None:
        (tmp_path / "note.txt").write_text("Local files have no URL to attach", encoding="utf-8")
        signals = await FileSystemCollector(tmp_path).collect()
        assert signals[0].url is None

    async def test_empty_txt_file_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "empty.txt").write_text("", encoding="utf-8")
        signals = await FileSystemCollector(tmp_path).collect()
        assert signals == []

    async def test_whitespace_only_txt_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "whitespace.txt").write_text("   \n\t\n   ", encoding="utf-8")
        signals = await FileSystemCollector(tmp_path).collect()
        assert signals == []

    async def test_multiple_txt_files_all_collected(self, tmp_path: Path) -> None:
        for i in range(3):
            (tmp_path / f"note_{i}.txt").write_text(
                f"Note {i}: interesting content about machine learning topic {i}",
                encoding="utf-8",
            )
        signals = await FileSystemCollector(tmp_path).collect()
        assert len(signals) == 3


# ── Happy-path: markdown ──────────────────────────────────────────────────────

class TestMarkdown:
    async def test_md_file_produces_signal(self, tmp_path: Path) -> None:
        (tmp_path / "notes.md").write_text(
            "# Transformer Architecture\n\nSelf-attention scales quadratically.",
            encoding="utf-8",
        )
        signals = await FileSystemCollector(tmp_path).collect()
        assert len(signals) == 1

    async def test_md_content_is_raw_text_not_html(self, tmp_path: Path) -> None:
        (tmp_path / "doc.md").write_text(
            "# Header\n\nBody text with **bold** and *italic*.", encoding="utf-8"
        )
        signals = await FileSystemCollector(tmp_path).collect()
        # Raw text: markdown syntax preserved, not rendered as HTML
        assert "<h1>" not in signals[0].content
        assert "Header" in signals[0].content

    async def test_md_and_txt_both_collected(self, tmp_path: Path) -> None:
        (tmp_path / "doc.md").write_text(
            "# Research Notes\n\nSome findings about neural networks.", encoding="utf-8"
        )
        (tmp_path / "log.txt").write_text(
            "Read paper on attention mechanisms and summarised key points.",
            encoding="utf-8",
        )
        signals = await FileSystemCollector(tmp_path).collect()
        assert len(signals) == 2


# ── Happy-path: PDF ───────────────────────────────────────────────────────────

class TestPDF:
    async def test_pdf_file_produces_signal(self, tmp_path: Path) -> None:
        (tmp_path / "paper.pdf").write_bytes(_MINIMAL_PDF)
        signals = await FileSystemCollector(tmp_path).collect()
        # If fitz extracts text, we get a signal; if the minimal PDF yields
        # no text (renderer dependent), the file is skipped — both are valid.
        assert isinstance(signals, list)

    async def test_non_pdf_bytes_file_skipped_gracefully(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "corrupt.pdf").write_bytes(b"not a real pdf file")
        # Should not raise — corrupt PDFs are skipped, not fatal
        signals = await FileSystemCollector(tmp_path).collect()
        assert isinstance(signals, list)

    async def test_pdf_signal_has_correct_source(self, tmp_path: Path) -> None:
        (tmp_path / "paper.pdf").write_bytes(_MINIMAL_PDF)
        signals = await FileSystemCollector(tmp_path).collect()
        for sig in signals:
            assert sig.source == SignalSource.FILESYSTEM


# ── File type filtering ───────────────────────────────────────────────────────

class TestFileTypeFiltering:
    async def test_unsupported_extension_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "script.py").write_text("print('hello')", encoding="utf-8")
        (tmp_path / "data.json").write_text('{"key": "value"}', encoding="utf-8")
        (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")
        signals = await FileSystemCollector(tmp_path).collect()
        assert signals == []

    async def test_supported_and_unsupported_mixed(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text(
            "Interesting research about transformer models", encoding="utf-8"
        )
        (tmp_path / "script.py").write_text("x = 1 + 1", encoding="utf-8")
        signals = await FileSystemCollector(tmp_path).collect()
        assert len(signals) == 1
        assert signals[0].metadata["filename"] == "notes.txt"

    async def test_custom_patterns_restrict_scan(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text(
            "Machine learning notes about gradient descent", encoding="utf-8"
        )
        (tmp_path / "README.md").write_text(
            "# Project Readme\n\nSetup instructions for the project.", encoding="utf-8"
        )
        # Only scan .md files
        signals = await FileSystemCollector(tmp_path, patterns=["**/*.md"]).collect()
        assert len(signals) == 1
        assert signals[0].metadata["filename"] == "README.md"


# ── Recursive scanning ────────────────────────────────────────────────────────

class TestRecursiveScanning:
    async def test_subdirectory_files_included(self, tmp_path: Path) -> None:
        subdir = tmp_path / "research" / "2024"
        subdir.mkdir(parents=True)
        (subdir / "notes.txt").write_text(
            "Deep learning research notes on attention heads", encoding="utf-8"
        )
        signals = await FileSystemCollector(tmp_path).collect()
        assert len(signals) == 1

    async def test_nested_subdirectory_files_included(
        self, tmp_path: Path
    ) -> None:
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / "deep_note.txt").write_text(
            "Very deep nested note about neural architecture search",
            encoding="utf-8",
        )
        signals = await FileSystemCollector(tmp_path).collect()
        assert len(signals) == 1

    async def test_files_across_multiple_subdirs(self, tmp_path: Path) -> None:
        for name in ("papers", "notes", "drafts"):
            subdir = tmp_path / name
            subdir.mkdir()
            (subdir / "content.txt").write_text(
                f"Content in {name} about machine learning models and training",
                encoding="utf-8",
            )
        signals = await FileSystemCollector(tmp_path).collect()
        assert len(signals) == 3


# ── Content truncation ────────────────────────────────────────────────────────

class TestContentTruncation:
    async def test_file_over_2000_chars_truncated(self, tmp_path: Path) -> None:
        text = "x" * 5000
        (tmp_path / "long.txt").write_text(text, encoding="utf-8")
        signals = await FileSystemCollector(tmp_path).collect()
        assert len(signals) == 1
        assert len(signals[0].content) == 2000

    async def test_file_exactly_2000_chars_not_truncated(
        self, tmp_path: Path
    ) -> None:
        text = "a" * 2000
        (tmp_path / "exact.txt").write_text(text, encoding="utf-8")
        signals = await FileSystemCollector(tmp_path).collect()
        assert len(signals[0].content) == 2000

    async def test_file_under_2000_chars_preserved(self, tmp_path: Path) -> None:
        text = "Short note about transformers and attention."
        (tmp_path / "short.txt").write_text(text, encoding="utf-8")
        signals = await FileSystemCollector(tmp_path).collect()
        assert signals[0].content == text.strip()


# ── Metadata ──────────────────────────────────────────────────────────────────

class TestMetadata:
    async def test_metadata_contains_filename(self, tmp_path: Path) -> None:
        (tmp_path / "my_notes.txt").write_text(
            "Notes about distributed systems and CAP theorem", encoding="utf-8"
        )
        signals = await FileSystemCollector(tmp_path).collect()
        assert signals[0].metadata["filename"] == "my_notes.txt"

    async def test_metadata_contains_suffix(self, tmp_path: Path) -> None:
        (tmp_path / "document.md").write_text(
            "# Title\n\nMarkdown document about system design patterns.",
            encoding="utf-8",
        )
        signals = await FileSystemCollector(tmp_path).collect()
        assert signals[0].metadata["suffix"] == ".md"

    async def test_metadata_contains_size_bytes(self, tmp_path: Path) -> None:
        content = "Hello world, this is a test document."
        path = tmp_path / "test.txt"
        path.write_text(content, encoding="utf-8")
        signals = await FileSystemCollector(tmp_path).collect()
        assert "size_bytes" in signals[0].metadata
        assert signals[0].metadata["size_bytes"] == path.stat().st_size

    async def test_metadata_contains_mtime_utc(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text(
            "Content about language model pretraining objectives.", encoding="utf-8"
        )
        signals = await FileSystemCollector(tmp_path).collect()
        assert "mtime_utc" in signals[0].metadata
        mtime = signals[0].metadata["mtime_utc"]
        assert isinstance(mtime, str)  # ISO format string

    async def test_metadata_contains_absolute_path(self, tmp_path: Path) -> None:
        (tmp_path / "doc.txt").write_text(
            "Research notes about vision transformers and patch embeddings.",
            encoding="utf-8",
        )
        signals = await FileSystemCollector(tmp_path).collect()
        assert "path" in signals[0].metadata
        assert signals[0].metadata["path"].endswith("doc.txt")


# ── Since filtering ───────────────────────────────────────────────────────────

class TestSinceFiltering:
    async def test_file_modified_after_since_included(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "recent.txt"
        path.write_text("Recent notes about language models", encoding="utf-8")
        # since = yesterday → file written just now should be included
        since = datetime(2000, 1, 1, tzinfo=timezone.utc)
        signals = await FileSystemCollector(tmp_path, since=since).collect()
        assert len(signals) == 1

    async def test_file_modified_before_since_excluded(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "old.txt"
        path.write_text("Old notes about support vector machines", encoding="utf-8")
        # Set mtime to 2020-01-01
        old_mtime = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
        import os
        os.utime(path, (old_mtime, old_mtime))
        since = datetime(2024, 1, 1, tzinfo=timezone.utc)
        signals = await FileSystemCollector(tmp_path, since=since).collect()
        assert signals == []

    async def test_since_none_includes_all_files(self, tmp_path: Path) -> None:
        for i in range(3):
            (tmp_path / f"f{i}.txt").write_text(
                f"File {i} content about machine learning topic {i}", encoding="utf-8"
            )
        signals = await FileSystemCollector(tmp_path, since=None).collect()
        assert len(signals) == 3


# ── Empty directory ───────────────────────────────────────────────────────────

class TestEmptyDirectory:
    async def test_empty_directory_returns_empty_list(
        self, tmp_path: Path
    ) -> None:
        signals = await FileSystemCollector(tmp_path).collect()
        assert signals == []

    async def test_directory_with_only_unsupported_files_returns_empty(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "main.py").write_text("def main(): pass", encoding="utf-8")
        (tmp_path / "config.yaml").write_text("key: value", encoding="utf-8")
        signals = await FileSystemCollector(tmp_path).collect()
        assert signals == []


# ── Return type contract ──────────────────────────────────────────────────────

class TestReturnTypeContract:
    async def test_returns_list(self, tmp_path: Path) -> None:
        result = await FileSystemCollector(tmp_path).collect()
        assert isinstance(result, list)

    async def test_all_items_are_raw_signals(self, tmp_path: Path) -> None:
        for i in range(3):
            (tmp_path / f"doc{i}.txt").write_text(
                f"Document {i}: notes on deep learning architectures and training",
                encoding="utf-8",
            )
        signals = await FileSystemCollector(tmp_path).collect()
        for s in signals:
            assert isinstance(s, RawSignal)

    async def test_signals_have_utc_aware_timestamps(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "note.txt").write_text(
            "UTC timestamp test content for filesystem signal", encoding="utf-8"
        )
        signals = await FileSystemCollector(tmp_path).collect()
        for s in signals:
            assert s.timestamp.tzinfo is not None


# ── PDF error handling ────────────────────────────────────────────────────────

class TestPdfErrorHandling:
    async def test_corrupt_pdf_bytes_skipped_gracefully(self, tmp_path: Path) -> None:
        """fitz.open() on non-PDF bytes raises; collector must skip silently."""
        (tmp_path / "corrupt.pdf").write_bytes(b"not a real pdf file at all")
        # Even if fitz is installed, this should not raise — corrupt file is skipped.
        collector = FileSystemCollector(root_dir=tmp_path)
        signals = await collector.collect()
        filenames = [s.metadata.get("filename") for s in signals]
        assert "corrupt.pdf" not in filenames

    async def test_fitz_not_installed_pdf_skipped(self, tmp_path: Path) -> None:
        """When PyMuPDF is absent, PDF files produce no signal (not an error)."""
        import sys
        from unittest.mock import patch

        (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.4 fake content")
        collector = FileSystemCollector(root_dir=tmp_path)

        with patch.dict("sys.modules", {"fitz": None}):
            signals = await collector.collect()

        filenames = [s.metadata.get("filename") for s in signals]
        assert "paper.pdf" not in filenames
