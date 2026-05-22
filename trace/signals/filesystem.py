"""
FileSystemCollector — reads local documents and emits RawSignal objects.

Supported formats:
  .txt  — read as UTF-8 text
  .md   — read as UTF-8 text (raw markdown, not rendered)
  .pdf  — text extracted via PyMuPDF (fitz); page text joined with newlines

Design decisions:

WHY THESE THREE FORMATS:
  These are the three document types a knowledge worker accumulates on their
  filesystem: plain notes (.txt), structured notes / docs (.md), and research
  papers / reports (.pdf). Source code (.py, .js) is intentionally excluded —
  it's not a curiosity signal, it's a work artifact.

WHY RAW MARKDOWN (not rendered):
  The topic extractor (Claude) handles semantic content, not HTML structure.
  Sending raw markdown preserves headings, bullet points and emphasis cues
  without introducing parser complexity or HTML noise.

WHY PyMuPDF (fitz) OVER pdfplumber / PyPDF2:
  PyMuPDF is the fastest PDF text extractor in Python, with the most reliable
  Unicode handling. It's already in requirements (pymupdf>=1.24).

WHY MTIME FOR `since` FILTERING (not ctime):
  mtime (modification time) reflects when the content last changed.
  ctime on Linux is metadata-change time, not creation time. On macOS,
  ctime is creation time, but we want the content-change semantic consistently.

WHY glob("**/*") + suffix filter (not glob per extension):
  A single rglob walk with a suffix check is a single directory traversal.
  Running three separate globs (*.txt, *.md, *.pdf) would traverse three times.

WHY skip empty / whitespace-only content:
  RawSignal.content has min_length=1. An empty file has no curiosity signal.
  Whitespace-only files (blank .md files, .txt stubs) are equally uninformative.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trace.models import RawSignal, SignalSource
from trace.signals.base import SignalCollectionError, SignalCollector

_SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".txt", ".md", ".pdf"})
_DEFAULT_PATTERNS: list[str] = ["**/*"]


class FileSystemCollector(SignalCollector):
    """
    Collects curiosity signals from local documents (.txt, .md, .pdf).

    Recursively scans `root_dir` for supported file types and emits one
    RawSignal per file with extractable content.

    Parameters:
        root_dir:  directory to scan (must exist, must be a directory).
        since:     if provided, files with mtime < since are excluded.
                   Must be UTC-aware.
        patterns:  glob patterns relative to root_dir (default: ["**/*"]).
                   Used to restrict which files are considered.
    """

    source = SignalSource.FILESYSTEM

    def __init__(
        self,
        root_dir: Path,
        since: datetime | None = None,
        patterns: list[str] | None = None,
    ) -> None:
        if not root_dir.exists():
            raise SignalCollectionError(
                self.source,
                f"Directory not found: {root_dir}",
            )
        if not root_dir.is_dir():
            raise SignalCollectionError(
                self.source,
                f"Path is not a directory: {root_dir}",
            )
        if since is not None and since.tzinfo is None:
            raise ValueError(
                "FileSystemCollector.since must be timezone-aware. "
                "Use datetime(..., tzinfo=timezone.utc)."
            )
        self._root = root_dir
        self._since = since
        self._patterns = patterns if patterns is not None else _DEFAULT_PATTERNS

    async def collect(self) -> list[RawSignal]:
        paths = await asyncio.to_thread(self._find_files)
        if not paths:
            return []
        results = await asyncio.gather(
            *(asyncio.to_thread(self._parse_file, p) for p in paths),
            return_exceptions=True,
        )
        return [r for r in results if isinstance(r, RawSignal)]

    def _find_files(self) -> list[Path]:
        seen: set[Path] = set()
        result: list[Path] = []
        for pattern in self._patterns:
            for path in self._root.glob(pattern):
                if path.is_file() and path.suffix in _SUPPORTED_SUFFIXES:
                    if path not in seen:
                        seen.add(path)
                        result.append(path)
        return result

    def _parse_file(self, path: Path) -> RawSignal | None:
        stat = path.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

        if self._since is not None and mtime < self._since:
            return None

        text = self._extract_text(path)
        if text is None:
            return None

        text = text.strip()
        if not text:
            return None

        return RawSignal(
            source=self.source,
            content=text[:2000],
            url=None,
            timestamp=mtime,
            metadata=self._build_metadata(path, stat, mtime),
        )

    def _extract_text(self, path: Path) -> str | None:
        suffix = path.suffix.lower()
        if suffix in (".txt", ".md"):
            return self._read_text(path)
        if suffix == ".pdf":
            return self._read_pdf(path)
        return None

    def _read_text(self, path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def _read_pdf(self, path: Path) -> str | None:
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(path)
            pages: list[str] = []
            for page in doc:
                page_text = page.get_text()
                if page_text:
                    pages.append(page_text)
            doc.close()
            return "\n".join(pages) if pages else None
        except Exception:
            # Corrupt PDFs, encrypted PDFs, import errors — all skipped silently.
            return None

    def _build_metadata(
        self, path: Path, stat: os.stat_result, mtime: datetime
    ) -> dict[str, Any]:
        return {
            "filename": path.name,
            "suffix": path.suffix,
            "size_bytes": stat.st_size,
            "mtime_utc": mtime.isoformat(),
            "path": str(path.resolve()),
        }
