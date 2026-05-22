"""
AuditWriter — appends AuditEntry records to a JSONL file.

Each pipeline run writes one record per stage so that every decision the
agent makes on a user's behalf is fully traceable.  The output is newline-
delimited JSON (JSONL), one object per line, so logs can be streamed,
grep'd, or ingested into any log analytics tool without parsing complexity.

WHY JSONL AND NOT A DATABASE:
  The audit log is append-only, sequential, and small.  A flat file is
  portable — it works on any laptop without a running database, and can
  be inspected with a text editor.  If the user wants to query it later
  they can load it into DuckDB or Pandas with a one-liner.

WHY ASYNCIO.TO_THREAD:
  File I/O is blocking.  The pipeline runs in an async event loop.
  Wrapping the append in asyncio.to_thread keeps the event loop unblocked
  even if the filesystem is slow (network mount, cold SSD).

WHY SILENT ON WRITE FAILURE:
  A failed audit write must never abort the newsletter generation.
  The audit trail is observability infrastructure, not business logic.
  Write failures are logged as WARNING so the operator knows, but the
  pipeline continues producing the user's newsletter.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from trace.models import AuditEntry

_log = logging.getLogger(__name__)


class AuditWriter:
    """
    Appends AuditEntry records to a JSONL file.

    Parameters:
        path: path to the audit log file.  Created (and its parent
              directories) if it does not exist.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    async def record(self, entry: AuditEntry) -> None:
        """Append one audit entry to the log.  Never raises."""
        line = entry.model_dump_json() + "\n"
        try:
            await asyncio.to_thread(self._append, line)
        except OSError as exc:
            _log.warning("Audit write failed (%s): %s", self._path, exc)

    def _append(self, line: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line)
