"""
Shared utilities used across trace modules.

Only tiny, truly-shared helpers live here.  Do not use this as a dumping
ground — prefer co-locating logic with its primary consumer.
"""

from __future__ import annotations

import re

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?(.*?)\n?```$", re.DOTALL)


def strip_markdown_fence(text: str) -> str:
    """Remove markdown code fences Claude sometimes wraps JSON responses in."""
    stripped = text.strip()
    m = _FENCE_RE.match(stripped)
    return m.group(1).strip() if m else stripped
