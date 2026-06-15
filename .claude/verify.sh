#!/bin/bash
# Fast BUILD-mode smoke gate (runs on every turn-end via the Stop hook). Keep it FAST.
python3 -m compileall -q -x '(\.venv|node_modules|\.git|build|dist)' .
